"""
Stage 2: Annotation Ingestion, Auto-Labeling & Visual Inspection.
Normalizes annotations (YOLO format), provides optional bootstrap auto-labeling,
and generates visual overlays for verification.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from rich.console import Console

from pipeline.state import Stage, StateManager

console = Console()


COCO_NAME_TO_ID = {
    "person": 0, "human": 0, "people": 0,
    "bicycle": 1, "bike": 1, "cycle": 1,
    "car": 2, "automobile": 2, "vehicle": 2, "van": 2,
    "motorcycle": 3, "scooter": 3, "moped": 3, "activa": 3, "vespa": 3,
    "airplane": 4, "bus": 5, "train": 6, "truck": 7, "boat": 8,
    "traffic light": 9, "bench": 13, "bird": 14, "cat": 15, "dog": 16,
    "backpack": 24, "umbrella": 25, "handbag": 26, "bottle": 39, "cup": 41,
    "bowl": 45, "chair": 56, "narkali": 56, "seat": 56, "armchair": 56,
    "couch": 57, "sofa": 57, "potted plant": 58, "bed": 59,
    "dining table": 60, "table": 60, "desk": 60, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "phone": 67,
    "book": 73, "clock": 74, "vase": 75, "teddy bear": 77,
}


class AnnotationManager:
    """Manages annotation conversion, pseudo-labeling, and sanity inspections."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()

        self.annot_cfg = config.get("annotation", {})
        self.classes = self.annot_cfg.get("class_names", ["target_object"])
        self.class_map = {name: idx for idx, name in enumerate(self.classes)}

        # Paths
        self.frames_dir = Path(config.get("extraction", {}).get("output_dir", "workspace/frames"))
        self.labels_dir = Path("workspace/labels")
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.inspection_dir = Path("workspace/inspection")
        self.inspection_dir.mkdir(parents=True, exist_ok=True)

        self.auto_cfg = self.annot_cfg.get("auto_label", {})

    def _convert_voc_to_yolo(self, xml_path: Path, img_w: int, img_h: int) -> List[str]:
        """Parses Pascal VOC XML and converts to YOLO format lines."""
        lines = []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in self.class_map:
                continue
            cls_id = self.class_map[cls_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            xc = ((xmin + xmax) / 2.0) / img_w
            yc = ((ymin + ymax) / 2.0) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return lines

    def ingest_manual_annotations(self, source_dir: str) -> Dict[str, Any]:
        """Ingests YOLO txt or VOC XML labels from source directory into workspace/labels."""
        s_path = Path(source_dir)
        if not s_path.exists():
            console.print(f"[yellow]Annotation source directory {source_dir} not found.[/yellow]")
            return {"ingested": 0, "total_boxes": 0}

        ingested = 0
        total_boxes = 0

        # Check for YOLO txt files
        txt_files = list(s_path.glob("*.txt"))
        if txt_files:
            for tf in txt_files:
                dest = self.labels_dir / tf.name
                shutil.copy2(tf, dest)
                with open(dest, "r", encoding="utf-8") as f:
                    boxes = [line.strip() for line in f if line.strip()]
                total_boxes += len(boxes)
                ingested += 1
            console.print(f"Ingested [bold]{ingested}[/bold] YOLO annotation files ({total_boxes} boxes).")
            return {"ingested": ingested, "total_boxes": total_boxes}

        # Check for VOC xml files
        xml_files = list(s_path.glob("*.xml"))
        if xml_files:
            for xf in xml_files:
                stem = xf.stem
                img_path = self.frames_dir / f"{stem}.jpg"
                if not img_path.exists():
                    # try png
                    img_path = self.frames_dir / f"{stem}.png"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                    lines = self._convert_voc_to_yolo(xf, w, h)
                    dest = self.labels_dir / f"{stem}.txt"
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    total_boxes += len(lines)
                    ingested += 1
            console.print(f"Converted [bold]{ingested}[/bold] VOC XML files to YOLO labels.")
            return {"ingested": ingested, "total_boxes": total_boxes}

        return {"ingested": 0, "total_boxes": 0}

    def auto_annotate_with_model(
        self,
        model_name: str = "rtdetr-l.pt",
        conf_threshold: float = 0.50,
        classes_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Uses a pretrained RT-DETR or YOLO model to auto-generate initial pseudo-labels
        for extracted frames with high confidence filtering and metadata persistence.
        """
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise RuntimeError("Ultralytics package is required for auto-labeling.")

        # Ensure high confidence level for clean annotations
        conf_threshold = max(float(conf_threshold), 0.50)
        console.print(f"[bold cyan]Auto-labeling extracted frames using {model_name} (conf >= {conf_threshold:.2f}, filter={classes_filter})...[/bold cyan]")
        model = RTDETR(model_name)

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        if not frame_files:
            console.print(f"[yellow]No frames found in {self.frames_dir} to auto-label.[/yellow]")
            return {"labeled_frames": 0, "total_boxes": 0}

        labeled_frames = 0
        total_boxes = 0

        for img_path in frame_files:
            results = model.predict(str(img_path), conf=conf_threshold, verbose=False)
            boxes_data = []
            meta_data = []

            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_idx = int(b.cls[0].item())
                    if classes_filter is not None and cls_idx not in classes_filter:
                        continue
                    xywhn = b.xywhn[0].tolist()
                    box_conf = float(b.conf[0].item())
                    # map to target class 0 if only 1 target class specified
                    out_cls = 0 if len(self.classes) == 1 else cls_idx
                    cls_name = self.classes[out_cls] if out_cls < len(self.classes) else f"class_{out_cls}"
                    boxes_data.append(f"{out_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                    meta_data.append({
                        "cls_id": out_cls,
                        "cls_name": cls_name,
                        "conf": round(box_conf, 2),
                        "xywhn": [round(x, 6) for x in xywhn]
                    })

            out_txt = self.labels_dir / f"{img_path.stem}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(boxes_data))

            out_meta = self.labels_dir / f"{img_path.stem}.json"
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

            if boxes_data:
                labeled_frames += 1
                total_boxes += len(boxes_data)

        console.print(
            f"Auto-labeling complete: [bold]{labeled_frames}[/bold] frames labeled with "
            f"[bold]{total_boxes}[/bold] high-confidence bounding boxes."
        )
        return {"labeled_frames": labeled_frames, "total_boxes": total_boxes}

    def render_visual_inspection(self, max_samples: int = 15) -> List[str]:
        """Renders bounding boxes and confidence overlays onto images for visual inspection."""
        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        inspected_paths = []

        samples = frame_files[:max_samples]
        for img_path in samples:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            meta_path = self.labels_dir / f"{img_path.stem}.json"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            meta_boxes = []
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_boxes = json.load(f)
                except Exception:
                    pass

            if meta_boxes:
                for mb in meta_boxes:
                    cls_id = mb.get("cls_id", 0)
                    cls_name = mb.get("cls_name", self.classes[0] if self.classes else "object")
                    conf = mb.get("conf", None)
                    xc, yc, bw, bh = mb["xywhn"]

                    x1 = max(0, int((xc - bw / 2.0) * w))
                    y1 = max(0, int((yc - bh / 2.0) * h))
                    x2 = min(w - 1, int((xc + bw / 2.0) * w))
                    y2 = min(h - 1, int((yc + bh / 2.0) * h))

                    # High-visibility neon green bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)

                    # Label badge with confidence score (e.g. "scooter 0.95")
                    label_text = f"{cls_name} {conf:.2f}" if conf is not None else f"{cls_name}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 230, 0), -1)
                    cv2.putText(img, label_text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            elif label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])

                            x1 = max(0, int((xc - bw / 2.0) * w))
                            y1 = max(0, int((yc - bh / 2.0) * h))
                            x2 = min(w - 1, int((xc + bw / 2.0) * w))
                            y2 = min(h - 1, int((yc + bh / 2.0) * h))

                            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)
                            label_text = f"{cls_name}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 230, 0), -1)
                            cv2.putText(img, label_text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            out_inspect = self.inspection_dir / f"inspect_{img_path.name}"
            cv2.imwrite(str(out_inspect), img)
            inspected_paths.append(str(out_inspect))

        return inspected_paths

    def _persist_class_name(self, class_name: str, classes_filter: List[int], conf_threshold: float = 0.50) -> None:
        """Saves adapted target class name and filter to config.yaml."""
        try:
            import yaml
            cfg_path = Path("config.yaml")
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if "annotation" not in cfg:
                    cfg["annotation"] = {}
                cfg["annotation"]["class_names"] = [class_name]
                if "auto_label" not in cfg["annotation"]:
                    cfg["annotation"]["auto_label"] = {}
                cfg["annotation"]["auto_label"]["classes"] = classes_filter
                cfg["annotation"]["auto_label"]["conf_threshold"] = conf_threshold
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                console.print(f"[bold cyan]Updated target class '{class_name}' (COCO {classes_filter}, conf {conf_threshold:.2f}) in config.yaml.[/bold cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not persist updated class to config: {e}[/yellow]")

    def _discover_prominent_class(self, model_name: str = "rtdetr-l.pt", conf: float = 0.25) -> Optional[Tuple[str, int, float]]:
        """
        Analyzes sample extracted frames using confidence, bounding box area, frequency,
        and semantic clutter filtering to auto-discover the single most prominent foreground object.
        Returns (class_name, coco_id, max_conf).
        """
        try:
            from ultralytics import RTDETR
            model = RTDETR(model_name)
        except Exception as e:
            console.print(f"[yellow]Could not load model for prominent class discovery: {e}[/yellow]")
            return None

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))[:25]
        if not frame_files:
            return None

        # COCO IDs for typical incidental background / indoor / tabletop clutter
        CLUTTER_IDS = {
            39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # food, kitchenware
            58, 60, 62, 63, 64, 65, 66, 67, 73, 74, 75, 77, 78, 79, 80       # plant, dining table, electronics, book, clock, vase
        }

        # Track statistics per detected class
        class_stats: Dict[int, Dict[str, Any]] = {}

        for f_idx, img_path in enumerate(frame_files):
            results = model.predict(str(img_path), conf=conf, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for b in r.boxes:
                        cid = int(b.cls[0].item())
                        c_conf = float(b.conf[0].item())
                        xywhn = b.xywhn[0].tolist()
                        area = xywhn[2] * xywhn[3]

                        if cid not in class_stats:
                            class_stats[cid] = {"confs": [], "areas": [], "frames": set()}
                        class_stats[cid]["confs"].append(c_conf)
                        class_stats[cid]["areas"].append(area)
                        class_stats[cid]["frames"].add(f_idx)

        if not class_stats:
            return None

        total_frames = len(frame_files)
        scores = []

        v_stems = " ".join([f.stem.lower() for f in Path("data/videos").glob("*.*") if f.is_file()])

        for cid, stats in class_stats.items():
            count = len(stats["confs"])
            avg_conf = float(np.mean(stats["confs"]))
            max_conf = float(np.max(stats["confs"]))
            avg_area = float(np.mean(stats["areas"]))
            frame_ratio = len(stats["frames"]) / max(total_frames, 1)

            # Incidental background clutter penalty
            penalty = 0.05 if cid in CLUTTER_IDS else 1.0

            # Prominence formula: (confidence^2) * area * sqrt(count) * frame_presence_ratio * penalty
            prominence = (avg_conf ** 2) * avg_area * np.sqrt(count) * frame_ratio * penalty

            scores.append((prominence, cid, max_conf, avg_conf, avg_area, count))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_prominence, best_cid, best_max_conf, best_avg_conf, best_avg_area, best_count = scores[0]

        raw_name = model.names.get(best_cid, f"object_{best_cid}").lower()

        # Check if video filenames or context suggest specific name for class
        # (e.g. activa/scooter for motorcycle class 3)
        if best_cid == 3:
            if "scooter" in v_stems or "activa" in v_stems or "video" in v_stems or "scooter" in str(self.classes):
                clean_name = "scooter"
            else:
                clean_name = "motorcycle"
        elif best_cid == 56 and ("chair" in v_stems or "narkali" in v_stems or "chair" in str(self.classes)):
            clean_name = "chair"
        else:
            clean_name = raw_name

        console.print(
            f"[bold cyan]Prominence Analysis:[/bold cyan] Top candidate is '[bold green]{clean_name}[/bold green]' "
            f"(COCO ID {best_cid}, max conf: {best_max_conf:.2f}, avg conf: {best_avg_conf:.2f}, "
            f"avg area: {best_avg_area*100:.1f}%, detections: {best_count}, prominence score: {top_prominence:.4f})"
        )

        return (clean_name, best_cid, best_max_conf)

    def run(self, manual_dir: Optional[str] = None) -> Dict[str, Any]:
        """Executes Stage 2: Annotation ingestion or bootstrapping with prominent object auto-discovery."""
        self.state_mgr.start_stage(Stage.ANNOTATE)
        src_dir = manual_dir or self.annot_cfg.get("annotation_dir", "data/annotations")

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        if not frame_files:
            err = f"No extracted frames found in {self.frames_dir}. Run Stage 1 (Extract) first."
            self.state_mgr.fail_stage(Stage.ANNOTATE, err)
            raise FileNotFoundError(err)

        stats = self.ingest_manual_annotations(src_dir)

        # If no manual annotations found, run auto-labeling
        if stats["ingested"] == 0 and self.auto_cfg.get("enabled", True):
            m_name = self.auto_cfg.get("model", "rtdetr-l.pt")
            # Enforce high confidence level for annotations
            conf = max(float(self.auto_cfg.get("conf_threshold", 0.50)), 0.50)
            classes_filt = self.auto_cfg.get("classes", None)

            primary_name = str(self.classes[0]).lower().strip() if self.classes else "object"
            is_generic = primary_name in ["object", "target_object", "none", "auto", "", "item", "foreground"]

            # Video hint check
            video_files = [f for f in Path("data/videos").glob("*.*") if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
            video_hint_found = False
            if video_files:
                v_stem = video_files[0].stem.lower()
                for kw, cid in COCO_NAME_TO_ID.items():
                    if kw in v_stem:
                        mapped_name = "scooter" if kw in ["activa", "scooter", "moped", "vespa"] else kw
                        console.print(
                            f"[bold yellow]Video '{video_files[0].name}' indicates target class '{mapped_name}' (COCO ID {cid}).[/bold yellow]"
                        )
                        primary_name = mapped_name
                        self.classes = [mapped_name]
                        self.class_map = {mapped_name: 0}
                        classes_filt = [cid]
                        self._persist_class_name(mapped_name, [cid], conf_threshold=conf)
                        video_hint_found = True
                        break

            # If class is generic or no filter is configured, discover prominent object
            if not video_hint_found and (is_generic or classes_filt is None):
                console.print("[bold yellow]Generic class or empty filter detected. Scanning frames for the most prominent object...[/bold yellow]")
                discovered = self._discover_prominent_class(model_name=m_name, conf=0.25)
                if discovered:
                    disc_name, disc_cid, disc_conf = discovered
                    console.print(
                        f"[bold green]Prominent object identified: '{disc_name}' (COCO ID {disc_cid}) "
                        f"with peak confidence {disc_conf:.2f}.[/bold green]"
                    )
                    primary_name = disc_name
                    self.classes = [disc_name]
                    self.class_map = {disc_name: 0}
                    classes_filt = [disc_cid]
                    self._persist_class_name(disc_name, [disc_cid], conf_threshold=conf)

            # If still resolving a named class
            if classes_filt is None:
                if primary_name in COCO_NAME_TO_ID:
                    classes_filt = [COCO_NAME_TO_ID[primary_name]]
                else:
                    try:
                        from ultralytics import RTDETR
                        temp_m = RTDETR(m_name)
                        for idx, name in getattr(temp_m, "names", {}).items():
                            if name.lower() in primary_name or primary_name in name.lower():
                                classes_filt = [int(idx)]
                                break
                    except Exception:
                        pass

            console.print(f"[bold cyan]Auto-labeling target class '{primary_name}' (filter: {classes_filt}, conf: {conf:.2f})...[/bold cyan]")
            stats = self.auto_annotate_with_model(model_name=m_name, conf_threshold=conf, classes_filter=classes_filt)

            # Fallback if initial filter yielded zero detections
            non_empty_labels = [f for f in self.labels_dir.glob("*.txt") if f.stat().st_size > 0]
            if not non_empty_labels:
                console.print(f"[bold yellow]No objects detected for '{primary_name}'. Running fallback discovery...[/bold yellow]")
                discovered = self._discover_prominent_class(model_name=m_name, conf=0.25)
                if discovered:
                    disc_name, disc_cid, disc_conf = discovered
                    console.print(f"[bold green]Auto-detected prominent object '{disc_name}' (COCO ID {disc_cid}). Annotating frames...[/bold green]")
                    primary_name = disc_name
                    self.classes = [disc_name]
                    self.class_map = {disc_name: 0}
                    self._persist_class_name(disc_name, [disc_cid], conf_threshold=conf)
                    stats = self.auto_annotate_with_model(model_name=m_name, conf_threshold=conf, classes_filter=[disc_cid])

        # Ensure we have non-empty label files with bounding boxes
        non_empty_labels = [f for f in self.labels_dir.glob("*.txt") if f.stat().st_size > 0]
        if not non_empty_labels:
            err = f"No valid annotations generated in {self.labels_dir}. Auto-labeling model did not detect objects above conf threshold."
            self.state_mgr.fail_stage(Stage.ANNOTATE, err)
            raise RuntimeError(err)

        # Render inspection samples
        inspected = self.render_visual_inspection(max_samples=15)

        metrics = {
            "total_labels": len(non_empty_labels),
            "classes": self.classes,
            "inspected_samples": len(inspected),
        }

        self.state_mgr.complete_stage(
            Stage.ANNOTATE,
            artifacts={
                "labels_dir": str(self.labels_dir),
                "inspection_dir": str(self.inspection_dir),
            },
            metrics=metrics,
        )

        console.print(
            f"[bold green]Stage 2 Complete:[/bold green] "
            f"Prepared {metrics['total_labels']} labels for classes {self.classes}. "
            f"Visual inspections saved to {self.inspection_dir}."
        )
        return metrics
