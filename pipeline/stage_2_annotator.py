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
        conf_threshold: float = 0.60,
        classes_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Uses a pretrained RT-DETR or YOLO model to auto-generate initial pseudo-labels
        for extracted frames.
        """
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise RuntimeError("Ultralytics package is required for auto-labeling.")

        console.print(f"[bold cyan]Auto-labeling extracted frames using {model_name}...[/bold cyan]")
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

            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_idx = int(b.cls[0].item())
                    if classes_filter is not None and cls_idx not in classes_filter:
                        continue
                    xywhn = b.xywhn[0].tolist()
                    # map to target class 0 if only 1 target class specified
                    out_cls = 0 if len(self.classes) == 1 else cls_idx
                    boxes_data.append(f"{out_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")

            out_txt = self.labels_dir / f"{img_path.stem}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(boxes_data))

            if boxes_data:
                labeled_frames += 1
                total_boxes += len(boxes_data)

        console.print(
            f"Auto-labeling complete: [bold]{labeled_frames}[/bold] frames labeled with "
            f"[bold]{total_boxes}[/bold] bounding boxes."
        )
        return {"labeled_frames": labeled_frames, "total_boxes": total_boxes}

    def render_visual_inspection(self, max_samples: int = 15) -> List[str]:
        """Renders bounding boxes onto a sample of images for visual inspection."""
        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        inspected_paths = []

        samples = frame_files[:max_samples]
        for img_path in samples:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])

                            x1 = int((xc - bw / 2.0) * w)
                            y1 = int((yc - bh / 2.0) * h)
                            x2 = int((xc + bw / 2.0) * w)
                            y2 = int((yc + bh / 2.0) * h)

                            cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"

                            # Draw rectangle & label tag
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_text = f"{cls_name}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                            cv2.putText(img, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            out_inspect = self.inspection_dir / f"inspect_{img_path.name}"
            cv2.imwrite(str(out_inspect), img)
            inspected_paths.append(str(out_inspect))

        return inspected_paths

    def _persist_class_name(self, class_name: str, classes_filter: List[int]) -> None:
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
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                console.print(f"[bold cyan]Updated target class '{class_name}' in config.yaml.[/bold cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not persist updated class to config: {e}[/yellow]")

    def _discover_dominant_class(self, model_name: str = "rtdetr-l.pt", conf: float = 0.25) -> Optional[Tuple[str, int]]:
        """Scans sample extracted frames to auto-discover the dominant object class."""
        from collections import Counter
        try:
            from ultralytics import RTDETR
            model = RTDETR(model_name)
        except Exception:
            return None

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))[:25]
        if not frame_files:
            return None

        class_counts = Counter()
        BACKGROUND_CLUTTER = {58, 45, 41, 46, 39, 44}  # potted plant, bowl, cup, banana, bottle, spoon

        for img_path in frame_files:
            results = model.predict(str(img_path), conf=conf, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for b in r.boxes:
                        cid = int(b.cls[0].item())
                        class_counts[cid] += 1

        if not class_counts:
            return None

        v_stems = " ".join([f.stem.lower() for f in Path("data/videos").glob("*.*")])
        for kw, cid in COCO_NAME_TO_ID.items():
            if kw in v_stems and class_counts[cid] > 0:
                name = "scooter" if kw in ["activa", "scooter", "moped", "vespa"] else kw
                return (name, cid)

        candidates = [(cid, cnt) for cid, cnt in class_counts.most_common() if cid not in BACKGROUND_CLUTTER]
        if not candidates:
            candidates = class_counts.most_common()

        best_cid = candidates[0][0]
        raw_name = model.names.get(best_cid, f"object_{best_cid}").lower()
        if raw_name in ["motorcycle", "bicycle"] and ("activa" in v_stems or "scooter" in v_stems):
            clean_name = "scooter"
        else:
            clean_name = raw_name
        return (clean_name, best_cid)

    def run(self, manual_dir: Optional[str] = None) -> Dict[str, Any]:
        """Executes Stage 2: Annotation ingestion or bootstrapping."""
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
            conf = float(self.auto_cfg.get("conf_threshold", 0.30))
            classes_filt = self.auto_cfg.get("classes", None)

            # Check if video filenames give a strong hint for target class
            video_files = [f for f in Path("data/videos").glob("*.*") if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
            primary_name = str(self.classes[0]).lower().strip() if self.classes else "object"

            video_hint_found = False
            if video_files:
                v_stem = video_files[0].stem.lower()
                for kw, cid in COCO_NAME_TO_ID.items():
                    if kw in v_stem:
                        mapped_name = "scooter" if kw in ["activa", "scooter", "moped", "vespa"] else kw
                        if mapped_name != primary_name:
                            console.print(
                                f"[bold yellow]Video '{video_files[0].name}' indicates target class '{mapped_name}' (COCO ID {cid}). "
                                f"Auto-switching from previous target class '{primary_name}' -> '{mapped_name}'.[/bold yellow]"
                            )
                            primary_name = mapped_name
                            self.classes = [mapped_name]
                            self.class_map = {mapped_name: 0}
                            classes_filt = [cid]
                            self._persist_class_name(mapped_name, [cid])
                            video_hint_found = True
                        break

            if not video_hint_found:
                # Override default [3, 1, 2] if target class is not a vehicle
                if classes_filt == [3, 1, 2] and primary_name not in ["scooter", "motorcycle", "bike", "bicycle", "car"]:
                    classes_filt = None

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

            console.print(f"[bold cyan]Auto-labeling target class '{primary_name}' with COCO filter {classes_filt}...[/bold cyan]")
            stats = self.auto_annotate_with_model(model_name=m_name, conf_threshold=conf, classes_filter=classes_filt)

            # If initial filter resulted in 0 bounding boxes, auto-discover dominant class
            non_empty_labels = [f for f in self.labels_dir.glob("*.txt") if f.stat().st_size > 0]
            if not non_empty_labels:
                console.print(f"[bold yellow]No objects detected for '{primary_name}'. Scanning frames to auto-discover dominant objects...[/bold yellow]")
                discovered = self._discover_dominant_class(model_name=m_name, conf=conf)
                if discovered:
                    disc_name, disc_cid = discovered
                    console.print(f"[bold green]Auto-detected dominant object '{disc_name}' (COCO ID {disc_cid}). Re-annotating frames...[/bold green]")
                    primary_name = disc_name
                    self.classes = [disc_name]
                    self.class_map = {disc_name: 0}
                    self._persist_class_name(disc_name, [disc_cid])
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
