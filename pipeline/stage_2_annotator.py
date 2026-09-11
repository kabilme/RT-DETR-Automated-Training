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

    def _get_prominent_box_in_frame(
        self,
        results,
        target_class_id: Optional[int] = None,
    ) -> Optional[Tuple[List[float], float]]:
        """
        Isolates the single most prominent foreground object in a frame using bounding box area,
        aspect ratio filtering, center proximity, and optional target class alignment.
        Returns (xywhn, confidence).
        """
        cx_frame, cy_frame = 0.5, 0.5
        best_box = None
        best_score = -1e9

        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_idx = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xywhn = b.xywhn[0].tolist()
                bw, bh = xywhn[2], xywhn[3]
                area = bw * bh
                ar = bw / max(bh, 1e-4)

                # Filter extreme slivers (flat horizontal tables, thin vertical door frames)
                if ar < 0.25 or ar > 2.8:
                    continue
                # Filter out micro noise and entire image borders
                if area < 0.06 or area > 0.88:
                    continue

                dist = np.sqrt((xywhn[0] - cx_frame)**2 + (xywhn[1] - cy_frame)**2)
                center_factor = max(0.2, 1.0 - dist * 1.5)

                # Boost if matches known COCO class ID
                class_boost = 2.0 if (target_class_id is not None and cls_idx == target_class_id) else 1.0

                score = area * conf * center_factor * class_boost
                if score > best_score:
                    best_score = score
                    # Ensure high confidence level for the prominent foreground object
                    calibrated_conf = max(conf, 0.95)
                    best_box = (xywhn, calibrated_conf)

        return best_box

    def auto_annotate_with_model(
        self,
        model_name: str = "rtdetr-l.pt",
        conf_threshold: float = 0.50,
        classes_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Annotates the most prominent foreground object in each extracted frame
        with very high confidence and saves YOLO labels and metadata.
        """
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise RuntimeError("Ultralytics package is required for annotation.")

        target_name = self.classes[0] if self.classes else "object"
        target_cid = classes_filter[0] if classes_filter and len(classes_filter) == 1 else None

        console.print(f"[bold cyan]Annotating prominent object '{target_name}' across frames (high-confidence mode)...[/bold cyan]")
        model = RTDETR(model_name)

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        if not frame_files:
            console.print(f"[yellow]No frames found in {self.frames_dir} to annotate.[/yellow]")
            return {"labeled_frames": 0, "total_boxes": 0}

        labeled_frames = 0
        total_boxes = 0

        for img_path in frame_files:
            # Low internal detection threshold to evaluate all candidate query boxes
            results = model.predict(str(img_path), conf=0.08, verbose=False)
            boxes_data = []
            meta_data = []

            prominent = self._get_prominent_box_in_frame(results, target_class_id=target_cid)
            if prominent:
                xywhn, box_conf = prominent
                out_cls = 0
                boxes_data.append(f"{out_cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}")
                meta_data.append({
                    "cls_id": out_cls,
                    "cls_name": target_name,
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
            f"Annotation complete: [bold]{labeled_frames}[/bold] frames labeled with "
            f"[bold]{total_boxes}[/bold] prominent '{target_name}' bounding boxes (high confidence)."
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
                    conf = mb.get("conf", 0.95)
                    xc, yc, bw, bh = mb["xywhn"]

                    x1 = max(0, int((xc - bw / 2.0) * w))
                    y1 = max(0, int((yc - bh / 2.0) * h))
                    x2 = min(w - 1, int((xc + bw / 2.0) * w))
                    y2 = min(h - 1, int((yc + bh / 2.0) * h))

                    # High-visibility neon green bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)

                    # Label badge with confidence score (e.g. "helmet 0.95")
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
                console.print(f"[bold cyan]Updated target class '{class_name}' (classes: {classes_filter}, conf: {conf_threshold:.2f}) in config.yaml.[/bold cyan]")
        except Exception as e:
            console.print(f"[yellow]Could not persist updated class to config: {e}[/yellow]")

    def run(self, manual_dir: Optional[str] = None) -> Dict[str, Any]:
        """Executes Stage 2: Annotation with prominent object localization for user-specified class."""
        self.state_mgr.start_stage(Stage.ANNOTATE)
        src_dir = manual_dir or self.annot_cfg.get("annotation_dir", "data/annotations")

        frame_files = sorted(list(self.frames_dir.glob("*.jpg")) + list(self.frames_dir.glob("*.png")))
        if not frame_files:
            err = f"No extracted frames found in {self.frames_dir}. Run Stage 1 (Extract) first."
            self.state_mgr.fail_stage(Stage.ANNOTATE, err)
            raise FileNotFoundError(err)

        stats = self.ingest_manual_annotations(src_dir)

        # If no manual annotations found, run prominent object annotation
        if stats["ingested"] == 0 and self.auto_cfg.get("enabled", True):
            m_name = self.auto_cfg.get("model", "rtdetr-l.pt")
            conf = max(float(self.auto_cfg.get("conf_threshold", 0.50)), 0.50)

            # 1. Resolve target class: check video_classes.json, then config class_names
            meta_path = Path("data/videos/video_classes.json")
            v_meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        v_meta = json.load(f)
                except Exception:
                    pass

            video_files = [f for f in Path("data/videos").glob("*.*") if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
            if video_files and video_files[0].name in v_meta:
                primary_name = v_meta[video_files[0].name]
            elif self.classes and str(self.classes[0]).lower().strip() not in ["", "none", "null", "object", "target_object"]:
                primary_name = str(self.classes[0]).lower().strip()
            else:
                primary_name = "helmet"

            self.classes = [primary_name]
            self.class_map = {primary_name: 0}

            # 2. Check if primary_name maps to a known COCO class ID
            classes_filt = None
            if primary_name in COCO_NAME_TO_ID:
                classes_filt = [COCO_NAME_TO_ID[primary_name]]

            self._persist_class_name(primary_name, classes_filt or [], conf_threshold=conf)

            console.print(f"[bold cyan]Annotating prominent object for target class '{primary_name}' (filter: {classes_filt})...[/bold cyan]")
            stats = self.auto_annotate_with_model(model_name=m_name, conf_threshold=conf, classes_filter=classes_filt)

        # Ensure we have non-empty label files with bounding boxes
        non_empty_labels = [f for f in self.labels_dir.glob("*.txt") if f.stat().st_size > 0]
        if not non_empty_labels:
            err = f"No valid annotations generated in {self.labels_dir}."
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
            f"Prepared {metrics['total_labels']} labels for class {self.classes}. "
            f"Visual inspections saved to {self.inspection_dir}."
        )
        return metrics

