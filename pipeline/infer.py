"""
Inference Runner with Calibrated Zero-False-Positive Filtering.
Executes inference on images/videos using trained RT-DETR weights (.pt or .onnx)
with precision-calibrated per-class thresholds and geometric noise filters.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console

console = Console()


COCO_80_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}


class FalsePositiveFreeDetector:
    """Detects objects while strictly suppressing false positives using calibrated operating profiles."""

    def __init__(
        self,
        model_path: str,
        calibration_path: Optional[str] = None,
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.is_onnx = self.model_path.suffix.lower() == ".onnx"
        self.is_pretrained_base = "rtdetr-l" in self.model_path.name.lower()

        # Load calibration profile if this is a custom-trained model
        if not self.is_pretrained_base:
            self.calib_data = self._load_calibration(calibration_path)
            self.class_thresholds = self.calib_data.get("class_thresholds", {})
            self.global_threshold = float(self.calib_data.get("global_calibrated_threshold", 0.15))
        else:
            self.calib_data = {}
            self.class_thresholds = {}
            self.global_threshold = 0.25

        geom = self.calib_data.get("geometric_filters", {})
        self.min_box_area = float(geom.get("min_box_area", 50 if self.is_pretrained_base else 100))
        self.aspect_range = geom.get("aspect_ratio_range", [0.05, 20.0] if self.is_pretrained_base else [0.1, 10.0])

        # Build class name mapping
        if self.is_pretrained_base:
            self.class_names = dict(COCO_80_NAMES)
        else:
            self.class_names = {}
            for cname, cinfo in self.class_thresholds.items():
                cid = cinfo.get("class_id")
                if cid is not None:
                    self.class_names[cid] = cname

            # Fallback to dataset.yaml or config.yaml
            if not self.class_names:
                try:
                    import yaml
                    for yml_path in ["workspace/dataset/dataset.yaml", "config.yaml"]:
                        if Path(yml_path).exists():
                            with open(yml_path, "r", encoding="utf-8") as f:
                                d_yaml = yaml.safe_load(f)
                                names = d_yaml.get("names", {}) or d_yaml.get("annotation", {}).get("class_names", [])
                                if isinstance(names, dict):
                                    self.class_names = {int(k): str(v) for k, v in names.items()}
                                    break
                                elif isinstance(names, list):
                                    self.class_names = {i: str(n) for i, n in enumerate(names)}
                                    break
                except Exception:
                    pass

        # Load backend engine
        if self.is_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.model = None
        else:
            from ultralytics import RTDETR
            self.model = RTDETR(str(self.model_path))
            self.session = None
            if hasattr(self.model, "names") and self.model.names:
                if self.is_pretrained_base or len(self.model.names) > 1:
                    self.class_names = {int(k): str(v) for k, v in self.model.names.items()}

    def _load_calibration(self, calib_path: Optional[str]) -> Dict[str, Any]:
        candidates = []
        if calib_path:
            candidates.append(Path(calib_path))
        candidates.extend([
            Path("workspace/exported_models/calibrated_thresholds.json"),
            Path("workspace/evaluation/calibrated_thresholds.json"),
        ])
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return {}

    def _filter_box(self, xyxy: List[float], cls_id: int, conf: float) -> bool:
        """Applies calibrated per-class threshold and geometric filters."""
        # Find threshold for this class
        req_conf = self.global_threshold
        for cname, cinfo in self.class_thresholds.items():
            if cinfo.get("class_id") == cls_id:
                req_conf = float(cinfo.get("calibrated_conf", self.global_threshold))
                break

        if conf < req_conf:
            return False

        # Geometric filtering
        w = max(0.0, xyxy[2] - xyxy[0])
        h = max(0.0, xyxy[3] - xyxy[1])
        area = w * h
        ar = w / max(h, 1e-4)

        if area < self.min_box_area:
            return False
        if ar < self.aspect_range[0] or ar > self.aspect_range[1]:
            return False

        return True

    def predict_image(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Runs false-positive filtered inference on a single BGR image."""
        h0, w0 = image_bgr.shape[:2]
        filtered_detections = []

        if not self.is_onnx:
            # Ultralytics RT-DETR prediction
            # Use raw low confidence to allow our per-class thresholding
            results = self.model.predict(image_bgr, conf=0.01, imgsz=self.imgsz, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = [max(0.0, float(x)) for x in b.xyxy[0].tolist()]

                    if self._filter_box(xyxy, cls_id, conf):
                        cls_name = r.names.get(cls_id, self.class_names.get(cls_id, f"class_{cls_id}"))
                        filtered_detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(x, 2) for x in xyxy],
                        })
        else:
            # ONNX Runtime prediction
            # Preprocess: letterbox/resize to imgsz
            img_resized = cv2.resize(image_bgr, (self.imgsz, self.imgsz))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            outputs = self.session.run(None, {self.input_name: input_tensor})
            # RT-DETR ONNX output is typically (1, 300, 6) -> [cx, cy, w, h, score, class_id]
            out = outputs[0][0]

            for row in out:
                if len(row) >= 6:
                    c0, c1, c2, c3 = [float(v) for v in row[:4]]
                    if len(row) == 6:
                        conf = float(row[4])
                        cls_id = int(row[5])
                    else:
                        class_probs = row[4:]
                        cls_id = int(np.argmax(class_probs))
                        conf = float(class_probs[cls_id])

                    # Check whether coordinates are normalized [0..1] or pixel [0..imgsz]
                    if max(c0, c1, c2, c3) <= 1.5:
                        # Normalized center-xywh format
                        x1 = max(0.0, (c0 - c2 / 2.0) * w0)
                        y1 = max(0.0, (c1 - c3 / 2.0) * h0)
                        x2 = min(float(w0), (c0 + c2 / 2.0) * w0)
                        y2 = min(float(h0), (c1 + c3 / 2.0) * h0)
                    else:
                        # Pixel coordinates
                        sx = w0 / self.imgsz
                        sy = h0 / self.imgsz
                        x1 = max(0.0, (c0 - c2 / 2.0) * sx)
                        y1 = max(0.0, (c1 - c3 / 2.0) * sy)
                        x2 = min(float(w0), (c0 + c2 / 2.0) * sx)
                        y2 = min(float(h0), (c1 + c3 / 2.0) * sy)

                    box = [x1, y1, x2, y2]
                    if self._filter_box(box, cls_id, conf):
                        cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        filtered_detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(x, 2) for x in box],
                        })

        return filtered_detections

    def annotate_image(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws bounding boxes and labels onto the image."""
        vis = image_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cname = d["class_name"]
            conf = d["confidence"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cname} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return vis


def run_inference_on_file(
    model_path: str,
    input_path: str,
    output_dir: str = "workspace/inference_output",
    calibration_path: Optional[str] = None,
) -> None:
    """Executes inference on an image file, video file, or directory."""
    detector = FalsePositiveFreeDetector(model_path, calibration_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if in_p.is_file() and in_p.suffix.lower() in image_exts:
        img = cv2.imread(str(in_p))
        dets = detector.predict_image(img)
        vis = detector.annotate_image(img, dets)
        out_file = out_dir / in_p.name
        cv2.imwrite(str(out_file), vis)
        console.print(f"Processed image: {in_p.name} -> Detected {len(dets)} object(s). Saved to {out_file}")

    elif in_p.is_file() and in_p.suffix.lower() in video_exts:
        cap = cv2.VideoCapture(str(in_p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_video_path = out_dir / f"detected_{in_p.name}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        console.print(f"Starting inference on video: {in_p.name} ({total_frames} frames)...")

        total_dets = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.predict_image(frame)
            vis = detector.annotate_image(frame, dets)
            writer.write(vis)
            total_dets += len(dets)
            frame_idx += 1
            if frame_idx % 25 == 0 or frame_idx == total_frames:
                console.print(f" - Progress: {frame_idx}/{total_frames} frames ({total_dets} detections found)")

        cap.release()
        writer.release()
        console.print(f"Processed video: {in_p.name} ({frame_idx} frames, {total_dets} total detections). Saved to {out_video_path}")

    elif in_p.is_dir():
        files = [p for p in in_p.iterdir() if p.suffix.lower() in image_exts]
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            dets = detector.predict_image(img)
            vis = detector.annotate_image(img, dets)
            cv2.imwrite(str(out_dir / f.name), vis)
        console.print(f"Processed {len(files)} image(s) in directory {in_p}. Saved to {out_dir}")
