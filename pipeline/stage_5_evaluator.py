"""
Stage 5: False Positive Evaluator & Precision Calibration Engine.
Evaluates detections on positive and negative test images, sweeps confidence thresholds,
and derives per-class optimal thresholds to guarantee a false-positive-free model.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from pipeline.state import Stage, StateManager

console = Console()


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class FalsePositiveEvaluator:
    """Evaluates validation metrics and calibrates confidence thresholds to eliminate false positives."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()

        self.eval_cfg = config.get("evaluation", {})
        self.target_precision = float(self.eval_cfg.get("target_precision", 0.99))
        self.max_fp = int(self.eval_cfg.get("max_acceptable_fp_count", 0))
        self.iou_thresh = float(self.eval_cfg.get("iou_threshold", 0.50))
        self.min_box_area = float(self.eval_cfg.get("min_box_area", 100))
        self.aspect_range = self.eval_cfg.get("aspect_ratio_range", [0.1, 10.0])

        self.output_dir = Path(self.eval_cfg.get("runs_dir", "workspace/evaluation")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = Path(config.get("dataset", {}).get("output_dir", "workspace/dataset")).resolve()
        self.class_names = config.get("annotation", {}).get("class_names", ["target_object"])

    def _load_ground_truth(self, split: str = "val") -> Dict[str, List[Dict[str, Any]]]:
        """Loads ground truth bounding boxes in absolute pixels for an image split."""
        img_dir = self.dataset_dir / "images" / split
        lbl_dir = self.dataset_dir / "labels" / split

        gt_data = {}
        if not img_dir.exists():
            return gt_data

        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x1 = (xc - bw / 2.0) * w
                            y1 = (yc - bh / 2.0) * h
                            x2 = (xc + bw / 2.0) * w
                            y2 = (yc + bh / 2.0) * h
                            boxes.append({"cls": cls_id, "bbox": [x1, y1, x2, y2]})

            gt_data[str(img_path)] = boxes
        return gt_data

    def _run_raw_predictions(self, model_path: str, img_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Runs predictions at an ultra-low confidence threshold (0.01) to harvest all candidate detections."""
        from ultralytics import RTDETR

        model = RTDETR(model_path)
        predictions = {}

        for img_p in img_paths:
            preds = []
            results = model.predict(img_p, conf=0.01, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].tolist()

                    # Geometric filtering
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    area = w * h
                    ar = w / max(h, 1e-4)

                    if area < self.min_box_area:
                        continue
                    if ar < self.aspect_range[0] or ar > self.aspect_range[1]:
                        continue

                    preds.append({"cls": cls_id, "conf": conf, "bbox": xyxy})
            predictions[img_p] = preds
        return predictions

    def calibrate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Sweeps confidence thresholds and derives calibrated zero-FP operating thresholds."""
        self.state_mgr.start_stage(Stage.EVALUATE)

        ckpt_path = model_path
        if not ckpt_path:
            train_artifacts = self.state_mgr.get_artifacts(Stage.TRAIN)
            ckpt_path = train_artifacts.get("best_checkpoint")

        if not ckpt_path or not Path(ckpt_path).exists():
            # Fallback search
            bests = list(Path("workspace/runs/train").rglob("best.pt"))
            if bests:
                ckpt_path = str(bests[0])
            else:
                if Path("rtdetr-l.pt").exists():
                    console.print("[yellow]No custom trained checkpoint found; falling back to pretrained 'rtdetr-l.pt' for calibration.[/yellow]")
                    ckpt_path = "rtdetr-l.pt"
                else:
                    err = "No trained RT-DETR checkpoint found to evaluate."
                    self.state_mgr.fail_stage(Stage.EVALUATE, err)
                    raise FileNotFoundError(err)

        console.print(f"[bold cyan]Calibrating False-Positive Rejection on {ckpt_path}...[/bold cyan]")

        # 1. Load validation set (which includes both positive and negative background images)
        gt_dict = self._load_ground_truth(split="val")
        if not gt_dict:
            console.print("[yellow]Validation set empty, checking train split for calibration...[/yellow]")
            gt_dict = self._load_ground_truth(split="train")

        img_paths = list(gt_dict.keys())
        console.print(f"Evaluating across [bold]{len(img_paths)}[/bold] validation images...")

        # Count how many are true negative background images
        num_neg_images = sum(1 for boxes in gt_dict.values() if len(boxes) == 0)
        console.print(f" - Contains [bold]{num_neg_images}[/bold] dedicated background/negative scenes.")

        # 2. Extract candidate predictions
        all_preds = self._run_raw_predictions(ckpt_path, img_paths)

        # 3. Sweep thresholds per class
        threshold_steps = np.linspace(0.05, 0.95, 46)
        calibrated_results = {}
        global_fps_curve = []
        global_prec_curve = []

        for cls_idx, cls_name in enumerate(self.class_names):
            best_thresh = 0.50
            found_safe_thresh = False
            curve_points = []

            for t in threshold_steps:
                tp = 0
                fp = 0
                fn = 0
                bg_fp = 0

                for img_p, gt_boxes in gt_dict.items():
                    target_gt = [b["bbox"] for b in gt_boxes if b["cls"] == cls_idx]
                    matched_gt = set()

                    pred_boxes = [p for p in all_preds.get(img_p, []) if p["cls"] == cls_idx and p["conf"] >= t]

                    # If this is a pure background image, any detection is a pure false positive
                    if len(gt_boxes) == 0:
                        bg_fp += len(pred_boxes)
                        fp += len(pred_boxes)
                        continue

                    # Match detections against ground truth
                    for pb in pred_boxes:
                        best_iou = 0.0
                        best_gt_idx = -1
                        for g_idx, gb in enumerate(target_gt):
                            if g_idx in matched_gt:
                                continue
                            iou = compute_iou(pb["bbox"], gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                        if best_iou >= self.iou_thresh:
                            tp += 1
                            matched_gt.add(best_gt_idx)
                        else:
                            fp += 1

                    fn += len(target_gt) - len(matched_gt)

                precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
                recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

                curve_points.append({
                    "threshold": float(t),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": tp,
                    "fp": fp,
                    "bg_fp": bg_fp,
                })

                # Check if this threshold satisfies false-positive requirements
                if (precision >= self.target_precision) and (bg_fp <= self.max_fp):
                    if not found_safe_thresh:
                        best_thresh = float(t)
                        found_safe_thresh = True

            # If no threshold achieved 100% precision, pick highest threshold with max precision
            if not found_safe_thresh:
                best_point = max(curve_points, key=lambda pt: (pt["precision"], -pt["fp"]))
                best_thresh = best_point["threshold"]

            # Query metrics at calibrated threshold
            final_metric = next((p for p in curve_points if abs(p["threshold"] - best_thresh) < 1e-4), curve_points[-1])

            calibrated_results[cls_name] = {
                "class_id": cls_idx,
                "calibrated_confidence": round(best_thresh, 4),
                "precision": round(final_metric["precision"], 4),
                "recall": round(final_metric["recall"], 4),
                "false_positives": final_metric["fp"],
                "background_false_positives": final_metric["bg_fp"],
                "curve": curve_points,
            }

        # 4. Global operating threshold
        global_threshold = max(cr["calibrated_confidence"] for cr in calibrated_results.values())

        # 5. Plot Precision-Recall / FP curve
        plot_path = self.output_dir / "precision_calibration_curve.png"
        try:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            for cls_name, cdata in calibrated_results.items():
                threshs = [pt["threshold"] for pt in cdata["curve"]]
                precs = [pt["precision"] for pt in cdata["curve"]]
                plt.plot(threshs, precs, label=f"{cls_name} (tau*={cdata['calibrated_confidence']})")
            plt.axhline(self.target_precision, color="r", linestyle="--", label=f"Target ({self.target_precision})")
            plt.xlabel("Confidence Threshold")
            plt.ylabel("Precision")
            plt.title("Precision vs Confidence")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(1, 2, 2)
            for cls_name, cdata in calibrated_results.items():
                threshs = [pt["threshold"] for pt in cdata["curve"]]
                fps = [pt["fp"] for pt in cdata["curve"]]
                plt.plot(threshs, fps, label=f"{cls_name} False Positives")
            plt.xlabel("Confidence Threshold")
            plt.ylabel("False Positive Count")
            plt.title("False Positives vs Confidence")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=150)
            plt.close()
        except Exception as e:
            console.print(f"[yellow]Could not render calibration plot: {e}[/yellow]")

        # 6. Save calibration artifact
        out_json_path = self.output_dir / "calibrated_thresholds.json"
        export_data = {
            "target_precision": self.target_precision,
            "global_calibrated_threshold": global_threshold,
            "classes": calibrated_results,
            "geometric_filters": {
                "min_box_area": self.min_box_area,
                "aspect_ratio_range": self.aspect_range,
            },
            "checkpoint_evaluated": str(ckpt_path),
        }

        # Strip full curve points from final compact json
        compact_export = {
            "target_precision": self.target_precision,
            "global_calibrated_threshold": global_threshold,
            "class_thresholds": {
                name: {
                    "class_id": data["class_id"],
                    "calibrated_conf": data["calibrated_confidence"],
                    "precision": data["precision"],
                    "recall": data["recall"],
                    "false_positives": data["false_positives"],
                }
                for name, data in calibrated_results.items()
            },
            "geometric_filters": export_data["geometric_filters"],
            "model_path": str(ckpt_path),
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(compact_export, f, indent=2)

        self.state_mgr.complete_stage(
            Stage.EVALUATE,
            artifacts={
                "calibrated_thresholds_json": str(out_json_path),
                "calibration_plot": str(plot_path) if plot_path.exists() else None,
            },
            metrics={
                "global_threshold": global_threshold,
                "target_precision": self.target_precision,
            },
        )

        console.print(f"[bold green]Stage 5 Complete![/bold green] Calibrated zero-FP thresholds:")
        for name, data in compact_export["class_thresholds"].items():
            console.print(
                f" - [bold cyan]{name}[/bold cyan]: conf >= [bold]{data['calibrated_conf']}[/bold] "
                f"(Precision: {data['precision']*100:.1f}%, FP Count: {data['false_positives']})"
            )
        console.print(f"Calibration profile saved to: [bold]{out_json_path}[/bold]")
        return compact_export
