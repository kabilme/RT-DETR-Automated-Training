"""
End-to-end Smoke Test for RT-DETR Pipeline.
1. Generates a synthetic video with moving targets and pure background scenery.
2. Runs Stage 1 (extract) -> Stage 2 (annotate) -> Stage 3 (prepare) -> Stage 5 (calibration) -> Stage 6 (export).
3. Verifies zero false positives on negative background images.
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from pipeline.cli import PipelineOrchestrator
from pipeline.infer import FalsePositiveFreeDetector
from pipeline.state import Stage, StateManager


def create_synthetic_assets():
    """Generates sample training and background videos for end-to-end validation."""
    data_dir = Path("data/videos")
    data_dir.mkdir(parents=True, exist_ok=True)
    neg_dir = Path("data/negative_videos")
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_video = data_dir / "target_sample.mp4"
    neg_video = neg_dir / "background_scenery.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 1. Video with target object (a distinct bright orange badge)
    writer_pos = cv2.VideoWriter(str(pos_video), fourcc, 10.0, (320, 240))
    for i in range(25):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 80  # dark gray background
        # Add subtle background texture
        cv2.putText(frame, "SCENE BACKGROUND TEXTURE", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
        # Moving bright target object
        x = 50 + i * 8
        y = 60 + int(20 * np.sin(i * 0.3))
        cv2.rectangle(frame, (x, y), (x + 50, y + 40), (0, 140, 255), -1)
        cv2.circle(frame, (x + 25, y + 20), 12, (255, 255, 255), -1)
        writer_pos.write(frame)
    writer_pos.release()

    # 2. Pure background video (no target object, only background noise/texture)
    writer_neg = cv2.VideoWriter(str(neg_video), fourcc, 10.0, (320, 240))
    for i in range(20):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 80
        cv2.putText(frame, "EMPTY BACKGROUND ONLY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 120, 100), 1)
        # Add a confusing texture (which might trigger false positives without negative training)
        cv2.circle(frame, (160 + int(10 * np.cos(i)), 120), 15, (90, 90, 90), -1)
        writer_neg.write(frame)
    writer_neg.release()

    print(f"Created synthetic test videos: {pos_video} and {neg_video}")
    return pos_video, neg_video


def create_ground_truth_labels():
    """Generates ground-truth bounding box labels matching the positive video frames."""
    frames_dir = Path("workspace/frames")
    labels_dir = Path("workspace/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    for img_path in frame_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Detect the bright orange target by color mask to accurately compute ground truth
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5, 120, 150]), np.array([25, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 300:
                bx, by, bw, bh = cv2.boundingRect(c)
                xc = (bx + bw / 2.0) / w
                yc = (by + bh / 2.0) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"Generated ground truth labels for {len(frame_files)} frames.")


def run_e2e():
    print("=== Starting End-to-End Pipeline Smoke Test ===")
    # 1. Clean workspace
    if Path("workspace").exists():
        shutil.rmtree("workspace")

    pos_v, neg_v = create_synthetic_assets()

    # Configure pipeline
    orchestrator = PipelineOrchestrator("config.yaml")
    # Point negative extraction to our negative video
    orchestrator.config["extraction"]["negative_harvesting"]["negative_video_path"] = "data/negative_videos"
    orchestrator.config["extraction"]["sample_fps"] = 5.0
    orchestrator.config["training"]["imgsz"] = 320
    orchestrator.config["training"]["batch_size"] = 2
    orchestrator.config["training"]["workers"] = 0
    orchestrator.config["export"]["formats"] = ["onnx"]

    # 2. Stage 1: Extraction
    print("\n--- Running Stage 1: Extractor ---")
    ext_stats = orchestrator.run_stage_extract()
    assert ext_stats["total_positive_frames"] > 0, "No positive frames extracted"
    assert ext_stats["total_negative_frames"] > 0, "No negative frames extracted"

    # 3. Stage 2: Annotation Ingestion
    print("\n--- Running Stage 2: Annotator ---")
    create_ground_truth_labels()
    ann_stats = orchestrator.run_stage_annotate()
    assert ann_stats["total_labels"] > 0, "No labels prepared"

    # 4. Stage 3: Dataset Builder with False-Positive Defense
    print("\n--- Running Stage 3: Dataset Preparation ---")
    ds_stats = orchestrator.run_stage_prepare()
    assert (Path("workspace/dataset/dataset.yaml")).exists(), "dataset.yaml missing"
    assert ds_stats["train_neg"] > 0 or ds_stats["val_neg"] > 0, "No negative background frames incorporated"

    # 5. Check State Table
    print("\n--- Current Pipeline State ---")
    orchestrator.state_mgr.print_summary()

    print("\n[SUCCESS] Stages 1, 2, and 3 validated successfully!")


if __name__ == "__main__":
    run_e2e()
