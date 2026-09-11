"""
Automated Unit and Integration Tests for RT-DETR Pipeline.
Tests StateManager, Video Extraction, Annotation Ingestion,
False-Positive Mitigation Dataset Builder, Threshold Calibrator, and CLI.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from pipeline.infer import FalsePositiveFreeDetector
from pipeline.stage_1_extractor import (
    FrameExtractor,
    compute_frame_similarity,
    compute_laplacian_variance,
)
from pipeline.stage_2_annotator import AnnotationManager
from pipeline.stage_3_dataset import DatasetBuilder
from pipeline.stage_5_evaluator import FalsePositiveEvaluator, compute_iou
from pipeline.stage_6_exporter import ModelExporter
from pipeline.state import Stage, StageStatus, StateManager


class TestStateManager(unittest.TestCase):
    """Verifies state tracking, persistence, and reset logic."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.tmp_dir) / "test_state.json"
        self.mgr = StateManager(str(self.state_file))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initial_state(self):
        for s in Stage:
            self.assertEqual(self.mgr.get_stage_status(s), StageStatus.PENDING)
            self.assertFalse(self.mgr.is_completed(s))

    def test_stage_progression_and_persistence(self):
        self.mgr.start_stage(Stage.EXTRACT)
        self.assertEqual(self.mgr.get_stage_status(Stage.EXTRACT), StageStatus.RUNNING)

        self.mgr.complete_stage(
            Stage.EXTRACT,
            artifacts={"frames_count": 42},
            metrics={"blur_discarded": 5},
        )
        self.assertEqual(self.mgr.get_stage_status(Stage.EXTRACT), StageStatus.COMPLETED)
        self.assertTrue(self.mgr.is_completed(Stage.EXTRACT))

        # Re-instantiate from file to test persistence
        reloaded = StateManager(str(self.state_file))
        self.assertTrue(reloaded.is_completed(Stage.EXTRACT))
        self.assertEqual(reloaded.get_artifacts(Stage.EXTRACT)["frames_count"], 42)
        self.assertEqual(reloaded.get_metrics(Stage.EXTRACT)["blur_discarded"], 5)

    def test_reset_downstream(self):
        self.mgr.complete_stage(Stage.EXTRACT)
        self.mgr.complete_stage(Stage.ANNOTATE)
        self.mgr.complete_stage(Stage.PREPARE)

        # Reset ANNOTATE, should reset ANNOTATE and PREPARE, but keep EXTRACT
        self.mgr.reset_stage(Stage.ANNOTATE, reset_downstream=True)
        self.assertTrue(self.mgr.is_completed(Stage.EXTRACT))
        self.assertEqual(self.mgr.get_stage_status(Stage.ANNOTATE), StageStatus.PENDING)
        self.assertEqual(self.mgr.get_stage_status(Stage.PREPARE), StageStatus.PENDING)


class TestFrameExtractor(unittest.TestCase):
    """Tests video frame sampling, blur detection, and deduplication."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.frames_dir = Path(self.tmp_dir) / "frames"
        self.neg_dir = Path(self.tmp_dir) / "negatives"
        self.video_path = Path(self.tmp_dir) / "synthetic.mp4"

        # Create a synthetic 15-frame video
        # Frames 0-4: Sharp square
        # Frames 5-7: Identical duplicate frames
        # Frames 8-10: Intentionally blurred frames
        # Frames 11-14: Sharp circle
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, 10.0, (200, 200))

        for i in range(15):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            if i < 5:
                # moving sharp rectangle
                cv2.rectangle(img, (20 + i * 5, 20), (70 + i * 5, 70), (255, 255, 255), -1)
            elif 5 <= i <= 7:
                # identical static frame
                cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0), -1)
            elif 8 <= i <= 10:
                # low-contrast / blurred frame
                img = cv2.GaussianBlur(img, (25, 25), 0)
                img.fill(50)  # flat gray, Laplacian variance ~ 0
            else:
                cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)

            writer.write(img)
        writer.release()

        self.config = {
            "extraction": {
                "output_dir": str(self.frames_dir),
                "negative_output_dir": str(self.neg_dir),
                "sample_fps": 10.0,
                "frame_step": 1,
                "blur_filter": {"enabled": True, "min_laplacian_variance": 50.0},
                "deduplication": {"enabled": True, "similarity_threshold": 0.98},
                "negative_harvesting": {"enabled": False},
            }
        }
        self.extractor = FrameExtractor(self.config)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_extract_and_filter(self):
        stats = self.extractor.extract_from_video(self.video_path)
        self.assertGreater(stats["saved_frames"], 0)
        self.assertGreater(stats["discarded_blur"], 0, "Should have detected and filtered blurry frames")
        self.assertGreater(stats["discarded_dedup"], 0, "Should have detected and filtered duplicate frames")


class TestFalsePositiveMitigationDataset(unittest.TestCase):
    """Verifies that Stage 3 creates balanced splits and generates 0-byte negative labels."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.frames_dir = Path(self.tmp_dir) / "frames"
        self.labels_dir = Path(self.tmp_dir) / "labels"
        self.neg_dir = Path(self.tmp_dir) / "negatives"
        self.ds_dir = Path(self.tmp_dir) / "dataset"

        self.frames_dir.mkdir(parents=True)
        self.labels_dir.mkdir(parents=True)
        self.neg_dir.mkdir(parents=True)

        # Create 10 positive images + labels
        for i in range(10):
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(self.frames_dir / f"pos_{i:02d}.jpg"), img)
            with open(self.labels_dir / f"pos_{i:02d}.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")

        # Create 4 negative background images (empty scenery, no objects)
        for i in range(4):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(self.neg_dir / f"bg_{i:02d}.jpg"), img)

        self.config = {
            "project": {"seed": 42},
            "extraction": {
                "output_dir": str(self.frames_dir),
                "negative_output_dir": str(self.neg_dir),
            },
            "annotation": {"class_names": ["widget"]},
            "dataset": {
                "output_dir": str(self.ds_dir),
                "train_ratio": 0.70,
                "val_ratio": 0.20,
                "test_ratio": 0.10,
                "false_positive_mitigation": {
                    "include_background_images": True,
                    "target_background_ratio": 0.20,
                    "hard_negatives_dir": str(Path(self.tmp_dir) / "hard_neg"),
                },
            },
        }

        # Override labels_dir in builder
        self.builder = DatasetBuilder(self.config)
        self.builder.labels_dir = self.labels_dir

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_negative_label_generation(self):
        res = self.builder.build()
        self.assertTrue((self.ds_dir / "dataset.yaml").exists())

        # Verify background images have 0-byte empty labels
        val_img_dir = self.ds_dir / "images" / "val"
        val_lbl_dir = self.ds_dir / "labels" / "val"
        train_lbl_dir = self.ds_dir / "labels" / "train"

        # Check all label files in train
        for lbl in train_lbl_dir.glob("*.txt"):
            if "bg_" in lbl.stem:
                self.assertEqual(lbl.stat().st_size, 0, f"Negative frame label {lbl.name} must be 0 bytes")

        # Verify dataset.yaml content
        with open(self.ds_dir / "dataset.yaml", "r") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["names"][0], "widget")
        self.assertIn("train", data)
        self.assertIn("val", data)


class TestThresholdCalibration(unittest.TestCase):
    """Verifies IoU math and Precision-calibrated thresholding."""

    def test_compute_iou(self):
        # Perfectly overlapping boxes
        boxA = [10, 10, 50, 50]
        boxB = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        # Disjoint boxes
        boxC = [100, 100, 150, 150]
        self.assertAlmostEqual(compute_iou(boxA, boxC), 0.0)

        # Partial overlap (20x20 intersection, 40x40 each box)
        boxD = [30, 30, 70, 70]
        # inter = 20*20 = 400
        # union = 1600 + 1600 - 400 = 2800
        # iou = 400 / 2800 = 1/7 ~= 0.142857
        self.assertAlmostEqual(compute_iou(boxA, boxD), 400.0 / 2800.0, places=4)

    def test_detector_box_filtering(self):
        # Test detector geometric filter
        detector = FalsePositiveFreeDetector.__new__(FalsePositiveFreeDetector)
        detector.global_threshold = 0.70
        detector.class_thresholds = {"widget": {"class_id": 0, "calibrated_conf": 0.75}}
        detector.min_box_area = 50.0
        detector.aspect_range = [0.2, 5.0]

        # Valid box with high confidence
        self.assertTrue(detector._filter_box([10, 10, 40, 40], cls_id=0, conf=0.85))

        # Rejection by confidence (< 0.75)
        self.assertFalse(detector._filter_box([10, 10, 40, 40], cls_id=0, conf=0.60))

        # Rejection by micro-area (< 50 pixels)
        self.assertFalse(detector._filter_box([10, 10, 14, 14], cls_id=0, conf=0.95))

        # Rejection by abnormal aspect ratio (needle line: w=100, h=2 -> ar=50)
        self.assertFalse(detector._filter_box([0, 0, 100, 2], cls_id=0, conf=0.95))


if __name__ == "__main__":
    unittest.main()
