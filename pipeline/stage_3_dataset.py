"""
Stage 3: Dataset Preparation & False-Positive Suppression Engine.
Partitions data into train/val/test splits, balances foreground vs. background images,
and generates 0-byte negative sample labels to eliminate false positives in RT-DETR.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.console import Console

from pipeline.state import Stage, StateManager

console = Console()


class DatasetBuilder:
    """Constructs an RT-DETR compatible dataset with negative background sample injection."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()

        self.ds_cfg = config.get("dataset", {})
        self.output_dir = Path(self.ds_cfg.get("output_dir", "workspace/dataset")).resolve()

        # Split ratios
        self.train_ratio = float(self.ds_cfg.get("train_ratio", 0.70))
        self.val_ratio = float(self.ds_cfg.get("val_ratio", 0.20))
        self.test_ratio = float(self.ds_cfg.get("test_ratio", 0.10))

        # False-positive mitigation configuration
        fp_cfg = self.ds_cfg.get("false_positive_mitigation", {})
        self.include_background = fp_cfg.get("include_background_images", True)
        self.target_bg_ratio = float(fp_cfg.get("target_background_ratio", 0.15))
        self.hard_negatives_dir = Path(fp_cfg.get("hard_negatives_dir", "data/hard_negatives"))

        # Source paths
        self.frames_dir = Path(config.get("extraction", {}).get("output_dir", "workspace/frames")).resolve()
        self.negatives_dir = Path(config.get("extraction", {}).get("negative_output_dir", "workspace/negatives")).resolve()
        self.labels_dir = Path("workspace/labels").resolve()

        self.class_names = config.get("annotation", {}).get("class_names", ["target_object"])
        self.seed = int(config.get("project", {}).get("seed", 42))

    def _gather_positive_samples(self) -> List[Tuple[Path, Path]]:
        """Gathers images that have non-empty annotation files."""
        pairs = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not self.frames_dir.exists():
            return pairs

        for img_path in sorted(self.frames_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue

            lbl_path = self.labels_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                # Check if it actually contains bounding boxes
                with open(lbl_path, "r", encoding="utf-8") as f:
                    content = [l.strip() for l in f if l.strip()]
                if content:
                    pairs.append((img_path, lbl_path))

        return pairs

    def _gather_negative_samples(self) -> List[Path]:
        """Gathers background images without objects (from extraction negatives & hard negatives)."""
        negatives = []
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # 1. Harvested negatives from Stage 1
        if self.negatives_dir.exists():
            for p in self.negatives_dir.iterdir():
                if p.suffix.lower() in image_exts:
                    negatives.append(p)

        # 2. Hard negatives directory (curated scenes triggering false positives)
        if self.hard_negatives_dir.exists():
            for p in self.hard_negatives_dir.rglob("*"):
                if p.suffix.lower() in image_exts:
                    negatives.append(p)

        # Note: Do NOT treat unlabeled positive frames as negatives!
        # Dedicated background scenes should only be harvested from negative_videos or hard_negatives.

        # Deduplicate
        unique_negatives = list({p.resolve(): p for p in negatives}.values())
        return sorted(unique_negatives)

    def _split_data(self, items: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        """Splits items into train, val, test based on configured ratios."""
        rng = random.Random(self.seed)
        shuffled = list(items)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * self.train_ratio))
        n_val = int(round(n * self.val_ratio))

        # Ensure at least 1 in val and test if dataset is very small
        if n > 2 and n_val == 0 and self.val_ratio > 0:
            n_val = 1
            n_train = max(1, n_train - 1)

        train_items = shuffled[:n_train]
        val_items = shuffled[n_train : n_train + n_val]
        test_items = shuffled[n_train + n_val :]

        return train_items, val_items, test_items

    def build(self) -> Dict[str, Any]:
        """Constructs the dataset folder structure and writes dataset.yaml."""
        self.state_mgr.start_stage(Stage.PREPARE)
        console.print("[bold cyan]Building RT-DETR Dataset with False-Positive Suppression...[/bold cyan]")

        pos_samples = self._gather_positive_samples()
        console.print(f"Found [bold]{len(pos_samples)}[/bold] labeled foreground images.")

        if len(pos_samples) == 0:
            err = (
                "Found 0 labeled foreground images! You cannot train an object detection model without annotations. "
                "Please run Stage 2 (Annotation / Auto-Labeling) first to generate bounding boxes."
            )
            self.state_mgr.fail_stage(Stage.PREPARE, err)
            raise ValueError(err)

        neg_samples = self._gather_negative_samples() if self.include_background else []
        console.print(f"Found [bold]{len(neg_samples)}[/bold] available background/negative images.")

        # Determine target background count based on target_bg_ratio
        # target_bg_ratio = num_neg / (num_pos + num_neg)
        # num_neg = target_bg_ratio * num_pos / (1 - target_bg_ratio)
        if pos_samples and neg_samples and self.target_bg_ratio > 0:
            target_count = int(round((self.target_bg_ratio * len(pos_samples)) / (1.0 - self.target_bg_ratio)))
            selected_negatives = neg_samples[: max(1, min(len(neg_samples), target_count))]
        else:
            selected_negatives = neg_samples

        console.print(f"Selected [bold]{len(selected_negatives)}[/bold] background images for false-positive defense.")

        # Create output directories
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Split positive samples
        pos_train, pos_val, pos_test = self._split_data(pos_samples)

        # Split negative samples
        neg_train, neg_val, neg_test = self._split_data(selected_negatives)

        # Helper to copy samples into destination splits
        def copy_positives(items: List[Tuple[Path, Path]], split_name: str):
            for img_src, lbl_src in items:
                img_dst = self.output_dir / "images" / split_name / img_src.name
                lbl_dst = self.output_dir / "labels" / split_name / lbl_src.name
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)

        def copy_negatives(items: List[Path], split_name: str):
            for img_src in items:
                img_dst = self.output_dir / "images" / split_name / img_src.name
                # Empty 0-byte label file for negative background image
                lbl_dst = self.output_dir / "labels" / split_name / f"{img_src.stem}.txt"
                shutil.copy2(img_src, img_dst)
                with open(lbl_dst, "w", encoding="utf-8") as f:
                    pass  # 0-byte empty file teaches RT-DETR: no objects here!

        copy_positives(pos_train, "train")
        copy_positives(pos_val, "val")
        copy_positives(pos_test, "test")

        copy_negatives(neg_train, "train")
        copy_negatives(neg_val, "val")
        copy_negatives(neg_test, "test")

        # Generate dataset.yaml for RT-DETR
        yaml_data = {
            "path": str(self.output_dir).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

        counts = {
            "train_pos": len(pos_train),
            "train_neg": len(neg_train),
            "val_pos": len(pos_val),
            "val_neg": len(neg_val),
            "test_pos": len(pos_test),
            "test_neg": len(neg_test),
            "total_images": len(pos_train) + len(pos_val) + len(pos_test) + len(selected_negatives),
            "yaml_path": str(yaml_path),
        }

        self.state_mgr.complete_stage(
            Stage.PREPARE,
            artifacts={
                "dataset_dir": str(self.output_dir),
                "dataset_yaml": str(yaml_path),
            },
            metrics=counts,
        )

        console.print(
            f"[bold green]Stage 3 Complete:[/bold green] Dataset constructed at {self.output_dir}\n"
            f" - Train: {counts['train_pos']} positive, {counts['train_neg']} background\n"
            f" - Val:   {counts['val_pos']} positive, {counts['val_neg']} background\n"
            f" - Test:  {counts['test_pos']} positive, {counts['test_neg']} background\n"
            f" - Dataset config: {yaml_path}"
        )
        return counts

    run = build

