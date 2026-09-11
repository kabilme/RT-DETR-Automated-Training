"""
Stage 4: RT-DETR Model Trainer.
Wraps Ultralytics RT-DETR training with automatic device management,
fault-tolerant checkpointing, and resume support.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from rich.console import Console

from pipeline.state import Stage, StateManager

console = Console()


class RTDETRTrainer:
    """Manages RT-DETR model training and checkpointing."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()

        self.train_cfg = config.get("training", {})
        self.proj_cfg = config.get("project", {})

        self.model_name = self.train_cfg.get("model_architecture", "rtdetr-l.pt")
        self.imgsz = int(self.train_cfg.get("imgsz", 640))
        self.epochs = int(self.train_cfg.get("epochs", 50))
        self.batch_size = int(self.train_cfg.get("batch_size", 4))
        self.workers = int(self.train_cfg.get("workers", 2))
        self.patience = int(self.train_cfg.get("patience", 15))
        self.optimizer = self.train_cfg.get("optimizer", "AdamW")
        self.lr0 = float(self.train_cfg.get("lr0", 0.0001))
        self.lrf = float(self.train_cfg.get("lrf", 0.01))
        self.weight_decay = float(self.train_cfg.get("weight_decay", 0.0001))
        self.save_period = int(self.train_cfg.get("save_period", 5))
        self.resume_allowed = self.train_cfg.get("resume", True)

        # Hardware setup
        req_device = self.proj_cfg.get("device", "auto")
        if req_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = req_device

        self.runs_dir = Path("workspace/runs/train").resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Locates the latest checkpoint (last.pt) if training was interrupted."""
        checkpoints = list(self.runs_dir.rglob("weights/last.pt"))
        if checkpoints:
            # Return the most recently modified last.pt
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None

    def train(self, dataset_yaml: Optional[str] = None, override_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Runs or resumes RT-DETR training."""
        from ultralytics import RTDETR

        self.state_mgr.start_stage(Stage.TRAIN)

        data_yaml_path = dataset_yaml
        if not data_yaml_path:
            # Check state or default path
            prep_artifacts = self.state_mgr.get_artifacts(Stage.PREPARE)
            data_yaml_path = prep_artifacts.get("dataset_yaml", "workspace/dataset/dataset.yaml")

        if not Path(data_yaml_path).exists():
            err = f"Dataset yaml not found at: {data_yaml_path}. Run 'prepare' stage first."
            self.state_mgr.fail_stage(Stage.TRAIN, err)
            raise FileNotFoundError(err)

        epochs = override_epochs if override_epochs is not None else self.epochs

        console.print(f"[bold cyan]Initializing RT-DETR Training[/bold cyan]")
        console.print(f" - Model Architecture: {self.model_name}")
        console.print(f" - Compute Device:     {self.device.upper()}")
        console.print(f" - Target Epochs:      {epochs}")
        console.print(f" - Batch Size:         {self.batch_size}")
        console.print(f" - Image Size:         {self.imgsz}x{self.imgsz}")

        # Check for resumable checkpoint
        latest_ckpt = self.find_latest_checkpoint() if self.resume_allowed else None
        if latest_ckpt and latest_ckpt.exists():
            console.print(f"[bold yellow]Found previous training checkpoint at {latest_ckpt}. Resuming...[/bold yellow]")
            model = RTDETR(str(latest_ckpt))
            train_kwargs = {
                "resume": True,
            }
        else:
            console.print(f"Starting fresh training from {self.model_name}...")
            model = RTDETR(self.model_name)
            train_kwargs = {
                "data": str(data_yaml_path),
                "epochs": epochs,
                "batch": self.batch_size,
                "imgsz": self.imgsz,
                "device": self.device,
                "workers": self.workers,
                "patience": self.patience,
                "optimizer": self.optimizer,
                "lr0": self.lr0,
                "lrf": self.lrf,
                "weight_decay": self.weight_decay,
                "save_period": self.save_period,
                "project": str(self.runs_dir),
                "name": "rtdetr_run",
                "exist_ok": True,
                "verbose": True,
            }

        try:
            results = model.train(**train_kwargs)
        except Exception as e:
            self.state_mgr.fail_stage(Stage.TRAIN, str(e))
            raise

        # Find best checkpoint
        best_ckpt = self.runs_dir / "rtdetr_run" / "weights" / "best.pt"
        if not best_ckpt.exists():
            # Fallback search
            all_bests = list(self.runs_dir.rglob("weights/best.pt"))
            best_ckpt = max(all_bests, key=lambda p: p.stat().st_mtime) if all_bests else None

        last_ckpt = self.runs_dir / "rtdetr_run" / "weights" / "last.pt"

        metrics = {
            "device": self.device,
            "best_checkpoint": str(best_ckpt) if best_ckpt else None,
            "last_checkpoint": str(last_ckpt) if last_ckpt and last_ckpt.exists() else None,
        }

        # Extract final metrics if available
        if hasattr(results, "results_dict"):
            metrics.update({k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))})

        self.state_mgr.complete_stage(
            Stage.TRAIN,
            artifacts={
                "best_checkpoint": str(best_ckpt) if best_ckpt else None,
                "last_checkpoint": str(last_ckpt) if last_ckpt and last_ckpt.exists() else None,
                "runs_dir": str(self.runs_dir / "rtdetr_run"),
            },
            metrics=metrics,
        )

        console.print(f"[bold green]Stage 4 Training Complete![/bold green]")
        if best_ckpt:
            console.print(f"Best model weights saved to: [bold]{best_ckpt}[/bold]")
        return metrics
