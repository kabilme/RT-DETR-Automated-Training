"""
Pipeline State Manager: Provides stage tracking, state persistence, checkpointing,
and resume capability so the user can pause and continue at any stage.
"""

import json
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Stage(str, Enum):
    EXTRACT = "extract"
    ANNOTATE = "annotate"
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"
    EXPORT = "export"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


STAGE_ORDER: List[Stage] = [
    Stage.EXTRACT,
    Stage.ANNOTATE,
    Stage.PREPARE,
    Stage.TRAIN,
    Stage.EVALUATE,
    Stage.EXPORT,
]


STAGE_ARTIFACTS: Dict[Stage, List[str]] = {
    Stage.EXTRACT: [
        "workspace/frames",
        "workspace/negatives",
        "workspace/frames_manifest.json",
    ],
    Stage.ANNOTATE: [
        "workspace/labels",
        "workspace/inspection",
    ],
    Stage.PREPARE: [
        "workspace/dataset",
    ],
    Stage.TRAIN: [
        "workspace/runs",
    ],
    Stage.EVALUATE: [
        "workspace/evaluation",
    ],
    Stage.EXPORT: [
        "workspace/exported_models",
    ],
}


class StateManager:
    """Manages persistent execution state for pipeline stages."""

    def __init__(self, state_file: str = "workspace/pipeline_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = self._load()

    def _init_empty_state(self) -> Dict[str, Any]:
        stages = {}
        for s in STAGE_ORDER:
            stages[s.value] = {
                "status": StageStatus.PENDING.value,
                "started_at": None,
                "completed_at": None,
                "duration_sec": 0.0,
                "artifacts": {},
                "metrics": {},
                "error": None,
            }
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": None,
            "stages": stages,
        }

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Backfill any missing stages
                    for s in STAGE_ORDER:
                        if s.value not in data.get("stages", {}):
                            data.setdefault("stages", {})[s.value] = {
                                "status": StageStatus.PENDING.value,
                                "started_at": None,
                                "completed_at": None,
                                "duration_sec": 0.0,
                                "artifacts": {},
                                "metrics": {},
                                "error": None,
                            }
                    return data
            except Exception:
                return self._init_empty_state()
        return self._init_empty_state()

    def save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_stage_status(self, stage: Stage) -> StageStatus:
        st_data = self.data["stages"].get(stage.value, {})
        return StageStatus(st_data.get("status", StageStatus.PENDING.value))

    def is_completed(self, stage: Stage) -> bool:
        if self.get_stage_status(stage) != StageStatus.COMPLETED:
            return False
        # Physically verify essential stage artifacts exist on disk
        try:
            if stage == Stage.EXTRACT:
                f_dir = Path(self.data["stages"][stage.value].get("artifacts", {}).get("frames_dir", "workspace/frames"))
                if not f_dir.exists() or len(list(f_dir.glob("*.jpg")) + list(f_dir.glob("*.png"))) == 0:
                    return False
            elif stage == Stage.ANNOTATE:
                l_dir = Path(self.data["stages"][stage.value].get("artifacts", {}).get("labels_dir", "workspace/labels"))
                if not l_dir.exists() or len(list(l_dir.glob("*.txt"))) == 0:
                    return False
            elif stage == Stage.PREPARE:
                yaml_p = Path("workspace/dataset/dataset.yaml")
                if not yaml_p.exists():
                    return False
            elif stage == Stage.TRAIN:
                ckpt = self.data["stages"][stage.value].get("artifacts", {}).get("best_checkpoint")
                if not ckpt or not Path(ckpt).exists():
                    if not list(Path("workspace/runs").rglob("best.pt")):
                        return False
            elif stage == Stage.EVALUATE:
                rep = Path("workspace/evaluation/calibration_report.json")
                if not rep.exists():
                    return False
            elif stage == Stage.EXPORT:
                exp_dir = Path("workspace/exported_models")
                if not exp_dir.exists() or len(list(exp_dir.glob("*.*"))) == 0:
                    return False
        except Exception:
            pass
        return True

    def start_stage(self, stage: Stage) -> None:
        self.data["stages"][stage.value]["status"] = StageStatus.RUNNING.value
        self.data["stages"][stage.value]["started_at"] = datetime.now().isoformat()
        self.data["stages"][stage.value]["error"] = None
        self.save()

    def complete_stage(
        self,
        stage: Stage,
        artifacts: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        st_dict = self.data["stages"][stage.value]
        st_dict["status"] = StageStatus.COMPLETED.value
        now = datetime.now()
        st_dict["completed_at"] = now.isoformat()
        if st_dict.get("started_at"):
            try:
                start_dt = datetime.fromisoformat(st_dict["started_at"])
                st_dict["duration_sec"] = round((now - start_dt).total_seconds(), 2)
            except Exception:
                st_dict["duration_sec"] = 0.0
        if artifacts:
            st_dict.setdefault("artifacts", {}).update(artifacts)
        if metrics:
            st_dict.setdefault("metrics", {}).update(metrics)
        self.data["last_completed_stage"] = stage.value
        self.save()

    def fail_stage(self, stage: Stage, error_message: str) -> None:
        st_dict = self.data["stages"][stage.value]
        st_dict["status"] = StageStatus.FAILED.value
        st_dict["error"] = error_message
        self.save()

    def reset_stage(self, stage: Stage, reset_downstream: bool = True, clean_disk: bool = True) -> None:
        """Reset a stage, and optionally reset downstream stages dependent on it, clearing disk data."""
        import gc
        import shutil

        stage_idx = STAGE_ORDER.index(stage)
        stages_to_reset = STAGE_ORDER[stage_idx:] if reset_downstream else [stage]

        for s in stages_to_reset:
            self.data["stages"][s.value] = {
                "status": StageStatus.PENDING.value,
                "started_at": None,
                "completed_at": None,
                "duration_sec": 0.0,
                "artifacts": {},
                "metrics": {},
                "error": None,
            }

            if clean_disk and s in STAGE_ARTIFACTS:
                gc.collect()
                for rel_path in STAGE_ARTIFACTS[s]:
                    p = Path(rel_path)
                    if p.exists():
                        try:
                            if p.is_dir():
                                shutil.rmtree(p, ignore_errors=True)
                                p.mkdir(parents=True, exist_ok=True)
                            elif p.is_file():
                                p.unlink(missing_ok=True)
                        except Exception:
                            pass

        # Update last completed stage
        last_completed = None
        for s in STAGE_ORDER:
            if self.data["stages"][s.value]["status"] == StageStatus.COMPLETED.value:
                last_completed = s.value
        self.data["last_completed_stage"] = last_completed
        self.save()

    def reset_all(self, clean_disk: bool = True) -> None:
        """Resets all stages to PENDING and clears all generated workspace artifacts."""
        import shutil
        self.reset_stage(STAGE_ORDER[0], reset_downstream=True, clean_disk=clean_disk)
        if clean_disk:
            for extra in ["workspace/inference_output", "workspace/web_uploads"]:
                p = Path(extra)
                if p.exists() and p.is_dir():
                    for item in p.iterdir():
                        if item.name != ".gitkeep":
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item, ignore_errors=True)
                                else:
                                    item.unlink(missing_ok=True)
                            except Exception:
                                pass

    def get_next_pending_stage(self) -> Optional[Stage]:
        for s in STAGE_ORDER:
            if self.data["stages"][s.value]["status"] != StageStatus.COMPLETED.value:
                return s
        return None

    def get_artifacts(self, stage: Stage) -> Dict[str, Any]:
        return self.data["stages"][stage.value].get("artifacts", {})

    def get_metrics(self, stage: Stage) -> Dict[str, Any]:
        return self.data["stages"][stage.value].get("metrics", {})

    def print_summary(self) -> None:
        """Prints formatted summary table of all stages."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="RT-DETR Pipeline Execution State", show_header=True)
            table.add_column("Stage", style="cyan", no_wrap=True)
            table.add_column("Status", style="bold")
            table.add_column("Duration (s)", justify="right")
            table.add_column("Details / Artifacts")

            status_colors = {
                StageStatus.COMPLETED.value: "[green]COMPLETED[/green]",
                StageStatus.RUNNING.value: "[yellow]RUNNING[/yellow]",
                StageStatus.FAILED.value: "[red]FAILED[/red]",
                StageStatus.PENDING.value: "[dim]PENDING[/dim]",
                StageStatus.SKIPPED.value: "[blue]SKIPPED[/blue]",
            }

            for s in STAGE_ORDER:
                info = self.data["stages"][s.value]
                stat_display = status_colors.get(info["status"], info["status"])
                dur = f"{info.get('duration_sec', 0.0):.1f}"
                details = []
                if info.get("error"):
                    details.append(f"[red]Error: {info['error']}[/red]")
                elif info.get("metrics"):
                    m_str = ", ".join(f"{k}={v}" for k, v in list(info["metrics"].items())[:3])
                    details.append(f"[dim]{m_str}[/dim]")
                elif info.get("artifacts"):
                    a_keys = ", ".join(info["artifacts"].keys())
                    details.append(f"[dim]Artifacts: {a_keys}[/dim]")
                detail_str = "; ".join(details) if details else "-"
                table.add_row(s.value.upper(), stat_display, dur, detail_str)

            console.print(table)
        except ImportError:
            print("\n=== RT-DETR Pipeline State ===")
            for s in STAGE_ORDER:
                info = self.data["stages"][s.value]
                print(f" - {s.value.upper():<10} : {info['status'].upper()} ({info.get('duration_sec', 0.0):.1f}s)")
            print("==============================\n")
