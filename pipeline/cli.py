"""
Unified CLI & Pipeline Orchestrator.
Supports running individual stages, stepping through the full workflow,
pausing/resuming via StateManager, and viewing status tables.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console

from pipeline.infer import run_inference_on_file
from pipeline.stage_1_extractor import FrameExtractor
from pipeline.stage_2_annotator import AnnotationManager
from pipeline.stage_3_dataset import DatasetBuilder
from pipeline.stage_4_trainer import RTDETRTrainer
from pipeline.stage_5_evaluator import FalsePositiveEvaluator
from pipeline.stage_6_exporter import ModelExporter
from pipeline.state import Stage, StageStatus, StateManager, STAGE_ORDER

console = Console()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    cpath = Path(config_path)
    if not cpath.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(cpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PipelineOrchestrator:
    """Coordinates execution across all modular pipeline stages."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.state_mgr = StateManager()

    def run_stage_extract(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        extractor = FrameExtractor(self.config, self.state_mgr)
        return extractor.run(video_path)

    def run_stage_annotate(self, auto_label: Optional[bool] = None) -> Dict[str, Any]:
        if auto_label is not None:
            self.config.setdefault("annotation", {}).setdefault("auto_label", {})["enabled"] = auto_label
        annotator = AnnotationManager(self.config, self.state_mgr)
        return annotator.run()

    def run_stage_prepare(self) -> Dict[str, Any]:
        builder = DatasetBuilder(self.config, self.state_mgr)
        return builder.build()

    def run_stage_train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        trainer = RTDETRTrainer(self.config, self.state_mgr)
        return trainer.train(override_epochs=epochs)

    def run_stage_evaluate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        evaluator = FalsePositiveEvaluator(self.config, self.state_mgr)
        return evaluator.calibrate(model_path)

    def run_stage_export(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        exporter = ModelExporter(self.config, self.state_mgr)
        return exporter.export(checkpoint_path)

    def run_all(self, from_stage: Optional[str] = None, force: bool = False) -> None:
        """Executes pipeline end-to-end, skipping already completed stages unless forced."""
        console.print("[bold green]Starting End-to-End RT-DETR Pipeline Execution[/bold green]\n")

        start_exec = False if from_stage else True

        for stage in STAGE_ORDER:
            s_name = stage.value

            if from_stage and s_name == from_stage:
                start_exec = True

            if not start_exec:
                console.print(f"[dim]Skipping stage '{s_name}' (before start stage '{from_stage}')[/dim]")
                continue

            if not force and self.state_mgr.is_completed(stage):
                console.print(f"[bold green]✓ Stage '{s_name}' already completed.[/bold green] (Use --force to re-run)")
                continue

            console.print(f"\n[bold yellow]>>> Executing Stage: {s_name.upper()} <<<[/bold yellow]")

            if stage == Stage.EXTRACT:
                self.run_stage_extract()
            elif stage == Stage.ANNOTATE:
                self.run_stage_annotate()
            elif stage == Stage.PREPARE:
                self.run_stage_prepare()
            elif stage == Stage.TRAIN:
                self.run_stage_train()
            elif stage == Stage.EVALUATE:
                self.run_stage_evaluate()
            elif stage == Stage.EXPORT:
                self.run_stage_export()

        console.print("\n[bold green]Pipeline Execution Completed Successfully![/bold green]")
        self.state_mgr.print_summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RT-DETR Automated Training & Deployment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run pipeline end-to-end (or resume)")
    p_run.add_argument("--from-stage", choices=[s.value for s in STAGE_ORDER], help="Start from specific stage")
    p_run.add_argument("--force", action="store_true", help="Re-run already completed stages")

    # Command: extract
    p_ext = subparsers.add_parser("extract", help="Stage 1: Extract frames from video")
    p_ext.add_argument("--video", help="Path to video file or directory")

    # Command: annotate
    p_ann = subparsers.add_parser("annotate", help="Stage 2: Ingest or auto-label annotations")
    p_ann.add_argument("--auto", action="store_true", help="Enable auto-labeling with pretrained model")

    # Command: prepare
    subparsers.add_parser("prepare", help="Stage 3: Prepare dataset & negative frames")

    # Command: train
    p_trn = subparsers.add_parser("train", help="Stage 4: Train RT-DETR model")
    p_trn.add_argument("--epochs", type=int, help="Override epochs count")

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Stage 5: Calibrate false-positive thresholds")
    p_eval.add_argument("--model", help="Path to trained checkpoint (.pt)")

    # Command: export
    p_exp = subparsers.add_parser("export", help="Stage 6: Export model (ONNX, OpenVINO, TorchScript)")
    p_exp.add_argument("--checkpoint", help="Path to checkpoint (.pt)")

    # Command: status
    subparsers.add_parser("status", help="Show pipeline state summary table")

    # Command: reset
    p_rst = subparsers.add_parser("reset", help="Reset a stage and downstream stages")
    p_rst.add_argument("--stage", required=True, choices=[s.value for s in STAGE_ORDER])

    # Command: infer
    p_inf = subparsers.add_parser("infer", help="Run false-positive-free inference on image/video")
    p_inf.add_argument("--model", required=True, help="Path to .pt or .onnx model")
    p_inf.add_argument("--input", required=True, help="Path to image, video, or folder")
    p_inf.add_argument("--output", default="workspace/inference_output", help="Output directory")
    p_inf.add_argument("--calibration", help="Path to calibrated_thresholds.json")

    # Command: web
    p_web = subparsers.add_parser("web", help="Launch interactive Web Studio dashboard")
    p_web.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    orchestrator = PipelineOrchestrator(args.config)

    if args.command == "status":
        orchestrator.state_mgr.print_summary()
    elif args.command == "reset":
        st = Stage(args.stage)
        orchestrator.state_mgr.reset_stage(st, reset_downstream=True)
        console.print(f"[yellow]Reset stage '{args.stage}' and all downstream stages.[/yellow]")
        orchestrator.state_mgr.print_summary()
    elif args.command == "extract":
        orchestrator.run_stage_extract(args.video)
    elif args.command == "annotate":
        orchestrator.run_stage_annotate(auto_label=args.auto)
    elif args.command == "prepare":
        orchestrator.run_stage_prepare()
    elif args.command == "train":
        orchestrator.run_stage_train(epochs=args.epochs)
    elif args.command == "evaluate":
        orchestrator.run_stage_evaluate(model_path=args.model)
    elif args.command == "export":
        orchestrator.run_stage_export(checkpoint_path=args.checkpoint)
    elif args.command == "run":
        orchestrator.run_all(from_stage=args.from_stage, force=args.force)
    elif args.command == "infer":
        run_inference_on_file(
            model_path=args.model,
            input_path=args.input,
            output_dir=args.output,
            calibration_path=args.calibration,
        )
    elif args.command == "web":
        from web.app import start_server
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
