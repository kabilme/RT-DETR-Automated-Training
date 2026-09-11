"""
Stage 6: RT-DETR Model Exporter & ONNX Runtime Verifier.
Exports trained models into deployment formats (ONNX, OpenVINO, TorchScript)
and verifies runtime execution with ONNX Runtime.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console

from pipeline.state import Stage, StateManager

console = Console()


class ModelExporter:
    """Exports trained RT-DETR models and verifies runtime deployment readiness."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()

        self.exp_cfg = config.get("export", {})
        self.output_dir = Path(self.exp_cfg.get("output_dir", "workspace/exported_models")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.formats = self.exp_cfg.get("formats", ["onnx"])
        self.onnx_cfg = self.exp_cfg.get("onnx", {})
        self.opset = int(self.onnx_cfg.get("opset", 17))
        self.dynamic = bool(self.onnx_cfg.get("dynamic", False))
        self.simplify = bool(self.onnx_cfg.get("simplify", True))
        self.half = bool(self.onnx_cfg.get("half", False))
        self.imgsz = int(config.get("training", {}).get("imgsz", 640))

    def _verify_onnx_model(self, onnx_file: Path) -> Dict[str, Any]:
        """Runs a test inference pass using ONNX Runtime to verify model integrity."""
        import onnx
        import onnxruntime as ort

        console.print(f"Verifying ONNX model integrity with onnxruntime: [bold]{onnx_file.name}[/bold]")

        # 1. Structural check
        model_proto = onnx.load(str(onnx_file))
        onnx.checker.check_model(model_proto)

        # 2. Runtime session check
        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()

        console.print(f" - Input name:  [cyan]{input_meta.name}[/cyan], shape: {input_meta.shape}, type: {input_meta.type}")
        for idx, out in enumerate(output_meta):
            console.print(f" - Output [{idx}]: [green]{out.name}[/green], shape: {out.shape}, type: {out.type}")

        # 3. Dummy inference pass
        batch_size = 1
        dummy_input = np.random.randn(batch_size, 3, self.imgsz, self.imgsz).astype(np.float32)
        outputs = session.run(None, {input_meta.name: dummy_input})

        console.print(f"[bold green]✓ ONNX verification passed![/bold green] Output shape: {outputs[0].shape}")
        return {
            "onnx_input_name": input_meta.name,
            "onnx_input_shape": input_meta.shape,
            "onnx_output_count": len(outputs),
            "onnx_output_shape": list(outputs[0].shape),
        }

    def export(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Exports the trained RT-DETR weights to requested formats."""
        from ultralytics import RTDETR

        self.state_mgr.start_stage(Stage.EXPORT)

        ckpt = checkpoint_path
        if not ckpt:
            train_art = self.state_mgr.get_artifacts(Stage.TRAIN)
            ckpt = train_art.get("best_checkpoint")

        if not ckpt or not Path(ckpt).exists():
            bests = list(Path("workspace/runs/train").rglob("best.pt"))
            if bests:
                ckpt = str(bests[0])
            elif Path("rtdetr-l.pt").exists():
                ckpt = "rtdetr-l.pt"
            else:
                err = "No trained RT-DETR checkpoint found to export."
                self.state_mgr.fail_stage(Stage.EXPORT, err)
                raise FileNotFoundError(err)

        console.print(f"[bold cyan]Exporting RT-DETR Model:[/bold cyan] {ckpt}")
        model = RTDETR(ckpt)

        exported_artifacts = {}
        verification_results = {}

        for fmt in self.formats:
            fmt_lower = fmt.lower()
            console.print(f"\n[bold]Exporting format: {fmt_lower.upper()}...[/bold]")
            try:
                export_kwargs = {
                    "format": fmt_lower,
                    "imgsz": self.imgsz,
                    "half": self.half,
                }
                if fmt_lower == "onnx":
                    export_kwargs.update({
                        "opset": self.opset,
                        "dynamic": self.dynamic,
                        "simplify": self.simplify,
                    })

                exported_file = model.export(**export_kwargs)
                exp_path = Path(exported_file)

                # Move/copy exported artifact to dedicated exported_models folder
                dest_path = self.output_dir / exp_path.name
                if exp_path.is_file():
                    shutil.copy2(exp_path, dest_path)
                elif exp_path.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(exp_path, dest_path)

                exported_artifacts[fmt_lower] = str(dest_path)
                console.print(f"[green]Saved {fmt_lower.upper()} model to: {dest_path}[/green]")

                # If ONNX, perform runtime verification
                if fmt_lower == "onnx" and dest_path.is_file():
                    v_res = self._verify_onnx_model(dest_path)
                    verification_results.update(v_res)

            except Exception as e:
                console.print(f"[yellow]Failed to export format {fmt}: {e}[/yellow]")

        # Copy calibrated thresholds file alongside exported model for ready-to-ship bundle
        calib_file = Path("workspace/evaluation/calibrated_thresholds.json")
        if calib_file.exists():
            bundle_calib = self.output_dir / "calibrated_thresholds.json"
            shutil.copy2(calib_file, bundle_calib)
            exported_artifacts["calibrated_thresholds"] = str(bundle_calib)

        metrics = {
            "exported_formats": list(exported_artifacts.keys()),
            "output_directory": str(self.output_dir),
            **verification_results,
        }

        self.state_mgr.complete_stage(
            Stage.EXPORT,
            artifacts=exported_artifacts,
            metrics=metrics,
        )

        console.print(f"\n[bold green]Stage 6 Complete! Model ready for deployment at:[/bold green] {self.output_dir}")
        return exported_artifacts
