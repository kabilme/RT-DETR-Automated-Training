"""
FastAPI Backend Server for RT-DETR Interactive Web Dashboard.
Provides REST APIs for stage execution, state monitoring, config updating,
dataset gallery, live inference, and artifact downloads.
"""

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.cli import load_config
from pipeline.infer import FalsePositiveFreeDetector
from pipeline.stage_1_extractor import FrameExtractor
from pipeline.stage_2_annotator import AnnotationManager
from pipeline.stage_3_dataset import DatasetBuilder
from pipeline.stage_4_trainer import RTDETRTrainer
from pipeline.stage_5_evaluator import FalsePositiveEvaluator
from pipeline.stage_6_exporter import ModelExporter
from pipeline.state import Stage, StageStatus, StateManager, STAGE_ORDER

app = FastAPI(title="RT-DETR Interactive Pipeline Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE = Path("workspace").resolve()
STATE_MGR = StateManager()
LOG_BUFFER: List[str] = []
MAX_LOGS = 250
PIPELINE_RUNNING = False
CURRENT_RUNNING_STAGE: Optional[str] = None


def append_log(msg: str):
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    LOG_BUFFER.append(formatted)
    if len(LOG_BUFFER) > MAX_LOGS:
        LOG_BUFFER.pop(0)


# Initialize static mounts
STATIC_DIR = Path("web/static").resolve()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path("workspace/web_uploads").resolve()
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/state")
def get_pipeline_state():
    """Returns the current state and metrics for all stages."""
    STATE_MGR.data = STATE_MGR._load()
    return {
        "stages": STATE_MGR.data.get("stages", {}),
        "last_completed_stage": STATE_MGR.data.get("last_completed_stage"),
        "is_running": PIPELINE_RUNNING,
        "running_stage": CURRENT_RUNNING_STAGE,
        "updated_at": STATE_MGR.data.get("updated_at"),
    }


@app.get("/api/logs")
def get_pipeline_logs():
    """Returns the latest execution logs."""
    return {"logs": LOG_BUFFER}


@app.get("/api/config")
def get_config():
    """Returns current config.yaml contents."""
    if Path("config.yaml").exists():
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# Video Storage Folders
VIDEOS_POS_DIR = Path("data/videos").resolve()
VIDEOS_NEG_DIR = Path("data/negative_videos").resolve()
VIDEOS_POS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_NEG_DIR.mkdir(parents=True, exist_ok=True)


def get_video_metadata(vpath: Path) -> Dict[str, Any]:
    """Inspects video file to get duration, fps, frame count, and dimensions."""
    meta = {
        "filename": vpath.name,
        "size_mb": round(vpath.stat().st_size / (1024 * 1024), 2),
        "fps": 0.0,
        "frames": 0,
        "duration_sec": 0.0,
        "resolution": "Unknown",
    }
    try:
        cap = cv2.VideoCapture(str(vpath))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            cap.release()

            meta["fps"] = round(fps, 1)
            meta["frames"] = frames
            meta["duration_sec"] = round(frames / max(fps, 1.0), 1)
            meta["resolution"] = f"{w}x{h}" if w > 0 else "Unknown"
    except Exception:
        pass
    return meta


@app.get("/api/videos")
def list_uploaded_videos():
    """Lists all uploaded training and negative background videos."""
    supported = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
    
    pos_videos = []
    if VIDEOS_POS_DIR.exists():
        for p in sorted(VIDEOS_POS_DIR.iterdir()):
            if p.suffix.lower() in supported and p.is_file():
                info = get_video_metadata(p)
                info["type"] = "positive"
                pos_videos.append(info)

    neg_videos = []
    if VIDEOS_NEG_DIR.exists():
        for p in sorted(VIDEOS_NEG_DIR.iterdir()):
            if p.suffix.lower() in supported and p.is_file():
                info = get_video_metadata(p)
                info["type"] = "negative"
                neg_videos.append(info)

    return {
        "positive_videos": pos_videos,
        "negative_videos": neg_videos,
        "total_count": len(pos_videos) + len(neg_videos),
    }


@app.post("/api/upload/video")
async def upload_video_file(
    file: UploadFile = File(...),
    video_type: str = Form("positive"),
):
    """Uploads a video file into data/videos (positive) or data/negative_videos (negative background)."""
    supported = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported video format '{suffix}'. Supported: {', '.join(supported)}")

    dest_dir = VIDEOS_NEG_DIR if video_type == "negative" else VIDEOS_POS_DIR
    dest_path = dest_dir / file.filename

    # Save video chunk by chunk to support large video files
    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    meta = get_video_metadata(dest_path)
    meta["type"] = video_type
    tag = "Background / Negative" if video_type == "negative" else "Training"
    append_log(f"Uploaded {tag} video: '{file.filename}' ({meta['size_mb']} MB, {meta['duration_sec']}s, {meta['resolution']})")

    return {
        "status": "success",
        "message": f"Successfully uploaded {tag} video",
        "video": meta,
    }


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    """Saves updated config parameters to config.yaml."""
    try:
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(req.config, f, sort_keys=False)
        append_log("Configuration updated successfully.")
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/videos/{video_type}/{filename}")
def delete_video_file(video_type: str, filename: str):
    """Deletes an uploaded video file."""
    import gc
    import urllib.parse

    unquoted_name = urllib.parse.unquote(filename)
    dest_dir = VIDEOS_NEG_DIR if video_type == "negative" else VIDEOS_POS_DIR
    target = dest_dir / unquoted_name

    if not target.exists() or not target.is_file():
        # Fallback check direct name
        target = dest_dir / filename

    if target.exists() and target.is_file():
        try:
            gc.collect()  # Release any OpenCV or Python file handles on Windows
            target.unlink()
            append_log(f"Deleted video file: '{target.name}' from {video_type} pool.")
            return {"status": "success", "message": f"Deleted {target.name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Permission error deleting file: {str(e)}")
    raise HTTPException(status_code=404, detail=f"Video file '{unquoted_name}' not found")


def _run_stage_task(stage_name: str, **kwargs):
    global PIPELINE_RUNNING, CURRENT_RUNNING_STAGE
    PIPELINE_RUNNING = True
    CURRENT_RUNNING_STAGE = stage_name
    append_log(f">>> Commencing Stage: {stage_name.upper()} <<<")

    try:
        cfg = load_config("config.yaml")
        state_mgr = StateManager()

        if stage_name == "extract":
            video_path = kwargs.get("video_path")
            extractor = FrameExtractor(cfg, state_mgr)
            stats = extractor.run(video_path)
            append_log(f"Extracted {stats['total_positive_frames']} positive, {stats['total_negative_frames']} background frames.")

        elif stage_name == "annotate":
            auto_label = kwargs.get("auto_label", True)
            if auto_label:
                cfg.setdefault("annotation", {}).setdefault("auto_label", {})["enabled"] = True
            classes_filter = kwargs.get("classes_filter", [3, 1, 2])
            if classes_filter:
                cfg.setdefault("annotation", {}).setdefault("auto_label", {})["classes"] = classes_filter
            conf_thresh = kwargs.get("conf_threshold", 0.40)
            if conf_thresh:
                cfg.setdefault("annotation", {}).setdefault("auto_label", {})["conf_threshold"] = float(conf_thresh)
            annotator = AnnotationManager(cfg, state_mgr)
            stats = annotator.run()
            append_log(f"Annotation completed: {stats.get('total_labels', 0)} frames ready with bounding boxes.")

        elif stage_name == "prepare":
            builder = DatasetBuilder(cfg, state_mgr)
            stats = builder.build()
            append_log(f"Dataset prepared. Train: {stats['train_pos']}+{stats['train_neg']}bg, Val: {stats['val_pos']}+{stats['val_neg']}bg.")

        elif stage_name == "train":
            epochs = kwargs.get("epochs")
            trainer = RTDETRTrainer(cfg, state_mgr)
            stats = trainer.train(override_epochs=epochs)
            append_log(f"Training completed. Best checkpoint: {stats.get('best_checkpoint')}")

        elif stage_name == "evaluate":
            model_path = kwargs.get("model_path")
            evaluator = FalsePositiveEvaluator(cfg, state_mgr)
            stats = evaluator.calibrate(model_path)
            append_log(f"Calibration completed. Global zero-FP threshold: {stats.get('global_calibrated_threshold')}")

        elif stage_name == "export":
            ckpt = kwargs.get("checkpoint_path")
            exporter = ModelExporter(cfg, state_mgr)
            stats = exporter.export(ckpt)
            append_log(f"Export completed. Ready formats: {list(stats.keys())}")

    except Exception as e:
        append_log(f"[ERROR in {stage_name.upper()}]: {str(e)}")
    finally:
        PIPELINE_RUNNING = False
        CURRENT_RUNNING_STAGE = None


@app.post("/api/stage/run/{stage_name}")
def run_stage(stage_name: str, payload: Dict[str, Any] = None):
    """Launches an individual pipeline stage in a background worker thread."""
    global PIPELINE_RUNNING
    if PIPELINE_RUNNING:
        raise HTTPException(status_code=400, detail="Another stage is currently running.")

    if stage_name not in [s.value for s in STAGE_ORDER]:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage_name}")

    kwargs = payload or {}
    t = threading.Thread(target=_run_stage_task, args=(stage_name,), kwargs=kwargs, daemon=True)
    t.start()
    return {"status": "started", "stage": stage_name}


@app.post("/api/stage/reset/{stage_name}")
def reset_stage(stage_name: str):
    """Resets a stage and downstream stages in the state manager."""
    try:
        st = Stage(stage_name)
        STATE_MGR.reset_stage(st, reset_downstream=True)
        if stage_name in ["train", "prepare", "annotate", "extract"]:
            # Clean old runs so fresh training doesn't resume old weights
            runs_train = Path("workspace/runs/train")
            if runs_train.exists():
                shutil.rmtree(runs_train, ignore_errors=True)
        append_log(f"Reset stage '{stage_name}' and dependent downstream stages.")
        return {"status": "success", "reset_stage": stage_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gallery/inspection")
def get_inspection_samples():
    """Lists inspection images with bounding box overlays."""
    ins_dir = Path("workspace/inspection")
    if not ins_dir.exists():
        return {"images": []}
    files = sorted(list(ins_dir.glob("*.jpg")) + list(ins_dir.glob("*.png")))
    return {"images": [f"/files/inspection/{f.name}" for f in files[:20]]}


@app.get("/api/gallery/frames")
def get_extracted_frames():
    """Lists sample extracted frames and negatives."""
    f_dir = Path("workspace/frames")
    n_dir = Path("workspace/negatives")

    pos_files = sorted(list(f_dir.glob("*.jpg")) + list(f_dir.glob("*.png"))) if f_dir.exists() else []
    neg_files = sorted(list(n_dir.glob("*.jpg")) + list(n_dir.glob("*.png"))) if n_dir.exists() else []

    return {
        "positive_count": len(pos_files),
        "negative_count": len(neg_files),
        "positive_samples": [f"/files/frames/{f.name}" for f in pos_files[:12]],
        "negative_samples": [f"/files/negatives/{f.name}" for f in neg_files[:12]],
    }


@app.get("/api/calibration")
def get_calibration_details():
    """Returns calibrated thresholds and calibration curve image URL."""
    c_path = Path("workspace/evaluation/calibrated_thresholds.json")
    plot_path = Path("workspace/evaluation/precision_calibration_curve.png")

    calib_json = {}
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                calib_json = json.load(f)
        except Exception:
            pass

    return {
        "calibrated_thresholds": calib_json,
        "curve_image": "/files/evaluation/precision_calibration_curve.png" if plot_path.exists() else None,
    }


@app.get("/api/artifacts")
def get_artifacts_list():
    """Lists exported model artifacts available for download."""
    exp_dir = Path("workspace/exported_models")
    artifacts = []
    if exp_dir.exists():
        for p in exp_dir.iterdir():
            if p.is_file():
                artifacts.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "url": f"/files/exported_models/{p.name}",
                })
            elif p.is_dir():
                artifacts.append({
                    "name": f"{p.name} (Folder)",
                    "size_mb": 0.0,
                    "url": f"/files/exported_models/{p.name}",
                })
    return {"artifacts": artifacts}


@app.post("/api/infer")
async def run_live_inference(
    file: UploadFile = File(...),
    model_choice: str = Form("onnx"),
    conf_override: Optional[float] = Form(None),
):
    """Executes live false-positive-free inference on an uploaded image or video."""
    try:
        contents = await file.read()
        filename_lower = file.filename.lower()
        is_video = any(filename_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])

        img = None
        if is_video:
            # Save temporary video to extract test frame
            temp_video_path = UPLOADS_DIR / f"temp_{file.filename}"
            with open(temp_video_path, "wb") as f:
                f.write(contents)

            cap = cv2.VideoCapture(str(temp_video_path))
            total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            # Sample frame at ~15% into video for representative content
            target_frame_no = min(15, total_v_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()
            try:
                temp_video_path.unlink()
            except Exception:
                pass

            if not ret or frame is None:
                raise HTTPException(status_code=400, detail="Could not extract a valid frame from the uploaded video.")
            img = frame
        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file format.")

        # Determine model path
        if model_choice == "onnx":
            candidates = [
                Path("workspace/exported_models/best.onnx"),
                Path("workspace/runs/train/rtdetr_run/weights/best.onnx"),
                Path("workspace/exported_models/rtdetr-l.onnx"),
                Path("rtdetr-l.onnx"),
            ]
        else:
            candidates = [
                Path("workspace/runs/train/rtdetr_run/weights/best.pt"),
                Path("workspace/exported_models/best.pt"),
                Path("rtdetr-l.pt"),
            ]

        model_path = None
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

        if not model_path:
            raise HTTPException(status_code=404, detail=f"No {model_choice.upper()} model found in workspace. Run Train & Export first.")

        detector = FalsePositiveFreeDetector(model_path)
        if conf_override is not None and conf_override > 0:
            detector.global_threshold = float(conf_override)
            for cname in detector.class_thresholds:
                detector.class_thresholds[cname]["calibrated_conf"] = float(conf_override)

        dets = detector.predict_image(img)
        vis = detector.annotate_image(img, dets)

        # Save result for viewing
        out_name = f"infer_{Path(file.filename).stem}.jpg"
        out_path = UPLOADS_DIR / out_name
        cv2.imwrite(str(out_path), vis)

        effective_conf = conf_override if conf_override is not None else detector.global_threshold

        return {
            "status": "success",
            "model_used": Path(model_path).name,
            "detections_count": len(dets),
            "detections": dets,
            "calibrated_threshold": effective_conf,
            "annotated_image_url": f"/files/web_uploads/{out_name}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static file serving routes
@app.get("/files/inspection/{filename}")
def serve_inspection(filename: str):
    p = Path("workspace/inspection") / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404)


@app.get("/files/frames/{filename}")
def serve_frames(filename: str):
    p = Path("workspace/frames") / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404)


@app.get("/files/negatives/{filename}")
def serve_negatives(filename: str):
    p = Path("workspace/negatives") / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404)


@app.get("/files/evaluation/{filename}")
def serve_evaluation(filename: str):
    p = Path("workspace/evaluation") / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404)


@app.get("/files/exported_models/{filename}")
def serve_exported_models(filename: str):
    p = Path("workspace/exported_models") / filename
    if p.exists() and p.is_file():
        return FileResponse(p, filename=filename)
    raise HTTPException(status_code=404)


@app.get("/files/web_uploads/{filename}")
def serve_uploads(filename: str):
    p = UPLOADS_DIR / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(status_code=404)


# Mount static assets (HTML/CSS/JS)
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")


def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"\n=======================================================")
    print(f" RT-DETR Interactive Web Studio running at:")
    print(f" >>> http://localhost:{port} <<<")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
