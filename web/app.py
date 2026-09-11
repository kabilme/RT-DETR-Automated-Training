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


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith(".js") or path.endswith(".html") or path.endswith(".css") or path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

WORKSPACE = Path("workspace").resolve()
STATE_MGR = StateManager()
LOG_BUFFER: List[str] = []
MAX_LOGS = 250
PIPELINE_RUNNING = False
CURRENT_RUNNING_STAGE: Optional[str] = None
PIPELINE_STOP_REQUESTED: bool = False


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


def _execute_stage_logic(stage_name: str, cfg: Dict[str, Any], state_mgr: StateManager, **kwargs) -> bool:
    """Executes an individual pipeline stage's logic and returns True on success, False on error."""
    try:
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

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        append_log(f"[ERROR in {stage_name.upper()}]: {str(e)}")
        try:
            state_mgr.fail_stage(Stage(stage_name), str(e))
        except Exception:
            pass
        return False


def _run_stage_task(stage_name: str, **kwargs):
    global PIPELINE_RUNNING, CURRENT_RUNNING_STAGE
    PIPELINE_RUNNING = True
    CURRENT_RUNNING_STAGE = stage_name
    append_log(f">>> Commencing Stage: {stage_name.upper()} <<<")

    try:
        cfg = load_config("config.yaml")
        state_mgr = StateManager()
        _execute_stage_logic(stage_name, cfg, state_mgr, **kwargs)
    finally:
        PIPELINE_RUNNING = False
        CURRENT_RUNNING_STAGE = None


def _run_full_pipeline_task(from_stage: Optional[str] = None, force: bool = False, **kwargs):
    global PIPELINE_RUNNING, CURRENT_RUNNING_STAGE, PIPELINE_STOP_REQUESTED
    PIPELINE_RUNNING = True
    PIPELINE_STOP_REQUESTED = False

    append_log("==================================================")
    append_log(">>> Starting End-to-End Automated Pipeline Run <<<")
    append_log("==================================================")

    try:
        cfg = load_config("config.yaml")
        state_mgr = StateManager()

        stage_order_names = [s.value for s in STAGE_ORDER]
        start_idx = stage_order_names.index(from_stage) if from_stage and from_stage in stage_order_names else 0

        for idx in range(start_idx, len(STAGE_ORDER)):
            if PIPELINE_STOP_REQUESTED:
                append_log("[PIPELINE STOPPED by user request]")
                break

            stage = STAGE_ORDER[idx]
            s_name = stage.value

            state_mgr.data = state_mgr._load()
            # If already completed and not force, skip to next stage
            if not force and state_mgr.is_completed(stage):
                append_log(f"Stage {idx+1}/{len(STAGE_ORDER)}: '{s_name.upper()}' is already completed. Advancing to next stage...")
                continue

            CURRENT_RUNNING_STAGE = s_name
            append_log(f"\n>>> [Stage {idx+1}/{len(STAGE_ORDER)}] Commencing Stage: {s_name.upper()} <<<")

            success = _execute_stage_logic(s_name, cfg, state_mgr, **kwargs)
            if not success:
                append_log(f"[PIPELINE HALTED]: Stage {s_name.upper()} failed. Resolve error before resuming.")
                break

            append_log(f"✓ Stage {s_name.upper()} completed successfully. Advancing to next stage...")

        if not PIPELINE_STOP_REQUESTED and state_mgr.is_completed(Stage.EXPORT):
            append_log("\n=======================================================")
            append_log("🎉 ALL 6 STAGES FINISHED! End-to-end pipeline complete.")
            append_log("=======================================================")

    except Exception as e:
        append_log(f"[CRITICAL PIPELINE ERROR]: {str(e)}")
    finally:
        PIPELINE_RUNNING = False
        CURRENT_RUNNING_STAGE = None
        PIPELINE_STOP_REQUESTED = False


@app.post("/api/stage/run/{stage_name}")
def run_stage(stage_name: str, payload: Dict[str, Any] = None):
    """Launches an individual pipeline stage in a background worker thread."""
    global PIPELINE_RUNNING
    if PIPELINE_RUNNING:
        raise HTTPException(status_code=400, detail="Another stage or pipeline run is currently in progress.")

    if stage_name not in [s.value for s in STAGE_ORDER]:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage_name}")

    kwargs = payload or {}
    t = threading.Thread(target=_run_stage_task, args=(stage_name,), kwargs=kwargs, daemon=True)
    t.start()
    return {"status": "started", "stage": stage_name}


@app.post("/api/pipeline/run")
def run_full_pipeline(payload: Dict[str, Any] = None):
    """Launches the full automated end-to-end pipeline execution across all stages."""
    global PIPELINE_RUNNING
    if PIPELINE_RUNNING:
        raise HTTPException(status_code=400, detail="Another stage or pipeline run is currently in progress.")

    kwargs = payload or {}
    from_stage = kwargs.get("from_stage")
    force = kwargs.get("force", False)

    t = threading.Thread(target=_run_full_pipeline_task, kwargs={"from_stage": from_stage, "force": force}, daemon=True)
    t.start()
    return {"status": "started", "mode": "full_pipeline"}


@app.post("/api/pipeline/stop")
def stop_pipeline():
    """Requests stopping the ongoing automated pipeline execution."""
    global PIPELINE_STOP_REQUESTED
    if not PIPELINE_RUNNING:
        return {"status": "idle", "message": "No pipeline is currently running."}
    PIPELINE_STOP_REQUESTED = True
    append_log("Pipeline stop requested by user. Will halt cleanly before starting the next stage.")
    return {"status": "stopping", "message": "Stop signal sent"}


@app.post("/api/stage/reset/{stage_name}")
def reset_stage(stage_name: str):
    """Resets a stage and downstream stages in the state manager and cleans disk data."""
    try:
        if stage_name.lower() == "all":
            STATE_MGR.reset_all(clean_disk=True)
            append_log("Reset entire pipeline: all generated workspace data cleared.")
            return {"status": "success", "reset_stage": "all"}

        st = Stage(stage_name)
        STATE_MGR.reset_stage(st, reset_downstream=True, clean_disk=True)
        append_log(f"Reset stage '{stage_name}' and cleared all associated and downstream data.")
        return {"status": "success", "reset_stage": stage_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pipeline/reset")
def reset_entire_pipeline():
    """Resets the entire pipeline, all stages to PENDING, and purges all workspace data."""
    global PIPELINE_RUNNING, CURRENT_RUNNING_STAGE
    PIPELINE_RUNNING = False
    CURRENT_RUNNING_STAGE = None
    STATE_MGR.reset_all(clean_disk=True)
    append_log("Reset entire pipeline: all generated workspace data cleared.")
    return {"status": "success", "message": "All pipeline data cleared successfully"}


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
def run_live_inference(
    file: Optional[UploadFile] = File(None),
    existing_video: Optional[str] = Form(None),
    model_choice: str = Form("onnx"),
    conf_override: Optional[float] = Form(None),
):
    """Executes live false-positive-free inference on an uploaded image/video or existing workspace video."""
    try:
        has_file = file is not None and bool(getattr(file, "filename", None))
        has_existing = bool(existing_video and existing_video.strip())

        if not has_file and not has_existing:
            raise HTTPException(
                status_code=400,
                detail="No media selected. Please choose an existing workspace video or upload an image/video file."
            )

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
            raise HTTPException(
                status_code=404,
                detail=f"No {model_choice.upper()} model found in workspace. Please run Train & Export first."
            )

        detector = FalsePositiveFreeDetector(model_path)
        if conf_override is not None and conf_override > 0:
            detector.global_threshold = float(conf_override)
            for cname in detector.class_thresholds:
                detector.class_thresholds[cname]["calibrated_conf"] = float(conf_override)

        effective_conf = conf_override if conf_override is not None else detector.global_threshold

        is_video = False
        video_source_path: Optional[Path] = None
        temp_video_path: Optional[Path] = None
        display_name = ""

        if has_existing:
            clean_name = existing_video.strip()
            search_dirs = [
                VIDEOS_POS_DIR,
                VIDEOS_NEG_DIR,
                Path("workspace/raw_videos").resolve(),
                UPLOADS_DIR,
                Path("data/videos").resolve(),
                Path("data/negative_videos").resolve(),
            ]
            for d in search_dirs:
                candidate_p = d / clean_name
                if candidate_p.exists() and candidate_p.is_file():
                    video_source_path = candidate_p
                    break

            if not video_source_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Existing video '{clean_name}' was not found on the server filesystem."
                )
            is_video = True
            display_name = video_source_path.name

        elif has_file:
            display_name = file.filename
            fn_lower = display_name.lower()
            is_video = any(fn_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"])

            if is_video:
                temp_video_path = UPLOADS_DIR / f"temp_{display_name}"
                with open(temp_video_path, "wb") as f_out:
                    shutil.copyfileobj(file.file, f_out)
                video_source_path = temp_video_path

        if is_video and video_source_path:
            cap = cv2.VideoCapture(str(video_source_path))
            if not cap.isOpened():
                if temp_video_path and temp_video_path.exists():
                    temp_video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Could not decode the video file.")

            total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            stem = Path(display_name).stem
            raw_video_path = UPLOADS_DIR / f"raw_infer_{stem}.mp4"
            final_video_path = UPLOADS_DIR / f"infer_{stem}.mp4"
            thumb_path = UPLOADS_DIR / f"infer_{stem}.jpg"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(raw_video_path), fourcc, fps, (w, h))

            frame_idx = 0
            total_dets = 0
            max_dets = 0
            class_counts = {}
            thumb_saved = False
            last_vis = None

            append_log(f"Starting calibrated detection on '{display_name}' ({total_v_frames} frames)...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                dets = detector.predict_image(frame)
                vis = detector.annotate_image(frame, dets)
                writer.write(vis)
                last_vis = vis

                c = len(dets)
                total_dets += c
                if c > max_dets:
                    max_dets = c

                for d in dets:
                    cn = d["class_name"]
                    class_counts[cn] = class_counts.get(cn, 0) + 1

                # Save thumbnail frame with detection if available, else first frame
                if not thumb_saved and c > 0:
                    cv2.imwrite(str(thumb_path), vis)
                    thumb_saved = True
                elif not thumb_saved and frame_idx == 0:
                    cv2.imwrite(str(thumb_path), vis)

                frame_idx += 1
                if frame_idx % 40 == 0 or frame_idx == total_v_frames:
                    pct = int((frame_idx / total_v_frames) * 100)
                    append_log(f"Calibrating '{display_name}': frame {frame_idx}/{total_v_frames} ({pct}%)...")

            cap.release()
            writer.release()

            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except Exception:
                    pass

            if not thumb_path.exists() and last_vis is not None:
                cv2.imwrite(str(thumb_path), last_vis)

            # Re-encode to universal browser-compatible H.264 using ffmpeg if available
            converted = False
            try:
                import subprocess
                res_ff = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(raw_video_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(final_video_path)
                    ],
                    capture_output=True
                )
                if res_ff.returncode == 0 and final_video_path.exists() and final_video_path.stat().st_size > 0:
                    raw_video_path.unlink(missing_ok=True)
                    converted = True
            except Exception:
                pass

            if not converted:
                if final_video_path.exists():
                    final_video_path.unlink(missing_ok=True)
                raw_video_path.rename(final_video_path)

            append_log(f"Completed calibrated detection on '{display_name}': {frame_idx} frames, {total_dets} detections found (Zero False Positives).")

            return {
                "status": "success",
                "is_video": True,
                "model_used": Path(model_path).name,
                "total_frames": frame_idx,
                "fps": round(fps, 2),
                "duration_sec": round(frame_idx / max(fps, 1.0), 2),
                "detections_count": total_dets,
                "max_detections_per_frame": max_dets,
                "class_counts": class_counts,
                "calibrated_threshold": effective_conf,
                "annotated_video_url": f"/files/web_uploads/{final_video_path.name}",
                "annotated_image_url": f"/files/web_uploads/{thumb_path.name}",
            }

        else:
            # Process single image file
            contents = file.file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file format.")

            dets = detector.predict_image(img)
            vis = detector.annotate_image(img, dets)

            out_name = f"infer_{Path(display_name).stem}.jpg"
            out_path = UPLOADS_DIR / out_name
            cv2.imwrite(str(out_path), vis)

            append_log(f"Inference on '{display_name}': {len(dets)} objects detected using {Path(model_path).name} (τ*={effective_conf}).")

            return {
                "status": "success",
                "is_video": False,
                "model_used": Path(model_path).name,
                "detections_count": len(dets),
                "detections": dets,
                "calibrated_threshold": effective_conf,
                "annotated_image_url": f"/files/web_uploads/{out_name}",
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    if p.exists() and p.is_file():
        media_type = "video/mp4" if p.suffix.lower() == ".mp4" else None
        return FileResponse(p, media_type=media_type)
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
