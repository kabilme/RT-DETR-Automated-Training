# RT-DETR Automated Training & Deployment Pipeline

A production-ready, modular Python application for automating the entire lifecycle of training and deploying **RT-DETR (Real-Time DEtection TRansformer)** models. The pipeline automates every step from raw video frame ingestion to generating deployment-optimized artifacts (**ONNX**, **OpenVINO**, **TorchScript**) with **mathematically verified zero/low false-positive operating profiles**.

---

## Key Features

1. **Modular & Resumable Architecture**:
   - Every stage (`extract` → `annotate` → `prepare` → `train` → `evaluate` → `export`) is decoupled.
   - Built-in persistent state manager (`workspace/pipeline_state.json`) lets you pause, stop, and continue at any stage without losing progress.
   - Stage reset support (`python main.py reset --stage train`) allows re-running specific stages without re-extracting frames.

2. **Guaranteed False-Positive Mitigation**:
   - **Background / Negative Frame Harvesting**: Extracts empty frames (scenery, clutter, shadows) and pairs them with 0-byte label files. This teaches the RT-DETR Hungarian matcher what is *not* an object.
   - **Hard Negative Mining**: Easily ingest challenging false-alarm scenes from `data/hard_negatives/`.
   - **Precision Calibration Engine**: Evaluates Precision-Recall and False-Positive curves on validation and background scenes, deriving optimal per-class confidence thresholds ($\tau^*$) that meet your target precision (e.g. 99–100%).
   - **Geometric Outlier Filtering**: Automatically suppresses micro-noise boxes and impossible aspect ratio detections.

3. **Multi-Format Export & Verification**:
   - Exports directly to **ONNX** (with Opset 17), **OpenVINO** (optimized for Intel CPUs/iGPUs), and **TorchScript**.
   - Automatic post-export verification with `onnxruntime` to ensure numerical and shape integrity.

4. **Hardware Optimized**:
   - Auto-detects CUDA / CPU.
   - Supports OpenVINO acceleration on Intel UHD / Iris graphics.

---

## Quickstart

### 1. Environment Setup

The application is configured to run in `.venv` (Python 3.10):

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Verify packages
python -m pip list
```

### 2. Prepare Your Data

Place your raw input videos into `data/videos/`:
```
d:\RT_DETR\
  ├── data/
  │   ├── videos/              # Put your positive training videos here (.mp4, .avi, etc.)
  │   ├── negative_videos/     # Optional: videos containing only background/no objects
  │   └── annotations/         # Optional: manual labels (.txt or Pascal VOC .xml)
```

---

## Pipeline Execution

### Option A: Run Full Pipeline (End-to-End)

Executes all stages sequentially. If a stage is already completed, it will automatically resume from the next pending stage:

```powershell
python main.py run
```

To force re-running all stages from scratch:
```powershell
python main.py run --force
```

To start from a specific stage (e.g. from training onward):
```powershell
python main.py run --from-stage train
```

---

### Option B: Interactive Web Studio Dashboard

You can launch and operate the entire pipeline through the browser dashboard:

```powershell
python main.py web --port 8000
```
Open your browser at: **`http://localhost:8000`**

The Web Studio provides:
- **Modular Pipeline Stepper**: Live glowing step indicators, execution times, and stage trigger/reset buttons.
- **Execution Console**: Live streaming logs with autoscroll.
- **Configuration Tuner**: Adjust blur thresholds, negative ratios, epochs, and precision targets interactively.
- **Visual Inspection Gallery**: Preview extracted frames and ground truth bounding boxes before training.
- **Precision & False-Positive Curves**: Review calibration curves and optimal per-class thresholds ($\tau^*$).
- **Live Inference Lab**: Drag & drop test images to test false-positive rejection with ONNX Runtime.
- **Artifact Downloads**: One-click download for `rtdetr-l.onnx`, `rtdetr-l.torchscript`, OpenVINO, and `calibrated_thresholds.json`.

---

### Option C: Modular Stage-by-Stage CLI Control

You can execute each stage independently, inspect intermediate artifacts, and continue when ready:

#### Stage 1: Video Ingestion & Frame Extraction
Extracts frames from video files with Laplacian blur filtering, perceptual deduplication, and negative frame extraction:
```powershell
# Extract frames from default video folder:
python main.py extract

# Or extract from a specific video file:
python main.py extract --video path/to/video.mp4
```
*Output: Extracted frames in `workspace/frames/` and background scenes in `workspace/negatives/`.*

#### Stage 2: Annotation Ingestion / Auto-Labeling
Ingests manual YOLO or Pascal VOC annotations, or bootstraps labels using a pretrained model:
```powershell
# Ingest manual annotations:
python main.py annotate

# Or enable bootstrap auto-labeling:
python main.py annotate --auto
```
*Visual inspection overlays are saved to `workspace/inspection/` for quality review.*

#### Stage 3: Dataset Preparation & Negative Sample Injection
Splits data into `train`, `val`, and `test` splits, enforces negative/background image ratios, and generates 0-byte negative label files:
```powershell
python main.py prepare
```
*Output: RT-DETR compatible dataset in `workspace/dataset/` with `dataset.yaml`.*

#### Stage 4: RT-DETR Model Training
Trains the RT-DETR model with auto-detected hardware, early stopping, and checkpoint recovery:
```powershell
# Train with default epochs from config.yaml:
python main.py train

# Or override epoch count:
python main.py train --epochs 30
```
*If interrupted, re-running `python main.py train` automatically resumes from `last.pt`.*

#### Stage 5: False-Positive Calibration
Evaluates validation metrics across confidence thresholds to determine the exact threshold $\tau^*$ per class required to attain $\ge 99\%$ precision with zero false positives:
```powershell
python main.py evaluate
```
*Outputs `workspace/evaluation/calibrated_thresholds.json` and visual curve `precision_calibration_curve.png`.*

#### Stage 6: Model Export & Verification
Exports the best checkpoint to deployment-ready formats and verifies them with ONNX Runtime:
```powershell
python main.py export
```
*Output: Ready-to-deploy bundle in `workspace/exported_models/` (`model.onnx`, `calibrated_thresholds.json`).*

---

## Status & State Management

Inspect the current execution state of all pipeline stages at any time:
```powershell
python main.py status
```
Example output:
```
                       RT-DETR Pipeline Execution State                        
+-----------------------------------------------------------------------------+
| Stage    | Status    | Duration (s) | Details / Artifacts                   |
|----------+-----------+--------------+---------------------------------------|
| EXTRACT  | COMPLETED |         14.2 | total_positive_frames=120, neg=25     |
| ANNOTATE | COMPLETED |          3.1 | total_labels=120, inspected_samples=10|
| PREPARE  | COMPLETED |          1.4 | train_pos=84, train_neg=17, val_pos=24|
| TRAIN    | COMPLETED |        420.5 | best_checkpoint=runs/train/best.pt    |
| EVALUATE | COMPLETED |         12.8 | global_threshold=0.74, precision=1.0  |
| EXPORT   | COMPLETED |          8.2 | formats=['onnx', 'openvino']          |
+-----------------------------------------------------------------------------+
```

To reset a stage (and any downstream stages dependent on it):
```powershell
python main.py reset --stage train
```

---

## Running Inference (Zero False Positives)

Deploy the trained model with precision calibration on any new image, video, or folder:

```powershell
# Inference on an image:
python main.py infer --model workspace/exported_models/best.onnx --input path/to/image.jpg

# Inference on a video:
python main.py infer --model workspace/exported_models/best.pt --input path/to/video.mp4

# Inference on an entire folder:
python main.py infer --model workspace/exported_models/best.onnx --input path/to/test_folder/
```

Detections will automatically apply the calibrated per-class threshold $\tau^*$ and geometric noise filters, discarding low-confidence spurious background hallucinations.

---

## Configuration Reference (`config.yaml`)

Edit `config.yaml` to tune pipeline parameters:

- `extraction.sample_fps`: Extraction rate (frames per second).
- `extraction.blur_filter.min_laplacian_variance`: Minimum sharpness threshold (higher = stricter blur rejection).
- `extraction.deduplication.similarity_threshold`: Consecutive frame similarity limit (default: 0.96).
- `dataset.false_positive_mitigation.target_background_ratio`: Fraction of pure background images to include (default: 0.15 = 15%).
- `training.model_architecture`: Backbone (`rtdetr-l.pt`, `rtdetr-x.pt`).
- `training.imgsz`: Image resolution (e.g. 640 or 320 for faster CPU training).
- `evaluation.target_precision`: Precision target for threshold calibration (default: 0.99 = 99%).
- `export.formats`: List of export formats (`onnx`, `openvino`, `torchscript`).
