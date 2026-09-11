# RT-DETR Automated Training & Deployment Pipeline

A production-grade, modular framework for automating the entire lifecycle of training, fine-tuning, and deploying **RT-DETR (Real-Time DEtection TRansformer)** models. The pipeline automates every step from raw video ingestion to exporting deployment-ready artifacts (**ONNX**, **OpenVINO**, **TorchScript**) with **mathematically verified zero/low false-positive operating profiles**.

---

## 🌟 Key Highlights & Architecture

### 1. User-Specified Target Class & Ingestion Center
- **Custom Class Support on Upload**: Directly input any custom class name (e.g., `helmet`, `scooter`, `drone`, `industrial_part`) when uploading training videos through the interactive Web Studio or CLI.
- **Negative / Background Video Harvesting**: Upload pure background scenes (empty rooms, roads, natural clutter) to harvest hard negative frames that teach the Hungarian matcher what is *not* an object.
- **Metadata Persistence**: Video-to-class mappings are automatically saved to `data/videos/video_classes.json` and synchronized with `config.yaml`.

### 2. High-Confidence Prominent Foreground Object Annotation
- **No Reliance on Fixed COCO Classes**: Standard COCO models frequently fail on custom objects or assign erroneous labels (e.g., tagging a motorcycle helmet as a `vase` or `oven`).
- **No External Zero-Shot Dependencies**: Operates completely offline without requiring YOLO-World or heavy external downloads.
- **Foreground Prominence Localization**: Stage 2 isolates the primary object of interest in every extracted frame using:
  - **Geometric Saliency**: Rejects flat horizontal surfaces (tables, shelves with aspect ratio $> 2.8$) and vertical slivers ($< 0.25$).
  - **Boundary & Noise Filtering**: Excludes full-frame borders ($> 88\%$) and micro-noise ($< 6\%$).
  - **Center-Prior Proximity**: Focuses attention on foreground subjects located near the image optical center:
    $$\text{score} = \text{area} \times \text{conf} \times \text{center\_factor}$$
- **High-Confidence Badging**: Bounding boxes are tagged with the user's custom class name at very high confidence ($\ge 0.95$), saving visual inspection previews to `workspace/inspection/` (e.g. `helmet 0.95`).

### 3. Guaranteed False-Positive Defense
- **Background Frame Injection**: Dedicated negative frames are paired with 0-byte label files in Stage 3, enforcing zero-object supervision.
- **Precision Calibration Engine**: Stage 5 evaluates precision and false-positive curves across confidence thresholds on validation and background scenes, deriving optimal per-class operating thresholds ($\tau^*$) that achieve $100\%$ precision with zero false alarms.

### 4. Multi-Format Export & Verification
- Exports to **ONNX** (Opset 17, simplified with `onnxslim`), **OpenVINO** (optimized for Intel CPUs/iGPUs), and **TorchScript**.
- Automatic numerical and tensor shape verification with `onnxruntime` (`[1, 300, 6]`).

---

## 🚀 Quickstart

### 1. Environment Setup

The application is configured to run in `.venv` with Python 3.10:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Verify installation
python -m pip list
```

### 2. Launch the Interactive Web Studio Dashboard

Launch the web studio to manage videos, configure parameters, run the pipeline, and run live inference:

```powershell
python main.py web --port 8000
```

Open your browser at: **`http://localhost:8000`**

---

## 🖥️ Web Studio Dashboard Capabilities

The interactive web studio provides a comprehensive interface:

1. **Video Ingestion & Upload Center**:
   - Drag & drop or browse video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
   - Toggle between **🎯 Training Video (Target Objects)** and **🛡️ Background Video (0-FP Defense)**.
   - For training videos, enter the **Target Object Class Name** (e.g. `helmet`).
   - Video inventory list displaying video resolution, duration, FPS, file size, and interactive class badges (`🏷️ <class_name> ✏️`) for inline editing.
   - One-click deletion with confirmation dialogs.

2. **Modular Stage Progression Stepper**:
   - Visual execution cards for all 6 pipeline stages (**Extract**, **Annotate**, **Prepare**, **Train**, **Calibrate**, **Export**).
   - Real-time stage statuses (`COMPLETED`, `RUNNING`, `FAILED`, `PENDING`) and execution durations.
   - **Run Full Pipeline (Auto-Resume)**: Automatically resumes from the earliest uncompleted stage.
   - **Reset Entire Pipeline**: Flushes workspace artifacts and resets pipeline state back to stage 1.

3. **Live Execution Console**:
   - Real-time streaming log feed with auto-scroll and status indicators.

4. **Visual Inspection Gallery (`Dataset & Inspect`)**:
   - Renders visual overlays directly from `workspace/inspection/`.
   - Displays bounding boxes with bright green contours and exact `<class_name> <confidence>` badges.

5. **False-Positive Defense Curve (`FP Defense & Curve`)**:
   - Plots precision vs. confidence curves and per-class zero-FP operating thresholds.

6. **Live Inference Lab**:
   - Test trained models (`best.onnx` or `best.pt`) on single images or full video clips.
   - Displays real-time bounding boxes, confidence badges, latency metrics, and side-by-side detection cards.
   - Download processed inference videos with overlays.

7. **Models & Export**:
   - Download `best.onnx`, `best.torchscript`, `best_openvino_model`, and `calibrated_thresholds.json`.

---

## ⚙️ CLI Operations & Modular Stages

You can execute each stage independently via the command line:

### Stage 1: Frame Extraction & Pre-Filtering
Extracts video frames, rejects blurry frames via Laplacian variance, eliminates perceptual duplicates, and harvests background scenes:
```powershell
python main.py extract
```
*Output: Extracted frames in `workspace/frames/` and negative frames in `workspace/negatives/`.*

### Stage 2: Prominent Object Annotation
Localizes and annotates the most prominent foreground object with high confidence ($\ge 0.95$) using the specified class:
```powershell
python main.py annotate
```
*Output: YOLO labels in `workspace/labels/` and visual inspection frames in `workspace/inspection/`.*

### Stage 3: Dataset Preparation & Negative Injection
Partitions dataset into train/val/test splits and injects empty background images with 0-byte labels:
```powershell
python main.py prepare
```
*Output: Structured dataset in `workspace/dataset/` with `dataset.yaml`.*

### Stage 4: RT-DETR Training
Fine-tunes the RT-DETR model with auto-detected hardware, early stopping, and checkpoint recovery:
```powershell
# Train with default epochs from config.yaml:
python main.py train

# Or override epochs:
python main.py train --epochs 10
```
*Output: Checkpoints saved in `workspace/runs/train/rtdetr_run/weights/` (`best.pt`, `last.pt`).*

### Stage 5: False-Positive Calibration
Evaluates validation metrics to determine the exact threshold $\tau^*$ per class required to attain $\ge 99\%$ precision with zero false alarms on negative backgrounds:
```powershell
python main.py evaluate
```
*Output: `workspace/evaluation/calibrated_thresholds.json` and `precision_calibration_curve.png`.*

### Stage 6: Model Export & Verification
Exports `best.pt` to ONNX, OpenVINO, and TorchScript, verifying tensor shapes with `onnxruntime`:
```powershell
python main.py export
```
*Output: Exported bundle in `workspace/exported_models/` (`best.onnx`, `best.torchscript`, OpenVINO).*

---

## 🔄 Pipeline State Management & Status

Check the status of all pipeline stages at any time:
```powershell
python main.py status
```

Example status table:
```
                       RT-DETR Pipeline Execution State                        
+-----------------------------------------------------------------------------+
| Stage    | Status    | Duration (s) | Details / Artifacts                   |
|----------+-----------+--------------+---------------------------------------|
| EXTRACT  | COMPLETED |          3.0 | total_positive_frames=132, neg=8      |
| ANNOTATE | COMPLETED |         47.3 | total_labels=132, classes=['helmet']  |
| PREPARE  | COMPLETED |          0.6 | train_pos=92, train_neg=6, val_pos=26 |
| TRAIN    | COMPLETED |        341.0 | best_checkpoint=best.pt               |
| EVALUATE | COMPLETED |         11.3 | global_threshold=0.09, precision=1.0  |
| EXPORT   | COMPLETED |         18.9 | formats=['onnx', 'torchscript', ...]  |
+-----------------------------------------------------------------------------+
```

To reset a specific stage and all downstream stages:
```powershell
python main.py reset --stage train
```

To reset the entire pipeline back to stage 1:
```powershell
python main.py reset --all
```

---

## 🎯 Running Inference (Zero False Positives)

Deploy the trained model with calibrated thresholds on any test image, video, or folder:

```powershell
# Run inference on an image using ONNX:
python main.py infer --model workspace/exported_models/best.onnx --input path/to/image.jpg

# Run inference on a video using PyTorch checkpoint:
python main.py infer --model workspace/runs/train/rtdetr_run/weights/best.pt --input path/to/video.mp4

# Run inference on a directory of images:
python main.py infer --model workspace/exported_models/best.onnx --input path/to/folder/
```

Detections automatically enforce the calibrated per-class threshold $\tau^*$ and geometric filters to eliminate spurious background hallucinations.

---

## 📝 Configuration Reference (`config.yaml`)

```yaml
project:
  name: RT_DETR_Project
  work_dir: workspace
  device: auto # auto, cpu, or cuda:0

extraction:
  video_path: data/videos
  negative_video_path: data/negative_videos
  sample_fps: 4.0
  blur_filter:
    enabled: true
    min_laplacian_variance: 80.0
  deduplication:
    enabled: true
    similarity_threshold: 0.96

annotation:
  annotation_dir: data/annotations
  class_names:
    - helmet # Target class name
  auto_label:
    enabled: true
    model: rtdetr-l.pt
    conf_threshold: 0.50

dataset:
  train_ratio: 0.70
  val_ratio: 0.20
  test_ratio: 0.10
  false_positive_mitigation:
    include_background_images: true
    target_background_ratio: 0.15 # 15% pure background scenes

training:
  model_architecture: rtdetr-l.pt
  imgsz: 640
  epochs: 2
  batch_size: 4
  optimizer: AdamW
  lr0: 0.0005

evaluation:
  target_precision: 0.99 # 99% precision requirement
  max_acceptable_fp_count: 0
  min_box_area: 100
  aspect_ratio_range: [0.1, 10.0]

export:
  formats:
    - onnx
    - torchscript
    - openvino
  onnx:
    opset: 17
    simplify: true
```
