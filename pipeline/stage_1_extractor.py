"""
Stage 1: Video Frame Acquisition & Quality Filtering.
Extracts frames from video files with intelligent blur filtering, perceptual deduplication,
and dedicated background/negative frame harvesting.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from pipeline.state import Stage, StateManager

console = Console()


def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    """Calculates blur metric using the variance of the Laplacian."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_frame_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Computes normalized similarity between two frames using downsampled grayscale correlation.
    Fast and effective for detecting near-identical consecutive video frames.
    """
    g1 = cv2.cvtColor(cv2.resize(frame1, (64, 64)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(frame2, (64, 64)), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(g1, g2, cv2.TM_CCORR_NORMED)
    return float(res[0][0])


class FrameExtractor:
    """Extracts, filters, and organizes training and negative frames from video files."""

    def __init__(self, config: Dict[str, Any], state_mgr: Optional[StateManager] = None):
        self.config = config
        self.state_mgr = state_mgr or StateManager()
        self.ext_cfg = config.get("extraction", {})

        # Paths
        self.output_dir = Path(self.ext_cfg.get("output_dir", "workspace/frames"))
        self.negative_dir = Path(self.ext_cfg.get("negative_output_dir", "workspace/negatives"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.negative_dir.mkdir(parents=True, exist_ok=True)

        # Sampling parameters
        self.sample_fps = float(self.ext_cfg.get("sample_fps", 2.0))
        self.frame_step = self.ext_cfg.get("frame_step", None)

        # Filters
        blur_cfg = self.ext_cfg.get("blur_filter", {})
        self.blur_enabled = blur_cfg.get("enabled", True)
        self.min_variance = float(blur_cfg.get("min_laplacian_variance", 80.0))

        dedup_cfg = self.ext_cfg.get("deduplication", {})
        self.dedup_enabled = dedup_cfg.get("enabled", True)
        self.sim_threshold = float(dedup_cfg.get("similarity_threshold", 0.96))

        # Negative harvesting
        neg_cfg = self.ext_cfg.get("negative_harvesting", {})
        self.neg_enabled = neg_cfg.get("enabled", True)
        self.neg_video_path = neg_cfg.get("negative_video_path", "data/negative_videos")

    def _find_video_files(self, video_input: str) -> List[Path]:
        """Locates all video files from a file path or directory."""
        vpath = Path(video_input)
        if not vpath.exists():
            raise FileNotFoundError(f"Video path not found: {video_input}")

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        if vpath.is_file():
            if vpath.suffix.lower() in video_exts:
                return [vpath]
            raise ValueError(f"File {vpath} does not have a supported video extension ({video_exts})")

        found = [p for p in vpath.rglob("*") if p.suffix.lower() in video_exts]
        if not found:
            raise FileNotFoundError(f"No video files found in directory: {video_input}")
        return sorted(found)

    def extract_from_video(
        self,
        video_path: Path,
        save_as_negatives: bool = False,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Extracts frames from a single video with blur and similarity filtering."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = self.frame_step if self.frame_step else max(1, int(round(fps / self.sample_fps)))

        target_dir = self.negative_dir if save_as_negatives else self.output_dir
        tag = "neg" if save_as_negatives else "pos"
        video_stem = video_path.stem

        saved_files = []
        discarded_blur = 0
        discarded_dedup = 0
        processed_count = 0
        last_saved_frame: Optional[np.ndarray] = None

        console.print(
            f"[cyan]Processing video:[/cyan] {video_path.name} "
            f"([dim]{total_frames} frames, {fps:.1f} FPS, sampling step: {step}[/dim])"
        )

        frame_idx = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Extracting {video_path.name}", total=total_frames)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    processed_count += 1

                    # Blur filtering (use more permissive threshold for negative background scenes)
                    if self.blur_enabled:
                        var = compute_laplacian_variance(frame)
                        effective_min_var = (self.min_variance * 0.15) if save_as_negatives else self.min_variance
                        if var < effective_min_var:
                            discarded_blur += 1
                            frame_idx += 1
                            progress.update(task, advance=1)
                            continue

                    # Deduplication filtering
                    if self.dedup_enabled and last_saved_frame is not None:
                        sim = compute_frame_similarity(last_saved_frame, frame)
                        if sim >= self.sim_threshold:
                            discarded_dedup += 1
                            frame_idx += 1
                            progress.update(task, advance=1)
                            continue

                    # Save frame
                    out_name = f"{prefix}{video_stem}_{tag}_f{frame_idx:06d}.jpg"
                    out_path = target_dir / out_name
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_files.append(str(out_path))
                    last_saved_frame = frame.copy()

                frame_idx += 1
                progress.update(task, advance=1)

        cap.release()

        return {
            "video": str(video_path),
            "is_negative": save_as_negatives,
            "processed_samples": processed_count,
            "saved_frames": len(saved_files),
            "discarded_blur": discarded_blur,
            "discarded_dedup": discarded_dedup,
            "saved_paths": saved_files,
        }

    def run(self, video_path: Optional[str] = None) -> Dict[str, Any]:
        """Runs Stage 1 extraction across all specified videos."""
        self.state_mgr.start_stage(Stage.EXTRACT)

        v_input = video_path or self.ext_cfg.get("video_path", "data/videos")
        video_files = self._find_video_files(v_input)

        total_saved_pos = 0
        total_saved_neg = 0
        total_blur = 0
        total_dedup = 0
        all_saved_paths = []
        all_neg_paths = []

        console.print(f"[bold green]Starting Frame Extraction Stage[/bold green]")
        console.print(f"Found [bold]{len(video_files)}[/bold] training video(s).")

        for idx, vf in enumerate(video_files):
            stats = self.extract_from_video(vf, save_as_negatives=False, prefix=f"v{idx:02d}_")
            total_saved_pos += stats["saved_frames"]
            total_blur += stats["discarded_blur"]
            total_dedup += stats["discarded_dedup"]
            all_saved_paths.extend(stats["saved_paths"])

        # Dedicated negative videos if provided
        if self.neg_enabled and self.neg_video_path:
            neg_path = Path(self.neg_video_path)
            if neg_path.exists():
                neg_videos = self._find_video_files(str(neg_path))
                console.print(f"Found [bold]{len(neg_videos)}[/bold] background/negative video(s).")
                for idx, nvf in enumerate(neg_videos):
                    stats = self.extract_from_video(nvf, save_as_negatives=True, prefix=f"neg{idx:02d}_")
                    total_saved_neg += stats["saved_frames"]
                    total_blur += stats["discarded_blur"]
                    total_dedup += stats["discarded_dedup"]
                    all_neg_paths.extend(stats["saved_paths"])

        summary = {
            "total_positive_frames": total_saved_pos,
            "total_negative_frames": total_saved_neg,
            "discarded_blurry": total_blur,
            "discarded_duplicates": total_dedup,
            "output_dir": str(self.output_dir),
            "negative_dir": str(self.negative_dir),
        }

        self.state_mgr.complete_stage(
            Stage.EXTRACT,
            artifacts={
                "frames_dir": str(self.output_dir),
                "negatives_dir": str(self.negative_dir),
                "sample_count": total_saved_pos + total_saved_neg,
            },
            metrics=summary,
        )

        console.print(
            f"[bold green]Stage 1 Complete:[/bold green] "
            f"Extracted {total_saved_pos} positive frames, {total_saved_neg} background frames. "
            f"(Filtered: {total_blur} blurry, {total_dedup} duplicates)"
        )
        return summary
