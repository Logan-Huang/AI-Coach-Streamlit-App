from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_COMPLEXITY_TO_MODEL = {0: "lite", 1: "full", 2: "heavy"}
_MODEL_MIN_BYTES = 1_000_000
_download_lock = threading.Lock()


class PoseIdx(IntEnum):
    """Pose landmark indices (unchanged between the legacy Solutions API and the Tasks API)."""

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# Standard 33-landmark skeleton topology (the Tasks API ships no drawing helper).
_POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

ProgressCallback = Optional[Callable[[int, Optional[int]], None]]


@dataclass
class AnalysisConfig:
    sample_stride: int = 2
    max_frames: Optional[int] = None
    min_det_conf: float = 0.5
    min_track_conf: float = 0.5
    model_complexity: int = 1
    annotate_video: bool = True


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def safe_filename(name: str, default: str = "uploaded_video.mp4") -> str:
    base = os.path.basename(name or default).strip().replace(" ", "_")
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return base or default


def _model_cache_dir() -> str:
    base = os.environ.get("TENNIS_AI_MODEL_DIR") or os.path.join(tempfile.gettempdir(), "tennis_ai_models")
    return ensure_dir(base)


def model_name_for_complexity(complexity: int) -> str:
    return _COMPLEXITY_TO_MODEL.get(int(complexity), "full")


def ensure_pose_model(model_name: str = "full") -> str:
    """Download the PoseLandmarker .task model once and cache it on disk."""
    if model_name not in _MODEL_URLS:
        raise ValueError(f"Unknown pose model: {model_name!r}")
    dest = os.path.join(_model_cache_dir(), f"pose_landmarker_{model_name}.task")
    if os.path.exists(dest) and os.path.getsize(dest) >= _MODEL_MIN_BYTES:
        return dest
    url = _MODEL_URLS[model_name]
    with _download_lock:
        if os.path.exists(dest) and os.path.getsize(dest) >= _MODEL_MIN_BYTES:
            return dest
        fd, tmp = tempfile.mkstemp(suffix=".task", dir=_model_cache_dir())
        os.close(fd)
        last_err: Optional[Exception] = None
        try:
            for _ in range(2):
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
                        shutil.copyfileobj(resp, out)
                    if os.path.getsize(tmp) < _MODEL_MIN_BYTES:
                        raise RuntimeError("Downloaded model file looks truncated.")
                    os.replace(tmp, dest)
                    return dest
                except Exception as exc:
                    last_err = exc
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    raise RuntimeError(f"Could not download pose model '{model_name}' from {url}: {last_err}")


def _draw_pose(frame: np.ndarray, landmarks, width: int, height: int) -> None:
    pts = [(int(round(lm.x * width)), int(round(lm.y * height))) for lm in landmarks]
    for a, b in _POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 200, 0), -1, cv2.LINE_AA)


def run_cmd(cmd: List[str], timeout_s: float = 600.0) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {int(timeout_s)}s: " + " ".join(cmd)) from exc
    if proc.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc


def ffmpeg_executable() -> Optional[str]:
    """Return a usable FFmpeg executable.

    Streamlit Cloud apt packages can be fragile across Debian image updates, so
    this app first checks for a system FFmpeg and then falls back to the
    imageio-ffmpeg Python wheel, which bundles a static FFmpeg binary.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg and os.path.exists(bundled_ffmpeg):
            return bundled_ffmpeg
    except Exception:
        return None

    return None


def ffmpeg_available() -> bool:
    return ffmpeg_executable() is not None


def convert_to_mp4_if_needed(video_path: str, out_dir: str, force: bool = False) -> str:
    """Convert videos such as MOV to H.264 MP4 for better OpenCV compatibility."""
    ensure_dir(out_dir)
    ext = Path(video_path).suffix.lower()
    if ext in {".mp4", ".m4v"} and not force:
        return video_path

    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return video_path

    output_name = safe_filename(Path(video_path).stem) + "_converted.mp4"
    out_path = os.path.join(out_dir, output_name)
    run_cmd([
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ])
    return out_path


def angle_deg(a: np.ndarray | None, b: np.ndarray | None, c: np.ndarray | None) -> float:
    if a is None or b is None or c is None:
        return float("nan")
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-9:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def safe_norm(v: np.ndarray | None) -> float:
    return float(np.linalg.norm(v)) if v is not None else float("nan")


def signed_angle_from_vertical_deg(vec: np.ndarray | None) -> float:
    """Signed lean angle from vertical in image coordinates."""
    if vec is None or safe_norm(vec) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(vec[0], -vec[1])))


def tilt_from_horizontal_deg(vec: np.ndarray | None) -> float:
    if vec is None or safe_norm(vec) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def fmt(x: object, nd: int = 1) -> str:
    try:
        value = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{nd}f}"


def finite_array(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def safe_nanmedian(values: object) -> float:
    arr = finite_array(values)
    return float(np.median(arr)) if arr.size else float("nan")


def safe_nanmin(values: object) -> float:
    arr = finite_array(values)
    return float(np.min(arr)) if arr.size else float("nan")


def safe_percentile(values: object, percentile: float, default: float = float("nan")) -> float:
    arr = finite_array(values)
    return float(np.percentile(arr, percentile)) if arr.size else default


def rowwise_nanmin_two(a: object, b: object) -> np.ndarray:
    arr = np.vstack([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
    mask = np.isfinite(arr)
    replaced = np.where(mask, arr, np.inf)
    mins = np.min(replaced, axis=0)
    mins[np.sum(mask, axis=0) == 0] = np.nan
    return mins


def _lm_xy(lm, idx: int, width: int, height: int) -> np.ndarray:
    return np.array([lm[idx].x * width, lm[idx].y * height], dtype=np.float32)


def _make_web_friendly_video(raw_path: str, final_path: str) -> Optional[str]:
    if not raw_path or not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return None

    ffmpeg = ffmpeg_executable()
    if ffmpeg:
        try:
            run_cmd([
                ffmpeg,
                "-y",
                "-i",
                raw_path,
                "-vcodec",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                final_path,
            ])
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                if os.path.abspath(raw_path) != os.path.abspath(final_path):
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                return final_path
        except Exception:
            pass

    # Without FFmpeg the raw mp4v file will not play in the browser's HTML5
    # player, so drop it rather than presenting an unplayable video.
    try:
        os.remove(raw_path)
    except OSError:
        pass
    return None


def analyze_video(
    video_path: str,
    out_dir: str,
    cfg: AnalysisConfig,
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV could not open this video. Try enabling force conversion to MP4, "
            "or upload an MP4 encoded with H.264."
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    raw_annotated_path = os.path.join(out_dir, "annotated_video_raw.mp4")
    annotated_path = os.path.join(out_dir, "annotated_video.mp4")

    if cfg.annotate_video and width > 0 and height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_annotated_path, fourcc, fps / max(1, cfg.sample_stride), (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None

    try:
        model_path = ensure_pose_model(model_name_for_complexity(cfg.model_complexity))
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=float(cfg.min_det_conf),
                min_tracking_confidence=float(cfg.min_track_conf),
            )
        )
    except Exception:
        cap.release()
        if writer is not None:
            writer.release()
        raise

    prev_pts = {"LEFT_WRIST": None, "RIGHT_WRIST": None}
    prev_frame_i = {"LEFT_WRIST": None, "RIGHT_WRIST": None}
    last_ts = -1
    rows: List[Dict[str, float | int]] = []
    frame_i = -1
    processed = 0
    sample_stride = max(1, int(cfg.sample_stride))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            if progress_callback and (frame_i % 10 == 0 or frame_i == total_frames - 1):
                progress_callback(frame_i + 1, total_frames if total_frames > 0 else None)

            if frame_i % sample_stride != 0:
                continue

            time_s = frame_i / fps
            timestamp_ms = int(round(time_s * 1000.0))
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            left_knee = right_knee = float("nan")
            left_elbow = right_elbow = float("nan")
            torso_lean = float("nan")
            shoulder_tilt = float("nan")
            hip_tilt = float("nan")
            stance_ratio = float("nan")
            left_wrist_speed = right_wrist_speed = float("nan")

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]

                l_sh = _lm_xy(lm, PoseIdx.LEFT_SHOULDER, width, height)
                r_sh = _lm_xy(lm, PoseIdx.RIGHT_SHOULDER, width, height)
                l_hp = _lm_xy(lm, PoseIdx.LEFT_HIP, width, height)
                r_hp = _lm_xy(lm, PoseIdx.RIGHT_HIP, width, height)
                l_kn = _lm_xy(lm, PoseIdx.LEFT_KNEE, width, height)
                r_kn = _lm_xy(lm, PoseIdx.RIGHT_KNEE, width, height)
                l_an = _lm_xy(lm, PoseIdx.LEFT_ANKLE, width, height)
                r_an = _lm_xy(lm, PoseIdx.RIGHT_ANKLE, width, height)
                l_el = _lm_xy(lm, PoseIdx.LEFT_ELBOW, width, height)
                r_el = _lm_xy(lm, PoseIdx.RIGHT_ELBOW, width, height)
                l_wr = _lm_xy(lm, PoseIdx.LEFT_WRIST, width, height)
                r_wr = _lm_xy(lm, PoseIdx.RIGHT_WRIST, width, height)

                left_knee = angle_deg(l_hp, l_kn, l_an)
                right_knee = angle_deg(r_hp, r_kn, r_an)
                left_elbow = angle_deg(l_sh, l_el, l_wr)
                right_elbow = angle_deg(r_sh, r_el, r_wr)

                shoulder_center = 0.5 * (l_sh + r_sh)
                hip_center = 0.5 * (l_hp + r_hp)
                torso_lean = signed_angle_from_vertical_deg(shoulder_center - hip_center)
                shoulder_tilt = tilt_from_horizontal_deg(r_sh - l_sh)
                hip_tilt = tilt_from_horizontal_deg(r_hp - l_hp)

                ankle_dist = safe_norm(r_an - l_an)
                hip_dist = safe_norm(r_hp - l_hp)
                if math.isfinite(ankle_dist) and math.isfinite(hip_dist) and hip_dist > 1e-6:
                    stance_ratio = float(ankle_dist / hip_dist)

                # Divide by the real elapsed time since the wrist was last seen,
                # so pose-detection gaps do not inflate speeds into false peaks.
                if prev_pts["LEFT_WRIST"] is not None and prev_frame_i["LEFT_WRIST"] is not None:
                    gap_s = (frame_i - prev_frame_i["LEFT_WRIST"]) / fps
                    if gap_s > 0:
                        left_wrist_speed = float(np.linalg.norm(l_wr - prev_pts["LEFT_WRIST"]) / gap_s)
                if prev_pts["RIGHT_WRIST"] is not None and prev_frame_i["RIGHT_WRIST"] is not None:
                    gap_s = (frame_i - prev_frame_i["RIGHT_WRIST"]) / fps
                    if gap_s > 0:
                        right_wrist_speed = float(np.linalg.norm(r_wr - prev_pts["RIGHT_WRIST"]) / gap_s)

                prev_pts["LEFT_WRIST"] = l_wr
                prev_pts["RIGHT_WRIST"] = r_wr
                prev_frame_i["LEFT_WRIST"] = frame_i
                prev_frame_i["RIGHT_WRIST"] = frame_i

                if writer is not None:
                    _draw_pose(frame, lm, width, height)

            if writer is not None:
                overlay = (
                    f"t={time_s:.2f}s | knee L/R={fmt(left_knee)}/{fmt(right_knee)} deg | "
                    f"lean={fmt(torso_lean)} deg | stance={fmt(stance_ratio)} | "
                    f"wrist speed L/R={fmt(left_wrist_speed, 0)}/{fmt(right_wrist_speed, 0)} px/s"
                )
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)

            rows.append({
                "time_s": float(time_s),
                "frame_i": int(frame_i),
                "left_knee_angle_deg": float(left_knee),
                "right_knee_angle_deg": float(right_knee),
                "left_elbow_angle_deg": float(left_elbow),
                "right_elbow_angle_deg": float(right_elbow),
                "torso_lean_deg": float(torso_lean),
                "torso_lean_abs_deg": abs(float(torso_lean)) if math.isfinite(float(torso_lean)) else float("nan"),
                "shoulder_tilt_deg": float(shoulder_tilt),
                "hip_tilt_deg": float(hip_tilt),
                "stance_width_ratio": float(stance_ratio),
                "left_wrist_speed_px_s": float(left_wrist_speed),
                "right_wrist_speed_px_s": float(right_wrist_speed),
            })

            processed += 1
            if cfg.max_frames is not None and processed >= int(cfg.max_frames):
                break
    finally:
        cap.release()
        landmarker.close()
        if writer is not None:
            writer.release()

    final_annotated = None
    if writer is not None and cfg.annotate_video:
        final_annotated = _make_web_friendly_video(raw_annotated_path, annotated_path)

    df = pd.DataFrame(rows)
    meta: Dict[str, object] = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "processed_frames": processed,
        "sample_stride": sample_stride,
        "config": asdict(cfg),
        "annotated_video": final_annotated,
    }
    return df, meta


def pick_hitting_arm(metrics_df: pd.DataFrame) -> str:
    left = finite_array(metrics_df.get("left_wrist_speed_px_s", []))
    right = finite_array(metrics_df.get("right_wrist_speed_px_s", []))
    if left.size == 0 and right.size == 0:
        return "RIGHT"
    left_p90 = safe_percentile(left, 90, default=-1.0)
    right_p90 = safe_percentile(right, 90, default=-1.0)
    return "LEFT" if left_p90 > right_p90 else "RIGHT"


def detect_peaks_1d(y: np.ndarray, thresh: float, min_gap: int) -> List[int]:
    peaks: List[int] = []
    min_gap = max(1, int(min_gap))
    for i in range(1, len(y) - 1):
        if not np.isfinite(y[i]):
            continue
        if y[i] > thresh and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)
            elif y[i] > y[peaks[-1]]:
                peaks[-1] = i
    return peaks


def build_strokes_df(metrics_df: pd.DataFrame, fps: float, sample_stride: int) -> Tuple[pd.DataFrame, str, float]:
    columns = [
        "stroke_id",
        "peak_time_s",
        "peak_frame_i",
        "hitting_arm_guess",
        "peak_speed_px_s",
        "min_knee_angle_deg",
        "stance_width_ratio_med",
        "torso_lean_abs_deg_med",
        "hitting_elbow_angle_deg_med",
    ]
    if metrics_df.empty:
        return pd.DataFrame(columns=columns), "RIGHT", 0.0

    hitting_arm = pick_hitting_arm(metrics_df)
    speed_col = "left_wrist_speed_px_s" if hitting_arm == "LEFT" else "right_wrist_speed_px_s"
    speed = metrics_df[speed_col].to_numpy(dtype=float)
    # p95 over real (finite) speeds only, so undetected frames don't drag the threshold down.
    p95 = safe_percentile(speed, 95, default=0.0)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    thresh = max(300.0, 0.35 * p95)
    min_gap = int(round(0.45 * float(fps) / max(1, int(sample_stride))))
    peaks = detect_peaks_1d(speed, thresh=thresh, min_gap=min_gap)

    win = int(round(0.25 * float(fps) / max(1, int(sample_stride))))
    rows: List[Dict[str, object]] = []

    for stroke_id, peak_idx in enumerate(peaks, start=1):
        start = max(0, peak_idx - win)
        stop = min(len(metrics_df) - 1, peak_idx + win)
        seg = metrics_df.iloc[start:stop + 1]

        knee_each = rowwise_nanmin_two(seg["left_knee_angle_deg"].to_numpy(), seg["right_knee_angle_deg"].to_numpy())
        knee_min = safe_nanmin(knee_each)
        stance_med = safe_nanmedian(seg["stance_width_ratio"].to_numpy())
        lean_abs_med = safe_nanmedian(seg["torso_lean_abs_deg"].to_numpy())
        elbow_col = "left_elbow_angle_deg" if hitting_arm == "LEFT" else "right_elbow_angle_deg"
        elbow_med = safe_nanmedian(seg[elbow_col].to_numpy())

        rows.append({
            "stroke_id": int(stroke_id),
            "peak_time_s": float(metrics_df.iloc[peak_idx]["time_s"]),
            "peak_frame_i": int(metrics_df.iloc[peak_idx]["frame_i"]),
            "hitting_arm_guess": hitting_arm,
            "peak_speed_px_s": float(speed[peak_idx]),
            "min_knee_angle_deg": float(knee_min),
            "stance_width_ratio_med": float(stance_med),
            "torso_lean_abs_deg_med": float(lean_abs_med),
            "hitting_elbow_angle_deg_med": float(elbow_med),
        })

    return pd.DataFrame(rows, columns=columns), hitting_arm, float(thresh)


def summarize_metrics(metrics_df: pd.DataFrame, strokes_df: pd.DataFrame) -> Dict[str, float | int]:
    summary: Dict[str, float | int] = {}
    summary["duration_s"] = float(metrics_df["time_s"].max()) if len(metrics_df) else 0.0
    summary["frames_processed"] = int(len(metrics_df))

    if len(metrics_df):
        knee_min_each = rowwise_nanmin_two(
            metrics_df["left_knee_angle_deg"].to_numpy(),
            metrics_df["right_knee_angle_deg"].to_numpy(),
        )
        summary["knee_min_deg_global_median"] = safe_nanmedian(knee_min_each)
        summary["torso_lean_abs_deg_global_median"] = safe_nanmedian(metrics_df["torso_lean_abs_deg"].to_numpy())
        summary["stance_ratio_global_median"] = safe_nanmedian(metrics_df["stance_width_ratio"].to_numpy())
    else:
        summary["knee_min_deg_global_median"] = float("nan")
        summary["torso_lean_abs_deg_global_median"] = float("nan")
        summary["stance_ratio_global_median"] = float("nan")

    if len(strokes_df):
        summary["strokes_detected"] = int(len(strokes_df))
        summary["knee_min_deg_at_strokes_median"] = safe_nanmedian(strokes_df["min_knee_angle_deg"].to_numpy())
        summary["torso_lean_abs_deg_at_strokes_median"] = safe_nanmedian(strokes_df["torso_lean_abs_deg_med"].to_numpy())
        summary["stance_ratio_at_strokes_median"] = safe_nanmedian(strokes_df["stance_width_ratio_med"].to_numpy())
        summary["elbow_angle_at_strokes_median"] = safe_nanmedian(strokes_df["hitting_elbow_angle_deg_med"].to_numpy())
        summary["peak_speed_px_s_median"] = safe_nanmedian(strokes_df["peak_speed_px_s"].to_numpy())
    else:
        summary["strokes_detected"] = 0

    return summary


def generate_suggestions(summary: Dict[str, float | int], hitting_arm: str) -> Tuple[List[str], List[str]]:
    good: List[str] = []
    focus: List[str] = []

    knee = float(summary.get("knee_min_deg_at_strokes_median", summary.get("knee_min_deg_global_median", float("nan"))))
    if math.isfinite(knee):
        if knee > 155:
            focus.append("Bend your knees more during the loading phase before you swing. Aim for a clearly athletic position, often around 120 to 145 degrees depending on the shot.")
        elif knee < 110:
            focus.append("You get very low on some swings. If it feels unstable, try keeping knee bend while staying stacked, with hips under shoulders, instead of collapsing.")
        else:
            good.append("Knee bend looks generally athletic on many swings.")

    stance = float(summary.get("stance_ratio_at_strokes_median", summary.get("stance_ratio_global_median", float("nan"))))
    if math.isfinite(stance):
        if stance < 0.95:
            focus.append("Your base looks narrow at times. Try a slightly wider stance to improve balance and power transfer.")
        elif stance > 1.9:
            focus.append("Your base can be very wide. Practice finding a balanced width that still lets you rotate and recover quickly.")
        else:
            good.append("Stance width looks balanced most of the time.")

    lean = float(summary.get("torso_lean_abs_deg_at_strokes_median", summary.get("torso_lean_abs_deg_global_median", float("nan"))))
    if math.isfinite(lean):
        if lean > 22:
            focus.append("You lean your torso a lot during swings. Focus on staying more centered and rotating through the shot rather than tipping sideways.")
        else:
            good.append("Torso stays relatively centered on many swings.")

    elbow = float(summary.get("elbow_angle_at_strokes_median", float("nan")))
    if math.isfinite(elbow):
        if elbow < 75:
            focus.append(f"Your {hitting_arm.lower()} hitting-arm elbow looks quite bent during fast swings. Work on creating space and extending through contact when appropriate.")
        elif elbow > 155:
            focus.append(f"Your {hitting_arm.lower()} hitting arm can look very straight at times. If contact feels late or jammed, practice a relaxed arm and smooth extension, not a locked arm.")
        else:
            good.append("Hitting-arm elbow position looks reasonable on many swings.")

    if int(summary.get("strokes_detected", 0)) < 4:
        focus.append("Only a few swing moments were detected. For best results, use a video where the player is clearly visible.")

    focus.append("For more accurate feedback, film with the full body visible, good lighting, minimal camera shake, and a camera near hip height from a side-on view.")
    return good, focus


def render_report_md(
    video_name: str,
    meta: Dict[str, object],
    summary: Dict[str, float | int],
    hitting_arm: str,
    good: List[str],
    focus: List[str],
) -> str:
    good_lines = "\n".join([f"- {x}" for x in good]) or "- n/a; insufficient landmarks or visibility in the video"
    focus_lines = "\n".join([f"- {x}" for x in focus]) or "- n/a"
    return f"""# Tennis AI Coach Report (Pose-based)

**Video:** `{video_name}`  
**Duration analyzed:** {fmt(summary.get('duration_s'))} s  
**Frames processed:** {int(summary.get('frames_processed', 0))} with stride={int(meta.get('sample_stride', 1))}  
**Detected swing moments:** {int(summary.get('strokes_detected', 0))}

## Snapshot metrics (median)

- Knee bend, smaller means more bend: **{fmt(summary.get('knee_min_deg_at_strokes_median', float('nan')))} deg** at swings
- Torso lean, absolute: **{fmt(summary.get('torso_lean_abs_deg_at_strokes_median', float('nan')))} deg** at swings
- Stance width ratio: **{fmt(summary.get('stance_ratio_at_strokes_median', float('nan')))}** at swings
- Hitting-arm elbow angle: **{fmt(summary.get('elbow_angle_at_strokes_median', float('nan')))} deg**; arm guess: **{hitting_arm}**
- Peak wrist speed: **{fmt(summary.get('peak_speed_px_s_median', float('nan')), 0)} px/s**

## What looks good

{good_lines}

## Focus next

{focus_lines}

## Important limitations

This is a 2D pose-estimation analysis from a single camera angle. Treat the output as general coaching cues, not as a clinical or professional biomechanical assessment.
"""


def save_output_bundle(
    out_dir: str,
    metrics_df: pd.DataFrame,
    strokes_df: pd.DataFrame,
    report_md: str,
    meta: Dict[str, object],
) -> Dict[str, str]:
    ensure_dir(out_dir)
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    strokes_csv = os.path.join(out_dir, "strokes.csv")
    report_path = os.path.join(out_dir, "report.md")
    meta_path = os.path.join(out_dir, "meta.json")
    zip_path = os.path.join(out_dir, "tennis_ai_outputs.zip")

    metrics_df.to_csv(metrics_csv, index=False)
    strokes_df.to_csv(strokes_csv, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(out_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                if os.path.abspath(full_path) == os.path.abspath(zip_path):
                    continue
                rel_path = os.path.relpath(full_path, out_dir)
                zf.write(full_path, arcname=os.path.join("tennis_ai_outputs", rel_path))

    return {
        "metrics_csv": metrics_csv,
        "strokes_csv": strokes_csv,
        "report_md": report_path,
        "meta_json": meta_path,
        "zip": zip_path,
    }


def read_bytes(path: Optional[str]) -> Optional[bytes]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()
