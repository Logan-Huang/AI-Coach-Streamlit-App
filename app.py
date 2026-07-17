from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from tennis_analysis import (
    AnalysisConfig,
    build_strokes_df,
    convert_to_mp4_if_needed,
    ensure_dir,
    fmt,
    generate_suggestions,
    read_bytes,
    render_report_md,
    safe_filename,
    save_output_bundle,
    summarize_metrics,
    analyze_video,
)

st.set_page_config(
    page_title="Tennis AI Coach",
    page_icon="🎾",
    layout="wide",
)


def metric_value(value: object, suffix: str = "", nd: int = 1) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{nd}f}{suffix}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_knee_series(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["time_s", "knee_bend_min_deg"])
    left = metrics_df["left_knee_angle_deg"].to_numpy(dtype=float)
    right = metrics_df["right_knee_angle_deg"].to_numpy(dtype=float)
    stacked = np.vstack([left, right])
    mask = np.isfinite(stacked)
    replaced = np.where(mask, stacked, np.inf)
    knee = np.min(replaced, axis=0)
    knee[np.sum(mask, axis=0) == 0] = np.nan
    return pd.DataFrame({"time_s": metrics_df["time_s"].to_numpy(), "knee_bend_min_deg": knee})


def run_pipeline(
    uploaded_file,
    sample_stride: int,
    max_frames: Optional[int],
    model_complexity: int,
    min_det_conf: float,
    min_track_conf: float,
    annotate_video: bool,
    force_convert: bool,
) -> Dict[str, object]:
    # One work dir per session run: drop the previous run's artifacts so temp
    # disk usage stays bounded on Streamlit Cloud.
    prev_dir = st.session_state.pop("work_dir", None)
    if prev_dir and os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir, ignore_errors=True)
    st.session_state.pop("result", None)

    work_dir = tempfile.mkdtemp(prefix="tennis_ai_")
    st.session_state["work_dir"] = work_dir
    input_name = safe_filename(uploaded_file.name)
    input_path = os.path.join(work_dir, input_name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    out_dir = ensure_dir(os.path.join(work_dir, "tennis_ai_outputs"))
    status = st.empty()
    progress = st.progress(0)

    status.info("Preparing video...")
    # Convert into work_dir (not out_dir) so the input video is not bundled
    # into the downloadable outputs ZIP.
    video_path = convert_to_mp4_if_needed(input_path, work_dir, force=force_convert)

    cfg = AnalysisConfig(
        sample_stride=sample_stride,
        max_frames=max_frames,
        min_det_conf=min_det_conf,
        min_track_conf=min_track_conf,
        model_complexity=model_complexity,
        annotate_video=annotate_video,
    )

    def on_progress(current: int, total: Optional[int]) -> None:
        if total:
            progress.progress(min(current / total, 1.0))
            status.info(f"Processing frame {current:,} of {total:,}...")
        else:
            status.info(f"Processing frame {current:,}...")

    metrics_df, meta = analyze_video(video_path, out_dir, cfg, progress_callback=on_progress)
    progress.progress(1.0)
    status.info("Detecting swing moments and generating report...")

    strokes_df, hitting_arm, peak_thresh = build_strokes_df(metrics_df, float(meta["fps"]), int(meta["sample_stride"]))
    summary = summarize_metrics(metrics_df, strokes_df)
    good, focus = generate_suggestions(summary, hitting_arm)
    report_md = render_report_md(input_name, meta, summary, hitting_arm, good, focus)
    bundle_paths = save_output_bundle(out_dir, metrics_df, strokes_df, report_md, meta)

    annotated_path = meta.get("annotated_video")
    status.success("Analysis complete.")

    # Large artifacts (annotated video, zip) stay on disk and are read lazily
    # at render time instead of living in session_state for the whole session.
    return {
        "input_name": input_name,
        "metrics_df": metrics_df,
        "strokes_df": strokes_df,
        "summary": summary,
        "hitting_arm": hitting_arm,
        "peak_thresh": peak_thresh,
        "report_md": report_md,
        "meta": meta,
        "metrics_csv_bytes": to_csv_bytes(metrics_df),
        "strokes_csv_bytes": to_csv_bytes(strokes_df),
        "report_bytes": report_md.encode("utf-8"),
        "zip_path": bundle_paths["zip"],
        "annotated_video_path": str(annotated_path) if annotated_path else None,
    }


st.title("Tennis AI Coach")
st.caption("Upload a tennis video, run MediaPipe pose tracking, detect likely swing moments, and download metrics, report, and annotated video.")

with st.sidebar:
    st.header("Analysis settings")
    sample_stride = st.slider("Process every Nth frame", min_value=1, max_value=8, value=2, help="Higher values are faster but less detailed.")
    max_processed_frames = st.number_input("Max processed frames", min_value=0, max_value=100000, value=0, step=100, help="0 means analyze the whole video.")
    model_complexity = st.select_slider("Pose model complexity", options=[0, 1, 2], value=1, help="0 = lite (fastest), 1 = full, 2 = heavy (most accurate, slowest). The matching pose model is downloaded on first use.")
    min_det_conf = st.slider("Minimum detection confidence", 0.1, 0.9, 0.5, 0.05)
    min_track_conf = st.slider("Minimum tracking confidence", 0.1, 0.9, 0.5, 0.05)
    annotate_video = st.checkbox("Create annotated video", value=True)
    force_convert = st.checkbox("Force conversion to web MP4 first", value=False, help="Useful when OpenCV cannot read a phone MOV/HEVC video.")

uploaded = st.file_uploader("Upload a tennis video", type=["mp4", "mov", "m4v", "avi"], accept_multiple_files=False)

left, right = st.columns([1, 2])
with left:
    run_clicked = st.button("Run analysis", type="primary", disabled=uploaded is None, width="stretch")
with right:
    if uploaded is not None:
        st.write(f"Selected: `{uploaded.name}`")
    else:
        st.write("Upload an MP4, MOV, M4V, or AVI file to begin.")

if run_clicked and uploaded is not None:
    try:
        st.session_state["result"] = run_pipeline(
            uploaded_file=uploaded,
            sample_stride=int(sample_stride),
            max_frames=None if int(max_processed_frames) == 0 else int(max_processed_frames),
            model_complexity=int(model_complexity),
            min_det_conf=float(min_det_conf),
            min_track_conf=float(min_track_conf),
            annotate_video=bool(annotate_video),
            force_convert=bool(force_convert),
        )
    except Exception as exc:
        st.error("Analysis failed. Try a shorter H.264 MP4 video, increase the sample stride, or enable force conversion.")
        st.exception(exc)

result = st.session_state.get("result")

if not result:
    st.info("This app runs pose estimation on the server. For best results, use a side-on clip with the full body visible and good lighting.")
    st.stop()

summary = result["summary"]
metrics_df = result["metrics_df"]
strokes_df = result["strokes_df"]
meta = result["meta"]

st.subheader("Results")
metric_cols = st.columns(5)
metric_cols[0].metric("Duration", metric_value(summary.get("duration_s"), " s"))
metric_cols[1].metric("Frames processed", f"{int(summary.get('frames_processed', 0)):,}")
metric_cols[2].metric("Detected swings", f"{int(summary.get('strokes_detected', 0)):,}")
metric_cols[3].metric("Hitting arm guess", str(result["hitting_arm"]))
metric_cols[4].metric("Peak threshold", metric_value(result.get("peak_thresh"), " px/s", nd=0))

report_tab, charts_tab, data_tab, video_tab, downloads_tab = st.tabs(["Report", "Charts", "Data", "Annotated video", "Downloads"])

with report_tab:
    st.markdown(result["report_md"])

with charts_tab:
    if metrics_df.empty:
        st.warning("No frame metrics were produced.")
    else:
        chart_df = metrics_df.set_index("time_s")
        st.write("Torso lean over time")
        st.line_chart(chart_df[["torso_lean_abs_deg"]])
        st.write("Stance width ratio over time")
        st.line_chart(chart_df[["stance_width_ratio"]])
        st.write("Knee bend over time, smaller means more bend")
        knee_df = build_knee_series(metrics_df).set_index("time_s")
        st.line_chart(knee_df)
        speed_col = "left_wrist_speed_px_s" if result["hitting_arm"] == "LEFT" else "right_wrist_speed_px_s"
        st.write(f"{result['hitting_arm'].title()} wrist speed over time")
        st.line_chart(chart_df[[speed_col]])

with data_tab:
    st.write("Swing moments")
    st.dataframe(strokes_df, width="stretch")
    st.write("Per-frame metrics")
    st.dataframe(metrics_df, width="stretch")

with video_tab:
    video_path = result.get("annotated_video_path")
    if video_path and os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("No annotated video was generated. Enable 'Create annotated video' and rerun if you want one.")

with downloads_tab:
    st.write("Download everything as a ZIP, or download individual files.")
    zip_bytes = read_bytes(result.get("zip_path"))
    if zip_bytes:
        st.download_button(
            "Download all outputs (.zip)",
            data=zip_bytes,
            file_name="tennis_ai_outputs.zip",
            mime="application/zip",
            width="stretch",
        )
    else:
        st.warning("The output bundle is no longer on disk. Rerun the analysis to regenerate it.")
    col1, col2, col3 = st.columns(3)
    col1.download_button("metrics.csv", data=result["metrics_csv_bytes"], file_name="metrics.csv", mime="text/csv", width="stretch")
    col2.download_button("strokes.csv", data=result["strokes_csv_bytes"], file_name="strokes.csv", mime="text/csv", width="stretch")
    col3.download_button("report.md", data=result["report_bytes"], file_name="report.md", mime="text/markdown", width="stretch")
    annotated_dl = read_bytes(result.get("annotated_video_path"))
    if annotated_dl:
        st.download_button(
            "annotated_video.mp4",
            data=annotated_dl,
            file_name="annotated_video.mp4",
            mime="video/mp4",
            width="stretch",
        )

with st.expander("Technical metadata"):
    st.json(meta)
