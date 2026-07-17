# Tennis AI Coach Streamlit App

This repository turns the original Colab notebook into a deployable Streamlit app.

The app lets a user upload a tennis video, runs MediaPipe pose tracking, extracts per-frame metrics, detects likely swing moments from wrist-speed peaks, and provides downloads for:

- `metrics.csv`: per-frame pose metrics
- `strokes.csv`: detected swing moments and per-stroke summaries
- `annotated_video.mp4`: pose skeleton overlay and metrics overlay
- `report.md`: rule-based coaching report
- `tennis_ai_outputs.zip`: all generated outputs

## Important Streamlit Cloud notes

This version targets **Python 3.14** and **mediapipe 0.10.35** using MediaPipe's current
Tasks API (`PoseLandmarker`). The legacy `mp.solutions` API this app originally used was
removed from mediapipe 0.10.31+, and older mediapipe versions have no Python 3.14 wheels.
The pose model (`pose_landmarker_lite/full/heavy.task`, ~5-29 MB) is downloaded from
Google's model storage on first use and cached in the temp directory.

`packages.txt` contains exactly one apt package: `libportaudio2`. mediapipe's Tasks
import chain pulls in `sounddevice`, which needs the PortAudio system library on Linux.
It is a tiny leaf package and does not reintroduce the heavy apt packages (`ffmpeg`,
`libgl1`, `libglib2.0-0`) that caused Debian dependency conflicts in earlier versions.
Those are still avoided by:

- using a headless OpenCV wheel, so no GUI/OpenGL Linux packages are needed;
- using `imageio-ffmpeg`, so FFmpeg is available from Python instead of apt;
- a small local compatibility shim so MediaPipe's `opencv-contrib-python` dependency resolves to `opencv-contrib-python-headless`.

## Project files

```text
tennis-ai-coach/
├── app.py
├── tennis_analysis.py
├── requirements.txt
├── packages.txt
├── opencv_contrib_python_stub/
│   └── pyproject.toml
├── .python-version
├── .gitignore
├── .streamlit/
│   └── config.toml
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Then run:

```powershell
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository, or update your existing repository.
2. Upload all files in this folder to the repository root (including `packages.txt`).
3. Go to Streamlit Community Cloud and create or redeploy the app.
4. Select the GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Select **Python 3.14** in Streamlit advanced settings, then deploy.

Note: only the "Python version" dropdown in Advanced settings controls the deployed
interpreter — no file in the repo does (`.python-version` is for local tooling only).
The Python version cannot be changed after deployment; to switch it you must delete
the app and redeploy. If you add `packages.txt` to an already-deployed app and the
deploy logs show no apt install phase, delete and redeploy to force a fresh container.

## Notes and limitations

- This is a 2D pose-estimation tool from a single camera angle, not a professional biomechanical assessment.
- Phone MOV/HEVC videos may need conversion. Use the app setting "Force conversion to web MP4 first" if OpenCV cannot read the file.
- Large videos take longer and may hit memory or time limits on free hosting. For faster runs, increase "Process every Nth frame" or set a max processed frame count.
- Do not upload sensitive, controlled, private, or regulated videos to a public deployment.
