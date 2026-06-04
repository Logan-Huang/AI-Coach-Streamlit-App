# Tennis AI Coach Streamlit App

This repository turns the original Colab notebook into a deployable Streamlit app.

The app lets a user upload a tennis video, runs MediaPipe pose tracking, extracts per-frame metrics, detects likely swing moments from wrist-speed peaks, and provides downloads for:

- `metrics.csv`: per-frame pose metrics
- `strokes.csv`: detected swing moments and per-stroke summaries
- `annotated_video.mp4`: pose skeleton overlay and metrics overlay
- `report.md`: rule-based coaching report
- `tennis_ai_outputs.zip`: all generated outputs

## Important Streamlit Cloud fix

This version intentionally does **not** include `packages.txt`.

The earlier package requested apt packages such as `ffmpeg`, `libgl1`, and `libglib2.0-0`. On current Streamlit Cloud images, that can produce Debian dependency conflicts before Python packages are installed. This version avoids those apt packages by:

- using a headless OpenCV wheel, so no GUI/OpenGL Linux packages are needed;
- using `imageio-ffmpeg`, so FFmpeg is available from Python instead of apt;
- adding a small local compatibility shim so MediaPipe's `opencv-contrib-python` dependency resolves to `opencv-contrib-python-headless`.

If your GitHub repo already has the old file, delete `packages.txt` from the repo before redeploying.

## Project files

```text
tennis-ai-coach/
├── app.py
├── tennis_analysis.py
├── requirements.txt
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
2. Upload all files in this folder to the repository root.
3. Make sure `packages.txt` is not in the repository.
4. Go to Streamlit Community Cloud and create or redeploy the app.
5. Select the GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Use Python 3.11 in Streamlit advanced settings, then deploy.

## Notes and limitations

- This is a 2D pose-estimation tool from a single camera angle, not a professional biomechanical assessment.
- Phone MOV/HEVC videos may need conversion. Use the app setting "Force conversion to web MP4 first" if OpenCV cannot read the file.
- Large videos take longer and may hit memory or time limits on free hosting. For faster runs, increase "Process every Nth frame" or set a max processed frame count.
- Do not upload sensitive, controlled, private, or regulated videos to a public deployment.
