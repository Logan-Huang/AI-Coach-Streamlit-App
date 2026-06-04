# Tennis AI Coach Streamlit App

This repository turns the original Colab notebook into a deployable Streamlit app.

The app lets a user upload a tennis video, runs MediaPipe pose tracking, extracts per-frame metrics, detects likely swing moments from wrist-speed peaks, and provides downloads for:

- `metrics.csv`: per-frame pose metrics
- `strokes.csv`: detected swing moments and per-stroke summaries
- `annotated_video.mp4`: pose skeleton overlay and metrics overlay
- `report.md`: rule-based coaching report
- `tennis_ai_outputs.zip`: all generated outputs

## Project files

```text
tennis-ai-coach/
├── app.py
├── tennis_analysis.py
├── requirements.txt
├── packages.txt
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

1. Create a new GitHub repository.
2. Upload all files in this folder to the repository root.
3. Go to Streamlit Community Cloud and create a new app.
4. Select the GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

If the app fails while installing MediaPipe, open the app's advanced settings and select Python 3.11, then redeploy.

## Notes and limitations

- This is a 2D pose-estimation tool from a single camera angle, not a professional biomechanical assessment.
- Phone MOV/HEVC videos may need conversion. Use the app setting "Force conversion to web MP4 first" if OpenCV cannot read the file.
- Large videos take longer and may hit memory or time limits on free hosting. For faster runs, increase "Process every Nth frame" or set a max processed frame count.
- Do not upload sensitive, controlled, private, or regulated videos to a public deployment.
