# Automatic Rat Scratching Detection

Source code for our published article: **"A robust deep learning–based video system for high-accuracy automated detection of scratching behavior in rat itch models"**

## Repository structure

```text
Rat-Scratching-Detection/
├── scratching_detection/
│   ├── detect_new_video.py
│   └── model.py
├── models/
│   └── README.md
├── demo_video/
│   └── README.md
├── results/
│   └── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
conda create -n scratchdl python=3.10 -y
conda activate scratchdl
pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with your NVIDIA driver/CUDA environment.

## Trained model

Place the trained checkpoint here:

```text
models/best.pt
```

The checkpoint is not included in the Git repository because model files can be large. Host it separately using a GitHub Release, Zenodo, Figshare, Google Drive, or a similar service.

## Input video

Place a new video in:

```text
demo_video/new_rat.mp4
```

The model was developed using overhead-view videos recorded at approximately 30 fps.

## Run detection

### Windows PowerShell

```powershell
python scratching_detection\detect_new_video.py --video demo_video\new_rat.mp4 --checkpoint models\best.pt --output-prefix results\new_rat --threshold 0.50 --stride-frames 6 --merge-gap-sec 0.50 --batch-size 2
```

### macOS/Linux

```bash
python scratching_detection/detect_new_video.py \
  --video demo_video/new_rat.mp4 \
  --checkpoint models/best.pt \
  --output-prefix results/new_rat \
  --threshold 0.50 \
  --stride-frames 6 \
  --merge-gap-sec 0.50 \
  --batch-size 2
```

## Output

Two CSV files are generated.

```text
results/new_rat_windows.csv
results/new_rat_events.csv
```

`new_rat_windows.csv` contains the probability for each sliding window.

`new_rat_events.csv` contains merged scratching events, including readable whole-second timing:

## License

See `LICENSE`.
