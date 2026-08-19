# Automatic Rat Scratching Detection

Inference code for automatic detection of rat scratching events from video using a trained R3D-18 network with temporal self-attention.

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

## Preprocessing and inference

Default settings are:

- 48 consecutive frames per input clip
- expected video frame rate: 30 fps
- sliding-window stride: 6 frames (0.2 s at 30 fps)
- resize to 640 x 360 using OpenCV area interpolation (`cv2.INTER_AREA`)
- aspect-ratio-preserving letterbox to 224 x 224 pixels
- scratching probability threshold: 0.50
- event merge gap: 0.50 s

For final evaluation, threshold and event-merging settings should be determined on the validation set and locked before analysis of an independent test set.

## Output

Two CSV files are generated.

```text
results/new_rat_windows.csv
results/new_rat_events.csv
```

`new_rat_windows.csv` contains the probability for each sliding window.

`new_rat_events.csv` contains merged scratching events, including readable whole-second timing:

```text
event,start_sec,end_sec,start_time,end_time,duration_sec,max_probability
1,620.4,622.2,10 min 20 sec,10 min 22 sec,1.8,0.9814
```

The continuous second values are retained for quantitative analysis, while `start_time` and `end_time` are rounded to whole seconds for readability.

## Model architecture

The detector uses a torchvision R3D-18 backbone with two temporal self-attention modules inserted after later R3D stages, followed by adaptive global pooling and binary classification.

## Citation

If you use this code, please cite the associated manuscript:

```text
[Add the final manuscript citation here]
```

## License

See `LICENSE`.
