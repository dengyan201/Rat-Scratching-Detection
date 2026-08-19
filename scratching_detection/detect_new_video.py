import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from model import ScratchR3DAttention

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def letterbox_square(frame_rgb, size=224, fill=0):
    h, w = frame_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def preprocess_frame(frame_bgr, size=224):
    frame = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return letterbox_square(frame, size=size)

def clip_to_tensor(clip_uint8):
    x = clip_uint8.astype(np.float32) / 255.0
    x = (x - KINETICS_MEAN) / KINETICS_STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).contiguous()

def merge_positive_windows(starts, probs, clip_sec, threshold=0.5, merge_gap_sec=0.5):
    positive = [[float(s), float(s + clip_sec), float(p)]
                for s, p in zip(starts, probs) if p >= threshold]
    if not positive:
        return []
    merged = [positive[0]]
    for s, e, p in positive[1:]:
        if s <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], p)
        else:
            merged.append([s, e, p])
    return merged

def run_batch(model, batch_tensors, batch_starts, device, all_starts, all_probs):
    if not batch_tensors:
        return [], []
    x = torch.stack(batch_tensors, 0).to(device, non_blocking=True)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    all_starts.extend(batch_starts)
    all_probs.extend(probs.tolist())
    return [], []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--clip-frames", type=int, default=48)
    ap.add_argument("--stride-frames", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--merge-gap-sec", type=float, default=0.50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--network-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ScratchR3DAttention(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Loaded checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height}, {fps:.3f} fps, {total_frames} frames")
    if abs(fps - 30.0) > 0.5:
        print("WARNING: model was developed using 30-fps videos.")

    frame_buffer = deque(maxlen=args.clip_frames)
    batch_tensors, batch_starts = [], []
    all_starts, all_probs = [], []
    frame_index = 0

    with tqdm(total=total_frames, desc="Scanning whole video", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_buffer.append(preprocess_frame(frame, args.network_size))
            if len(frame_buffer) == args.clip_frames:
                start_frame = frame_index - args.clip_frames + 1
                if start_frame % args.stride_frames == 0:
                    clip = np.stack(frame_buffer, axis=0)
                    batch_tensors.append(clip_to_tensor(clip))
                    batch_starts.append(start_frame / fps)
                    if len(batch_tensors) >= args.batch_size:
                        batch_tensors, batch_starts = run_batch(
                            model, batch_tensors, batch_starts, device,
                            all_starts, all_probs
                        )
            frame_index += 1
            pbar.update(1)
    cap.release()

    batch_tensors, batch_starts = run_batch(
        model, batch_tensors, batch_starts, device, all_starts, all_probs
    )

    clip_sec = args.clip_frames / fps
    windows_df = pd.DataFrame({
        "start_sec": all_starts,
        "end_sec": [s + clip_sec for s in all_starts],
        "scratch_probability": all_probs,
    })
    windows_df["predicted_positive"] = windows_df["scratch_probability"] >= args.threshold

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    windows_path = Path(str(prefix) + "_windows.csv")
    windows_df.to_csv(windows_path, index=False)

    events = merge_positive_windows(
        all_starts, all_probs, clip_sec,
        threshold=args.threshold,
        merge_gap_sec=args.merge_gap_sec
    )
    def format_min_sec(seconds):
        total_seconds = int(round(seconds))
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes} min {secs} sec"

    events_df = pd.DataFrame(
        [{
            "event": i + 1,
            "start_sec": s,
            "end_sec": e,
            "start_time": format_min_sec(s),
            "end_time": format_min_sec(e),
            "duration_sec": e - s,
            "max_probability": p
        }
         for i, (s, e, p) in enumerate(events)]
    )
    events_path = Path(str(prefix) + "_events.csv")
    events_df.to_csv(events_path, index=False)

    print("\nDetection completed")
    print("Windows analyzed:", len(windows_df))
    print("Positive windows:", int(windows_df["predicted_positive"].sum()))
    print("Detected scratching events:", len(events_df))
    print("Windows file:", windows_path)
    print("Events file:", events_path)

    if len(events_df) > 0:
        print("\nDetected scratching events:")
        print(
            events_df[
                ["event", "start_time", "end_time", "duration_sec", "max_probability"]
            ].to_string(index=False)
        )

if __name__ == "__main__":
    main()
