"""Score videos with a fine-tuned deepfake frame classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from deepfake_detector.data import load_sampled_frames
from deepfake_detector.model import build_model, make_transform, select_device


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "efficientnet_b0":
        raise ValueError("Unsupported checkpoint model; expected efficientnet_b0")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.face_crop = bool(checkpoint.get("face_crop", False))
    model.to(device).eval()
    return model


@torch.inference_mode()
def score_video(
    model: torch.nn.Module,
    video_path: str | Path,
    device: torch.device,
    max_frames: int = 16,
) -> dict[str, str | float | int]:
    frames = load_sampled_frames(
        video_path,
        count=max_frames,
        training=False,
        face_crop=bool(getattr(model, "face_crop", False)),
    )
    transform = make_transform(training=False)
    batch = torch.stack([transform(frame) for frame in frames]).to(device)
    frame_logits = model(batch).view(-1)
    frame_probabilities = torch.sigmoid(frame_logits).cpu().numpy()
    video_logit = frame_logits.mean()
    fake_probability = float(torch.sigmoid(video_logit).cpu())
    return {
        "video": Path(video_path).name,
        "prediction": "potentially_manipulated" if fake_probability >= 0.5 else "likely_real",
        "fake_probability": fake_probability,
        "frame_count": len(frames),
        "frame_score_std": float(frame_probabilities.std()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Path to one video")
    source.add_argument("--video-dir", help="Directory of MP4 files to score")
    parser.add_argument("--frames", type=int, default=16, help="Frames sampled from each video")
    parser.add_argument("--output-csv", help="Optional CSV output path (directory mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    model = load_checkpoint(args.checkpoint, device)
    if args.video:
        results = [score_video(model, args.video, device, args.frames)]
        print(json.dumps(results[0], indent=2))
        return

    video_dir = Path(args.video_dir).expanduser()
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 files found in {video_dir}")
    results = []
    for path in videos:
        try:
            results.append(score_video(model, path, device, args.frames))
        except RuntimeError as error:
            print(f"Skipping {path.name}: {error}")
    if args.output_csv:
        if not results:
            raise SystemExit("No videos could be scored")
        output_path = Path(args.output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} predictions to {output_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
