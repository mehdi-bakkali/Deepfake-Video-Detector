"""Fine-tune EfficientNet-B0 on DFDC video frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detector.data import VideoFrameDataset, limit_records, load_records, split_records
from deepfake_detector.model import IMAGE_SIZE, build_model, make_transform, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Root directory containing DFDC shards")
    parser.add_argument("--output", default="artifacts/best_model.pt", help="Checkpoint output path")
    parser.add_argument(
        "--max-videos", type=int, help="Optional class-balanced, source-grouped subset size"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Videos per batch")
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for a simple Mac setup")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop after this many stale epochs")
    parser.add_argument(
        "--face-crop", action=argparse.BooleanOptionalAction, default=True,
        help="Use an OpenCV face crop when detected; otherwise keep the full frame",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[int] = []
    probabilities_all: list[float] = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        batch_size, frame_count = images.shape[:2]
        images = images.flatten(0, 1)
        labels_float = labels.to(device, dtype=torch.float32).view(-1, 1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            frame_logits = model(images).view(batch_size, frame_count)
            video_logits = frame_logits.mean(dim=1, keepdim=True)
            loss = criterion(video_logits, labels_float)
            if training:
                loss.backward()
                optimizer.step()
        probabilities = torch.sigmoid(video_logits.detach()).view(-1).cpu().numpy()
        losses.append(float(loss.detach().cpu()))
        labels_all.extend(labels.numpy().astype(int).tolist())
        probabilities_all.extend(probabilities.tolist())

    auc = roc_auc_score(labels_all, probabilities_all)
    average_precision = average_precision_score(labels_all, probabilities_all)
    balanced_accuracy = balanced_accuracy_score(labels_all, np.asarray(probabilities_all) >= 0.5)
    return float(np.mean(losses)), float(auc), float(average_precision), float(balanced_accuracy)


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.frames_per_video < 1
        or args.num_workers < 0
        or args.patience < 1
    ):
        raise SystemExit("epochs, batch-size, frames-per-video, and patience must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    records = load_records(args.data_dir)
    records = limit_records(records, args.max_videos, seed=args.seed)
    train_records, validation_records = split_records(
        records, validation_fraction=args.validation_fraction, seed=args.seed
    )
    print(
        f"Loaded {len(records)} videos: {len(train_records)} train, "
        f"{len(validation_records)} validation"
    )

    train_dataset = VideoFrameDataset(
        train_records,
        make_transform(training=True),
        training=True,
        frames_per_video=args.frames_per_video,
        face_crop=args.face_crop,
    )
    validation_dataset = VideoFrameDataset(
        validation_records,
        make_transform(training=False),
        training=False,
        frames_per_video=args.frames_per_video,
        face_crop=args.face_crop,
    )
    device = select_device()
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(pretrained=True).to(device)
    positive_count = sum(record.label == 1 for record in train_records)
    negative_count = len(train_records) - positive_count
    positive_weight = torch.tensor([negative_count / max(positive_count, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    stale_epochs = 0

    print(f"Training on {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, train_ap, train_balanced_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_auc, val_ap, val_balanced_accuracy = run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f}, ROC-AUC {train_auc:.4f}, AP {train_ap:.4f}, "
            f"balanced acc {train_balanced_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, ROC-AUC {val_auc:.4f}, AP {val_ap:.4f}, "
            f"balanced acc {val_balanced_accuracy:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_name": "efficientnet_b0",
                    "image_size": IMAGE_SIZE,
                    "state_dict": model.state_dict(),
                    "class_mapping": {"real": 0, "fake": 1},
                    "frames_per_video": args.frames_per_video,
                    "face_crop": args.face_crop,
                    "validation_auc": val_auc,
                    "validation_average_precision": val_ap,
                    "validation_balanced_accuracy": val_balanced_accuracy,
                },
                output_path,
            )
            print(f"Saved best checkpoint to {output_path}")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without ROC-AUC improvement")
                break


if __name__ == "__main__":
    main()
