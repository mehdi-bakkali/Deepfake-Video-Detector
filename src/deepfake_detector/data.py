"""DFDC metadata loading and on-demand video frame dataset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

_FACE_CASCADE: cv2.CascadeClassifier | None = None


@dataclass(frozen=True)
class VideoRecord:
    path: Path
    label: int  # 0 = REAL, 1 = FAKE
    name: str
    original: str | None = None


def _normalize_label(value: Any) -> int:
    label = str(value).strip().upper()
    if label in {"FAKE", "1", "TRUE"}:
        return 1
    if label in {"REAL", "0", "FALSE"}:
        return 0
    raise ValueError(f"Unrecognized video label: {value!r}")


def _video_lookup(root: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    for path in root.rglob("*.mp4"):
        lookup.setdefault(path.name, []).append(path)
    return lookup


def _resolve_video(
    root: Path, metadata_dir: Path, name: str, lookup: dict[str, list[Path]]
) -> Path | None:
    candidate = metadata_dir / name
    if candidate.is_file():
        return candidate
    candidate = root / name
    if candidate.is_file():
        return candidate
    matches = lookup.get(Path(name).name, [])
    if len(matches) == 1:
        return matches[0]
    return None


def load_records(data_dir: str | Path) -> list[VideoRecord]:
    """Load labeled records from DFDC metadata or DFD label-named folders."""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    metadata_files = sorted(root.rglob("metadata.json"))
    if not metadata_files:
        return _load_folder_labeled_records(root)
    lookup = _video_lookup(root)
    records: list[VideoRecord] = []
    missing: list[str] = []

    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a filename-to-record object in {metadata_path}")
        for name, details in metadata.items():
            if not isinstance(details, dict) or "label" not in details:
                continue
            video_path = _resolve_video(root, metadata_path.parent, name, lookup)
            if video_path is None:
                missing.append(f"{metadata_path.parent / name}")
                continue
            original = details.get("original")
            records.append(
                VideoRecord(
                    path=video_path,
                    label=_normalize_label(details["label"]),
                    name=Path(name).name,
                    original=Path(str(original)).name if original else None,
                )
            )

    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Could not find {len(missing)} video file(s) referenced in metadata. Examples: {preview}"
        )
    if not records:
        return _load_folder_labeled_records(root)
    return records


def _label_from_folder(video_path: Path, root: Path) -> int | None:
    """Infer a class from an ancestor directory such as DFD_original_sequences."""
    for component in reversed(video_path.relative_to(root).parts[:-1]):
        name = "".join(character for character in component.lower() if character.isalnum())
        if any(token in name for token in ("manipulated", "deepfake", "fake", "synthesis")):
            return 1
        if any(token in name for token in ("original", "authentic", "real")):
            return 0
    return None


def _load_folder_labeled_records(root: Path) -> list[VideoRecord]:
    videos = sorted(root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(
            f"No labeled metadata.json or MP4 files found under {root}"
        )

    records: list[VideoRecord] = []
    unlabeled: list[str] = []
    for video_path in videos:
        label = _label_from_folder(video_path, root)
        if label is None:
            unlabeled.append(video_path.relative_to(root).as_posix())
            continue
        records.append(
            VideoRecord(
                path=video_path,
                label=label,
                name=video_path.relative_to(root).as_posix(),
            )
        )

    if not records:
        examples = ", ".join(unlabeled[:3])
        raise ValueError(
            "Could not infer labels from folder names. Put MP4 files under folders named "
            f"original/real and manipulated/fake. Unrecognized examples: {examples}"
        )
    if {record.label for record in records} != {0, 1}:
        raise ValueError(
            "Folder-labeled data must include both real/original and fake/manipulated videos"
        )
    if unlabeled:
        print(f"Ignoring {len(unlabeled)} MP4 file(s) outside recognized label folders")
    return records


def split_records(
    records: list[VideoRecord], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Create a grouped split so a fake and its source do not cross splits."""
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    source_to_index: dict[str, int] = {}
    for index, record in enumerate(records):
        source_key = record.original or record.name
        if source_key in source_to_index:
            union(index, source_to_index[source_key])
        else:
            source_to_index[source_key] = index

    groups = [find(index) for index in range(len(records))]
    group_count = len(set(groups))
    if group_count < 2:
        raise ValueError("At least two independent video groups are needed for a split")

    # Use the closest fold count to the requested validation fraction.
    folds = max(2, min(10, round(1 / validation_fraction), group_count))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [record.label for record in records]
    try:
        train_indices, validation_indices = next(
            splitter.split(range(len(records)), labels, groups)
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified grouped split. Include enough real and fake "
            "source groups in the dataset."
        ) from error

    train = [records[int(index)] for index in train_indices]
    validation = [records[int(index)] for index in validation_indices]
    if {record.label for record in train} != {0, 1} or {
        record.label for record in validation
    } != {0, 1}:
        raise ValueError("The train and validation splits must each include real and fake videos")
    return train, validation


def limit_records(records: list[VideoRecord], maximum: int | None, seed: int = 42) -> list[VideoRecord]:
    """Select a class-balanced, source-grouped subset for quick experiments."""
    if maximum is None or maximum >= len(records):
        return records
    if maximum < 2:
        raise ValueError("max_videos must be at least 2")

    grouped: dict[str, list[VideoRecord]] = {}
    for record in records:
        grouped.setdefault(record.original or record.name, []).append(record)
    keys = list(grouped)
    random.Random(seed).shuffle(keys)
    target_by_class = [maximum / 2, maximum / 2]
    selected: list[VideoRecord] = []
    selected_by_class = [0, 0]
    selected_keys: set[str] = set()

    while len(selected) < maximum:
        candidates = [
            key for key in keys
            if key not in selected_keys and len(grouped[key]) <= maximum - len(selected)
        ]
        if not candidates:
            break

        def subset_score(key: str) -> float:
            group = grouped[key]
            after = [
                selected_by_class[label] + sum(record.label == label for record in group)
                for label in (0, 1)
            ]
            class_error = sum(abs(after[label] - target_by_class[label]) for label in (0, 1))
            remaining = maximum - len(selected) - len(group)
            return class_error + 0.1 * remaining

        chosen = min(candidates, key=subset_score)
        group = grouped[chosen]
        selected.extend(group)
        selected_keys.add(chosen)
        for record in group:
            selected_by_class[record.label] += 1

    if {record.label for record in selected} != {0, 1}:
        raise ValueError("The selected sample needs both REAL and FAKE videos; increase --max-videos")
    return selected


def _face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(cascade_path))
    return _FACE_CASCADE


def _focus_face(frame: Any) -> Any:
    """Return an expanded square crop around the largest detected face, if any."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cascade = _face_cascade()
    if cascade.empty():
        return frame
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    if len(faces) == 0:
        return frame

    x, y, face_width, face_height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
    inv_scale = 1.0 / scale
    x, y = int(x * inv_scale), int(y * inv_scale)
    face_width, face_height = int(face_width * inv_scale), int(face_height * inv_scale)
    side = int(max(face_width, face_height) * 1.55)
    center_x = x + face_width // 2
    center_y = y + face_height // 2 + int(face_height * 0.06)
    left = max(0, center_x - side // 2)
    top = max(0, center_y - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    return frame[top:bottom, left:right] if right > left and bottom > top else frame


def load_sampled_frames(
    video_path: str | Path,
    count: int,
    training: bool = False,
    face_crop: bool = True,
) -> list[Image.Image]:
    """Read a fixed number of random or uniformly spaced frames from a video."""
    if count < 1:
        raise ValueError("Frame count must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    if training:
        indices = sorted(random.choices(range(frame_count), k=count))
    elif count == 1:
        indices = [frame_count // 2]
    else:
        indices = [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]

    decoded: dict[int, Any] = {}
    for frame_index in sorted(set(indices)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if success and frame is not None:
            decoded[frame_index] = _focus_face(frame) if face_crop else frame
    capture.release()
    if not decoded:
        raise RuntimeError(f"Could not decode frames from: {video_path}")

    decoded_indices = sorted(decoded)
    frames: list[Image.Image] = []
    for frame_index in indices:
        actual_index = min(decoded_indices, key=lambda candidate: abs(candidate - frame_index))
        rgb = cv2.cvtColor(decoded[actual_index], cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    return frames


class VideoFrameDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load multiple frames per video, random for training and evenly spaced for validation."""

    def __init__(
        self,
        records: list[VideoRecord],
        transform: Callable[[Image.Image], Any],
        training: bool,
        frames_per_video: int = 4,
        face_crop: bool = True,
    ) -> None:
        self.records = records
        self.transform = transform
        self.training = training
        self.frames_per_video = frames_per_video
        self.face_crop = face_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        record = self.records[index]
        frames = load_sampled_frames(
            record.path,
            count=self.frames_per_video,
            training=self.training,
            face_crop=self.face_crop,
        )
        return torch.stack([self.transform(frame) for frame in frames]), record.label
