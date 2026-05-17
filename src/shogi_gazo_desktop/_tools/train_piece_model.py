from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from detect_board_grid import detect_grid
from recognize_board_pieces import initial_position_labels
from learned_piece_recognizer import build_model, save_model
from position_label_utils import load_position_label, resolve_label_image_path


def load_training_samples(
    labels_dir: Path,
    screenshots_dir: Path,
    exclude_sample: str | None,
    calibration_dir: Path | None,
) -> list[tuple[str, Image.Image, object, list[dict]]]:
    samples = []
    for label_path in sorted(labels_dir.rglob("*.json"), key=training_label_sort_key):
        source = label_path.stem
        if source == exclude_sample:
            continue
        label = load_position_label(label_path)
        if label.get("exclude_from_benchmark"):
            continue
        image_path = resolve_label_image_path(label_path, label, screenshots_dir)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        detection = detect_grid(image)
        if detection is None:
            continue
        samples.append((source, image, detection, label["cells"]))
    if calibration_dir is not None and calibration_dir.exists():
        for image_path in calibration_image_paths(calibration_dir):
            source = f"initial:{image_path.stem}"
            if exclude_sample is not None and image_path.stem == exclude_sample:
                continue
            image = Image.open(image_path).convert("RGB")
            detection = detect_grid(image)
            if detection is None:
                continue
            samples.append((source, image, detection, initial_position_cells()))
    return samples


def training_label_sort_key(label_path: Path) -> tuple[int, str]:
    try:
        label = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (9, str(label_path))
    metadata = label.get("metadata") if isinstance(label.get("metadata"), dict) else {}
    label_source = str(label.get("label_source") or "")
    if label_source == "user_corrected_from_augmented_recognition" or metadata.get("user_feedback_date"):
        priority = 0
    elif metadata.get("reviewed_at"):
        priority = 1
    else:
        priority = 2
    return (priority, str(label_path))


def calibration_image_paths(calibration_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    if calibration_dir.is_file():
        return [calibration_dir]
    image_paths = sorted(path for path in calibration_dir.rglob("*") if path.suffix.lower() in suffixes)
    initial_paths = [
        path
        for path in image_paths
        if "初期配置" in path.stem or any(parent.name == "初期配置" for parent in path.parents)
    ]
    return initial_paths or image_paths


def initial_position_cells() -> list[dict]:
    return [
        {
            "row": row,
            "col": col,
            "state": "piece",
            "color": color,
            "piece": piece,
        }
        for (row, col), (color, piece) in sorted(initial_position_labels().items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a non-leaking learned board-piece model from teacher labels.")
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--screenshots-dir", type=Path, default=Path("tools/samples/screenshots"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("tools/samples/screenshots/初期配置"))
    parser.add_argument("--exclude-sample", help="Sample stem to leave out for holdout evaluation.")
    parser.add_argument("--include-hands", action="store_true", help="Bundle hand-piece classifier assets into the model.")
    parser.add_argument("--out", type=Path, default=Path("tools/out/models/piece_model.pkl"))
    args = parser.parse_args()

    samples = load_training_samples(args.labels_dir, args.screenshots_dir, args.exclude_sample, args.calibration_dir)
    model = build_model(samples, excluded_source=args.exclude_sample, include_hands=args.include_hands)
    save_model(args.out, model)
    print(
        f"OK: trained {args.out} "
        f"samples={len(samples)} templates={len(model['templates'])} excluded={args.exclude_sample or '-'}",
    )


if __name__ == "__main__":
    main()
