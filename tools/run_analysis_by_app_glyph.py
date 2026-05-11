from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from learned_piece_recognizer import build_model, load_model, recognize_image, save_model
from make_visual_review import write_visual_review
from recognize_board_pieces import initial_position_labels
from train_piece_model import load_training_samples


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def group_key(image_path: Path, screenshots_dir: Path) -> tuple[str, str]:
    relative = image_path.relative_to(screenshots_dir)
    parts = relative.parts
    app = parts[0] if len(parts) >= 1 else "未分類"
    glyph = parts[1] if len(parts) >= 2 else "未分類"
    return app, glyph


def is_initial_image(image_path: Path) -> bool:
    return "初期配置" in image_path.stem


def initial_rows() -> list[list[str]]:
    labels = initial_position_labels()
    rows: list[list[str]] = []
    for row in range(1, 10):
        values = []
        for col in range(1, 10):
            color_piece = labels.get((row, col))
            values.append(f"{color_piece[0]}:{color_piece[1]}" if color_piece else "empty")
        rows.append(values)
    return rows


def empty_hands() -> dict[str, dict[str, int]]:
    pieces = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
    return {color: {piece: 0 for piece in pieces} for color in ("black", "white")}


def label_lookup(labels_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(labels_dir.rglob("*.json"))}


def prepare_review_labels(
    out_dir: Path,
    labels_dir: Path,
    image_paths: list[Path],
    screenshots_dir: Path,
) -> tuple[Path, dict[str, str]]:
    review_labels = out_dir / "_review_labels"
    if review_labels.exists():
        shutil.rmtree(review_labels)
    review_labels.mkdir(parents=True, exist_ok=True)

    existing = label_lookup(labels_dir)
    label_status: dict[str, str] = {}
    rows = initial_rows()
    for image_path in image_paths:
        app, glyph = group_key(image_path, screenshots_dir)
        destination = review_labels / app / glyph / f"{image_path.stem}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing_label = existing.get(image_path.stem)
        if existing_label is not None:
            shutil.copy2(existing_label, destination)
            label_status[image_path.stem] = "教師ラベルあり"
            continue
        if is_initial_image(image_path):
            data = {
                "schema_version": 2,
                "image": str(image_path.resolve()),
                "orientation": "black_bottom",
                "rows": rows,
                "hands": empty_hands(),
                "label_source": "implicit_initial_position",
            }
            destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            label_status[image_path.stem] = "初期配置ラベル"
            continue
        label_status[image_path.stem] = "教師ラベル未作成"
    return review_labels, label_status


def train_or_load_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_model and args.model.exists():
        return load_model(args.model)
    samples = load_training_samples(args.labels_dir, args.screenshots_dir, None, args.calibration_dir)
    model = build_model(samples, include_hands=args.include_hands)
    save_model(args.model, model)
    return model


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "app",
        "glyph",
        "sample",
        "kind",
        "label_status",
        "image",
        "report",
        "seconds",
        "piece",
        "empty",
        "unknown",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recognition for every screenshot grouped by app and glyph count.")
    parser.add_argument("--screenshots-dir", type=Path, default=Path("tools/samples/screenshots_by_app_glyph"))
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards_by_app_glyph"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("tools/samples/screenshots_by_app_glyph"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/analysis_by_app_glyph"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--reuse-model", action="store_true")
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--low-confidence", type=float, default=0.55)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.model is None:
        args.model = args.out / "models" / "all_labeled_piece_model.pkl"
    args.model.parent.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(args.screenshots_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {args.screenshots_dir}")
    review_labels, label_status = prepare_review_labels(args.out, args.labels_dir, image_paths, args.screenshots_dir)
    model = train_or_load_model(args)

    manifest_rows: list[dict[str, Any]] = []
    groups: set[tuple[str, str]] = set()
    started = time.perf_counter()
    for index, image_path in enumerate(image_paths, start=1):
        app, glyph = group_key(image_path, args.screenshots_dir)
        groups.add((app, glyph))
        kind = "初期配置" if is_initial_image(image_path) else "通常"
        sample_out = args.out / app / glyph / image_path.stem / "piece_report.json"
        report = recognize_image(image_path, args.model, model=model, include_hands=args.include_hands, out_path=sample_out)
        summary = report.get("summary") or {}
        seconds = float((report.get("timing") or {}).get("processing_time_seconds") or 0.0)
        manifest_rows.append(
            {
                "app": app,
                "glyph": glyph,
                "sample": image_path.stem,
                "kind": kind,
                "label_status": label_status.get(image_path.stem, "教師ラベル未作成"),
                "image": str(image_path),
                "report": str(sample_out),
                "seconds": round(seconds, 4),
                "piece": summary.get("piece", ""),
                "empty": summary.get("empty", ""),
                "unknown": summary.get("unknown", ""),
            }
        )
        print(f"[{index}/{len(image_paths)}] {app}/{glyph}/{image_path.stem}: {seconds:.3f}s", flush=True)

    write_manifest(args.out / "manifest.csv", manifest_rows)
    for app, glyph in sorted(groups):
        group_dir = args.out / app / glyph
        try:
            html_path = write_visual_review(
                group_dir,
                review_labels,
                group_dir / "visual_review.html",
                include_hands=args.include_hands,
                low_confidence=args.low_confidence,
            )
            print(f"HTML: {html_path}")
        except FileNotFoundError:
            pass

    elapsed = time.perf_counter() - started
    summary = {
        "images": len(image_paths),
        "groups": len(groups),
        "elapsed_seconds": round(elapsed, 3),
        "model": str(args.model),
        "review_labels": str(review_labels),
        "manifest": str(args.out / "manifest.csv"),
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
