from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from learned_piece_recognizer import hand_classifier_assets, load_model, recognize_image
from make_visual_review import write_visual_review
from recognize_hand_pieces import digit_templates


DEFAULT_PATTERNS = ("*.png", "*.jpg", "*.jpeg")


def resolve_images(images: list[Path], screenshots_dir: Path, patterns: list[str]) -> list[Path]:
    if images:
        return sorted({path.resolve() for path in images})
    found: set[Path] = set()
    for pattern in patterns or list(DEFAULT_PATTERNS):
        found.update(path.resolve() for path in screenshots_dir.glob(pattern) if path.is_file())
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learned piece recognition for multiple images with one model load.")
    parser.add_argument("images", nargs="*", type=Path, help="Images to recognize. If omitted, --pattern is used.")
    parser.add_argument("--screenshots-dir", type=Path, default=Path("tools/samples/screenshots"))
    parser.add_argument("--pattern", action="append", default=[], help="Glob under --screenshots-dir when no images are passed.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tools/out/piece_inference"))
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--no-visual-review", action="store_true")
    parser.add_argument("--visual-review-low-confidence", type=float, default=0.55)
    args = parser.parse_args()

    image_paths = resolve_images(args.images, args.screenshots_dir, args.pattern)
    if not image_paths:
        raise SystemExit("no images matched")

    args.out.mkdir(parents=True, exist_ok=True)
    model_load_started = time.perf_counter()
    model = load_model(args.model)
    if args.include_hands:
        hand_model, hand_templates = hand_classifier_assets()
        model["hand_classifier"] = {"model": hand_model, "templates": hand_templates}
        digit_templates()
    model_load_seconds = time.perf_counter() - model_load_started

    timing_rows = []
    for image_path in image_paths:
        report_path = args.out / image_path.stem / "piece_report.json"
        started = time.perf_counter()
        report = recognize_image(
            image_path,
            args.model,
            model=model,
            include_hands=args.include_hands,
            out_path=report_path,
        )
        seconds = time.perf_counter() - started
        internal_seconds = float(report["timing"]["processing_time_seconds"])
        row = {
            "sample": image_path.stem,
            "image": str(image_path),
            "seconds": round(seconds, 4),
            "internal_seconds": round(internal_seconds, 4),
            "ok": seconds <= args.max_seconds,
        }
        timing_rows.append(row)
        print(
            f"{image_path.stem}: {seconds:.3f}s "
            f"internal={internal_seconds:.3f}s ok={row['ok']}",
            flush=True,
        )

    summary = {
        "samples": len(timing_rows),
        "model": str(args.model),
        "model_load_seconds": round(model_load_seconds, 4),
        "startup_seconds": round(model_load_seconds, 4),
        "max_seconds": args.max_seconds,
        "slow_samples": sum(1 for row in timing_rows if not row["ok"]),
        "timing": timing_rows,
    }
    (args.out / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out / "timing.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "image", "seconds", "internal_seconds", "ok"])
        writer.writeheader()
        writer.writerows(timing_rows)

    if not args.no_visual_review:
        visual_path = write_visual_review(
            args.out,
            args.labels_dir,
            include_hands=args.include_hands,
            low_confidence=args.visual_review_low_confidence,
        )
        summary["visual_review_html"] = str(visual_path)
        (args.out / "inference_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"visual_review_html={visual_path}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["slow_samples"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
