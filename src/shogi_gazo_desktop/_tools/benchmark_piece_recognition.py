from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from detect_board_grid import detect_grid
from evaluate_piece_recognition import evaluate_one
from learned_piece_recognizer import build_model, load_model, recognize_image, save_model
from position_label_utils import find_label_path, inventory_errors, load_position_label, resolve_label_image_path
from recognize_board_pieces import initial_position_labels


TIMING_FIELDS = [
    "sample",
    "seconds",
    "internal_seconds",
    "ok",
    "resumed",
    "reused_model",
    "build_seconds",
    "save_model_seconds",
    "load_model_seconds",
    "inference_seconds",
    "evaluation_seconds",
    "fold_wall_seconds",
]


def label_paths(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.rglob("*.json"))


def load_samples(labels_dir: Path, screenshots_dir: Path) -> dict[str, tuple[Path, Image.Image, object, dict]]:
    samples = {}
    for label_path in label_paths(labels_dir):
        label = load_position_label(label_path, require_hands=False)
        if label.get("exclude_from_benchmark"):
            continue
        image_path = resolve_label_image_path(label_path, label, screenshots_dir)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        detection = detect_grid(image)
        if detection is None:
            continue
        samples[label_path.stem] = (image_path, image, detection, label)
    return samples


def build_fold_model(samples: dict, calibration_samples: list[tuple], excluded: str, include_hands: bool):
    training = [
        (source, image, detection, label["cells"])
        for source, (_, image, detection, label) in samples.items()
        if source != excluded
    ]
    training.extend(calibration_samples)
    return build_model(training, excluded_source=excluded, include_hands=include_hands)


def normalize_sample_name(value: str) -> str:
    text = value.strip().strip('"')
    if not text:
        return ""
    return Path(text).stem


def selected_sample_names(samples: dict, requested: list[str], limit: int | None) -> list[str]:
    if requested:
        seen = set()
        selected = []
        for value in requested:
            sample = normalize_sample_name(value)
            if sample and sample in samples and sample not in seen:
                selected.append(sample)
                seen.add(sample)
    else:
        selected = sorted(samples)
    if limit:
        selected = selected[:limit]
    return selected


def load_samples_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_labels_for_benchmark(labels_dir: Path, include_hands: bool) -> list[dict]:
    failures = []
    for label_path in label_paths(labels_dir):
        try:
            label = load_position_label(label_path, require_hands=include_hands)
            if label.get("exclude_from_benchmark"):
                continue
            if include_hands:
                errors = inventory_errors(label["cells"], label["hands"])
                if errors:
                    failures.append({"sample": label_path.stem, "reason": "inventory_mismatch", "errors": errors})
        except Exception as exc:
            failures.append({"sample": label_path.stem, "reason": f"{type(exc).__name__}: {exc}", "errors": []})
    return failures


def load_calibration_samples(calibration_dir: Path) -> list[tuple[str, Image.Image, object, list[dict]]]:
    if not calibration_dir.exists():
        return []
    samples = []
    cells = initial_position_cells()
    for image_path in calibration_image_paths(calibration_dir):
        image = Image.open(image_path).convert("RGB")
        detection = detect_grid(image)
        if detection is None:
            continue
        samples.append((f"initial:{image_path.stem}", image, detection, cells))
    return samples


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
    parser = argparse.ArgumentParser(description="Run non-leaking leave-one-image-out piece recognition benchmark.")
    parser.add_argument("--mode", choices=("leave-one-image-out",), default="leave-one-image-out")
    parser.add_argument("--screenshots", type=Path, default=Path("tools/samples/screenshots"))
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("tools/samples/screenshots/初期配置"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/benchmark_piece_recognition"))
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--target-accuracy", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Optional sample limit for smoke tests.")
    parser.add_argument("--sample", action="append", default=[], help="Evaluate one sample name. Can be repeated.")
    parser.add_argument("--samples-file", type=Path, help="Text file containing sample names or image/label paths.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing fold reports in the output directory.")
    parser.add_argument("--reuse-models", action="store_true", help="Reuse existing fold model files when reports are missing.")
    parser.add_argument(
        "--quarantine-invalid-labels",
        action="store_true",
        help="Exclude labels with structural inventory failures from the scored benchmark and report them separately.",
    )
    parser.add_argument("--no-visual-review", action="store_true", help="Skip writing visual_review.html.")
    parser.add_argument("--visual-review-low-confidence", type=float, default=0.55)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    label_failures = validate_labels_for_benchmark(args.labels_dir, args.include_hands)
    quarantined_samples = {failure["sample"] for failure in label_failures} if args.quarantine_invalid_labels else set()
    samples = load_samples(args.labels_dir, args.screenshots)
    calibration_samples = load_calibration_samples(args.calibration_dir)
    requested_samples = [*load_samples_file(args.samples_file), *args.sample]
    selected_names = [
        sample
        for sample in selected_sample_names(samples, requested_samples, args.limit)
        if sample not in quarantined_samples
    ]

    results = []
    timing_rows = []
    benchmark_started = time.perf_counter()
    for sample in selected_names:
        image_path, _, _, _ = samples[sample]
        report_path = args.out / sample / "piece_report.json"
        model_path = args.out / "models" / f"{sample}.pkl"
        fold_started = time.perf_counter()
        build_seconds = 0.0
        save_model_seconds = 0.0
        load_model_seconds = 0.0
        inference_seconds = 0.0
        evaluation_seconds = 0.0
        reused_model = False
        resumed = bool(args.resume and report_path.exists())
        if resumed:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            internal_seconds = float(report.get("timing", {}).get("processing_time_seconds") or 0.0)
            seconds = internal_seconds
        else:
            if args.reuse_models and model_path.exists():
                load_started = time.perf_counter()
                model = load_model(model_path)
                load_model_seconds = time.perf_counter() - load_started
                reused_model = True
            else:
                build_started = time.perf_counter()
                model = build_fold_model(samples, calibration_samples, sample, args.include_hands)
                build_seconds = time.perf_counter() - build_started
                save_started = time.perf_counter()
                save_model(model_path, model)
                save_model_seconds = time.perf_counter() - save_started
            inference_started = time.perf_counter()
            report = recognize_image(image_path, model_path, model=model, include_hands=args.include_hands, out_path=report_path)
            inference_seconds = time.perf_counter() - inference_started
            seconds = inference_seconds
            internal_seconds = float(report["timing"]["processing_time_seconds"])
        evaluation_started = time.perf_counter()
        evaluation = evaluate_one(
            report_path,
            find_label_path(args.labels_dir, sample),
            high_confidence_threshold=0.75,
            include_hands=args.include_hands,
            strict_leak_guard=True,
            forbidden_sources=(sample,),
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        timing_rows.append(
            {
                "sample": sample,
                "seconds": round(seconds, 4),
                "internal_seconds": round(internal_seconds, 4),
                "ok": seconds <= args.max_seconds,
                "resumed": resumed,
                "reused_model": reused_model,
                "build_seconds": round(build_seconds, 4),
                "save_model_seconds": round(save_model_seconds, 4),
                "load_model_seconds": round(load_model_seconds, 4),
                "inference_seconds": round(inference_seconds, 4),
                "evaluation_seconds": round(evaluation_seconds, 4),
                "fold_wall_seconds": round(time.perf_counter() - fold_started, 4),
            }
        )
        results.append(evaluation)
        print(
            f"{sample}: {seconds:.3f}s "
            f"errors={evaluation['metrics']['errors']} hand_errors={evaluation['metrics'].get('hand_errors', 0)} "
            f"leak_errors={evaluation['metrics'].get('leak_errors', 0)}"
            f"{' resumed' if resumed else ''}",
            f"{'reused_model' if reused_model else ''}",
            flush=True,
        )
        write_progress_report(
            args.out,
            args,
            results,
            timing_rows,
            label_failures,
            quarantined_samples,
            calibration_samples,
            selected_names,
            benchmark_started,
        )

    scored_label_failures = [] if args.quarantine_invalid_labels else label_failures
    summary = aggregate_benchmark(results, timing_rows, scored_label_failures, args.target_accuracy, args.max_seconds)
    summary["quarantined_samples"] = len(quarantined_samples)
    summary["data_quality_passed"] = not label_failures
    summary["limited"] = bool(args.limit)
    report = {
        "mode": args.mode,
        "screenshots": str(args.screenshots),
        "labels_dir": str(args.labels_dir),
        "calibration_dir": str(args.calibration_dir),
        "calibration_samples": len(calibration_samples),
        "include_hands": args.include_hands,
        "max_seconds": args.max_seconds,
        "target_accuracy": args.target_accuracy,
        "elapsed_seconds": round(time.perf_counter() - benchmark_started, 3),
        "summary": summary,
        "label_failures": label_failures,
        "quarantine_invalid_labels": args.quarantine_invalid_labels,
        "quarantined_samples": sorted(quarantined_samples),
        "timing": timing_rows,
        "results": results,
    }
    (args.out / "benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "timing.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(timing_rows)
    if not args.no_visual_review:
        from make_visual_review import write_visual_review

        visual_review_path = write_visual_review(
            args.out,
            args.labels_dir,
            include_hands=args.include_hands,
            low_confidence=args.visual_review_low_confidence,
        )
        report["visual_review_html"] = str(visual_review_path)
        (args.out / "benchmark_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"visual_review_html={visual_review_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


def write_progress_report(
    out_dir: Path,
    args: argparse.Namespace,
    results: list[dict],
    timing_rows: list[dict],
    label_failures: list[dict],
    quarantined_samples: set[str],
    calibration_samples: list[tuple],
    selected_names: list[str],
    benchmark_started: float,
) -> None:
    scored_label_failures = [] if args.quarantine_invalid_labels else label_failures
    summary = aggregate_benchmark(results, timing_rows, scored_label_failures, args.target_accuracy, args.max_seconds)
    summary["quarantined_samples"] = len(quarantined_samples)
    summary["data_quality_passed"] = not label_failures
    summary["limited"] = bool(args.limit)
    report = {
        "mode": args.mode,
        "screenshots": str(args.screenshots),
        "labels_dir": str(args.labels_dir),
        "calibration_dir": str(args.calibration_dir),
        "calibration_samples": len(calibration_samples),
        "include_hands": args.include_hands,
        "max_seconds": args.max_seconds,
        "target_accuracy": args.target_accuracy,
        "elapsed_seconds": round(time.perf_counter() - benchmark_started, 3),
        "completed_samples": len(results),
        "selected_samples": len(selected_names),
        "summary": summary,
        "label_failures": label_failures,
        "quarantine_invalid_labels": args.quarantine_invalid_labels,
        "quarantined_samples": sorted(quarantined_samples),
        "timing": timing_rows,
        "results": results,
        "partial": len(results) < len(selected_names),
    }
    (out_dir / "benchmark_progress.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "timing_progress.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(timing_rows)


def aggregate_benchmark(results: list[dict], timing_rows: list[dict], label_failures: list[dict], target_accuracy: float, max_seconds: float) -> dict:
    if not results:
        return {
            "passed": False,
            "samples": 0,
            "accuracy": 0.0,
            "errors": 0,
            "hand_errors": 0,
            "leak_errors": 0,
            "high_confidence_errors": 0,
            "unknown_on_piece": 0,
            "slow_samples": 0,
            "label_failures": len(label_failures),
            "reason": "no_samples_evaluated",
        }
    true_piece = sum(result["metrics"]["true_piece"] for result in results)
    errors = sum(result["metrics"]["errors"] for result in results)
    hand_errors = sum(result["metrics"].get("hand_errors", 0) for result in results)
    leak_errors = sum(result["metrics"].get("leak_errors", 0) for result in results)
    high_confidence_errors = sum(result["metrics"]["high_confidence_errors"] for result in results)
    unknown_on_piece = sum(result["metrics"]["unknown_on_piece"] for result in results)
    accuracy = 1.0 if true_piece and errors == 0 else (1.0 - errors / max(1, true_piece))
    slow = [row for row in timing_rows if row["seconds"] > max_seconds]
    passed = (
        not label_failures
        and not slow
        and errors == 0
        and hand_errors == 0
        and leak_errors == 0
        and high_confidence_errors == 0
        and unknown_on_piece == 0
        and accuracy >= target_accuracy
    )
    return {
        "passed": passed,
        "samples": len(results),
        "accuracy": round(accuracy, 4),
        "errors": errors,
        "hand_errors": hand_errors,
        "leak_errors": leak_errors,
        "high_confidence_errors": high_confidence_errors,
        "unknown_on_piece": unknown_on_piece,
        "slow_samples": len(slow),
        "label_failures": len(label_failures),
    }


if __name__ == "__main__":
    main()
