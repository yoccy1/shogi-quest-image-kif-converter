from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import ExportError, export_json, export_kif, export_sfen
from .models import RecognitionOptions
from .paths import DEFAULT_LABELS_DIR, DEFAULT_MODEL_PATH, DEFAULT_OUTPUTS_DIR, DEFAULT_SCREENSHOTS_DIR, ensure_tools_on_path
from .recognition import load_result, recognize_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shogi-gazo", description="Recognize shogi positions from screenshots.")
    sub = parser.add_subparsers(dest="command", required=True)

    recognize = sub.add_parser("recognize", help="Recognize one screenshot.")
    recognize.add_argument("image", type=Path)
    add_common_recognition_args(recognize)

    batch = sub.add_parser("batch", help="Recognize a directory of screenshots.")
    batch.add_argument("input", type=Path, nargs="?", default=DEFAULT_SCREENSHOTS_DIR)
    add_common_recognition_args(batch)
    batch.add_argument(
        "--no-leak",
        action="store_true",
        help="Train/reuse a per-sample model that excludes the target sample. Slow but suitable for holdout evaluation.",
    )
    batch.add_argument("--sample", action="append", default=[], help="Only process samples whose stem matches this value. Can be used more than once.")
    batch.add_argument("--limit", type=int, help="Process at most this many images after filtering.")

    export = sub.add_parser("export", help="Export a recognition JSON as KIF, SFEN, or JSON.")
    export.add_argument("result", type=Path)
    export.add_argument("--format", choices=("kif", "sfen", "json"), default="kif")
    export.add_argument("--side-to-move", choices=("black", "white"), default="black")
    export.add_argument("--out", type=Path)

    train = sub.add_parser("train-model", help="Train a reusable recognition model from labeled samples.")
    train.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    train.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    train.add_argument("--calibration-dir", type=Path)
    train.add_argument("--include-hands", action="store_true")
    train.add_argument("--exclude-sample")
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)

    review = sub.add_parser("review", help="Create an HTML visual review for a run directory.")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    review.add_argument("--html", type=Path)
    review.add_argument("--include-hands", action="store_true")

    analysis_html = sub.add_parser("analysis-html", help="Create a static analysis HTML for recognition reports.")
    analysis_html.add_argument("run_dir", type=Path)
    analysis_html.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    analysis_html.add_argument("--no-labels", action="store_true")
    analysis_html.add_argument("--evaluation", type=Path)
    analysis_html.add_argument("--html", type=Path)
    analysis_html.add_argument("--include-hands", action="store_true")
    analysis_html.add_argument("--low-confidence", type=float, default=0.55)

    kif_ui = sub.add_parser("kif-ui", help="Start a local HTML UI for selecting an image and exporting KIF.")
    kif_ui.add_argument("--host", default="127.0.0.1")
    kif_ui.add_argument("--port", type=int, default=8765)
    kif_ui.add_argument("--out", type=Path, default=DEFAULT_OUTPUTS_DIR / "kif_ui")
    kif_ui.add_argument("--model", type=Path, help="Override the bundled model selected by recognition style.")
    kif_ui.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    kif_ui.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    kif_ui.add_argument("--calibration-dir", type=Path)
    kif_ui.add_argument("--no-hands", action="store_true", help="Do not read captured pieces in the UI recognizer.")
    kif_ui.add_argument("--no-train", action="store_true", help="Fail instead of training a model when --model is missing.")

    evaluate = sub.add_parser("evaluate", help="Evaluate recognized piece_report.json files against labels.")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    evaluate.add_argument("--include-hands", action="store_true")
    evaluate.add_argument("--strict-leak-guard", action="store_true")
    evaluate.add_argument("--require-perfect", action="store_true")
    evaluate.add_argument("--out", type=Path)
    evaluate.add_argument("--sample", action="append", default=[], help="Only evaluate samples whose output directory name matches this value. Can be used more than once.")
    evaluate.add_argument("--limit", type=int, help="Evaluate at most this many reports after filtering.")

    validate = sub.add_parser("validate-labels", help="Validate labels and inventory.")
    validate.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)

    args = parser.parse_args(argv)
    try:
        if args.command == "recognize":
            return command_recognize(args)
        if args.command == "batch":
            return command_batch(args)
        if args.command == "export":
            return command_export(args)
        if args.command == "train-model":
            return command_train_model(args)
        if args.command == "review":
            return command_review(args)
        if args.command == "analysis-html":
            return command_analysis_html(args)
        if args.command == "kif-ui":
            return command_kif_ui(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        if args.command == "validate-labels":
            return command_validate_labels(args)
    except ExportError as exc:
        print(f"export error: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"file not found: {exc}")
        return 2
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    return 1


def add_common_recognition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Fail instead of training a model when --model is missing.")
    parser.add_argument("--exclude-sample", help="Exclude this sample stem while training/loading the model.")


def options_from_args(args: argparse.Namespace) -> RecognitionOptions:
    model_path = args.model
    exclude_sample = getattr(args, "exclude_sample", None)
    if exclude_sample and model_path == DEFAULT_MODEL_PATH:
        model_path = DEFAULT_OUTPUTS_DIR / "models" / "no_leak" / f"{exclude_sample}.pkl"
    return RecognitionOptions(
        model_path=model_path,
        screenshots_dir=args.screenshots_dir,
        labels_dir=args.labels,
        calibration_dir=args.calibration_dir or args.screenshots_dir,
        include_hands=args.include_hands,
        train_if_missing=not args.no_train,
        exclude_sample=exclude_sample,
        out_dir=args.out,
    )


def command_recognize(args: argparse.Namespace) -> int:
    result = recognize_image(args.image, options_from_args(args))
    print(json.dumps({"output": result.output_path, "needs_review": result.needs_review}, ensure_ascii=False))
    return 0 if not result.needs_review else 3


def command_batch(args: argparse.Namespace) -> int:
    validate_limit(args.limit)
    images = sorted(path for path in args.input.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    images = filter_named_paths(images, args.sample)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"no images found under {args.input}")
    options = options_from_args(args)
    manifest = []
    review_count = 0
    for index, image in enumerate(images, start=1):
        if args.no_leak:
            options = options_for_no_leak_sample(args, image.stem)
        result = recognize_image(image, options)
        review_count += int(result.needs_review)
        manifest.append({"image": str(image), "output": result.output_path, "needs_review": result.needs_review})
        print(f"[{index}/{len(images)}] {image.name}: {'review' if result.needs_review else 'ok'}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "needs_review": review_count, "manifest": str(args.out / "manifest.json")}, ensure_ascii=False))
    return 0


def command_export(args: argparse.Namespace) -> int:
    result = load_result(args.result)
    if args.format == "kif":
        text = export_kif(result, args.side_to_move)
    elif args.format == "sfen":
        text = export_sfen(result, args.side_to_move) + "\n"
    else:
        text = export_json(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(str(args.out))
    else:
        print(text, end="")
    return 0


def command_train_model(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from learned_piece_recognizer import build_model, save_model
    from train_piece_model import load_training_samples

    calibration_dir = args.calibration_dir or args.screenshots_dir
    samples = load_training_samples(args.labels, args.screenshots_dir, args.exclude_sample, calibration_dir)
    if not samples:
        raise ValueError(
            "no training samples found. Provide --screenshots-dir and --labels with matching labeled screenshots."
        )
    model = build_model(samples, excluded_source=args.exclude_sample, include_hands=args.include_hands)
    save_model(args.out, model)
    print(
        json.dumps(
            {
                "model": str(args.out),
                "samples": len(samples),
                "templates": len(model.get("templates", [])),
                "excluded_source": args.exclude_sample,
            },
            ensure_ascii=False,
        )
    )
    return 0


def command_review(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from make_visual_review import write_visual_review

    html = args.html or args.run_dir / "visual_review.html"
    write_visual_review(args.run_dir, args.labels, html, include_hands=args.include_hands)
    print(str(html))
    return 0


def command_analysis_html(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from make_image_analysis_html import write_image_analysis_html

    html = args.html or args.run_dir / "image_analysis.html"
    write_image_analysis_html(
        args.run_dir,
        labels_dir=None if args.no_labels else args.labels,
        out_path=html,
        evaluation_path=args.evaluation,
        include_hands=args.include_hands,
        low_confidence=args.low_confidence,
    )
    print(str(html))
    return 0


def command_kif_ui(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from serve_image_kif_ui import KifUiConfig, serve_kif_ui

    serve_kif_ui(
        KifUiConfig(
            host=args.host,
            port=args.port,
            out_dir=args.out,
            model_path=args.model,
            screenshots_dir=args.screenshots_dir,
            labels_dir=args.labels,
            calibration_dir=args.calibration_dir,
            include_hands=not args.no_hands,
            train_if_missing=not args.no_train,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from evaluate_piece_recognition import aggregate, evaluate_one
    from position_label_utils import find_label_path

    validate_limit(args.limit)
    reports = sorted(args.run_dir.rglob("piece_report.json"))
    reports = filter_report_paths(reports, args.sample)
    if args.limit is not None:
        reports = reports[: args.limit]
    if not reports:
        raise FileNotFoundError(f"no piece_report.json found under {args.run_dir}")
    results = []
    skipped = []
    for report in reports:
        sample = report.parent.name
        label = find_label_path(args.labels, sample)
        if not label.exists():
            skipped.append({"sample": sample, "report": str(report), "reason": "missing label"})
            continue
        results.append(
            evaluate_one(
                report,
                label,
                0.75,
                include_hands=args.include_hands,
                strict_leak_guard=args.strict_leak_guard,
                forbidden_sources=[sample] if args.strict_leak_guard else [],
                require_excluded_source=args.strict_leak_guard,
            )
        )
    summary = {
        "run_dir": str(args.run_dir),
        "labels": str(args.labels),
        "reports": len(reports),
        "evaluated": len(results),
        "skipped": skipped,
        "results": results,
        "metrics": aggregate(results) if results else {},
    }
    out = args.out or args.run_dir / "evaluation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    if args.strict_leak_guard and summary["metrics"].get("leak_errors", 0):
        return 5
    if args.require_perfect:
        metrics = summary["metrics"]
        failed = (
            bool(skipped)
            or metrics.get("errors", 0)
            or metrics.get("hand_errors", 0)
            or metrics.get("leak_errors", 0)
            or metrics.get("unknown_on_piece", 0)
            or metrics.get("false_empty_on_piece", 0)
            or metrics.get("false_piece_on_empty", 0)
            or metrics.get("high_confidence_errors", 0)
        )
        return 5 if failed else 0
    return 0


def validate_limit(limit: int | None) -> None:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")


def filter_named_paths(paths: list[Path], names: list[str]) -> list[Path]:
    if not names:
        return paths
    allowed = set(names)
    return [path for path in paths if path.stem in allowed]


def filter_report_paths(paths: list[Path], names: list[str]) -> list[Path]:
    if not names:
        return paths
    allowed = set(names)
    return [path for path in paths if path.parent.name in allowed]


def options_for_no_leak_sample(args: argparse.Namespace, sample: str) -> RecognitionOptions:
    model_path = args.out / "_models" / "no_leak" / f"{sample}.pkl"
    return RecognitionOptions(
        model_path=model_path,
        screenshots_dir=args.screenshots_dir,
        labels_dir=args.labels,
        calibration_dir=args.calibration_dir or args.screenshots_dir,
        include_hands=args.include_hands,
        train_if_missing=not args.no_train,
        exclude_sample=sample,
        out_dir=args.out,
    )


def command_validate_labels(args: argparse.Namespace) -> int:
    ensure_tools_on_path()
    from validate_position_labels import validate_label
    from audit_position_label_inventory import inventory_detail, flatten_rows, write_csv, write_html

    paths = sorted(args.labels.rglob("*.json"))
    results = [validate_label(path, require_hands=True) for path in paths]
    validation = {
        "labels_dir": str(args.labels),
        "label_count": len(results),
        "ok_count": sum(1 for result in results if result["ok"]),
        "failed_count": sum(1 for result in results if not result["ok"]),
        "require_hands": True,
    }
    details = []
    inventory_exceptions = []
    for path in paths:
        try:
            item = inventory_detail(path)
        except Exception as exc:
            inventory_exceptions.append({"label": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if item is not None:
            details.append(item)
    out_dir = DEFAULT_OUTPUTS_DIR / "label_inventory_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(details)
    write_csv(out_dir / "label_inventory_audit.csv", rows)
    write_html(out_dir / "label_inventory_audit.html", details)
    print(
        json.dumps(
            {
                "validation": validation,
                "inventory_errors": len(details),
                "inventory_exceptions": inventory_exceptions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if validation.get("failed_count", 0) == 0 and not details and not inventory_exceptions else 4


if __name__ == "__main__":
    raise SystemExit(main())
