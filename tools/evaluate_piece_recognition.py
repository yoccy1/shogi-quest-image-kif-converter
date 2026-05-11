from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from position_label_utils import (
    HAND_PIECES,
    find_label_path,
    identity,
    load_position_label,
)


def default_labels_dir() -> Path:
    return Path(__file__).resolve().parent / "samples" / "labels" / "boards"


def default_reports_dir() -> Path:
    return Path(__file__).resolve().parent / "out" / "analysis_opencv"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def load_labels(path: Path, require_hands: bool = False) -> dict[str, Any]:
    return load_position_label(path, require_hands=require_hands)


def read_key_value_metadata(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        metadata[key.strip()] = value.strip()
    return metadata


def apply_companion_report_metadata(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    metadata = read_key_value_metadata(report_path.with_name("sample_meta.txt"))
    excluded_source = metadata.get("excluded_template_source")
    if not excluded_source:
        return report

    merged = dict(report)
    raw_model = merged.get("model")
    model = dict(raw_model) if isinstance(raw_model, dict) else {}
    if not model.get("excluded_source") and not model.get("excluded_sources"):
        model["excluded_source"] = excluded_source
    model.setdefault("metadata_schema", "android_eval_sample_meta_v1")
    merged["model"] = model
    return merged


def load_report(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if "piece_recognition" in data:
        return apply_companion_report_metadata(data["piece_recognition"], path)
    if "cells" in data:
        return apply_companion_report_metadata(data, path)
    if "rows" in data:
        cells = []
        for row_index, row in enumerate(data["rows"], start=1):
            for col_index, cell in enumerate(row, start=1):
                cells.append({"row": row_index, "col": col_index, **cell, "candidates": []})
        return apply_companion_report_metadata(
            {"method": data.get("method"), "summary": data.get("summary", {}), "cells": cells},
            path,
        )
    raise ValueError(f"{path}: unsupported report format")


def resolve_report_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    direct = path / "piece_report.json"
    if direct.exists():
        return [direct]
    reports = sorted(child / "piece_report.json" for child in path.iterdir() if (child / "piece_report.json").exists())
    if not reports:
        raise FileNotFoundError(f"no piece_report.json found under {path}")
    return reports


def report_sample_name(path: Path) -> str:
    if path.name in {"piece_report.json", "analysis_report.json", "recognized_board.json"}:
        return path.parent.name
    return path.stem


def report_group_context(path: Path) -> tuple[str | None, str | None]:
    if path.name not in {"piece_report.json", "analysis_report.json", "recognized_board.json"}:
        return None, None
    sample_dir = path.parent
    style_dir = sample_dir.parent
    app_dir = style_dir.parent
    if style_dir == sample_dir or app_dir == style_dir:
        return None, None
    return app_dir.name, style_dir.name


def top_candidates(cell: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = cell.get("candidates") or []
    if candidates:
        return candidates
    if cell.get("state") == "piece":
        return [
            {
                "color": cell.get("color"),
                "piece": cell.get("piece"),
                "score": cell.get("confidence"),
                "source": "confirmed",
            },
        ]
    return []


def candidate_key(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "none"
    return identity(candidate.get("color"), candidate.get("piece"))


def prediction_key(cell: dict[str, Any]) -> str:
    if cell.get("state") == "empty":
        return "empty"
    return candidate_key(top_candidates(cell)[0] if top_candidates(cell) else None)


def evaluate_one(
    report_path: Path,
    label_path: Path,
    high_confidence_threshold: float,
    include_hands: bool = False,
    strict_leak_guard: bool = False,
    forbidden_sources: Sequence[str] = (),
    require_excluded_source: bool = True,
) -> dict[str, Any]:
    report = load_report(report_path)
    labels = load_labels(label_path, require_hands=include_hands)
    leak_errors = leak_guard_errors(
        report,
        forbidden_sources,
        require_excluded_source=require_excluded_source,
    ) if strict_leak_guard else []
    reported_cells = {(cell["row"], cell["col"]): cell for cell in report.get("cells", [])}

    metrics = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    piece_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    errors = []
    high_confidence_errors = []
    top3_misses = []

    for expected in labels["cells"]:
        if expected["state"] == "unknown":
            metrics["ignored_unknown"] += 1
            continue
        key = (expected["row"], expected["col"])
        if key not in reported_cells:
            raise ValueError(f"{report_path}: missing reported cell r{key[0]} c{key[1]}")
        actual = reported_cells[key]
        expected_key = "empty" if expected["state"] == "empty" else identity(expected["color"], expected["piece"])
        predicted_key = prediction_key(actual)
        candidates = top_candidates(actual)
        top1 = candidates[0] if candidates else None
        top3_keys = [candidate_key(candidate) for candidate in candidates[:3]]

        metrics["total"] += 1
        if expected["state"] == "empty":
            metrics["true_empty"] += 1
            if actual.get("state") == "empty":
                metrics["empty_correct"] += 1
            else:
                metrics["false_piece_on_empty"] += 1
        elif expected["state"] == "piece":
            metrics["true_piece"] += 1
            if actual.get("state") != "empty":
                metrics["piece_presence_correct"] += 1
            else:
                metrics["false_empty_on_piece"] += 1
            if actual.get("state") == "unknown":
                metrics["unknown_on_piece"] += 1
            if actual.get("state") == "piece" and actual.get("color") == expected["color"] and actual.get("piece") == expected["piece"]:
                metrics["confirmed_identity_correct"] += 1
            if top1 and top1.get("color") == expected["color"]:
                metrics["top1_color_correct"] += 1
            if top1 and top1.get("piece") == expected["piece"]:
                metrics["top1_piece_type_correct"] += 1
            if top1 and top1.get("color") == expected["color"] and top1.get("piece") == expected["piece"]:
                metrics["top1_identity_correct"] += 1
            if expected_key in top3_keys:
                metrics["top3_contains_identity"] += 1
            else:
                top3_misses.append(expected["square"])

        confusion[expected_key][predicted_key] += 1
        if expected["state"] == "piece":
            piece_confusion[expected["piece"]][top1.get("piece") if top1 else "none"] += 1

        is_exact = expected_key == predicted_key
        if expected["state"] == "empty":
            is_exact = actual.get("state") == "empty"
        if not is_exact:
            entry = {
                "square": expected["square"],
                "row": expected["row"],
                "col": expected["col"],
                "expected": expected_key,
                "actual_state": actual.get("state"),
                "predicted_top1": predicted_key,
                "confirmed": identity(actual.get("color"), actual.get("piece")) if actual.get("state") == "piece" else actual.get("state"),
                "confidence": actual.get("confidence"),
                "top3": [
                    {
                        "identity": candidate_key(candidate),
                        "score": candidate.get("score"),
                        "source": candidate.get("source"),
                    }
                    for candidate in candidates[:3]
                ],
            }
            errors.append(entry)
            score = top1.get("score") if top1 else actual.get("confidence")
            if isinstance(score, (int, float)) and score >= high_confidence_threshold:
                high_confidence_errors.append(entry)

    hand_result = evaluate_hands(report, labels) if include_hands else None
    true_piece = metrics["true_piece"]
    true_empty = metrics["true_empty"]
    result = {
        "sample": report_sample_name(report_path),
        "report": str(report_path),
        "labels": str(label_path),
        "recognition_summary": report.get("summary", {}),
        "label_summary": labels["summary"],
        "metrics": {
            "total": metrics["total"],
            "true_piece": true_piece,
            "true_empty": true_empty,
            "empty_accuracy": rate(metrics["empty_correct"], true_empty),
            "piece_presence_accuracy": rate(metrics["piece_presence_correct"], true_piece),
            "confirmed_identity_accuracy": rate(metrics["confirmed_identity_correct"], true_piece),
            "top1_identity_accuracy": rate(metrics["top1_identity_correct"], true_piece),
            "top1_piece_type_accuracy": rate(metrics["top1_piece_type_correct"], true_piece),
            "top1_color_accuracy": rate(metrics["top1_color_correct"], true_piece),
            "top3_contains_identity_accuracy": rate(metrics["top3_contains_identity"], true_piece),
            "unknown_on_piece": metrics["unknown_on_piece"],
            "false_empty_on_piece": metrics["false_empty_on_piece"],
            "false_piece_on_empty": metrics["false_piece_on_empty"],
            "ignored_unknown": metrics["ignored_unknown"],
            "errors": len(errors),
            "high_confidence_errors": len(high_confidence_errors),
            "hand_errors": hand_result["errors"] if hand_result else 0,
            "leak_errors": len(leak_errors),
        },
        "hands": hand_result,
        "leak_errors": leak_errors,
        "confusion_matrix": {expected: dict(predicted) for expected, predicted in sorted(confusion.items())},
        "piece_confusion_matrix": {expected: dict(predicted) for expected, predicted in sorted(piece_confusion.items())},
        "errors": errors,
        "high_confidence_errors": high_confidence_errors,
        "top3_misses": top3_misses,
    }
    return result


def leak_guard_errors(
    report: dict[str, Any],
    forbidden_sources: Sequence[str] = (),
    require_excluded_source: bool = True,
) -> list[str]:
    errors: list[str] = []
    if report.get("method") == "known_board_labels":
        errors.append("method is known_board_labels")
    model = report.get("model") or {}
    excluded_sources, excluded_source_errors = model_excluded_sources(model)
    errors.extend(excluded_source_errors)
    no_leak_excluded_sources, no_leak_option_errors = model_no_leak_option_excluded_sources(model)
    errors.extend(no_leak_option_errors)
    if require_excluded_source:
        for source_name in forbidden_sources:
            if not excluded_sources:
                errors.append(f"model excluded_source is missing, expected {source_name}")
            elif source_name not in excluded_sources:
                errors.append(
                    f"model excluded_sources do not include expected {source_name}: "
                    f"{sorted(excluded_sources)}"
                )
    else:
        for source_name in forbidden_sources:
            if excluded_sources and source_name not in excluded_sources:
                errors.append(
                    f"model excluded_sources do not include expected {source_name}: "
                    f"{sorted(excluded_sources)}"
                )
    for source_name in forbidden_sources:
        if no_leak_excluded_sources and source_name not in no_leak_excluded_sources:
            errors.append(
                "model no_leak_options excluded sources do not include expected "
                f"{source_name}: {sorted(no_leak_excluded_sources)}"
            )
    if excluded_sources and no_leak_excluded_sources and not no_leak_excluded_sources.issubset(excluded_sources):
        errors.append(
            "model no_leak_options excluded sources are not a subset of model excluded sources: "
            f"model={sorted(excluded_sources)} no_leak_options={sorted(no_leak_excluded_sources)}"
        )
    for source_name in forbidden_sources:
        for field in ("training_sources", "template_sources"):
            values = model.get(field) or []
            if isinstance(values, (list, tuple, set)) and source_name in values:
                errors.append(f"model {field} contains forbidden source {source_name}")
    label_corrections = report.get("label_corrections") or {}
    if label_corrections.get("applied"):
        errors.append("label_corrections.applied is true")
    scan_source_fields(report, forbidden_sources, errors, "$")
    return errors


def model_excluded_sources(model: Any) -> tuple[set[str], list[str]]:
    if not isinstance(model, dict):
        return set(), []
    errors: list[str] = []
    sources: set[str] = set()
    excluded_source = model.get("excluded_source")
    if isinstance(excluded_source, str) and excluded_source:
        sources.add(excluded_source)
    elif excluded_source is not None:
        errors.append(f"model excluded_source must be a non-empty string when present: {excluded_source!r}")

    excluded_sources = model.get("excluded_sources")
    if excluded_sources is None:
        return sources, errors
    if not isinstance(excluded_sources, list):
        errors.append(f"model excluded_sources must be a list when present: {excluded_sources!r}")
        return sources, errors
    for index, source in enumerate(excluded_sources):
        if isinstance(source, str) and source:
            sources.add(source)
        else:
            errors.append(f"model excluded_sources[{index}] must be a non-empty string: {source!r}")
    return sources, errors


def model_no_leak_option_excluded_sources(model: Any) -> tuple[set[str], list[str]]:
    if not isinstance(model, dict):
        return set(), []
    options = model.get("no_leak_options")
    if options is None:
        return set(), []
    if not isinstance(options, dict):
        return set(), [f"model no_leak_options must be an object when present: {options!r}"]

    errors: list[str] = []
    sources: set[str] = set()
    for key in ("excluded_template_source", "excludedTemplateSource"):
        excluded_source = options.get(key)
        if isinstance(excluded_source, str) and excluded_source:
            sources.add(excluded_source)
        elif excluded_source is not None:
            errors.append(
                f"model no_leak_options {key} must be a non-empty string "
                f"when present: {excluded_source!r}"
            )

    for key in ("excluded_template_sources", "excludedTemplateSources"):
        excluded_sources = options.get(key)
        if excluded_sources is None:
            continue
        if not isinstance(excluded_sources, list):
            errors.append(
                f"model no_leak_options {key} must be a list when present: "
                f"{excluded_sources!r}"
            )
            continue
        for index, source in enumerate(excluded_sources):
            if isinstance(source, str) and source:
                sources.add(source)
            else:
                errors.append(
                    f"model no_leak_options {key}"
                    f"[{index}] must be a non-empty string: {source!r}"
                )
    return sources, errors


def is_allowed_model_exclusion_path(path: str) -> bool:
    return path in {
        "$.model.excluded_source",
        "$.model.excluded_sources",
        "$.model.no_leak_options.excluded_template_source",
        "$.model.no_leak_options.excluded_template_sources",
        "$.model.no_leak_options.excludedTemplateSource",
        "$.model.no_leak_options.excludedTemplateSources",
    }


def scan_source_fields(value: Any, forbidden_sources: Sequence[str], errors: list[str], path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lowered = str(key).lower()
            if not is_allowed_model_exclusion_path(child_path) and (
                lowered in {"source", "sources"} or lowered.endswith("_source") or lowered.endswith("_sources")
            ):
                check_source_value(child, forbidden_sources, errors, child_path)
            if lowered == "trusted_label" and child:
                errors.append(f"{child_path} is true")
            scan_source_fields(child, forbidden_sources, errors, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_source_fields(child, forbidden_sources, errors, f"{path}[{index}]")


def check_source_value(value: Any, forbidden_sources: Sequence[str], errors: list[str], path: str) -> None:
    if isinstance(value, str):
        if value.startswith("label:") or value.startswith("known_sample:"):
            errors.append(f"{path} is {value}")
        for source_name in forbidden_sources:
            if source_field_contains_source(value, source_name):
                errors.append(f"{path} leaks forbidden source {source_name}: {value}")
    elif isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            check_source_value(child, forbidden_sources, errors, f"{path}[{index}]")


def source_field_contains_source(source_value: str, source_name: str) -> bool:
    for token in source_value.split("+"):
        normalized = token
        prefixes = (
            "hand_learned:",
            "hand_hog:",
            "learned:",
            "app_template:",
            "hand_template:",
            "known_sample:",
            "label:",
            "initial:",
            "calibration:",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True
        if normalized == source_name or normalized.startswith(f"{source_name}:"):
            return True
    return False


def report_hands(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    hands = report.get("hands")
    if isinstance(hands, dict):
        return {
            color: {
                piece: int((hands.get(color) or {}).get(piece, 0))
                for piece in HAND_PIECES
            }
            for color in ("black", "white")
        }
    return {
        color: {piece: 0 for piece in HAND_PIECES}
        for color in ("black", "white")
    }


def evaluate_hands(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    expected = labels.get("hands")
    if expected is None:
        raise ValueError(f"{labels['path']}: hands are required for --include-hands")
    actual = report_hands(report)
    errors = []
    for color in ("black", "white"):
        for piece in HAND_PIECES:
            expected_count = int(expected[color][piece])
            actual_count = int(actual[color][piece])
            if expected_count != actual_count:
                errors.append(
                    {
                        "owner": color,
                        "piece": piece,
                        "expected": expected_count,
                        "actual": actual_count,
                    },
                )
    return {
        "expected": expected,
        "actual": actual,
        "errors": len(errors),
        "error_details": errors,
        "accuracy": 1.0 if not errors else 0.0,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for result in results:
        metrics = result["metrics"]
        totals["true_piece"] += metrics["true_piece"]
        totals["true_empty"] += metrics["true_empty"]
        totals["empty_correct"] += round((metrics["empty_accuracy"] or 0.0) * metrics["true_empty"])
        totals["piece_presence_correct"] += round((metrics["piece_presence_accuracy"] or 0.0) * metrics["true_piece"])
        totals["confirmed_identity_correct"] += round((metrics["confirmed_identity_accuracy"] or 0.0) * metrics["true_piece"])
        totals["top1_identity_correct"] += round((metrics["top1_identity_accuracy"] or 0.0) * metrics["true_piece"])
        totals["top1_piece_type_correct"] += round((metrics["top1_piece_type_accuracy"] or 0.0) * metrics["true_piece"])
        totals["top1_color_correct"] += round((metrics["top1_color_accuracy"] or 0.0) * metrics["true_piece"])
        totals["top3_contains_identity"] += round((metrics["top3_contains_identity_accuracy"] or 0.0) * metrics["true_piece"])
        totals["unknown_on_piece"] += metrics["unknown_on_piece"]
        totals["false_empty_on_piece"] += metrics["false_empty_on_piece"]
        totals["false_piece_on_empty"] += metrics["false_piece_on_empty"]
        totals["ignored_unknown"] += metrics.get("ignored_unknown", 0)
        totals["errors"] += metrics["errors"]
        totals["high_confidence_errors"] += metrics["high_confidence_errors"]
        totals["hand_errors"] += metrics.get("hand_errors", 0)
        totals["leak_errors"] += metrics.get("leak_errors", 0)

    true_piece = totals["true_piece"]
    true_empty = totals["true_empty"]
    return {
        "samples": len(results),
        "true_piece": true_piece,
        "true_empty": true_empty,
        "empty_accuracy": rate(totals["empty_correct"], true_empty),
        "piece_presence_accuracy": rate(totals["piece_presence_correct"], true_piece),
        "confirmed_identity_accuracy": rate(totals["confirmed_identity_correct"], true_piece),
        "top1_identity_accuracy": rate(totals["top1_identity_correct"], true_piece),
        "top1_piece_type_accuracy": rate(totals["top1_piece_type_correct"], true_piece),
        "top1_color_accuracy": rate(totals["top1_color_correct"], true_piece),
        "top3_contains_identity_accuracy": rate(totals["top3_contains_identity"], true_piece),
        "unknown_on_piece": totals["unknown_on_piece"],
        "false_empty_on_piece": totals["false_empty_on_piece"],
        "false_piece_on_empty": totals["false_piece_on_empty"],
        "ignored_unknown": totals["ignored_unknown"],
        "errors": totals["errors"],
        "high_confidence_errors": totals["high_confidence_errors"],
        "hand_errors": totals["hand_errors"],
        "leak_errors": totals["leak_errors"],
    }


def strict_failure_reasons(
    output: dict[str, Any],
    require_perfect: bool,
    fail_on_skipped: bool,
    fail_on_leak: bool = False,
) -> list[str]:
    reasons: list[str] = []
    skipped = output.get("skipped") or []
    if skipped and (require_perfect or fail_on_skipped):
        reasons.append(f"skipped_samples={len(skipped)}")
    summary = output.get("summary") or {}
    if require_perfect:
        for key in (
            "errors",
            "hand_errors",
            "false_empty_on_piece",
            "false_piece_on_empty",
            "unknown_on_piece",
            "high_confidence_errors",
            "leak_errors",
        ):
            if int(summary.get(key) or 0) != 0:
                reasons.append(f"{key}={summary.get(key)}")
    elif fail_on_leak and int(summary.get("leak_errors") or 0) != 0:
        reasons.append(f"leak_errors={summary.get('leak_errors')}")
    return reasons


def print_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"{result['sample']}:")
    print(
        "  empty={empty_accuracy} piece_presence={piece_presence_accuracy} "
        "confirmed_identity={confirmed_identity_accuracy} top1_identity={top1_identity_accuracy} "
        "top3={top3_contains_identity_accuracy}".format(**metrics),
    )
    print(
        "  unknown_on_piece={unknown_on_piece} false_empty={false_empty_on_piece} "
        "false_piece_on_empty={false_piece_on_empty} high_conf_errors={high_confidence_errors} "
        "hand_errors={hand_errors} leak_errors={leak_errors}".format(**metrics),
    )
    if result["high_confidence_errors"]:
        preview = ", ".join(
            f"{entry['square']} {entry['expected']}->{entry['predicted_top1']}({entry['confidence']})"
            for entry in result["high_confidence_errors"][:8]
        )
        print(f"  high_confidence_error_preview: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shogi piece recognition reports against board labels.")
    parser.add_argument("reports", nargs="?", type=Path, default=default_reports_dir(), help="piece_report.json or analysis output directory.")
    parser.add_argument("--labels-dir", type=Path, default=default_labels_dir(), help="Directory containing <sample>.json board labels.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--include-hands", action="store_true", help="Also compare hands.black/white counts.")
    parser.add_argument("--strict-leak-guard", action="store_true", help="Fail reports that contain teacher-label leakage.")
    parser.add_argument("--forbidden-source", action="append", default=[], help="Sample stem that must not appear in learned candidate sources.")
    parser.add_argument("--allow-missing-excluded-source", action="store_true", help="Do not require model.excluded_source when checking app-generated reports.")
    parser.add_argument("--fail-on-skipped", action="store_true", help="Return non-zero if any report is skipped, including missing labels.")
    parser.add_argument("--require-perfect", action="store_true", help="Return non-zero unless board, hands, leak, and high-confidence error metrics are all zero.")
    args = parser.parse_args()

    results = []
    skipped = []
    for report_path in resolve_report_paths(args.reports):
        sample = report_sample_name(report_path)
        app, piece_style = report_group_context(report_path)
        label_path = find_label_path(args.labels_dir, sample, app, piece_style)
        if not label_path.exists():
            skipped.append({"sample": sample, "report": str(report_path), "reason": "label_missing"})
            continue
        results.append(
            evaluate_one(
                report_path,
                label_path,
                args.high_confidence_threshold,
                include_hands=args.include_hands,
                strict_leak_guard=args.strict_leak_guard,
                forbidden_sources=args.forbidden_source,
                require_excluded_source=not args.allow_missing_excluded_source,
            ),
        )

    output = {
        "reports": str(args.reports),
        "labels_dir": str(args.labels_dir),
        "high_confidence_threshold": args.high_confidence_threshold,
        "include_hands": args.include_hands,
        "strict_leak_guard": args.strict_leak_guard,
        "summary": aggregate(results) if results else {},
        "results": results,
        "skipped": skipped,
    }

    for result in results:
        print_result(result)
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  {item['sample']}: {item['reason']}")
    if results:
        print("Aggregate:")
        print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    if args.out:
        write_json(args.out, output)
    failure_reasons = strict_failure_reasons(output, args.require_perfect, args.fail_on_skipped, args.strict_leak_guard)
    if failure_reasons:
        print("Evaluation failed strict checks: " + ", ".join(failure_reasons), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
