from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_piece_recognition import evaluate_one, load_report
from position_label_utils import find_label_path


def report_paths(path: Path) -> dict[str, Path]:
    direct = path / "piece_report.json"
    if path.is_file():
        report = load_report(path)
        return {sample_name(path, report): path}
    if direct.exists():
        report = load_report(direct)
        return {sample_name(direct, report): direct}
    paths: dict[str, Path] = {}
    for report_path in sorted(path.glob("*/piece_report.json")):
        report = load_report(report_path)
        paths[sample_name(report_path, report)] = report_path
    return paths


def sample_name(report_path: Path, report: dict[str, Any]) -> str:
    image = report.get("image")
    if image:
        return Path(image).stem
    return report_path.parent.name


def error_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(error["square"]): error for error in result.get("errors", [])}


def timing_map(path: Path) -> dict[str, dict[str, Any]]:
    timing_path = path / "timing.csv"
    if not timing_path.exists():
        return {}
    with timing_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample"]: row for row in csv.DictReader(handle)}


def as_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def prediction(error: dict[str, Any] | None) -> str:
    if not error:
        return "correct"
    return str(error.get("predicted_top1") or error.get("confirmed") or "")


def top3_text(error: dict[str, Any] | None) -> str:
    if not error:
        return ""
    return " / ".join(
        f"{candidate.get('identity')}:{candidate.get('score')}"
        for candidate in (error.get("top3") or [])
    )


def correct_rank(error: dict[str, Any] | None) -> int | str:
    if not error:
        return ""
    expected = str(error.get("expected"))
    for index, candidate in enumerate(error.get("top3") or [], start=1):
        if candidate.get("identity") == expected:
            return index
    return ""


def top_source(error: dict[str, Any] | None) -> str:
    if not error:
        return ""
    top3 = error.get("top3") or []
    if not top3:
        return ""
    return str(top3[0].get("source") or "")


def hand_rows(sample: str, before_result: dict[str, Any], after_result: dict[str, Any]) -> list[dict[str, Any]]:
    before_hands = before_result.get("hands") or {}
    after_hands = after_result.get("hands") or {}
    expected = before_hands.get("expected") or after_hands.get("expected") or {}
    before_actual = before_hands.get("actual") or {}
    after_actual = after_hands.get("actual") or {}
    rows = []
    pieces = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
    for owner in ("black", "white"):
        for piece in pieces:
            expected_count = int((expected.get(owner) or {}).get(piece, 0))
            before_count = int((before_actual.get(owner) or {}).get(piece, 0))
            after_count = int((after_actual.get(owner) or {}).get(piece, 0))
            before_ok = before_count == expected_count
            after_ok = after_count == expected_count
            if before_ok and after_ok and before_count == after_count:
                continue
            if not before_ok and after_ok:
                status = "fixed"
            elif before_ok and not after_ok:
                status = "regressed"
            elif before_count != after_count:
                status = "still_wrong_changed"
            else:
                status = "still_wrong"
            rows.append(
                {
                    "sample": sample,
                    "owner": owner,
                    "piece": piece,
                    "status": status,
                    "expected": expected_count,
                    "before_actual": before_count,
                    "after_actual": after_count,
                },
            )
    return rows


def compare_runs(before: Path, after: Path, labels_dir: Path, include_hands: bool, high_confidence_threshold: float) -> dict[str, Any]:
    before_reports = report_paths(before)
    after_reports = report_paths(after)
    before_timing = timing_map(before)
    after_timing = timing_map(after)
    samples = sorted(set(before_reports) & set(after_reports))
    missing = []
    for sample in sorted(set(before_reports) - set(after_reports)):
        missing.append({"sample": sample, "reason": "missing_after"})
    for sample in sorted(set(after_reports) - set(before_reports)):
        missing.append({"sample": sample, "reason": "missing_before"})
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    hand_change_rows: list[dict[str, Any]] = []
    for sample in samples:
        label_path = find_label_path(labels_dir, sample)
        if not label_path.exists():
            missing.append({"sample": sample, "reason": "missing_label"})
            continue
        try:
            before_result = evaluate_one(before_reports[sample], label_path, high_confidence_threshold, include_hands=include_hands)
            after_result = evaluate_one(after_reports[sample], label_path, high_confidence_threshold, include_hands=include_hands)
        except Exception as exc:
            missing.append({"sample": sample, "reason": f"evaluate_failed: {type(exc).__name__}: {exc}"})
            continue
        before_errors = error_map(before_result)
        after_errors = error_map(after_result)
        before_count = int(before_result["metrics"]["errors"])
        after_count = int(after_result["metrics"]["errors"])
        before_seconds = as_float((before_timing.get(sample) or {}).get("seconds"))
        after_seconds = as_float((after_timing.get(sample) or {}).get("seconds"))
        sample_rows.append(
            {
                "sample": sample,
                "before_errors": before_count,
                "after_errors": after_count,
                "delta_errors": after_count - before_count,
                "before_accuracy": before_result["metrics"].get("confirmed_identity_accuracy"),
                "after_accuracy": after_result["metrics"].get("confirmed_identity_accuracy"),
                "before_top3_accuracy": before_result["metrics"].get("top3_contains_identity_accuracy"),
                "after_top3_accuracy": after_result["metrics"].get("top3_contains_identity_accuracy"),
                "before_unknown_on_piece": before_result["metrics"].get("unknown_on_piece", 0),
                "after_unknown_on_piece": after_result["metrics"].get("unknown_on_piece", 0),
                "before_high_confidence_errors": before_result["metrics"].get("high_confidence_errors", 0),
                "after_high_confidence_errors": after_result["metrics"].get("high_confidence_errors", 0),
                "before_hand_errors": before_result["metrics"].get("hand_errors", 0),
                "after_hand_errors": after_result["metrics"].get("hand_errors", 0),
                "before_seconds": before_seconds if before_seconds is not None else "",
                "after_seconds": after_seconds if after_seconds is not None else "",
                "delta_seconds": round(after_seconds - before_seconds, 4) if before_seconds is not None and after_seconds is not None else "",
            },
        )
        if include_hands:
            hand_change_rows.extend(hand_rows(sample, before_result, after_result))
        for square in sorted(set(before_errors) | set(after_errors)):
            before_error = before_errors.get(square)
            after_error = after_errors.get(square)
            if before_error and not after_error:
                status = "fixed"
                expected = before_error.get("expected")
            elif after_error and not before_error:
                status = "regressed"
                expected = after_error.get("expected")
            else:
                status = "still_wrong_changed" if prediction(before_error) != prediction(after_error) else "still_wrong"
                expected = after_error.get("expected") if after_error else before_error.get("expected")
            rows.append(
                {
                    "sample": sample,
                    "square": square,
                    "row": (after_error or before_error or {}).get("row", ""),
                    "col": (after_error or before_error or {}).get("col", ""),
                    "status": status,
                    "expected": expected,
                    "before_prediction": prediction(before_error),
                    "after_prediction": prediction(after_error),
                    "before_actual_state": before_error.get("actual_state", "") if before_error else "",
                    "after_actual_state": after_error.get("actual_state", "") if after_error else "",
                    "before_confirmed": before_error.get("confirmed", "") if before_error else "",
                    "after_confirmed": after_error.get("confirmed", "") if after_error else "",
                    "before_confidence": before_error.get("confidence", "") if before_error else "",
                    "after_confidence": after_error.get("confidence", "") if after_error else "",
                    "confidence_delta": (
                        round(float(after_error.get("confidence")) - float(before_error.get("confidence")), 4)
                        if before_error
                        and after_error
                        and isinstance(before_error.get("confidence"), (int, float))
                        and isinstance(after_error.get("confidence"), (int, float))
                        else ""
                    ),
                    "before_correct_rank": correct_rank(before_error),
                    "after_correct_rank": correct_rank(after_error),
                    "before_source": top_source(before_error),
                    "after_source": top_source(after_error),
                    "before_top3": top3_text(before_error),
                    "after_top3": top3_text(after_error),
                    "is_high_conf_regression": (
                        status == "regressed"
                        and isinstance(after_error.get("confidence") if after_error else None, (int, float))
                        and float(after_error.get("confidence")) >= high_confidence_threshold
                    ),
                },
            )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "before": str(before),
        "after": str(after),
        "labels_dir": str(labels_dir),
        "include_hands": include_hands,
        "high_confidence_threshold": high_confidence_threshold,
        "samples": sample_rows,
        "changes": rows,
        "hand_changes": hand_change_rows,
        "missing": missing,
        "status_counts": status_counts,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two recognition output directories against teacher labels.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/recognition_run_diff"))
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.75)
    args = parser.parse_args()

    result = compare_runs(args.before, args.after, args.labels_dir, args.include_hands, args.high_confidence_threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        args.out / "samples.csv",
        result["samples"],
        [
            "sample",
            "before_errors",
            "after_errors",
            "delta_errors",
            "before_accuracy",
            "after_accuracy",
            "before_top3_accuracy",
            "after_top3_accuracy",
            "before_unknown_on_piece",
            "after_unknown_on_piece",
            "before_high_confidence_errors",
            "after_high_confidence_errors",
            "before_hand_errors",
            "after_hand_errors",
            "before_seconds",
            "after_seconds",
            "delta_seconds",
        ],
    )
    write_csv(
        args.out / "changes.csv",
        result["changes"],
        [
            "sample",
            "square",
            "row",
            "col",
            "status",
            "expected",
            "before_prediction",
            "after_prediction",
            "before_actual_state",
            "after_actual_state",
            "before_confirmed",
            "after_confirmed",
            "before_confidence",
            "after_confidence",
            "confidence_delta",
            "before_correct_rank",
            "after_correct_rank",
            "before_source",
            "after_source",
            "before_top3",
            "after_top3",
            "is_high_conf_regression",
        ],
    )
    write_csv(
        args.out / "hand_changes.csv",
        result["hand_changes"],
        ["sample", "owner", "piece", "status", "expected", "before_actual", "after_actual"],
    )
    write_csv(args.out / "missing.csv", result["missing"], ["sample", "reason"])
    print(
        json.dumps(
            {
                "samples": len(result["samples"]),
                "status_counts": result["status_counts"],
                "hand_changes": len(result["hand_changes"]),
                "missing": result["missing"],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
