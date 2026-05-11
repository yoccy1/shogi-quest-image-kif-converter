from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_piece_recognition import evaluate_one, load_report
from position_label_utils import find_label_path


def report_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    benchmark = path / "benchmark_report.json"
    if benchmark.exists():
        data = json.loads(benchmark.read_text(encoding="utf-8"))
        paths = []
        for result in data.get("results", []):
            report_path = Path(result.get("report", ""))
            if report_path.exists():
                paths.append(report_path)
        if paths:
            return paths
    direct = path / "piece_report.json"
    if direct.exists():
        return [direct]
    return sorted(path.glob("*/piece_report.json"))


def sample_name(report_path: Path, report: dict[str, Any]) -> str:
    image = report.get("image")
    if image:
        return Path(image).stem
    return report_path.parent.name


def candidate_summary(error: dict[str, Any]) -> tuple[bool, bool, int | None, str, str, float | None, float | None]:
    expected = str(error.get("expected"))
    top3 = error.get("top3") or []
    correct_rank = None
    for index, candidate in enumerate(top3, start=1):
        if candidate.get("identity") == expected:
            correct_rank = index
            break
    scores = [float(candidate.get("score") or 0.0) for candidate in top3]
    selected_score = scores[0] if scores else 0.0
    best = max(top3, key=lambda candidate: float(candidate.get("score") or 0.0), default=None)
    best_score = float(best.get("score") or 0.0) if best else None
    best_identity = str(best.get("identity")) if best else ""
    selected_not_max = bool(scores and best_score is not None and selected_score < best_score - 1e-6)
    selected_score_gap = round(best_score - selected_score, 4) if best_score is not None else None
    top3_text = " / ".join(f"{item.get('identity')}:{item.get('score')}" for item in top3)
    return correct_rank is not None, selected_not_max, correct_rank, top3_text, best_identity, best_score, selected_score_gap


def final_change_by_square(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    changes = ((report.get("constraint_postprocess") or {}).get("changes") or [])
    by_square: dict[str, dict[str, Any]] = {}
    for change in changes:
        square = str(change.get("square") or "")
        if square:
            by_square[square] = change
    return by_square


def category(error: dict[str, Any], correct_in_top3: bool, selected_not_max: bool) -> str:
    expected = str(error.get("expected"))
    predicted = str(error.get("predicted_top1"))
    actual_state = str(error.get("actual_state"))
    if expected == "empty":
        return "false_piece"
    if actual_state == "unknown":
        return "unknown_on_piece"
    if predicted == "empty":
        return "false_empty"
    if selected_not_max:
        return "postprocess_or_ordering"
    if correct_in_top3:
        return "near_miss"
    exp_color, _, exp_piece = expected.partition(":")
    pred_color, _, pred_piece = predicted.partition(":")
    if exp_piece == pred_piece and exp_color != pred_color:
        return "color_flip"
    if exp_color == pred_color and exp_piece != pred_piece:
        return "piece_confusion"
    return "color_and_piece_confusion"


def summarize(reports: Path, labels_dir: Path, include_hands: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for report_path in report_paths(reports):
        report = load_report(report_path)
        sample = sample_name(report_path, report)
        changes = final_change_by_square(report)
        label_path = find_label_path(labels_dir, sample)
        if not label_path.exists():
            skipped.append({"sample": sample, "reason": "label_missing", "report": str(report_path)})
            continue
        try:
            result = evaluate_one(report_path, label_path, 0.75, include_hands=include_hands)
        except Exception as exc:
            skipped.append({"sample": sample, "reason": f"{type(exc).__name__}: {exc}", "report": str(report_path)})
            continue
        metrics = result["metrics"]
        sample_rows.append(
            {
                "sample": sample,
                "errors": metrics.get("errors", 0),
                "true_piece": metrics.get("true_piece", 0),
                "confirmed_identity_accuracy": metrics.get("confirmed_identity_accuracy"),
                "top3_contains_identity_accuracy": metrics.get("top3_contains_identity_accuracy"),
                "unknown_on_piece": metrics.get("unknown_on_piece", 0),
                "high_confidence_errors": metrics.get("high_confidence_errors", 0),
                "hand_errors": metrics.get("hand_errors", 0),
            }
        )
        for error in result.get("errors", []):
            correct_in_top3, selected_not_max, correct_rank, top3_text, best_identity, best_score, selected_score_gap = candidate_summary(error)
            change = changes.get(str(error.get("square"))) or {}
            rows.append(
                {
                    "sample": sample,
                    "square": error.get("square"),
                    "expected": error.get("expected"),
                    "predicted_top1": error.get("predicted_top1"),
                    "confirmed": error.get("confirmed"),
                    "actual_state": error.get("actual_state"),
                    "confidence": error.get("confidence"),
                    "correct_in_top3": correct_in_top3,
                    "correct_rank": correct_rank or "",
                    "selected_not_max": selected_not_max,
                    "best_candidate": best_identity,
                    "best_score": best_score if best_score is not None else "",
                    "selected_score_gap": selected_score_gap if selected_score_gap is not None else "",
                    "postprocess_reason": change.get("reason", ""),
                    "postprocess_from": change.get("from", ""),
                    "postprocess_to": change.get("to", ""),
                    "category": category(error, correct_in_top3, selected_not_max),
                    "top3": top3_text,
                }
            )
    by_category: dict[str, int] = {}
    for row in rows:
        key = str(row["category"])
        by_category[key] = by_category.get(key, 0) + 1
    return {"samples": sample_rows, "errors": rows, "by_category": by_category, "skipped": skipped}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recognition errors into actionable CSV/JSON.")
    parser.add_argument("reports", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/error_summary"))
    parser.add_argument("--include-hands", action="store_true")
    args = parser.parse_args()

    result = summarize(args.reports, args.labels_dir, args.include_hands)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "error_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        args.out / "errors.csv",
        result["errors"],
        [
            "sample",
            "square",
            "expected",
            "predicted_top1",
            "confirmed",
            "actual_state",
            "confidence",
            "correct_in_top3",
            "correct_rank",
            "selected_not_max",
            "best_candidate",
            "best_score",
            "selected_score_gap",
            "postprocess_reason",
            "postprocess_from",
            "postprocess_to",
            "category",
            "top3",
        ],
    )
    write_csv(
        args.out / "samples.csv",
        result["samples"],
        [
            "sample",
            "errors",
            "true_piece",
            "confirmed_identity_accuracy",
            "top3_contains_identity_accuracy",
            "unknown_on_piece",
            "high_confidence_errors",
            "hand_errors",
        ],
    )
    print(json.dumps({"samples": len(result["samples"]), "errors": len(result["errors"]), "by_category": result["by_category"], "skipped": result["skipped"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
