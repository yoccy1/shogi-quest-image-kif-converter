from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_piece_recognition import (
    default_labels_dir,
    default_reports_dir,
    evaluate_one,
    load_labels,
    report_sample_name,
    resolve_report_paths,
    write_json,
)


PIECE_NAMES = {
    "OU": "玉",
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
    "RY": "龍",
    "UM": "馬",
    "NG": "成銀",
    "NK": "成桂",
    "NY": "成香",
    "TO": "と",
}
PHYSICAL_PIECE_LIMITS = {
    "OU": 2,
    "HI": 2,
    "KA": 2,
    "KI": 4,
    "GI": 4,
    "KE": 4,
    "KY": 4,
    "FU": 18,
}
PHYSICAL_PIECE_GROUPS = {
    "RY": "HI",
    "UM": "KA",
    "NG": "GI",
    "NK": "KE",
    "NY": "KY",
    "TO": "FU",
}


def identity_text(identity: str) -> str:
    if identity in {"empty", "unknown", "none"}:
        return identity
    if ":" not in identity:
        return identity
    color, piece = identity.split(":", 1)
    color_text = "先手" if color == "black" else "後手"
    return f"{color_text}:{PIECE_NAMES.get(piece, piece)}"


def physical_piece_group(piece: str) -> str:
    return PHYSICAL_PIECE_GROUPS.get(piece, piece)


def physical_count_issues(labels: dict[str, Any]) -> list[dict[str, Any]]:
    counts = Counter(
        physical_piece_group(cell["piece"])
        for cell in labels["cells"]
        if cell["state"] == "piece"
    )
    issues: list[dict[str, Any]] = []
    for group, limit in PHYSICAL_PIECE_LIMITS.items():
        if group == "OU":
            continue
        count = counts[group]
        if count > limit:
            issues.append(
                {
                    "kind": "physical_count_over_limit",
                    "piece_group": group,
                    "piece_group_text": PIECE_NAMES.get(group, group),
                    "count": count,
                    "limit": limit,
                },
            )
    if counts["OU"] != 2:
        issues.append(
            {
                "kind": "king_count_not_two",
                "piece_group": "OU",
                "piece_group_text": PIECE_NAMES["OU"],
                "count": counts["OU"],
                "limit": 2,
            },
        )
    return issues


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def validate_labels(labels_dir: Path, reports_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        sample = label_path.stem
        try:
            labels = load_labels(label_path)
            state_counts = Counter(cell["state"] for cell in labels["cells"])
            label_counts = Counter(
                f"{cell['color']}:{cell['piece']}"
                for cell in labels["cells"]
                if cell["state"] == "piece"
            )
            physical_counts = Counter(
                physical_piece_group(cell["piece"])
                for cell in labels["cells"]
                if cell["state"] == "piece"
            )
            count_issues = physical_count_issues(labels)
            image_value = labels.get("image")
            image_path = resolve_label_image_path(label_path, image_value)
            report_path = reports_dir / sample / "piece_report.json"
            row = {
                "sample": sample,
                "label": str(label_path),
                "image": str(image_path) if image_path else None,
                "image_exists": bool(image_path and image_path.exists()),
                "report": str(report_path),
                "report_exists": report_path.exists(),
                "piece": state_counts["piece"],
                "empty": state_counts["empty"],
                "unknown": state_counts["unknown"],
                "by_label": dict(sorted(label_counts.items())),
                "by_physical_group": dict(sorted(physical_counts.items())),
                "physical_count_issues": count_issues,
            }
            summaries.append(row)
            if not row["image_exists"]:
                issues.append({"sample": sample, "severity": "error", "message": "image_missing", "label": str(label_path)})
            if not row["report_exists"]:
                issues.append({"sample": sample, "severity": "warning", "message": "report_missing", "label": str(label_path)})
            if state_counts["piece"] == 0:
                issues.append({"sample": sample, "severity": "warning", "message": "no_piece_labels", "label": str(label_path)})
            for count_issue in count_issues:
                issues.append(
                    {
                        "sample": sample,
                        "severity": "review",
                        "message": count_issue["kind"],
                        "label": str(label_path),
                        **count_issue,
                    },
                )
        except Exception as exc:
            issues.append(
                {
                    "sample": sample,
                    "severity": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "label": str(label_path),
                },
            )
    return summaries, issues


def resolve_label_image_path(label_path: Path, image_value: str | None) -> Path | None:
    if image_value:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = label_path.parent / image_path
        return image_path.resolve()
    default_path = Path(__file__).resolve().parent / "samples" / "screenshots" / f"{label_path.stem}.png"
    return default_path


def evaluate_reports(
    reports_dir: Path,
    labels_dir: Path,
    high_confidence_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for report_path in resolve_report_paths(reports_dir):
        sample = report_sample_name(report_path)
        label_path = labels_dir / f"{sample}.json"
        if not label_path.exists():
            skipped.append({"sample": sample, "report": str(report_path), "reason": "label_missing"})
            continue
        results.append(evaluate_one(report_path, label_path, high_confidence_threshold))
    return results, skipped


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    for result in results:
        metrics = result["metrics"]
        for key in (
            "true_piece",
            "true_empty",
            "unknown_on_piece",
            "false_empty_on_piece",
            "false_piece_on_empty",
            "ignored_unknown",
            "errors",
            "high_confidence_errors",
        ):
            total[key] += metrics.get(key, 0)
        total["top1_identity_correct"] += round((metrics.get("top1_identity_accuracy") or 0.0) * metrics["true_piece"])
        total["top3_identity_correct"] += round((metrics.get("top3_contains_identity_accuracy") or 0.0) * metrics["true_piece"])
        total["piece_type_correct"] += round((metrics.get("top1_piece_type_accuracy") or 0.0) * metrics["true_piece"])
        total["color_correct"] += round((metrics.get("top1_color_accuracy") or 0.0) * metrics["true_piece"])
    true_piece = total["true_piece"]
    return {
        "samples": len(results),
        "true_piece": true_piece,
        "true_empty": total["true_empty"],
        "top1_identity_accuracy": round(total["top1_identity_correct"] / true_piece, 4) if true_piece else None,
        "top3_contains_identity_accuracy": round(total["top3_identity_correct"] / true_piece, 4) if true_piece else None,
        "top1_piece_type_accuracy": round(total["piece_type_correct"] / true_piece, 4) if true_piece else None,
        "top1_color_accuracy": round(total["color_correct"] / true_piece, 4) if true_piece else None,
        "unknown_on_piece": total["unknown_on_piece"],
        "false_empty_on_piece": total["false_empty_on_piece"],
        "false_piece_on_empty": total["false_piece_on_empty"],
        "ignored_unknown": total["ignored_unknown"],
        "errors": total["errors"],
        "high_confidence_errors": total["high_confidence_errors"],
    }


def review_rows(results: list[dict[str, Any]], reports_dir: Path, include_top3_misses: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for result in results:
        sample = result["sample"]
        sample_dir = reports_dir / sample
        entries = [("high_confidence", item) for item in result["high_confidence_errors"]]
        if include_top3_misses:
            top3_miss_squares = set(result.get("top3_misses", []))
            entries.extend(
                ("top3_miss", item)
                for item in result["errors"]
                if item["square"] in top3_miss_squares
            )
        for reason, item in entries:
            key = (sample, item["row"], item["col"], reason)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "reason": reason,
                    "sample": sample,
                    "square": item["square"],
                    "row": item["row"],
                    "col": item["col"],
                    "expected": item["expected"],
                    "expected_text": identity_text(item["expected"]),
                    "predicted_top1": item["predicted_top1"],
                    "predicted_text": identity_text(item["predicted_top1"]),
                    "confirmed": item["confirmed"],
                    "confidence": item["confidence"],
                    "top3": " | ".join(f"{candidate['identity']}:{candidate.get('score')}" for candidate in item["top3"]),
                    "comparison_png": str(sample_dir / "position_comparison.png"),
                    "candidate_grid_png": str(sample_dir / "candidate_grid.png"),
                    "cell_png": str(sample_dir / "recognition_cells" / f"r{item['row']:02d}_c{item['col']:02d}.png"),
                    "action_note": "認識器側の改善候補。教師ラベルは正として扱い、物理枚数矛盾があるサンプルだけ別途監査対象にする。",
                },
            )
    rows.sort(key=lambda row: (row["reason"] != "high_confidence", row["sample"], -float(row["confidence"] or 0.0)))
    return rows


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reason",
        "sample",
        "square",
        "expected",
        "expected_text",
        "predicted_top1",
        "predicted_text",
        "confirmed",
        "confidence",
        "top3",
        "comparison_png",
        "candidate_grid_png",
        "cell_png",
        "action_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_review_markdown(path: Path, rows: list[dict[str, Any]], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 駒認識レビュー候補",
        "",
        "この一覧は、認識器が高い confidence で外しているセルを優先した改善用ログです。",
        "教師ラベルは正として扱い、物理枚数などの構造矛盾が残るサンプルだけ別途監査対象です。",
        "",
        "| # | reason | sample | square | expected | predicted | confidence | top3 |",
        "|---:|---|---|---|---|---|---:|---|",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        lines.append(
            "| {index} | {reason} | {sample} | {square} | {expected} | {predicted} | {confidence} | {top3} |".format(
                index=index,
                reason=row["reason"],
                sample=row["sample"],
                square=row["square"],
                expected=row["expected_text"],
                predicted=row["predicted_text"],
                confidence=row["confidence"],
                top3=row["top3"].replace("|", "/"),
            ),
        )
    if len(rows) > limit:
        lines.extend(["", f"ほか {len(rows) - limit} 件はCSVに出力しています。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit board labels and create a recognizer improvement list.")
    parser.add_argument("--reports", type=Path, default=default_reports_dir())
    parser.add_argument("--labels-dir", type=Path, default=default_labels_dir())
    parser.add_argument("--out-dir", type=Path, default=default_reports_dir() / "audit")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--include-top3-misses", action="store_true")
    parser.add_argument("--markdown-limit", type=int, default=80)
    args = parser.parse_args()

    label_summaries, label_issues = validate_labels(args.labels_dir, args.reports)
    results, skipped = evaluate_reports(args.reports, args.labels_dir, args.high_confidence_threshold)
    rows = review_rows(results, args.reports, args.include_top3_misses)

    report = {
        "reports": str(args.reports),
        "labels_dir": str(args.labels_dir),
        "high_confidence_threshold": args.high_confidence_threshold,
        "label_count": len(label_summaries),
        "label_issues": label_issues,
        "evaluation_summary": aggregate_results(results),
        "skipped": skipped,
        "review_count": len(rows),
        "review_rows": rows,
        "labels": label_summaries,
    }
    write_json(args.out_dir / "label_audit.json", report)
    write_review_csv(args.out_dir / "recognition_improvement_candidates.csv", rows)
    write_review_markdown(args.out_dir / "recognition_improvement_candidates.md", rows, args.markdown_limit)
    print(f"OK: labels={len(label_summaries)} issues={len(label_issues)} review_rows={len(rows)} out={args.out_dir}")


if __name__ == "__main__":
    main()
