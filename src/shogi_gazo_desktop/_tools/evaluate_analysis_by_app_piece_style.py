from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_piece_recognition import aggregate, candidate_key, evaluate_one, load_report
from position_label_utils import find_label_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_ASSET = ROOT / "app" / "src" / "main" / "assets" / "app_piece_templates.json"

PIECE_JP = {
    "OU": "玉",
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
    "RY": "竜",
    "UM": "馬",
    "TO": "と",
    "NY": "成香",
    "NK": "成桂",
    "NG": "成銀",
}

COLOR_JP = {
    "black": "先手",
    "white": "後手",
}

PRESET_BY_APP_STYLE = {
    ("将棋ウォーズ", "一文字"): "wars_one",
    ("将棋ウォーズ", "二文字"): "wars_two",
    ("将棋クエスト", "一文字駒"): "quest_one",
    ("将棋クエスト", "書籍風"): "quest_book",
    ("将棋クエスト", "クラシック二文字駒"): "quest_classic_two",
    ("ぴよ将棋", "一文字駒"): "piyo_one",
    ("ぴよ将棋", "二文字駒"): "piyo_two",
    ("ぴよ将棋", "太文字駒"): "piyo_bold",
    ("ぴよ将棋", "ひよこ駒"): "piyo_chick",
    ("ぴよ将棋", "昇竜"): "piyo_shoryu",
    ("ぴよ将棋", "昇竜一文字"): "piyo_shoryu_one",
    ("ぴよ将棋", "風波一文字"): "piyo_kazanami_one",
}

ASSET_COLOR_BY_REPORT_COLOR = {
    "black": "BLACK",
    "white": "WHITE",
}

COMPONENT_GLYPH_DIAGNOSTIC_FIELDS = [
    "source_connected_component_count",
    "target_connected_component_count",
    "source_largest_component_area_ratio",
    "target_largest_component_area_ratio",
    "source_red_dominant_component_area_ratio",
    "target_red_dominant_component_area_ratio",
    "source_edge_touching_component_area_ratio",
    "target_edge_touching_component_area_ratio",
    "source_component_pruned_glyph_bbox_width",
    "source_component_pruned_glyph_bbox_height",
    "target_component_pruned_glyph_bbox_width",
    "target_component_pruned_glyph_bbox_height",
    "source_component_pruned_glyph_bbox_center_x",
    "source_component_pruned_glyph_bbox_center_y",
    "target_component_pruned_glyph_bbox_center_x",
    "target_component_pruned_glyph_bbox_center_y",
    "source_component_pruned_glyph_ink_count",
    "target_component_pruned_glyph_ink_count",
    "source_component_pruned_glyph_clean_ink_count",
    "target_component_pruned_glyph_clean_ink_count",
]


def identity_jp(value: Any) -> str:
    text = str(value or "")
    if text == "empty":
        return "空"
    if text in {"unknown", "none", ""}:
        return "不明"
    color, separator, piece = text.partition(":")
    if separator:
        return f"{COLOR_JP.get(color, color)} {PIECE_JP.get(piece, piece)}"
    return PIECE_JP.get(text, text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_template_assets(path: Path = DEFAULT_TEMPLATE_ASSET) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    templates = data.get("templates") if isinstance(data, dict) else []
    return [template for template in templates if isinstance(template, dict)]


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return round(ordered[index], 4)


def timing_summary(rows: list[dict[str, str]], max_seconds: float) -> dict[str, Any]:
    parsed: list[tuple[dict[str, str], float]] = []
    missing = 0
    for row in rows:
        raw_seconds = (row.get("seconds") or "").strip()
        if not raw_seconds:
            missing += 1
            continue
        try:
            parsed.append((row, float(raw_seconds)))
        except ValueError:
            missing += 1
    seconds = [seconds for _, seconds in parsed]
    over = [(row, seconds) for row, seconds in parsed if seconds > max_seconds]
    return {
        "images": len(rows),
        "timed_images": len(parsed),
        "missing_seconds_count": missing,
        "average_seconds": round(mean(seconds), 4) if seconds else None,
        "p95_seconds": percentile(seconds, 0.95),
        "max_seconds": round(max_seconds, 4),
        "max_observed_seconds": round(max(seconds), 4) if seconds else None,
        "over_limit_count": len(over),
        "over_limit_samples": [
            {
                "sample": row.get("sample"),
                "seconds": round(seconds, 4),
                "app": row.get("app"),
                "piece_style": row.get("piece_style") or row.get("glyph"),
            }
            for row, seconds in over
        ],
    }


def summarize_group(
    app: str,
    piece_style: str,
    rows: list[dict[str, str]],
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    out_dir: Path,
    max_seconds: float,
) -> dict[str, Any]:
    metrics = aggregate(results) if results else {}
    summary: dict[str, Any] = {
        "app": app,
        "piece_style": piece_style,
        "manifest_images": len(rows),
        "evaluated_samples": len(results),
        "skipped_samples": len(skipped),
        "labeled": sum(row.get("label_status") == "教師ラベルあり" for row in rows),
        "initial": sum(row.get("label_status") == "初期配置ラベル" for row in rows),
        "unlabeled": sum(row.get("label_status") == "教師ラベル未作成" for row in rows),
        "visual_review_html": str(out_dir / app / piece_style / "visual_review.html"),
        "visual_review_exists": (out_dir / app / piece_style / "visual_review.html").exists(),
        "timing": timing_summary(rows, max_seconds),
    }
    summary.update(metrics)
    return summary


def evaluate_manifest(
    out_dir: Path,
    labels_dir: Path,
    include_hands: bool,
    high_confidence_threshold: float,
    max_seconds: float,
    strict_leak_guard: bool,
    require_excluded_source: bool,
) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.csv"
    rows = read_csv(manifest_path)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in rows:
        app = row.get("app") or "未分類"
        piece_style = row.get("piece_style") or row.get("glyph") or "未分類"
        groups[(app, piece_style)].append(row)

        sample = row.get("sample") or ""
        report_path = Path(row.get("report") or "")
        label_path = find_label_path(labels_dir, sample, app, piece_style)
        if not label_path.exists():
            row["label_status"] = "教師ラベル未作成"
            skipped.append(
                {
                    "sample": sample,
                    "app": app,
                    "piece_style": piece_style,
                    "report": str(report_path),
                    "reason": "label_missing",
                }
            )
            continue
        row["label_status"] = "教師ラベルあり"
        try:
            result = evaluate_one(
                report_path,
                label_path,
                high_confidence_threshold,
                include_hands=include_hands,
                strict_leak_guard=strict_leak_guard,
                forbidden_sources=(sample,),
                require_excluded_source=require_excluded_source,
            )
        except Exception as exc:
            skipped.append(
                {
                    "sample": sample,
                    "app": app,
                    "piece_style": piece_style,
                    "report": str(report_path),
                    "labels": str(label_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        result["app"] = app
        result["piece_style"] = piece_style
        raw_seconds = (row.get("seconds") or "").strip()
        try:
            result["seconds"] = float(raw_seconds) if raw_seconds else None
        except ValueError:
            result["seconds"] = None
        results.append(result)

    by_group_results: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_group_skipped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_group_results[(result["app"], result["piece_style"])].append(result)
    for item in skipped:
        by_group_skipped[(item["app"], item["piece_style"])].append(item)

    group_summaries = [
        summarize_group(
            app,
            piece_style,
            group_rows,
            by_group_results.get((app, piece_style), []),
            by_group_skipped.get((app, piece_style), []),
            out_dir,
            max_seconds,
        )
        for (app, piece_style), group_rows in sorted(groups.items())
    ]

    overall = aggregate(results) if results else {}
    overall.update(
        {
            "manifest_images": len(rows),
            "evaluated_samples": len(results),
            "skipped_samples": len(skipped),
            "groups": len(groups),
            "timing": timing_summary(rows, max_seconds),
        }
    )
    return {
        "analysis_dir": str(out_dir),
        "labels_dir": str(labels_dir),
        "include_hands": include_hands,
        "high_confidence_threshold": high_confidence_threshold,
        "max_seconds": max_seconds,
        "strict_leak_guard": strict_leak_guard,
        "require_excluded_source": require_excluded_source,
        "overall": overall,
        "groups": group_summaries,
        "skipped": skipped,
        "results": results,
    }


def collect_board_error_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for error in result.get("errors") or []:
            rows.append(
                {
                    "app": result.get("app"),
                    "piece_style": result.get("piece_style"),
                    "sample": result.get("sample"),
                    "square": error.get("square"),
                    "expected": error.get("expected"),
                    "expected_jp": identity_jp(error.get("expected")),
                    "predicted_top1": error.get("predicted_top1"),
                    "predicted_top1_jp": identity_jp(error.get("predicted_top1")),
                    "confirmed": error.get("confirmed"),
                    "confirmed_jp": identity_jp(error.get("confirmed")),
                    "actual_state": error.get("actual_state"),
                    "confidence": error.get("confidence"),
                    "top3_jp": " / ".join(
                        f"{identity_jp(candidate.get('identity'))}:{candidate.get('score')}"
                        for candidate in (error.get("top3") or [])
                    ),
                }
            )
    return rows


def cell_position(cell: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(cell.get("row")), int(cell.get("col"))
    except (TypeError, ValueError):
        return None


def report_cells_by_position(report_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    report = load_report(report_path)
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in report.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        position = cell_position(cell)
        if position is not None:
            cells[position] = cell
    return cells


def candidate_list_for_diagnostics(cell: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for key in ("diagnostic_candidates", "debug_candidates", "candidates"):
        candidates = cell.get(key) or []
        if isinstance(candidates, list) and candidates:
            return key, [candidate for candidate in candidates if isinstance(candidate, dict)]
    return "", []


def candidate_score(candidate: dict[str, Any]) -> Any:
    if "score" in candidate:
        return candidate.get("score")
    return candidate.get("confidence")


def score_breakdown(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("score_breakdown") or {}
    return value if isinstance(value, dict) else {}


def collect_board_error_candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cell_cache: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    breakdown_fields = [
        "shape",
        "raw_dice",
        "raw_iou",
        "clean_dice",
        "clean_iou",
        "clean_area",
        "clean_shape",
        "bbox",
        "center",
        "density",
        "red",
        "position_boost",
        "template_position_boost",
        "position_prior_boost",
        "score_without_template_position",
        "score_without_position_prior",
        "score_without_position_boost",
        "unweighted_template_score",
        "template_weight_contribution",
        "base_template_score",
        "weighted_template_score",
        "exact_template_weight_contribution",
        "score_formula_residual",
        "template_weight",
        "source_bbox_width",
        "source_bbox_height",
        "target_bbox_width",
        "target_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
        "target_bbox_center_x",
        "target_bbox_center_y",
        "source_ink_ratio",
        "target_ink_ratio",
        "source_clean_ink_ratio",
        "target_clean_ink_ratio",
        "source_ink_count",
        "target_ink_count",
        "source_clean_ink_count",
        "target_clean_ink_count",
        *COMPONENT_GLYPH_DIAGNOSTIC_FIELDS,
        "source_red_share",
        "target_red_share",
        "source_central_red_share",
        "source_edge_red_share",
        "target_central_red_share",
        "target_edge_red_share",
        "source_red_center_x",
        "source_red_center_y",
        "target_red_center_x",
        "target_red_center_y",
    ]
    for result in results:
        report_value = str(result.get("report") or "")
        if report_value and report_value not in cell_cache:
            try:
                cell_cache[report_value] = report_cells_by_position(Path(report_value))
            except Exception:
                cell_cache[report_value] = {}
        cells = cell_cache.get(report_value, {})
        for error in result.get("errors") or []:
            try:
                position = (int(error.get("row")), int(error.get("col")))
            except (TypeError, ValueError):
                position = None
            cell = cells.get(position) if position is not None else None
            candidate_set, candidates = candidate_list_for_diagnostics(cell or {})
            expected = error.get("expected")
            predicted_top1 = error.get("predicted_top1")
            identities = [candidate_key(candidate) for candidate in candidates]
            expected_rank = next(
                (
                    candidate.get("rank") or index
                    for index, candidate in enumerate(candidates, start=1)
                    if identities[index - 1] == expected
                ),
                "",
            )
            base_row = {
                "app": result.get("app"),
                "piece_style": result.get("piece_style"),
                "sample": result.get("sample"),
                "square": error.get("square"),
                "row": error.get("row"),
                "col": error.get("col"),
                "expected": expected,
                "expected_jp": identity_jp(expected),
                "predicted_top1": predicted_top1,
                "predicted_top1_jp": identity_jp(predicted_top1),
                "confirmed": error.get("confirmed"),
                "confirmed_jp": identity_jp(error.get("confirmed")),
                "actual_state": error.get("actual_state"),
                "confidence": error.get("confidence"),
                "candidate_set": candidate_set,
                "expected_in_candidate_set": bool(expected_rank),
                "expected_candidate_rank": expected_rank,
            }
            if not candidates:
                rows.append(base_row)
                continue
            for index, candidate in enumerate(candidates, start=1):
                identity = identities[index - 1]
                row = dict(base_row)
                row.update(
                    {
                        "candidate_index": index,
                        "candidate_rank": candidate.get("rank") or index,
                        "candidate_identity": identity,
                        "candidate_jp": identity_jp(identity),
                        "is_expected": identity == expected,
                        "is_predicted_top1": identity == predicted_top1,
                        "score": candidate_score(candidate),
                        "source": candidate.get("source"),
                    }
                )
                breakdown = score_breakdown(candidate)
                row.update({field: breakdown.get(field) for field in breakdown_fields})
                rows.append(row)
    return rows


def numeric_delta(left: Any, right: Any) -> float | str:
    try:
        return round(float(left) - float(right), 4)
    except (TypeError, ValueError):
        return ""


def numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_numeric(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [value for row in rows if (value := numeric_value(row.get(field))) is not None]
    return round(mean(values), 4) if values else ""


def sort_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int]:
        for field in ("candidate_index", "candidate_rank"):
            try:
                return int(row.get(field) or 9999), int(row.get("candidate_rank") or 9999)
            except (TypeError, ValueError):
                continue
        return 9999, 9999

    return sorted(rows, key=key)


def collect_board_error_candidate_gap_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (
            row.get("app"),
            row.get("piece_style"),
            row.get("sample"),
            row.get("square"),
            row.get("row"),
            row.get("col"),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    delta_fields = [
        "shape",
        "raw_dice",
        "clean_shape",
        "bbox",
        "center",
        "density",
        "red",
        "position_boost",
        "template_position_boost",
        "position_prior_boost",
        "score_without_template_position",
        "score_without_position_prior",
        "score_without_position_boost",
        "unweighted_template_score",
        "template_weight_contribution",
        "base_template_score",
        "weighted_template_score",
        "exact_template_weight_contribution",
        "score_formula_residual",
        "source_bbox_width",
        "source_bbox_height",
        "target_bbox_width",
        "target_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
        "target_bbox_center_x",
        "target_bbox_center_y",
        "source_ink_ratio",
        "target_ink_ratio",
        "source_clean_ink_ratio",
        "target_clean_ink_ratio",
        "source_ink_count",
        "target_ink_count",
        "source_clean_ink_count",
        "target_clean_ink_count",
        *COMPONENT_GLYPH_DIAGNOSTIC_FIELDS,
    ]
    for group_rows in groups.values():
        ordered = sort_candidate_rows(group_rows)
        first = ordered[0]
        top = next((row for row in ordered if row.get("candidate_index") in {1, "1"}), ordered[0])
        expected = next((row for row in ordered if row.get("is_expected") in {True, "True", "true", "1"}), None)
        runner_up = next((row for row in ordered if row is not top), None)
        row = {
            "app": first.get("app"),
            "piece_style": first.get("piece_style"),
            "sample": first.get("sample"),
            "square": first.get("square"),
            "row": first.get("row"),
            "col": first.get("col"),
            "expected": first.get("expected"),
            "expected_jp": first.get("expected_jp"),
            "predicted_top1": first.get("predicted_top1"),
            "predicted_top1_jp": first.get("predicted_top1_jp"),
            "confirmed": first.get("confirmed"),
            "confirmed_jp": first.get("confirmed_jp"),
            "actual_state": first.get("actual_state"),
            "confidence": first.get("confidence"),
            "candidate_set": first.get("candidate_set"),
            "expected_in_candidate_set": first.get("expected_in_candidate_set"),
            "expected_candidate_rank": first.get("expected_candidate_rank"),
            "top_identity": top.get("candidate_identity"),
            "top_jp": top.get("candidate_jp"),
            "top_score": top.get("score"),
            "top_source": top.get("source"),
            "top_position_boost": top.get("position_boost"),
            "top_template_position_boost": top.get("template_position_boost"),
            "top_position_prior_boost": top.get("position_prior_boost"),
            "runner_up_identity": runner_up.get("candidate_identity") if runner_up else "",
            "runner_up_jp": runner_up.get("candidate_jp") if runner_up else "",
            "runner_up_score": runner_up.get("score") if runner_up else "",
            "runner_up_source": runner_up.get("source") if runner_up else "",
            "expected_score": expected.get("score") if expected else "",
            "expected_source": expected.get("source") if expected else "",
            "score_gap_top_minus_expected": numeric_delta(top.get("score"), expected.get("score") if expected else None),
            "rank_gap_top_to_expected": numeric_delta(first.get("expected_candidate_rank"), 1),
        }
        for field in delta_fields:
            row[f"top_{field}"] = top.get(field)
            row[f"expected_{field}"] = expected.get(field) if expected else ""
            row[f"delta_{field}"] = numeric_delta(top.get(field), expected.get(field) if expected else None)
        rows.append(row)
    return rows


def collect_board_error_candidate_gap_summary_rows(gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in gap_rows:
        key = (
            row.get("app"),
            row.get("piece_style"),
            row.get("predicted_top1"),
            row.get("expected"),
        )
        groups.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (app, piece_style, predicted_top1, expected), group_rows in sorted(groups.items()):
        ranks = [
            value
            for row in group_rows
            if (value := numeric_value(row.get("expected_candidate_rank"))) is not None
        ]
        row = {
            "app": app,
            "piece_style": piece_style,
            "predicted_top1": predicted_top1,
            "predicted_top1_jp": identity_jp(predicted_top1),
            "expected": expected,
            "expected_jp": identity_jp(expected),
            "count": len(group_rows),
            "avg_score_gap_top_minus_expected": average_numeric(group_rows, "score_gap_top_minus_expected"),
            "avg_delta_shape": average_numeric(group_rows, "delta_shape"),
            "avg_delta_raw_dice": average_numeric(group_rows, "delta_raw_dice"),
            "avg_delta_clean_shape": average_numeric(group_rows, "delta_clean_shape"),
            "avg_delta_bbox": average_numeric(group_rows, "delta_bbox"),
            "avg_delta_center": average_numeric(group_rows, "delta_center"),
            "avg_delta_density": average_numeric(group_rows, "delta_density"),
            "avg_delta_red": average_numeric(group_rows, "delta_red"),
            "avg_delta_position_boost": average_numeric(group_rows, "delta_position_boost"),
            "avg_delta_template_position_boost": average_numeric(group_rows, "delta_template_position_boost"),
            "avg_delta_position_prior_boost": average_numeric(group_rows, "delta_position_prior_boost"),
            "avg_delta_score_without_template_position": average_numeric(
                group_rows,
                "delta_score_without_template_position",
            ),
            "avg_delta_score_without_position_prior": average_numeric(
                group_rows,
                "delta_score_without_position_prior",
            ),
            "avg_delta_score_without_position_boost": average_numeric(
                group_rows,
                "delta_score_without_position_boost",
            ),
            "avg_delta_unweighted_template_score": average_numeric(
                group_rows,
                "delta_unweighted_template_score",
            ),
            "avg_delta_template_weight_contribution": average_numeric(
                group_rows,
                "delta_template_weight_contribution",
            ),
            "avg_delta_base_template_score": average_numeric(
                group_rows,
                "delta_base_template_score",
            ),
            "avg_delta_weighted_template_score": average_numeric(
                group_rows,
                "delta_weighted_template_score",
            ),
            "avg_delta_exact_template_weight_contribution": average_numeric(
                group_rows,
                "delta_exact_template_weight_contribution",
            ),
            "avg_delta_score_formula_residual": average_numeric(
                group_rows,
                "delta_score_formula_residual",
            ),
            "avg_delta_source_bbox_width": average_numeric(group_rows, "delta_source_bbox_width"),
            "avg_delta_source_bbox_height": average_numeric(group_rows, "delta_source_bbox_height"),
            "avg_delta_target_bbox_width": average_numeric(group_rows, "delta_target_bbox_width"),
            "avg_delta_target_bbox_height": average_numeric(group_rows, "delta_target_bbox_height"),
            "avg_delta_source_bbox_center_x": average_numeric(group_rows, "delta_source_bbox_center_x"),
            "avg_delta_source_bbox_center_y": average_numeric(group_rows, "delta_source_bbox_center_y"),
            "avg_delta_target_bbox_center_x": average_numeric(group_rows, "delta_target_bbox_center_x"),
            "avg_delta_target_bbox_center_y": average_numeric(group_rows, "delta_target_bbox_center_y"),
            "avg_delta_source_ink_ratio": average_numeric(group_rows, "delta_source_ink_ratio"),
            "avg_delta_target_ink_ratio": average_numeric(group_rows, "delta_target_ink_ratio"),
            "avg_delta_source_clean_ink_ratio": average_numeric(group_rows, "delta_source_clean_ink_ratio"),
            "avg_delta_target_clean_ink_ratio": average_numeric(group_rows, "delta_target_clean_ink_ratio"),
            "avg_delta_source_ink_count": average_numeric(group_rows, "delta_source_ink_count"),
            "avg_delta_target_ink_count": average_numeric(group_rows, "delta_target_ink_count"),
            "avg_delta_source_clean_ink_count": average_numeric(group_rows, "delta_source_clean_ink_count"),
            "avg_delta_target_clean_ink_count": average_numeric(group_rows, "delta_target_clean_ink_count"),
            "expected_rank_min": round(min(ranks), 4) if ranks else "",
            "expected_rank_avg": round(mean(ranks), 4) if ranks else "",
            "expected_rank_max": round(max(ranks), 4) if ranks else "",
        }
        row.update(
            {
                f"avg_delta_{field}": average_numeric(group_rows, f"delta_{field}")
                for field in COMPONENT_GLYPH_DIAGNOSTIC_FIELDS
            }
        )
        rows.append(row)
    return rows


def identity_parts(value: Any) -> tuple[str, str] | None:
    color, separator, piece = str(value or "").partition(":")
    if not separator or not color or not piece:
        return None
    return color, piece


def template_source_label(template: dict[str, Any]) -> str:
    source = template.get("source") or ""
    row = template.get("row")
    col = template.get("col")
    if row is None or col is None:
        return str(source)
    return f"{source}:r{row}:c{col}"


def matching_templates(
    templates: list[dict[str, Any]],
    preset: str | None,
    identity_value: Any,
) -> list[dict[str, Any]]:
    parts = identity_parts(identity_value)
    if preset is None or parts is None:
        return []
    color, piece = parts
    asset_color = ASSET_COLOR_BY_REPORT_COLOR.get(color)
    return [
        template
        for template in templates
        if template.get("preset") == preset and
            template.get("color") == asset_color and
            template.get("piece") == piece
    ]


def template_supply_details(
    templates: list[dict[str, Any]],
    sample: Any,
) -> dict[str, Any]:
    sample_name = str(sample or "")
    available = [template for template in templates if template.get("source") != sample_name]
    excluded = [template for template in templates if template.get("source") == sample_name]
    return {
        "asset_template_count": len(templates),
        "available_template_count": len(available),
        "excluded_template_count": len(excluded),
        "available_template_sources": " / ".join(template_source_label(template) for template in available),
        "excluded_template_sources": " / ".join(template_source_label(template) for template in excluded),
    }


def collect_board_error_template_supply_rows(
    gap_rows: list[dict[str, Any]],
    templates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gap in gap_rows:
        app = gap.get("app")
        piece_style = gap.get("piece_style")
        preset = PRESET_BY_APP_STYLE.get((str(app), str(piece_style)))
        expected_templates = matching_templates(templates, preset, gap.get("expected"))
        predicted_templates = matching_templates(templates, preset, gap.get("predicted_top1"))
        expected_supply = template_supply_details(expected_templates, gap.get("sample"))
        predicted_supply = template_supply_details(predicted_templates, gap.get("sample"))
        rows.append(
            {
                "app": app,
                "piece_style": piece_style,
                "preset": preset or "",
                "sample": gap.get("sample"),
                "square": gap.get("square"),
                "expected": gap.get("expected"),
                "expected_jp": gap.get("expected_jp"),
                "predicted_top1": gap.get("predicted_top1"),
                "predicted_top1_jp": gap.get("predicted_top1_jp"),
                "expected_candidate_rank": gap.get("expected_candidate_rank"),
                "score_gap_top_minus_expected": gap.get("score_gap_top_minus_expected"),
                "expected_report_source": gap.get("expected_source"),
                "top_report_source": gap.get("top_source"),
                "expected_asset_template_count": expected_supply["asset_template_count"],
                "expected_available_template_count": expected_supply["available_template_count"],
                "expected_excluded_template_count": expected_supply["excluded_template_count"],
                "expected_available_template_sources": expected_supply["available_template_sources"],
                "expected_excluded_template_sources": expected_supply["excluded_template_sources"],
                "predicted_asset_template_count": predicted_supply["asset_template_count"],
                "predicted_available_template_count": predicted_supply["available_template_count"],
                "predicted_excluded_template_count": predicted_supply["excluded_template_count"],
                "predicted_available_template_sources": predicted_supply["available_template_sources"],
                "predicted_excluded_template_sources": predicted_supply["excluded_template_sources"],
            }
        )
    return rows


def collect_hand_error_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        hands = result.get("hands") or {}
        for error in hands.get("error_details") or []:
            owner = error.get("owner")
            piece = error.get("piece")
            rows.append(
                {
                    "app": result.get("app"),
                    "piece_style": result.get("piece_style"),
                    "sample": result.get("sample"),
                    "owner": owner,
                    "owner_jp": COLOR_JP.get(str(owner), str(owner)),
                    "piece": piece,
                    "piece_jp": PIECE_JP.get(str(piece), str(piece)),
                    "expected": error.get("expected"),
                    "actual": error.get("actual"),
                    "diff_actual_minus_expected": int(error.get("actual") or 0) - int(error.get("expected") or 0),
                }
            )
    return rows


def collect_leak_error_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for error in result.get("leak_errors") or []:
            rows.append(
                {
                    "app": result.get("app"),
                    "piece_style": result.get("piece_style"),
                    "sample": result.get("sample"),
                    "leak_error": error,
                }
            )
    return rows


def strict_failure_reasons(
    result: dict[str, Any],
    require_perfect: bool,
    fail_on_skipped: bool,
    require_speed: bool,
    fail_on_missing_timing: bool,
    fail_on_leak: bool = False,
) -> list[str]:
    reasons: list[str] = []
    overall = result.get("overall") or {}
    skipped = int(overall.get("skipped_samples") or 0)
    if skipped and (require_perfect or fail_on_skipped):
        reasons.append(f"skipped_samples={skipped}")
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
            if int(overall.get(key) or 0) != 0:
                reasons.append(f"{key}={overall.get(key)}")
    elif fail_on_leak and int(overall.get("leak_errors") or 0) != 0:
        reasons.append(f"leak_errors={overall.get('leak_errors')}")
    timing = overall.get("timing") or {}
    over_limit = int(timing.get("over_limit_count") or 0)
    missing_timing = int(timing.get("missing_seconds_count") or 0)
    if require_speed and over_limit:
        reasons.append(f"over_limit_count={over_limit}")
    if (require_speed or fail_on_missing_timing) and missing_timing:
        reasons.append(f"missing_seconds_count={missing_timing}")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grouped app/piece-style analysis output.")
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--include-hands", action="store_true")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--strict-leak-guard", action="store_true")
    parser.add_argument("--allow-missing-excluded-source", action="store_true")
    parser.add_argument("--fail-on-skipped", action="store_true", help="Return non-zero if any sample is skipped, including missing labels.")
    parser.add_argument("--fail-on-missing-timing", action="store_true", help="Return non-zero if any manifest row lacks a valid seconds value.")
    parser.add_argument("--require-speed", action="store_true", help="Return non-zero if any valid timing exceeds --max-seconds; missing timing also fails.")
    parser.add_argument("--require-perfect", action="store_true", help="Return non-zero unless board, hands, leak, and high-confidence error metrics are all zero.")
    args = parser.parse_args()

    labels_dir = args.labels_dir or (args.analysis_dir / "_review_labels")
    result = evaluate_manifest(
        args.analysis_dir,
        labels_dir,
        args.include_hands,
        args.high_confidence_threshold,
        args.max_seconds,
        args.strict_leak_guard,
        not args.allow_missing_excluded_source,
    )

    write_json(args.analysis_dir / "piece_style_evaluation_summary.json", result)
    group_fields = [
        "app",
        "piece_style",
        "manifest_images",
        "evaluated_samples",
        "skipped_samples",
        "labeled",
        "initial",
        "unlabeled",
        "max_seconds",
        "max_observed_seconds",
        "over_limit_count",
        "confirmed_identity_accuracy",
        "top1_identity_accuracy",
        "top3_contains_identity_accuracy",
        "errors",
        "high_confidence_errors",
        "hand_errors",
        "leak_errors",
        "visual_review_html",
    ]
    group_rows = []
    for group in result["groups"]:
        timing = group.get("timing") or {}
        row = dict(group)
        row["max_seconds"] = timing.get("max_seconds")
        row["max_observed_seconds"] = timing.get("max_observed_seconds")
        row["over_limit_count"] = timing.get("over_limit_count")
        group_rows.append({field: row.get(field, "") for field in group_fields})
    write_csv(args.analysis_dir / "piece_style_evaluation_summary.csv", group_rows, group_fields)

    sample_fields = [
        "app",
        "piece_style",
        "sample",
        "seconds",
        "errors",
        "high_confidence_errors",
        "hand_errors",
        "leak_errors",
        "confirmed_identity_accuracy",
        "top1_identity_accuracy",
        "top3_contains_identity_accuracy",
    ]
    sample_rows = []
    for item in result["results"]:
        metrics = item.get("metrics") or {}
        sample_rows.append(
            {
                "app": item.get("app"),
                "piece_style": item.get("piece_style"),
                "sample": item.get("sample"),
                "seconds": item.get("seconds"),
                "errors": metrics.get("errors"),
                "high_confidence_errors": metrics.get("high_confidence_errors"),
                "hand_errors": metrics.get("hand_errors"),
                "leak_errors": metrics.get("leak_errors"),
                "confirmed_identity_accuracy": metrics.get("confirmed_identity_accuracy"),
                "top1_identity_accuracy": metrics.get("top1_identity_accuracy"),
                "top3_contains_identity_accuracy": metrics.get("top3_contains_identity_accuracy"),
            }
        )
    write_csv(args.analysis_dir / "piece_style_evaluation_samples.csv", sample_rows, sample_fields)
    write_json(args.analysis_dir / "piece_style_evaluation_skipped.json", result["skipped"])

    board_error_fields = [
        "app",
        "piece_style",
        "sample",
        "square",
        "expected",
        "expected_jp",
        "predicted_top1",
        "predicted_top1_jp",
        "confirmed",
        "confirmed_jp",
        "actual_state",
        "confidence",
        "top3_jp",
    ]
    write_csv(
        args.analysis_dir / "piece_style_board_errors.csv",
        collect_board_error_rows(result["results"]),
        board_error_fields,
    )
    board_error_candidate_fields = [
        "app",
        "piece_style",
        "sample",
        "square",
        "row",
        "col",
        "expected",
        "expected_jp",
        "predicted_top1",
        "predicted_top1_jp",
        "confirmed",
        "confirmed_jp",
        "actual_state",
        "confidence",
        "candidate_set",
        "expected_in_candidate_set",
        "expected_candidate_rank",
        "candidate_index",
        "candidate_rank",
        "candidate_identity",
        "candidate_jp",
        "is_expected",
        "is_predicted_top1",
        "score",
        "source",
        "shape",
        "raw_dice",
        "raw_iou",
        "clean_dice",
        "clean_iou",
        "clean_area",
        "clean_shape",
        "bbox",
        "center",
        "density",
        "red",
        "position_boost",
        "template_position_boost",
        "position_prior_boost",
        "score_without_template_position",
        "score_without_position_prior",
        "score_without_position_boost",
        "unweighted_template_score",
        "template_weight_contribution",
        "base_template_score",
        "weighted_template_score",
        "exact_template_weight_contribution",
        "score_formula_residual",
        "template_weight",
        "source_bbox_width",
        "source_bbox_height",
        "target_bbox_width",
        "target_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
        "target_bbox_center_x",
        "target_bbox_center_y",
        "source_ink_ratio",
        "target_ink_ratio",
        "source_clean_ink_ratio",
        "target_clean_ink_ratio",
        "source_ink_count",
        "target_ink_count",
        "source_clean_ink_count",
        "target_clean_ink_count",
        *COMPONENT_GLYPH_DIAGNOSTIC_FIELDS,
        "source_red_share",
        "target_red_share",
        "source_central_red_share",
        "source_edge_red_share",
        "target_central_red_share",
        "target_edge_red_share",
        "source_red_center_x",
        "source_red_center_y",
        "target_red_center_x",
        "target_red_center_y",
    ]
    board_error_candidate_rows = collect_board_error_candidate_rows(result["results"])
    write_csv(
        args.analysis_dir / "piece_style_board_error_candidates.csv",
        board_error_candidate_rows,
        board_error_candidate_fields,
    )
    board_error_candidate_gap_fields = [
        "app",
        "piece_style",
        "sample",
        "square",
        "row",
        "col",
        "expected",
        "expected_jp",
        "predicted_top1",
        "predicted_top1_jp",
        "confirmed",
        "confirmed_jp",
        "actual_state",
        "confidence",
        "candidate_set",
        "expected_in_candidate_set",
        "expected_candidate_rank",
        "top_identity",
        "top_jp",
        "top_score",
        "top_source",
        "runner_up_identity",
        "runner_up_jp",
        "runner_up_score",
        "runner_up_source",
        "expected_score",
        "expected_source",
        "score_gap_top_minus_expected",
        "rank_gap_top_to_expected",
    ]
    for field in [
        "shape",
        "raw_dice",
        "clean_shape",
        "bbox",
        "center",
        "density",
        "red",
        "position_boost",
        "template_position_boost",
        "position_prior_boost",
        "score_without_template_position",
        "score_without_position_prior",
        "score_without_position_boost",
        "unweighted_template_score",
        "template_weight_contribution",
        "base_template_score",
        "weighted_template_score",
        "exact_template_weight_contribution",
        "score_formula_residual",
        "source_bbox_width",
        "source_bbox_height",
        "target_bbox_width",
        "target_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
        "target_bbox_center_x",
        "target_bbox_center_y",
        "source_ink_ratio",
        "target_ink_ratio",
        "source_clean_ink_ratio",
        "target_clean_ink_ratio",
        "source_ink_count",
        "target_ink_count",
        "source_clean_ink_count",
        "target_clean_ink_count",
        *COMPONENT_GLYPH_DIAGNOSTIC_FIELDS,
    ]:
        board_error_candidate_gap_fields += [
            f"top_{field}",
            f"expected_{field}",
            f"delta_{field}",
        ]
    board_error_candidate_gap_rows = collect_board_error_candidate_gap_rows(board_error_candidate_rows)
    write_csv(
        args.analysis_dir / "piece_style_board_error_candidate_gaps.csv",
        board_error_candidate_gap_rows,
        board_error_candidate_gap_fields,
    )
    board_error_candidate_gap_summary_fields = [
        "app",
        "piece_style",
        "predicted_top1",
        "predicted_top1_jp",
        "expected",
        "expected_jp",
        "count",
        "avg_score_gap_top_minus_expected",
        "avg_delta_shape",
        "avg_delta_raw_dice",
        "avg_delta_clean_shape",
        "avg_delta_bbox",
        "avg_delta_center",
        "avg_delta_density",
        "avg_delta_red",
        "avg_delta_position_boost",
        "avg_delta_template_position_boost",
        "avg_delta_position_prior_boost",
        "avg_delta_score_without_template_position",
        "avg_delta_score_without_position_prior",
        "avg_delta_score_without_position_boost",
        "avg_delta_unweighted_template_score",
        "avg_delta_template_weight_contribution",
        "avg_delta_base_template_score",
        "avg_delta_weighted_template_score",
        "avg_delta_exact_template_weight_contribution",
        "avg_delta_score_formula_residual",
        "avg_delta_source_bbox_width",
        "avg_delta_source_bbox_height",
        "avg_delta_target_bbox_width",
        "avg_delta_target_bbox_height",
        "avg_delta_source_bbox_center_x",
        "avg_delta_source_bbox_center_y",
        "avg_delta_target_bbox_center_x",
        "avg_delta_target_bbox_center_y",
        "avg_delta_source_ink_ratio",
        "avg_delta_target_ink_ratio",
        "avg_delta_source_clean_ink_ratio",
        "avg_delta_target_clean_ink_ratio",
        "avg_delta_source_ink_count",
        "avg_delta_target_ink_count",
        "avg_delta_source_clean_ink_count",
        "avg_delta_target_clean_ink_count",
        *[f"avg_delta_{field}" for field in COMPONENT_GLYPH_DIAGNOSTIC_FIELDS],
        "expected_rank_min",
        "expected_rank_avg",
        "expected_rank_max",
    ]
    write_csv(
        args.analysis_dir / "piece_style_board_error_candidate_gap_summary.csv",
        collect_board_error_candidate_gap_summary_rows(board_error_candidate_gap_rows),
        board_error_candidate_gap_summary_fields,
    )
    board_error_template_supply_fields = [
        "app",
        "piece_style",
        "preset",
        "sample",
        "square",
        "expected",
        "expected_jp",
        "predicted_top1",
        "predicted_top1_jp",
        "expected_candidate_rank",
        "score_gap_top_minus_expected",
        "expected_report_source",
        "top_report_source",
        "expected_asset_template_count",
        "expected_available_template_count",
        "expected_excluded_template_count",
        "expected_available_template_sources",
        "expected_excluded_template_sources",
        "predicted_asset_template_count",
        "predicted_available_template_count",
        "predicted_excluded_template_count",
        "predicted_available_template_sources",
        "predicted_excluded_template_sources",
    ]
    write_csv(
        args.analysis_dir / "piece_style_board_error_template_supply.csv",
        collect_board_error_template_supply_rows(board_error_candidate_gap_rows, load_template_assets()),
        board_error_template_supply_fields,
    )
    hand_error_fields = [
        "app",
        "piece_style",
        "sample",
        "owner",
        "owner_jp",
        "piece",
        "piece_jp",
        "expected",
        "actual",
        "diff_actual_minus_expected",
    ]
    write_csv(
        args.analysis_dir / "piece_style_hand_errors.csv",
        collect_hand_error_rows(result["results"]),
        hand_error_fields,
    )
    leak_error_fields = [
        "app",
        "piece_style",
        "sample",
        "leak_error",
    ]
    write_csv(
        args.analysis_dir / "piece_style_leak_errors.csv",
        collect_leak_error_rows(result["results"]),
        leak_error_fields,
    )

    print(
        json.dumps(
            {
                "overall": result["overall"],
                "groups": [
                    {
                        "app": group["app"],
                        "piece_style": group["piece_style"],
                        "evaluated_samples": group["evaluated_samples"],
                        "errors": group.get("errors"),
                        "hand_errors": group.get("hand_errors"),
                        "leak_errors": group.get("leak_errors"),
                        "max_seconds": (group.get("timing") or {}).get("max_seconds"),
                        "max_observed_seconds": (group.get("timing") or {}).get("max_observed_seconds"),
                        "over_limit_count": (group.get("timing") or {}).get("over_limit_count"),
                    }
                    for group in result["groups"]
                ],
                "skipped": result["skipped"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    failure_reasons = strict_failure_reasons(
        result,
        args.require_perfect,
        args.fail_on_skipped,
        args.require_speed,
        args.fail_on_missing_timing,
        args.strict_leak_guard,
    )
    if failure_reasons:
        print("Evaluation failed strict checks: " + ", ".join(failure_reasons), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
