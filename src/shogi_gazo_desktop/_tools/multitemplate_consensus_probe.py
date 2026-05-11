from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from chamfer_offline_probe import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_SCREENSHOTS_DIR,
    DEFAULT_TEMPLATE_ASSET,
    FEATURE_HEIGHT,
    FEATURE_SIZE,
    FEATURE_WIDTH,
    cell_key,
    crop_cell_image,
    decode_hex_mask,
    dilate,
    empty_source_features,
    extract_piece_masks,
    load_report_cells,
    load_reports_by_sample,
    mask_bounds,
    parse_float,
    parse_int,
    read_csv,
    round_float,
    write_csv,
)


PROMOTED_KINDS = {"TO", "NY", "NK", "NG", "UM", "RY"}
PIYO_RED_SENSITIVE_PROMOTED_KINDS = {"RY", "UM", "NY"}
CELL_KEY_FIELDS = ("app", "piece_style", "sample", "square", "row", "col")


def normalize_identity(color: str, piece: str) -> str:
    return f"{str(color or '').lower()}:{piece}"


def similarity_by_distance(left: float, right: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 1.0 if abs(left - right) <= 0.0001 else 0.0
    return max(0.0, min(1.0, 1.0 - abs(left - right) / tolerance))


def mask_similarity(left: list[bool], right: list[bool], left_count: int, right_count: int) -> dict[str, float]:
    intersection = sum(1 for left_ink, right_ink in zip(left, right) if left_ink and right_ink)
    total = left_count + right_count
    if total == 0:
        return {"dice": 0.0, "iou": 0.0}
    union = max(1, total - intersection)
    return {
        "dice": max(0.0, min(1.0, 2.0 * intersection / total)),
        "iou": max(0.0, min(1.0, intersection / union)),
    }


def red_location_stats(mask: list[bool]) -> dict[str, float]:
    red_pixels = 0
    central_pixels = 0
    edge_pixels = 0
    sum_x = 0.0
    sum_y = 0.0
    for index, is_red in enumerate(mask):
        if not is_red:
            continue
        x = index % FEATURE_WIDTH
        y = index // FEATURE_WIDTH
        normalized_x = (x + 0.5) / FEATURE_WIDTH
        normalized_y = (y + 0.5) / FEATURE_HEIGHT
        red_pixels += 1
        sum_x += normalized_x
        sum_y += normalized_y
        if 0.28 <= normalized_x <= 0.72 and 0.18 <= normalized_y <= 0.84:
            central_pixels += 1
        if normalized_x < 0.18 or normalized_x > 0.82 or normalized_y < 0.12 or normalized_y > 0.88:
            edge_pixels += 1
    if red_pixels <= 0:
        return {"red_center_x": 0.5, "red_center_y": 0.5, "central_red_share": 0.0, "edge_red_share": 0.0}
    return {
        "red_center_x": max(0.0, min(1.0, sum_x / red_pixels)),
        "red_center_y": max(0.0, min(1.0, sum_y / red_pixels)),
        "central_red_share": max(0.0, min(1.0, central_pixels / red_pixels)),
        "edge_red_share": max(0.0, min(1.0, edge_pixels / red_pixels)),
    }


def load_templates(path: Path, preset: str = "piyo_chick") -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    templates: list[dict[str, Any]] = []
    for item in data.get("templates") or []:
        if not isinstance(item, dict) or item.get("preset") != preset:
            continue
        raw_mask = decode_hex_mask(str(item.get("mask") or ""))
        ink_mask = dilate(raw_mask)
        if not any(ink_mask):
            continue
        raw_red_mask = decode_hex_mask(str(item.get("redMask") or "")) if item.get("redMask") else [False] * FEATURE_SIZE
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else None
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [parse_int(value) for value in bbox[:4]]
            x1 = min(max(0, x1), FEATURE_WIDTH - 1)
            y1 = min(max(0, y1), FEATURE_HEIGHT - 1)
            x2 = min(max(x1 + 1, x2), FEATURE_WIDTH)
            y2 = min(max(y1 + 1, y2), FEATURE_HEIGHT)
        else:
            x1, y1, x2, y2 = mask_bounds(ink_mask, FEATURE_WIDTH, FEATURE_HEIGHT)
        red_stats = red_location_stats(raw_red_mask)
        red_share = parse_float(item.get("redShare"), 0.0)
        red_count = sum(raw_red_mask) if any(raw_red_mask) else int(sum(ink_mask) * red_share)
        templates.append(
            {
                "preset": item.get("preset"),
                "identity": normalize_identity(str(item.get("color") or ""), str(item.get("piece") or "")),
                "color": item.get("color"),
                "piece": item.get("piece"),
                "source": item.get("source") or "",
                "row": item.get("row"),
                "col": item.get("col"),
                "ink_mask": ink_mask,
                "clean_mask": ink_mask,
                "red_mask": raw_red_mask,
                "ink_count": sum(ink_mask),
                "clean_ink_count": sum(ink_mask),
                "red_ink_count": red_count,
                "original_ink_ratio": parse_float(item.get("darkRatio"), 0.01),
                "red_share": red_share,
                "bbox_width_ratio": (x2 - x1) / FEATURE_WIDTH,
                "bbox_height_ratio": (y2 - y1) / FEATURE_HEIGHT,
                "bbox_center_x": (x1 + x2) / (2 * FEATURE_WIDTH),
                "bbox_center_y": (y1 + y2) / (2 * FEATURE_HEIGHT),
                "red_center_x": parse_float(item.get("redCenterX"), red_stats["red_center_x"]),
                "red_center_y": parse_float(item.get("redCenterY"), red_stats["red_center_y"]),
                "central_red_share": parse_float(item.get("centralRedShare"), red_stats["central_red_share"]),
                "edge_red_share": parse_float(item.get("edgeRedShare"), red_stats["edge_red_share"]),
            }
        )
    return templates


def source_features_for_cell(source_masks: dict[str, Any], source_row: dict[str, str]) -> dict[str, Any]:
    features = dict(source_masks)
    features.update(
        {
            "original_ink_ratio": parse_float(source_row.get("source_ink_ratio"), features.get("original_ink_ratio", 0.0)),
            "red_share": parse_float(source_row.get("source_red_share"), features.get("red_share", 0.0)),
            "bbox_width_ratio": parse_float(source_row.get("source_bbox_width"), features.get("bbox_width_ratio", 0.0)),
            "bbox_height_ratio": parse_float(source_row.get("source_bbox_height"), features.get("bbox_height_ratio", 0.0)),
            "bbox_center_x": parse_float(source_row.get("source_bbox_center_x"), features.get("bbox_center_x", 0.5)),
            "bbox_center_y": parse_float(source_row.get("source_bbox_center_y"), features.get("bbox_center_y", 0.5)),
            "red_center_x": parse_float(source_row.get("source_red_center_x"), 0.5),
            "red_center_y": parse_float(source_row.get("source_red_center_y"), 0.5),
            "central_red_share": parse_float(source_row.get("source_central_red_share"), 0.0),
            "edge_red_share": parse_float(source_row.get("source_edge_red_share"), 0.0),
        }
    )
    return features


def has_enough_ink(features: dict[str, Any]) -> bool:
    return parse_int(features.get("ink_count"), 0) >= 10 and parse_float(features.get("original_ink_ratio"), 0.0) >= 0.006


def looks_like_piyo_promoted_glyph(features: dict[str, Any]) -> bool:
    return (
        parse_float(features.get("red_share")) >= 0.80
        and parse_float(features.get("central_red_share")) >= 0.18
        and parse_float(features.get("edge_red_share")) <= 0.72
    )


def should_skip_piyo_chick_template(source: dict[str, Any], template: dict[str, Any]) -> bool:
    target_red_share = parse_float(template.get("red_share"))
    if looks_like_piyo_promoted_glyph(source) and template.get("piece") not in PROMOTED_KINDS and target_red_share < 0.05:
        return True
    if (
        parse_float(source.get("red_share")) < 0.012
        and template.get("piece") in PIYO_RED_SENSITIVE_PROMOTED_KINDS
        and target_red_share > 0.03
    ):
        return True
    return False


def piyo_chick_red_compatibility(source: dict[str, Any], target: dict[str, Any], kind: str) -> float:
    promoted = kind in PROMOTED_KINDS
    score = similarity_by_distance(parse_float(source.get("red_share")), parse_float(target.get("red_share")), 0.16)
    source_edge_decoration = (
        parse_float(source.get("red_share")) >= 0.45
        and parse_float(source.get("central_red_share")) <= 0.13
        and parse_float(source.get("edge_red_share")) >= 0.58
    )
    source_promoted_glyph = looks_like_piyo_promoted_glyph(source)
    target_promoted_glyph = (
        parse_float(target.get("red_share")) >= 0.45
        and parse_float(target.get("central_red_share")) >= 0.55
        and parse_float(target.get("edge_red_share")) <= 0.25
    )
    if promoted and source_edge_decoration and target_promoted_glyph:
        score = min(score, 0.12)
    elif promoted and source_promoted_glyph and target_promoted_glyph:
        score += 0.08
    elif promoted and parse_float(source.get("red_share")) < 0.012 and parse_float(target.get("red_share")) > 0.03:
        score -= 0.12
    elif not promoted and source_promoted_glyph and parse_float(target.get("red_share")) < 0.05:
        score -= 0.24
    elif not promoted and parse_float(source.get("red_share")) > 0.055:
        score -= 0.18
    return max(0.0, min(1.0, score))


def template_weight(template: dict[str, Any]) -> float:
    return 1.085 if template.get("preset") == "piyo_chick" else 0.935


def template_position_boost(
    template: dict[str, Any],
    row: int,
    col: int,
    base_score: float,
    red_score: float,
) -> tuple[float, str]:
    if template.get("row") != row or template.get("col") != col:
        return 0.0, ""
    if template.get("piece") in PROMOTED_KINDS and red_score < 0.20:
        return 0.0, ""
    base = 0.118 if template.get("piece") == "GI" else 0.105
    return base, "template_position"


def score_template(source: dict[str, Any], template: dict[str, Any], row: int, col: int) -> dict[str, Any] | None:
    if should_skip_piyo_chick_template(source, template) or not has_enough_ink(source) or not has_enough_ink(template):
        return None
    raw_similarity = mask_similarity(
        source.get("ink_mask") or [False] * FEATURE_SIZE,
        template.get("ink_mask") or [False] * FEATURE_SIZE,
        parse_int(source.get("ink_count"), 0),
        parse_int(template.get("ink_count"), 0),
    )
    clean_similarity = mask_similarity(
        source.get("clean_mask") or [False] * FEATURE_SIZE,
        template.get("clean_mask") or [False] * FEATURE_SIZE,
        parse_int(source.get("clean_ink_count"), 0),
        parse_int(template.get("clean_ink_count"), 0),
    )
    clean_area_score = similarity_by_distance(
        parse_int(source.get("clean_ink_count"), 0) / FEATURE_SIZE,
        parse_int(template.get("clean_ink_count"), 0) / FEATURE_SIZE,
        0.18,
    )
    clean_score = max(
        0.0,
        min(1.0, clean_similarity["dice"] * 0.58 + clean_similarity["iou"] * 0.30 + clean_area_score * 0.12),
    )
    bbox_score = (
        similarity_by_distance(parse_float(source.get("bbox_width_ratio")), parse_float(template.get("bbox_width_ratio")), 0.34)
        + similarity_by_distance(parse_float(source.get("bbox_height_ratio")), parse_float(template.get("bbox_height_ratio")), 0.40)
    ) / 2
    center_score = (
        similarity_by_distance(parse_float(source.get("bbox_center_x")), parse_float(template.get("bbox_center_x")), 0.24)
        + similarity_by_distance(parse_float(source.get("bbox_center_y")), parse_float(template.get("bbox_center_y")), 0.28)
    ) / 2
    density_score = similarity_by_distance(
        parse_float(source.get("original_ink_ratio")),
        parse_float(template.get("original_ink_ratio")),
        0.15,
    )
    red_score = piyo_chick_red_compatibility(source, template, str(template.get("piece") or ""))
    red_mismatch_penalty = 0.035 if template.get("piece") in PIYO_RED_SENSITIVE_PROMOTED_KINDS and red_score <= 0.05 else 0.0
    base_template_score = max(
        0.0,
        min(
            1.0,
            raw_similarity["dice"] * 0.42
            + clean_score * 0.31
            + bbox_score * 0.10
            + center_score * 0.05
            + density_score * 0.05
            + red_score * 0.07
            - red_mismatch_penalty,
        ),
    )
    weight = template_weight(template)
    position_boost, position_source = template_position_boost(template, row, col, base_template_score, red_score)
    weighted_score = base_template_score * weight
    final_score = max(0.0, min(1.0, weighted_score + position_boost))
    return {
        "identity": template["identity"],
        "template_source": template.get("source") or "",
        "template_row": template.get("row") or "",
        "template_col": template.get("col") or "",
        "base_template_score": base_template_score,
        "weighted_template_score": weighted_score,
        "template_position_boost": position_boost,
        "template_position_source": position_source,
        "template_final_score": final_score,
        "shape": max(0.0, min(1.0, raw_similarity["dice"] * 0.58 + clean_score * 0.42)),
        "raw_dice": raw_similarity["dice"],
        "raw_iou": raw_similarity["iou"],
        "clean_dice": clean_similarity["dice"],
        "clean_iou": clean_similarity["iou"],
        "clean_shape": clean_score,
        "bbox": bbox_score,
        "center": center_score,
        "density": density_score,
        "red": red_score,
        "template_ink_count": template.get("ink_count"),
        "template_clean_ink_count": template.get("clean_ink_count"),
        "template_red_share": template.get("red_share"),
    }


def top_mean(values: list[float], count: int) -> float:
    if not values:
        return 0.0
    return mean(values[: min(count, len(values))])


def consensus_metrics(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "template_count": 0,
            "source_count": 0,
            "unique_template_position_count": 0,
            "best_template_score": 0.0,
            "second_template_score": 0.0,
            "top2_mean_score": 0.0,
            "top3_mean_score": 0.0,
            "top3_trimmed_mean_excluding_best": 0.0,
            "top5_mean_score": 0.0,
            "mean_template_score": 0.0,
            "median_template_score": 0.0,
            "template_score_std": 0.0,
            "best_to_top3_mean_gap": 0.0,
            "support_count_within_010": 0,
            "support_count_within_020": 0,
            "support_count_within_030": 0,
            "support_count_ge_034": 0,
            "support_count_ge_036": 0,
            "support_count_ge_038": 0,
            "support_count_ge_040": 0,
            "source_count_ge_034": 0,
            "source_count_ge_036": 0,
            "source_entropy_count": 0.0,
            "source_entropy_weighted": 0.0,
            "source_diversity_top2_mean_score": 0.0,
            "source_diversity_top3_mean_score": 0.0,
            "soft_support_mass_025": 0.0,
            "support_bonus_score": 0.0,
            "source_diversity_bonus_score": 0.0,
        }
    ordered = sorted((parse_float(row["weighted_template_score"]) for row in scores), reverse=True)
    best = ordered[0]
    per_source_best: dict[str, float] = {}
    source_counts: dict[str, int] = defaultdict(int)
    source_soft_weights: dict[str, float] = defaultdict(float)
    positions = set()
    for row in scores:
        source = str(row.get("template_source") or "")
        score = parse_float(row.get("weighted_template_score"))
        per_source_best[source] = max(per_source_best.get(source, 0.0), score)
        source_counts[source] += 1
        source_soft_weights[source] += math.exp((score - best) / 0.025)
        position = (row.get("template_row"), row.get("template_col"))
        if position != ("", ""):
            positions.add(position)
    source_values = sorted(per_source_best.values(), reverse=True)
    soft_mass = sum(math.exp((score - best) / 0.025) for score in ordered)
    support_within_020 = sum(1 for score in ordered if score >= best - 0.020)
    support_ge_036 = sum(1 for score in ordered if score >= 0.36)
    source_ge_034 = sum(1 for score in source_values if score >= 0.34)
    source_ge_036 = sum(1 for score in source_values if score >= 0.36)
    support_bonus = (
        best
        + 0.018 * math.log1p(max(0, support_within_020 - 1)) / math.log1p(5)
        + 0.012 * math.log1p(source_ge_034) / math.log1p(6)
        + 0.010 * math.log1p(max(0.0, soft_mass - 1.0)) / math.log1p(6)
    )
    source_diversity_bonus = (
        top_mean(source_values, 3)
        + 0.012 * math.log1p(source_ge_036) / math.log1p(6)
        + 0.008 * math.log1p(max(0, support_ge_036 - 1)) / math.log1p(6)
    )
    score_mean = mean(ordered)
    score_std = math.sqrt(sum((score - score_mean) ** 2 for score in ordered) / len(ordered))
    source_entropy_count = normalized_entropy([float(value) for value in source_counts.values()])
    source_entropy_weighted = normalized_entropy(list(source_soft_weights.values()))
    return {
        "template_count": len(scores),
        "source_count": len(per_source_best),
        "unique_template_position_count": len(positions),
        "best_template_score": best,
        "second_template_score": ordered[1] if len(ordered) >= 2 else 0.0,
        "top2_mean_score": top_mean(ordered, 2),
        "top3_mean_score": top_mean(ordered, 3),
        "top3_trimmed_mean_excluding_best": mean(ordered[1:3]) if len(ordered) >= 3 else (ordered[1] if len(ordered) >= 2 else 0.0),
        "top5_mean_score": top_mean(ordered, 5),
        "mean_template_score": score_mean,
        "median_template_score": sorted(ordered)[len(ordered) // 2],
        "template_score_std": score_std,
        "best_to_top3_mean_gap": best - top_mean(ordered, 3),
        "support_count_within_010": sum(1 for score in ordered if score >= best - 0.010),
        "support_count_within_020": support_within_020,
        "support_count_within_030": sum(1 for score in ordered if score >= best - 0.030),
        "support_count_ge_034": sum(1 for score in ordered if score >= 0.34),
        "support_count_ge_036": support_ge_036,
        "support_count_ge_038": sum(1 for score in ordered if score >= 0.38),
        "support_count_ge_040": sum(1 for score in ordered if score >= 0.40),
        "source_count_ge_034": source_ge_034,
        "source_count_ge_036": source_ge_036,
        "source_entropy_count": source_entropy_count,
        "source_entropy_weighted": source_entropy_weighted,
        "source_diversity_top2_mean_score": top_mean(source_values, 2),
        "source_diversity_top3_mean_score": top_mean(source_values, 3),
        "soft_support_mass_025": soft_mass,
        "support_bonus_score": support_bonus,
        "source_diversity_bonus_score": source_diversity_bonus,
    }


def normalized_entropy(values: list[float]) -> float:
    total = sum(value for value in values if value > 0)
    if total <= 0 or len(values) <= 1:
        return 0.0
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(values))


def rank_rows(rows: list[dict[str, Any]], score_field: str, rank_field: str) -> None:
    for rank, row in enumerate(sorted(rows, key=lambda item: parse_float(item.get(score_field), -1.0), reverse=True), start=1):
        row[rank_field] = rank


def load_excluded_sources(report_path: Path | None) -> set[str]:
    if report_path is None:
        return set()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    output: set[str] = set()
    excluded_source = model.get("excluded_source")
    if isinstance(excluded_source, str) and excluded_source:
        output.add(excluded_source)
    excluded_sources = model.get("excluded_sources")
    if isinstance(excluded_sources, list):
        output.update(str(source) for source in excluded_sources if source)
    options = model.get("no_leak_options") if isinstance(model.get("no_leak_options"), dict) else {}
    option_source = options.get("excludedTemplateSource")
    if isinstance(option_source, str) and option_source:
        output.add(option_source)
    return output


def average_rank(rows: list[dict[str, Any]], field: str) -> float:
    values = [parse_float(row.get(field), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else math.nan


def run_probe(
    analysis_dir: Path,
    screenshots_dir: Path,
    template_asset: Path,
    out_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    candidate_path = analysis_dir / "piece_style_board_error_candidates.csv"
    gap_path = analysis_dir / "piece_style_board_error_candidate_gaps.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    if not gap_path.exists():
        raise FileNotFoundError(gap_path)

    candidate_rows = [
        row
        for row in read_csv(candidate_path)
        if row.get("app") == "ぴよ将棋" and row.get("piece_style") == "ひよこ駒" and parse_int(row.get("candidate_rank"), 9999) <= top_n
    ]
    templates = load_templates(template_asset)
    reports_by_sample = load_reports_by_sample(analysis_dir)
    gaps_by_cell = {cell_key(row): row for row in read_csv(gap_path)}
    cells_cache: dict[Path, dict[tuple[int, int], dict[str, Any]]] = {}
    exclusions_cache: dict[Path, set[str]] = {}
    source_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    excluded_sources_by_cell: dict[tuple[str, ...], set[str]] = {}
    source_rows: dict[tuple[str, ...], dict[str, str]] = {}
    for row in candidate_rows:
        source_rows.setdefault(cell_key(row), row)

    template_score_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    grouped_candidates: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[cell_key(row)].append(row)

    for key, rows in sorted(grouped_candidates.items()):
        first = source_rows[key]
        report_path = reports_by_sample.get(str(first.get("sample")))
        if report_path is not None and report_path not in exclusions_cache:
            exclusions_cache[report_path] = load_excluded_sources(report_path)
        excluded_sources_by_cell[key] = exclusions_cache.get(report_path, set()) if report_path is not None else set()
        if key not in source_cache:
            cell: dict[str, Any] | None = None
            if report_path is not None:
                if report_path not in cells_cache:
                    cells_cache[report_path] = load_report_cells(report_path)
                cell = cells_cache[report_path].get((parse_int(first.get("row")), parse_int(first.get("col"))))
            crop = crop_cell_image(screenshots_dir, first, cell or {}) if cell is not None else None
            masks = extract_piece_masks(crop) if crop is not None else empty_source_features()
            source_cache[key] = source_features_for_cell(masks, first)
        source = source_cache[key]
        row_number = parse_int(first.get("row"))
        col_number = parse_int(first.get("col"))
        scores_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for template in templates:
            if template.get("source") in excluded_sources_by_cell[key]:
                continue
            scored = score_template(source, template, row_number, col_number)
            if scored is None:
                continue
            scores_by_identity[str(scored["identity"])].append(scored)
            template_score_rows.append(
                {
                    **{field: first.get(field, "") for field in CELL_KEY_FIELDS},
                    "expected": first.get("expected"),
                    "predicted_top1": first.get("predicted_top1"),
                    **{name: round_float(value) if isinstance(value, float) else value for name, value in scored.items()},
                }
            )

        for candidate in sorted(rows, key=lambda row: parse_int(row.get("candidate_rank"), 9999)):
            identity = str(candidate.get("candidate_identity") or "")
            template_scores = scores_by_identity.get(identity, [])
            metrics = consensus_metrics(template_scores)
            best_template = max(template_scores, key=lambda item: parse_float(item.get("weighted_template_score")), default={})
            best_final_template = max(template_scores, key=lambda item: parse_float(item.get("template_final_score")), default={})
            base_score = parse_float(candidate.get("score"))
            identity_rows.append(
                {
                    **{field: candidate.get(field, "") for field in CELL_KEY_FIELDS},
                    "excluded_sources": "|".join(sorted(excluded_sources_by_cell[key])),
                    "expected": candidate.get("expected"),
                    "predicted_top1": candidate.get("predicted_top1"),
                    "candidate_index": candidate.get("candidate_index"),
                    "candidate_rank": candidate.get("candidate_rank"),
                    "candidate_identity": identity,
                    "is_expected": candidate.get("is_expected"),
                    "is_predicted_top1": candidate.get("is_predicted_top1"),
                    "base_score": candidate.get("score"),
                    "candidate_source": candidate.get("source"),
                    "best_template_source": best_template.get("template_source", ""),
                    "best_template_row": best_template.get("template_row", ""),
                    "best_template_col": best_template.get("template_col", ""),
                    "best_template_final_score": round_float(parse_float(best_final_template.get("template_final_score"), 0.0)),
                    "best_final_template_source": best_final_template.get("template_source", ""),
                    "best_final_template_row": best_final_template.get("template_row", ""),
                    "best_final_template_col": best_final_template.get("template_col", ""),
                    **{name: round_float(value) if isinstance(value, float) else value for name, value in metrics.items()},
                    "base_plus_support_bonus_score": round_float(base_score + parse_float(metrics["support_bonus_score"]) * 0.10),
                    "base_plus_source_diversity_score": round_float(base_score + parse_float(metrics["source_diversity_bonus_score"]) * 0.10),
                }
            )

    grouped_identity: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in identity_rows:
        grouped_identity[cell_key(row)].append(row)

    cell_summary_rows: list[dict[str, Any]] = []
    rank_specs = [
        ("top3_mean_score", "top3_mean_rank"),
        ("support_bonus_score", "support_bonus_rank"),
        ("source_diversity_bonus_score", "source_diversity_rank"),
        ("base_plus_support_bonus_score", "base_plus_support_bonus_rank"),
        ("base_plus_source_diversity_score", "base_plus_source_diversity_rank"),
    ]
    for key, rows in sorted(grouped_identity.items()):
        rows.sort(key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        for score_field, rank_field in rank_specs:
            rank_rows(rows, score_field, rank_field)
        base_top = min(rows, key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        expected = str(base_top.get("expected") or "")
        expected_row = next((row for row in rows if row.get("candidate_identity") == expected), None)
        gap = gaps_by_cell.get(key, {})
        summary = {
            **{field: base_top.get(field, "") for field in CELL_KEY_FIELDS},
            "expected": expected,
            "base_top1_identity": base_top.get("candidate_identity"),
            "base_top1_score": base_top.get("base_score"),
            "expected_base_rank": expected_row.get("candidate_rank") if expected_row else gap.get("expected_candidate_rank", ""),
            "expected_base_score": expected_row.get("base_score") if expected_row else "",
            "expected_in_topn": expected_row is not None,
            "topn": len(rows),
        }
        for score_field, rank_field in rank_specs:
            top = min(rows, key=lambda row: parse_int(row.get(rank_field), 9999))
            prefix = rank_field.removesuffix("_rank")
            summary[f"{prefix}_top1_identity"] = top.get("candidate_identity")
            summary[f"{prefix}_top1_score"] = top.get(score_field)
            summary[f"{prefix}_top1_is_expected"] = top.get("candidate_identity") == expected
            summary[f"expected_{rank_field}"] = expected_row.get(rank_field) if expected_row else ""
            summary[f"expected_{score_field}"] = expected_row.get(score_field) if expected_row else ""
        cell_summary_rows.append(summary)

    summary: dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "top_n": top_n,
        "candidate_rows": len(identity_rows),
        "template_score_rows": len(template_score_rows),
        "cell_rows": len(cell_summary_rows),
        "expected_in_topn_count": sum(1 for row in cell_summary_rows if row.get("expected_in_topn")),
        "base_errors_if_top1": len(cell_summary_rows),
        "expected_base_rank_avg": average_rank(cell_summary_rows, "expected_base_rank"),
    }
    for _score_field, rank_field in rank_specs:
        prefix = rank_field.removesuffix("_rank")
        summary[f"{prefix}_errors_if_top1"] = sum(1 for row in cell_summary_rows if not row.get(f"{prefix}_top1_is_expected"))
        summary[f"fixed_by_{prefix}_count"] = sum(1 for row in cell_summary_rows if row.get(f"{prefix}_top1_is_expected"))
        summary[f"expected_{rank_field}_avg"] = average_rank(cell_summary_rows, f"expected_{rank_field}")

    identity_fieldnames = [
            *CELL_KEY_FIELDS,
            "excluded_sources",
            "expected",
        "predicted_top1",
        "candidate_index",
        "candidate_rank",
        "candidate_identity",
        "is_expected",
        "is_predicted_top1",
        "base_score",
        "candidate_source",
        "best_template_source",
        "best_template_row",
        "best_template_col",
        "best_template_final_score",
        "best_final_template_source",
        "best_final_template_row",
        "best_final_template_col",
        "template_count",
        "source_count",
        "unique_template_position_count",
        "best_template_score",
        "second_template_score",
        "top2_mean_score",
        "top3_mean_score",
        "top3_trimmed_mean_excluding_best",
        "top5_mean_score",
        "mean_template_score",
        "median_template_score",
        "template_score_std",
        "best_to_top3_mean_gap",
        "support_count_within_010",
        "support_count_within_020",
        "support_count_within_030",
        "support_count_ge_034",
        "support_count_ge_036",
        "support_count_ge_038",
        "support_count_ge_040",
        "source_count_ge_034",
        "source_count_ge_036",
        "source_entropy_count",
        "source_entropy_weighted",
        "source_diversity_top2_mean_score",
        "source_diversity_top3_mean_score",
        "soft_support_mass_025",
        "support_bonus_score",
        "source_diversity_bonus_score",
        "base_plus_support_bonus_score",
        "base_plus_source_diversity_score",
        "top3_mean_rank",
        "support_bonus_rank",
        "source_diversity_rank",
        "base_plus_support_bonus_rank",
        "base_plus_source_diversity_rank",
    ]
    template_fieldnames = [
        *CELL_KEY_FIELDS,
        "expected",
        "predicted_top1",
        "identity",
        "template_source",
        "template_row",
        "template_col",
        "base_template_score",
        "weighted_template_score",
        "template_position_boost",
        "template_position_source",
        "template_final_score",
        "shape",
        "raw_dice",
        "raw_iou",
        "clean_dice",
        "clean_iou",
        "clean_shape",
        "bbox",
        "center",
        "density",
        "red",
        "template_ink_count",
        "template_clean_ink_count",
        "template_red_share",
    ]
    cell_fieldnames = list(cell_summary_rows[0].keys()) if cell_summary_rows else []
    summary_rows = [{key: round_float(value) if isinstance(value, float) else value for key, value in summary.items()}]

    write_csv(out_dir / "piece_style_multitemplate_template_scores.csv", template_score_rows, template_fieldnames)
    write_csv(out_dir / "piece_style_multitemplate_identity_consensus.csv", identity_rows, identity_fieldnames)
    write_csv(out_dir / "piece_style_multitemplate_cell_summary.csv", cell_summary_rows, cell_fieldnames)
    write_csv(out_dir / "piece_style_multitemplate_summary.csv", summary_rows, list(summary_rows[0].keys()))
    return {"summary": summary_rows[0], "identity_rows": identity_rows, "cell_summary_rows": cell_summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline identity-level multi-template consensus/source diversity probe.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--template-assets", type=Path, default=DEFAULT_TEMPLATE_ASSET)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.analysis_dir / "multitemplate_consensus_probe"
    result = run_probe(
        analysis_dir=args.analysis_dir,
        screenshots_dir=args.screenshots_dir,
        template_asset=args.template_assets,
        out_dir=out_dir,
        top_n=args.top_n,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
