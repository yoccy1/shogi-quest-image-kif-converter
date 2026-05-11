from __future__ import annotations

import argparse
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
from multitemplate_consensus_probe import (
    load_excluded_sources,
    load_templates,
    red_location_stats,
    score_template,
)


CELL_KEY_FIELDS = ("app", "piece_style", "sample", "square", "row", "col")
MASK_VARIANTS = (
    "current_clean",
    "red_pruned",
    "edge_band_pruned",
    "red_edge_pruned",
    "interior_only",
    "skeleton_like",
)
ATTRACTION_KINDS = {"KY", "HI", "GI"}
RED_FEATURE_KEYS = (
    "red_mask",
    "red_share",
    "red_center_x",
    "red_center_y",
    "central_red_share",
    "edge_red_share",
)


def expanded_mask(mask: list[bool], iterations: int = 1) -> list[bool]:
    output = mask.copy()
    for _ in range(max(0, iterations)):
        output = dilate(output)
    return output


def edge_band_mask(edge_x: int, edge_y: int) -> list[bool]:
    return [
        x < edge_x or x >= FEATURE_WIDTH - edge_x or y < edge_y or y >= FEATURE_HEIGHT - edge_y
        for y in range(FEATURE_HEIGHT)
        for x in range(FEATURE_WIDTH)
    ]


def interior_region_mask(left: int, right: int, top: int, bottom: int) -> list[bool]:
    return [
        left <= x < right and top <= y < bottom
        for y in range(FEATURE_HEIGHT)
        for x in range(FEATURE_WIDTH)
    ]


def skeleton_like_mask(mask: list[bool], max_iterations: int = 40) -> list[bool]:
    output = mask.copy()

    def value(x: int, y: int) -> int:
        if not (0 <= x < FEATURE_WIDTH and 0 <= y < FEATURE_HEIGHT):
            return 0
        return 1 if output[y * FEATURE_WIDTH + x] else 0

    def neighbor_values(x: int, y: int) -> list[int]:
        return [
            value(x, y - 1),
            value(x + 1, y - 1),
            value(x + 1, y),
            value(x + 1, y + 1),
            value(x, y + 1),
            value(x - 1, y + 1),
            value(x - 1, y),
            value(x - 1, y - 1),
        ]

    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            remove: list[int] = []
            for y in range(1, FEATURE_HEIGHT - 1):
                for x in range(1, FEATURE_WIDTH - 1):
                    index = y * FEATURE_WIDTH + x
                    if not output[index]:
                        continue
                    neighbors = neighbor_values(x, y)
                    neighbor_count = sum(neighbors)
                    transitions = sum(
                        1
                        for left, right in zip(neighbors, neighbors[1:] + neighbors[:1])
                        if left == 0 and right == 1
                    )
                    if neighbor_count < 2 or neighbor_count > 6 or transitions != 1:
                        continue
                    p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
                    if step == 0:
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                            continue
                    remove.append(index)
            if remove:
                changed = True
                for index in remove:
                    output[index] = False
        if not changed:
            break
    return output if any(output) else mask.copy()


def build_mask_variants(
    source_features: dict[str, Any],
    edge_band_ratio: float = 0.12,
    interior_x_ratio: float = 0.22,
    interior_y_ratio: float = 0.18,
    red_dilate_iterations: int = 2,
) -> dict[str, list[bool]]:
    current = list(source_features.get("clean_mask") or source_features.get("ink_mask") or [False] * FEATURE_SIZE)
    red = list(source_features.get("red_mask") or [False] * FEATURE_SIZE)
    expanded_red = expanded_mask(red, red_dilate_iterations)
    edge_x = max(1, round(FEATURE_WIDTH * edge_band_ratio))
    edge_y = max(1, round(FEATURE_HEIGHT * edge_band_ratio))
    edge = edge_band_mask(edge_x, edge_y)
    left = max(0, round(FEATURE_WIDTH * interior_x_ratio))
    right = min(FEATURE_WIDTH, FEATURE_WIDTH - left)
    top = max(0, round(FEATURE_HEIGHT * interior_y_ratio))
    bottom = min(FEATURE_HEIGHT, FEATURE_HEIGHT - top)
    interior = interior_region_mask(left, right, top, bottom)
    red_edge = [ink and not expanded_red[index] and not edge[index] for index, ink in enumerate(current)]
    interior_only = [ink and interior[index] for index, ink in enumerate(current)]
    return {
        "current_clean": current,
        "red_pruned": [ink and not expanded_red[index] for index, ink in enumerate(current)],
        "edge_band_pruned": [ink and not edge[index] for index, ink in enumerate(current)],
        "red_edge_pruned": red_edge,
        "interior_only": interior_only,
        "skeleton_like": skeleton_like_mask(red_edge if any(red_edge) else interior_only),
    }


def source_variant_features(source_features: dict[str, Any], variant_mask: list[bool]) -> dict[str, Any]:
    red_mask = list(source_features.get("red_mask") or [False] * FEATURE_SIZE)
    variant_red = [ink and red_mask[index] for index, ink in enumerate(variant_mask)]
    ink_count = sum(variant_mask)
    red_count = sum(variant_red)
    x1, y1, x2, y2 = mask_bounds(variant_mask, FEATURE_WIDTH, FEATURE_HEIGHT)
    red_stats = red_location_stats(variant_red)
    if ink_count <= 0:
        return {
            **empty_source_features(),
            "ink_mask": [False] * FEATURE_SIZE,
            "clean_mask": [False] * FEATURE_SIZE,
            "red_mask": [False] * FEATURE_SIZE,
        }
    return {
        "ink_mask": variant_mask,
        "clean_mask": variant_mask,
        "red_mask": variant_red,
        "ink_count": ink_count,
        "clean_ink_count": ink_count,
        "original_ink_ratio": ink_count / FEATURE_SIZE,
        "red_share": red_count / max(1, ink_count),
        "bbox_width_ratio": (x2 - x1) / FEATURE_WIDTH,
        "bbox_height_ratio": (y2 - y1) / FEATURE_HEIGHT,
        "bbox_center_x": (x1 + x2) / (2 * FEATURE_WIDTH),
        "bbox_center_y": (y1 + y2) / (2 * FEATURE_HEIGHT),
        "red_center_x": red_stats["red_center_x"],
        "red_center_y": red_stats["red_center_y"],
        "central_red_share": red_stats["central_red_share"],
        "edge_red_share": red_stats["edge_red_share"],
    }


def with_original_red_features(variant_features: dict[str, Any], original_features: dict[str, Any]) -> dict[str, Any]:
    output = dict(variant_features)
    for key in RED_FEATURE_KEYS:
        if key in original_features:
            output[key] = original_features[key]
    return output


def projection(mask: list[bool], axis: str) -> list[float]:
    if axis == "x":
        return [float(sum(1 for y in range(FEATURE_HEIGHT) if mask[y * FEATURE_WIDTH + x])) for x in range(FEATURE_WIDTH)]
    if axis == "y":
        return [float(sum(1 for x in range(FEATURE_WIDTH) if mask[y * FEATURE_WIDTH + x])) for y in range(FEATURE_HEIGHT)]
    raise ValueError(f"unknown projection axis: {axis}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def projection_metrics(source_mask: list[bool], template_mask: list[bool]) -> dict[str, float]:
    x_similarity = cosine_similarity(projection(source_mask, "x"), projection(template_mask, "x"))
    y_similarity = cosine_similarity(projection(source_mask, "y"), projection(template_mask, "y"))
    return {
        "projection_x_similarity": x_similarity,
        "projection_y_similarity": y_similarity,
        "projection_mean_similarity": (x_similarity + y_similarity) / 2.0,
    }


def identity_piece(identity: str) -> str:
    return (identity or "").split(":", 1)[1] if ":" in (identity or "") else ""


def is_attraction_identity(identity: str) -> bool:
    return identity_piece(identity) in ATTRACTION_KINDS


def average_rank(rows: list[dict[str, Any]], field: str) -> float:
    values = [parse_float(row.get(field), math.nan) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else math.nan


def rank_change_class(delta: Any, expected_in_topn: bool = True) -> str:
    if not expected_in_topn:
        return "missing_expected"
    value = parse_float(delta, 0.0)
    if value > 0:
        return "improved"
    if value < 0:
        return "worsened"
    return "unchanged"


def split_sources(value: Any) -> set[str]:
    return {part for part in str(value or "").replace(";", "|").split("|") if part}


def rank_rows(rows: list[dict[str, Any]], score_field: str, rank_field: str) -> None:
    for rank, row in enumerate(sorted(rows, key=lambda item: parse_float(item.get(score_field), -1.0), reverse=True), start=1):
        row[rank_field] = rank


def maybe_row(rows: list[dict[str, Any]], identity: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("candidate_identity") == identity), None)


def run_probe(
    analysis_dir: Path,
    screenshots_dir: Path,
    template_asset: Path,
    out_dir: Path,
    top_n: int,
    virtual_weight: float,
    edge_band_ratio: float,
    interior_x_ratio: float,
    interior_y_ratio: float,
    red_dilate_iterations: int,
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
    grouped_candidates: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[cell_key(row)].append(row)

    cells_cache: dict[Path, dict[tuple[int, int], dict[str, Any]]] = {}
    exclusions_cache: dict[Path, set[str]] = {}
    source_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    variant_masks_by_cell: dict[tuple[str, ...], dict[str, list[bool]]] = {}
    variant_features_by_cell: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    excluded_sources_by_cell: dict[tuple[str, ...], set[str]] = {}

    mask_rows: list[dict[str, Any]] = []
    template_score_rows: list[dict[str, Any]] = []
    shape_only_template_score_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    no_leak_audit_rows: list[dict[str, Any]] = []

    for key, rows in sorted(grouped_candidates.items()):
        rows = sorted(rows, key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        first = rows[0]
        report_path = reports_by_sample.get(str(first.get("sample")))
        if report_path is not None and report_path not in exclusions_cache:
            exclusions_cache[report_path] = load_excluded_sources(report_path)
        excluded_sources = exclusions_cache.get(report_path, set()) if report_path is not None else set()
        excluded_sources_by_cell[key] = excluded_sources
        if key not in source_cache:
            cell: dict[str, Any] | None = None
            if report_path is not None:
                if report_path not in cells_cache:
                    cells_cache[report_path] = load_report_cells(report_path)
                cell = cells_cache[report_path].get((parse_int(first.get("row")), parse_int(first.get("col"))))
            crop = crop_cell_image(screenshots_dir, first, cell or {}) if cell is not None else None
            source_cache[key] = extract_piece_masks(crop) if crop is not None else empty_source_features()
        source_features = source_cache[key]
        variant_masks = build_mask_variants(
            source_features,
            edge_band_ratio=edge_band_ratio,
            interior_x_ratio=interior_x_ratio,
            interior_y_ratio=interior_y_ratio,
            red_dilate_iterations=red_dilate_iterations,
        )
        variant_masks_by_cell[key] = variant_masks
        variant_features = {name: source_variant_features(source_features, mask) for name, mask in variant_masks.items()}
        variant_features_by_cell[key] = variant_features
        current_count = sum(variant_masks["current_clean"])
        current_red_count = sum(source_features.get("red_mask") or [])
        for variant_name in MASK_VARIANTS:
            feature = variant_features[variant_name]
            mask_rows.append(
                {
                    **{field: first.get(field, "") for field in CELL_KEY_FIELDS},
                    "variant": variant_name,
                    "expected": first.get("expected"),
                    "base_top1_identity": first.get("candidate_identity"),
                    "excluded_sources": "|".join(sorted(excluded_sources)),
                    "current_clean_count": current_count,
                    "current_red_count": current_red_count,
                    "variant_mask_count": feature["clean_ink_count"],
                    "variant_red_count": sum(feature.get("red_mask") or []),
                    "variant_removed_from_current_count": current_count - feature["clean_ink_count"],
                    "variant_retained_ratio": feature["clean_ink_count"] / max(1, current_count),
                    "variant_red_share": feature["red_share"],
                    "variant_bbox_width": feature["bbox_width_ratio"],
                    "variant_bbox_height": feature["bbox_height_ratio"],
                    "variant_bbox_center_x": feature["bbox_center_x"],
                    "variant_bbox_center_y": feature["bbox_center_y"],
                }
            )

        row_number = parse_int(first.get("row"))
        col_number = parse_int(first.get("col"))
        for variant_name in MASK_VARIANTS:
            scores_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
            shape_only_scores_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
            leak_hits = 0
            shape_only_leak_hits = 0
            shape_only_feature = with_original_red_features(variant_features[variant_name], variant_features["current_clean"])
            for template in templates:
                if template.get("source") in excluded_sources:
                    continue
                scored = score_template(variant_features[variant_name], template, row_number, col_number)
                if scored is not None:
                    scored.update(projection_metrics(variant_features[variant_name]["clean_mask"], template["clean_mask"]))
                    if scored.get("template_source") in excluded_sources:
                        leak_hits += 1
                    scores_by_identity[str(scored["identity"])].append(scored)
                    template_score_rows.append(
                        {
                            **{field: first.get(field, "") for field in CELL_KEY_FIELDS},
                            "variant": variant_name,
                            "expected": first.get("expected"),
                            "base_top1_identity": first.get("candidate_identity"),
                            "excluded_sources": "|".join(sorted(excluded_sources)),
                            **{name: round_float(value) if isinstance(value, float) else value for name, value in scored.items()},
                        }
                    )
                shape_only_scored = score_template(shape_only_feature, template, row_number, col_number)
                if shape_only_scored is not None:
                    shape_only_scored.update(projection_metrics(variant_features[variant_name]["clean_mask"], template["clean_mask"]))
                    if shape_only_scored.get("template_source") in excluded_sources:
                        shape_only_leak_hits += 1
                    shape_only_scores_by_identity[str(shape_only_scored["identity"])].append(shape_only_scored)
                    shape_only_template_score_rows.append(
                        {
                            **{field: first.get(field, "") for field in CELL_KEY_FIELDS},
                            "variant": variant_name,
                            "expected": first.get("expected"),
                            "base_top1_identity": first.get("candidate_identity"),
                            "excluded_sources": "|".join(sorted(excluded_sources)),
                            **{name: round_float(value) if isinstance(value, float) else value for name, value in shape_only_scored.items()},
                        }
                    )
            no_leak_audit_rows.append(
                {
                    **{field: first.get(field, "") for field in CELL_KEY_FIELDS},
                    "variant": variant_name,
                    "excluded_sources": "|".join(sorted(excluded_sources)),
                    "template_score_rows": sum(len(values) for values in scores_by_identity.values()),
                    "shape_only_template_score_rows": sum(len(values) for values in shape_only_scores_by_identity.values()),
                    "leak_template_score_rows": leak_hits,
                    "shape_only_leak_template_score_rows": shape_only_leak_hits,
                }
            )
            for candidate in rows:
                identity = str(candidate.get("candidate_identity") or "")
                template_scores = scores_by_identity.get(identity, [])
                best = max(template_scores, key=lambda item: parse_float(item.get("template_final_score")), default={})
                variant_score = parse_float(best.get("template_final_score"), 0.0)
                shape_only_template_scores = shape_only_scores_by_identity.get(identity, [])
                shape_only_best = max(shape_only_template_scores, key=lambda item: parse_float(item.get("template_final_score")), default={})
                shape_only_score = parse_float(shape_only_best.get("template_final_score"), 0.0)
                base_score = parse_float(candidate.get("score"))
                candidate_sources = split_sources(candidate.get("source"))
                identity_rows.append(
                    {
                        **{field: candidate.get(field, "") for field in CELL_KEY_FIELDS},
                        "variant": variant_name,
                        "excluded_sources": "|".join(sorted(excluded_sources)),
                        "expected": candidate.get("expected"),
                        "predicted_top1": candidate.get("predicted_top1"),
                        "candidate_index": candidate.get("candidate_index"),
                        "candidate_rank": candidate.get("candidate_rank"),
                        "candidate_identity": identity,
                        "is_expected": candidate.get("is_expected"),
                        "is_predicted_top1": candidate.get("is_predicted_top1"),
                        "base_score": candidate.get("score"),
                        "candidate_source": candidate.get("source"),
                        "candidate_source_leak": bool(candidate_sources & excluded_sources),
                        "variant_score": round_float(variant_score),
                        "base_plus_variant_score": round_float((1.0 - virtual_weight) * base_score + virtual_weight * variant_score),
                        "shape_only_variant_score": round_float(shape_only_score),
                        "base_plus_shape_only_variant_score": round_float((1.0 - virtual_weight) * base_score + virtual_weight * shape_only_score),
                        "best_template_source": best.get("template_source", ""),
                        "best_template_row": best.get("template_row", ""),
                        "best_template_col": best.get("template_col", ""),
                        "best_base_template_score": round_float(parse_float(best.get("base_template_score"), 0.0)),
                        "best_weighted_template_score": round_float(parse_float(best.get("weighted_template_score"), 0.0)),
                        "best_template_position_boost": round_float(parse_float(best.get("template_position_boost"), 0.0)),
                        "best_shape": round_float(parse_float(best.get("shape"), 0.0)),
                        "best_raw_dice": round_float(parse_float(best.get("raw_dice"), 0.0)),
                        "best_clean_dice": round_float(parse_float(best.get("clean_dice"), 0.0)),
                        "best_bbox": round_float(parse_float(best.get("bbox"), 0.0)),
                        "best_center": round_float(parse_float(best.get("center"), 0.0)),
                        "best_density": round_float(parse_float(best.get("density"), 0.0)),
                        "best_red": round_float(parse_float(best.get("red"), 0.0)),
                        "best_projection_x_similarity": round_float(parse_float(best.get("projection_x_similarity"), 0.0)),
                        "best_projection_y_similarity": round_float(parse_float(best.get("projection_y_similarity"), 0.0)),
                        "best_projection_mean_similarity": round_float(parse_float(best.get("projection_mean_similarity"), 0.0)),
                        "shape_only_best_template_source": shape_only_best.get("template_source", ""),
                        "shape_only_best_red": round_float(parse_float(shape_only_best.get("red"), 0.0)),
                        "shape_only_best_projection_mean_similarity": round_float(parse_float(shape_only_best.get("projection_mean_similarity"), 0.0)),
                        "variant_mask_count": variant_features[variant_name]["clean_ink_count"],
                        "variant_red_share": round_float(parse_float(variant_features[variant_name]["red_share"])),
                    }
                )

    grouped_identity: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in identity_rows:
        grouped_identity[(*cell_key(row), str(row.get("variant") or ""))].append(row)

    cell_summary_rows: list[dict[str, Any]] = []
    for _key, rows in sorted(grouped_identity.items()):
        rows.sort(key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        rank_rows(rows, "variant_score", "variant_rank")
        rank_rows(rows, "base_plus_variant_score", "base_plus_variant_rank")
        rank_rows(rows, "shape_only_variant_score", "shape_only_variant_rank")
        rank_rows(rows, "base_plus_shape_only_variant_score", "base_plus_shape_only_variant_rank")
        base_top = min(rows, key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        variant_top = min(rows, key=lambda row: parse_int(row.get("variant_rank"), 9999))
        base_plus_top = min(rows, key=lambda row: parse_int(row.get("base_plus_variant_rank"), 9999))
        shape_only_top = min(rows, key=lambda row: parse_int(row.get("shape_only_variant_rank"), 9999))
        base_plus_shape_only_top = min(rows, key=lambda row: parse_int(row.get("base_plus_shape_only_variant_rank"), 9999))
        expected = str(base_top.get("expected") or "")
        expected_row = maybe_row(rows, expected)
        base_top_identity = str(base_top.get("candidate_identity") or "")
        base_top_is_expected = base_top_identity == expected
        variant_top_is_expected = variant_top.get("candidate_identity") == expected
        base_plus_top_is_expected = base_plus_top.get("candidate_identity") == expected
        shape_only_top_is_expected = shape_only_top.get("candidate_identity") == expected
        base_plus_shape_only_top_is_expected = base_plus_shape_only_top.get("candidate_identity") == expected
        attraction_row = maybe_row(rows, base_top_identity)
        gap = gaps_by_cell.get(cell_key(base_top), {})
        expected_base_rank = expected_row.get("candidate_rank") if expected_row else gap.get("expected_candidate_rank", "")
        expected_variant_rank = expected_row.get("variant_rank") if expected_row else ""
        expected_base_plus_rank = expected_row.get("base_plus_variant_rank") if expected_row else ""
        expected_shape_only_rank = expected_row.get("shape_only_variant_rank") if expected_row else ""
        expected_base_plus_shape_only_rank = expected_row.get("base_plus_shape_only_variant_rank") if expected_row else ""
        expected_in_topn = expected_row is not None
        expected_rank_delta_variant = parse_int(expected_base_rank, 9999) - parse_int(expected_variant_rank, 9999) if expected_row else ""
        expected_rank_delta_base_plus = parse_int(expected_base_rank, 9999) - parse_int(expected_base_plus_rank, 9999) if expected_row else ""
        expected_rank_delta_shape_only = parse_int(expected_base_rank, 9999) - parse_int(expected_shape_only_rank, 9999) if expected_row else ""
        expected_rank_delta_base_plus_shape_only = (
            parse_int(expected_base_rank, 9999) - parse_int(expected_base_plus_shape_only_rank, 9999) if expected_row else ""
        )
        cell_summary_rows.append(
            {
                **{field: base_top.get(field, "") for field in CELL_KEY_FIELDS},
                "variant": base_top.get("variant"),
                "excluded_sources": base_top.get("excluded_sources"),
                "expected": expected,
                "base_top1_identity": base_top_identity,
                "base_top1_score": base_top.get("base_score"),
                "variant_top1_identity": variant_top.get("candidate_identity"),
                "variant_top1_score": variant_top.get("variant_score"),
                "base_plus_variant_top1_identity": base_plus_top.get("candidate_identity"),
                "base_plus_variant_top1_score": base_plus_top.get("base_plus_variant_score"),
                "shape_only_variant_top1_identity": shape_only_top.get("candidate_identity"),
                "shape_only_variant_top1_score": shape_only_top.get("shape_only_variant_score"),
                "base_plus_shape_only_variant_top1_identity": base_plus_shape_only_top.get("candidate_identity"),
                "base_plus_shape_only_variant_top1_score": base_plus_shape_only_top.get("base_plus_shape_only_variant_score"),
                "expected_piece": identity_piece(expected),
                "base_top1_piece": identity_piece(base_top_identity),
                "expected_in_topn": expected_in_topn,
                "expected_base_rank": expected_base_rank,
                "expected_variant_rank": expected_variant_rank,
                "expected_base_plus_variant_rank": expected_base_plus_rank,
                "expected_shape_only_variant_rank": expected_shape_only_rank,
                "expected_base_plus_shape_only_variant_rank": expected_base_plus_shape_only_rank,
                "expected_base_score": expected_row.get("base_score") if expected_row else "",
                "expected_variant_score": expected_row.get("variant_score") if expected_row else "",
                "expected_base_plus_variant_score": expected_row.get("base_plus_variant_score") if expected_row else "",
                "expected_shape_only_variant_score": expected_row.get("shape_only_variant_score") if expected_row else "",
                "expected_base_plus_shape_only_variant_score": expected_row.get("base_plus_shape_only_variant_score") if expected_row else "",
                "expected_rank_delta_variant": expected_rank_delta_variant,
                "expected_rank_delta_base_plus": expected_rank_delta_base_plus,
                "expected_rank_delta_shape_only_variant": expected_rank_delta_shape_only,
                "expected_rank_delta_base_plus_shape_only": expected_rank_delta_base_plus_shape_only,
                "expected_rank_change_variant": rank_change_class(expected_rank_delta_variant, expected_in_topn),
                "expected_rank_change_base_plus": rank_change_class(expected_rank_delta_base_plus, expected_in_topn),
                "expected_rank_change_shape_only_variant": rank_change_class(expected_rank_delta_shape_only, expected_in_topn),
                "expected_rank_change_base_plus_shape_only": rank_change_class(expected_rank_delta_base_plus_shape_only, expected_in_topn),
                "base_top1_is_expected": base_top_is_expected,
                "variant_top1_is_expected": variant_top_is_expected,
                "base_plus_variant_top1_is_expected": base_plus_top_is_expected,
                "shape_only_variant_top1_is_expected": shape_only_top_is_expected,
                "base_plus_shape_only_variant_top1_is_expected": base_plus_shape_only_top_is_expected,
                "fixed_by_variant": (not base_top_is_expected) and variant_top_is_expected,
                "fixed_by_base_plus_variant": (not base_top_is_expected) and base_plus_top_is_expected,
                "fixed_by_shape_only_variant": (not base_top_is_expected) and shape_only_top_is_expected,
                "fixed_by_base_plus_shape_only_variant": (not base_top_is_expected) and base_plus_shape_only_top_is_expected,
                "base_top1_is_ky_hi_gi": is_attraction_identity(base_top_identity),
                "variant_top1_is_ky_hi_gi": is_attraction_identity(str(variant_top.get("candidate_identity") or "")),
                "base_plus_variant_top1_is_ky_hi_gi": is_attraction_identity(str(base_plus_top.get("candidate_identity") or "")),
                "shape_only_variant_top1_is_ky_hi_gi": is_attraction_identity(str(shape_only_top.get("candidate_identity") or "")),
                "base_plus_shape_only_variant_top1_is_ky_hi_gi": is_attraction_identity(str(base_plus_shape_only_top.get("candidate_identity") or "")),
                "base_attraction_identity": base_top_identity if is_attraction_identity(base_top_identity) else "",
                "base_attraction_variant_rank": attraction_row.get("variant_rank") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_base_plus_rank": attraction_row.get("base_plus_variant_rank") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_shape_only_variant_rank": attraction_row.get("shape_only_variant_rank") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_base_plus_shape_only_rank": attraction_row.get("base_plus_shape_only_variant_rank") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_variant_score": attraction_row.get("variant_score") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_base_score": attraction_row.get("base_score") if attraction_row and is_attraction_identity(base_top_identity) else "",
                "base_attraction_weakened_by_variant": bool(attraction_row and is_attraction_identity(base_top_identity) and parse_int(attraction_row.get("variant_rank"), 1) > 1),
                "base_attraction_weakened_by_base_plus": bool(attraction_row and is_attraction_identity(base_top_identity) and parse_int(attraction_row.get("base_plus_variant_rank"), 1) > 1),
                "base_attraction_weakened_by_shape_only_variant": bool(attraction_row and is_attraction_identity(base_top_identity) and parse_int(attraction_row.get("shape_only_variant_rank"), 1) > 1),
                "base_attraction_weakened_by_base_plus_shape_only": bool(attraction_row and is_attraction_identity(base_top_identity) and parse_int(attraction_row.get("base_plus_shape_only_variant_rank"), 1) > 1),
                "topn": len(rows),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for variant_name in MASK_VARIANTS:
        rows = [row for row in cell_summary_rows if row.get("variant") == variant_name]
        mask_rows_for_variant = [row for row in mask_rows if row.get("variant") == variant_name]
        audit_rows_for_variant = [row for row in no_leak_audit_rows if row.get("variant") == variant_name]
        expected_in_topn_count = sum(1 for row in rows if row.get("expected_in_topn"))
        base_error_count = sum(1 for row in rows if not row.get("base_top1_is_expected"))
        variant_errors = sum(1 for row in rows if not row.get("variant_top1_is_expected"))
        base_plus_errors = sum(1 for row in rows if not row.get("base_plus_variant_top1_is_expected"))
        shape_only_errors = sum(1 for row in rows if not row.get("shape_only_variant_top1_is_expected"))
        base_plus_shape_only_errors = sum(1 for row in rows if not row.get("base_plus_shape_only_variant_top1_is_expected"))
        improved = sum(1 for row in rows if parse_float(row.get("expected_rank_delta_variant"), 0.0) > 0)
        worsened = sum(1 for row in rows if parse_float(row.get("expected_rank_delta_variant"), 0.0) < 0)
        unchanged = sum(1 for row in rows if parse_float(row.get("expected_rank_delta_variant"), 0.0) == 0)
        base_attraction_rows = [row for row in rows if row.get("base_top1_is_ky_hi_gi")]
        summary_rows.append(
            {
                "analysis_dir": str(analysis_dir),
                "variant": variant_name,
                "top_n": top_n,
                "virtual_weight": virtual_weight,
                "edge_band_ratio": edge_band_ratio,
                "interior_x_ratio": interior_x_ratio,
                "interior_y_ratio": interior_y_ratio,
                "red_dilate_iterations": red_dilate_iterations,
                "cell_rows": len(rows),
                "expected_in_topn_count": expected_in_topn_count,
                "base_errors_if_top1": base_error_count,
                "variant_errors_if_top1": variant_errors,
                "base_plus_variant_errors_if_top1": base_plus_errors,
                "shape_only_variant_errors_if_top1": shape_only_errors,
                "base_plus_shape_only_variant_errors_if_top1": base_plus_shape_only_errors,
                "fixed_by_variant_count": sum(1 for row in rows if row.get("fixed_by_variant")),
                "fixed_by_base_plus_variant_count": sum(1 for row in rows if row.get("fixed_by_base_plus_variant")),
                "fixed_by_shape_only_variant_count": sum(1 for row in rows if row.get("fixed_by_shape_only_variant")),
                "fixed_by_base_plus_shape_only_variant_count": sum(1 for row in rows if row.get("fixed_by_base_plus_shape_only_variant")),
                "expected_base_rank_avg": round_float(average_rank(rows, "expected_base_rank")),
                "expected_variant_rank_avg": round_float(average_rank(rows, "expected_variant_rank")),
                "expected_base_plus_variant_rank_avg": round_float(average_rank(rows, "expected_base_plus_variant_rank")),
                "expected_shape_only_variant_rank_avg": round_float(average_rank(rows, "expected_shape_only_variant_rank")),
                "expected_base_plus_shape_only_variant_rank_avg": round_float(average_rank(rows, "expected_base_plus_shape_only_variant_rank")),
                "expected_rank_improved_count": improved,
                "expected_rank_worsened_count": worsened,
                "expected_rank_unchanged_count": unchanged,
                "base_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("base_top1_is_ky_hi_gi")),
                "variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("variant_top1_is_ky_hi_gi")),
                "base_plus_variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("base_plus_variant_top1_is_ky_hi_gi")),
                "shape_only_variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("shape_only_variant_top1_is_ky_hi_gi")),
                "base_plus_shape_only_variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("base_plus_shape_only_variant_top1_is_ky_hi_gi")),
                "variant_top1_ky_count": sum(1 for row in rows if identity_piece(str(row.get("variant_top1_identity") or "")) == "KY"),
                "variant_top1_hi_count": sum(1 for row in rows if identity_piece(str(row.get("variant_top1_identity") or "")) == "HI"),
                "variant_top1_gi_count": sum(1 for row in rows if identity_piece(str(row.get("variant_top1_identity") or "")) == "GI"),
                "base_attraction_weakened_by_variant_count": sum(1 for row in base_attraction_rows if row.get("base_attraction_weakened_by_variant")),
                "base_attraction_weakened_by_base_plus_count": sum(1 for row in base_attraction_rows if row.get("base_attraction_weakened_by_base_plus")),
                "base_attraction_weakened_by_shape_only_variant_count": sum(1 for row in base_attraction_rows if row.get("base_attraction_weakened_by_shape_only_variant")),
                "base_attraction_weakened_by_base_plus_shape_only_count": sum(1 for row in base_attraction_rows if row.get("base_attraction_weakened_by_base_plus_shape_only")),
                "avg_variant_retained_ratio": round_float(mean(parse_float(row.get("variant_retained_ratio")) for row in mask_rows_for_variant)) if mask_rows_for_variant else "",
                "min_variant_mask_count": min((parse_int(row.get("variant_mask_count")) for row in mask_rows_for_variant), default=""),
                "avg_variant_red_share": round_float(mean(parse_float(row.get("variant_red_share")) for row in mask_rows_for_variant)) if mask_rows_for_variant else "",
                "template_score_rows": sum(parse_int(row.get("template_score_rows")) for row in audit_rows_for_variant),
                "shape_only_template_score_rows": sum(parse_int(row.get("shape_only_template_score_rows")) for row in audit_rows_for_variant),
                "leak_template_score_rows": sum(parse_int(row.get("leak_template_score_rows")) for row in audit_rows_for_variant),
                "shape_only_leak_template_score_rows": sum(parse_int(row.get("shape_only_leak_template_score_rows")) for row in audit_rows_for_variant),
                "candidate_source_leak_rows": sum(1 for row in identity_rows if row.get("variant") == variant_name and row.get("candidate_source_leak")),
                "candidate_source_empty_rows": sum(
                    1 for row in identity_rows if row.get("variant") == variant_name and not split_sources(row.get("candidate_source"))
                ),
            }
        )

    ky_summary_rows = grouped_summary_rows(
        cell_summary_rows,
        group_name="ky_attraction",
        group_filter=lambda row: identity_piece(str(row.get("base_top1_identity") or "")) == "KY",
    )
    expected_identity_summary_rows = grouped_summary_rows(
        cell_summary_rows,
        group_name="expected_identity",
        group_key=lambda row: str(row.get("expected") or ""),
    )
    rank_change_rows = rank_change_summary_rows(cell_summary_rows)

    write_csv(out_dir / "piece_style_mask_variant_masks.csv", mask_rows, list(mask_rows[0].keys()) if mask_rows else [])
    write_csv(out_dir / "piece_style_mask_variant_template_scores.csv", template_score_rows, list(template_score_rows[0].keys()) if template_score_rows else [])
    write_csv(
        out_dir / "piece_style_mask_variant_shape_only_template_scores.csv",
        shape_only_template_score_rows,
        list(shape_only_template_score_rows[0].keys()) if shape_only_template_score_rows else [],
    )
    identity_fieldnames = [
        *CELL_KEY_FIELDS,
        "variant",
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
        "candidate_source_leak",
        "variant_score",
        "base_plus_variant_score",
        "shape_only_variant_score",
        "base_plus_shape_only_variant_score",
        "variant_rank",
        "base_plus_variant_rank",
        "shape_only_variant_rank",
        "base_plus_shape_only_variant_rank",
        "best_template_source",
        "best_template_row",
        "best_template_col",
        "best_base_template_score",
        "best_weighted_template_score",
        "best_template_position_boost",
        "best_shape",
        "best_raw_dice",
        "best_clean_dice",
        "best_bbox",
        "best_center",
        "best_density",
        "best_red",
        "best_projection_x_similarity",
        "best_projection_y_similarity",
        "best_projection_mean_similarity",
        "shape_only_best_template_source",
        "shape_only_best_red",
        "shape_only_best_projection_mean_similarity",
        "variant_mask_count",
        "variant_red_share",
    ]
    write_csv(out_dir / "piece_style_mask_variant_identity_scores.csv", identity_rows, identity_fieldnames)
    write_csv(out_dir / "piece_style_mask_variant_cell_summary.csv", cell_summary_rows, list(cell_summary_rows[0].keys()) if cell_summary_rows else [])
    write_csv(out_dir / "piece_style_mask_variant_summary.csv", summary_rows, list(summary_rows[0].keys()) if summary_rows else [])
    write_csv(out_dir / "piece_style_mask_variant_ky_attraction_summary.csv", ky_summary_rows, list(ky_summary_rows[0].keys()) if ky_summary_rows else [])
    write_csv(
        out_dir / "piece_style_mask_variant_expected_identity_summary.csv",
        expected_identity_summary_rows,
        list(expected_identity_summary_rows[0].keys()) if expected_identity_summary_rows else [],
    )
    write_csv(out_dir / "piece_style_mask_variant_rank_change_summary.csv", rank_change_rows, list(rank_change_rows[0].keys()) if rank_change_rows else [])
    write_csv(out_dir / "piece_style_mask_variant_no_leak_audit.csv", no_leak_audit_rows, list(no_leak_audit_rows[0].keys()) if no_leak_audit_rows else [])
    return {
        "summary": [{key: round_float(value) if isinstance(value, float) else value for key, value in row.items()} for row in summary_rows],
        "cell_summary_rows": cell_summary_rows,
        "identity_rows": identity_rows,
        "template_score_rows": template_score_rows,
        "shape_only_template_score_rows": shape_only_template_score_rows,
        "out_dir": str(out_dir),
    }


def grouped_summary_rows(
    cell_summary_rows: list[dict[str, Any]],
    group_name: str,
    group_filter: Any | None = None,
    group_key: Any | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_summary_rows:
        if group_filter is not None and not group_filter(row):
            continue
        key = group_key(row) if group_key is not None else group_name
        grouped[(str(row.get("variant") or ""), str(key))].append(row)
    output: list[dict[str, Any]] = []
    for (variant, key), rows in sorted(grouped.items()):
        output.append(
            {
                "variant": variant,
                group_name: key,
                "cell_rows": len(rows),
                "variant_errors_if_top1": sum(1 for row in rows if not row.get("variant_top1_is_expected")),
                "base_plus_variant_errors_if_top1": sum(1 for row in rows if not row.get("base_plus_variant_top1_is_expected")),
                "shape_only_variant_errors_if_top1": sum(1 for row in rows if not row.get("shape_only_variant_top1_is_expected")),
                "base_plus_shape_only_variant_errors_if_top1": sum(1 for row in rows if not row.get("base_plus_shape_only_variant_top1_is_expected")),
                "fixed_by_variant_count": sum(1 for row in rows if row.get("fixed_by_variant")),
                "fixed_by_base_plus_variant_count": sum(1 for row in rows if row.get("fixed_by_base_plus_variant")),
                "fixed_by_shape_only_variant_count": sum(1 for row in rows if row.get("fixed_by_shape_only_variant")),
                "fixed_by_base_plus_shape_only_variant_count": sum(1 for row in rows if row.get("fixed_by_base_plus_shape_only_variant")),
                "expected_base_rank_avg": round_float(average_rank(rows, "expected_base_rank")),
                "expected_variant_rank_avg": round_float(average_rank(rows, "expected_variant_rank")),
                "expected_base_plus_variant_rank_avg": round_float(average_rank(rows, "expected_base_plus_variant_rank")),
                "expected_shape_only_variant_rank_avg": round_float(average_rank(rows, "expected_shape_only_variant_rank")),
                "expected_base_plus_shape_only_variant_rank_avg": round_float(average_rank(rows, "expected_base_plus_shape_only_variant_rank")),
                "expected_rank_improved_count": sum(1 for row in rows if parse_float(row.get("expected_rank_delta_variant"), 0.0) > 0),
                "expected_rank_worsened_count": sum(1 for row in rows if parse_float(row.get("expected_rank_delta_variant"), 0.0) < 0),
                "variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("variant_top1_is_ky_hi_gi")),
                "shape_only_variant_top1_ky_hi_gi_count": sum(1 for row in rows if row.get("shape_only_variant_top1_is_ky_hi_gi")),
                "base_attraction_weakened_by_variant_count": sum(1 for row in rows if row.get("base_attraction_weakened_by_variant")),
                "base_attraction_weakened_by_shape_only_variant_count": sum(1 for row in rows if row.get("base_attraction_weakened_by_shape_only_variant")),
            }
        )
    return output


def rank_change_summary_rows(cell_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modes = (
        ("variant", "expected_rank_change_variant", "fixed_by_variant", "variant_top1_is_ky_hi_gi"),
        ("base_plus_variant", "expected_rank_change_base_plus", "fixed_by_base_plus_variant", "base_plus_variant_top1_is_ky_hi_gi"),
        ("shape_only_variant", "expected_rank_change_shape_only_variant", "fixed_by_shape_only_variant", "shape_only_variant_top1_is_ky_hi_gi"),
        (
            "base_plus_shape_only_variant",
            "expected_rank_change_base_plus_shape_only",
            "fixed_by_base_plus_shape_only_variant",
            "base_plus_shape_only_variant_top1_is_ky_hi_gi",
        ),
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_summary_rows:
        variant = str(row.get("variant") or "")
        for mode, change_field, _fixed_field, _attraction_field in modes:
            grouped[(variant, mode, str(row.get(change_field) or ""))].append(row)
    output: list[dict[str, Any]] = []
    for (variant, mode, change), rows in sorted(grouped.items()):
        fixed_field = next(item[2] for item in modes if item[0] == mode)
        attraction_field = next(item[3] for item in modes if item[0] == mode)
        output.append(
            {
                "variant": variant,
                "score_mode": mode,
                "rank_change": change,
                "cell_rows": len(rows),
                "fixed_count": sum(1 for row in rows if row.get(fixed_field)),
                "top1_ky_hi_gi_count": sum(1 for row in rows if row.get(attraction_field)),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline source mask variant probe for piyo_chick residual board cells.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--template-assets", type=Path, default=DEFAULT_TEMPLATE_ASSET)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--virtual-weight", type=float, default=0.50)
    parser.add_argument("--edge-band-ratio", type=float, default=0.12)
    parser.add_argument("--interior-x-ratio", type=float, default=0.22)
    parser.add_argument("--interior-y-ratio", type=float, default=0.18)
    parser.add_argument("--red-dilate-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.analysis_dir / "mask_variant_offline_probe"
    result = run_probe(
        analysis_dir=args.analysis_dir,
        screenshots_dir=args.screenshots_dir,
        template_asset=args.template_assets,
        out_dir=out_dir,
        top_n=args.top_n,
        virtual_weight=args.virtual_weight,
        edge_band_ratio=args.edge_band_ratio,
        interior_x_ratio=args.interior_x_ratio,
        interior_y_ratio=args.interior_y_ratio,
        red_dilate_iterations=args.red_dilate_iterations,
    )
    print(json.dumps({"out_dir": result["out_dir"], "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
