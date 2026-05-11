#!/usr/bin/env python3
"""Audit cells that improved in the mask-variant offline probe.

This script is intentionally offline-only. It reads the B5-29 mask-variant
probe outputs, compares the expected identity against the top competing
identity for a small fixed target set, and emits CSV/Markdown/HTML diagnostics.
It does not change production recognition logic or template assets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chamfer_offline_probe import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_SCREENSHOTS_DIR,
    DEFAULT_TEMPLATE_ASSET,
    cell_key,
    crop_cell_image,
    empty_source_features,
    extract_piece_masks,
    load_report_cells,
    load_reports_by_sample,
    parse_float,
    parse_int,
    read_csv,
    round_float,
    write_csv,
)
from mask_variant_offline_probe import (
    CELL_KEY_FIELDS,
    build_mask_variants,
    source_variant_features,
    split_sources,
)
from multitemplate_consensus_probe import load_templates


DEFAULT_MASK_VARIANT_DIR = (
    DEFAULT_ANALYSIS_DIR / "mask_variant_offline_probe"
)
DEFAULT_OUT_DIR = DEFAULT_ANALYSIS_DIR / "mask_variant_cell_audit"
DEFAULT_VARIANTS = ("current_clean", "interior_only", "skeleton_like")
SCORE_MODES = {
    "base": {
        "rank": "candidate_rank",
        "score": "base_score",
        "template_table": "variant",
    },
    "variant": {
        "rank": "variant_rank",
        "score": "variant_score",
        "template_table": "variant",
    },
    "base_plus_variant": {
        "rank": "base_plus_variant_rank",
        "score": "base_plus_variant_score",
        "template_table": "variant",
    },
    "shape_only_variant": {
        "rank": "shape_only_variant_rank",
        "score": "shape_only_variant_score",
        "template_table": "shape_only",
    },
    "base_plus_shape_only_variant": {
        "rank": "base_plus_shape_only_variant_rank",
        "score": "base_plus_shape_only_variant_score",
        "template_table": "shape_only",
    },
}
OUTPUT_SCORE_MODES = tuple(SCORE_MODES.keys())


@dataclass(frozen=True)
class TargetCell:
    sample: str
    square: str
    expected: str


TARGET_CELLS = (
    TargetCell("ぴよ将棋_ひよこ駒_通常_01", "6八", "black:OU"),
    TargetCell("ぴよ将棋_ひよこ駒_通常_02", "3八", "black:OU"),
    TargetCell("ぴよ将棋_ひよこ駒_通常_02", "6六", "white:KA"),
    TargetCell("ぴよ将棋_ひよこ駒_通常_02", "4二", "white:KI"),
    TargetCell("ぴよ将棋_ひよこ駒_通常_02", "4四", "white:KI"),
    TargetCell("ぴよ将棋_ひよこ駒_通常_02", "6三", "white:KI"),
)


IDENTITY_OUTPUT_FIELDS = [
    *CELL_KEY_FIELDS,
    "variant",
    "score_mode",
    "role",
    "identity",
    "expected_identity",
    "rank",
    "score",
    "top_identity",
    "top_rank",
    "top_score",
    "candidate_source",
    "candidate_source_leak",
    "excluded_sources",
    "best_template_source",
    "best_template_row",
    "best_template_col",
    "best_template_score",
    "best_template_position_boost",
    "best_shape",
    "best_raw_dice",
    "best_clean_dice",
    "best_raw_iou",
    "best_clean_iou",
    "best_clean_shape",
    "best_bbox",
    "best_center",
    "best_density",
    "best_red",
    "best_projection_x_similarity",
    "best_projection_y_similarity",
    "best_projection_mean_similarity",
    "source_ink_count",
    "source_clean_ink_count",
    "source_density",
    "source_red_share",
    "source_central_red_share",
    "source_edge_red_share",
    "source_red_center_x",
    "source_red_center_y",
    "source_bbox_width",
    "source_bbox_height",
    "source_bbox_center_x",
    "source_bbox_center_y",
    "source_retained_ratio",
    "template_ink_count",
    "template_clean_ink_count",
    "template_red_share",
    "template_central_red_share",
    "template_edge_red_share",
    "template_red_center_x",
    "template_red_center_y",
    "template_bbox_width",
    "template_bbox_height",
    "template_bbox_center_x",
    "template_bbox_center_y",
]

PAIR_DELTA_FIELDS = [
    *CELL_KEY_FIELDS,
    "variant",
    "score_mode",
    "expected_identity",
    "competitor_identity",
    "expected_rank",
    "competitor_rank",
    "rank_delta_expected_minus_competitor",
    "rank_advantage_expected",
    "expected_score",
    "competitor_score",
    "score_delta_expected_minus_competitor",
    "score_advantage_expected",
    "delta_best_bbox",
    "delta_best_center",
    "delta_best_density",
    "delta_best_red",
    "delta_best_projection_x_similarity",
    "delta_best_projection_y_similarity",
    "delta_best_projection_mean_similarity",
    "delta_template_ink_count",
    "delta_template_clean_ink_count",
    "delta_template_red_share",
    "delta_template_central_red_share",
    "delta_template_edge_red_share",
    "delta_template_bbox_width",
    "delta_template_bbox_height",
    "delta_template_bbox_center_x",
    "delta_template_bbox_center_y",
    "source_ink_count",
    "source_density",
    "source_red_share",
    "source_central_red_share",
    "source_edge_red_share",
    "source_bbox_width",
    "source_bbox_height",
    "source_bbox_center_x",
    "source_bbox_center_y",
]

NO_LEAK_FIELDS = [
    *CELL_KEY_FIELDS,
    "variant",
    "excluded_sources",
    "identity_score_rows",
    "candidate_source_leak_count",
    "candidate_source_empty_count",
    "template_score_rows",
    "template_source_leak_count",
    "shape_only_template_score_rows",
    "shape_only_template_source_leak_count",
]


def target_key(target: TargetCell) -> Tuple[str, str, str]:
    return target.sample, target.square, target.expected


def row_target_key(row: Mapping[str, str]) -> Tuple[str, str, str]:
    return row.get("sample", ""), row.get("square", ""), row.get("expected", "")


def key_without_variant(row: Mapping[str, str]) -> Tuple[str, ...]:
    return tuple(row.get(field, "") for field in CELL_KEY_FIELDS)


def key_with_variant(row: Mapping[str, str]) -> Tuple[str, ...]:
    return (*key_without_variant(row), row.get("variant", ""))


def truthy(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def load_required_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_csv(path)


def parse_csv_sources(value: str) -> List[str]:
    sources = []
    for part in str(value or "").replace(";", "|").split("|"):
        text = part.strip()
        if text:
            sources.append(text)
    return sources


def source_matches_excluded(source: str, excluded: Sequence[str]) -> bool:
    if not source:
        return False
    return any(item and item in source for item in excluded)


def safe_float(value: object) -> Optional[float]:
    parsed = parse_float(value)
    return parsed


def safe_int(value: object) -> Optional[int]:
    parsed = parse_int(value)
    return parsed


def num_for_delta(row: Mapping[str, str], field: str) -> Optional[float]:
    return safe_float(row.get(field, ""))


def numeric_delta(
    expected_row: Mapping[str, str],
    competitor_row: Mapping[str, str],
    field: str,
) -> str:
    left = num_for_delta(expected_row, field)
    right = num_for_delta(competitor_row, field)
    if left is None or right is None:
        return ""
    return str(round_float(left - right))


def sort_by_rank(rows: Iterable[Mapping[str, str]], rank_field: str) -> List[Mapping[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            safe_int(row.get(rank_field)) is None,
            safe_int(row.get(rank_field)) or 10**9,
            -(safe_float(row.get(SCORE_MODES["variant"]["score"])) or -10**9),
            row.get("candidate_identity", ""),
        ),
    )


def best_template_rows(
    rows: Sequence[Mapping[str, str]],
) -> Dict[Tuple[str, ...], Mapping[str, str]]:
    best: Dict[Tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = (*key_with_variant(row), row.get("identity", ""))
        current = best.get(key)
        row_score = safe_float(row.get("template_final_score")) or -10**9
        current_score = (
            safe_float(current.get("template_final_score")) if current else None
        )
        if current is None or row_score > (current_score if current_score is not None else -10**9):
            best[key] = row
    return best


def load_template_feature_index(template_asset: Path) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}

    def fmt(value: object) -> str:
        parsed = safe_float(value)
        return "" if parsed is None else str(round_float(parsed))

    try:
        templates = load_templates(template_asset)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return index
    for template in templates:
        identity = str(template.get("identity", ""))
        source = str(template.get("source", ""))
        row = "" if template.get("row") is None else str(template.get("row"))
        col = "" if template.get("col") is None else str(template.get("col"))
        index[(identity, source, row, col)] = {
            "template_central_red_share": fmt(template.get("central_red_share")),
            "template_edge_red_share": fmt(template.get("edge_red_share")),
            "template_red_center_x": fmt(template.get("red_center_x")),
            "template_red_center_y": fmt(template.get("red_center_y")),
            "template_bbox_width": fmt(template.get("bbox_width_ratio")),
            "template_bbox_height": fmt(template.get("bbox_height_ratio")),
            "template_bbox_center_x": fmt(template.get("bbox_center_x")),
            "template_bbox_center_y": fmt(template.get("bbox_center_y")),
        }
    return index


def load_mask_rows(mask_rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, ...], Mapping[str, str]]:
    return {key_with_variant(row): row for row in mask_rows}


def mask_row_fallback_features(row: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if row is None:
        return {}
    count = safe_float(row.get("variant_mask_count"))
    width = safe_float(row.get("variant_bbox_width"))
    height = safe_float(row.get("variant_bbox_height"))
    density = None
    if count is not None and width and height:
        density = count / (width * height)
    current_count = safe_float(row.get("current_mask_count"))
    retained = None
    if count is not None and current_count:
        retained = count / current_count
    return {
        "source_ink_count": "" if count is None else str(int(count)),
        "source_clean_ink_count": "" if count is None else str(int(count)),
        "source_density": "" if density is None else str(round_float(density)),
        "source_red_share": row.get("variant_red_share", ""),
        "source_bbox_width": row.get("variant_bbox_width", ""),
        "source_bbox_height": row.get("variant_bbox_height", ""),
        "source_bbox_center_x": row.get("variant_bbox_center_x", ""),
        "source_bbox_center_y": row.get("variant_bbox_center_y", ""),
        "source_retained_ratio": "" if retained is None else str(round_float(retained)),
    }


def source_features_from_report(
    analysis_dir: Path,
    screenshots_dir: Path,
    candidate_rows: Sequence[Mapping[str, str]],
    variants: Sequence[str],
) -> Dict[Tuple[str, ...], Dict[str, str]]:
    report_by_sample = load_reports_by_sample(analysis_dir)
    report_cache: Dict[Path, Dict[Tuple[str, str], Mapping[str, object]]] = {}
    by_key: Dict[Tuple[str, ...], Dict[str, str]] = {}
    first_candidate: Dict[Tuple[str, ...], Mapping[str, str]] = {}
    for row in candidate_rows:
        first_candidate.setdefault(key_without_variant(row), row)

    for key, row in first_candidate.items():
        sample = row.get("sample", "")
        report_path = report_by_sample.get(sample)
        crop = None
        if report_path:
            report = report_cache.get(report_path)
            if report is None:
                report = load_report_cells(report_path)
                report_cache[report_path] = report
            cell = report.get((safe_int(row.get("row")) or -1, safe_int(row.get("col")) or -1))
            if cell is not None:
                crop = crop_cell_image(screenshots_dir, row, cell)
        if crop is None:
            source_masks = empty_source_features()
            variant_map = build_mask_variants(source_masks)
        else:
            source_masks = extract_piece_masks(crop)
            variant_map = build_mask_variants(source_masks)
        for variant in variants:
            features = source_variant_features(
                source_masks,
                variant_map.get(variant, variant_map.get("current_clean", [])),
            )
            by_key[(*key, variant)] = {
                "source_ink_count": str(features.get("ink_count", "")),
                "source_clean_ink_count": str(features.get("clean_ink_count", "")),
                "source_density": str(features.get("density", features.get("original_ink_ratio", ""))),
                "source_red_share": str(features.get("red_share", "")),
                "source_central_red_share": str(features.get("central_red_share", "")),
                "source_edge_red_share": str(features.get("edge_red_share", "")),
                "source_red_center_x": str(features.get("red_center_x", "")),
                "source_red_center_y": str(features.get("red_center_y", "")),
                "source_bbox_width": str(features.get("bbox_width", features.get("bbox_width_ratio", ""))),
                "source_bbox_height": str(features.get("bbox_height", features.get("bbox_height_ratio", ""))),
                "source_bbox_center_x": str(features.get("bbox_center_x", "")),
                "source_bbox_center_y": str(features.get("bbox_center_y", "")),
                "source_retained_ratio": str(features.get("retained_ratio", "")),
            }
    return by_key


def merged_source_features(
    key: Tuple[str, ...],
    extracted: Mapping[Tuple[str, ...], Dict[str, str]],
    mask_rows: Mapping[Tuple[str, ...], Mapping[str, str]],
) -> Dict[str, str]:
    values = dict(extracted.get(key, {}))
    fallback = mask_row_fallback_features(mask_rows.get(key))
    for field, fallback_value in fallback.items():
        current = values.get(field, "")
        if current in {"", "None", "0", "0.0"} and fallback_value != "":
            values[field] = fallback_value
    for field in (
        "source_ink_count",
        "source_clean_ink_count",
        "source_density",
        "source_red_share",
        "source_central_red_share",
        "source_edge_red_share",
        "source_red_center_x",
        "source_red_center_y",
        "source_bbox_width",
        "source_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
        "source_retained_ratio",
    ):
        values.setdefault(field, "")
    return values


def select_expected_and_competitor(
    rows: Sequence[Mapping[str, str]],
    expected_identity: str,
    score_mode: str,
) -> Tuple[Optional[Mapping[str, str]], Optional[Mapping[str, str]], Optional[Mapping[str, str]]]:
    rank_field = SCORE_MODES[score_mode]["rank"]
    sorted_rows = sort_by_rank(rows, rank_field)
    expected_row = next(
        (row for row in rows if row.get("candidate_identity", "") == expected_identity),
        None,
    )
    top_row = sorted_rows[0] if sorted_rows else None
    competitor_row = next(
        (row for row in sorted_rows if row.get("candidate_identity", "") != expected_identity),
        None,
    )
    return expected_row, competitor_row, top_row


def selected_template_row(
    selected_identity_row: Mapping[str, str],
    score_mode: str,
    template_best: Mapping[str, Mapping[Tuple[str, ...], Mapping[str, str]]],
) -> Mapping[str, str]:
    table_name = SCORE_MODES[score_mode]["template_table"]
    key = (
        *key_with_variant(selected_identity_row),
        selected_identity_row.get("candidate_identity", ""),
    )
    return template_best.get(table_name, {}).get(key, {})


def template_metadata_for(
    identity: str,
    template_row: Mapping[str, str],
    template_features: Mapping[Tuple[str, str, str, str], Dict[str, str]],
) -> Dict[str, str]:
    source = template_row.get("template_source", "")
    row = template_row.get("template_row", "")
    col = template_row.get("template_col", "")
    return dict(template_features.get((identity, source, row, col), {}))


def identity_output_row(
    selected_identity_row: Mapping[str, str],
    top_row: Optional[Mapping[str, str]],
    role: str,
    score_mode: str,
    source_features: Mapping[str, str],
    template_best: Mapping[str, Mapping[Tuple[str, ...], Mapping[str, str]]],
    template_features: Mapping[Tuple[str, str, str, str], Dict[str, str]],
) -> Dict[str, str]:
    config = SCORE_MODES[score_mode]
    identity = selected_identity_row.get("candidate_identity", "")
    template_row = selected_template_row(selected_identity_row, score_mode, template_best)
    template_meta = template_metadata_for(identity, template_row, template_features)
    row = {field: selected_identity_row.get(field, "") for field in CELL_KEY_FIELDS}
    row.update(
        {
            "variant": selected_identity_row.get("variant", ""),
            "score_mode": score_mode,
            "role": role,
            "identity": identity,
            "expected_identity": selected_identity_row.get("expected", ""),
            "rank": selected_identity_row.get(config["rank"], ""),
            "score": selected_identity_row.get(config["score"], ""),
            "top_identity": "" if top_row is None else top_row.get("candidate_identity", ""),
            "top_rank": "" if top_row is None else top_row.get(config["rank"], ""),
            "top_score": "" if top_row is None else top_row.get(config["score"], ""),
            "candidate_source": selected_identity_row.get("candidate_source", ""),
            "candidate_source_leak": selected_identity_row.get("candidate_source_leak", ""),
            "excluded_sources": selected_identity_row.get("excluded_sources", ""),
            "best_template_source": template_row.get("template_source", ""),
            "best_template_row": template_row.get("template_row", ""),
            "best_template_col": template_row.get("template_col", ""),
            "best_template_score": template_row.get("template_final_score", ""),
            "best_template_position_boost": template_row.get("template_position_boost", ""),
            "best_shape": template_row.get("shape", ""),
            "best_raw_dice": template_row.get("raw_dice", ""),
            "best_clean_dice": template_row.get("clean_dice", ""),
            "best_raw_iou": template_row.get("raw_iou", ""),
            "best_clean_iou": template_row.get("clean_iou", ""),
            "best_clean_shape": template_row.get("clean_shape", ""),
            "best_bbox": template_row.get("bbox", ""),
            "best_center": template_row.get("center", ""),
            "best_density": template_row.get("density", ""),
            "best_red": template_row.get("red", ""),
            "best_projection_x_similarity": template_row.get("projection_x_similarity", ""),
            "best_projection_y_similarity": template_row.get("projection_y_similarity", ""),
            "best_projection_mean_similarity": template_row.get("projection_mean_similarity", ""),
            "template_ink_count": template_row.get("template_ink_count", ""),
            "template_clean_ink_count": template_row.get("template_clean_ink_count", ""),
            "template_red_share": template_row.get("template_red_share", ""),
        }
    )
    row.update(source_features)
    row.update(
        {
            "template_central_red_share": template_meta.get("template_central_red_share", ""),
            "template_edge_red_share": template_meta.get("template_edge_red_share", ""),
            "template_red_center_x": template_meta.get("template_red_center_x", ""),
            "template_red_center_y": template_meta.get("template_red_center_y", ""),
            "template_bbox_width": template_meta.get("template_bbox_width", ""),
            "template_bbox_height": template_meta.get("template_bbox_height", ""),
            "template_bbox_center_x": template_meta.get("template_bbox_center_x", ""),
            "template_bbox_center_y": template_meta.get("template_bbox_center_y", ""),
        }
    )
    return {field: row.get(field, "") for field in IDENTITY_OUTPUT_FIELDS}


def pair_delta_row(
    expected_row: Mapping[str, str],
    competitor_row: Mapping[str, str],
) -> Dict[str, str]:
    row = {field: expected_row.get(field, "") for field in CELL_KEY_FIELDS}
    expected_rank = safe_float(expected_row.get("rank"))
    competitor_rank = safe_float(competitor_row.get("rank"))
    expected_score = safe_float(expected_row.get("score"))
    competitor_score = safe_float(competitor_row.get("score"))
    row.update(
        {
            "variant": expected_row.get("variant", ""),
            "score_mode": expected_row.get("score_mode", ""),
            "expected_identity": expected_row.get("identity", ""),
            "competitor_identity": competitor_row.get("identity", ""),
            "expected_rank": expected_row.get("rank", ""),
            "competitor_rank": competitor_row.get("rank", ""),
            "rank_delta_expected_minus_competitor": (
                "" if expected_rank is None or competitor_rank is None else str(round_float(expected_rank - competitor_rank))
            ),
            "rank_advantage_expected": (
                "" if expected_rank is None or competitor_rank is None else str(round_float(competitor_rank - expected_rank))
            ),
            "expected_score": expected_row.get("score", ""),
            "competitor_score": competitor_row.get("score", ""),
            "score_delta_expected_minus_competitor": (
                "" if expected_score is None or competitor_score is None else str(round_float(expected_score - competitor_score))
            ),
            "score_advantage_expected": (
                "" if expected_score is None or competitor_score is None else str(round_float(expected_score - competitor_score))
            ),
        }
    )
    for field in (
        "best_bbox",
        "best_center",
        "best_density",
        "best_red",
        "best_projection_x_similarity",
        "best_projection_y_similarity",
        "best_projection_mean_similarity",
        "template_ink_count",
        "template_clean_ink_count",
        "template_red_share",
        "template_central_red_share",
        "template_edge_red_share",
        "template_bbox_width",
        "template_bbox_height",
        "template_bbox_center_x",
        "template_bbox_center_y",
    ):
        row[f"delta_{field}"] = numeric_delta(expected_row, competitor_row, field)
    for field in (
        "source_ink_count",
        "source_density",
        "source_red_share",
        "source_central_red_share",
        "source_edge_red_share",
        "source_bbox_width",
        "source_bbox_height",
        "source_bbox_center_x",
        "source_bbox_center_y",
    ):
        row[field] = expected_row.get(field, "")
    return {field: row.get(field, "") for field in PAIR_DELTA_FIELDS}


def mode_values(
    identity_groups: Mapping[Tuple[str, ...], Sequence[Mapping[str, str]]],
    base_key: Tuple[str, ...],
    variant: str,
    expected: str,
    mode: str,
) -> Dict[str, str]:
    rows = identity_groups.get((*base_key, variant), [])
    expected_row, competitor_row, top_row = select_expected_and_competitor(rows, expected, mode)
    rank_field = SCORE_MODES[mode]["rank"]
    score_field = SCORE_MODES[mode]["score"]
    return {
        "top_identity": "" if top_row is None else top_row.get("candidate_identity", ""),
        "top_rank": "" if top_row is None else top_row.get(rank_field, ""),
        "top_score": "" if top_row is None else top_row.get(score_field, ""),
        "expected_rank": "" if expected_row is None else expected_row.get(rank_field, ""),
        "expected_score": "" if expected_row is None else expected_row.get(score_field, ""),
        "competitor_identity": "" if competitor_row is None else competitor_row.get("candidate_identity", ""),
        "competitor_rank": "" if competitor_row is None else competitor_row.get(rank_field, ""),
        "competitor_score": "" if competitor_row is None else competitor_row.get(score_field, ""),
    }


def classify_variant(
    current_rank: Optional[int],
    variant_values: Mapping[str, str],
    base_plus_values: Mapping[str, str],
    shape_values: Mapping[str, str],
    expected: str,
) -> str:
    variant_rank = safe_int(variant_values.get("expected_rank"))
    variant_top = variant_values.get("top_identity", "")
    base_plus_top = base_plus_values.get("top_identity", "")
    shape_top = shape_values.get("top_identity", "")
    labels = []
    if variant_top == expected:
        labels.append("variant_top1_fix")
    elif current_rank is not None and variant_rank is not None and variant_rank < current_rank:
        labels.append("rank_improved_only")
    else:
        labels.append("no_rank_improvement")
    if variant_top == expected and base_plus_top != expected:
        labels.append("base_plus_lost")
    elif variant_top == expected and base_plus_top == expected:
        labels.append("base_plus_preserved")
    if variant_top == expected and shape_top != expected:
        labels.append("shape_only_lost")
    elif variant_top == expected and shape_top == expected:
        labels.append("shape_only_preserved")
    return ";".join(labels)


def build_cell_summary_rows(
    target_rows: Sequence[Mapping[str, str]],
    identity_groups: Mapping[Tuple[str, ...], Sequence[Mapping[str, str]]],
    source_feature_map: Mapping[Tuple[str, ...], Mapping[str, str]],
    variants: Sequence[str],
) -> List[Dict[str, str]]:
    summary_rows: List[Dict[str, str]] = []
    seen: set[Tuple[str, ...]] = set()
    for target in target_rows:
        base_key = key_without_variant(target)
        if base_key in seen:
            continue
        seen.add(base_key)
        expected = target.get("expected", "")
        row = {field: target.get(field, "") for field in CELL_KEY_FIELDS}
        row["expected_identity"] = expected
        current_values = mode_values(identity_groups, base_key, "current_clean", expected, "variant")
        current_rank = safe_int(current_values.get("expected_rank"))
        for variant in variants:
            source_features = source_feature_map.get((*base_key, variant), {})
            for field in (
                "source_ink_count",
                "source_density",
                "source_red_share",
                "source_central_red_share",
                "source_edge_red_share",
                "source_retained_ratio",
            ):
                row[f"{variant}_{field}"] = source_features.get(field, "")
            for mode in OUTPUT_SCORE_MODES:
                values = mode_values(identity_groups, base_key, variant, expected, mode)
                for name, value in values.items():
                    row[f"{variant}_{mode}_{name}"] = value
            variant_values = mode_values(identity_groups, base_key, variant, expected, "variant")
            base_plus_values = mode_values(identity_groups, base_key, variant, expected, "base_plus_variant")
            shape_values = mode_values(identity_groups, base_key, variant, expected, "shape_only_variant")
            row[f"{variant}_classification"] = classify_variant(
                current_rank,
                variant_values,
                base_plus_values,
                shape_values,
                expected,
            )
        row["observation"] = observation_for_summary_row(row, variants)
        summary_rows.append(row)
    return summary_rows


def observation_for_summary_row(row: Mapping[str, str], variants: Sequence[str]) -> str:
    notes = []
    expected = row.get("expected_identity", "")
    for variant in variants:
        if variant == "current_clean":
            continue
        top = row.get(f"{variant}_variant_top_identity", "")
        rank = row.get(f"{variant}_variant_expected_rank", "")
        base_plus_top = row.get(f"{variant}_base_plus_variant_top_identity", "")
        shape_top = row.get(f"{variant}_shape_only_variant_top_identity", "")
        classification = row.get(f"{variant}_classification", "")
        if top == expected:
            if base_plus_top == expected:
                notes.append(f"{variant}: variant-only top1 and base+ keeps it")
            else:
                notes.append(f"{variant}: variant-only top1 but base+ reverts to {base_plus_top}")
        elif "rank_improved_only" in classification:
            notes.append(f"{variant}: rank improves to {rank}, top remains {top}")
        else:
            notes.append(f"{variant}: no useful rank lift, top {top}")
        if top == expected and shape_top != expected:
            notes.append(f"{variant}: shape-only loses expected to {shape_top}")
    return "; ".join(notes)


def build_no_leak_rows(
    target_identity_rows: Sequence[Mapping[str, str]],
    template_rows: Sequence[Mapping[str, str]],
    shape_only_rows: Sequence[Mapping[str, str]],
    variants: Sequence[str],
) -> List[Dict[str, str]]:
    by_cell_variant: Dict[Tuple[str, ...], List[Mapping[str, str]]] = {}
    for row in target_identity_rows:
        by_cell_variant.setdefault(key_with_variant(row), []).append(row)

    template_by_cell_variant: Dict[Tuple[str, ...], List[Mapping[str, str]]] = {}
    for row in template_rows:
        template_by_cell_variant.setdefault(key_with_variant(row), []).append(row)
    shape_template_by_cell_variant: Dict[Tuple[str, ...], List[Mapping[str, str]]] = {}
    for row in shape_only_rows:
        shape_template_by_cell_variant.setdefault(key_with_variant(row), []).append(row)

    output = []
    for key, rows in sorted(by_cell_variant.items()):
        if key[-1] not in variants:
            continue
        first = rows[0]
        excluded = parse_csv_sources(first.get("excluded_sources", ""))
        template_items = template_by_cell_variant.get(key, [])
        shape_items = shape_template_by_cell_variant.get(key, [])
        row = {field: first.get(field, "") for field in CELL_KEY_FIELDS}
        row.update(
            {
                "variant": key[-1],
                "excluded_sources": first.get("excluded_sources", ""),
                "identity_score_rows": str(len(rows)),
                "candidate_source_leak_count": str(
                    sum(
                        1
                        for item in rows
                        if truthy(item.get("candidate_source_leak"))
                        or source_matches_excluded(item.get("candidate_source", ""), excluded)
                    )
                ),
                "candidate_source_empty_count": str(
                    sum(1 for item in rows if not split_sources(item.get("candidate_source", "")))
                ),
                "template_score_rows": str(len(template_items)),
                "template_source_leak_count": str(
                    sum(
                        1
                        for item in template_items
                        if source_matches_excluded(item.get("template_source", ""), excluded)
                    )
                ),
                "shape_only_template_score_rows": str(len(shape_items)),
                "shape_only_template_source_leak_count": str(
                    sum(
                        1
                        for item in shape_items
                        if source_matches_excluded(item.get("template_source", ""), excluded)
                    )
                ),
            }
        )
        output.append({field: row.get(field, "") for field in NO_LEAK_FIELDS})
    return output


def markdown_summary(summary_rows: Sequence[Mapping[str, str]], variants: Sequence[str]) -> str:
    lines = [
        "# Mask Variant Cell Audit Summary",
        "",
        "Offline audit only. `variant-only` and high-weight/rank improvements are not production recognition improvements.",
        "",
        "| sample | square | expected | variant | top identity | expected rank | base+ top | base+ expected rank | shape-only top | classification | note |",
        "|---|---:|---|---|---|---:|---|---:|---|---|---|",
    ]
    for row in summary_rows:
        for variant in variants:
            lines.append(
                "| {sample} | {square} | {expected} | {variant} | {top} | {rank} | {base_top} | {base_rank} | {shape_top} | {classification} | {note} |".format(
                    sample=row.get("sample", ""),
                    square=row.get("square", ""),
                    expected=row.get("expected_identity", ""),
                    variant=variant,
                    top=row.get(f"{variant}_variant_top_identity", ""),
                    rank=row.get(f"{variant}_variant_expected_rank", ""),
                    base_top=row.get(f"{variant}_base_plus_variant_top_identity", ""),
                    base_rank=row.get(f"{variant}_base_plus_variant_expected_rank", ""),
                    shape_top=row.get(f"{variant}_shape_only_variant_top_identity", ""),
                    classification=row.get(f"{variant}_classification", ""),
                    note=row.get("observation", "") if variant != "current_clean" else "",
                )
            )
    return "\n".join(lines) + "\n"


def html_summary(summary_rows: Sequence[Mapping[str, str]], variants: Sequence[str]) -> str:
    rows = []
    for row in summary_rows:
        for variant in variants:
            rows.append(
                "<tr>"
                f"<td>{html.escape(row.get('sample', ''))}</td>"
                f"<td>{html.escape(row.get('square', ''))}</td>"
                f"<td>{html.escape(row.get('expected_identity', ''))}</td>"
                f"<td>{html.escape(variant)}</td>"
                f"<td>{html.escape(row.get(f'{variant}_variant_top_identity', ''))}</td>"
                f"<td>{html.escape(row.get(f'{variant}_variant_expected_rank', ''))}</td>"
                f"<td>{html.escape(row.get(f'{variant}_base_plus_variant_top_identity', ''))}</td>"
                f"<td>{html.escape(row.get(f'{variant}_base_plus_variant_expected_rank', ''))}</td>"
                f"<td>{html.escape(row.get(f'{variant}_shape_only_variant_top_identity', ''))}</td>"
                f"<td>{html.escape(row.get(f'{variant}_classification', ''))}</td>"
                f"<td>{html.escape(row.get('observation', '') if variant != 'current_clean' else '')}</td>"
                "</tr>"
            )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Mask Variant Cell Audit</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;font-size:13px;}th,td{border:1px solid #ccc;padding:4px 6px;}"
        "th{background:#f4f4f4;}td{vertical-align:top;}</style></head><body>"
        "<h1>Mask Variant Cell Audit Summary</h1>"
        "<p>Offline audit only. <code>variant-only</code> and rank improvements are not production recognition improvements.</p>"
        "<table><thead><tr><th>sample</th><th>square</th><th>expected</th><th>variant</th>"
        "<th>top identity</th><th>expected rank</th><th>base+ top</th><th>base+ expected rank</th>"
        "<th>shape-only top</th><th>classification</th><th>note</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>\n"
    )


def filter_target_rows(
    rows: Sequence[Mapping[str, str]],
    targets: Sequence[TargetCell],
    variants: Sequence[str],
) -> List[Mapping[str, str]]:
    allowed = {target_key(target) for target in targets}
    return [
        row
        for row in rows
        if row_target_key(row) in allowed and row.get("variant", "") in variants
    ]


def filter_target_cell_rows(
    rows: Sequence[Mapping[str, str]],
    targets: Sequence[TargetCell],
) -> List[Mapping[str, str]]:
    allowed = {target_key(target) for target in targets}
    return [row for row in rows if row_target_key(row) in allowed]


def group_identity_rows(
    rows: Sequence[Mapping[str, str]],
) -> Dict[Tuple[str, ...], List[Mapping[str, str]]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault(key_with_variant(row), []).append(row)
    return groups


def run_audit(
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    mask_variant_dir: Path = DEFAULT_MASK_VARIANT_DIR,
    screenshots_dir: Path = DEFAULT_SCREENSHOTS_DIR,
    template_asset: Path = DEFAULT_TEMPLATE_ASSET,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_n: int = 30,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    targets: Sequence[TargetCell] = TARGET_CELLS,
) -> Dict[str, Path]:
    del top_n  # Inputs are already top-N probe outputs; keep CLI parity/documentation.
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_summary_rows = load_required_csv(mask_variant_dir / "piece_style_mask_variant_cell_summary.csv")
    identity_rows = load_required_csv(mask_variant_dir / "piece_style_mask_variant_identity_scores.csv")
    mask_rows = load_required_csv(mask_variant_dir / "piece_style_mask_variant_masks.csv")
    template_rows = load_required_csv(mask_variant_dir / "piece_style_mask_variant_template_scores.csv")
    shape_template_rows = load_required_csv(mask_variant_dir / "piece_style_mask_variant_shape_only_template_scores.csv")
    candidate_rows = load_required_csv(analysis_dir / "piece_style_board_error_candidates.csv")

    target_identity_rows = filter_target_rows(identity_rows, targets, variants)
    target_cell_rows = filter_target_cell_rows(candidate_rows, targets)
    target_mask_rows = filter_target_rows(mask_rows, targets, variants)
    target_template_rows = filter_target_rows(template_rows, targets, variants)
    target_shape_template_rows = filter_target_rows(shape_template_rows, targets, variants)
    target_summary_rows = filter_target_rows(cell_summary_rows, targets, variants)
    if not target_identity_rows:
        raise ValueError("No target identity rows found")
    if not target_cell_rows:
        target_cell_rows = [
            {field: row.get(field, "") for field in CELL_KEY_FIELDS}
            for row in target_identity_rows
        ]

    identity_groups = group_identity_rows(target_identity_rows)
    mask_row_index = load_mask_rows(target_mask_rows)
    extracted_source = source_features_from_report(
        analysis_dir,
        screenshots_dir,
        target_cell_rows,
        variants,
    )
    source_feature_map: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for key in set(mask_row_index.keys()) | set(extracted_source.keys()):
        source_feature_map[key] = merged_source_features(key, extracted_source, mask_row_index)

    template_best = {
        "variant": best_template_rows(target_template_rows),
        "shape_only": best_template_rows(target_shape_template_rows),
    }
    template_feature_index = load_template_feature_index(template_asset)

    identity_output: List[Dict[str, str]] = []
    pair_output: List[Dict[str, str]] = []
    expected_role_rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    competitor_role_rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for key, rows in sorted(identity_groups.items()):
        variant = key[-1]
        source_features = source_feature_map.get(key, {})
        expected_identity = rows[0].get("expected", "")
        for score_mode in OUTPUT_SCORE_MODES:
            expected_row, competitor_row, top_row = select_expected_and_competitor(
                rows,
                expected_identity,
                score_mode,
            )
            if expected_row is None or competitor_row is None:
                continue
            expected_out = identity_output_row(
                expected_row,
                top_row,
                "expected",
                score_mode,
                source_features,
                template_best,
                template_feature_index,
            )
            competitor_out = identity_output_row(
                competitor_row,
                top_row,
                "competitor",
                score_mode,
                source_features,
                template_best,
                template_feature_index,
            )
            identity_output.extend([expected_out, competitor_out])
            pair_output.append(pair_delta_row(expected_out, competitor_out))
            role_key = (*key, score_mode)
            expected_role_rows[role_key] = expected_out
            competitor_role_rows[role_key] = competitor_out

    # Touch the input summary in a visible way, so schema drift is caught while
    # keeping identity rows as the authoritative source for this audit.
    summary_keys = {key_with_variant(row) for row in target_summary_rows}
    missing_summary_keys = sorted(set(identity_groups.keys()) - summary_keys)
    if missing_summary_keys:
        raise ValueError(f"Missing cell summary rows for {len(missing_summary_keys)} target variants")

    cell_output = build_cell_summary_rows(
        target_cell_rows,
        identity_groups,
        source_feature_map,
        variants,
    )
    no_leak_output = build_no_leak_rows(
        target_identity_rows,
        target_template_rows,
        target_shape_template_rows,
        variants,
    )

    identity_path = out_dir / "piece_style_mask_variant_cell_audit_identity_rows.csv"
    pair_path = out_dir / "piece_style_mask_variant_cell_audit_pair_deltas.csv"
    cell_path = out_dir / "piece_style_mask_variant_cell_audit_cell_summary.csv"
    no_leak_path = out_dir / "piece_style_mask_variant_cell_audit_no_leak_audit.csv"
    md_path = out_dir / "mask_variant_cell_audit_summary.md"
    html_path = out_dir / "mask_variant_cell_audit_summary.html"

    write_csv(identity_path, identity_output, IDENTITY_OUTPUT_FIELDS)
    write_csv(pair_path, pair_output, PAIR_DELTA_FIELDS)
    cell_fields = sorted({field for row in cell_output for field in row.keys()})
    write_csv(cell_path, cell_output, cell_fields)
    write_csv(no_leak_path, no_leak_output, NO_LEAK_FIELDS)
    md_path.write_text(markdown_summary(cell_output, variants), encoding="utf-8")
    html_path.write_text(html_summary(cell_output, variants), encoding="utf-8")

    return {
        "identity_rows": identity_path,
        "pair_deltas": pair_path,
        "cell_summary": cell_path,
        "no_leak_audit": no_leak_path,
        "summary_md": md_path,
        "summary_html": html_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--mask-variant-dir", type=Path, default=DEFAULT_MASK_VARIANT_DIR)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--template-assets", type=Path, default=DEFAULT_TEMPLATE_ASSET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_audit(
        analysis_dir=args.analysis_dir,
        mask_variant_dir=args.mask_variant_dir,
        screenshots_dir=args.screenshots_dir,
        template_asset=args.template_assets,
        out_dir=args.out_dir,
        top_n=args.top_n,
        variants=tuple(args.variants),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
