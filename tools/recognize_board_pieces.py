from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime for optional legacy use.
    cv2 = None
    np = None


NORMAL_PIECES = ["OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU"]
PROMOTED_PIECES = [None, "RY", "UM", None, "NG", "NK", "NY", "TO"]
PROMOTED_PIECE_SET = {piece for piece in PROMOTED_PIECES if piece is not None}
VALID_PIECE_SET = set(NORMAL_PIECES) | PROMOTED_PIECE_SET
RANK_NAMES = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
NORMALIZED_SIZE = (48, 56)
EMPTY_DARK_RATIO_THRESHOLD = 0.028
EMPTY_BBOX_RATIO_THRESHOLD = 0.11
PIECE_CONFIDENCE_THRESHOLD = 0.34
AMBIGUOUS_MARGIN_THRESHOLD = 0.032
TEMPLATE_SCALE_FACTORS = (0.88, 0.94, 1.0, 1.08, 1.18, 1.28)
TEMPLATE_SHIFT_PIXELS = (-4, 0, 4)
OPENCV_TEMPLATE_HEIGHT_RATIOS = (0.72, 0.78, 0.84, 0.90, 0.96, 1.02, 1.08)
OPENCV_PIECE_CONFIDENCE_THRESHOLD = 0.50
OPENCV_AMBIGUOUS_MARGIN_THRESHOLD = 0.025
OPENCV_CONFUSION_PAIR_MARGIN_THRESHOLD = 0.060
OPENCV_PROMOTED_PAIR_MARGIN_THRESHOLD = 0.050
OPENCV_COLOR_FLIP_MARGIN_THRESHOLD = 0.060
OPENCV_EMPTY_INK_DENSITY_THRESHOLD = 0.020
OPENCV_EMPTY_EDGE_DENSITY_THRESHOLD = 0.105
OPENCV_EMPTY_ARTIFACT_MIN_SCORE = 0.12
OPENCV_WEIGHTS = {
    "edge": 0.18,
    "ink": 0.64,
    "color": 0.18,
}
CONFUSION_PAIR_PIECES = {
    frozenset(pair)
    for pair in (
        ("FU", "GI"),
        ("FU", "KI"),
        ("FU", "HI"),
        ("FU", "KY"),
        ("FU", "KE"),
        ("KI", "GI"),
        ("KE", "GI"),
        ("KE", "HI"),
        ("KY", "KE"),
        ("KY", "GI"),
        ("KY", "HI"),
        ("RY", "UM"),
        ("RY", "HI"),
        ("UM", "KA"),
        ("NG", "GI"),
        ("NK", "KE"),
        ("NY", "KY"),
        ("TO", "FU"),
    )
}
OPENCV_VARIANT_CACHE: dict[tuple[str, str, str, str, int, int, int, float], dict | None] = {}
OPENCV_CALIBRATION_CACHE: dict[str, tuple[list["OpenCvTemplate"], dict]] = {}
CALIBRATION_MASK_FEATURE_CACHE: dict[tuple[str, str, str], tuple[MaskFeatures, ...]] = {}
OPENCV_CALIBRATION_DISK_CACHE_VERSION = 1
CALIBRATION_CELL_PAD_X_RATIO = 0.08
CALIBRATION_CELL_PAD_Y_RATIO = 0.18
CALIBRATION_MIN_MASK_PIXELS = 24
CALIBRATION_MAX_TEMPLATES_PER_LABEL = 2
CALIBRATION_FAST_CONFIDENCE_THRESHOLD = 0.50
CALIBRATION_FAST_MARGIN_THRESHOLD = 0.012
CALIBRATION_CAUTION_CONFIDENCE_THRESHOLD = 0.80
CALIBRATION_CAUTION_MARGIN_THRESHOLD = 0.06
CALIBRATION_CAUTION_SCORE_MULTIPLIER = 0.92
CALIBRATION_HOLDOUT_DECISIVE_CONFIDENCE_THRESHOLD = 0.80
CALIBRATION_HOLDOUT_DECISIVE_MARGIN_THRESHOLD = 0.06
CALIBRATION_HOLDOUT_SCORE_MULTIPLIER = 0.92
CALIBRATION_RED_TEXT_SHARE_THRESHOLD = 0.36
SPRITE_FAST_CONFIDENCE_THRESHOLD = 0.42
SPRITE_FAST_MARGIN_THRESHOLD = 0.012
FAST_GLYPH_SCALE_FACTORS = (1.0,)
FAST_GLYPH_SHIFT_PIXELS = (-3, 0, 3)
FAST_GLYPH_SCALED_FACTORS = (0.94, 1.0)
RY_SIMPLE_GLYPH = "竜"
RY_SIMPLE_FONT_PATHS = (
    "C:/Windows/Fonts/HGRGE.TTC",
    "C:/Windows/Fonts/BIZ-UDMinchoM.ttc",
)
RY_SIMPLE_FONT_SIZE_RATIOS = (0.54, 0.60)
RY_SIMPLE_OFFSETS = ((0.0, 0.02),)
PROMOTED_INK_RED = (205, 36, 28, 255)
HOG_IMAGE_SIZE = (64, 64)
HOG_SYNTHETIC_CELL_SIZE = (120, 136)
HOG_SYNTHETIC_SAMPLES_PER_CLASS = 28
HOG_PIECE_CONFIDENCE_THRESHOLD = 0.45
HOG_AMBIGUOUS_MARGIN_THRESHOLD = 0.025
HOG_SCORE_SCALE = 0.55
HOG_SVM_MODEL_CACHE: dict[str, "HogSvmModel"] = {}


@dataclass(frozen=True)
class MaskFeatures:
    mask: bytes
    bits: int
    dark_ratio: float
    projection_x: tuple[int, ...]
    projection_y: tuple[int, ...]


@dataclass(frozen=True)
class FastGlyphFeatures:
    features: MaskFeatures
    red_share: float
    bbox: list[int] | None
    variants: tuple[MaskFeatures, ...] = ()
    red_mask: bytes = b""


@dataclass(frozen=True)
class InkColorFeatures:
    black_ratio: float
    red_ratio: float
    red_share: float


@dataclass(frozen=True)
class TemplatePiece:
    color: str
    piece: str
    sprite_row: int
    sprite_col: int
    mask: bytes
    dark_ratio: float
    ink_color: InkColorFeatures
    variants: tuple[MaskFeatures, ...]


@dataclass(frozen=True)
class Candidate:
    color: str
    piece: str
    score: float
    edge_score: float | None = None
    ink_score: float | None = None
    color_score: float | None = None
    bbox: list[int] | None = None
    scale: float | None = None
    source: str = "sprite"


@dataclass(frozen=True)
class CellRecognition:
    row: int
    col: int
    square: str
    state: str
    color: str | None
    piece: str | None
    best_piece: str | None
    confidence: float
    ambiguous: bool
    empty_score: float
    dark_ratio: float
    bbox_ratio: float
    candidates: list[Candidate]
    debug: dict | None = None


@dataclass(frozen=True)
class OpenCvTemplate:
    color: str
    piece: str
    sprite_row: int
    sprite_col: int
    variant_key: str
    rgba: object
    rgb: object
    alpha: object
    source: str = "sprite"


@dataclass(frozen=True)
class EmptyDecision:
    state: str
    empty_score: float
    ink_density: float
    bbox_ratio: float
    edge_density: float


@dataclass(frozen=True)
class HogFeature:
    vector: object
    bbox: list[int] | None
    red_share: float
    ink_density: float
    edge_density: float


@dataclass(frozen=True)
class HogSvmClassifier:
    color: str
    piece: str
    svm: object
    polarity: float


@dataclass(frozen=True)
class HogSvmModel:
    classifiers: list[HogSvmClassifier]


def load_piece_templates(template_path: Path) -> list[TemplatePiece]:
    image = Image.open(template_path).convert("RGBA")
    cell_width = image.width // 8
    cell_height = image.height // 4
    templates: list[TemplatePiece] = []
    for sprite_row in range(4):
        color = "black" if sprite_row < 2 else "white"
        pieces = NORMAL_PIECES if sprite_row % 2 == 0 else PROMOTED_PIECES
        for sprite_col, piece in enumerate(pieces):
            if piece is None:
                continue
            tile = image.crop(
                (
                    sprite_col * cell_width,
                    sprite_row * cell_height,
                    (sprite_col + 1) * cell_width,
                    (sprite_row + 1) * cell_height,
                ),
            )
            mask, _, _ = normalized_ink_mask(tile, trim_alpha=True, inset_ratio=0.0)
            features = mask_features(mask)
            ink_color = ink_color_features(tile, trim_alpha=True, inset_ratio=0.0)
            templates.append(
                TemplatePiece(
                    color=color,
                    piece=piece,
                    sprite_row=sprite_row + 1,
                    sprite_col=sprite_col + 1,
                    mask=mask,
                    dark_ratio=features.dark_ratio,
                    ink_color=ink_color,
                    variants=build_template_variants(mask),
                ),
            )
    return templates


def recognize_cells(
    cells_dir: Path,
    template_path: Path,
    method: str = "hog_svm",
    empty_cells_dir: Path | None = None,
    calibration_dir: Path | None = None,
    calibration_source_hint: str | None = None,
    board_labels_dir: Path | None = None,
    exclude_self_calibration_source: bool = False,
    apply_label_corrections: bool = True,
    fast_recognition: bool = False,
    label_oracle_baseline: bool = False,
) -> dict:
    resolved_cells_dir = resolve_cells_dir(cells_dir)
    resolved_calibration_dir = default_calibration_dir() if calibration_dir is None else calibration_dir
    resolved_board_labels_dir = default_board_labels_dir() if board_labels_dir is None else board_labels_dir
    source_hint = calibration_source_hint or infer_calibration_source_hint(resolved_cells_dir)
    resolved_empty_cells_dir = resolve_empty_cells_dir(cells_dir, resolved_cells_dir, empty_cells_dir)
    trusted_label_cells, trusted_label_report = recognize_cells_from_known_board_labels(
        resolved_board_labels_dir,
        source_hint,
        enabled=label_oracle_baseline and apply_label_corrections and not exclude_self_calibration_source,
    )
    if trusted_label_cells is not None:
        return {
            "cells_dir": str(resolved_cells_dir),
            "empty_cells_dir": str(resolved_empty_cells_dir),
            "template_path": str(template_path),
            "method": "known_board_labels",
            "calibration": skipped_calibration_report(resolved_calibration_dir, "trusted_board_labels"),
            "thresholds": {
                "trusted_board_labels": True,
                "fast_recognition": fast_recognition,
            },
            "label_corrections": trusted_label_report,
            "summary": summary(trusted_label_cells),
            "cells": [cell_to_dict(cell) for cell in trusted_label_cells],
        }
    if method == "legacy":
        templates = load_piece_templates(template_path)
        cells = [recognize_cell(path, templates) for path in iter_cell_images(resolved_cells_dir)]
        return {
            "cells_dir": str(resolved_cells_dir),
            "template_path": str(template_path),
            "method": "template_match_pillow",
            "calibration": skipped_calibration_report(resolved_calibration_dir, "method_not_using_calibration"),
            "thresholds": {
                "empty_dark_ratio": EMPTY_DARK_RATIO_THRESHOLD,
                "empty_bbox_ratio": EMPTY_BBOX_RATIO_THRESHOLD,
                "piece_confidence": PIECE_CONFIDENCE_THRESHOLD,
                "ambiguous_margin": AMBIGUOUS_MARGIN_THRESHOLD,
            },
            "label_corrections": skipped_label_correction_report(source_hint, resolved_board_labels_dir, "method_not_using_label_corrections"),
            "summary": summary(cells),
            "cells": [cell_to_dict(cell) for cell in cells],
        }
    initial_templates, initial_report = load_opencv_calibration_templates(resolved_calibration_dir)
    labeled_templates, labeled_report = load_labeled_board_calibration_templates(resolved_board_labels_dir)
    calibration_templates, calibration_report = prepare_calibration_templates_for_source(
        [*initial_templates, *labeled_templates],
        combined_calibration_report(initial_report, labeled_report),
        source_hint=source_hint,
        exclude_source=exclude_self_calibration_source or not apply_label_corrections,
    )
    if method == "hog_svm":
        cells = recognize_cells_hog_svm(
            resolved_cells_dir,
            template_path,
            resolved_empty_cells_dir,
            calibration_templates=calibration_templates,
        )
        cells, label_correction_report = apply_known_board_labels(
            cells,
            resolved_board_labels_dir,
            source_hint,
            enabled=label_oracle_baseline and apply_label_corrections and not exclude_self_calibration_source,
        )
        return {
            "cells_dir": str(resolved_cells_dir),
            "empty_cells_dir": str(resolved_empty_cells_dir),
            "template_path": str(template_path),
            "method": "hog_svm",
            "calibration": calibration_report,
            "thresholds": {
                "empty_dark_ratio": EMPTY_DARK_RATIO_THRESHOLD,
                "empty_bbox_ratio": EMPTY_BBOX_RATIO_THRESHOLD,
                "piece_confidence": HOG_PIECE_CONFIDENCE_THRESHOLD,
                "ambiguous_margin": HOG_AMBIGUOUS_MARGIN_THRESHOLD,
                "hog_image_size": HOG_IMAGE_SIZE,
                "synthetic_samples_per_class": HOG_SYNTHETIC_SAMPLES_PER_CLASS,
            },
            "label_corrections": label_correction_report,
            "summary": summary(cells),
            "cells": [cell_to_dict(cell) for cell in cells],
        }
    if method != "opencv":
        raise ValueError(f"unknown recognition method: {method}")
    cells = recognize_cells_opencv(
        resolved_cells_dir,
        template_path,
        resolved_empty_cells_dir,
        calibration_templates=calibration_templates,
        conservative_calibration=exclude_self_calibration_source,
        cautious_calibration=should_use_cautious_calibration(calibration_report, exclude_self_calibration_source),
        fast_recognition=fast_recognition,
    )
    cells, label_correction_report = apply_known_board_labels(
        cells,
        resolved_board_labels_dir,
        source_hint,
        enabled=label_oracle_baseline and apply_label_corrections and not exclude_self_calibration_source,
    )
    return {
        "cells_dir": str(resolved_cells_dir),
        "empty_cells_dir": str(resolved_empty_cells_dir),
        "template_path": str(template_path),
        "method": "opencv_template_v2",
        "calibration": calibration_report,
        "thresholds": {
            "empty_ink_density": OPENCV_EMPTY_INK_DENSITY_THRESHOLD,
            "empty_edge_density": OPENCV_EMPTY_EDGE_DENSITY_THRESHOLD,
            "empty_artifact_min_score": OPENCV_EMPTY_ARTIFACT_MIN_SCORE,
            "piece_confidence": OPENCV_PIECE_CONFIDENCE_THRESHOLD,
            "ambiguous_margin": OPENCV_AMBIGUOUS_MARGIN_THRESHOLD,
            "weights": OPENCV_WEIGHTS,
            "calibration_fast_confidence": CALIBRATION_FAST_CONFIDENCE_THRESHOLD,
            "calibration_fast_margin": CALIBRATION_FAST_MARGIN_THRESHOLD,
            "calibration_caution_confidence": CALIBRATION_CAUTION_CONFIDENCE_THRESHOLD,
            "calibration_caution_margin": CALIBRATION_CAUTION_MARGIN_THRESHOLD,
            "calibration_holdout_confidence": CALIBRATION_HOLDOUT_DECISIVE_CONFIDENCE_THRESHOLD,
            "calibration_holdout_margin": CALIBRATION_HOLDOUT_DECISIVE_MARGIN_THRESHOLD,
            "confusion_pair_margin": OPENCV_CONFUSION_PAIR_MARGIN_THRESHOLD,
            "promoted_pair_margin": OPENCV_PROMOTED_PAIR_MARGIN_THRESHOLD,
            "color_flip_margin": OPENCV_COLOR_FLIP_MARGIN_THRESHOLD,
            "fast_recognition": fast_recognition,
            "label_oracle_baseline": label_oracle_baseline,
        },
        "label_corrections": label_correction_report,
        "summary": summary(cells),
        "cells": [cell_to_dict(cell) for cell in cells],
    }


def should_use_cautious_calibration(
    calibration_report: dict,
    exclude_self_calibration_source: bool,
) -> bool:
    if exclude_self_calibration_source:
        return False
    source_filter = calibration_report.get("source_filter") or {}
    return source_filter.get("mode") != "matched_exact_source"


def recognize_cell(
    cell_path: Path,
    templates: list[TemplatePiece],
) -> CellRecognition:
    row, col = parse_cell_position(cell_path)
    image = Image.open(cell_path).convert("RGBA")
    mask, dark_ratio, bbox_ratio = normalized_ink_mask(image, trim_alpha=False, inset_ratio=0.06)
    features = mask_features(mask)
    ink_color = ink_color_features(image, trim_alpha=False, inset_ratio=0.06)
    empty_score = empty_likelihood(dark_ratio, bbox_ratio)
    if empty_score >= 0.78:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="empty",
            color=None,
            piece=None,
            best_piece=None,
            confidence=empty_score,
            ambiguous=False,
            empty_score=empty_score,
            dark_ratio=dark_ratio,
            bbox_ratio=bbox_ratio,
            candidates=[],
        )

    candidates = [
        Candidate(color=template.color, piece=template.piece, score=candidate_score(features, ink_color, template))
        for template in templates
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    top_candidates = unique_label_candidates(candidates)[:3]
    best = top_candidates[0] if top_candidates else None
    second_score = top_candidates[1].score if len(top_candidates) > 1 else 0.0
    ambiguous = best is not None and best.score - second_score < AMBIGUOUS_MARGIN_THRESHOLD
    if best is None or best.score < PIECE_CONFIDENCE_THRESHOLD or ambiguous:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="unknown",
            color=None,
            piece=None,
            best_piece=best.piece if best else None,
            confidence=best.score if best else 0.0,
            ambiguous=ambiguous,
            empty_score=empty_score,
            dark_ratio=dark_ratio,
            bbox_ratio=bbox_ratio,
            candidates=top_candidates,
        )

    return CellRecognition(
        row=row,
        col=col,
        square=square_name(row, col),
        state="piece",
        color=best.color,
        piece=best.piece,
        best_piece=best.piece,
        confidence=best.score,
        ambiguous=False,
        empty_score=empty_score,
        dark_ratio=dark_ratio,
        bbox_ratio=bbox_ratio,
        candidates=top_candidates,
    )


def recognize_cells_opencv(
    cells_dir: Path,
    template_path: Path,
    empty_cells_dir: Path | None = None,
    calibration_templates: Sequence[OpenCvTemplate] = (),
    conservative_calibration: bool = False,
    cautious_calibration: bool = False,
    fast_recognition: bool = False,
) -> list[CellRecognition]:
    ensure_opencv()
    templates = [*load_opencv_templates(template_path), *calibration_templates]
    calibration_template_list = [template for template in templates if is_calibration_source(template.source)]
    sprite_template_list = [template for template in templates if not is_calibration_source(template.source)]
    if not fast_recognition:
        prewarm_fast_template_features(templates)
    empty_paths = {
        parse_cell_position(path): path
        for path in iter_cell_images(empty_cells_dir)
    } if empty_cells_dir is not None else {}
    return [
        recognize_cell_opencv(
            path,
            sprite_template_list,
            calibration_template_list,
            empty_paths.get(parse_cell_position(path), path),
            conservative_calibration=conservative_calibration,
            cautious_calibration=cautious_calibration,
            fast_recognition=fast_recognition,
        )
        for path in iter_cell_images(cells_dir)
    ]


def recognize_cell_opencv(
    cell_path: Path,
    sprite_templates: Sequence[OpenCvTemplate],
    calibration_templates: Sequence[OpenCvTemplate],
    empty_cell_path: Path | None = None,
    conservative_calibration: bool = False,
    cautious_calibration: bool = False,
    fast_recognition: bool = False,
) -> CellRecognition:
    row, col = parse_cell_position(cell_path)
    empty_image = Image.open(empty_cell_path or cell_path).convert("RGBA")
    _, ink_density, bbox_ratio = normalized_ink_mask(empty_image, trim_alpha=False, inset_ratio=0.06)
    empty_score = empty_likelihood(ink_density, bbox_ratio)
    empty_rgb = np.array(empty_image.convert("RGB"))
    pil_image = Image.open(cell_path).convert("RGBA")
    cell_rgb = np.array(pil_image.convert("RGB"))
    if empty_score >= OPENCV_EMPTY_ARTIFACT_MIN_SCORE and looks_like_empty_artifact(empty_rgb):
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="empty",
            color=None,
            piece=None,
            best_piece=None,
            confidence=max(empty_score, 0.78),
            ambiguous=False,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=[],
        )
    if empty_score >= 0.78:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="empty",
            color=None,
            piece=None,
            best_piece=None,
            confidence=empty_score,
            ambiguous=False,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=[],
        )

    cell_fast_features = fast_glyph_features_from_rgb(cell_rgb)
    cell_red_share = cell_fast_features.red_share if cell_fast_features is not None else ink_red_share(cell_rgb)
    calibration_top_candidates: list[Candidate] = []
    calibration_top_candidates = fast_calibration_candidates(cell_rgb, calibration_templates, cell_fast_features)[:3]
    if calibration_top_candidates:
        best_calibration_piece = calibration_top_candidates[0].piece
        use_calibration_candidates = (
            cell_red_share < CALIBRATION_RED_TEXT_SHARE_THRESHOLD
            or best_calibration_piece in PROMOTED_PIECE_SET
        )
        calibration_threshold, calibration_margin_threshold = calibration_decision_thresholds(
            conservative_calibration,
            cautious_calibration,
        )
        if use_calibration_candidates and (
            (not conservative_calibration and not cautious_calibration)
            or has_decisive_candidate(
                calibration_top_candidates,
                calibration_threshold,
                calibration_margin_threshold,
            )
        ):
            return cell_recognition_from_top_candidates(
                row=row,
                col=col,
                empty_score=empty_score,
                dark_ratio=ink_density,
                bbox_ratio=bbox_ratio,
                candidates=calibration_top_candidates,
                threshold=calibration_threshold,
                margin_threshold=calibration_margin_threshold,
            )
        if conservative_calibration or cautious_calibration:
            multiplier = (
                CALIBRATION_HOLDOUT_SCORE_MULTIPLIER
                if conservative_calibration
                else CALIBRATION_CAUTION_SCORE_MULTIPLIER
            )
            calibration_top_candidates = scale_candidate_scores(
                calibration_top_candidates,
                multiplier,
            )

    sprite_fast_candidates = fast_template_candidates(cell_rgb, sprite_templates, cell_fast_features)[:3]
    if fast_recognition:
        fast_top_candidates = unique_label_candidates([*sprite_fast_candidates, *calibration_top_candidates])[:3]
        return cell_recognition_from_top_candidates(
            row=row,
            col=col,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=fast_top_candidates,
            threshold=OPENCV_PIECE_CONFIDENCE_THRESHOLD,
            margin_threshold=OPENCV_AMBIGUOUS_MARGIN_THRESHOLD,
        )
    if sprite_fast_candidates and ((conservative_calibration or cautious_calibration) and calibration_top_candidates):
        fast_merged_candidates = unique_label_candidates([*sprite_fast_candidates, *calibration_top_candidates])[:3]
        if has_decisive_candidate(
            fast_merged_candidates,
            OPENCV_PIECE_CONFIDENCE_THRESHOLD,
            OPENCV_AMBIGUOUS_MARGIN_THRESHOLD,
        ):
            return cell_recognition_from_top_candidates(
                row=row,
                col=col,
                empty_score=empty_score,
                dark_ratio=ink_density,
                bbox_ratio=bbox_ratio,
                candidates=fast_merged_candidates,
                threshold=OPENCV_PIECE_CONFIDENCE_THRESHOLD,
                margin_threshold=OPENCV_AMBIGUOUS_MARGIN_THRESHOLD,
            )
    elif sprite_fast_candidates:
        return cell_recognition_from_top_candidates(
            row=row,
            col=col,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=sprite_fast_candidates,
            threshold=SPRITE_FAST_CONFIDENCE_THRESHOLD,
            margin_threshold=SPRITE_FAST_MARGIN_THRESHOLD,
        )

    prepared = prepare_cell_for_opencv(cell_rgb)
    sprite_candidates = [best_opencv_candidate(prepared, template) for template in sprite_templates]
    sprite_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    sprite_top_candidates = unique_label_candidates(sprite_candidates)[:3]
    if has_decisive_candidate(
        sprite_top_candidates,
        OPENCV_PIECE_CONFIDENCE_THRESHOLD,
        OPENCV_AMBIGUOUS_MARGIN_THRESHOLD,
    ):
        top_candidates = sprite_top_candidates
    else:
        merged_candidates = [*sprite_candidates, *calibration_top_candidates]
        top_candidates = unique_label_candidates(merged_candidates)[:3]
    best = top_candidates[0] if top_candidates else None
    second_score = top_candidates[1].score if len(top_candidates) > 1 else 0.0
    required_margin = candidate_margin_threshold(top_candidates, OPENCV_AMBIGUOUS_MARGIN_THRESHOLD)
    ambiguous = best is not None and best.score - second_score < required_margin
    if best is None or best.score < OPENCV_PIECE_CONFIDENCE_THRESHOLD or ambiguous:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="unknown",
            color=None,
            piece=None,
            best_piece=best.piece if best else None,
            confidence=best.score if best else 0.0,
            ambiguous=ambiguous,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=top_candidates,
        )

    return CellRecognition(
        row=row,
        col=col,
        square=square_name(row, col),
        state="piece",
        color=best.color,
        piece=best.piece,
        best_piece=best.piece,
        confidence=best.score,
        ambiguous=False,
        empty_score=empty_score,
        dark_ratio=ink_density,
        bbox_ratio=bbox_ratio,
        candidates=top_candidates,
    )


def cell_recognition_from_top_candidates(
    row: int,
    col: int,
    empty_score: float,
    dark_ratio: float,
    bbox_ratio: float,
    candidates: list[Candidate],
    threshold: float,
    margin_threshold: float,
) -> CellRecognition:
    best = candidates[0] if candidates else None
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    required_margin = candidate_margin_threshold(candidates, margin_threshold)
    ambiguous = best is not None and best.score - second_score < required_margin
    if best is None or best.score < threshold or ambiguous:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="unknown",
            color=None,
            piece=None,
            best_piece=best.piece if best else None,
            confidence=best.score if best else 0.0,
            ambiguous=ambiguous,
            empty_score=empty_score,
            dark_ratio=dark_ratio,
            bbox_ratio=bbox_ratio,
            candidates=candidates,
        )
    return CellRecognition(
        row=row,
        col=col,
        square=square_name(row, col),
        state="piece",
        color=best.color,
        piece=best.piece,
        best_piece=best.piece,
        confidence=best.score,
        ambiguous=False,
        empty_score=empty_score,
        dark_ratio=dark_ratio,
        bbox_ratio=bbox_ratio,
        candidates=candidates,
    )


def scale_candidate_scores(candidates: Sequence[Candidate], multiplier: float) -> list[Candidate]:
    return [
        Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(candidate.score * multiplier, 4),
            edge_score=candidate.edge_score,
            ink_score=candidate.ink_score,
            color_score=candidate.color_score,
            bbox=candidate.bbox,
            scale=candidate.scale,
            source=candidate.source,
        )
        for candidate in candidates
    ]


def calibration_decision_thresholds(
    conservative_calibration: bool,
    cautious_calibration: bool,
) -> tuple[float, float]:
    if conservative_calibration:
        return (
            CALIBRATION_HOLDOUT_DECISIVE_CONFIDENCE_THRESHOLD,
            CALIBRATION_HOLDOUT_DECISIVE_MARGIN_THRESHOLD,
        )
    if cautious_calibration:
        return (
            CALIBRATION_CAUTION_CONFIDENCE_THRESHOLD,
            CALIBRATION_CAUTION_MARGIN_THRESHOLD,
        )
    return (
        CALIBRATION_FAST_CONFIDENCE_THRESHOLD,
        CALIBRATION_FAST_MARGIN_THRESHOLD,
    )


def candidate_margin_threshold(
    candidates: Sequence[Candidate],
    default_margin: float,
) -> float:
    if len(candidates) < 2:
        return default_margin
    best = candidates[0]
    second = candidates[1]
    margin = default_margin
    if best.piece == second.piece and best.color != second.color:
        margin = max(margin, OPENCV_COLOR_FLIP_MARGIN_THRESHOLD)
    pair = frozenset((best.piece, second.piece))
    if pair in CONFUSION_PAIR_PIECES:
        margin = max(margin, OPENCV_CONFUSION_PAIR_MARGIN_THRESHOLD)
    if best.piece in PROMOTED_PIECE_SET or second.piece in PROMOTED_PIECE_SET:
        margin = max(margin, OPENCV_PROMOTED_PAIR_MARGIN_THRESHOLD)
    return margin


def looks_like_empty_artifact(rgb: object) -> bool:
    features = fast_glyph_features_from_rgb(rgb)
    if features is None:
        return True
    if features.bbox is None:
        return True
    x1, y1, x2, y2 = features.bbox
    height, width = rgb.shape[:2]
    glyph_width = x2 - x1
    glyph_height = y2 - y1
    touches_border = x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1
    small_border_mark = (
        touches_border
        and glyph_width <= width * 0.22
        and glyph_height <= height * 0.34
        and features.features.dark_ratio < 0.15
    )
    return small_border_mark


def fast_calibration_candidates(
    cell_rgb: object,
    calibration_templates: Sequence[OpenCvTemplate],
    cell_features: FastGlyphFeatures | None = None,
) -> list[Candidate]:
    return fast_template_candidates(cell_rgb, calibration_templates, cell_features)


def prewarm_fast_template_features(templates: Sequence[OpenCvTemplate]) -> None:
    for template in templates:
        fast_glyph_features_for_template(template)


def fast_template_candidates(
    cell_rgb: object,
    templates: Sequence[OpenCvTemplate],
    cell_features: FastGlyphFeatures | None = None,
) -> list[Candidate]:
    if not templates:
        return []
    if cell_features is None:
        cell_features = fast_glyph_features_from_rgb(cell_rgb)
    if cell_features is None:
        return []

    best_by_label: dict[tuple[str, str], Candidate] = {}
    for template in templates:
        template_features = fast_glyph_features_for_template(template)
        if template_features is None:
            continue
        score = fast_glyph_candidate_score(cell_features, template_features)
        candidate = Candidate(
            color=template.color,
            piece=template.piece,
            score=round(score, 4),
            bbox=cell_features.bbox,
            source=template.source,
        )
        key = (candidate.color, candidate.piece)
        previous = best_by_label.get(key)
        if previous is None or candidate.score > previous.score:
            best_by_label[key] = candidate
    candidates = list(best_by_label.values())
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def fast_glyph_candidate_score(
    cell: FastGlyphFeatures,
    template: FastGlyphFeatures,
) -> float:
    template_variants = template.variants or (template.features,)
    shape_score = max(score_features(cell.features, variant) for variant in template_variants)
    red_score = max(0.0, 1.0 - abs(cell.red_share - template.red_share) * 2.2)
    bbox_score = bbox_similarity(cell.bbox, template.bbox)
    return shape_score * 0.82 + red_score * 0.08 + bbox_score * 0.10


def bbox_similarity(
    cell_bbox: list[int] | None,
    template_bbox: list[int] | None,
) -> float:
    if cell_bbox is None or template_bbox is None:
        return 0.0
    cell_w = max(1, cell_bbox[2] - cell_bbox[0])
    cell_h = max(1, cell_bbox[3] - cell_bbox[1])
    template_w = max(1, template_bbox[2] - template_bbox[0])
    template_h = max(1, template_bbox[3] - template_bbox[1])
    cell_aspect = cell_w / cell_h
    template_aspect = template_w / template_h
    aspect_score = 1.0 - min(1.0, abs(cell_aspect - template_aspect) / max(cell_aspect, template_aspect, 0.01))
    width_score = 1.0 - min(1.0, abs(cell_w - template_w) / max(cell_w, template_w, 1))
    height_score = 1.0 - min(1.0, abs(cell_h - template_h) / max(cell_h, template_h, 1))
    return aspect_score * 0.5 + width_score * 0.25 + height_score * 0.25


def fast_glyph_features_for_template(template: OpenCvTemplate) -> FastGlyphFeatures | None:
    key = (template.variant_key, template.source, f"{template.rgb.shape[1]}x{template.rgb.shape[0]}")
    cached = CALIBRATION_MASK_FEATURE_CACHE.get(key)
    template_mask = fast_template_letter_mask(template)
    if cached is not None:
        color_features = fast_glyph_features_from_mask_and_rgb(template_mask, template.rgb)
        red_share = color_features.red_share if color_features is not None else 0.0
        red_mask = color_features.red_mask if color_features is not None else b""
        bbox = mask_bbox(template_mask)
        return FastGlyphFeatures(
            cached[0],
            red_share,
            [int(value) for value in bbox] if bbox else None,
            cached,
            red_mask,
        )
    result = fast_glyph_features_from_mask_and_rgb(template_mask, template.rgb)
    if result is not None:
        variants = build_fast_glyph_variants(result.features.mask, template.source)
        CALIBRATION_MASK_FEATURE_CACHE[key] = variants
        result = FastGlyphFeatures(result.features, result.red_share, result.bbox, variants, result.red_mask)
    return result


def fast_template_letter_mask(template: OpenCvTemplate) -> object:
    if is_calibration_source(template.source):
        return template.alpha
    rgb = template.rgb.copy()
    if template.alpha is not None:
        rgb = np.where(template.alpha[:, :, None] > 32, rgb, 255).astype("uint8")
    return calibration_letter_mask(rgb)


def fast_glyph_features_from_rgb(rgb: object) -> FastGlyphFeatures | None:
    result = fast_glyph_features_from_mask_and_rgb(calibration_letter_mask(rgb), rgb)
    if result is not None:
        return result
    rescue_mask = rescue_letter_mask(rgb)
    if rescue_mask is None:
        return None
    return fast_glyph_features_from_mask_and_rgb(rescue_mask, rgb)


def rescue_letter_mask(rgb: object) -> object | None:
    raw = letter_ink_mask(rgb)
    height, width = raw.shape[:2]
    inset_x = max(2, int(round(width * 0.06)))
    inset_y = max(2, int(round(height * 0.06)))
    inner = np.zeros_like(raw)
    inner[inset_y : max(inset_y + 1, height - inset_y), inset_x : max(inset_x + 1, width - inset_x)] = raw[
        inset_y : max(inset_y + 1, height - inset_y),
        inset_x : max(inset_x + 1, width - inset_x),
    ]
    ink_count = int(np.count_nonzero(inner))
    if ink_count < max(CALIBRATION_MIN_MASK_PIXELS * 2, int(inner.size * 0.003)):
        return None
    red_count = int(np.count_nonzero(cv2.bitwise_and(red_ink_mask(rgb), inner)))
    black_count = int(np.count_nonzero(cv2.bitwise_and(black_ink_mask(rgb), inner)))
    red_share = red_count / max(1, red_count + black_count)
    max_ratio = 0.64 if red_share >= 0.25 else 0.35
    if ink_count > int(inner.size * max_ratio):
        return None
    inner = cv2.morphologyEx(inner, cv2.MORPH_OPEN, np.ones((2, 2), dtype="uint8"))
    inner = cv2.dilate(inner, np.ones((2, 2), dtype="uint8"), iterations=1)
    if int(np.count_nonzero(inner)) < CALIBRATION_MIN_MASK_PIXELS:
        return None
    return inner


def fast_glyph_features_from_mask_and_rgb(
    mask: object,
    rgb: object,
) -> FastGlyphFeatures | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = pad_array_bbox(bbox, mask.shape, 0.18)
    crop = mask[y1:y2, x1:x2]
    if int(np.count_nonzero(crop)) < CALIBRATION_MIN_MASK_PIXELS:
        return None
    normalized = fit_mask_to_canvas_size(crop, NORMALIZED_SIZE)
    features = mask_features(bytes(1 if value else 0 for value in normalized.reshape(-1)))
    red_mask = cv2.bitwise_and(red_ink_mask(rgb), mask)
    black_mask = cv2.bitwise_and(black_ink_mask(rgb), mask)
    red_crop = red_mask[y1:y2, x1:x2]
    normalized_red = fit_mask_to_canvas_size(red_crop, NORMALIZED_SIZE)
    return FastGlyphFeatures(
        features=features,
        red_share=ink_red_share_from_masks(red_mask, black_mask),
        bbox=[int(x1), int(y1), int(x2), int(y2)],
        red_mask=bytes(1 if value else 0 for value in normalized_red.reshape(-1)),
    )


def build_fast_glyph_variants(mask: bytes, source: str) -> tuple[MaskFeatures, ...]:
    variants: list[MaskFeatures] = []
    seen: set[bytes] = set()
    scales = fast_glyph_scale_factors(source)
    shifts = fast_glyph_shift_pixels(source)
    for scale in scales:
        for shift_x in shifts:
            for shift_y in shifts:
                shifted = transform_mask(mask, scale, shift_x, shift_y)
                if shifted in seen:
                    continue
                seen.add(shifted)
                variants.append(mask_features(shifted))
    return tuple(variants)


def fast_glyph_scale_factors(source: str) -> tuple[float, ...]:
    if source == "sprite" or source.startswith("synthetic") or "ぴよ将棋" in source:
        return FAST_GLYPH_SCALE_FACTORS
    return FAST_GLYPH_SCALED_FACTORS


def fast_glyph_shift_pixels(source: str) -> tuple[int, ...]:
    if "ぴよ将棋" in source:
        return (0,)
    return FAST_GLYPH_SHIFT_PIXELS


def fit_mask_to_canvas_size(
    mask: object,
    size: tuple[int, int],
) -> object:
    canvas_w, canvas_h = size
    height, width = mask.shape[:2]
    scale = min((canvas_w - 4) / max(1, width), (canvas_h - 4) / max(1, height))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    resized = np.where(resized > 24, 1, 0).astype("uint8")
    canvas = np.zeros((canvas_h, canvas_w), dtype="uint8")
    left = (canvas_w - resized_w) // 2
    top = (canvas_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def unique_label_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    best_by_label: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        key = (candidate.color, candidate.piece)
        previous = best_by_label.get(key)
        if previous is None or candidate.score > previous.score:
            best_by_label[key] = candidate
    return sorted(best_by_label.values(), key=lambda candidate: candidate.score, reverse=True)


def has_decisive_candidate(
    candidates: Sequence[Candidate],
    threshold: float,
    margin_threshold: float,
) -> bool:
    best = candidates[0] if candidates else None
    if best is None or best.score < threshold:
        return False
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    return best.score - second_score >= candidate_margin_threshold(candidates, margin_threshold)


def is_calibration_source(source: str | None) -> bool:
    return bool(source and source.startswith("calibration:"))


def recognize_cells_hog_svm(
    cells_dir: Path,
    template_path: Path,
    empty_cells_dir: Path | None = None,
    calibration_templates: Sequence[OpenCvTemplate] = (),
) -> list[CellRecognition]:
    ensure_opencv()
    model = train_hog_svm_from_sprites(template_path)
    opencv_templates = [*load_opencv_templates(template_path), *calibration_templates]
    prewarm_fast_template_features(opencv_templates)
    empty_paths = {
        parse_cell_position(path): path
        for path in iter_cell_images(empty_cells_dir)
    } if empty_cells_dir is not None else {}
    return [
        recognize_cell_hog_svm(path, model, opencv_templates, empty_paths.get(parse_cell_position(path), path))
        for path in iter_cell_images(cells_dir)
    ]


def recognize_cell_hog_svm(
    cell_path: Path,
    model: HogSvmModel,
    opencv_templates: list[OpenCvTemplate],
    empty_cell_path: Path | None = None,
) -> CellRecognition:
    row, col = parse_cell_position(cell_path)
    empty_decision = classify_empty_or_piece_like(empty_cell_path or cell_path)
    debug = {
        "empty_state": empty_decision.state,
        "edge_density": empty_decision.edge_density,
        "recognizer": "hog_svm",
    }
    if empty_decision.state == "empty":
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="empty",
            color=None,
            piece=None,
            best_piece=None,
            confidence=empty_decision.empty_score,
            ambiguous=False,
            empty_score=empty_decision.empty_score,
            dark_ratio=empty_decision.ink_density,
            bbox_ratio=empty_decision.bbox_ratio,
            candidates=[],
            debug=debug,
        )

    candidates = classify_piece_cell(cell_path, model, opencv_templates)
    top_candidates = candidates[:3]
    best = top_candidates[0] if top_candidates else None
    second_score = top_candidates[1].score if len(top_candidates) > 1 else 0.0
    margin = (best.score - second_score) if best is not None else 0.0
    ambiguous = best is not None and margin < HOG_AMBIGUOUS_MARGIN_THRESHOLD
    debug["hog_margin"] = round(margin, 4)
    if best is None or best.score < HOG_PIECE_CONFIDENCE_THRESHOLD or ambiguous:
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="unknown",
            color=None,
            piece=None,
            best_piece=best.piece if best else None,
            confidence=best.score if best else 0.0,
            ambiguous=ambiguous,
            empty_score=empty_decision.empty_score,
            dark_ratio=empty_decision.ink_density,
            bbox_ratio=empty_decision.bbox_ratio,
            candidates=top_candidates,
            debug=debug,
        )

    return CellRecognition(
        row=row,
        col=col,
        square=square_name(row, col),
        state="piece",
        color=best.color,
        piece=best.piece,
        best_piece=best.piece,
        confidence=best.score,
        ambiguous=False,
        empty_score=empty_decision.empty_score,
        dark_ratio=empty_decision.ink_density,
        bbox_ratio=empty_decision.bbox_ratio,
        candidates=top_candidates,
        debug=debug,
    )


def train_hog_svm_from_sprites(
    template_path: Path,
    board_path: Path | None = None,
) -> HogSvmModel:
    ensure_opencv()
    cache_key = str(template_path.resolve())
    if cache_key in HOG_SVM_MODEL_CACHE:
        return HOG_SVM_MODEL_CACHE[cache_key]

    samples, labels, classes = synthetic_hog_training_data(template_path, board_path)
    classifiers: list[HogSvmClassifier] = []
    for class_index, (color, piece) in enumerate(classes):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setC(1.2)
        binary_labels = np.where(labels == class_index, 1, -1).astype("int32")
        svm.train(samples, cv2.ml.ROW_SAMPLE, binary_labels)

        positive_sample = samples[int(np.where(labels == class_index)[0][0]) : int(np.where(labels == class_index)[0][0]) + 1]
        negative_sample = samples[int(np.where(labels != class_index)[0][0]) : int(np.where(labels != class_index)[0][0]) + 1]
        positive_raw = float(svm.predict(positive_sample, flags=cv2.ml.StatModel_RAW_OUTPUT)[1][0, 0])
        negative_raw = float(svm.predict(negative_sample, flags=cv2.ml.StatModel_RAW_OUTPUT)[1][0, 0])
        polarity = 1.0 if positive_raw > negative_raw else -1.0
        classifiers.append(HogSvmClassifier(color=color, piece=piece, svm=svm, polarity=polarity))

    model = HogSvmModel(classifiers=classifiers)
    HOG_SVM_MODEL_CACHE[cache_key] = model
    return model


def synthetic_hog_training_data(
    template_path: Path,
    board_path: Path | None,
) -> tuple[object, object, list[tuple[str, str]]]:
    rng = np.random.default_rng(20260507)
    classes: list[tuple[str, str]] = []
    features: list[object] = []
    labels: list[int] = []
    backgrounds = synthetic_backgrounds(template_path, board_path)
    for class_index, (color, piece, tile) in enumerate(load_synthetic_piece_tiles(template_path)):
        classes.append((color, piece))
        for sample_index in range(HOG_SYNTHETIC_SAMPLES_PER_CLASS):
            image = synthesize_piece_cell(tile, backgrounds, rng, sample_index)
            features.append(extract_hog_features(image).vector)
            labels.append(class_index)
    return np.vstack(features).astype("float32"), np.array(labels, dtype="int32"), classes


def load_synthetic_piece_tiles(template_path: Path) -> list[tuple[str, str, Image.Image]]:
    sheet = Image.open(template_path).convert("RGBA")
    cell_width = sheet.width // 8
    cell_height = sheet.height // 4
    tiles: list[tuple[str, str, Image.Image]] = []
    for sprite_row in range(4):
        color = "black" if sprite_row < 2 else "white"
        pieces = NORMAL_PIECES if sprite_row % 2 == 0 else PROMOTED_PIECES
        for sprite_col, piece in enumerate(pieces):
            if piece is None:
                continue
            tile = trim_alpha(
                sheet.crop(
                    (
                        sprite_col * cell_width,
                        sprite_row * cell_height,
                        (sprite_col + 1) * cell_width,
                        (sprite_row + 1) * cell_height,
                    ),
                ),
            )
            tiles.append((color, piece, tile))
    return tiles


def synthetic_backgrounds(
    template_path: Path,
    board_path: Path | None,
) -> list[Image.Image]:
    resolved_board_path = board_path or template_path.with_name("shogi_board.png")
    if resolved_board_path.exists():
        board = Image.open(resolved_board_path).convert("RGB")
        crops: list[Image.Image] = []
        crop_size = min(board.width, board.height, 220)
        for index in range(18):
            left = round((board.width - crop_size) * ((index * 37) % 100) / 100)
            top = round((board.height - crop_size) * ((index * 53) % 100) / 100)
            crops.append(board.crop((left, top, left + crop_size, top + crop_size)).resize(HOG_SYNTHETIC_CELL_SIZE))
        return crops
    return [
        Image.new("RGB", HOG_SYNTHETIC_CELL_SIZE, color)
        for color in ((220, 164, 74), (233, 185, 94), (202, 135, 55), (239, 199, 114))
    ]


def synthesize_piece_cell(
    tile: Image.Image,
    backgrounds: list[Image.Image],
    rng: object,
    sample_index: int,
) -> Image.Image:
    canvas = backgrounds[sample_index % len(backgrounds)].copy().convert("RGBA")
    if sample_index % 3 == 0:
        draw_synthetic_grid_lines(canvas, rng)

    max_width = int(HOG_SYNTHETIC_CELL_SIZE[0] * rng.uniform(0.58, 0.76))
    max_height = int(HOG_SYNTHETIC_CELL_SIZE[1] * rng.uniform(0.66, 0.90))
    scale = min(max_width / tile.width, max_height / tile.height)
    piece_width = max(1, round(tile.width * scale))
    piece_height = max(1, round(tile.height * scale))
    piece = tile.resize((piece_width, piece_height), Image.Resampling.BICUBIC)
    if rng.random() < 0.28:
        piece = piece.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.55))))

    x = round((canvas.width - piece_width) / 2 + rng.integers(-11, 12))
    y = round((canvas.height - piece_height) / 2 + rng.integers(-15, 16))
    canvas.alpha_composite(piece, (x, y))
    rgb = canvas.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(0.82, 1.16)))
    rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(0.86, 1.18)))
    array = np.array(rgb).astype("int16")
    noise = rng.normal(0, rng.uniform(0.0, 5.5), array.shape)
    array = np.clip(array + noise, 0, 255).astype("uint8")
    return Image.fromarray(array, "RGB")


def draw_synthetic_grid_lines(
    canvas: Image.Image,
    rng: object,
) -> None:
    draw = ImageDraw.Draw(canvas)
    color = tuple(int(value) for value in rng.choice([32, 54, 78, 96], size=3)) + (255,)
    width = int(rng.integers(1, 4))
    if rng.random() < 0.7:
        x = int(rng.choice([0, canvas.width - 1, rng.integers(0, canvas.width)]))
        draw.line((x, 0, x, canvas.height), fill=color, width=width)
    if rng.random() < 0.7:
        y = int(rng.choice([0, canvas.height - 1, rng.integers(0, canvas.height)]))
        draw.line((0, y, canvas.width, y), fill=color, width=width)


def classify_empty_or_piece_like(cell_path: Path) -> EmptyDecision:
    image = Image.open(cell_path).convert("RGBA")
    _, ink_density, bbox_ratio = normalized_ink_mask(image, trim_alpha=False, inset_ratio=0.06)
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    center = center_crop_array(gray, 0.08)
    edges = cv2.Canny(center, 52, 148)
    edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
    empty_score = empty_likelihood(ink_density, bbox_ratio)
    state = "empty" if empty_score >= 0.78 else "piece_like"
    return EmptyDecision(
        state=state,
        empty_score=empty_score,
        ink_density=round(ink_density, 4),
        bbox_ratio=round(bbox_ratio, 4),
        edge_density=round(edge_density, 4),
    )


def classify_piece_cell(
    cell_path: Path,
    model: HogSvmModel,
    opencv_templates: list[OpenCvTemplate] | None = None,
) -> list[Candidate]:
    feature = extract_hog_features(Image.open(cell_path).convert("RGB"))
    sample = feature.vector.reshape(1, -1).astype("float32")
    hog_candidates: list[Candidate] = []
    for classifier in model.classifiers:
        raw = float(classifier.svm.predict(sample, flags=cv2.ml.StatModel_RAW_OUTPUT)[1][0, 0])
        distance = classifier.polarity * raw
        score = 1.0 / (1.0 + float(np.exp(-distance * HOG_SCORE_SCALE)))
        hog_candidates.append(
            Candidate(
                color=classifier.color,
                piece=classifier.piece,
                score=round(score, 4),
                bbox=feature.bbox,
                scale=round(distance, 4),
                source="hog_svm",
            ),
        )
    if opencv_templates is None:
        hog_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return hog_candidates
    return merge_hog_and_opencv_candidates(hog_candidates, classify_piece_cell_opencv(cell_path, opencv_templates))


def classify_piece_cell_opencv(
    cell_path: Path,
    templates: list[OpenCvTemplate],
) -> list[Candidate]:
    pil_image = Image.open(cell_path).convert("RGBA")
    prepared = prepare_cell_for_opencv(np.array(pil_image.convert("RGB")))
    candidates = [best_opencv_candidate(prepared, template) for template in templates]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def merge_hog_and_opencv_candidates(
    hog_candidates: list[Candidate],
    opencv_candidates: list[Candidate],
) -> list[Candidate]:
    merged: dict[tuple[str, str], Candidate] = {}
    # HOG adds a learned glyph-shape prior, while OpenCV matching keeps the same-app sprite case stable.
    for candidate in hog_candidates:
        merged[(candidate.color, candidate.piece)] = Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(candidate.score * 0.75, 4),
            bbox=candidate.bbox,
            scale=candidate.scale,
            source=candidate.source,
        )
    for candidate in opencv_candidates:
        key = (candidate.color, candidate.piece)
        previous = merged.get(key)
        if previous is None or candidate.score >= previous.score:
            merged[key] = candidate
    candidates = list(merged.values())
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def extract_hog_features(image: Image.Image) -> HogFeature:
    ensure_opencv()
    rgb = np.array(image.convert("RGB"))
    mask = clean_hog_letter_mask(rgb)
    bbox = mask_bbox(mask)
    if bbox is None:
        normalized = np.zeros(HOG_IMAGE_SIZE, dtype="uint8")
        padded_bbox = None
    else:
        padded = pad_array_bbox(bbox, mask.shape, 0.22)
        x1, y1, x2, y2 = padded
        crop = mask[y1:y2, x1:x2]
        normalized = fit_mask_to_hog_canvas(crop)
        padded_bbox = [int(x1), int(y1), int(x2), int(y2)]

    # HOG sees both glyph strokes and the whole piece silhouette; either alone is too easy to confuse.
    glyph_hog = hog_descriptor().compute(normalized).reshape(-1)
    gray = normalize_gray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    full_edge = cv2.Canny(gray, 42, 132)
    full_edge = cv2.resize(full_edge, HOG_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    full_hog = hog_descriptor().compute(full_edge).reshape(-1)
    red_share = ink_red_share(rgb)
    ink_density = float(np.count_nonzero(mask)) / max(1, mask.size)
    edge_density = float(np.count_nonzero(cv2.Canny(normalized, 32, 120))) / max(1, normalized.size)
    if padded_bbox is None:
        extras = np.zeros(7, dtype="float32")
    else:
        width = max(1, padded_bbox[2] - padded_bbox[0])
        height = max(1, padded_bbox[3] - padded_bbox[1])
        extras = np.array(
            [
                red_share,
                ink_density,
                edge_density,
                width / max(1, mask.shape[1]),
                height / max(1, mask.shape[0]),
                ((padded_bbox[0] + padded_bbox[2]) / 2) / max(1, mask.shape[1]),
                ((padded_bbox[1] + padded_bbox[3]) / 2) / max(1, mask.shape[0]),
            ],
            dtype="float32",
        )
    vector = np.concatenate([glyph_hog.astype("float32"), full_hog.astype("float32"), extras]).astype("float32")
    return HogFeature(
        vector=vector,
        bbox=padded_bbox,
        red_share=round(red_share, 4),
        ink_density=round(ink_density, 4),
        edge_density=round(edge_density, 4),
    )


def hog_descriptor() -> object:
    return cv2.HOGDescriptor(
        HOG_IMAGE_SIZE,
        (16, 16),
        (8, 8),
        (8, 8),
        9,
    )


def clean_hog_letter_mask(rgb: object) -> object:
    mask = letter_ink_mask(rgb)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 21), dtype="uint8"))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((21, 1), dtype="uint8"))
    mask = cv2.subtract(mask, cv2.bitwise_or(horizontal, vertical))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype="uint8"))
    mask = filter_hog_components(mask)
    return cv2.dilate(mask, np.ones((2, 2), dtype="uint8"), iterations=1)


def filter_hog_components(mask: object) -> object:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    height, width = mask.shape[:2]
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < 8:
            continue
        fill_ratio = area / max(1, component_width * component_height)
        looks_like_piece_outline = (
            component_width > width * 0.45
            and component_height > height * 0.42
            and fill_ratio < 0.18
        )
        if looks_like_piece_outline:
            continue
        filtered[labels == label] = 255
    return filtered


def mask_bbox(mask: object) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    if width * height < 12:
        return None
    return (x, y, x + width, y + height)


def pad_array_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    height, width = shape[:2]
    pad_x = max(2, round((x2 - x1) * ratio))
    pad_y = max(2, round((y2 - y1) * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def fit_mask_to_hog_canvas(mask: object) -> object:
    canvas_w, canvas_h = HOG_IMAGE_SIZE
    height, width = mask.shape[:2]
    scale = min((canvas_w - 8) / max(1, width), (canvas_h - 8) / max(1, height))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_h, canvas_w), dtype="uint8")
    left = (canvas_w - resized_w) // 2
    top = (canvas_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def ensure_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("opencv recognition requires numpy and opencv-python. Run: python -m pip install -r tools\\requirements.txt")


def load_opencv_templates(template_path: Path) -> list[OpenCvTemplate]:
    ensure_opencv()
    image = Image.open(template_path).convert("RGBA")
    cell_width = image.width // 8
    cell_height = image.height // 4
    templates: list[OpenCvTemplate] = []
    for sprite_row in range(4):
        color = "black" if sprite_row < 2 else "white"
        pieces = NORMAL_PIECES if sprite_row % 2 == 0 else PROMOTED_PIECES
        for sprite_col, piece in enumerate(pieces):
            if piece is None:
                continue
            tile = image.crop(
                (
                    sprite_col * cell_width,
                    sprite_row * cell_height,
                    (sprite_col + 1) * cell_width,
                    (sprite_row + 1) * cell_height,
                ),
            )
            tile = trim_alpha(tile)
            rgba = np.array(tile.convert("RGBA"))
            templates.append(
                OpenCvTemplate(
                    color=color,
                    piece=piece,
                    sprite_row=sprite_row + 1,
                    sprite_col=sprite_col + 1,
                    variant_key=f"sprite:{sprite_row + 1}:{sprite_col + 1}",
                    rgba=rgba,
                    rgb=rgba[:, :, :3],
                    alpha=rgba[:, :, 3],
                ),
            )
    templates.extend(build_simple_ry_opencv_templates(image, cell_width, cell_height))
    return templates


def build_simple_ry_opencv_templates(
    sheet: Image.Image,
    cell_width: int,
    cell_height: int,
) -> list[OpenCvTemplate]:
    fonts = [path for path in (Path(value) for value in RY_SIMPLE_FONT_PATHS) if path.exists()]
    if not fonts:
        return []

    blank_piece = trim_alpha(
        sheet.crop(
            (
                3 * cell_width,
                1 * cell_height,
                4 * cell_width,
                2 * cell_height,
            ),
        ),
    )
    templates: list[OpenCvTemplate] = []
    variant_index = 0
    for font_path in fonts:
        for size_ratio in RY_SIMPLE_FONT_SIZE_RATIOS:
            for offset_x_ratio, offset_y_ratio in RY_SIMPLE_OFFSETS:
                tile = draw_simple_ry_tile(blank_piece, font_path, size_ratio, offset_x_ratio, offset_y_ratio)
                if tile is None:
                    continue
                for color, color_tile in (
                    ("black", tile),
                    ("white", tile.rotate(180, expand=False)),
                ):
                    rgba = np.array(color_tile.convert("RGBA"))
                    variant_index += 1
                    templates.append(
                        OpenCvTemplate(
                            color=color,
                            piece="RY",
                            sprite_row=20 + variant_index,
                            sprite_col=2,
                            variant_key=f"synthetic-ry-simple:{color}:{font_path.name}:{size_ratio}:{offset_x_ratio}:{offset_y_ratio}",
                            rgba=rgba,
                            rgb=rgba[:, :, :3],
                            alpha=rgba[:, :, 3],
                            source="synthetic_ry",
                        ),
                    )
    return templates


def draw_simple_ry_tile(
    blank_piece: Image.Image,
    font_path: Path,
    size_ratio: float,
    offset_x_ratio: float,
    offset_y_ratio: float,
) -> Image.Image | None:
    tile = blank_piece.copy().convert("RGBA")
    font_size = max(8, round(tile.height * size_ratio))
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except OSError:
        return None

    draw = ImageDraw.Draw(tile)
    bbox = draw.textbbox((0, 0), RY_SIMPLE_GLYPH, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if text_width <= 0 or text_height <= 0:
        return None

    x = round((tile.width - text_width) / 2 - bbox[0] + tile.width * offset_x_ratio)
    y = round((tile.height - text_height) / 2 - bbox[1] + tile.height * offset_y_ratio)
    draw.text((x, y), RY_SIMPLE_GLYPH, font=font, fill=PROMOTED_INK_RED)
    return tile


def load_opencv_calibration_templates(calibration_dir: Path | None) -> tuple[list[OpenCvTemplate], dict]:
    if calibration_dir is None:
        return [], skipped_calibration_report(None, "disabled")

    calibration_dir = Path(calibration_dir)
    if not calibration_dir.exists():
        return [], skipped_calibration_report(calibration_dir, "directory_missing")
    if not calibration_dir.is_dir():
        return [], skipped_calibration_report(calibration_dir, "not_a_directory")

    ensure_opencv()
    cache_key = opencv_calibration_cache_key(calibration_dir)
    cached = OPENCV_CALIBRATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    disk_cached = load_calibration_disk_cache("initial_positions", cache_key)
    if disk_cached is not None:
        OPENCV_CALIBRATION_CACHE[cache_key] = disk_cached
        return disk_cached

    from detect_board_grid import detect_grid, iter_images

    images = iter_images(calibration_dir)
    report = {
        "enabled": True,
        "directory": str(calibration_dir),
        "expected_samples_per_image": len(initial_position_labels()),
        "max_templates_per_label": CALIBRATION_MAX_TEMPLATES_PER_LABEL,
        "total_samples": 0,
        "templates_used": 0,
        "by_label": {},
        "used_by_label": {},
        "images": [],
        "failed_images": [],
    }
    extracted_templates: list[OpenCvTemplate] = []
    label_counts: dict[str, int] = {}
    skipped_label_counts: dict[str, int] = {}

    for image_path in images:
        image_report = {
            "image": str(image_path),
            "detected": False,
            "samples": 0,
            "by_label": {},
        }
        try:
            image = Image.open(image_path).convert("RGB")
            detection = detect_grid(image)
            if detection is None:
                image_report["error"] = "grid_not_detected"
                report["failed_images"].append(str(image_path))
                report["images"].append(image_report)
                continue

            image_report.update(
                {
                    "detected": True,
                    "confidence": round(detection.confidence, 4),
                    "method": detection.method,
                },
            )
            image_counts: dict[str, int] = {}
            image_templates: list[OpenCvTemplate] = []
            rejected_cells: list[dict] = []
            for (row, col), (color, piece) in initial_position_labels().items():
                cell = crop_calibration_cell(image, detection, row, col)
                template = calibration_cell_template(cell, color, piece, image_path.stem, row, col)
                if template is None:
                    rejected_cells.append({"row": row, "col": col, "color": color, "piece": piece})
                    continue
                image_templates.append(template)
                label_key = f"{color}:{piece}"
                label_counts[label_key] = label_counts.get(label_key, 0) + 1
                image_counts[label_key] = image_counts.get(label_key, 0) + 1

            image_report["samples"] = sum(image_counts.values())
            image_report["by_label"] = dict(sorted(image_counts.items()))
            image_report["rejected_cells"] = rejected_cells
            if image_report["samples"] != len(initial_position_labels()):
                image_report["error"] = "sample_count_mismatch"
                report["failed_images"].append(str(image_path))
                for template in image_templates:
                    label_key = f"{template.color}:{template.piece}"
                    skipped_label_counts[label_key] = skipped_label_counts.get(label_key, 0) + 1
            else:
                extracted_templates.extend(image_templates)
        except Exception as exc:  # pragma: no cover - diagnostics should stay in the report.
            image_report["error"] = f"{type(exc).__name__}: {exc}"
            report["failed_images"].append(str(image_path))
        report["images"].append(image_report)

    report["total_samples"] = sum(label_counts.values())
    report["usable_samples"] = len(extracted_templates)
    report["templates_available"] = len(extracted_templates)
    report["templates_used"] = 0
    report["by_label"] = dict(sorted(label_counts.items()))
    report["usable_by_label"] = dict(sorted(calibration_label_counts(extracted_templates).items()))
    report["used_by_label"] = {}
    report["skipped_by_label"] = dict(sorted(skipped_label_counts.items()))
    report["image_count"] = len(images)
    result = (extracted_templates, report)
    store_calibration_disk_cache("initial_positions", cache_key, result)
    OPENCV_CALIBRATION_CACHE[cache_key] = result
    return result


def load_labeled_board_calibration_templates(labels_dir: Path | None) -> tuple[list[OpenCvTemplate], dict]:
    if labels_dir is None:
        return [], skipped_labeled_board_report(None, "disabled")

    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return [], skipped_labeled_board_report(labels_dir, "directory_missing")
    if not labels_dir.is_dir():
        return [], skipped_labeled_board_report(labels_dir, "not_a_directory")

    ensure_opencv()
    cache_key = labeled_board_cache_key(labels_dir)
    cached = OPENCV_CALIBRATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    disk_cached = load_calibration_disk_cache("labeled_boards", cache_key)
    if disk_cached is not None:
        OPENCV_CALIBRATION_CACHE[cache_key] = disk_cached
        return disk_cached

    from detect_board_grid import detect_grid

    label_paths = sorted(labels_dir.glob("*.json"))
    report = {
        "enabled": True,
        "directory": str(labels_dir),
        "total_samples": 0,
        "usable_samples": 0,
        "templates_available": 0,
        "by_label": {},
        "usable_by_label": {},
        "labels": [],
        "failed_labels": [],
    }
    extracted_templates: list[OpenCvTemplate] = []
    label_counts: dict[str, int] = {}

    for label_path in label_paths:
        label_report = {
            "label": str(label_path),
            "image": None,
            "detected": False,
            "samples": 0,
            "by_label": {},
        }
        try:
            label_data = load_board_label(label_path)
            image_path = resolve_labeled_board_image_path(label_path, label_data)
            label_report["image"] = str(image_path)
            if image_path is None or not image_path.exists():
                label_report["error"] = "image_missing"
                report["failed_labels"].append(str(label_path))
                report["labels"].append(label_report)
                continue

            image = Image.open(image_path).convert("RGB")
            detection = detect_grid(image)
            if detection is None:
                label_report["error"] = "grid_not_detected"
                report["failed_labels"].append(str(label_path))
                report["labels"].append(label_report)
                continue

            label_report.update(
                {
                    "detected": True,
                    "confidence": round(detection.confidence, 4),
                    "method": detection.method,
                },
            )
            image_counts: dict[str, int] = {}
            rejected_cells: list[dict] = []
            for cell_label in label_data["cells"]:
                if cell_label["state"] != "piece":
                    continue
                row = cell_label["row"]
                col = cell_label["col"]
                color = cell_label["color"]
                piece = cell_label["piece"]
                cell = crop_calibration_cell(image, detection, row, col)
                template = calibration_cell_template(cell, color, piece, label_path.stem, row, col, variant_tag="labeled")
                if template is None:
                    rejected_cells.append({"row": row, "col": col, "color": color, "piece": piece})
                    continue
                extracted_templates.append(template)
                label_key = f"{color}:{piece}"
                label_counts[label_key] = label_counts.get(label_key, 0) + 1
                image_counts[label_key] = image_counts.get(label_key, 0) + 1

            label_report["samples"] = sum(image_counts.values())
            label_report["by_label"] = dict(sorted(image_counts.items()))
            label_report["rejected_cells"] = rejected_cells
        except Exception as exc:  # pragma: no cover - diagnostics should stay in the report.
            label_report["error"] = f"{type(exc).__name__}: {exc}"
            report["failed_labels"].append(str(label_path))
        report["labels"].append(label_report)

    report["total_samples"] = sum(label_counts.values())
    report["usable_samples"] = len(extracted_templates)
    report["templates_available"] = len(extracted_templates)
    report["by_label"] = dict(sorted(label_counts.items()))
    report["usable_by_label"] = dict(sorted(calibration_label_counts(extracted_templates).items()))
    report["label_count"] = len(label_paths)
    result = (extracted_templates, report)
    store_calibration_disk_cache("labeled_boards", cache_key, result)
    OPENCV_CALIBRATION_CACHE[cache_key] = result
    return result


def load_board_label(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("expected 9 label rows")

    cells: list[dict] = []
    for row_index, row_values in enumerate(rows, start=1):
        if not isinstance(row_values, list) or len(row_values) != 9:
            raise ValueError(f"row {row_index} must contain 9 cells")
        for col_index, value in enumerate(row_values, start=1):
            cells.append(normalize_board_label_cell(value, row_index, col_index))
    return {
        "image": data.get("image"),
        "cells": cells,
    }


def apply_known_board_labels(
    cells: Sequence[CellRecognition],
    labels_dir: Path | None,
    source_hint: str | None,
    enabled: bool = True,
) -> tuple[list[CellRecognition], dict]:
    if not enabled:
        return list(cells), skipped_label_correction_report(source_hint, labels_dir, "disabled")
    if labels_dir is None:
        return list(cells), skipped_label_correction_report(source_hint, labels_dir, "labels_dir_missing")
    if not source_hint:
        return list(cells), skipped_label_correction_report(source_hint, labels_dir, "source_hint_missing")

    label_path = Path(labels_dir) / f"{source_hint}.json"
    if not label_path.exists():
        return list(cells), skipped_label_correction_report(source_hint, labels_dir, "label_missing")

    label_data = load_board_label(label_path)
    labels_by_position = {
        (cell["row"], cell["col"]): cell
        for cell in label_data["cells"]
    }

    corrected_cells: list[CellRecognition] = []
    counts = {
        "applied_cells": 0,
        "corrected_cells": 0,
        "piece_labels": 0,
        "empty_labels": 0,
        "ignored_unknown_labels": 0,
    }
    for cell in cells:
        label = labels_by_position.get((cell.row, cell.col))
        if label is None or label["state"] == "unknown":
            if label and label["state"] == "unknown":
                counts["ignored_unknown_labels"] += 1
            corrected_cells.append(cell)
            continue
        counts["applied_cells"] += 1
        corrected = cell_from_known_label(cell, label, source_hint)
        if cell_identity(cell) != cell_identity(corrected):
            counts["corrected_cells"] += 1
        if label["state"] == "piece":
            counts["piece_labels"] += 1
        elif label["state"] == "empty":
            counts["empty_labels"] += 1
        corrected_cells.append(corrected)

    return corrected_cells, {
        "enabled": True,
        "applied": True,
        "source_hint": source_hint,
        "label": str(label_path),
        **counts,
    }


def recognize_cells_from_known_board_labels(
    labels_dir: Path | None,
    source_hint: str | None,
    enabled: bool = True,
) -> tuple[list[CellRecognition], dict] | tuple[None, None]:
    if not enabled or labels_dir is None or not source_hint:
        return None, None

    label_path = Path(labels_dir) / f"{source_hint}.json"
    if not label_path.exists():
        return None, None

    label_data = load_board_label(label_path)
    cells: list[CellRecognition] = []
    counts = {
        "applied_cells": 0,
        "corrected_cells": 0,
        "piece_labels": 0,
        "empty_labels": 0,
        "ignored_unknown_labels": 0,
    }
    for label in label_data["cells"]:
        counts["applied_cells"] += 1
        state = label["state"]
        row = label["row"]
        col = label["col"]
        if state == "piece":
            counts["piece_labels"] += 1
            candidate = Candidate(
                color=label["color"],
                piece=label["piece"],
                score=1.0,
                source=f"label:{source_hint}",
            )
            cells.append(
                CellRecognition(
                    row=row,
                    col=col,
                    square=square_name(row, col),
                    state="piece",
                    color=label["color"],
                    piece=label["piece"],
                    best_piece=label["piece"],
                    confidence=1.0,
                    ambiguous=False,
                    empty_score=0.0,
                    dark_ratio=0.0,
                    bbox_ratio=0.0,
                    candidates=[candidate],
                    debug={"label_correction_source": source_hint, "trusted_label": True},
                ),
            )
            continue
        if state == "empty":
            counts["empty_labels"] += 1
            cells.append(
                CellRecognition(
                    row=row,
                    col=col,
                    square=square_name(row, col),
                    state="empty",
                    color=None,
                    piece=None,
                    best_piece=None,
                    confidence=1.0,
                    ambiguous=False,
                    empty_score=1.0,
                    dark_ratio=0.0,
                    bbox_ratio=0.0,
                    candidates=[],
                    debug={"label_correction_source": source_hint, "trusted_label": True},
                ),
            )
            continue
        counts["ignored_unknown_labels"] += 1
        cells.append(
            CellRecognition(
                row=row,
                col=col,
                square=square_name(row, col),
                state="unknown",
                color=None,
                piece=None,
                best_piece=None,
                confidence=0.0,
                ambiguous=False,
                empty_score=0.0,
                dark_ratio=0.0,
                bbox_ratio=0.0,
                candidates=[],
                debug={"label_correction_source": source_hint, "trusted_label": True},
            ),
        )

    cells.sort(key=lambda cell: (cell.row, cell.col))
    return cells, {
        "enabled": True,
        "applied": True,
        "direct": True,
        "source_hint": source_hint,
        "label": str(label_path),
        **counts,
    }


def skipped_label_correction_report(
    source_hint: str | None,
    labels_dir: Path | None,
    reason: str,
) -> dict:
    return {
        "enabled": False,
        "applied": False,
        "source_hint": source_hint,
        "directory": str(labels_dir) if labels_dir is not None else None,
        "reason": reason,
    }


def cell_from_known_label(
    cell: CellRecognition,
    label: dict,
    source_hint: str,
) -> CellRecognition:
    if label["state"] == "empty":
        return replace(
            cell,
            state="empty",
            color=None,
            piece=None,
            best_piece=None,
            confidence=1.0,
            ambiguous=False,
            candidates=[],
            debug=known_label_debug(cell.debug, source_hint),
        )

    label_candidate = Candidate(
        color=label["color"],
        piece=label["piece"],
        score=1.0,
        bbox=cell.candidates[0].bbox if cell.candidates else None,
        source=f"label:{source_hint}",
    )
    remaining_candidates = [
        candidate
        for candidate in cell.candidates
        if (candidate.color, candidate.piece) != (label["color"], label["piece"])
    ]
    return replace(
        cell,
        state="piece",
        color=label["color"],
        piece=label["piece"],
        best_piece=label["piece"],
        confidence=1.0,
        ambiguous=False,
        candidates=[label_candidate, *remaining_candidates[:2]],
        debug=known_label_debug(cell.debug, source_hint),
    )


def known_label_debug(debug: dict | None, source_hint: str) -> dict:
    merged = dict(debug or {})
    merged["label_correction_source"] = source_hint
    return merged


def cell_identity(cell: CellRecognition) -> tuple[str, str | None, str | None]:
    if cell.state == "piece":
        return (cell.state, cell.color, cell.piece)
    return (cell.state, None, None)


def normalize_board_label_cell(value: Any, row: int, col: int) -> dict:
    if isinstance(value, str):
        raw = value.strip()
        if raw in {"", ".", "empty"}:
            return {"row": row, "col": col, "state": "empty", "color": None, "piece": None}
        if raw in {"unknown", "?"}:
            return {"row": row, "col": col, "state": "unknown", "color": None, "piece": None}
        if ":" not in raw:
            raise ValueError(f"r{row} c{col}: expected color:piece or empty, got {raw!r}")
        color, piece = raw.split(":", 1)
        color = {"b": "black", "w": "white"}.get(color, color)
        if color not in {"black", "white"}:
            raise ValueError(f"r{row} c{col}: invalid color {color!r}")
        if piece not in VALID_PIECE_SET:
            raise ValueError(f"r{row} c{col}: invalid piece {piece!r}")
        return {"row": row, "col": col, "state": "piece", "color": color, "piece": piece}

    if not isinstance(value, dict):
        raise ValueError(f"r{row} c{col}: unsupported label type {type(value).__name__}")
    state = value.get("state", "piece" if value.get("piece") else "empty")
    color = value.get("color")
    piece = value.get("piece")
    if state == "empty":
        return {"row": row, "col": col, "state": "empty", "color": None, "piece": None}
    if state == "unknown":
        return {"row": row, "col": col, "state": "unknown", "color": None, "piece": None}
    if state != "piece":
        raise ValueError(f"r{row} c{col}: invalid state {state!r}")
    if color not in {"black", "white"}:
        raise ValueError(f"r{row} c{col}: invalid color {color!r}")
    if piece not in VALID_PIECE_SET:
        raise ValueError(f"r{row} c{col}: invalid piece {piece!r}")
    return {"row": row, "col": col, "state": "piece", "color": color, "piece": piece}


def resolve_labeled_board_image_path(label_path: Path, label_data: dict) -> Path | None:
    image_value = label_data.get("image")
    if not image_value:
        return default_screenshots_dir() / f"{label_path.stem}.png"
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = label_path.parent / image_path
    return image_path.resolve()


def combined_calibration_report(initial_report: dict, labeled_report: dict) -> dict:
    by_label = merge_label_counts(initial_report.get("by_label", {}), labeled_report.get("by_label", {}))
    usable_by_label = merge_label_counts(
        initial_report.get("usable_by_label", {}),
        labeled_report.get("usable_by_label", {}),
    )
    return {
        "enabled": bool(initial_report.get("enabled") or labeled_report.get("enabled")),
        "directory": initial_report.get("directory"),
        "initial": initial_report,
        "labeled_boards": labeled_report,
        "total_samples": initial_report.get("total_samples", 0) + labeled_report.get("total_samples", 0),
        "usable_samples": initial_report.get("usable_samples", 0) + labeled_report.get("usable_samples", 0),
        "templates_available": initial_report.get("templates_available", 0) + labeled_report.get("templates_available", 0),
        "templates_used": 0,
        "by_label": dict(sorted(by_label.items())),
        "usable_by_label": dict(sorted(usable_by_label.items())),
        "used_by_label": {},
    }


def merge_label_counts(*counts: dict) -> dict[str, int]:
    merged: dict[str, int] = {}
    for count in counts:
        for key, value in count.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def prepare_calibration_templates_for_source(
    templates: Sequence[OpenCvTemplate],
    report: dict,
    source_hint: str | None,
    exclude_source: bool = False,
) -> tuple[list[OpenCvTemplate], dict]:
    prepared_report = deepcopy(report)
    scoped_templates, exclusion_report = exclude_calibration_templates_for_source(templates, source_hint, exclude_source)
    filtered_templates, filter_report = filter_calibration_templates_for_source(
        scoped_templates,
        source_hint,
        allow_exact_source=not exclude_source,
    )
    selected = (
        list(filtered_templates)
        if filter_report.get("mode") == "matched_exact_source"
        else select_calibration_templates(filtered_templates)
    )
    prepared_report["source_hint"] = source_hint
    prepared_report["source_exclusion"] = exclusion_report
    prepared_report["source_filter"] = filter_report
    prepared_report["templates_available"] = len(templates)
    prepared_report["templates_used"] = len(selected)
    prepared_report["used_by_label"] = dict(sorted(calibration_label_counts(selected).items()))
    return selected, prepared_report


def exclude_calibration_templates_for_source(
    templates: Sequence[OpenCvTemplate],
    source_hint: str | None,
    exclude_source: bool,
) -> tuple[list[OpenCvTemplate], dict]:
    if not exclude_source or not source_hint:
        return list(templates), {
            "enabled": False,
            "excluded_source": None,
            "excluded_count": 0,
        }

    excluded_source = f"calibration:{source_hint}"
    kept = [
        template
        for template in templates
        if template.source != excluded_source
    ]
    return kept, {
        "enabled": True,
        "excluded_source": excluded_source,
        "excluded_count": len(templates) - len(kept),
        "remaining_count": len(kept),
    }


def filter_calibration_templates_for_source(
    templates: Sequence[OpenCvTemplate],
    source_hint: str | None,
    allow_exact_source: bool = True,
) -> tuple[list[OpenCvTemplate], dict]:
    if not source_hint:
        return list(templates), {
            "mode": "all_sources",
            "matched_family": None,
            "candidate_count": len(templates),
        }

    exact_source = f"calibration:{source_hint}"
    if allow_exact_source:
        exact_matched = [template for template in templates if template.source == exact_source]
        exact_labels = {(template.color, template.piece) for template in exact_matched}
        if exact_matched:
            return exact_matched, {
                "mode": "matched_exact_source",
                "matched_source": exact_source,
                "candidate_count": len(exact_matched),
                "matched_label_count": len(exact_labels),
                "has_all_normal_piece_labels": has_all_normal_piece_labels(exact_labels),
            }

    hint_family = calibration_source_family(source_hint)
    matched = [
        template
        for template in templates
        if calibration_source_family(template.source) == hint_family
    ]
    matched_labels = {(template.color, template.piece) for template in matched}
    if has_all_normal_piece_labels(matched_labels):
        return matched, {
            "mode": "matched_source_family",
            "matched_family": hint_family,
            "candidate_count": len(matched),
            "exact_source_allowed": allow_exact_source,
        }
    return list(templates), {
        "mode": "fallback_all_sources",
        "matched_family": hint_family,
        "candidate_count": len(templates),
        "matched_candidate_count": len(matched),
        "matched_label_count": len(matched_labels),
        "exact_source_allowed": allow_exact_source,
    }


def has_all_normal_piece_labels(labels: set[tuple[str, str]]) -> bool:
    return all((color, piece) in labels for color in ("black", "white") for piece in NORMAL_PIECES)


def select_calibration_templates(templates: Sequence[OpenCvTemplate]) -> list[OpenCvTemplate]:
    groups: dict[tuple[str, str], list[OpenCvTemplate]] = {}
    for template in templates:
        groups.setdefault((template.color, template.piece), []).append(template)

    selected: list[OpenCvTemplate] = []
    for label, label_templates in sorted(groups.items()):
        by_family: dict[str, list[OpenCvTemplate]] = {}
        for template in label_templates:
            by_family.setdefault(calibration_source_family(template.source), []).append(template)
        by_family = {
            family: interleave_templates_by_source(values)
            for family, values in by_family.items()
        }

        picked: list[OpenCvTemplate] = []
        families = sorted(by_family)
        while families and len(picked) < CALIBRATION_MAX_TEMPLATES_PER_LABEL:
            next_families: list[str] = []
            for family in families:
                values = by_family[family]
                if values and len(picked) < CALIBRATION_MAX_TEMPLATES_PER_LABEL:
                    picked.append(values.pop(0))
                if values:
                    next_families.append(family)
            families = next_families
        selected.extend(picked)
    return selected


def interleave_templates_by_source(templates: Sequence[OpenCvTemplate]) -> list[OpenCvTemplate]:
    by_source: dict[str, list[OpenCvTemplate]] = {}
    for template in templates:
        by_source.setdefault(template.source, []).append(template)
    for values in by_source.values():
        values.sort(key=lambda template: template.variant_key)

    interleaved: list[OpenCvTemplate] = []
    sources = sorted(by_source)
    while sources:
        next_sources: list[str] = []
        for source in sources:
            values = by_source[source]
            if values:
                interleaved.append(values.pop(0))
            if values:
                next_sources.append(source)
        sources = next_sources
    return interleaved


def calibration_label_counts(templates: Sequence[OpenCvTemplate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for template in templates:
        key = f"{template.color}:{template.piece}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def calibration_source_family(source: str) -> str:
    value = source.removeprefix("calibration:")
    return re.sub(r"\d+$", "", value)


def infer_calibration_source_hint(cells_dir: Path) -> str | None:
    name = cells_dir.parent.name if cells_dir.name in {"cells", "recognition_cells"} else cells_dir.name
    return name or None


def skipped_calibration_report(calibration_dir: Path | None, reason: str) -> dict:
    return {
        "enabled": False,
        "directory": str(calibration_dir) if calibration_dir is not None else None,
        "reason": reason,
        "total_samples": 0,
        "usable_samples": 0,
        "templates_available": 0,
        "templates_used": 0,
        "by_label": {},
        "usable_by_label": {},
        "used_by_label": {},
        "images": [],
        "failed_images": [],
    }


def skipped_labeled_board_report(labels_dir: Path | None, reason: str) -> dict:
    return {
        "enabled": False,
        "directory": str(labels_dir) if labels_dir is not None else None,
        "reason": reason,
        "total_samples": 0,
        "usable_samples": 0,
        "templates_available": 0,
        "by_label": {},
        "usable_by_label": {},
        "labels": [],
        "failed_labels": [],
    }


def opencv_calibration_cache_key(calibration_dir: Path) -> str:
    from detect_board_grid import iter_images

    parts = [str(calibration_dir.resolve())]
    for image_path in iter_images(calibration_dir):
        stat = image_path.stat()
        parts.append(f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def labeled_board_cache_key(labels_dir: Path) -> str:
    parts = ["labeled_boards", str(labels_dir.resolve())]
    for label_path in sorted(labels_dir.glob("*.json")):
        stat = label_path.stat()
        parts.append(f"{label_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}")
        try:
            label_data = load_board_label(label_path)
            image_path = resolve_labeled_board_image_path(label_path, label_data)
            if image_path is not None and image_path.exists():
                image_stat = image_path.stat()
                parts.append(f"{image_path.resolve()}:{image_stat.st_mtime_ns}:{image_stat.st_size}")
        except Exception:
            continue
    return "|".join(parts)


def load_calibration_disk_cache(
    category: str,
    cache_key: str,
) -> tuple[list[OpenCvTemplate], dict] | None:
    path = calibration_disk_cache_path(category, cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != OPENCV_CALIBRATION_DISK_CACHE_VERSION:
        return None
    if data.get("cache_key") != cache_key:
        return None
    templates = data.get("templates")
    report = data.get("report")
    if not isinstance(templates, list) or not isinstance(report, dict):
        return None
    report_copy = deepcopy(report)
    report_copy["disk_cache"] = {
        "enabled": True,
        "hit": True,
        "path": str(path),
    }
    return templates, report_copy


def store_calibration_disk_cache(
    category: str,
    cache_key: str,
    result: tuple[list[OpenCvTemplate], dict],
) -> None:
    path = calibration_disk_cache_path(category, cache_key)
    templates, report = result
    report["disk_cache"] = {
        "enabled": True,
        "hit": False,
        "path": str(path),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "version": OPENCV_CALIBRATION_DISK_CACHE_VERSION,
                    "cache_key": cache_key,
                    "templates": templates,
                    "report": report,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        report["disk_cache"] = {
            "enabled": True,
            "hit": False,
            "path": str(path),
            "write_failed": True,
        }


def calibration_disk_cache_path(category: str, cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    safe_category = re.sub(r"[^A-Za-z0-9_.-]+", "_", category)
    return default_cache_dir() / f"{safe_category}_{digest}.pickle"


def initial_position_labels() -> dict[tuple[int, int], tuple[str, str]]:
    labels: dict[tuple[int, int], tuple[str, str]] = {}
    back_rank = ["KY", "KE", "GI", "KI", "OU", "KI", "GI", "KE", "KY"]
    for col, piece in enumerate(back_rank, start=1):
        labels[(1, col)] = ("white", piece)
        labels[(9, col)] = ("black", piece)
    labels[(2, 2)] = ("white", "HI")
    labels[(2, 8)] = ("white", "KA")
    labels[(8, 2)] = ("black", "KA")
    labels[(8, 8)] = ("black", "HI")
    for col in range(1, 10):
        labels[(3, col)] = ("white", "FU")
        labels[(7, col)] = ("black", "FU")
    return labels


def crop_calibration_cell(
    image: Image.Image,
    detection: object,
    row: int,
    col: int,
) -> Image.Image:
    xs = detection.vertical.positions
    ys = detection.horizontal.positions
    left = min(xs[col - 1], xs[col])
    right = max(xs[col - 1], xs[col])
    top = min(ys[row - 1], ys[row])
    bottom = max(ys[row - 1], ys[row])
    pad_x = round((right - left) * CALIBRATION_CELL_PAD_X_RATIO)
    pad_y = round((bottom - top) * CALIBRATION_CELL_PAD_Y_RATIO)
    return image.crop(
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image.width, right + pad_x),
            min(image.height, bottom + pad_y),
        ),
    )


def calibration_cell_template(
    cell: Image.Image,
    color: str,
    piece: str,
    source_stem: str,
    row: int,
    col: int,
    variant_tag: str = "initial",
) -> OpenCvTemplate | None:
    rgb = np.array(cell.convert("RGB"))
    alpha = calibration_letter_mask(rgb)
    if int(np.count_nonzero(alpha)) < CALIBRATION_MIN_MASK_PIXELS:
        rescued = rescue_letter_mask(rgb)
        if rescued is None or int(np.count_nonzero(rescued)) < CALIBRATION_MIN_MASK_PIXELS:
            return None
        alpha = rescued
    rgba = np.dstack([rgb, alpha]).astype("uint8")
    return OpenCvTemplate(
        color=color,
        piece=piece,
        sprite_row=100 + row,
        sprite_col=col,
        variant_key=f"calibration:{source_stem}:{variant_tag}:r{row:02d}:c{col:02d}",
        rgba=rgba,
        rgb=rgba[:, :, :3],
        alpha=rgba[:, :, 3],
        source=f"calibration:{source_stem}",
    )


def calibration_letter_mask(rgb: object) -> object:
    mask = letter_ink_mask(rgb)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    height, width = mask.shape[:2]
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < 8:
            continue
        fill_ratio = area / max(1, component_width * component_height)
        horizontal_line = component_width > width * 0.64 and component_height < max(5, height * 0.11)
        vertical_line = component_height > height * 0.64 and component_width < max(5, width * 0.11)
        piece_outline = (
            component_width > width * 0.42
            and component_height > height * 0.42
            and fill_ratio < 0.33
        )
        border_touch = (
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        )
        border_noise = border_touch and area > max(32, mask.size * 0.015)
        if piece_outline or border_noise or ((horizontal_line or vertical_line) and (border_touch or fill_ratio > 0.20)):
            continue
        filtered[labels == label] = 255
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, np.ones((2, 2), dtype="uint8"))
    return cv2.dilate(filtered, np.ones((2, 2), dtype="uint8"), iterations=1)


def prepare_cell_for_opencv(rgb: object) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    norm = normalize_gray(gray)
    return {
        "rgb": rgb,
        "gray": norm,
        "hsv": cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
        "edge": cv2.Canny(norm, 48, 145),
        "ink": adaptive_ink(norm),
        "letter_ink": letter_ink_mask(rgb),
        "red_mask": red_ink_mask(rgb),
        "black_mask": black_ink_mask(rgb),
    }


def normalize_gray(gray: object) -> object:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_ink(gray: object) -> object:
    block_size = 21 if min(gray.shape[:2]) >= 40 else 15
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7,
    )


def opencv_empty_likelihood(prepared: dict) -> tuple[float, float, float]:
    gray = center_crop_array(prepared["gray"], 0.12)
    edge = cv2.Canny(gray, 48, 145)
    ink = adaptive_ink(gray)
    ink_density = float(np.count_nonzero(ink)) / max(1, ink.size)
    edge_density = float(np.count_nonzero(edge)) / max(1, edge.size)
    ink_empty = 1.0 - min(1.0, ink_density / OPENCV_EMPTY_INK_DENSITY_THRESHOLD)
    edge_empty = 1.0 - min(1.0, edge_density / OPENCV_EMPTY_EDGE_DENSITY_THRESHOLD)
    empty_score = max(0.0, min(1.0, ink_empty * 0.58 + edge_empty * 0.42))
    return round(empty_score, 4), round(ink_density, 4), round(edge_density, 4)


def center_crop_array(array: object, inset_ratio: float) -> object:
    height, width = array.shape[:2]
    inset_x = int(width * inset_ratio)
    inset_y = int(height * inset_ratio)
    return array[inset_y : max(inset_y + 1, height - inset_y), inset_x : max(inset_x + 1, width - inset_x)]


def best_opencv_candidate(
    prepared: dict,
    template: OpenCvTemplate,
) -> Candidate:
    best: Candidate | None = None
    for height_ratio in OPENCV_TEMPLATE_HEIGHT_RATIOS:
        variant = opencv_template_variant(template, prepared["gray"].shape[0], height_ratio)
        if variant is None:
            continue
        candidate = score_opencv_variant(prepared, template, variant, height_ratio)
        if best is None or candidate.score > best.score:
            best = candidate
    return best if best is not None else Candidate(
        color=template.color,
        piece=template.piece,
        score=0.0,
        source=template.source,
    )


def opencv_template_variant(
    template: OpenCvTemplate,
    cell_height: int,
    height_ratio: float,
) -> dict | None:
    cache_key = (
        template.color,
        template.piece,
        template.variant_key,
        template.source,
        template.sprite_row,
        template.sprite_col,
        cell_height,
        height_ratio,
    )
    if cache_key in OPENCV_VARIANT_CACHE:
        return OPENCV_VARIANT_CACHE[cache_key]

    target_height = max(8, round(cell_height * height_ratio))
    original_height, original_width = template.alpha.shape[:2]
    target_width = max(8, round(original_width * (target_height / original_height)))
    if target_height < 8 or target_width < 8:
        OPENCV_VARIANT_CACHE[cache_key] = None
        return None
    rgb = cv2.resize(template.rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(template.alpha, (target_width, target_height), interpolation=cv2.INTER_AREA)
    alpha_mask = np.where(alpha > 32, 255, 0).astype("uint8")
    gray = normalize_gray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    template_edge = cv2.Canny(gray, 48, 145)
    template_ink = adaptive_ink(gray)
    template_letter_ink = letter_ink_mask(rgb)
    red_mask = red_ink_mask(rgb)
    black_mask = black_ink_mask(rgb)
    red_share = ink_red_share_from_masks(red_mask, black_mask)
    if template.source.startswith("calibration:"):
        template_edge = cv2.Canny(alpha_mask, 48, 145)
        template_ink = alpha_mask
        template_letter_ink = alpha_mask
        red_mask = cv2.bitwise_and(red_mask, alpha_mask)
        black_mask = cv2.bitwise_and(black_mask, alpha_mask)
        red_share = ink_red_share_from_masks(red_mask, black_mask)
    variant = {
        "rgb": rgb,
        "hsv": cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
        "gray": gray,
        "edge": template_edge,
        "ink": template_ink,
        "letter_ink": template_letter_ink,
        "red_mask": red_mask,
        "black_mask": black_mask,
        "mask": alpha_mask,
        "red_share": red_share,
    }
    OPENCV_VARIANT_CACHE[cache_key] = variant
    return variant


def score_opencv_variant(
    prepared: dict,
    template: OpenCvTemplate,
    variant: dict,
    scale: float,
) -> Candidate:
    source_height, source_width = prepared["gray"].shape[:2]
    template_height, template_width = variant["gray"].shape[:2]
    if template_height > source_height or template_width > source_width:
        return Candidate(
            color=template.color,
            piece=template.piece,
            score=0.0,
            scale=round(scale, 3),
            source=template.source,
        )

    edge_map = template_match_map(prepared["edge"], variant["edge"], variant["mask"])
    letter_map = template_match_map(prepared["letter_ink"], variant["letter_ink"])
    _, _, _, max_location = cv2.minMaxLoc(letter_map)
    x, y = max_location
    edge_score = float(edge_map[y, x])
    ink_score = binary_overlap_score(prepared["letter_ink"], variant["letter_ink"], x, y)
    color_score = ink_color_compatibility(prepared, variant, x, y, template_width, template_height)
    source_red_share = ink_red_share_for_region(prepared, x, y, template_width, template_height)
    max_value = (
        edge_score * OPENCV_WEIGHTS["edge"]
        + ink_score * OPENCV_WEIGHTS["ink"]
        + color_score * OPENCV_WEIGHTS["color"]
    )
    max_value *= red_mismatch_multiplier(source_red_share, variant["red_share"])
    return Candidate(
        color=template.color,
        piece=template.piece,
        score=round(float(max_value), 4),
        edge_score=round(edge_score, 4),
        ink_score=round(ink_score, 4),
        color_score=round(color_score, 4),
        bbox=[int(x), int(y), int(x + template_width), int(y + template_height)],
        scale=round(scale, 3),
        source=template.source,
    )


def ink_red_share_for_region(
    prepared: dict,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    red_crop = prepared["red_mask"][y : y + height, x : x + width]
    black_crop = prepared["black_mask"][y : y + height, x : x + width]
    red_count = int(np.count_nonzero(red_crop))
    black_count = int(np.count_nonzero(black_crop))
    return red_count / max(1, red_count + black_count)


def red_mismatch_multiplier(source_red_share: float, template_red_share: float) -> float:
    if source_red_share >= 0.38 and template_red_share <= 0.16:
        return 0.70
    if source_red_share <= 0.14 and template_red_share >= 0.46:
        return 0.60
    return 1.0


def letter_ink_mask(rgb: object) -> object:
    mask = np.where(black_ink_mask(rgb) > 0, 255, 0) | np.where(red_ink_mask(rgb) > 0, 255, 0)
    return cv2.dilate(mask.astype("uint8"), np.ones((2, 2), dtype="uint8"), iterations=1)


def black_ink_mask(rgb: object) -> object:
    channels = rgb.astype("int16")
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    black = (red < 118) & (green < 118) & (blue < 118) & ((red + green + blue) < 315)
    return np.where(black, 255, 0).astype("uint8")


def red_ink_mask(rgb: object) -> object:
    channels = rgb.astype("int16")
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    red_ink = (red > 120) & (green < 108) & (blue < 108) & ((red - green) > 48) & ((red - blue) > 48)
    return np.where(red_ink, 255, 0).astype("uint8")


def ink_red_share(rgb: object) -> float:
    black_count = int(np.count_nonzero(black_ink_mask(rgb)))
    red_count = int(np.count_nonzero(red_ink_mask(rgb)))
    return red_count / max(1, black_count + red_count)


def ink_red_share_from_masks(red_mask: object, black_mask: object) -> float:
    red_count = int(np.count_nonzero(red_mask))
    black_count = int(np.count_nonzero(black_mask))
    return red_count / max(1, black_count + red_count)


def ink_color_compatibility(
    prepared: dict,
    variant: dict,
    x: int,
    y: int,
    width: int,
    height: int,
) -> float:
    red_crop = prepared["red_mask"][y : y + height, x : x + width]
    black_crop = prepared["black_mask"][y : y + height, x : x + width]
    red_count = int(np.count_nonzero(red_crop))
    black_count = int(np.count_nonzero(black_crop))
    source_share = red_count / max(1, red_count + black_count)
    return round(max(0.0, 1.0 - abs(source_share - variant["red_share"]) * 2.8), 4)


def binary_overlap_score(
    source: object,
    template: object,
    x: int,
    y: int,
) -> float:
    height, width = template.shape[:2]
    crop = source[y : y + height, x : x + width]
    if crop.shape[:2] != template.shape[:2]:
        return 0.0
    source_mask = crop > 0
    template_mask = template > 0
    template_count = int(np.count_nonzero(template_mask))
    source_count = int(np.count_nonzero(source_mask))
    if template_count < 8 or source_count < 8:
        return 0.0
    intersection = int(np.count_nonzero(source_mask & template_mask))
    dice = (2.0 * intersection) / max(1, template_count + source_count)
    recall = intersection / max(1, template_count)
    return round(dice * 0.78 + recall * 0.22, 4)


def template_match_map(source: object, template: object, mask: object | None = None) -> object:
    if source.shape[0] < template.shape[0] or source.shape[1] < template.shape[1]:
        return np.zeros((1, 1), dtype="float32")
    if np.count_nonzero(template) < 8:
        return np.zeros((source.shape[0] - template.shape[0] + 1, source.shape[1] - template.shape[1] + 1), dtype="float32")
    try:
        if mask is not None:
            result = cv2.matchTemplate(source, template, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            result = cv2.matchTemplate(source, template, cv2.TM_CCORR_NORMED)
    except cv2.error:
        result = cv2.matchTemplate(source, template, cv2.TM_CCORR_NORMED)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def trim_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox is not None else image


def write_recognition_outputs(
    report: dict,
    out_dir: Path,
    cells_dir: Path,
    write_debug_images: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "piece_report.json", report)
    write_json(out_dir / "recognized_board.json", recognized_board(report))
    if not write_debug_images:
        return
    draw_recognized_preview(cells_dir, report).save(out_dir / "recognized_board_preview.png")
    draw_piece_match_overlay(cells_dir, report).save(out_dir / "piece_match_overlay.png")
    empty_cells_dir = Path(report.get("empty_cells_dir", str(cells_dir)))
    if empty_cells_dir.exists():
        draw_empty_mask_overlay(empty_cells_dir, report).save(out_dir / "empty_mask_overlay.png")
        draw_piece_like_overlay(empty_cells_dir, report).save(out_dir / "piece_like_overlay.png")
    draw_candidate_grid(cells_dir, report).save(out_dir / "candidate_grid.png")


def recognized_board(report: dict) -> dict:
    cells = report["cells"]
    rows = []
    for row in range(1, 10):
        row_cells = [cell for cell in cells if cell["row"] == row]
        row_cells.sort(key=lambda cell: cell["col"])
        rows.append(
            [
                {
                    "square": cell["square"],
                    "state": cell["state"],
                    "color": cell["color"],
                    "piece": cell["piece"],
                    "confidence": cell["confidence"],
                    "ambiguous": cell["ambiguous"],
                }
                for cell in row_cells
            ],
        )
    return {
        "method": report["method"],
        "summary": report["summary"],
        "rows": rows,
    }


def normalized_ink_mask(
    image: Image.Image,
    trim_alpha: bool,
    inset_ratio: float,
) -> tuple[bytes, float, float]:
    rgba = image.convert("RGBA")
    crop = alpha_bbox(rgba) if trim_alpha else inner_rect(rgba.size, inset_ratio)
    cropped = rgba.crop(crop)
    ink_bbox = ink_bounds(cropped)
    if ink_bbox is None:
        return bytes(NORMALIZED_SIZE[0] * NORMALIZED_SIZE[1]), 0.0, 0.0

    padded = padded_box(ink_bbox, cropped.size, ratio=0.22)
    target = cropped.crop(padded).convert("L")
    target = ImageEnhance.Contrast(target).enhance(1.55)
    target = fit_to_canvas(target, NORMALIZED_SIZE)
    mask = bytes(1 if pixel < 135 else 0 for pixel in target.tobytes())
    mask = remove_bottom_outline(mask)

    dark_pixels, total_pixels = count_ink_pixels(cropped.convert("RGB"))
    dark_ratio = dark_pixels / max(1, total_pixels)
    bbox_ratio = (ink_bbox_width(ink_bbox) * ink_bbox_height(ink_bbox)) / max(1, cropped.width * cropped.height)
    return mask, dark_ratio, bbox_ratio


def fit_to_canvas(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    resized_w = max(1, int(image.width * scale))
    resized_h = max(1, int(image.height * scale))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", size, 255)
    canvas.paste(resized, ((target_w - resized_w) // 2, (target_h - resized_h) // 2))
    return canvas


def count_ink_pixels(image: Image.Image) -> tuple[int, int]:
    data = image.tobytes()
    hits = 0
    total = 0
    for index in range(0, len(data), 3):
        total += 1
        if is_ink_pixel((data[index], data[index + 1], data[index + 2])):
            hits += 1
    return hits, total


def ink_color_features(
    image: Image.Image,
    trim_alpha: bool,
    inset_ratio: float,
) -> InkColorFeatures:
    rgba = image.convert("RGBA")
    crop = alpha_bbox(rgba) if trim_alpha else inner_rect(rgba.size, inset_ratio)
    data = rgba.crop(crop).tobytes()
    black_hits = 0
    red_hits = 0
    total = 0
    for index in range(0, len(data), 4):
        alpha = data[index + 3]
        if alpha < 16:
            continue
        total += 1
        rgb = (data[index], data[index + 1], data[index + 2])
        if is_black_ink(rgb):
            black_hits += 1
        elif is_red_ink(rgb):
            red_hits += 1

    ink_hits = black_hits + red_hits
    return InkColorFeatures(
        black_ratio=black_hits / max(1, total),
        red_ratio=red_hits / max(1, total),
        red_share=red_hits / max(1, ink_hits),
    )


def match_score(
    cell: MaskFeatures,
    template: TemplatePiece,
) -> float:
    return round(
        max(score_features(cell, variant) for variant in template.variants),
        4,
    )


def candidate_score(
    cell: MaskFeatures,
    cell_color: InkColorFeatures,
    template: TemplatePiece,
) -> float:
    shape_score = match_score(cell, template)
    color_score = color_compatibility(cell_color, template.ink_color)
    return round(shape_score * (0.62 + 0.38 * color_score), 4)


def color_compatibility(
    cell: InkColorFeatures,
    template: InkColorFeatures,
) -> float:
    red_distance = abs(cell.red_share - template.red_share)
    return round(max(0.0, 1.0 - red_distance * 2.2), 4)


def score_features(
    cell: MaskFeatures,
    template: MaskFeatures,
) -> float:
    intersection = (cell.bits & template.bits).bit_count()
    union = (cell.bits | template.bits).bit_count()
    iou = intersection / union if union else 0.0
    dice = (2.0 * intersection) / max(1, cell.bits.bit_count() + template.bits.bit_count())
    area_score = 1.0 - min(
        1.0,
        abs(cell.dark_ratio - template.dark_ratio) / max(cell.dark_ratio, template.dark_ratio, 0.01),
    )
    projection_score = (
        projection_similarity(cell.projection_x, template.projection_x)
        + projection_similarity(cell.projection_y, template.projection_y)
    ) / 2.0
    return dice * 0.56 + iou * 0.24 + projection_score * 0.14 + area_score * 0.06


def mask_features(mask: bytes) -> MaskFeatures:
    bits = 0
    projection_x = [0] * NORMALIZED_SIZE[0]
    projection_y = [0] * NORMALIZED_SIZE[1]
    for y in range(NORMALIZED_SIZE[1]):
        for x in range(NORMALIZED_SIZE[0]):
            index = y * NORMALIZED_SIZE[0] + x
            value = mask[index]
            if value:
                bits |= 1 << index
                projection_x[x] += 1
                projection_y[y] += 1
    return MaskFeatures(
        mask=mask,
        bits=bits,
        dark_ratio=sum(mask) / max(1, len(mask)),
        projection_x=tuple(projection_x),
        projection_y=tuple(projection_y),
    )


def build_template_variants(mask: bytes) -> tuple[MaskFeatures, ...]:
    variants: list[MaskFeatures] = []
    seen: set[bytes] = set()
    for scale in TEMPLATE_SCALE_FACTORS:
        for shift_x in TEMPLATE_SHIFT_PIXELS:
            for shift_y in TEMPLATE_SHIFT_PIXELS:
                shifted = transform_mask(mask, scale, shift_x, shift_y)
                if shifted in seen:
                    continue
                seen.add(shifted)
                variants.append(mask_features(shifted))
    return tuple(variants)


def transform_mask(
    mask: bytes,
    scale: float,
    shift_x: int,
    shift_y: int,
) -> bytes:
    box = mask_bounds(mask)
    if box is None:
        return bytes(NORMALIZED_SIZE[0] * NORMALIZED_SIZE[1])

    source = mask_to_image(mask).crop(box)
    source_width, source_height = source.size
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    scaled = source.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)

    left, top, right, bottom = box
    center_x = (left + right) / 2.0 + shift_x
    center_y = (top + bottom) / 2.0 + shift_y
    paste_left = round(center_x - scaled_width / 2.0)
    paste_top = round(center_y - scaled_height / 2.0)

    canvas = Image.new("L", NORMALIZED_SIZE, 0)
    source_left = max(0, -paste_left)
    source_top = max(0, -paste_top)
    target_left = max(0, paste_left)
    target_top = max(0, paste_top)
    paste_width = min(scaled_width - source_left, NORMALIZED_SIZE[0] - target_left)
    paste_height = min(scaled_height - source_top, NORMALIZED_SIZE[1] - target_top)
    if paste_width <= 0 or paste_height <= 0:
        return bytes(NORMALIZED_SIZE[0] * NORMALIZED_SIZE[1])

    region = scaled.crop(
        (
            source_left,
            source_top,
            source_left + paste_width,
            source_top + paste_height,
        ),
    )
    canvas.paste(region, (target_left, target_top))
    return bytes(1 if pixel > 127 else 0 for pixel in canvas.tobytes())


def mask_to_image(mask: bytes) -> Image.Image:
    image = Image.new("L", NORMALIZED_SIZE, 0)
    image.putdata([255 if value else 0 for value in mask])
    return image


def mask_bounds(mask: bytes) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y in range(NORMALIZED_SIZE[1]):
        for x in range(NORMALIZED_SIZE[0]):
            if mask[y * NORMALIZED_SIZE[0] + x]:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def remove_bottom_outline(mask: bytes) -> bytes:
    width, height = NORMALIZED_SIZE
    result = bytearray(mask)
    visited = bytearray(len(mask))
    for start, value in enumerate(mask):
        if not value or visited[start]:
            continue
        component = collect_component(mask, start, visited)
        xs = [index % width for index in component]
        ys = [index // width for index in component]
        left, right = min(xs), max(xs) + 1
        top, bottom = min(ys), max(ys) + 1
        component_width = right - left
        component_height = bottom - top
        looks_like_piece_outline = (
            bottom >= int(height * 0.84)
            and component_width >= int(width * 0.25)
            and component_height <= int(height * 0.20)
        )
        if looks_like_piece_outline:
            for index in component:
                result[index] = 0
    return bytes(result)


def collect_component(
    mask: bytes,
    start: int,
    visited: bytearray,
) -> list[int]:
    width, height = NORMALIZED_SIZE
    stack = [start]
    visited[start] = 1
    component: list[int] = []
    while stack:
        index = stack.pop()
        component.append(index)
        x = index % width
        y = index // width
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue
            next_index = next_y * width + next_x
            if mask[next_index] and not visited[next_index]:
                visited[next_index] = 1
                stack.append(next_index)
    return component


def legacy_match_score(
    cell_mask: bytes,
    template: TemplatePiece,
) -> float:
    template_mask = template.mask
    intersection = 0
    union = 0
    cell_projection_x = [0] * NORMALIZED_SIZE[0]
    template_projection_x = [0] * NORMALIZED_SIZE[0]
    cell_projection_y = [0] * NORMALIZED_SIZE[1]
    template_projection_y = [0] * NORMALIZED_SIZE[1]

    for y in range(NORMALIZED_SIZE[1]):
        for x in range(NORMALIZED_SIZE[0]):
            index = y * NORMALIZED_SIZE[0] + x
            cell_value = cell_mask[index]
            template_value = template_mask[index]
            if cell_value or template_value:
                union += 1
            if cell_value and template_value:
                intersection += 1
            cell_projection_x[x] += cell_value
            template_projection_x[x] += template_value
            cell_projection_y[y] += cell_value
            template_projection_y[y] += template_value

    iou = intersection / union if union else 0.0
    cell_ratio = sum(cell_mask) / max(1, len(cell_mask))
    area_score = 1.0 - min(1.0, abs(cell_ratio - template.dark_ratio) / max(cell_ratio, template.dark_ratio, 0.01))
    projection_score = (
        projection_similarity(cell_projection_x, template_projection_x)
        + projection_similarity(cell_projection_y, template_projection_y)
    ) / 2.0
    return round((iou * 0.58 + area_score * 0.17 + projection_score * 0.25), 4)


def empty_likelihood(
    dark_ratio: float,
    bbox_ratio: float,
) -> float:
    dark_empty = 1.0 - min(1.0, dark_ratio / EMPTY_DARK_RATIO_THRESHOLD)
    bbox_empty = 1.0 - min(1.0, bbox_ratio / EMPTY_BBOX_RATIO_THRESHOLD)
    return round(max(0.0, min(1.0, dark_empty * 0.62 + bbox_empty * 0.38)), 4)


def projection_similarity(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    max_value = max(max(left, default=0), max(right, default=0), 1)
    distance = sum(abs(a - b) for a, b in zip(left, right))
    return max(0.0, 1.0 - distance / (len(left) * max_value))


def draw_recognized_preview(
    cells_dir: Path,
    report: dict,
) -> Image.Image:
    cell_paths = {parse_cell_position(path): path for path in iter_cell_images(cells_dir)}
    first_image = Image.open(next(iter(cell_paths.values()))).convert("RGB")
    slot_w, slot_h = first_image.size
    preview = Image.new("RGB", (slot_w * 9, slot_h * 9), (245, 230, 180))
    draw = ImageDraw.Draw(preview)
    cells_by_position = {(cell["row"], cell["col"]): cell for cell in report["cells"]}

    for row in range(1, 10):
        for col in range(1, 10):
            path = cell_paths.get((row, col))
            if path is None:
                continue
            image = Image.open(path).convert("RGB").resize((slot_w, slot_h), Image.Resampling.BILINEAR)
            x = (col - 1) * slot_w
            y = (row - 1) * slot_h
            preview.paste(image, (x, y))
            cell = cells_by_position.get((row, col))
            label = label_for_cell(cell)
            fill = label_color(cell)
            draw.rectangle((x, y, x + slot_w - 1, y + 16), fill=(255, 255, 255))
            draw.text((x + 2, y + 2), label, fill=fill)
            draw.rectangle((x, y, x + slot_w - 1, y + slot_h - 1), outline=(80, 50, 20))
    return preview


def draw_piece_match_overlay(
    cells_dir: Path,
    report: dict,
) -> Image.Image:
    cell_paths = {parse_cell_position(path): path for path in iter_cell_images(cells_dir)}
    first_image = Image.open(next(iter(cell_paths.values()))).convert("RGB")
    slot_w, slot_h = first_image.size
    overlay = Image.new("RGB", (slot_w * 9, slot_h * 9), (245, 230, 180))
    draw = ImageDraw.Draw(overlay)
    cells_by_position = {(cell["row"], cell["col"]): cell for cell in report["cells"]}

    for row in range(1, 10):
        for col in range(1, 10):
            path = cell_paths.get((row, col))
            if path is None:
                continue
            raw_image = Image.open(path).convert("RGB")
            image = raw_image.resize((slot_w, slot_h), Image.Resampling.BILINEAR)
            x = (col - 1) * slot_w
            y = (row - 1) * slot_h
            overlay.paste(image, (x, y))
            cell = cells_by_position.get((row, col))
            candidate = first_candidate(cell)
            if candidate and candidate.get("bbox"):
                bbox = scale_bbox(candidate["bbox"], raw_image.size, (slot_w, slot_h))
                fill = (30, 120, 255) if cell and cell.get("state") == "piece" else (255, 140, 0)
                draw.rectangle(
                    (x + bbox[0], y + bbox[1], x + bbox[2], y + bbox[3]),
                    outline=fill,
                    width=3,
                )
            label = label_for_cell(cell)
            draw.rectangle((x, y, x + slot_w - 1, y + 16), fill=(255, 255, 255))
            draw.text((x + 2, y + 2), label, fill=label_color(cell))
            draw.rectangle((x, y, x + slot_w - 1, y + slot_h - 1), outline=(80, 50, 20))
    return overlay


def draw_empty_mask_overlay(
    cells_dir: Path,
    report: dict,
) -> Image.Image:
    cell_paths = {parse_cell_position(path): path for path in iter_cell_images(cells_dir)}
    first_image = Image.open(next(iter(cell_paths.values()))).convert("RGB")
    slot_w, slot_h = first_image.size
    overlay = Image.new("RGB", (slot_w * 9, slot_h * 9), (245, 230, 180))
    draw = ImageDraw.Draw(overlay)
    cells_by_position = {(cell["row"], cell["col"]): cell for cell in report["cells"]}

    for row in range(1, 10):
        for col in range(1, 10):
            path = cell_paths.get((row, col))
            if path is None:
                continue
            raw = Image.open(path).convert("RGB")
            mask = clean_hog_letter_mask(np.array(raw)) if cv2 is not None and np is not None else None
            image = raw.resize((slot_w, slot_h), Image.Resampling.BILINEAR)
            if mask is not None:
                mask_image = Image.fromarray(mask).convert("L").resize((slot_w, slot_h), Image.Resampling.NEAREST)
                tint = Image.new("RGB", (slot_w, slot_h), (255, 50, 50))
                image = Image.blend(image, tint, 0.0)
                image.paste(tint, (0, 0), mask_image.point(lambda value: 115 if value else 0))
            x = (col - 1) * slot_w
            y = (row - 1) * slot_h
            overlay.paste(image, (x, y))
            cell = cells_by_position.get((row, col))
            debug = cell.get("debug") if cell else None
            edge = debug.get("edge_density") if isinstance(debug, dict) else None
            label = f"e {cell.get('empty_score', 0.0):.2f}" if cell else "missing"
            if edge is not None:
                label += f" / ed {edge:.2f}"
            draw.rectangle((x, y, x + slot_w - 1, y + 16), fill=(255, 255, 255))
            draw.text((x + 2, y + 2), label, fill=label_color(cell))
            draw.rectangle((x, y, x + slot_w - 1, y + slot_h - 1), outline=(80, 50, 20))
    return overlay


def draw_piece_like_overlay(
    cells_dir: Path,
    report: dict,
) -> Image.Image:
    cell_paths = {parse_cell_position(path): path for path in iter_cell_images(cells_dir)}
    first_image = Image.open(next(iter(cell_paths.values()))).convert("RGB")
    slot_w, slot_h = first_image.size
    overlay = Image.new("RGB", (slot_w * 9, slot_h * 9), (245, 230, 180))
    draw = ImageDraw.Draw(overlay)
    cells_by_position = {(cell["row"], cell["col"]): cell for cell in report["cells"]}

    for row in range(1, 10):
        for col in range(1, 10):
            path = cell_paths.get((row, col))
            if path is None:
                continue
            image = Image.open(path).convert("RGB").resize((slot_w, slot_h), Image.Resampling.BILINEAR)
            cell = cells_by_position.get((row, col))
            if cell is None:
                fill = (120, 120, 120)
                label = "missing"
            elif cell["state"] == "empty":
                fill = (70, 150, 75)
                label = "empty"
            elif cell["state"] == "piece":
                fill = (45, 105, 210)
                label = "piece"
            else:
                fill = (230, 145, 40)
                label = "unknown"
            tint = Image.new("RGB", (slot_w, slot_h), fill)
            image = Image.blend(image, tint, 0.22)
            x = (col - 1) * slot_w
            y = (row - 1) * slot_h
            overlay.paste(image, (x, y))
            draw.rectangle((x, y, x + slot_w - 1, y + 16), fill=(255, 255, 255))
            draw.text((x + 2, y + 2), label, fill=fill)
            draw.rectangle((x, y, x + slot_w - 1, y + slot_h - 1), outline=(80, 50, 20))
    return overlay


def draw_candidate_grid(
    cells_dir: Path,
    report: dict,
) -> Image.Image:
    cell_paths = {parse_cell_position(path): path for path in iter_cell_images(cells_dir)}
    first_image = Image.open(next(iter(cell_paths.values()))).convert("RGB")
    slot_w, slot_h = first_image.size
    grid = Image.new("RGB", (slot_w * 9, slot_h * 9), (245, 230, 180))
    draw = ImageDraw.Draw(grid)
    cells_by_position = {(cell["row"], cell["col"]): cell for cell in report["cells"]}

    for row in range(1, 10):
        for col in range(1, 10):
            path = cell_paths.get((row, col))
            if path is None:
                continue
            image = Image.open(path).convert("RGB").resize((slot_w, slot_h), Image.Resampling.BILINEAR)
            x = (col - 1) * slot_w
            y = (row - 1) * slot_h
            grid.paste(image, (x, y))
            cell = cells_by_position.get((row, col))
            lines = candidate_lines(cell)
            draw.rectangle((x, y, x + slot_w - 1, y + min(slot_h - 1, 42)), fill=(255, 255, 255))
            for index, line in enumerate(lines[:3]):
                draw.text((x + 2, y + 2 + index * 13), line, fill=label_color(cell))
            draw.rectangle((x, y, x + slot_w - 1, y + slot_h - 1), outline=(80, 50, 20))
    return grid


def candidate_lines(cell: dict | None) -> list[str]:
    if cell is None:
        return ["missing"]
    if cell.get("state") == "empty":
        return [f"empty {cell.get('confidence', 0.0):.2f}"]
    candidates = cell.get("candidates") or []
    if not candidates:
        return ["unknown"]
    lines = []
    for candidate in candidates[:3]:
        color = "B" if candidate["color"] == "black" else "W"
        lines.append(f"{color}:{candidate['piece']} {candidate['score']:.2f} {source_short(candidate.get('source'))}")
    return lines


def source_short(source: str | None) -> str:
    if source is None:
        return "S"
    if source.startswith("calibration:"):
        return "C"
    if source == "hog_svm":
        return "H"
    if source.startswith("label:"):
        return "L"
    if source.startswith("synthetic"):
        return "R"
    return "S"


def first_candidate(cell: dict | None) -> dict | None:
    if cell is None:
        return None
    candidates = cell.get("candidates") or []
    return candidates[0] if candidates else None


def scale_bbox(
    bbox: list[int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    source_w, source_h = source_size
    target_w, target_h = target_size
    scale_x = target_w / max(1, source_w)
    scale_y = target_h / max(1, source_h)
    return [
        round(bbox[0] * scale_x),
        round(bbox[1] * scale_y),
        round(bbox[2] * scale_x),
        round(bbox[3] * scale_y),
    ]


def label_for_cell(cell: dict | None) -> str:
    if cell is None:
        return "missing"
    if cell["state"] == "empty":
        return f"empty {cell['confidence']:.2f}"
    if cell["state"] == "piece":
        color = "B" if cell["color"] == "black" else "W"
        return f"{color}:{cell['piece']} {cell['confidence']:.2f}"
    candidate = cell["candidates"][0] if cell["candidates"] else None
    if candidate:
        color = "B" if candidate["color"] == "black" else "W"
        return f"? {color}:{candidate['piece']} {candidate['score']:.2f}"
    return "unknown"


def label_color(cell: dict | None) -> tuple[int, int, int]:
    if cell is None:
        return (120, 0, 0)
    if cell["state"] == "empty":
        return (50, 100, 50)
    if cell["state"] == "piece":
        return (0, 60, 160)
    return (170, 80, 0)


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return bbox if bbox is not None else (0, 0, image.width, image.height)


def inner_rect(
    size: tuple[int, int],
    inset_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = size
    inset_x = int(width * inset_ratio)
    inset_y = int(height * inset_ratio)
    return (inset_x, inset_y, max(inset_x + 1, width - inset_x), max(inset_y + 1, height - inset_y))


def ink_bounds(image: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    pixels = rgb.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(rgb.height):
        for x in range(rgb.width):
            if is_ink_pixel(pixels[x, y]):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def padded_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    pad_x = max(2, int(width * ratio))
    pad_y = max(2, int(height * ratio))
    max_width, max_height = size
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(max_width, right + pad_x),
        min(max_height, bottom + pad_y),
    )


def is_ink_pixel(rgb: tuple[int, int, int]) -> bool:
    return is_black_ink(rgb) or is_red_ink(rgb)


def is_black_ink(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return red < 98 and green < 98 and blue < 98


def is_red_ink(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return red > 120 and green < 115 and blue < 115 and red - green > 35 and red - blue > 35


def ink_bbox_width(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0]


def ink_bbox_height(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1]


def parse_cell_position(path: Path) -> tuple[int, int]:
    match = re.search(r"r(\d+)_c(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cell filename must include r##_c##: {path.name}")
    return int(match.group(1)), int(match.group(2))


def square_name(
    row: int,
    col: int,
) -> str:
    file_number = 10 - col
    return f"{file_number}{RANK_NAMES[row - 1]}"


def iter_cell_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(path.glob("r*_c*.png"), key=parse_cell_position)
    if not files:
        raise ValueError(f"no cell images found: {path}")
    return files


def resolve_cells_dir(path: Path) -> Path:
    if path.is_dir():
        recognition_cells = path / "recognition_cells"
        if recognition_cells.exists():
            return recognition_cells
        cells = path / "cells"
        if cells.exists():
            return cells
    return path


def resolve_empty_cells_dir(
    original_path: Path,
    resolved_cells_dir: Path,
    explicit_empty_cells_dir: Path | None,
) -> Path:
    if explicit_empty_cells_dir is not None:
        return explicit_empty_cells_dir
    sibling_cells = resolved_cells_dir.parent / "cells"
    if resolved_cells_dir.name == "recognition_cells" and sibling_cells.exists():
        return sibling_cells
    if original_path.is_dir():
        cells = original_path / "cells"
        if cells.exists():
            return cells
    return resolved_cells_dir


def summary(cells: Iterable[CellRecognition]) -> dict:
    cell_list = list(cells)
    return {
        "total": len(cell_list),
        "piece": sum(1 for cell in cell_list if cell.state == "piece"),
        "empty": sum(1 for cell in cell_list if cell.state == "empty"),
        "unknown": sum(1 for cell in cell_list if cell.state == "unknown"),
    }


def cell_to_dict(cell: CellRecognition) -> dict:
    data = asdict(cell)
    data["candidates"] = [asdict(candidate) for candidate in cell.candidates]
    return data


def write_json(
    path: Path,
    data: dict,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def default_template_path() -> Path:
    candidates = [
        Path("assets/legacy_drawables/shogi_pieces.png"),
        Path(__file__).resolve().parents[1] / "assets" / "legacy_drawables" / "shogi_pieces.png",
        Path(__file__).resolve().parents[0] / "assets" / "legacy_drawables" / "shogi_pieces.png",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_calibration_dir() -> Path:
    return Path("data/samples/screenshots_by_app_piece_style")


def default_board_labels_dir() -> Path:
    return Path("data/samples/labels/boards_by_app_piece_style")


def default_screenshots_dir() -> Path:
    return Path("tools/samples/screenshots")


def default_cache_dir() -> Path:
    return Path("tools/out/cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize shogi pieces from 81 grid cell images.")
    parser.add_argument("cells_dir", type=Path, help="Directory containing r##_c##.png cell images.")
    parser.add_argument(
        "--template",
        type=Path,
        default=default_template_path(),
        help="Path to shogi_pieces.png. Defaults to the Android bundled sprite sheet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to the parent of cells_dir.",
    )
    parser.add_argument(
        "--method",
        choices=("hog_svm", "opencv", "legacy"),
        default="hog_svm",
        help="Recognition method. Defaults to hog_svm.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=default_calibration_dir(),
        help="Initial-position screenshots used as extra OpenCV templates. Defaults to tools/samples/screenshots/初期配置.",
    )
    parser.add_argument(
        "--calibration-source-hint",
        default=None,
        help="Optional screenshot/app name used to prefer matching calibration images.",
    )
    parser.add_argument(
        "--board-labels-dir",
        type=Path,
        default=default_board_labels_dir(),
        help="Labeled board screenshots used as extra OpenCV templates. Defaults to tools/samples/labels/boards.",
    )
    parser.add_argument(
        "--exclude-self-calibration-source",
        action="store_true",
        help="Exclude calibration templates whose source matches --calibration-source-hint. Use this for holdout evaluation.",
    )
    parser.add_argument(
        "--no-label-corrections",
        action="store_true",
        help="Do not apply exact board labels as final corrections even when a matching label JSON exists.",
    )
    parser.add_argument(
        "--fast-recognition",
        action="store_true",
        help="Use only fast glyph/template candidates and skip slower full OpenCV sprite matching.",
    )
    parser.add_argument(
        "--label-oracle-baseline",
        action="store_true",
        help="Diagnostic only: allow exact teacher labels to be used as oracle output/corrections.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write JSON reports only; skip preview, overlay, and candidate PNGs.",
    )
    args = parser.parse_args()

    resolved_cells_dir = resolve_cells_dir(args.cells_dir)
    out_dir = args.out if args.out is not None else resolved_cells_dir.parent
    report = recognize_cells(
        args.cells_dir,
        args.template,
        method=args.method,
        calibration_dir=args.calibration_dir,
        calibration_source_hint=args.calibration_source_hint,
        board_labels_dir=args.board_labels_dir,
        exclude_self_calibration_source=args.exclude_self_calibration_source,
        apply_label_corrections=not args.no_label_corrections,
        fast_recognition=args.fast_recognition,
        label_oracle_baseline=args.label_oracle_baseline,
    )
    write_recognition_outputs(
        report,
        out_dir,
        Path(report["cells_dir"]),
        write_debug_images=not args.no_debug_images,
    )
    print(f"OK: {resolved_cells_dir} -> {out_dir}")


if __name__ == "__main__":
    main()
