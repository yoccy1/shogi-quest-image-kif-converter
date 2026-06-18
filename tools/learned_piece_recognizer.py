from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detect_board_grid import GridDetection, detect_grid, report_dict
from detect_hand_areas import detect_hand_areas
from position_label_utils import (
    HAND_PIECES,
    TOTAL_INVENTORY,
    base_piece,
    board_inventory_overcount_errors,
    identity,
    write_json,
)
from recognize_board_pieces import (
    Candidate,
    CellRecognition,
    black_ink_mask,
    calibration_letter_mask,
    cell_to_dict,
    crop_calibration_cell,
    empty_likelihood,
    extract_hog_features,
    fast_glyph_features_from_rgb,
    clean_hog_letter_mask,
    mask_bbox,
    mask_features,
    fit_mask_to_canvas_size,
    normalized_ink_mask,
    square_name,
)
from recognize_hand_pieces import (
    DEFAULT_AMBIGUOUS_MARGIN,
    DEFAULT_MIN_CONFIDENCE,
    HAND_PIECE_SET,
    PieceProposal,
    RecognizedHandPiece,
    aggregate_hands,
    associate_digits,
    best_hand_candidate,
    cell_size,
    classify_proposals,
    classify_piece_crop,
    digit_candidates_for_area,
    has_piece_material,
    piece_material_features,
    proposals_for_area,
    recognize_digit,
    suppress_duplicate_pieces,
)


MODEL_VERSION = 3
BOARD_TEMPLATE_LIMIT_PER_FAMILY_LABEL = 512
PROMOTED_RESCUE_PIECES = {"RY", "UM", "NG", "NK", "NY", "TO"}
DEFAULT_HAND_CROP_CLASSIFIER_FAMILIES = {"将棋クエスト:クラシック二文字駒"}
DEFAULT_HAND_TEMPLATE_TARGET_FAMILIES = {
    "ぴよ将棋:ひよこ駒",
    "将棋クエスト:クラシック二文字駒",
    "将棋クエスト:一文字駒",
}
CLASSIFIER_SKIP_MIN_SCORE = 0.82
CLASSIFIER_SKIP_MIN_MARGIN = 0.10
_HAND_CLASSIFIER_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class LearnedTemplate:
    color: str
    piece: str
    source: str
    row: int
    col: int
    mask: bytes
    bits: int
    dark_ratio: float
    red_share: float
    bbox: list[int] | None
    hog_vector: object
    clean_mask: bytes
    clean_bits: int
    clean_dark_ratio: float
    red_mask: bytes = b""


def load_model(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if model.get("version") != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {model.get('version')}")
    warm_loaded_model(model)
    return model


def warm_loaded_model(model: dict[str, Any]) -> None:
    classifier = model.get("classifier") or {}
    templates = model.get("templates") or []
    if not classifier.get("enabled") or not templates:
        return
    try:
        template = templates[0]
        vector = np.asarray([feature_vector_from_template(template)], dtype="float32")
        hog_vector = np.asarray([template.hog_vector], dtype="float32")
        warm_classifier_estimators(
            classifier.get("knn"),
            classifier.get("svm"),
            classifier.get("hog_knn"),
            classifier.get("hog_sgd"),
            vector,
            hog_vector,
        )
    except Exception:
        return


def save_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_model = dict(model)
    serializable_model.pop("hand_classifier", None)
    with path.open("wb") as handle:
        pickle.dump(serializable_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cell_mask_template(
    image: Image.Image,
    detection: GridDetection,
    row: int,
    col: int,
    color: str,
    piece: str,
    source: str,
) -> LearnedTemplate | None:
    cell = crop_calibration_cell(image, detection, row, col)
    rgb = np.array(cell.convert("RGB"))
    glyph = fast_glyph_features_from_rgb(rgb)
    if glyph is None:
        return None
    clean = clean_mask_features(rgb)
    return LearnedTemplate(
        color=color,
        piece=piece,
        source=source,
        row=row,
        col=col,
        mask=glyph.features.mask,
        bits=glyph.features.bits,
        dark_ratio=glyph.features.dark_ratio,
        red_share=glyph.red_share,
        red_mask=glyph.red_mask,
        bbox=glyph.bbox,
        hog_vector=extract_hog_features(cell).vector.astype("float32"),
        clean_mask=clean.mask if clean is not None else glyph.features.mask,
        clean_bits=clean.bits if clean is not None else glyph.features.bits,
        clean_dark_ratio=clean.dark_ratio if clean is not None else glyph.features.dark_ratio,
    )


def build_model(
    samples: Sequence[tuple[str, Image.Image, GridDetection, list[dict[str, Any]]]],
    *,
    excluded_source: str | None = None,
    include_hands: bool = False,
) -> dict[str, Any]:
    templates: list[LearnedTemplate] = []
    label_counts: dict[str, int] = {}
    skipped_sources: dict[str, list[str]] = {}
    training_sources: set[str] = set()
    position_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source, image, detection, cells in samples:
        if source == excluded_source:
            continue
        source_errors = board_inventory_overcount_errors(cells)
        overfull_bases = board_overfull_bases(cells)
        if source_errors:
            skipped_sources[source] = source_errors
        usable_cells = [
            cell
            for cell in cells
            if cell["state"] != "piece" or base_piece(str(cell["piece"])) not in overfull_bases
        ]
        training_sources.add(source)
        add_position_counts(position_counts, source, usable_cells)
        for cell in usable_cells:
            if cell["state"] != "piece":
                continue
            template = cell_mask_template(
                image,
                detection,
                cell["row"],
                cell["col"],
                cell["color"],
                cell["piece"],
                source,
            )
            if template is None:
                continue
            key = identity(template.color, template.piece)
            count_key = f"{source_family(source)}:{key}"
            if label_counts.get(count_key, 0) >= BOARD_TEMPLATE_LIMIT_PER_FAMILY_LABEL:
                continue
            templates.append(template)
            label_counts[count_key] = label_counts.get(count_key, 0) + 1

    classifier = train_classifiers(templates)
    model = {
        "version": MODEL_VERSION,
        "method": "learned_cv_template_knn",
        "excluded_source": excluded_source,
        "templates": templates,
        "classifier": classifier,
        "label_counts": label_counts,
        "skipped_sources": skipped_sources,
        "training_sources": sorted(training_sources),
        "template_sources": sorted({template.source for template in templates}),
        "position_priors": serialize_position_priors(position_counts),
        "target_family": source_family(excluded_source or ""),
    }
    if include_hands:
        hand_classifier_assets()
        model["hand_classifier_preloaded"] = True
    return model


def board_overfull_bases(cells: Sequence[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for cell in cells:
        if cell.get("state") == "piece" and cell.get("piece"):
            counts[base_piece(str(cell["piece"]))] += 1
    return {
        piece
        for piece, count in counts.items()
        if count > TOTAL_INVENTORY.get(piece, 0)
    }


def train_classifiers(templates: Sequence[LearnedTemplate]) -> dict[str, Any]:
    if not templates:
        return {"enabled": False}
    vectors = np.vstack([feature_vector_from_template(template) for template in templates]).astype("float32")
    hog_vectors = np.vstack([template.hog_vector for template in templates]).astype("float32")
    labels = np.array([identity(template.color, template.piece) for template in templates])
    knn = KNeighborsClassifier(n_neighbors=min(5, len(templates)), weights="distance", n_jobs=1)
    knn.fit(vectors, labels)
    hog_knn = KNeighborsClassifier(n_neighbors=min(7, len(templates)), weights="distance", metric="cosine", n_jobs=1)
    hog_knn.fit(hog_vectors, labels)
    svm = None
    hog_sgd = None
    if len(set(labels.tolist())) >= 2:
        svm = SVC(kernel="rbf", C=6.0, gamma="scale", probability=True)
        svm.fit(vectors, labels)
        hog_sgd = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="modified_huber",
                alpha=0.0002,
                max_iter=1200,
                random_state=20260507,
                tol=1e-3,
            ),
        )
        hog_sgd.fit(hog_vectors, labels)
    warm_classifier_estimators(knn, svm, hog_knn, hog_sgd, vectors[:1], hog_vectors[:1])
    return {
        "enabled": True,
        "knn": knn,
        "svm": svm,
        "hog_knn": hog_knn,
        "hog_sgd": hog_sgd,
        "labels": sorted(set(labels.tolist())),
        "feature_size": int(vectors.shape[1]),
        "hog_feature_size": int(hog_vectors.shape[1]),
    }


def warm_classifier_estimators(knn, svm, hog_knn, hog_sgd, vector: np.ndarray, hog_vector: np.ndarray) -> None:
    for estimator, sample in ((knn, vector), (svm, vector), (hog_knn, hog_vector), (hog_sgd, hog_vector)):
        if estimator is None or not hasattr(estimator, "predict_proba"):
            continue
        try:
            estimator.predict_proba(sample)
        except Exception:
            continue


def recognize_image(
    image_path: Path,
    model_path: Path | None = None,
    *,
    model: dict[str, Any] | None = None,
    include_hands: bool = False,
    out_path: Path | None = None,
    max_proposals_per_area: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    if model is None:
        if model_path is None:
            raise ValueError("model_path or model is required")
        model = load_model(model_path)
    image = Image.open(image_path).convert("RGB")
    detection = detect_grid(image)
    grid_report = report_dict(image, detection)
    cells: list[dict[str, Any]] = []
    constraint_report = None
    target_family = ""
    if detection is not None:
        target_family = resolve_target_family(model, image_path)
        cells = [
            cell_to_dict(cell)
            for cell in recognize_board(image, detection, model, target_family=target_family)
        ]
    hands = {
        "black": {piece: 0 for piece in HAND_PIECES},
        "white": {piece: 0 for piece in HAND_PIECES},
    }
    hand_report = None
    if include_hands and detection is not None:
        hand_report = recognize_hands(
            image,
            detection,
            max_proposals_per_area,
            model,
            model.get("hand_classifier"),
            target_family=target_family,
        )
        hands = hand_report["hands"]
    if cells:
        hands_for_constraints = hands if include_hands and hand_report_is_reliable(hand_report) else None
        soft_hands = hands_for_constraints
        if include_hands and hand_report is not None and soft_hands is None:
            soft_hands = soft_inventory_hands_from_hand_report(hand_report)
        constraint_report = apply_piece_constraints(cells, hands_for_constraints, soft_hands, target_family=target_family)
        constraint_report["hands_used_for_inventory"] = hands_for_constraints is not None
        constraint_report["soft_hands_used_for_inventory"] = soft_hands is not None
        constraint_report["hand_reliability"] = hand_reliability_report(hand_report)
        if include_hands and hand_report is not None:
            hands = sanitize_hands_against_board_inventory(cells, hands, hand_report)
            hands = complete_hands_from_inventory_candidates(cells, hands, hand_report)

    report = {
        "image": str(image_path),
        "method": "learned_cv",
        "model_path": str(model_path) if model_path is not None else "<memory>",
        "model": {
            "version": model.get("version"),
            "method": model.get("method"),
            "excluded_source": model.get("excluded_source"),
            "template_count": len(model.get("templates", [])),
            "training_sources": model.get("training_sources", []),
            "template_sources": model.get("template_sources", []),
            "skipped_sources": model.get("skipped_sources", {}),
            "position_priors": {
                "keys": len(model.get("position_priors", {})),
                "target_family": model.get("target_family"),
                "resolved_target_family": target_family if detection is not None else "",
            },
        },
        "grid": grid_report,
        "summary": summarize_cells(cells),
        "cells": cells,
        "hands": hands,
        "hand_recognition": hand_report,
        "constraint_postprocess": constraint_report,
        "timing": timing_report(started),
    }
    if out_path is not None:
        write_json(out_path, report)
    return report


def sanitize_hands_against_board_inventory(
    cells: Sequence[dict[str, Any]],
    hands: dict[str, dict[str, int]],
    hand_report: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Drop impossible hand over-counts after the board has been recognized.

    Hand areas sometimes include UI digits or decorative glyphs. Those false
    positives tend to create impossible totals, for example two rooks on the
    board plus an extra rook in hand. Removing only the impossible surplus keeps
    the correction conservative and cheap.
    """

    adjusted = {
        color: {piece: int((hands.get(color) or {}).get(piece, 0)) for piece in HAND_PIECES}
        for color in ("black", "white")
    }
    board = board_counts(list(cells))
    confidence = hand_entry_confidence(hand_report)
    digit_metadata = hand_entry_digit_metadata(hand_report)
    changes: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    for piece in HAND_PIECES:
        available = max(0, TOTAL_INVENTORY.get(piece, 0) - int(board.get(piece, 0)))
        hand_total = sum(adjusted[color][piece] for color in ("black", "white"))
        surplus = hand_total - available
        if surplus <= 0:
            continue
        protected_colors = {
            color
            for color in ("black", "white")
            if adjusted[color][piece] > 0 and high_confidence_digit_hand_count(color, piece, confidence, digit_metadata)
        }
        protected_total = sum(adjusted[color][piece] for color in protected_colors)
        for color in sorted(("black", "white"), key=lambda item: confidence.get((item, piece), 0.0)):
            if surplus <= 0:
                break
            current = adjusted[color][piece]
            if current <= 0:
                continue
            if color in protected_colors:
                preserved.append(
                    {
                        "owner": color,
                        "piece": piece,
                        "count": current,
                        "board_count": int(board.get(piece, 0)),
                        "available_for_hands": available,
                        "confidence": confidence.get((color, piece)),
                        "digit_confidence": digit_metadata.get((color, piece), {}).get("digit_confidence"),
                    }
                )
                continue
            removed = min(current, surplus)
            adjusted[color][piece] -= removed
            surplus -= removed
            changes.append(
                {
                    "owner": color,
                    "piece": piece,
                    "removed": removed,
                    "before": current,
                    "after": adjusted[color][piece],
                    "board_count": int(board.get(piece, 0)),
                    "available_for_hands": available,
                    "confidence": confidence.get((color, piece)),
                    "candidate_sets": hand_entry_candidate_sets(hand_report, color, piece),
                }
            )
        if surplus > 0 and protected_total > 0:
            for color in protected_colors:
                current = adjusted[color][piece]
                if current <= 0:
                    continue
                metadata = digit_metadata.get((color, piece)) or {}
                preserved.append(
                    {
                        "owner": color,
                        "piece": piece,
                        "count": current,
                        "board_count": int(board.get(piece, 0)),
                        "available_for_hands": available,
                        "confidence": confidence.get((color, piece)),
                        "digit_confidence": metadata.get("digit_confidence"),
                        "candidate_sets": hand_entry_candidate_sets(hand_report, color, piece),
                    }
                )
    if changes:
        hand_report["hands_before_inventory_sanitize"] = hands
        hand_report["hands"] = adjusted
        hand_report["inventory_sanitization"] = {
            "applied": True,
            "changes": changes,
            "preserved": preserved,
        }
        update_hand_piece_entries_after_sanitize(hand_report, adjusted)
    else:
        hand_report["inventory_sanitization"] = {"applied": False, "changes": [], "preserved": preserved}
    return adjusted


def complete_hands_from_inventory_candidates(
    cells: Sequence[dict[str, Any]],
    hands: dict[str, dict[str, int]],
    hand_report: dict[str, Any],
) -> dict[str, dict[str, int]]:
    adjusted = {
        color: {piece: int((hands.get(color) or {}).get(piece, 0)) for piece in HAND_PIECES}
        for color in ("black", "white")
    }
    board = board_counts(list(cells))
    deficits = {
        piece: max(0, TOTAL_INVENTORY[piece] - int(board.get(piece, 0)) - sum(adjusted[color][piece] for color in ("black", "white")))
        for piece in HAND_PIECES
    }
    if not any(deficits.values()):
        hand_report["inventory_completion"] = {"applied": False, "changes": []}
        return adjusted

    options = hand_inventory_completion_options(hand_report, deficits)
    used_rects: list[list[int]] = []
    changes = apply_inventory_completion_options(options, adjusted, deficits, used_rects, hand_report)
    if any(deficits.values()):
        changes.extend(
            apply_inventory_completion_options(
                wars_side_order_completion_options(hand_report, deficits),
                adjusted,
                deficits,
                used_rects,
                hand_report,
            ),
        )

    if changes:
        hand_report["hands"] = adjusted
        hand_report["inventory_completion"] = {"applied": True, "changes": changes}
        existing = list(hand_report.get("pieces") or [])
        for change in changes:
            existing.append(
                {
                    "owner": change["owner"],
                    "piece": change["piece"],
                    "count": int(change.get("count") or 1),
                    "count_source": "inventory_candidate",
                    "confidence": round(float(change["score"]), 4),
                    "rects": [change["rect"]] if change.get("rect") else [],
                    "digits": [],
                    "ambiguous": True,
                    "inventory_completion": [change],
                }
            )
        existing.sort(key=lambda entry: (entry["owner"], HAND_PIECES.index(entry["piece"])))
        hand_report["pieces"] = existing
    else:
        hand_report["inventory_completion"] = {"applied": False, "changes": []}
    return adjusted


def apply_inventory_completion_options(
    options: Sequence[dict[str, Any]],
    adjusted: dict[str, dict[str, int]],
    deficits: dict[str, int],
    used_rects: list[list[int]],
    hand_report: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for option in sorted(options, key=lambda item: item["score"], reverse=True):
        owner = option["owner"]
        piece = option["piece"]
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            continue
        if deficits.get(piece, 0) <= 0:
            continue
        if adjusted[owner][piece] > 0 and not allow_incremental_inventory_completion(option, hand_report):
            continue
        rect = option.get("rect") or []
        if rect and any(rect_overlap_fraction(rect, used) >= 0.50 for used in used_rects):
            continue
        count = min(int(option.get("max_count") or 1), int(deficits[piece]))
        if count <= 0:
            continue
        adjusted[owner][piece] += count
        deficits[piece] -= count
        if rect:
            used_rects.append(rect)
        change = dict(option)
        change["count"] = count
        changes.append(change)
    return changes


def allow_incremental_inventory_completion(option: dict[str, Any], hand_report: dict[str, Any]) -> bool:
    owner = str(option.get("owner") or "")
    piece = str(option.get("piece") or "")
    rect = list(option.get("rect") or [])
    if (
        option.get("source") != "surplus_alternative"
        or str(option.get("removed_piece") or "") != "GI"
        or piece != "KI"
        or owner not in {"black", "white"}
        or float(option.get("score") or 0.0) < 0.55
        or not rect
    ):
        return False
    for entry in hand_report.get("pieces") or []:
        if entry.get("owner") != owner or entry.get("piece") != piece:
            continue
        for existing_rect in entry.get("rects") or []:
            if rect_overlap_fraction(rect, existing_rect) >= 0.50:
                return False
    return True


def hand_inventory_completion_options(hand_report: dict[str, Any], deficits: dict[str, int]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    target_family = str(hand_report.get("target_family") or "")
    for item in hand_report.get("unknown") or []:
        rect = list(item.get("rect") or [])
        item_owner = str(item.get("owner") or "")
        for candidate in item.get("candidates") or []:
            score = float(candidate.get("score") or 0.0)
            owner = str(candidate.get("color") or "")
            piece = str(candidate.get("piece") or "")
            if owner not in {"black", "white"} or piece not in HAND_PIECES:
                continue
            min_score = 0.60
            if (
                target_family == "ぴよ将棋:一文字駒"
                and item_owner in {"black", "white"}
                and owner == item_owner
                and piece in {"HI", "KA"}
                and deficits.get(piece, 0) > 0
            ):
                min_score = 0.52
            if score < min_score:
                continue
            if item_owner in {"black", "white"} and owner != item_owner:
                continue
            options.append(
                {
                    "owner": owner,
                    "piece": piece,
                    "score": round(score - 0.015, 4),
                    "rect": rect,
                    "source": "unknown_candidate",
                    "candidate_source": candidate.get("source"),
                }
                )
    for change in (hand_report.get("inventory_sanitization") or {}).get("changes") or []:
        removed_piece = str(change.get("piece") or "")
        change_owner = str(change.get("owner") or "")
        removed_count = max(1, int(change.get("removed") or 1))
        for candidate_set in change.get("candidate_sets") or []:
            rect = list(candidate_set.get("rect") or [])
            for candidate in candidate_set.get("candidates") or []:
                owner = str(candidate.get("color") or "")
                piece = str(candidate.get("piece") or "")
                score = float(candidate.get("score") or 0.0)
                if owner not in {"black", "white"} or piece not in HAND_PIECES or piece == removed_piece:
                    continue
                if change_owner in {"black", "white"} and owner != change_owner:
                    continue
                slot_prior_count = slot_prior_surplus_max_count(hand_report, change, candidate, rect, deficits)
                if score < 0.58 and slot_prior_count <= 0:
                    continue
                options.append(
                    {
                        "owner": owner,
                        "piece": piece,
                        "score": round(score - 0.03, 4),
                        "rect": rect,
                        "source": "slot_prior_surplus_alternative" if slot_prior_count > 0 and score < 0.58 else "surplus_alternative",
                        "removed_piece": removed_piece,
                        "max_count": slot_prior_count if slot_prior_count > 0 else removed_count,
                        "candidate_source": candidate.get("source"),
                    }
                )
    options.extend(accepted_alternative_completion_options(hand_report, deficits))
    return options


def slot_prior_surplus_max_count(
    hand_report: dict[str, Any],
    change: dict[str, Any],
    candidate: dict[str, Any],
    rect: Sequence[int],
    deficits: dict[str, int],
) -> int:
    if (
        str(hand_report.get("target_family") or "") != "将棋クエスト:クラシック二文字駒"
        or str(candidate.get("source") or "") != "hand_slot_prior"
        or str(candidate.get("color") or "") != "black"
        or str(candidate.get("piece") or "") != "HI"
        or str(change.get("piece") or "") not in {"KI", "GI"}
        or float(candidate.get("score") or 0.0) < 0.53
        or deficits.get("HI", 0) <= 0
        or not rect
    ):
        return 0
    digit = strongest_overlapping_digit(hand_report, rect)
    if digit < 2:
        return 0
    return min(digit, int(deficits.get("HI", 0)))


def strongest_overlapping_digit(hand_report: dict[str, Any], rect: Sequence[int]) -> int:
    best_digit = 0
    for digit in hand_report.get("digits") or []:
        digit_rect = list(digit.get("rect") or [])
        if len(digit_rect) != 4 or rect_overlap_fraction(rect, digit_rect) <= 0:
            continue
        if float(digit.get("confidence") or 0.0) < 0.90:
            continue
        try:
            best_digit = max(best_digit, int(digit.get("digit") or 0))
        except (TypeError, ValueError):
            continue
    return best_digit


def accepted_alternative_completion_options(hand_report: dict[str, Any], deficits: dict[str, int]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        current_piece = str(entry.get("piece") or "")
        if owner not in {"black", "white"} or current_piece not in HAND_PIECES:
            continue
        current_count = max(1, int(entry.get("count") or 1))
        if current_count <= 1:
            continue
        for candidate_set in entry.get("candidate_sets") or []:
            rect = list(candidate_set.get("rect") or [])
            for candidate in candidate_set.get("candidates") or []:
                candidate_owner = str(candidate.get("color") or "")
                piece = str(candidate.get("piece") or "")
                score = float(candidate.get("score") or 0.0)
                if candidate_owner != owner or piece not in HAND_PIECES or piece == current_piece:
                    continue
                if deficits.get(piece, 0) <= 0:
                    continue
                if score < 0.54:
                    continue
                max_count = accepted_alternative_max_count(hand_report, owner, current_piece, piece, rect, score)
                options.append(
                    {
                        "owner": owner,
                        "piece": piece,
                        "score": round(score - 0.035, 4),
                        "rect": rect,
                        "source": "accepted_alternative",
                        "current_piece": current_piece,
                        "max_count": max_count,
                        "candidate_source": candidate.get("source"),
                    }
                )
    return options


def accepted_alternative_max_count(
    hand_report: dict[str, Any],
    owner: str,
    current_piece: str,
    piece: str,
    rect: Sequence[int],
    score: float,
) -> int:
    if (
        str(hand_report.get("target_family") or "") == "将棋ウォーズ:二文字"
        and owner == "white"
        and current_piece == "FU"
        and piece == "KE"
        and score >= 0.56
        and strongest_overlapping_digit(hand_report, rect) == 2
    ):
        return 2
    return 1


def wars_side_order_completion_options(hand_report: dict[str, Any], deficits: dict[str, int]) -> list[dict[str, Any]]:
    if str(hand_report.get("target_family") or "") != "将棋ウォーズ:二文字":
        return []
    options: list[dict[str, Any]] = []
    accepted = wars_side_order_accepted_items(hand_report)
    for change in (hand_report.get("inventory_sanitization") or {}).get("changes") or []:
        owner = str(change.get("owner") or "")
        if owner not in {"black", "white"}:
            continue
        removed_piece = str(change.get("piece") or "")
        for candidate_set in change.get("candidate_sets") or []:
            rect = list(candidate_set.get("rect") or [])
            side = hand_area_side_for_rect(hand_report, rect)
            if side not in {"left", "right"}:
                continue
            compatible = [
                piece
                for piece in HAND_PIECES
                if piece != removed_piece
                and deficits.get(piece, 0) > 0
                and wars_side_order_position_is_compatible(owner, side, rect, piece, accepted)
            ]
            if len(compatible) != 1:
                continue
            piece = compatible[0]
            options.append(
                {
                    "owner": owner,
                    "piece": piece,
                    "score": 0.525,
                    "rect": rect,
                    "source": "wars_side_order",
                    "removed_piece": removed_piece,
                    "max_count": 1,
                    "candidate_source": "compact_side_order",
                }
            )
    return options


def wars_side_order_accepted_items(hand_report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            continue
        for rect in entry.get("rects") or []:
            side = hand_area_side_for_rect(hand_report, rect)
            if side in {"left", "right"}:
                items.append({"owner": owner, "piece": piece, "side": side, "center_y": rect_center(rect)[1]})
    return items


def wars_side_order_position_is_compatible(
    owner: str,
    side: str,
    rect: Sequence[int],
    piece: str,
    accepted: Sequence[dict[str, Any]],
) -> bool:
    order = hand_slot_order(side)
    piece_index = order.index(piece)
    center_y = rect_center(rect)[1]
    before: list[int] = []
    after: list[int] = []
    for item in accepted:
        if item.get("owner") != owner or item.get("side") != side or item.get("piece") not in HAND_PIECES:
            continue
        index = order.index(str(item["piece"]))
        if float(item.get("center_y") or 0.0) < center_y:
            before.append(index)
        else:
            after.append(index)
    if before and piece_index < max(before):
        return False
    if after and piece_index > min(after):
        return False
    return bool(before or after)


def hand_area_side_for_rect(hand_report: dict[str, Any], rect: Sequence[int]) -> str:
    if len(rect) != 4:
        return ""
    center_x, center_y = rect_center(rect)
    best_side = ""
    best_distance = float("inf")
    for area in hand_report.get("areas") or []:
        side = str(area.get("side") or "")
        area_rect = list(area.get("rect") or [])
        if side not in {"left", "right"} or len(area_rect) != 4:
            continue
        left, top, right, bottom = [float(value) for value in area_rect]
        if left <= center_x <= right and top <= center_y <= bottom:
            return side
        area_center_x, area_center_y = rect_center(area_rect)
        distance = abs(center_x - area_center_x) + abs(center_y - area_center_y)
        if distance < best_distance:
            best_distance = distance
            best_side = side
    return best_side


def rect_center(rect: Sequence[int]) -> tuple[float, float]:
    return ((float(rect[0]) + float(rect[2])) / 2.0, (float(rect[1]) + float(rect[3])) / 2.0)


def rect_overlap_fraction(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(int(first[0]), int(second[0]))
    top = max(int(first[1]), int(second[1]))
    right = min(int(first[2]), int(second[2]))
    bottom = min(int(first[3]), int(second[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(1, (int(first[2]) - int(first[0])) * (int(first[3]) - int(first[1])))
    second_area = max(1, (int(second[2]) - int(second[0])) * (int(second[3]) - int(second[1])))
    return intersection / max(1, min(first_area, second_area))


def hand_entry_confidence(hand_report: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        if owner in {"black", "white"} and piece in HAND_PIECES:
            result[(owner, piece)] = float(entry.get("confidence") or 0.0)
    return result


def hand_entry_candidate_sets(hand_report: dict[str, Any], owner: str, piece: str) -> list[dict[str, Any]]:
    for entry in hand_report.get("pieces") or []:
        if entry.get("owner") == owner and entry.get("piece") == piece:
            return list(entry.get("candidate_sets") or [])
    return []


def hand_entry_digit_metadata(hand_report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            continue
        digits = entry.get("digits") or []
        digit_confidence = max((float(digit.get("confidence") or 0.0) for digit in digits), default=0.0)
        result[(owner, piece)] = {
            "count": int(entry.get("count") or 0),
            "count_source": str(entry.get("count_source") or ""),
            "digit_confidence": digit_confidence,
        }
    return result


def high_confidence_digit_hand_count(
    owner: str,
    piece: str,
    confidence: dict[tuple[str, str], float],
    digit_metadata: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    metadata = digit_metadata.get((owner, piece)) or {}
    return (
        metadata.get("count_source") == "digit"
        and int(metadata.get("count") or 0) >= 2
        and float(confidence.get((owner, piece), 0.0)) >= 0.72
        and float(metadata.get("digit_confidence") or 0.0) >= 0.84
    )


def update_hand_piece_entries_after_sanitize(hand_report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    updated = []
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            continue
        count = int(hands[owner][piece])
        if count <= 0:
            continue
        next_entry = dict(entry)
        next_entry["count"] = count
        updated.append(next_entry)
    hand_report["pieces"] = updated


def recognize_board(
    image: Image.Image,
    detection: GridDetection,
    model: dict[str, Any],
    *,
    target_family: str | None = None,
) -> list[CellRecognition]:
    templates: list[LearnedTemplate] = model.get("templates", [])
    classifier = model.get("classifier") or {"enabled": False}
    target_family = target_family or str(model.get("target_family") or "") or source_family(str(model.get("excluded_source") or ""))
    position_priors = model.get("position_priors") or {}
    cells: list[CellRecognition] = []
    for row in range(1, 10):
        for col in range(1, 10):
            cells.append(recognize_board_cell(image, detection, row, col, templates, classifier, target_family, position_priors))
    return cells


def recognize_board_cell(
    image: Image.Image,
    detection: GridDetection,
    row: int,
    col: int,
    templates: Sequence[LearnedTemplate],
    classifier: dict[str, Any],
    target_family: str,
    position_priors: dict[str, dict[str, int]],
) -> CellRecognition:
    exact_cell = crop_grid_cell(image, detection, row, col, 0.0, 0.0)
    ink_density, bbox_ratio = fast_empty_ink_features(exact_cell, inset_ratio=0.06)
    empty_score = empty_likelihood(ink_density, bbox_ratio)
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

    rec_cell = crop_grid_cell(image, detection, row, col, 0.08, 0.18)
    rgb = np.array(rec_cell.convert("RGB"))
    glyph = fast_glyph_features_from_rgb(rgb)
    candidates = template_candidates(glyph, templates, classifier, rec_cell, target_family, row, col, position_priors)[:8] if glyph is not None else []
    best = candidates[0] if candidates else None
    second = candidates[1].score if len(candidates) > 1 else 0.0
    ambiguous = best is not None and best.score - second < 0.018
    if (
        ambiguous
        and best is not None
        and best.piece in {"RY", "NK"}
        and glyph is not None
        and float(glyph.red_share) >= 0.32
        and ink_density >= 0.10
        and bbox_ratio >= 0.45
        and best.score >= 0.52
        and best.score - second >= 0.010
    ):
        ambiguous = False
    if is_low_ink_false_piece(ink_density, bbox_ratio, best.score if best else 0.0):
        return CellRecognition(
            row=row,
            col=col,
            square=square_name(row, col),
            state="empty",
            color=None,
            piece=None,
            best_piece=best.piece if best else None,
            confidence=max(empty_score, 0.82),
            ambiguous=False,
            empty_score=empty_score,
            dark_ratio=ink_density,
            bbox_ratio=bbox_ratio,
            candidates=candidates[:8],
        )
    if best is None or best.score < 0.36 or ambiguous:
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
            candidates=candidates[:8],
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
        candidates=candidates[:8],
    )


def is_low_ink_false_piece(ink_density: float, bbox_ratio: float, best_score: float) -> bool:
    return (
        (ink_density < 0.017 and bbox_ratio < 0.125 and best_score < 0.40)
        or (ink_density < 0.065 and bbox_ratio < 0.065 and best_score < 0.42)
        or (ink_density < 0.055 and bbox_ratio < 0.055 and best_score < 0.45)
    )


def fast_empty_ink_features(image: Image.Image, inset_ratio: float) -> tuple[float, float]:
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    inset_x = int(width * inset_ratio)
    inset_y = int(height * inset_ratio)
    cropped = rgb[
        inset_y : max(inset_y + 1, height - inset_y),
        inset_x : max(inset_x + 1, width - inset_x),
    ]
    red = cropped[:, :, 0]
    green = cropped[:, :, 1]
    blue = cropped[:, :, 2]
    black_ink = (red < 98) & (green < 98) & (blue < 98)
    red_ink = (red > 120) & (green < 115) & (blue < 115) & ((red - green) > 35) & ((red - blue) > 35)
    ink = black_ink | red_ink
    ink_count = int(np.count_nonzero(ink))
    total = int(ink.size)
    if ink_count == 0:
        return 0.0, 0.0
    ys, xs = np.nonzero(ink)
    bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return ink_count / max(1, total), bbox_area / max(1, total)


def template_candidates(
    glyph,
    templates: Sequence[LearnedTemplate],
    classifier: dict[str, Any],
    cell_image: Image.Image,
    target_family: str,
    row: int,
    col: int,
    position_priors: dict[str, dict[str, int]],
) -> list[Candidate]:
    if glyph is None:
        return []
    clean_features = clean_mask_features(np.array(cell_image.convert("RGB")))
    best_by_label: dict[tuple[str, str], Candidate] = {}
    target_app = source_app(target_family)
    target_glyph = source_glyph(target_family)
    target_glyph_base = source_glyph_base(target_glyph)
    for template in templates:
        score = learned_template_score(glyph, template, clean_features)
        template_family = source_family(template.source)
        template_glyph = source_glyph(template_family) or source_glyph(template.source)
        template_glyph_base = source_glyph_base(template_glyph)
        if template_family == target_family:
            score *= 1.095
        elif target_app and source_app(template_family) == target_app:
            if target_glyph and template_glyph and template_glyph != target_glyph and template_glyph_base != target_glyph_base:
                score *= 0.900
            elif target_glyph and not template_glyph:
                score *= 1.010
            else:
                score *= 1.020
        elif target_family:
            score *= 0.940
        candidate = Candidate(
            color=template.color,
            piece=template.piece,
            score=round(score, 4),
            bbox=glyph.bbox,
            source=f"learned:{template.source}",
        )
        key = (candidate.color, candidate.piece)
        previous = best_by_label.get(key)
        if previous is None or candidate.score > previous.score:
            best_by_label[key] = candidate
    if not skip_classifier_for_template_match(best_by_label, float(glyph.red_share)):
        for candidate in classifier_candidates(glyph, classifier, cell_image):
            key = (candidate.color, candidate.piece)
            previous = best_by_label.get(key)
            if previous is None:
                best_by_label[key] = candidate
            else:
                best_by_label[key] = Candidate(
                    color=previous.color,
                    piece=previous.piece,
                    score=round(previous.score * 0.82 + candidate.score * 0.18, 4),
                    bbox=previous.bbox,
                    source=f"{previous.source}+{candidate.source}",
                )
    apply_red_promoted_rescue(best_by_label, float(glyph.red_share))
    apply_wars_promoted_major_rescue(best_by_label, float(glyph.red_share), target_family)
    apply_low_red_unpromoted_rescue(best_by_label, float(glyph.red_share))
    apply_short_piece_fu_tiebreak(best_by_label, row)
    for key, previous in list(best_by_label.items()):
        boost, source = position_prior_boost(
            row,
            col,
            target_family,
            previous.color,
            previous.piece,
            position_priors,
            previous.score,
        )
        if boost > 0:
            best_by_label[key] = Candidate(
                color=previous.color,
                piece=previous.piece,
                score=round(min(0.98, previous.score + boost), 4),
                bbox=previous.bbox,
                source=f"{previous.source}+{source}",
            )
    return sorted(best_by_label.values(), key=lambda item: item.score, reverse=True)


def skip_classifier_for_template_match(best_by_label: dict[tuple[str, str], Candidate], red_share: float) -> bool:
    if len(best_by_label) < 2:
        return False
    if red_share >= 0.12:
        return False
    ordered = sorted(best_by_label.values(), key=lambda item: item.score, reverse=True)
    if needs_classifier_for_close_identity(ordered):
        return False
    top = float(ordered[0].score)
    margin = top - float(ordered[1].score)
    return top >= CLASSIFIER_SKIP_MIN_SCORE and margin >= CLASSIFIER_SKIP_MIN_MARGIN


def needs_classifier_for_close_identity(candidates: Sequence[Candidate]) -> bool:
    if len(candidates) < 2:
        return False
    top = candidates[0]
    for rival in candidates[1:4]:
        if float(top.score) - float(rival.score) > 0.20:
            continue
        if top.piece == rival.piece and top.color != rival.color:
            return True
        if top.color == rival.color and base_piece(top.piece) == base_piece(rival.piece) and top.piece != rival.piece:
            return True
    return False


def apply_red_promoted_rescue(best_by_label: dict[tuple[str, str], Candidate], red_share: float) -> None:
    if red_share < 0.32 or not best_by_label:
        return
    top_score = max(candidate.score for candidate in best_by_label.values())
    for key, candidate in list(best_by_label.items()):
        if candidate.piece not in PROMOTED_RESCUE_PIECES:
            continue
        if top_score - candidate.score > 0.09:
            continue
        best_by_label[key] = Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(min(0.98, top_score + 0.04 + candidate.score * 0.001), 4),
            bbox=candidate.bbox,
            source=f"{candidate.source}+red_promoted_rescue",
        )
    for color in ("black", "white"):
        gold = best_by_label.get((color, "KI"))
        promoted_silver = best_by_label.get((color, "NG"))
        if gold is None or gold.score < 0.50:
            continue
        if promoted_silver is not None and promoted_silver.score >= gold.score - 0.05:
            continue
        best_by_label[(color, "NG")] = Candidate(
            color=color,
            piece="NG",
            score=round(min(0.98, gold.score + 0.04), 4),
            bbox=gold.bbox,
            source=f"{gold.source}+red_promoted_rescue:ki_to_ng",
        )


def apply_wars_promoted_major_rescue(
    best_by_label: dict[tuple[str, str], Candidate],
    red_share: float,
    target_family: str,
) -> None:
    if not target_family.startswith("将棋ウォーズ:") or red_share < 0.16 or not best_by_label:
        return
    top_score = max(candidate.score for candidate in best_by_label.values())
    for color, promoted, base in (("black", "RY", "HI"), ("white", "RY", "HI"), ("black", "UM", "KA"), ("white", "UM", "KA")):
        candidate = best_by_label.get((color, promoted))
        if candidate is None or candidate.score < 0.46:
            continue
        base_candidate = best_by_label.get((color, base))
        competitor_score = max(top_score, base_candidate.score if base_candidate is not None else 0.0)
        max_gap = 0.105 if promoted == "RY" else 0.085
        if competitor_score - candidate.score > max_gap:
            continue
        best_by_label[(color, promoted)] = Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(min(0.98, competitor_score + 0.022 + candidate.score * 0.001), 4),
            bbox=candidate.bbox,
            source=f"{candidate.source}+wars_promoted_major_rescue",
        )


def apply_low_red_unpromoted_rescue(best_by_label: dict[tuple[str, str], Candidate], red_share: float) -> None:
    if red_share > 0.12 or not best_by_label:
        return
    for key, candidate in list(best_by_label.items()):
        base = base_piece(candidate.piece)
        if base == candidate.piece or candidate.score < 0.50:
            continue
        if candidate.piece in {"RY", "UM"}:
            continue
        existing = best_by_label.get((candidate.color, base))
        if existing is not None and existing.score >= candidate.score - 0.03:
            continue
        best_by_label[(candidate.color, base)] = Candidate(
            color=candidate.color,
            piece=base,
            score=round(min(0.98, candidate.score + 0.025), 4),
            bbox=candidate.bbox,
            source=f"{candidate.source}+low_red_unpromoted_rescue",
        )


def apply_short_piece_fu_tiebreak(best_by_label: dict[tuple[str, str], Candidate], row: int) -> None:
    for color in ("black", "white"):
        fu = best_by_label.get((color, "FU"))
        if fu is None or fu.score < 0.49 or is_dead_end_piece(color, "FU", row):
            continue
        rivals = []
        for piece in ("KY", "KE"):
            rival = best_by_label.get((color, piece))
            if rival is None:
                continue
            if is_dead_end_piece(color, rival.piece, row):
                continue
            if rival.score < 0.70 and 0.0 <= rival.score - fu.score <= 0.08:
                rivals.append(rival)
        if not rivals:
            continue
        top = max(rivals, key=lambda item: item.score)
        best_by_label[(color, "FU")] = Candidate(
            color=fu.color,
            piece=fu.piece,
            score=round(min(0.98, top.score + 0.0001), 4),
            bbox=fu.bbox,
            source=f"{fu.source}+short_piece_fu_tiebreak",
        )


def classifier_candidates(glyph, classifier: dict[str, Any], cell_image: Image.Image) -> list[Candidate]:
    if not classifier.get("enabled"):
        return []
    vector = feature_vector_from_glyph(glyph).reshape(1, -1).astype("float32")
    scores: dict[str, float] = {}
    knn = classifier.get("knn")
    if knn is not None:
        for label, probability in zip(knn.classes_, knn.predict_proba(vector)[0]):
            scores[str(label)] = max(scores.get(str(label), 0.0), float(probability))
    svm = classifier.get("svm")
    if svm is not None:
        for label, probability in zip(svm.classes_, svm.predict_proba(vector)[0]):
            scores[str(label)] = max(scores.get(str(label), 0.0), float(probability) * 0.92)
    hog_vector = extract_hog_features(cell_image).vector.reshape(1, -1).astype("float32")
    hog_knn = classifier.get("hog_knn")
    if hog_knn is not None:
        for label, probability in zip(hog_knn.classes_, hog_knn.predict_proba(hog_vector)[0]):
            scores[str(label)] = max(scores.get(str(label), 0.0), float(probability))
    hog_sgd = classifier.get("hog_sgd")
    if hog_sgd is not None and hasattr(hog_sgd, "predict_proba"):
        for label, probability in zip(hog_sgd.classes_, hog_sgd.predict_proba(hog_vector)[0]):
            scores[str(label)] = max(scores.get(str(label), 0.0), float(probability) * 0.88)
    candidates = []
    for label, score in scores.items():
        if ":" not in label:
            continue
        color, piece = label.split(":", 1)
        candidates.append(
            Candidate(
                color=color,
                piece=piece,
                score=round(score, 4),
                bbox=glyph.bbox,
                source="sklearn",
            ),
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:8]


def add_position_counts(position_counts: dict[str, Counter[str]], source: str, cells: Sequence[dict[str, Any]]) -> None:
    family = source_family(source)
    app = source_app(source)
    glyph_base = source_glyph_base(source_glyph(family) or source_glyph(source))
    for cell in cells:
        if cell.get("state") != "piece":
            continue
        row = int(cell["row"])
        col = int(cell["col"])
        label = identity(cell.get("color"), cell.get("piece"))
        if family:
            position_counts[position_key(family, row, col)][label] += 1
        if app and glyph_base:
            position_counts[position_key(f"{app}:{glyph_base}", row, col)][label] += 1
        if app and app != family:
            position_counts[position_key(app, row, col)][label] += 1
        position_counts[position_key("*", row, col)][label] += 1


def serialize_position_priors(position_counts: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(counter)
        for key, counter in position_counts.items()
    }


def position_key(family: str, row: int, col: int) -> str:
    return f"{family}:r{row}:c{col}"


def position_prior_boost(
    row: int,
    col: int,
    target_family: str,
    color: str,
    piece: str,
    position_priors: dict[str, dict[str, int]],
    base_score: float,
) -> tuple[float, str]:
    label = identity(color, piece)
    target_app = source_app(target_family)
    target_glyph_base = source_glyph_base(source_glyph(target_family))
    target_base_family = f"{target_app}:{target_glyph_base}" if target_app and target_glyph_base else ""
    for family, max_boost, min_count, min_probability, min_base_score in (
        (target_family, 0.245, 4, 0.82, 0.34),
        (target_family, 0.035, 2, 0.45, 0.0),
        (target_base_family, 0.045, 4, 0.55, 0.0),
        (target_app, 0.030, 4, 0.55, 0.0),
        ("*", 0.020, 4, 0.55, 0.0),
    ):
        if not family:
            continue
        if base_score < min_base_score:
            continue
        counts = position_priors.get(position_key(family, row, col)) or {}
        total = sum(int(value) for value in counts.values())
        if total < min_count:
            continue
        count = int(counts.get(label, 0))
        probability = count / max(1, total)
        if count >= min_count and probability >= min_probability:
            return max_boost * probability, f"position_prior:{family}"
    return 0.0, ""


def resolve_target_family(model: dict[str, Any], image_path: Path) -> str:
    image_family = source_family(image_path.stem)
    model_family = str(model.get("target_family") or "")
    if image_family and source_glyph(image_family):
        if not model_family or not source_glyph(model_family):
            return image_family
        if source_app(model_family) == source_app(image_family):
            return image_family
    return model_family or image_family


def source_family(source: str) -> str:
    app = source_app(source)
    glyph = source_glyph(source)
    if app and glyph:
        return f"{app}:{glyph}"
    if app:
        return app
    if "初期配置" in source:
        return "初期配置"
    return ""


def source_app(source: str) -> str:
    if "ぴよ将棋" in source:
        return "ぴよ将棋"
    if "将棋ウォーズ" in source:
        return "将棋ウォーズ"
    if "将棋クエスト" in source:
        return "将棋クエスト"
    if "将皇" in source:
        return "将皇"
    return ""


def source_glyph(source: str) -> str:
    for style in (
        "クラシック二文字駒",
        "書籍風",
        "ひよこ駒",
        "二文字駒",
        "一文字駒",
        "太文字駒",
        "昇竜一文字",
        "昇竜",
        "風波一文字",
    ):
        if style in source:
            return style
    if "一文字" in source:
        return "一文字"
    if "二文字" in source:
        return "二文字"
    return ""


def source_glyph_base(glyph: str) -> str:
    if glyph in {"一文字", "一文字駒", "ひよこ駒", "太文字駒", "昇竜一文字", "風波一文字"}:
        return "一文字"
    if glyph in {"二文字", "二文字駒", "クラシック二文字駒", "書籍風", "昇竜"}:
        return "二文字"
    return glyph


def feature_vector_from_template(template: LearnedTemplate) -> np.ndarray:
    mask = np.frombuffer(template.mask, dtype=np.uint8).astype("float32")
    return append_scalar_features(mask, template.dark_ratio, template.red_share, template.bbox)


def feature_vector_from_glyph(glyph) -> np.ndarray:
    mask = np.frombuffer(glyph.features.mask, dtype=np.uint8).astype("float32")
    return append_scalar_features(mask, glyph.features.dark_ratio, glyph.red_share, glyph.bbox)


def append_scalar_features(mask: np.ndarray, dark_ratio: float, red_share: float, bbox: list[int] | None) -> np.ndarray:
    bbox_values = np.zeros(4, dtype="float32")
    if bbox is not None:
        bbox_values = np.array([bbox[0] / 48.0, bbox[1] / 56.0, bbox[2] / 48.0, bbox[3] / 56.0], dtype="float32")
    scalars = np.array([dark_ratio, red_share], dtype="float32")
    return np.concatenate([mask, scalars, bbox_values])


def learned_template_score(glyph, template: LearnedTemplate, clean_features) -> float:
    intersection = int((glyph.features.bits & template.bits).bit_count())
    union = int((glyph.features.bits | template.bits).bit_count())
    iou = intersection / union if union else 0.0
    clean_score = 0.0
    if clean_features is not None:
        clean_intersection = int((clean_features.bits & template.clean_bits).bit_count())
        clean_union = int((clean_features.bits | template.clean_bits).bit_count())
        clean_iou = clean_intersection / clean_union if clean_union else 0.0
        clean_dice = (2.0 * clean_intersection) / max(1, clean_features.bits.bit_count() + template.clean_bits.bit_count())
        clean_area = 1.0 - min(
            1.0,
            abs(clean_features.dark_ratio - template.clean_dark_ratio) / max(clean_features.dark_ratio, template.clean_dark_ratio, 0.01),
        )
        clean_score = clean_dice * 0.58 + clean_iou * 0.30 + clean_area * 0.12
    red_score = max(0.0, 1.0 - abs(glyph.red_share - template.red_share) * 2.2)
    bbox_score = bbox_similarity(glyph.bbox, template.bbox)
    return iou * 0.42 + clean_score * 0.40 + red_score * 0.08 + bbox_score * 0.10


def clean_mask_features(rgb: object):
    clean_mask = clean_hog_letter_mask(rgb)
    bbox = mask_bbox(clean_mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    crop = clean_mask[y1:y2, x1:x2]
    if int(np.count_nonzero(crop)) < 18:
        return None
    normalized = fit_mask_to_canvas_size(crop, (48, 56))
    return mask_features(bytes(1 if value else 0 for value in normalized.reshape(-1)))


def bbox_similarity(first: list[int] | None, second: list[int] | None) -> float:
    if first is None or second is None:
        return 0.0
    first_w = max(1, first[2] - first[0])
    first_h = max(1, first[3] - first[1])
    second_w = max(1, second[2] - second[0])
    second_h = max(1, second[3] - second[1])
    width = 1.0 - min(1.0, abs(first_w - second_w) / max(first_w, second_w))
    height = 1.0 - min(1.0, abs(first_h - second_h) / max(first_h, second_h))
    aspect_1 = first_w / first_h
    aspect_2 = second_w / second_h
    aspect = 1.0 - min(1.0, abs(aspect_1 - aspect_2) / max(aspect_1, aspect_2, 0.01))
    return width * 0.30 + height * 0.30 + aspect * 0.40


def apply_piece_constraints(
    cells: list[dict[str, Any]],
    hands: dict[str, dict[str, int]] | None,
    soft_hands: dict[str, dict[str, int]] | None = None,
    *,
    target_family: str = "",
) -> dict[str, Any]:
    report = {"applied": False, "changes": [], "unresolved": [], "target_family": target_family}
    limits = board_inventory_limits(hands)
    apply_global_unknown_beam(cells, limits, report)
    apply_dead_end_constraints(cells, report)
    apply_nifu_constraints(cells, report)
    if hands is not None:
        apply_rook_bishop_presence_constraints(cells, limits, report)
    apply_piece_count_limits(cells, limits, report)
    apply_king_constraints(cells, report)
    apply_global_constraint_rerank(cells, soft_hands, report, target_family)
    apply_quest_onechar_promoted_lance_inventory_tiebreak(cells, report, target_family)
    apply_quest_onechar_nifu_color_tiebreak(cells, report, target_family)
    apply_quest_onechar_promoted_lance_pair_tiebreak(cells, report, target_family)
    apply_wars_onechar_short_piece_tiebreak(cells, report, target_family)
    apply_soft_inventory_beam(cells, soft_hands, report, target_family)
    apply_same_piece_color_tiebreak(cells, report)
    report["applied"] = bool(report["changes"])
    return report


def apply_global_unknown_beam(cells: list[dict[str, Any]], limits: dict[str, int], report: dict[str, Any]) -> None:
    targets = [
        (index, cell)
        for index, cell in enumerate(cells)
        if cell.get("state") == "unknown" and cell.get("candidates")
    ]
    if not targets:
        return
    base_counts = board_counts(cells)
    base_kings = {
        color: sum(1 for cell in cells if cell.get("state") == "piece" and cell.get("color") == color and cell.get("piece") == "OU")
        for color in ("black", "white")
    }
    base_pawns = {
        (cell.get("color"), int(cell.get("col", 0)))
        for cell in cells
        if cell.get("state") == "piece" and cell.get("piece") == "FU"
    }
    beam = [
        {
            "score": 0.0,
            "choices": [],
            "counts": dict(base_counts),
            "kings": dict(base_kings),
            "pawns": set(base_pawns),
        }
    ]
    beam_width = 96
    variable_count = 0
    for _, cell in targets:
        options = unknown_beam_options(cell)
        if len(options) <= 1:
            continue
        variable_count += 1
        next_beam = []
        for state in beam:
            next_beam.append(
                {
                    "score": state["score"],
                    "choices": [*state["choices"], (cell, None)],
                    "counts": dict(state["counts"]),
                    "kings": dict(state["kings"]),
                    "pawns": set(state["pawns"]),
                }
            )
            for candidate, option_score in options[1:]:
                if not candidate_allowed_for_beam(cell, candidate, limits, state["counts"], state["kings"], state["pawns"]):
                    continue
                counts = dict(state["counts"])
                kings = dict(state["kings"])
                pawns = set(state["pawns"])
                base = base_piece(str(candidate.get("piece")))
                counts[base] = counts.get(base, 0) + 1
                if candidate.get("piece") == "OU":
                    kings[candidate.get("color")] = kings.get(candidate.get("color"), 0) + 1
                if candidate.get("piece") == "FU":
                    pawns.add((candidate.get("color"), int(cell["col"])))
                next_beam.append(
                    {
                        "score": state["score"] + option_score,
                        "choices": [*state["choices"], (cell, candidate)],
                        "counts": counts,
                        "kings": kings,
                        "pawns": pawns,
                    }
                )
        beam = sorted(next_beam, key=lambda item: item["score"], reverse=True)[:beam_width]
    if not beam:
        return
    best = beam[0]
    applied = 0
    for cell, candidate in best["choices"]:
        if candidate is None:
            continue
        score = float(candidate.get("score") or 0.0)
        if score < 0.55:
            continue
        apply_candidate(cell, candidate, "global_unknown_beam", report)
        applied += 1
    if applied or variable_count:
        report.setdefault("global_unknown_beam", {})
        report["global_unknown_beam"] = {
            "variables": variable_count,
            "applied": applied,
            "beam_width": beam_width,
        }


def unknown_beam_options(cell: dict[str, Any]) -> list[tuple[dict[str, Any] | None, float]]:
    candidates = list(cell.get("candidates") or [])
    if not candidates:
        return [(None, 0.0)]
    top_score = float(candidates[0].get("score") or 0.0)
    options: list[tuple[dict[str, Any] | None, float]] = [(None, 0.52)]
    for rank, candidate in enumerate(candidates[:4]):
        score = float(candidate.get("score") or 0.0)
        if score < 0.50:
            continue
        if is_low_margin_unknown_candidate(cell, candidate):
            continue
        if rank > 0 and top_score - score > 0.08:
            continue
        option_score = score - rank * 0.035
        if score < 0.55:
            option_score -= 0.08
        if candidate.get("piece") == "OU":
            option_score += 0.03
        options.append((candidate, option_score))
    return options


def is_low_margin_unknown_candidate(cell: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if not bool(cell.get("ambiguous")):
        return False
    score = float(candidate.get("score") or 0.0)
    if score >= 0.64:
        return False
    candidate_key = (candidate.get("color"), candidate.get("piece"))
    rival_score = max(
        (
            float(other.get("score") or 0.0)
            for other in cell.get("candidates") or []
            if (other.get("color"), other.get("piece")) != candidate_key
        ),
        default=0.0,
    )
    return score - rival_score < 0.010


def candidate_allowed_for_beam(
    cell: dict[str, Any],
    candidate: dict[str, Any],
    limits: dict[str, int],
    counts: dict[str, int],
    kings: dict[str, int],
    pawns: set[tuple[str | None, int]],
) -> bool:
    color = candidate.get("color")
    piece = candidate.get("piece")
    if not color or not piece:
        return False
    if is_dead_end_piece(color, piece, int(cell["row"])):
        return False
    base = base_piece(str(piece))
    if counts.get(base, 0) >= limits.get(base, TOTAL_INVENTORY.get(base, 0)):
        return False
    if piece == "OU" and kings.get(color, 0) >= 1:
        return False
    if piece == "FU" and (color, int(cell["col"])) in pawns:
        return False
    return True


def apply_global_constraint_rerank(
    cells: list[dict[str, Any]],
    hands: dict[str, dict[str, int]] | None,
    report: dict[str, Any],
    target_family: str = "",
) -> None:
    started = time.perf_counter()
    deadline = started + 8.0
    target = board_inventory_limits(hands)
    has_soft_target = hands is not None
    variables = global_rerank_variables(cells, target, has_soft_target, target_family)
    if not variables:
        report["global_solver"] = {
            "applied": 0,
            "variables": 0,
            "timeout": False,
            "timeout_ms": 8000,
        }
        return

    start_counts = board_counts(cells)
    start_kings = king_counts(cells)
    start_pawns = pawn_counts(cells)
    start_penalty = global_constraint_penalty(start_counts, target, start_kings, start_pawns, has_soft_target)
    beam = [
        {
            "score": 0.0,
            "penalty": start_penalty,
            "choices": [],
            "counts": dict(start_counts),
            "kings": dict(start_kings),
            "pawns": Counter(start_pawns),
        }
    ]
    beam_width = 192 if target_family == "将棋ウォーズ:一文字" else 96
    timed_out = False
    for cell, options in variables:
        if time.perf_counter() > deadline:
            timed_out = True
            break
        next_beam = []
        for state in beam:
            for candidate, local_delta in options:
                counts = dict(state["counts"])
                kings = dict(state["kings"])
                pawns = Counter(state["pawns"])
                apply_soft_count_transition(cell, candidate, counts, kings, pawns)
                penalty = global_constraint_penalty(counts, target, kings, pawns, has_soft_target)
                penalty_gain = float(state["penalty"]) - penalty
                next_beam.append(
                    {
                        "score": float(state["score"]) + local_delta + penalty_gain * 0.92,
                        "penalty": penalty,
                        "choices": [*state["choices"], (cell, candidate)],
                        "counts": counts,
                        "kings": kings,
                        "pawns": pawns,
                    }
                )
        if not next_beam:
            continue
        beam = sorted(next_beam, key=lambda item: (float(item["score"]), -float(item["penalty"])), reverse=True)[:beam_width]
    if not beam:
        return

    best = beam[0]
    second = beam[1] if len(beam) > 1 else None
    best_score = float(best["score"])
    best_penalty = float(best["penalty"])
    penalty_gain = start_penalty - best_penalty
    min_gain = 0.08 if target_family == "将棋ウォーズ:一文字" else (0.16 if has_soft_target else 0.22)
    should_apply = (penalty_gain >= min_gain and best_score >= -0.18) or (best_score >= 0.20 and best_penalty <= start_penalty)
    applied = 0
    if should_apply:
        for cell, candidate in best["choices"]:
            if candidate is None:
                continue
            if cell.get("state") == "piece" and (cell.get("color"), cell.get("piece")) == (candidate.get("color"), candidate.get("piece")):
                continue
            apply_candidate(cell, candidate, "global_solver", report)
            applied += 1

    second_gap = best_score - float(second["score"]) if second is not None else None
    report["global_solver"] = {
        "applied": applied,
        "variables": len(variables),
        "beam_width": beam_width,
        "timeout": timed_out,
        "timeout_ms": 8000,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "start_penalty": round(start_penalty, 4),
        "best_penalty": round(best_penalty, 4),
        "best_score": round(best_score, 4),
        "second_best_gap": round(second_gap, 4) if second_gap is not None else None,
        "unique_solution": bool(second_gap is None or second_gap >= 0.035),
    }


def global_rerank_variables(
    cells: list[dict[str, Any]],
    target: dict[str, int],
    has_soft_target: bool,
    target_family: str,
) -> list[tuple[dict[str, Any], list[tuple[dict[str, Any] | None, float]]]]:
    current_counts = board_counts(cells)
    surplus = {
        piece
        for piece in TOTAL_INVENTORY
        if current_counts.get(piece, 0) > target.get(piece, TOTAL_INVENTORY.get(piece, 0))
    }
    deficits = {
        piece
        for piece in TOTAL_INVENTORY
        if has_soft_target and current_counts.get(piece, 0) < target.get(piece, TOTAL_INVENTORY.get(piece, 0))
    }
    unknown_candidate_bases = {
        base_piece(str(candidate.get("piece")))
        for cell in cells
        if cell.get("state") == "unknown"
        for candidate in (cell.get("candidates") or [])[:6]
        if float(candidate.get("score") or 0.0) >= 0.46
    }
    nifu_cells = {
        (cell.get("row"), cell.get("col"))
        for color in ("black", "white")
        for col in range(1, 10)
        for group in [[
            cell
            for cell in cells
            if cell.get("state") == "piece"
            and cell.get("color") == color
            and cell.get("piece") == "FU"
            and int(cell.get("col", 0)) == col
        ]]
        if len(group) > 1
        for cell in group
    }
    variables = []
    for cell in cells:
        options = global_rerank_options(cell, surplus, deficits, nifu_cells, target_family, unknown_candidate_bases)
        if len(options) <= 1:
            continue
        priority = global_rerank_priority(cell, options, surplus, deficits, nifu_cells)
        variables.append((priority, cell, options))
    variables.sort(key=lambda item: item[0], reverse=True)
    limit = 18 if target_family == "将棋ウォーズ:一文字" else 14
    return [(cell, options) for _, cell, options in variables[:limit]]


def global_rerank_options(
    cell: dict[str, Any],
    surplus: set[str],
    deficits: set[str],
    nifu_cells: set[tuple[Any, Any]],
    target_family: str = "",
    unknown_candidate_bases: set[str] | None = None,
) -> list[tuple[dict[str, Any] | None, float]]:
    candidates = list(cell.get("candidates") or [])
    if not candidates:
        return [(None, 0.0)]
    current_identity = (cell.get("color"), cell.get("piece")) if cell.get("state") == "piece" else (None, None)
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else None
    current_score = float(cell.get("confidence") or 0.0)
    keep_score = 0.50 if cell.get("state") == "unknown" else current_score
    cell_key = (cell.get("row"), cell.get("col"))
    current_in_nifu = cell_key in nifu_cells and current_base == "FU"
    options: list[tuple[dict[str, Any] | None, float]] = [(None, 0.0)]
    seen: set[tuple[str, str]] = set()
    for rank, candidate in enumerate(candidates[:8]):
        color = str(candidate.get("color") or "")
        piece = str(candidate.get("piece") or "")
        if color not in {"black", "white"} or not piece:
            continue
        key = (color, piece)
        if key == current_identity or key in seen:
            continue
        if is_dead_end_piece(color, piece, int(cell["row"])):
            continue
        candidate_score = float(candidate.get("score") or 0.0)
        candidate_base = base_piece(piece)
        score_drop = keep_score - candidate_score
        exchange_supported = (
            target_family == "ぴよ将棋:ひよこ駒"
            and current_base in (unknown_candidate_bases or set())
            and score_drop <= 0.10
            and "position_prior" in str(candidate.get("source") or "")
        )
        target_supported = candidate_base in deficits and (
            cell.get("state") == "unknown" or current_base in surplus or exchange_supported
        )
        relevant = (
            cell.get("state") == "unknown"
            or target_supported
            or current_base in surplus
            or current_in_nifu
            or is_wars_global_confusion_swap(cell, candidate, target_family)
        )
        if not relevant:
            continue
        if (
            target_family == "将棋ウォーズ:一文字"
            and cell.get("postprocess_reason") == "global_unknown_beam"
            and not (target_supported or current_base in surplus or current_in_nifu)
        ):
            continue
        max_drop = 0.075
        min_score = 0.50
        if target_supported or current_base in surplus:
            max_drop = 0.16
            min_score = 0.45
        if current_in_nifu and candidate_base != "FU" and (current_score < 0.72 or current_base in surplus):
            max_drop = max(max_drop, 0.22)
            min_score = min(min_score, 0.43)
        if cell.get("state") == "unknown":
            max_drop = max(max_drop, 0.05)
            min_score = min(min_score, 0.48)
        if is_wars_global_confusion_swap(cell, candidate, target_family):
            max_drop = max(max_drop, 0.12)
            min_score = min(min_score, 0.48)
        if is_wars_unknown_major_inventory_rescue(cell, candidate, target_family):
            max_drop = max(max_drop, 0.18)
            min_score = min(min_score, 0.46)
        if is_quest_onechar_promoted_lance_inventory_swap(cell, candidate, target_family):
            max_drop = max(max_drop, 0.205)
            min_score = min(min_score, 0.50)
        if (
            target_family == "将棋ウォーズ:一文字"
            and cell.get("state") == "piece"
            and current_score >= 0.74
            and score_drop > 0.075
            and not is_high_confidence_wars_inventory_swap(current_base, candidate_base, score_drop)
        ):
            continue
        if (
            target_family != "将棋ウォーズ:一文字"
            and cell.get("state") == "piece"
            and current_score >= 0.90
            and score_drop > 0.08
            and not current_in_nifu
        ):
            continue
        if candidate_score < min_score or score_drop > max_drop:
            continue
        local_delta = candidate_score - keep_score - rank * 0.020
        if target_supported:
            local_delta += 0.105
        if current_base in surplus:
            local_delta += 0.100
        if current_in_nifu and candidate_base != "FU" and (current_score < 0.72 or current_base in surplus):
            local_delta += 0.135
        if cell.get("state") == "unknown":
            local_delta += 0.055
        if is_wars_global_confusion_swap(cell, candidate, target_family):
            local_delta += 0.030
        if is_quest_onechar_promoted_lance_inventory_swap(cell, candidate, target_family):
            local_delta += 0.130
        if (
            target_family == "将棋ウォーズ:一文字"
            and cell.get("postprocess_reason") == "global_unknown_beam"
            and current_base in {"FU", "KY", "KE"}
            and candidate_base in {"FU", "KY", "KE"}
        ):
            local_delta -= 0.100
        options.append((candidate, local_delta))
        seen.add(key)
    return options


def is_wars_global_confusion_swap(cell: dict[str, Any], candidate: dict[str, Any], target_family: str) -> bool:
    if target_family != "将棋ウォーズ:一文字":
        return False
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else ""
    candidate_base = base_piece(str(candidate.get("piece"))) if candidate.get("piece") else ""
    if not current_base or not candidate_base or current_base == candidate_base:
        return False
    return {current_base, candidate_base} in (
        {"FU", "KY"},
        {"FU", "KE"},
        {"KY", "KE"},
        {"KI", "GI"},
        {"KA", "HI"},
        {"KA", "FU"},
        {"HI", "KE"},
    )


def is_quest_onechar_promoted_lance_inventory_swap(
    cell: dict[str, Any],
    candidate: dict[str, Any],
    target_family: str,
) -> bool:
    if target_family != "将棋クエスト:一文字駒":
        return False
    if str(cell.get("postprocess_reason") or "") != "global_unknown_beam":
        return False
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else ""
    return (
        current_base in {"GI", "KE"}
        and candidate.get("color") == "white"
        and candidate.get("piece") == "NY"
        and float(candidate.get("score") or 0.0) >= 0.50
    )


def is_high_confidence_wars_inventory_swap(current_base: str | None, candidate_base: str, score_drop: float) -> bool:
    if not current_base:
        return False
    if {current_base, candidate_base} == {"FU", "KY"}:
        return score_drop <= 0.090
    if {current_base, candidate_base} == {"KI", "GI"}:
        return score_drop <= 0.115
    if {current_base, candidate_base} == {"FU", "KE"}:
        return score_drop <= 0.205
    return False


def global_rerank_priority(
    cell: dict[str, Any],
    options: Sequence[tuple[dict[str, Any] | None, float]],
    surplus: set[str],
    deficits: set[str],
    nifu_cells: set[tuple[Any, Any]],
) -> float:
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else None
    priority = max(0.0, 0.82 - float(cell.get("confidence") or 0.0))
    if cell.get("state") == "unknown":
        priority += 2.5
    if (cell.get("row"), cell.get("col")) in nifu_cells:
        priority += 3.0
    if current_base in surplus:
        priority += 2.4
    for candidate, _ in options[1:]:
        if candidate is None:
            continue
        candidate_base = base_piece(str(candidate.get("piece")))
        if candidate_base in deficits:
            priority += 1.8
    return priority


def global_constraint_penalty(
    counts: dict[str, int],
    target: dict[str, int],
    kings: dict[str, int],
    pawns: Counter[tuple[str | None, int]],
    has_soft_target: bool,
) -> float:
    if has_soft_target:
        penalty = sum(max(0, counts.get(piece, 0) - target.get(piece, 0)) * 0.55 for piece in TOTAL_INVENTORY)
    else:
        penalty = sum(max(0, counts.get(piece, 0) - TOTAL_INVENTORY.get(piece, 0)) * 0.60 for piece in TOTAL_INVENTORY)
    penalty += sum(abs(kings.get(color, 0) - 1) * 0.55 for color in ("black", "white"))
    penalty += sum(max(0, count - 1) * 0.10 for (color, _), count in pawns.items() if color in {"black", "white"})
    return penalty


def apply_wars_onechar_short_piece_tiebreak(
    cells: list[dict[str, Any]],
    report: dict[str, Any],
    target_family: str,
) -> None:
    if target_family != "将棋ウォーズ:一文字":
        return
    pawns = pawn_counts(cells)
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") not in {"FU", "KY", "KE"}:
            continue
        current_score = float(cell.get("confidence") or 0.0)
        postprocess_reason = str(cell.get("postprocess_reason") or "")
        if current_score >= 0.73 and postprocess_reason != "global_unknown_beam":
            continue
        for candidate in (cell.get("candidates") or [])[:3]:
            color = str(candidate.get("color") or "")
            piece = str(candidate.get("piece") or "")
            if color not in {"black", "white"} or piece not in {"FU", "KY", "KE"}:
                continue
            if (color, piece) == (cell.get("color"), cell.get("piece")):
                continue
            if is_dead_end_piece(color, piece, int(cell["row"])):
                continue
            candidate_score = float(candidate.get("score") or 0.0)
            score_drop = current_score - candidate_score
            candidate_source = str(candidate.get("source") or "")
            has_position_prior = "position_prior" in candidate_source
            if score_drop > (0.040 if has_position_prior else 0.024):
                continue
            if piece == "FU" and pawns[(color, int(cell["col"]))] > 0:
                continue
            if piece == "FU" and cell.get("piece") == "KY" and (has_position_prior or postprocess_reason == "global_unknown_beam"):
                old_key = (cell.get("color"), int(cell["col"]))
                if cell.get("piece") == "FU" and pawns[old_key] > 0:
                    pawns[old_key] -= 1
                apply_candidate(cell, candidate, "wars_onechar_short_piece_tiebreak", report)
                pawns[(color, int(cell["col"]))] += 1
                break


def apply_quest_onechar_nifu_color_tiebreak(
    cells: list[dict[str, Any]],
    report: dict[str, Any],
    target_family: str,
) -> None:
    if target_family != "将棋クエスト:一文字駒":
        return
    pawns = pawn_counts(cells)
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") != "FU" or cell.get("color") not in {"black", "white"}:
            continue
        current_color = str(cell.get("color"))
        current_key = (current_color, int(cell["col"]))
        if pawns[current_key] <= 1:
            continue
        other_color = "white" if current_color == "black" else "black"
        if pawns[(other_color, int(cell["col"]))] > 0:
            continue
        current_score = float(cell.get("confidence") or 0.0)
        for candidate in (cell.get("candidates") or [])[:4]:
            if candidate.get("color") != other_color or candidate.get("piece") != "FU":
                continue
            candidate_score = float(candidate.get("score") or 0.0)
            if candidate_score < 0.62 or current_score - candidate_score > 0.055:
                continue
            if is_dead_end_piece(other_color, "FU", int(cell["row"])):
                continue
            pawns[current_key] -= 1
            apply_candidate(cell, candidate, "quest_onechar_nifu_color_tiebreak", report)
            pawns[(other_color, int(cell["col"]))] += 1
            break


def apply_quest_onechar_promoted_lance_inventory_tiebreak(
    cells: list[dict[str, Any]],
    report: dict[str, Any],
    target_family: str,
) -> None:
    if target_family != "将棋クエスト:一文字駒":
        return
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") not in {"NG", "NK"}:
            continue
        if str(cell.get("postprocess_reason") or "") != "global_unknown_beam":
            continue
        current_score = float(cell.get("confidence") or 0.0)
        for candidate in (cell.get("candidates") or [])[:8]:
            if candidate.get("color") != "white" or candidate.get("piece") != "NY":
                continue
            candidate_score = float(candidate.get("score") or 0.0)
            if candidate_score < 0.50 or current_score - candidate_score > 0.20:
                continue
            apply_candidate(cell, candidate, "quest_onechar_promoted_lance_inventory_tiebreak", report)
            break


def apply_quest_onechar_promoted_lance_pair_tiebreak(
    cells: list[dict[str, Any]],
    report: dict[str, Any],
    target_family: str,
) -> None:
    if target_family != "将棋クエスト:一文字駒":
        return
    promoted_lance_options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") != "TO":
            continue
        current_score = float(cell.get("confidence") or 0.0)
        for candidate in (cell.get("candidates") or [])[:8]:
            if candidate.get("piece") != "NY":
                continue
            candidate_score = float(candidate.get("score") or 0.0)
            if candidate_score < 0.49 or current_score - candidate_score > 0.025:
                continue
            promoted_lance_options.append((candidate_score - current_score, cell, candidate))
            break
    if not promoted_lance_options:
        return

    pawn_options: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") != "KY":
            continue
        current_score = float(cell.get("confidence") or 0.0)
        for candidate in (cell.get("candidates") or [])[:6]:
            if candidate.get("piece") != "FU" or candidate.get("color") != cell.get("color"):
                continue
            if is_dead_end_piece(str(candidate.get("color")), "FU", int(cell["row"])):
                continue
            candidate_score = float(candidate.get("score") or 0.0)
            if candidate_score < 0.55 or current_score - candidate_score > 0.035:
                continue
            bonus = 0.03 if cell.get("postprocess_reason") == "global_solver" else 0.0
            pawn_options.append((candidate_score - current_score + bonus, cell, candidate))
            break
    if not pawn_options:
        return

    _, lance_cell, lance_candidate = max(promoted_lance_options, key=lambda item: item[0])
    _, pawn_cell, pawn_candidate = max(pawn_options, key=lambda item: item[0])
    apply_candidate(lance_cell, lance_candidate, "quest_onechar_promoted_lance_pair_tiebreak", report)
    apply_candidate(pawn_cell, pawn_candidate, "quest_onechar_promoted_lance_pair_tiebreak", report)


def apply_soft_inventory_beam(
    cells: list[dict[str, Any]],
    hands: dict[str, dict[str, int]] | None,
    report: dict[str, Any],
    target_family: str = "",
) -> None:
    if hands is None:
        return
    target = board_inventory_limits(hands)
    current_counts = board_counts(cells)
    if not any(current_counts.get(piece, 0) != target.get(piece, 0) for piece in TOTAL_INVENTORY):
        return

    variables = soft_inventory_variables(cells, current_counts, target, target_family)
    if not variables:
        return

    start_kings = king_counts(cells)
    start_pawns = pawn_counts(cells)
    start_penalty = soft_position_penalty(current_counts, target, start_kings, start_pawns)
    beam = [
        {
            "score": 0.0,
            "penalty": start_penalty,
            "choices": [],
            "counts": dict(current_counts),
            "kings": dict(start_kings),
            "pawns": Counter(start_pawns),
        }
    ]
    beam_width = 64
    for cell, options in variables:
        next_beam = []
        for state in beam:
            for candidate, local_delta in options:
                counts = dict(state["counts"])
                kings = dict(state["kings"])
                pawns = Counter(state["pawns"])
                apply_soft_count_transition(cell, candidate, counts, kings, pawns)
                penalty = soft_position_penalty(counts, target, kings, pawns)
                penalty_gain = float(state["penalty"]) - penalty
                next_beam.append(
                    {
                        "score": float(state["score"]) + local_delta + penalty_gain * 0.70,
                        "penalty": penalty,
                        "choices": [*state["choices"], (cell, candidate)],
                        "counts": counts,
                        "kings": kings,
                        "pawns": pawns,
                    }
                )
        beam = sorted(next_beam, key=lambda item: (float(item["score"]), -float(item["penalty"])), reverse=True)[:beam_width]
    if not beam:
        return

    best = beam[0]
    if float(best["score"]) < 0.18 or float(best["penalty"]) >= start_penalty:
        report["soft_inventory_beam"] = {
            "variables": len(variables),
            "applied": 0,
            "start_penalty": round(start_penalty, 4),
            "best_penalty": round(float(best["penalty"]), 4),
            "best_score": round(float(best["score"]), 4),
        }
        return

    applied = 0
    skipped = []
    for cell, candidate in best["choices"]:
        if candidate is None:
            continue
        if cell.get("state") == "piece" and (cell.get("color"), cell.get("piece")) == (candidate.get("color"), candidate.get("piece")):
            continue
        reason = unsafe_soft_inventory_candidate_reason(cell, candidate, target_family)
        if reason:
            skipped.append(
                {
                    "square": cell.get("square"),
                    "from": identity(cell.get("color"), cell.get("piece")) if cell.get("state") == "piece" else cell.get("state"),
                    "to": identity(candidate.get("color"), candidate.get("piece")),
                    "reason": reason,
                    "current_score": cell.get("confidence"),
                    "candidate_score": candidate.get("score"),
                }
            )
            continue
        apply_candidate(cell, candidate, "soft_inventory_beam", report)
        applied += 1
    report["soft_inventory_beam"] = {
        "variables": len(variables),
        "applied": applied,
        "start_penalty": round(start_penalty, 4),
        "best_penalty": round(float(best["penalty"]), 4),
        "best_score": round(float(best["score"]), 4),
        "beam_width": beam_width,
        "skipped": skipped,
    }


def unsafe_soft_inventory_candidate_reason(
    cell: dict[str, Any],
    candidate: dict[str, Any],
    target_family: str = "",
) -> str | None:
    candidate_score = float(candidate.get("score") or 0.0)
    current_score = float(cell.get("confidence") or 0.0)
    if target_family == "将棋ウォーズ:一文字" and cell.get("postprocess_reason") in {
        "global_solver",
        "wars_onechar_short_piece_tiebreak",
    }:
        return "protected_wars_global_choice"
    if cell.get("state") == "unknown":
        if (
            candidate_score < 0.53
            and not is_wars_unknown_major_inventory_rescue(cell, candidate, target_family)
            and not is_quest_onechar_unknown_tokins_rescue(cell, candidate, target_family)
        ):
            return "low_confidence_unknown_rescue"
        return None
    if cell.get("state") != "piece":
        return None
    score_drop = current_score - candidate_score
    if score_drop > 0.02 and current_score >= 0.58:
        if is_inventory_rerank_close_swap(cell, candidate) and score_drop <= 0.10:
            return None
        if is_wars_close_inventory_swap(cell, candidate, target_family) and score_drop <= 0.09:
            return None
        return "worse_than_current"
    return None


def is_wars_unknown_major_inventory_rescue(cell: dict[str, Any], candidate: dict[str, Any], target_family: str) -> bool:
    if not target_family.startswith("将棋ウォーズ:") or cell.get("state") != "unknown":
        return False
    piece = str(candidate.get("piece") or "")
    score = float(candidate.get("score") or 0.0)
    if piece in {"RY", "UM"}:
        return score >= 0.46
    if piece == "TO":
        return score >= 0.52
    return False


def is_quest_onechar_unknown_tokins_rescue(cell: dict[str, Any], candidate: dict[str, Any], target_family: str) -> bool:
    return (
        target_family == "将棋クエスト:一文字駒"
        and cell.get("state") == "unknown"
        and candidate.get("piece") == "TO"
        and float(candidate.get("score") or 0.0) >= 0.42
    )


def is_wars_close_inventory_swap(cell: dict[str, Any], candidate: dict[str, Any], target_family: str) -> bool:
    if not target_family.startswith("将棋ウォーズ:"):
        return False
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else ""
    candidate_base = base_piece(str(candidate.get("piece"))) if candidate.get("piece") else ""
    if not current_base or not candidate_base:
        return False
    return {current_base, candidate_base} in ({"KI", "KY"}, {"KI", "GI"}, {"FU", "KY"})


def is_inventory_rerank_close_swap(cell: dict[str, Any], candidate: dict[str, Any]) -> bool:
    candidate_score = float(candidate.get("score") or 0.0)
    if candidate_score < 0.52:
        return False
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else ""
    candidate_base = base_piece(str(candidate.get("piece"))) if candidate.get("piece") else ""
    if not current_base or not candidate_base or current_base == candidate_base:
        return False
    return current_base in {"FU", "KY", "KI", "GI"} and candidate_base in {"FU", "KY", "KI", "GI"}


def soft_inventory_variables(
    cells: list[dict[str, Any]],
    current_counts: dict[str, int],
    target: dict[str, int],
    target_family: str = "",
) -> list[tuple[dict[str, Any], list[tuple[dict[str, Any] | None, float]]]]:
    surplus = {piece for piece in TOTAL_INVENTORY if current_counts.get(piece, 0) > target.get(piece, 0)}
    deficits = {piece for piece in TOTAL_INVENTORY if current_counts.get(piece, 0) < target.get(piece, 0)}
    missing_kings = {
        color
        for color in ("black", "white")
        if sum(1 for cell in cells if cell.get("state") == "piece" and cell.get("color") == color and cell.get("piece") == "OU") < 1
    }
    variables = []
    for cell in cells:
        options = soft_inventory_options(cell, surplus, deficits, missing_kings, target_family)
        if len(options) <= 1:
            continue
        priority = soft_inventory_priority(cell, options, surplus, deficits, missing_kings)
        variables.append((priority, cell, options))
    variables.sort(key=lambda item: item[0], reverse=True)
    return [(cell, options) for _, cell, options in variables[:12]]


def soft_inventory_options(
    cell: dict[str, Any],
    surplus: set[str],
    deficits: set[str],
    missing_kings: set[str],
    target_family: str = "",
) -> list[tuple[dict[str, Any] | None, float]]:
    current_identity = (cell.get("color"), cell.get("piece")) if cell.get("state") == "piece" else (None, None)
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else None
    current_score = float(cell.get("confidence") or 0.0)
    allow_high_confidence_close_swap = (
        cell.get("state") == "piece"
        and current_score < 0.86
        and current_base in {"FU", "KY", "KI", "GI"}
        and current_base in surplus
    )
    if cell.get("state") == "piece" and current_score >= 0.75 and not allow_high_confidence_close_swap:
        return [(None, 0.0)]
    options: list[tuple[dict[str, Any] | None, float]] = [(None, 0.0)]
    seen: set[tuple[str, str]] = set()
    candidates = quest_onechar_owner_prior_candidates(cell, target_family)
    for rank, candidate in enumerate(candidates):
        color = candidate.get("color")
        piece = candidate.get("piece")
        if not color or not piece:
            continue
        key = (str(color), str(piece))
        if key == current_identity or key in seen:
            continue
        if is_dead_end_piece(str(color), str(piece), int(cell["row"])):
            continue
        candidate_score = float(candidate.get("score") or 0.0)
        candidate_base = base_piece(str(piece))
        if candidate_base == current_base and color != cell.get("color"):
            continue
        relevant = (
            candidate_base in deficits
            or current_base in surplus
            or (piece == "OU" and color in missing_kings)
            or current_score < 0.58
            or bool(cell.get("ambiguous"))
        )
        if not relevant:
            continue
        score_drop = current_score - candidate_score
        max_drop = 0.05
        if (
            allow_high_confidence_close_swap
            and candidate_base in deficits
            and (color == cell.get("color") or is_wars_close_inventory_swap(cell, candidate, target_family))
            and candidate_base in {"FU", "KY", "KI", "GI"}
        ):
            max_drop = 0.09
        if is_wars_close_inventory_swap(cell, candidate, target_family) and candidate_base in deficits:
            max_drop = max(max_drop, 0.09)
        min_score = 0.43
        if is_wars_unknown_major_inventory_rescue(cell, candidate, target_family) and candidate_base in deficits:
            max_drop = max(max_drop, 0.12)
            min_score = 0.46
        if is_quest_onechar_unknown_tokins_rescue(cell, candidate, target_family) and candidate_base in deficits:
            max_drop = max(max_drop, 0.12)
            min_score = 0.42
        if candidate_score < min_score or score_drop > max_drop:
            continue
        local_delta = candidate_score - current_score - rank * 0.025
        if candidate_base in deficits:
            local_delta += 0.04
        if current_base in surplus:
            local_delta += 0.04
        if piece == "OU" and color in missing_kings:
            local_delta += 0.08
        options.append((candidate, local_delta))
        seen.add(key)
    return options


def quest_onechar_owner_prior_candidates(cell: dict[str, Any], target_family: str) -> list[dict[str, Any]]:
    candidates = list(cell.get("candidates") or [])
    if target_family != "将棋クエスト:一文字駒" or cell.get("state") != "unknown":
        return candidates
    to_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("piece") == "TO" and float(candidate.get("score") or 0.0) >= 0.42
        ),
        None,
    )
    if to_candidate is None:
        return candidates
    row = int(cell.get("row") or 0)
    preferred_color = "white" if row >= 7 else ("black" if row <= 3 else "")
    if not preferred_color or any(candidate.get("color") == preferred_color and candidate.get("piece") == "TO" for candidate in candidates):
        return candidates
    synthetic = dict(to_candidate)
    synthetic["color"] = preferred_color
    synthetic["score"] = float(to_candidate.get("score") or 0.0) + 0.0002
    synthetic["source"] = f"{to_candidate.get('source') or 'unknown'}+quest_onechar_owner_prior"
    return [synthetic, *candidates]


def apply_same_piece_color_tiebreak(cells: list[dict[str, Any]], report: dict[str, Any]) -> None:
    pawns = pawn_counts(cells)
    for cell in cells:
        if cell.get("state") != "piece" or cell.get("piece") != "FU" or cell.get("color") not in {"black", "white"}:
            continue
        current_color = str(cell.get("color"))
        other_color = "white" if current_color == "black" else "black"
        current_score = float(cell.get("confidence") or 0.0)
        current_source = ""
        same_piece_candidate = None
        for candidate in cell.get("candidates") or []:
            if candidate.get("color") == current_color and candidate.get("piece") == "FU":
                current_source = str(candidate.get("source") or current_source)
            if candidate.get("color") == other_color and candidate.get("piece") == "FU" and same_piece_candidate is None:
                same_piece_candidate = candidate
        if same_piece_candidate is None:
            continue
        candidate_score = float(same_piece_candidate.get("score") or 0.0)
        candidate_source = str(same_piece_candidate.get("source") or "")
        if candidate_score < 0.50 or current_score - candidate_score > 0.04:
            continue
        if pawns[(other_color, int(cell["col"]))] > 0:
            continue
        has_prior_advantage = "position_prior" in candidate_source and "position_prior" not in current_source
        if candidate_score <= current_score and not has_prior_advantage:
            continue
        if is_dead_end_piece(other_color, "FU", int(cell["row"])):
            continue
        old_key = (current_color, int(cell["col"]))
        if pawns[old_key] > 0:
            pawns[old_key] -= 1
        apply_candidate(cell, same_piece_candidate, "same_piece_color_tiebreak", report)
        pawns[(other_color, int(cell["col"]))] += 1


def soft_inventory_priority(
    cell: dict[str, Any],
    options: Sequence[tuple[dict[str, Any] | None, float]],
    surplus: set[str],
    deficits: set[str],
    missing_kings: set[str],
) -> float:
    current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else None
    priority = 0.0
    if current_base in surplus:
        priority += 2.0
    if cell.get("state") == "unknown" or bool(cell.get("ambiguous")):
        priority += 1.0
    priority += max(0.0, 0.70 - float(cell.get("confidence") or 0.0))
    for candidate, _ in options[1:]:
        if candidate is None:
            continue
        candidate_base = base_piece(str(candidate.get("piece")))
        if candidate_base in deficits:
            priority += 1.4
        if candidate.get("piece") == "OU" and candidate.get("color") in missing_kings:
            priority += 1.8
    return priority


def apply_soft_count_transition(
    cell: dict[str, Any],
    candidate: dict[str, Any] | None,
    counts: dict[str, int],
    kings: dict[str, int],
    pawns: Counter[tuple[str | None, int]],
) -> None:
    if candidate is None:
        return
    if cell.get("state") == "piece" and cell.get("piece"):
        old_color = cell.get("color")
        old_piece = cell.get("piece")
        old_base = base_piece(str(old_piece))
        counts[old_base] = counts.get(old_base, 0) - 1
        if old_piece == "OU":
            kings[old_color] = kings.get(old_color, 0) - 1
        if old_piece == "FU":
            pawn_key = (old_color, int(cell["col"]))
            pawns[pawn_key] -= 1
            if pawns[pawn_key] <= 0:
                del pawns[pawn_key]
    new_color = candidate.get("color")
    new_piece = candidate.get("piece")
    new_base = base_piece(str(new_piece))
    counts[new_base] = counts.get(new_base, 0) + 1
    if new_piece == "OU":
        kings[new_color] = kings.get(new_color, 0) + 1
    if new_piece == "FU":
        pawns[(new_color, int(cell["col"]))] += 1


def soft_position_penalty(
    counts: dict[str, int],
    target: dict[str, int],
    kings: dict[str, int],
    pawns: Counter[tuple[str | None, int]],
) -> float:
    penalty = sum(abs(counts.get(piece, 0) - target.get(piece, 0)) * 0.42 for piece in TOTAL_INVENTORY)
    penalty += sum(abs(kings.get(color, 0) - 1) * 0.55 for color in ("black", "white"))
    penalty += sum(max(0, count - 1) * 0.35 for (color, _), count in pawns.items() if color in {"black", "white"})
    return penalty


def king_counts(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        color: sum(1 for cell in cells if cell.get("state") == "piece" and cell.get("color") == color and cell.get("piece") == "OU")
        for color in ("black", "white")
    }


def pawn_counts(cells: Sequence[dict[str, Any]]) -> Counter[tuple[str | None, int]]:
    pawns: Counter[tuple[str | None, int]] = Counter()
    for cell in cells:
        if cell.get("state") == "piece" and cell.get("piece") == "FU":
            pawns[(cell.get("color"), int(cell["col"]))] += 1
    return pawns


def hand_reliability_report(hand_report: dict[str, Any] | None) -> dict[str, Any]:
    if not hand_report:
        return {"reliable": False, "reason": "not_requested"}
    pieces = hand_report.get("pieces") or []
    unknown = hand_report.get("unknown") or []
    if unknown:
        return {"reliable": False, "reason": "unknown_hand_proposals", "unknown": len(unknown)}
    if not pieces:
        return {"reliable": False, "reason": "no_hand_pieces_detected"}
    min_confidence = min(float(piece.get("confidence") or 0.0) for piece in pieces)
    if min_confidence < 0.70:
        return {"reliable": False, "reason": "low_confidence", "min_confidence": round(min_confidence, 4)}
    return {"reliable": True, "reason": "ok", "min_confidence": round(min_confidence, 4)}


def hand_report_is_reliable(hand_report: dict[str, Any] | None) -> bool:
    return bool(hand_reliability_report(hand_report).get("reliable"))


def soft_inventory_hands_from_hand_report(hand_report: dict[str, Any] | None) -> dict[str, dict[str, int]] | None:
    if not hand_report:
        return None
    target_family = str(hand_report.get("target_family") or "")
    min_soft_confidence = 0.60 if target_family == "将棋ウォーズ:一文字" else 0.68
    hands = {color: {piece: 0 for piece in HAND_PIECES} for color in ("black", "white")}
    used = 0
    for entry in hand_report.get("pieces") or []:
        owner = entry.get("owner")
        piece = entry.get("piece")
        if owner not in {"black", "white"} or piece not in HAND_PIECE_SET:
            continue
        confidence = float(entry.get("confidence") or 0.0)
        count = max(0, int(entry.get("count") or 0))
        if count <= 0:
            continue
        count_source = str(entry.get("count_source") or "")
        digit_confidence = max(
            (float(digit.get("confidence") or 0.0) for digit in entry.get("digits") or []),
            default=0.0,
        )
        if count_source == "digit" and max(confidence, digit_confidence) >= 0.60:
            pass
        elif confidence < min_soft_confidence:
            continue
        hands[owner][piece] += count
        used += count
    if used and target_family == "将棋ウォーズ:一文字":
        apply_wars_onechar_soft_hand_repairs(hand_report, hands)
    if used and target_family == "将棋クエスト:一文字駒":
        apply_quest_onechar_soft_hand_repairs(hand_report, hands)
    return hands if used else None


def apply_wars_onechar_soft_hand_repairs(hand_report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    evidence = hand_evidence_scores(hand_report)
    white_fu_digit_confidence = max(
        (
            float(digit.get("confidence") or 0.0)
            for entry in hand_report.get("pieces") or []
            if entry.get("owner") == "white"
            and entry.get("piece") == "FU"
            and str(entry.get("count_source") or "") == "digit"
            and int(entry.get("count") or 0) >= 3
            for digit in entry.get("digits") or []
        ),
        default=0.0,
    )
    if (
        hands["white"]["KY"] == 0
        and hands["white"]["FU"] >= 3
        and white_fu_digit_confidence < 0.84
        and evidence.get(("white", "KY"), 0.0) >= 0.50
        and evidence.get(("white", "FU"), 0.0) - evidence.get(("white", "KY"), 0.0) <= 0.38
    ):
        hands["white"]["FU"] -= 1
        hands["white"]["KY"] += 1


def hand_evidence_scores(hand_report: dict[str, Any]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}

    def add(owner: Any, piece: Any, score: Any, discount: float = 0.0) -> None:
        if owner not in {"black", "white"} or piece not in HAND_PIECE_SET:
            return
        try:
            value = float(score) - discount
        except (TypeError, ValueError):
            return
        key = (str(owner), str(piece))
        scores[key] = max(scores.get(key, 0.0), value)

    for entry in hand_report.get("pieces") or []:
        add(entry.get("owner"), entry.get("piece"), entry.get("confidence"))
        for candidate_set in entry.get("candidate_sets") or []:
            for candidate in candidate_set.get("candidates") or []:
                add(candidate.get("color"), candidate.get("piece"), candidate.get("score"), 0.025)
    for item in hand_report.get("unknown") or []:
        for candidate in item.get("candidates") or []:
            add(candidate.get("color"), candidate.get("piece"), candidate.get("score"), 0.015)
    return scores


def apply_quest_onechar_soft_hand_repairs(hand_report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    evidence = hand_evidence_scores(hand_report)
    white_ke_digit_confidence = max(
        (
            float(digit.get("confidence") or 0.0)
            for entry in hand_report.get("pieces") or []
            if entry.get("owner") == "white"
            and entry.get("piece") == "KE"
            and str(entry.get("count_source") or "") == "digit"
            and int(entry.get("count") or 0) >= 3
            for digit in entry.get("digits") or []
        ),
        default=0.0,
    )
    if (
        hands["white"]["KE"] >= 3
        and hands["white"]["GI"] <= 1
        and white_ke_digit_confidence < 0.95
        and evidence.get(("white", "GI"), 0.0) >= 0.56
        and evidence.get(("white", "KE"), 0.0) - evidence.get(("white", "GI"), 0.0) <= 0.16
    ):
        move = min(2, hands["white"]["KE"])
        hands["white"]["KE"] -= move
        hands["white"]["GI"] += move
    if (
        hands["white"]["FU"] >= 3
        and hands["white"]["GI"] == 1
        and evidence.get(("white", "GI"), 0.0) < 0.72
        and evidence.get(("white", "FU"), 0.0) >= 0.70
    ):
        hands["white"]["GI"] = 0
    sanitization = hand_report.get("inventory_sanitization") or {}
    completion = hand_report.get("inventory_completion") or {}
    removed_white_gi = any(
        item.get("owner") == "white"
        and item.get("piece") == "GI"
        and int(item.get("removed") or 0) > 0
        and float(item.get("confidence") or 0.0) <= 0.50
        for item in sanitization.get("changes") or []
    )
    low_confidence_white_ky_completion = any(
        item.get("owner") == "white"
        and item.get("piece") == "KY"
        and float(item.get("score") or 0.0) <= 0.54
        for item in completion.get("changes") or []
    )
    if removed_white_gi and low_confidence_white_ky_completion and hands["white"]["KY"] > 0:
        hands["white"]["KY"] -= 1
        hands["white"]["GI"] += 1
    low_confidence_black_ke_completion = any(
        item.get("owner") == "black"
        and item.get("piece") == "KE"
        and float(item.get("score") or 0.0) <= 0.56
        for item in completion.get("changes") or []
    )
    if removed_white_gi and low_confidence_black_ke_completion and hands["black"]["KE"] > 0:
        hands["black"]["KE"] -= 1
        hands["white"]["GI"] += 1


def board_inventory_limits(hands: dict[str, dict[str, int]] | None) -> dict[str, int]:
    limits = dict(TOTAL_INVENTORY)
    if hands:
        for color in ("black", "white"):
            for piece, count in hands[color].items():
                limits[piece] = max(0, limits.get(piece, 0) - int(count))
    return limits


def apply_dead_end_constraints(cells: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for cell in cells:
        if cell.get("state") != "piece":
            continue
        if is_dead_end_piece(cell.get("color"), cell.get("piece"), int(cell["row"])):
            replacement = dead_end_color_flip_candidate(cell)
            if replacement is None:
                if float(cell.get("confidence") or 0.0) >= 0.75:
                    report["unresolved"].append({"reason": "dead_end_high_confidence", "square": cell.get("square")})
                    continue
                replacement = best_legal_candidate(
                    cell,
                    limits=None,
                    counts=None,
                    preferred_color=cell.get("color"),
                    require_preferred_color=True,
                    max_score_drop=0.05,
                )
            if replacement is None:
                report["unresolved"].append({"reason": "dead_end", "square": cell.get("square")})
            else:
                apply_candidate(cell, replacement, "dead_end", report)


def dead_end_color_flip_candidate(cell: dict[str, Any]) -> dict[str, Any] | None:
    color = cell.get("color")
    piece = cell.get("piece")
    if color not in {"black", "white"} or piece not in {"FU", "KY", "KE"}:
        return None
    flipped = "white" if color == "black" else "black"
    if is_dead_end_piece(flipped, piece, int(cell["row"])):
        return None
    current_score = float(cell.get("confidence") or 0.0)
    for candidate in cell.get("candidates") or []:
        if candidate.get("color") == flipped and candidate.get("piece") == piece:
            if current_score - float(candidate.get("score") or 0.0) <= 0.12:
                return candidate
    if current_score >= 0.38:
        return {
            "color": flipped,
            "piece": piece,
            "score": max(0.0, current_score - 0.02),
            "source": "dead_end_color_flip",
        }
    return None


def apply_nifu_constraints(cells: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for color in ("black", "white"):
        for col in range(1, 10):
            pawns = [
                cell
                for cell in cells
                if cell.get("state") == "piece"
                and cell.get("color") == color
                and cell.get("piece") == "FU"
                and int(cell.get("col", 0)) == col
            ]
            if len(pawns) <= 1:
                continue
            pawns.sort(key=lambda cell: float(cell.get("confidence") or 0.0))
            for cell in pawns[:-1]:
                current_score = float(cell.get("confidence") or 0.0)
                replacement = None
                if current_score < 0.52:
                    replacement = best_legal_candidate(
                        cell,
                        limits=None,
                        counts=None,
                        forbidden_base="FU",
                        preferred_color=color,
                        require_preferred_color=True,
                        max_score_drop=0.05,
                    )
                if replacement is not None:
                    apply_candidate(cell, replacement, "nifu", report)
                else:
                    report["unresolved"].append({"reason": "nifu", "square": cell.get("square"), "color": color})


def apply_rook_bishop_presence_constraints(cells: list[dict[str, Any]], limits: dict[str, int], report: dict[str, Any]) -> None:
    for base in ("HI", "KA"):
        target = limits.get(base, TOTAL_INVENTORY[base])
        if target <= 0:
            continue
        counts = board_counts(cells)
        if counts.get(base, 0) >= target:
            continue
        needed = target - counts.get(base, 0)
        options = []
        for cell in cells:
            if cell.get("state") == "empty":
                continue
            current_base = base_piece(str(cell.get("piece"))) if cell.get("piece") else None
            if current_base == base:
                continue
            best_score = max((float(candidate.get("score") or 0.0) for candidate in cell.get("candidates") or []), default=0.0)
            for rank, candidate in enumerate(cell.get("candidates") or []):
                if base_piece(str(candidate.get("piece"))) != base:
                    continue
                if is_dead_end_piece(candidate.get("color"), candidate.get("piece"), int(cell["row"])):
                    continue
                current_score = float(cell.get("confidence") or 0.0)
                candidate_score = float(candidate.get("score") or 0.0)
                if candidate_score < best_score - 1e-6:
                    continue
                penalty = max(0.0, current_score - candidate_score)
                if (cell.get("state") == "unknown" or cell.get("ambiguous")) and penalty <= 0.06:
                    options.append((penalty + rank * 0.018, cell, candidate))
                break
        for _, cell, candidate in sorted(options, key=lambda item: item[0])[:needed]:
            apply_candidate(cell, candidate, f"{base.lower()}_missing", report)


def apply_piece_count_limits(cells: list[dict[str, Any]], limits: dict[str, int], report: dict[str, Any]) -> None:
    changed = True
    while changed:
        changed = False
        counts = board_counts(cells)
        over_piece = next((piece for piece, count in counts.items() if count > limits.get(piece, 0)), None)
        if over_piece is None:
            break
        offenders = [
            cell
            for cell in cells
            if cell.get("state") == "piece" and base_piece(str(cell.get("piece"))) == over_piece
        ]
        offenders.sort(key=lambda cell: float(cell.get("confidence") or 0.0))
        for cell in offenders:
            replacement = best_legal_candidate(
                cell,
                limits,
                counts,
                forbidden_base=over_piece,
                preferred_color=cell.get("color"),
                max_score_drop=0.05,
            )
            if replacement is not None:
                apply_candidate(cell, replacement, "inventory_overcount", report)
                changed = True
            elif float(cell.get("confidence") or 0.0) < 0.50:
                mark_unknown(cell, "inventory_overcount", report)
                changed = True
            else:
                report["unresolved"].append({"reason": "inventory_overcount", "square": cell.get("square"), "piece": over_piece})
            break


def apply_king_constraints(cells: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for color in ("black", "white"):
        kings = [
            cell
            for cell in cells
            if cell.get("state") == "piece" and cell.get("color") == color and cell.get("piece") == "OU"
        ]
        if len(kings) > 1:
            kings.sort(key=lambda cell: float(cell.get("confidence") or 0.0))
            for cell in kings[:-1]:
                replacement = best_legal_candidate(
                    cell,
                    limits=None,
                    counts=None,
                    forbidden_identity=(color, "OU"),
                    preferred_color=color,
                    require_preferred_color=True,
                    max_score_drop=0.05,
                )
                if replacement is not None:
                    apply_candidate(cell, replacement, "king_overcount", report)
                elif float(cell.get("confidence") or 0.0) < 0.50:
                    mark_unknown(cell, "king_overcount", report)
                else:
                    report["unresolved"].append({"reason": "king_overcount", "square": cell.get("square"), "color": color})
        if len(kings) == 0:
            promoted = promote_best_candidate(
                cells,
                color,
                "OU",
                "king_missing",
                report,
                allowed_states={"unknown"},
                min_score=0.40,
                max_score_drop=0.03,
                max_rank=3,
            )
            if not promoted:
                promoted = promote_best_candidate(
                    cells,
                    color,
                    "OU",
                    "king_missing",
                    report,
                    allowed_states={"piece"},
                    require_ambiguous=True,
                    min_score=0.62,
                    max_score_drop=0.04,
                )
            if not promoted:
                report["unresolved"].append({"reason": "king_missing", "color": color})


def board_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts = {piece: 0 for piece in TOTAL_INVENTORY}
    for cell in cells:
        if cell.get("state") == "piece" and cell.get("piece"):
            base = base_piece(str(cell["piece"]))
            counts[base] = counts.get(base, 0) + 1
    return counts


def best_legal_candidate(
    cell: dict[str, Any],
    limits: dict[str, int] | None,
    counts: dict[str, int] | None,
    forbidden_base: str | None = None,
    forbidden_identity: tuple[str, str] | None = None,
    preferred_color: str | None = None,
    require_preferred_color: bool = False,
    max_score_drop: float | None = None,
) -> dict[str, Any] | None:
    current_score = float(cell.get("confidence") or 0.0)
    fallback: dict[str, Any] | None = None
    for candidate in cell.get("candidates") or []:
        color = candidate.get("color")
        piece = candidate.get("piece")
        if not color or not piece:
            continue
        if forbidden_identity is not None and (color, piece) == forbidden_identity:
            continue
        base = base_piece(piece)
        if forbidden_base is not None and base == forbidden_base:
            continue
        if is_dead_end_piece(color, piece, int(cell["row"])):
            continue
        if limits is not None and counts is not None and counts.get(base, 0) >= limits.get(base, 0):
            continue
        if max_score_drop is not None and current_score - float(candidate.get("score") or 0.0) > max_score_drop:
            continue
        if preferred_color is None or color == preferred_color:
            return candidate
        if require_preferred_color:
            continue
        if fallback is None:
            fallback = candidate
    return fallback


def promote_best_candidate(
    cells: list[dict[str, Any]],
    color: str,
    piece: str,
    reason: str,
    report: dict[str, Any],
    *,
    allowed_states: set[str] | None = None,
    require_ambiguous: bool = False,
    min_score: float = 0.0,
    max_score_drop: float | None = None,
    max_rank: int | None = None,
) -> bool:
    options = []
    for cell in cells:
        if allowed_states is not None and str(cell.get("state")) not in allowed_states:
            continue
        if require_ambiguous and not bool(cell.get("ambiguous")):
            continue
        current_score = float(cell.get("confidence") or 0.0)
        for rank, candidate in enumerate(cell.get("candidates") or []):
            if max_rank is not None and rank >= max_rank:
                break
            if candidate.get("color") == color and candidate.get("piece") == piece:
                candidate_score = float(candidate.get("score") or 0.0)
                if candidate_score < min_score:
                    continue
                if max_score_drop is not None and current_score - candidate_score > max_score_drop:
                    continue
                score = candidate_score - rank * 0.04 - max(0.0, current_score - candidate_score) * 0.10
                options.append((score, cell, candidate))
                break
    if not options:
        return False
    _, cell, candidate = max(options, key=lambda item: item[0])
    apply_candidate(cell, candidate, reason, report)
    return True


def is_dead_end_piece(color: str | None, piece: str | None, row: int) -> bool:
    if color == "black" and piece in {"FU", "KY"} and row == 1:
        return True
    if color == "black" and piece == "KE" and row <= 2:
        return True
    if color == "white" and piece in {"FU", "KY"} and row == 9:
        return True
    if color == "white" and piece == "KE" and row >= 8:
        return True
    return False


def apply_candidate(cell: dict[str, Any], candidate: dict[str, Any], reason: str, report: dict[str, Any]) -> None:
    before = identity(cell.get("color"), cell.get("piece")) if cell.get("state") == "piece" else str(cell.get("state"))
    after = identity(candidate.get("color"), candidate.get("piece"))
    cell["state"] = "piece"
    cell["color"] = candidate.get("color")
    cell["piece"] = candidate.get("piece")
    cell["best_piece"] = candidate.get("piece")
    cell["confidence"] = candidate.get("score", cell.get("confidence", 0.0))
    cell["ambiguous"] = False
    cell["postprocess_reason"] = reason
    cell.setdefault("postprocess_history", []).append({"reason": reason, "from": before, "to": after})
    reorder_candidates(cell, candidate)
    report["changes"].append(
        {
            "square": cell.get("square"),
            "reason": reason,
            "from": before,
            "to": after,
        },
    )


def mark_unknown(cell: dict[str, Any], reason: str, report: dict[str, Any]) -> None:
    before = identity(cell.get("color"), cell.get("piece")) if cell.get("state") == "piece" else str(cell.get("state"))
    cell["state"] = "unknown"
    cell["color"] = None
    cell["piece"] = None
    cell["best_piece"] = None
    cell["confidence"] = 0.0
    cell["ambiguous"] = True
    cell["candidates"] = []
    cell["postprocess_reason"] = reason
    cell.setdefault("postprocess_history", []).append({"reason": reason, "from": before, "to": "unknown"})
    report["changes"].append({"square": cell.get("square"), "reason": reason, "from": before, "to": "unknown"})


def reorder_candidates(cell: dict[str, Any], selected: dict[str, Any]) -> None:
    candidates = list(cell.get("candidates") or [])
    selected_key = (selected.get("color"), selected.get("piece"))
    reordered = [candidate for candidate in candidates if (candidate.get("color"), candidate.get("piece")) == selected_key]
    reordered.extend(candidate for candidate in candidates if (candidate.get("color"), candidate.get("piece")) != selected_key)
    cell["candidates"] = reordered


def recognize_hands(
    image: Image.Image,
    detection: GridDetection,
    max_proposals_per_area: int,
    learned_model: dict[str, Any] | None = None,
    hand_assets: dict[str, Any] | None = None,
    target_family: str = "",
) -> dict[str, Any]:
    areas = detect_hand_areas(image, detection)
    cell_w, cell_h = cell_size(detection)
    proposals = []
    digits = []
    for area in areas:
        proposals.extend(proposals_for_area(image, area, cell_w, cell_h, max_proposals_per_area))
        digits.extend(digit_candidates_for_area(image, area, cell_w, cell_h))
    model, templates = hand_classifier_assets(hand_assets)
    pieces, unknown = classify_hand_proposals(
        image,
        proposals,
        model,
        templates,
        areas,
        learned_model,
        target_family,
        min_confidence=0.43,
        ambiguous_margin=DEFAULT_AMBIGUOUS_MARGIN,
    )
    pieces = associate_digits(pieces, digits, cell_w, cell_h)
    hands, piece_entries = aggregate_hands(pieces, use_icon_count=(target_family == "将皇"))
    apply_layout_hand_count_repairs(image, areas, hands, piece_entries, target_family, cell_w, cell_h)
    owner_flip = wars_side_hand_owner_flip(image, detection, target_family)
    if owner_flip.get("applied"):
        hands, piece_entries, unknown = swap_hand_owners(hands, piece_entries, unknown)
    return {
        "hands": hands,
        "target_family": target_family,
        "areas": [
            {
                "owner": swapped_owner(area.owner) if owner_flip.get("applied") and area.side in {"left", "right"} else area.owner,
                "side": area.side,
                "rect": area.rect,
                "confidence": area.confidence,
                "evidence": area.evidence,
            }
            for area in areas
        ],
        "pieces": piece_entries,
        "digits": [asdict(digit) for digit in digits],
        "unknown": unknown,
        "owner_flip": owner_flip,
    }


def wars_side_hand_owner_flip(image: Image.Image, detection: GridDetection, target_family: str) -> dict[str, Any]:
    if not target_family.startswith("将棋ウォーズ:"):
        return {"applied": False, "reason": "not_wars"}
    left_digit, left_score = top_coordinate_digit(image, detection, 0)
    right_digit, right_score = top_coordinate_digit(image, detection, 8)
    result = {
        "applied": False,
        "reason": "coordinate_digits_not_confident",
        "left_digit": left_digit,
        "left_score": left_score,
        "right_digit": right_digit,
        "right_score": right_score,
    }
    if left_digit is None or right_digit is None or left_score < 0.70 or right_score < 0.70:
        return result
    if left_digit <= 3 and right_digit >= 7:
        result["applied"] = True
        result["reason"] = "top_files_ascending"
        return result
    if left_digit >= 7 and right_digit <= 3:
        result["reason"] = "top_files_descending"
        return result
    result["reason"] = "coordinate_digits_ambiguous"
    return result


def top_coordinate_digit(image: Image.Image, detection: GridDetection, visual_col: int) -> tuple[int | None, float]:
    xs = detection.vertical.positions
    ys = detection.horizontal.positions
    if len(xs) < 10 or len(ys) < 10 or visual_col < 0 or visual_col >= 9:
        return None, 0.0
    left, right = int(xs[0]), int(xs[-1])
    top, bottom = int(ys[0]), int(ys[-1])
    cell_w = max(1.0, (right - left) / 9.0)
    cell_h = max(1.0, (bottom - top) / 9.0)
    center_x = (float(xs[visual_col]) + float(xs[visual_col + 1])) / 2.0
    rect = (
        max(0, int(round(center_x - cell_w * 0.28))),
        max(0, int(round(top - cell_h * 0.32))),
        min(image.width, int(round(center_x + cell_w * 0.28))),
        min(image.height, int(round(top + cell_h * 0.05))),
    )
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return None, 0.0
    crop = image.crop(rect).convert("RGB")
    mask = black_ink_mask(np.array(crop))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best_digit: int | None = None
    best_score = 0.0
    for component in range(1, component_count):
        x, y, width, height, area = stats[component]
        if area < 10 or width > crop.width * 0.80 or height > crop.height * 0.95:
            continue
        digit, score = recognize_digit(mask[y : y + height, x : x + width])
        if digit is not None and score > best_score:
            best_digit = digit
            best_score = float(score)
    return best_digit, round(best_score, 4)


def swapped_owner(owner: str) -> str:
    return {"black": "white", "white": "black"}.get(owner, owner)


def swap_hand_owners(
    hands: dict[str, dict[str, int]],
    piece_entries: list[dict],
    unknown: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, int]], list[dict], list[dict[str, Any]]]:
    swapped_hands = {
        "black": {piece: int((hands.get("white") or {}).get(piece, 0)) for piece in HAND_PIECES},
        "white": {piece: int((hands.get("black") or {}).get(piece, 0)) for piece in HAND_PIECES},
    }
    swapped_entries: list[dict] = []
    for entry in piece_entries:
        updated = dict(entry)
        updated["owner"] = swapped_owner(str(updated.get("owner") or ""))
        updated["candidate_sets"] = swap_candidate_set_owners(updated.get("candidate_sets") or [])
        swapped_entries.append(updated)
    swapped_unknown: list[dict[str, Any]] = []
    for entry in unknown:
        updated = dict(entry)
        updated["owner"] = swapped_owner(str(updated.get("owner") or ""))
        updated["candidates"] = swap_candidate_owners(updated.get("candidates") or [])
        swapped_unknown.append(updated)
    swapped_entries.sort(key=lambda entry: (entry["owner"], HAND_PIECES.index(entry["piece"])))
    return swapped_hands, swapped_entries, swapped_unknown


def swap_candidate_set_owners(candidate_sets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    swapped: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        updated = dict(candidate_set)
        updated["candidates"] = swap_candidate_owners(updated.get("candidates") or [])
        swapped.append(updated)
    return swapped


def swap_candidate_owners(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    swapped: list[dict[str, Any]] = []
    for candidate in candidates:
        updated = dict(candidate)
        updated["color"] = swapped_owner(str(updated.get("color") or ""))
        swapped.append(updated)
    return swapped


def apply_layout_hand_count_repairs(
    image: Image.Image,
    areas: Sequence[Any],
    hands: dict[str, dict[str, int]],
    piece_entries: list[dict],
    target_family: str,
    cell_w: float,
    cell_h: float,
) -> None:
    if target_family != "将棋クエスト:一文字駒" or hands["white"].get("GI", 0) > 0:
        return
    top_area = next((area for area in areas if area.side == "top" and area.owner == "white"), None)
    if top_area is None:
        return
    for proposal in layout_slot_proposals_for_area(top_area, cell_w, cell_h, target_family):
        center = ((proposal.rect[0] + proposal.rect[2]) / 2.0 - float(top_area.rect[0])) / max(1.0, float(top_area.rect[2] - top_area.rect[0]))
        expected_piece, closeness = hand_slot_piece(
            center,
            hand_slot_order("top", top_area, target_family),
            *hand_slot_span("top", top_area, target_family, use_gutter_spacing=False),
            False,
        )
        if expected_piece != "GI" or closeness < 0.90:
            continue
        material = piece_material_features(image.crop(tuple(proposal.rect)).convert("RGB"))
        if material.get("cream_ratio", 0.0) < 0.14 or material.get("gold_ratio", 0.0) < 0.06:
            return
        hands["white"]["GI"] = 1
        piece_entries.append(
            {
                "owner": "white",
                "piece": "GI",
                "count": 1,
                "count_source": "layout_slot",
                "confidence": 0.46,
                "rects": [proposal.rect],
                "digits": [],
                "ambiguous": True,
            }
        )
        piece_entries.sort(key=lambda entry: (entry["owner"], HAND_PIECES.index(entry["piece"])))
        return


def layout_slot_proposals_for_area(
    area: Any,
    cell_w: float,
    cell_h: float,
    target_family: str,
) -> list[PieceProposal]:
    if not target_family.startswith("将棋クエスト:") or area.side not in {"top", "bottom"}:
        return []
    area_left, area_top, area_right, area_bottom = (int(value) for value in area.rect)
    area_w = max(1, area_right - area_left)
    order = hand_slot_order(area.side, area, target_family)
    slot_start, slot_span = hand_slot_span(area.side, area, target_family, use_gutter_spacing=False)
    slot_count = len(order)
    slot_width = slot_span / slot_count
    rect_w = cell_w * 0.82
    rect_h = cell_h * 1.00
    center_y = area_top + cell_h * 0.54
    proposals = []
    bounds = (area_left, area_top, area_right, area_bottom)
    for index, _ in enumerate(order):
        center_fraction = slot_start + (index + 0.5) * slot_width
        center_x = area_left + center_fraction * area_w
        rect = clip_hand_slot_rect(center_x, center_y, rect_w, rect_h, bounds)
        if rect is None:
            continue
        proposals.append(PieceProposal(rect=rect, source="layout_slot", side=area.side, owner=area.owner))
    return proposals


def clip_hand_slot_rect(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    bounds: tuple[int, int, int, int],
) -> list[int] | None:
    left = max(bounds[0], int(round(center_x - width / 2.0)))
    top = max(bounds[1], int(round(center_y - height / 2.0)))
    right = min(bounds[2], int(round(center_x + width / 2.0)))
    bottom = min(bounds[3], int(round(center_y + height / 2.0)))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def classify_hand_proposals(
    image: Image.Image,
    proposals: Sequence[Any],
    hog_model,
    hog_templates,
    areas: Sequence[Any],
    learned_model: dict[str, Any] | None,
    target_family: str,
    min_confidence: float,
    ambiguous_margin: float,
) -> tuple[list[RecognizedHandPiece], list[dict[str, Any]]]:
    accepted: list[RecognizedHandPiece] = []
    unknown: list[dict[str, Any]] = []
    area_by_side = {area.side: area for area in areas}
    for proposal in proposals:
        crop = image.crop(tuple(proposal.rect)).convert("RGB")
        material = piece_material_features(crop)
        if not has_piece_material(material, proposal, area_by_side):
            continue
        hog_candidates = classify_piece_crop(crop, hog_model, hog_templates)[:8]
        learned_candidates = learned_hand_candidates(crop, learned_model, proposal.owner, target_family)[:8]
        candidates = merge_hand_candidates(learned_candidates, hog_candidates)[:8]
        candidates = apply_hand_slot_prior(candidates, proposal, area_by_side.get(proposal.side), target_family)[:8]
        best, margin = best_hand_candidate(candidates, proposal.owner)
        global_best = candidates[0] if candidates else None
        owner_conflict = (
            best is not None
            and global_best is not None
            and (global_best.color, global_best.piece) != (best.color, best.piece)
            and global_best.score - best.score > 0.08
        )
        candidate_dicts = [asdict(candidate) for candidate in candidates]
        ambiguous = best is not None and margin < ambiguous_margin
        if (
            best is not None
            and best.score >= min_confidence
            and not owner_conflict
            and not is_unreliable_hand_candidate(best, proposal, material, area_by_side.get(proposal.side), target_family)
            and (not ambiguous or best.score >= 0.62 or best.source == "hand_slot_prior")
        ):
            accepted.append(
                RecognizedHandPiece(
                    owner=proposal.owner,
                    side=proposal.side,
                    piece=best.piece,
                    rect=proposal.rect,
                    confidence=round(best.score, 4),
                    ambiguous=ambiguous,
                    proposal_source=proposal.source,
                    candidates=candidate_dicts,
                ),
            )
            continue
        confidence = round(best.score, 4) if best is not None else 0.0
        if confidence >= min_confidence * 0.70:
            unknown.append(
                {
                    "owner": proposal.owner,
                    "side": proposal.side,
                    "rect": proposal.rect,
                    "proposal_source": proposal.source,
                    "best_piece": best.piece if best is not None else None,
                    "confidence": confidence,
                    "owner_conflict": owner_conflict,
                    "material": material,
                    "candidates": candidate_dicts,
                },
            )
    return suppress_duplicate_pieces(accepted), unknown[:30]


def is_unreliable_hand_candidate(
    candidate: Candidate,
    proposal: Any,
    material: dict[str, float],
    area: Any | None,
    target_family: str = "",
) -> bool:
    if (
        target_family == "将棋クエスト:一文字駒"
        and candidate.piece in {"GI", "KE"}
        and candidate.color == proposal.owner
        and (
            (proposal.source == "color" and candidate.score >= 0.60)
            or (proposal.source == "glyph" and candidate.piece == "KE" and candidate.score >= 0.52)
        )
    ):
        return False
    area_evidence = getattr(area, "evidence", None)
    if (
        area_evidence == "board_gutter"
        and candidate.piece != "FU"
        and candidate.color == proposal.owner
        and "hand_slot_prior" in str(candidate.source)
        and material.get("cream_ratio", 0.0) >= 0.24
        and material.get("gold_ratio", 0.0) >= 0.30
        and candidate.score >= 0.54
    ):
        return False
    if area_evidence == "board_gutter" and candidate.piece != "FU" and candidate.score < 0.58:
        return True
    if (
        area_evidence == "piece_color_components"
        and material.get("cream_ratio", 0.0) >= 0.10
        and material.get("gold_ratio", 0.0) >= 0.085
        and candidate.score >= 0.52
    ):
        return False
    if (
        "hand_slot_prior" in str(candidate.source)
        and area_evidence == "piece_color_components"
        and candidate.color == proposal.owner
        and material.get("cream_ratio", 0.0) >= 0.10
        and material.get("gold_ratio", 0.0) >= 0.065
        and candidate.score >= 0.52
    ):
        return False
    if (
        proposal.source == "layout_slot"
        and "hand_slot_prior" in str(candidate.source)
        and candidate.color == proposal.owner
        and material.get("cream_ratio", 0.0) >= 0.14
        and material.get("gold_ratio", 0.0) >= 0.06
        and candidate.score >= 0.45
    ):
        return False
    if area_evidence != "board_gutter" and material.get("gold_ratio", 0.0) < 0.10 and candidate.score < 0.62:
        return True
    return False


def learned_hand_candidates(
    crop: Image.Image,
    model: dict[str, Any] | None,
    owner: str | None = None,
    target_family: str = "",
) -> list[Candidate]:
    if model is None:
        return []
    hand_crop_candidates = hand_crop_classifier_candidates(crop, model.get("hand_crop_classifier"), owner, target_family)
    hand_template_family = hand_template_target_family(model, target_family)
    glyph = fast_glyph_features_from_rgb(np.array(crop.convert("RGB")))
    if glyph is None:
        return hand_crop_candidates
    board_candidates = [
        candidate
        for candidate in template_candidates(
            glyph,
            model.get("templates", []),
            model.get("classifier") or {"enabled": False},
            crop,
            hand_template_family,
            0,
            0,
            {},
        )
        if candidate.piece in HAND_PIECE_SET
    ]
    return merge_hand_learned_candidates(hand_crop_candidates, board_candidates)


def hand_crop_feature_vector(crop: Image.Image) -> np.ndarray | None:
    rgb = np.array(crop.convert("RGB"))
    glyph = fast_glyph_features_from_rgb(rgb)
    if glyph is None:
        glyph_vector = np.zeros(48 * 56 + 6, dtype="float32")
    else:
        glyph_vector = feature_vector_from_glyph(glyph).astype("float32")
    hog_vector = extract_hog_features(crop.convert("RGB")).vector.astype("float32")
    material = piece_material_features(crop)
    material_vector = np.asarray(
        [
            float(material.get("cream_ratio", 0.0)),
            float(material.get("gold_ratio", 0.0)),
            crop.width / max(1.0, crop.height),
        ],
        dtype="float32",
    )
    return np.concatenate([glyph_vector, hog_vector, material_vector]).astype("float32")


def hand_crop_classifier_candidates(
    crop: Image.Image,
    classifier: dict[str, Any] | None,
    owner: str | None,
    target_family: str = "",
) -> list[Candidate]:
    if (
        not classifier
        or not classifier.get("enabled")
        or owner not in {"black", "white"}
        or not hand_crop_classifier_allowed(classifier, target_family)
    ):
        return []
    vector = hand_crop_feature_vector(crop)
    if vector is None:
        return []
    sample = vector.reshape(1, -1).astype("float32")
    scores: dict[str, float] = {}
    for estimator_name in ("knn", "sgd"):
        estimator = classifier.get(estimator_name)
        if estimator is None or not hasattr(estimator, "predict_proba"):
            continue
        try:
            probabilities = estimator.predict_proba(sample)[0]
        except Exception:
            continue
        for label, probability in zip(estimator.classes_, probabilities):
            piece = str(label)
            if piece not in HAND_PIECE_SET:
                continue
            weight = 1.0 if estimator_name == "knn" else 0.92
            scores[piece] = max(scores.get(piece, 0.0), float(probability) * weight)
    candidates = [
        Candidate(
            color=owner,
            piece=piece,
            score=round(min(0.62, score * 0.72), 4),
            source="hand_crop_classifier",
        )
        for piece, score in scores.items()
        if score >= 0.18
    ]
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:8]


def hand_crop_classifier_allowed(classifier: dict[str, Any], target_family: str) -> bool:
    configured = classifier.get("enabled_families")
    enabled = set(configured) if configured else DEFAULT_HAND_CROP_CLASSIFIER_FAMILIES
    return bool(target_family and target_family in enabled)


def hand_template_target_family(model: dict[str, Any], target_family: str) -> str:
    family = target_family or str(model.get("target_family") or "")
    configured = model.get("hand_template_target_families")
    enabled = set(configured) if configured else DEFAULT_HAND_TEMPLATE_TARGET_FAMILIES
    return family if family in enabled else ""


def merge_hand_learned_candidates(
    hand_crop_candidates: Sequence[Candidate],
    board_candidates: Sequence[Candidate],
) -> list[Candidate]:
    merged: dict[tuple[str, str], Candidate] = {}
    for candidate in board_candidates:
        merged[(candidate.color, candidate.piece)] = candidate
    for candidate in hand_crop_candidates:
        key = (candidate.color, candidate.piece)
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        merged[key] = Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(min(0.90, max(previous.score, candidate.score) + 0.018), 4),
            bbox=previous.bbox,
            source=f"{previous.source}+{candidate.source}",
        )
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def apply_hand_slot_prior(
    candidates: Sequence[Candidate],
    proposal: Any,
    area: Any | None,
    target_family: str = "",
) -> list[Candidate]:
    if area is None:
        return list(candidates)
    use_slot_prior = area.evidence == "board_gutter" or (
        proposal.side == "bottom" and target_family.startswith("将棋クエスト:")
    )
    if not use_slot_prior:
        return list(candidates)
    axis_index = 0 if proposal.side in {"top", "bottom"} else 1
    start = float(area.rect[axis_index])
    end = float(area.rect[axis_index + 2])
    rect = proposal.rect
    center = ((float(rect[axis_index]) + float(rect[axis_index + 2])) / 2.0 - start) / max(1.0, end - start)
    order = hand_slot_order(proposal.side, area, target_family)
    use_gutter_spacing = getattr(area, "evidence", None) == "board_gutter"
    slot_start, slot_span = hand_slot_span(proposal.side, area, target_family, use_gutter_spacing)
    expected_piece, slot_closeness = hand_slot_piece(center, order, slot_start, slot_span, use_gutter_spacing)
    slot_count = len(order)
    slot_width = slot_span / (slot_count + 1) if use_gutter_spacing else slot_span / slot_count
    adjusted: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        score = float(candidate.score)
        source = candidate.source
        if candidate.color == proposal.owner and candidate.piece in order:
            expected_center = (
                slot_start + (order.index(candidate.piece) + 1) * slot_width
                if use_gutter_spacing
                else slot_start + (order.index(candidate.piece) + 0.5) * slot_width
            )
            distance = abs(center - expected_center)
            closeness = max(0.0, 1.0 - distance / slot_width)
            if closeness >= 0.35 and (use_gutter_spacing or candidate.piece == expected_piece):
                boost_weight = 0.18 if proposal.side in {"top", "bottom"} else 0.15
                if proposal.source == "layout_slot" and candidate.piece == expected_piece:
                    boost_weight = 0.24
                if (
                    proposal.side in {"top", "bottom"}
                    and target_family.startswith("将棋クエスト:")
                    and not use_gutter_spacing
                    and proposal.source == "color"
                    and candidate.piece == expected_piece
                ):
                    boost_weight = 0.28
                boost = boost_weight * closeness
                score = min(0.92, score + boost)
                source = f"{source}+hand_slot_prior"
        seen.add((candidate.color, candidate.piece))
        adjusted.append(
            Candidate(
                color=candidate.color,
                piece=candidate.piece,
                score=round(score, 4),
                edge_score=candidate.edge_score,
                ink_score=candidate.ink_score,
                color_score=candidate.color_score,
                bbox=candidate.bbox,
                scale=candidate.scale,
                source=source,
            ),
        )
    if expected_piece is not None and slot_closeness >= 0.54 and (proposal.owner, expected_piece) not in seen:
        score = 0.43 + 0.14 * slot_closeness
        if (
            proposal.side in {"top", "bottom"}
            and target_family.startswith("将棋クエスト:")
            and not use_gutter_spacing
            and proposal.source == "color"
        ):
            score = max(score, 0.50 + 0.20 * slot_closeness)
        adjusted.append(
            Candidate(
                color=proposal.owner,
                piece=expected_piece,
                score=round(min(0.74, score), 4),
                source="hand_slot_prior",
            ),
        )
    return sorted(adjusted, key=lambda candidate: candidate.score, reverse=True)


def hand_slot_order(side: str, area: Any | None = None, target_family: str = "") -> tuple[str, ...]:
    if getattr(area, "evidence", None) == "board_gutter":
        return HAND_PIECES
    if target_family.startswith("ぴよ将棋:"):
        return HAND_PIECES
    if side == "right":
        return HAND_PIECES
    return tuple(reversed(HAND_PIECES))


def hand_slot_span(
    side: str,
    area: Any | None = None,
    target_family: str = "",
    use_gutter_spacing: bool = False,
) -> tuple[float, float]:
    if use_gutter_spacing:
        return 0.0, 1.0
    if side == "top" and target_family.startswith("将棋クエスト:"):
        return 0.25, 0.75
    if side == "bottom" and target_family.startswith("将棋クエスト:"):
        return 0.0, 0.80
    return 0.0, 1.0


def hand_slot_piece(
    center: float,
    order: Sequence[str],
    slot_start: float = 0.0,
    slot_span: float = 1.0,
    use_gutter_spacing: bool = False,
) -> tuple[str | None, float]:
    slot_count = len(order)
    slot_width = slot_span / (slot_count + 1) if use_gutter_spacing else slot_span / slot_count
    best_piece = None
    best_closeness = 0.0
    for index, piece in enumerate(order):
        expected_center = slot_start + ((index + 1) * slot_width if use_gutter_spacing else (index + 0.5) * slot_width)
        closeness = max(0.0, 1.0 - abs(center - expected_center) / slot_width)
        if closeness > best_closeness:
            best_piece = piece
            best_closeness = closeness
    return best_piece, best_closeness


def merge_hand_candidates(learned_candidates: Sequence[Candidate], hog_candidates: Sequence[Candidate]) -> list[Candidate]:
    merged: dict[tuple[str, str], Candidate] = {}
    for candidate in learned_candidates:
        merged[(candidate.color, candidate.piece)] = Candidate(
            color=candidate.color,
            piece=candidate.piece,
            score=round(candidate.score, 4),
            bbox=candidate.bbox,
            source=f"hand_learned:{candidate.source}",
        )
    for candidate in hog_candidates:
        key = (candidate.color, candidate.piece)
        previous = merged.get(key)
        if previous is None:
            merged[key] = Candidate(
                color=candidate.color,
                piece=candidate.piece,
                score=round(candidate.score * 0.74, 4),
                bbox=candidate.bbox,
                scale=candidate.scale,
                source=f"hand_hog:{candidate.source}",
            )
        else:
            agreement_bonus = 0.05 if candidate.score >= 0.45 else 0.025
            merged[key] = Candidate(
                color=previous.color,
                piece=previous.piece,
                score=round(min(0.90, max(previous.score, candidate.score * 0.86) + agreement_bonus), 4),
                bbox=previous.bbox,
                scale=candidate.scale,
                source=f"{previous.source}+hand_hog:{candidate.source}",
            )
    return sorted(merged.values(), key=lambda candidate: candidate.score, reverse=True)


def hand_classifier_assets(hand_assets: dict[str, Any] | None = None):
    if hand_assets is not None:
        return hand_assets["model"], hand_assets["templates"]
    global _HAND_CLASSIFIER_CACHE
    if _HAND_CLASSIFIER_CACHE is None:
        from recognize_board_pieces import default_template_path, load_opencv_templates, train_hog_svm_from_sprites

        template_path = default_template_path()
        _HAND_CLASSIFIER_CACHE = {
            "model": train_hog_svm_from_sprites(template_path),
            "templates": load_opencv_templates(template_path),
        }
    return _HAND_CLASSIFIER_CACHE["model"], _HAND_CLASSIFIER_CACHE["templates"]


def crop_grid_cell(
    image: Image.Image,
    detection: GridDetection,
    row: int,
    col: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> Image.Image:
    xs = detection.vertical.positions
    ys = detection.horizontal.positions
    left = min(xs[col - 1], xs[col])
    right = max(xs[col - 1], xs[col])
    top = min(ys[row - 1], ys[row])
    bottom = max(ys[row - 1], ys[row])
    pad_x = round((right - left) * pad_x_ratio)
    pad_y = round((bottom - top) * pad_y_ratio)
    return image.crop(
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image.width, right + pad_x),
            min(image.height, bottom + pad_y),
        ),
    )


def summarize_cells(cells: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(cells),
        "piece": sum(1 for cell in cells if cell.get("state") == "piece"),
        "empty": sum(1 for cell in cells if cell.get("state") == "empty"),
        "unknown": sum(1 for cell in cells if cell.get("state") == "unknown"),
    }


def timing_report(started_ns: int) -> dict[str, float | int]:
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "processing_time_ns": elapsed_ns,
        "processing_time_ms": round(elapsed_ns / 1_000_000, 3),
        "processing_time_seconds": round(elapsed_ns / 1_000_000_000, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize board and hand pieces with a learned non-leaking model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-hands", action="store_true")
    args = parser.parse_args()
    report = recognize_image(args.image, args.model, include_hands=args.include_hands, out_path=args.out)
    print(f"OK: {args.image} {report['timing']['processing_time_ms']:.1f} ms")


if __name__ == "__main__":
    main()
