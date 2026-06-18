from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from detect_board_grid import GridDetection, detect_grid, iter_images, report_dict
from detect_hand_areas import (
    HandArea,
    cell_size,
    detect_hand_areas,
    detect_piece_color_components,
    draw_hand_area_overlay,
    grid_rect,
    write_json,
)
from recognize_board_pieces import (
    Candidate,
    black_ink_mask,
    calibration_letter_mask,
    classify_piece_cell_opencv,
    classify_piece_cell,
    default_template_path,
    extract_hog_features,
    load_opencv_templates,
    train_hog_svm_from_sprites,
    HOG_SCORE_SCALE,
    merge_hog_and_opencv_candidates,
)

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime.
    cv2 = None
    np = None


HAND_PIECES = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
HAND_PIECE_SET = set(HAND_PIECES)
DEFAULT_MIN_CONFIDENCE = 0.48
DEFAULT_AMBIGUOUS_MARGIN = 0.014
DIGIT_CANVAS = (32, 48)


@dataclass(frozen=True)
class PieceProposal:
    rect: list[int]
    source: str
    side: str
    owner: str


@dataclass(frozen=True)
class DigitCandidate:
    rect: list[int]
    digit: int
    confidence: float


@dataclass(frozen=True)
class RecognizedHandPiece:
    owner: str
    side: str
    piece: str
    rect: list[int]
    confidence: float
    ambiguous: bool
    proposal_source: str
    candidates: list[dict]
    digit: DigitCandidate | None = None


def ensure_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("hand piece recognition requires numpy and opencv-python. Run: python -m pip install -r tools\\requirements.txt")


def empty_hand_counts() -> dict[str, int]:
    return {piece: 0 for piece in HAND_PIECES}


def rect_width(rect: Sequence[int]) -> int:
    return rect[2] - rect[0]


def rect_height(rect: Sequence[int]) -> int:
    return rect[3] - rect[1]


def rect_center(rect: Sequence[int]) -> tuple[float, float]:
    return (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0


def clip_rect(rect: tuple[int, int, int, int], bounds: tuple[int, int, int, int]) -> list[int] | None:
    left, top, right, bottom = rect
    min_x, min_y, max_x, max_y = bounds
    clipped = [max(min_x, left), max(min_y, top), min(max_x, right), min(max_y, bottom)]
    if clipped[2] - clipped[0] < 8 or clipped[3] - clipped[1] < 8:
        return None
    return clipped


def pad_to_min_piece_box(
    rect: Sequence[int],
    bounds: tuple[int, int, int, int],
    cell_w: float,
    cell_h: float,
) -> list[int] | None:
    cx, cy = rect_center(rect)
    width = max(rect_width(rect) * 1.18, cell_w * 0.66)
    height = max(rect_height(rect) * 1.16, cell_h * 0.72)
    return clip_rect(
        (
            int(round(cx - width / 2)),
            int(round(cy - height / 2)),
            int(round(cx + width / 2)),
            int(round(cy + height / 2)),
        ),
        bounds,
    )


def glyph_component_boxes(area_rgb: object, cell_w: float, cell_h: float) -> list[list[int]]:
    ensure_opencv()
    mask = calibration_letter_mask(area_rgb)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[list[int]] = []
    for label in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < 20:
            continue
        if width < cell_w * 0.08 or height < cell_h * 0.10:
            continue
        if width > cell_w * 0.86 or height > cell_h * 0.90:
            continue
        fill = area / max(1, width * height)
        if fill < 0.06 or fill > 0.82:
            continue
        boxes.append([x, y, x + width, y + height])
    boxes.sort(key=lambda item: (item[1], item[0]))
    return boxes


def proposal_from_glyph(
    glyph_box: Sequence[int],
    area_rect: Sequence[int],
    cell_w: float,
    cell_h: float,
    side: str,
    owner: str,
) -> PieceProposal | None:
    area_left, area_top, area_right, area_bottom = area_rect
    local_cx, local_cy = rect_center(glyph_box)
    width = cell_w * 0.82
    height = cell_h * 0.98
    global_rect = (
        int(round(area_left + local_cx - width / 2)),
        int(round(area_top + local_cy - height * 0.50)),
        int(round(area_left + local_cx + width / 2)),
        int(round(area_top + local_cy + height * 0.50)),
    )
    rect = clip_rect(global_rect, (area_left, area_top, area_right, area_bottom))
    if rect is None:
        return None
    return PieceProposal(rect=rect, source="glyph", side=side, owner=owner)


def proposal_from_component(
    component: Sequence[int],
    area_rect: Sequence[int],
    cell_w: float,
    cell_h: float,
    side: str,
    owner: str,
) -> PieceProposal | None:
    rect = pad_to_min_piece_box(component, tuple(area_rect), cell_w, cell_h)
    if rect is None:
        return None
    return PieceProposal(rect=rect, source="color", side=side, owner=owner)


def rect_iou(first: Sequence[int], second: Sequence[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = rect_width(first) * rect_height(first)
    second_area = rect_width(second) * rect_height(second)
    return intersection / max(1, first_area + second_area - intersection)


def close_centers(first: Sequence[int], second: Sequence[int], cell_w: float, cell_h: float) -> bool:
    first_x, first_y = rect_center(first)
    second_x, second_y = rect_center(second)
    return abs(first_x - second_x) <= cell_w * 0.28 and abs(first_y - second_y) <= cell_h * 0.30


def merge_proposals(proposals: Iterable[PieceProposal], cell_w: float, cell_h: float) -> list[PieceProposal]:
    merged: list[PieceProposal] = []
    source_rank = {"color": 2, "glyph": 1}
    for proposal in sorted(proposals, key=lambda item: (item.side, item.rect[1], item.rect[0], -source_rank.get(item.source, 0))):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if existing.owner == proposal.owner
                and existing.side == proposal.side
                and (rect_iou(existing.rect, proposal.rect) >= 0.42 or close_centers(existing.rect, proposal.rect, cell_w, cell_h))
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(proposal)
            continue
        existing = merged[duplicate_index]
        if source_rank.get(proposal.source, 0) > source_rank.get(existing.source, 0):
            merged[duplicate_index] = proposal
    return merged


def proposals_for_area(
    image: Image.Image,
    area: HandArea,
    cell_w: float,
    cell_h: float,
    max_proposals: int,
) -> list[PieceProposal]:
    area_rect = tuple(area.rect)
    proposals: list[PieceProposal] = []

    components = detect_piece_color_components(image, area_rect, cell_w, cell_h)
    for component in components:
        proposal = proposal_from_component(component, area.rect, cell_w, cell_h, area.side, area.owner)
        if proposal is not None:
            proposals.append(proposal)

    area_rgb = np.array(image.crop(area_rect).convert("RGB"))
    for glyph_box in glyph_component_boxes(area_rgb, cell_w, cell_h):
        proposal = proposal_from_glyph(glyph_box, area.rect, cell_w, cell_h, area.side, area.owner)
        if proposal is not None:
            proposals.append(proposal)

    merged = merge_proposals(proposals, cell_w, cell_h)
    return merged[:max_proposals]


def shoko_icon_proposals_for_area(
    image: Image.Image,
    area: HandArea,
    cell_w: float,
    cell_h: float,
) -> list[PieceProposal]:
    ensure_opencv()
    area_rect = tuple(area.rect)
    area_left, area_top, area_right, area_bottom = area_rect
    area_rgb = np.array(image.crop(area_rect).convert("RGB"))
    proposals: list[PieceProposal] = []
    for glyph_box in glyph_component_boxes(area_rgb, cell_w, cell_h):
        glyph_w = glyph_box[2] - glyph_box[0]
        glyph_h = glyph_box[3] - glyph_box[1]
        if glyph_w < cell_w * 0.30 or glyph_w > cell_w * 0.80:
            continue
        if glyph_h < cell_h * 0.30 or glyph_h > cell_h * 0.55:
            continue
        local_cx = (glyph_box[0] + glyph_box[2]) / 2.0
        local_cy = (glyph_box[1] + glyph_box[3]) / 2.0
        global_rect = (
            int(round(area_left + local_cx - cell_w / 2)),
            int(round(area_top + local_cy - cell_h / 2)),
            int(round(area_left + local_cx + cell_w / 2)),
            int(round(area_top + local_cy + cell_h / 2)),
        )
        rect = clip_rect(global_rect, (area_left, area_top, area_right, area_bottom))
        if rect is None:
            continue
        proposals.append(PieceProposal(rect=rect, source="shoko_icon", side=area.side, owner=area.owner))
    return proposals


def candidate_to_dict(candidate: Candidate) -> dict:
    return asdict(candidate)


def piece_material_features(image: Image.Image) -> dict[str, float]:
    ensure_opencv()
    rgb = np.array(image.convert("RGB")).astype("int16")
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    hsv = cv2.cvtColor(rgb.astype("uint8"), cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    cream = (
        (red > 190)
        & (green > 170)
        & (blue > 125)
        & (blue < 235)
        & (np.abs(red - green) < 55)
        & (red > blue + 20)
        & (green > blue + 5)
        & (saturation < 115)
    )
    gold = (
        (red > 165)
        & (green > 105)
        & (blue < 120)
        & (red > green + 20)
        & (green > blue + 20)
        & (saturation > 80)
    )
    total = max(1, cream.size)
    return {
        "cream_ratio": round(float(np.count_nonzero(cream)) / total, 4),
        "gold_ratio": round(float(np.count_nonzero(gold)) / total, 4),
    }


def has_piece_material(features: dict[str, float], proposal: PieceProposal, area_by_side: dict[str, HandArea]) -> bool:
    if proposal.source == "color":
        return True
    area = area_by_side.get(proposal.side)
    if area is not None and area.evidence == "board_gutter":
        return features["cream_ratio"] >= 0.12
    return features["cream_ratio"] >= 0.09 or features["gold_ratio"] >= 0.16


def best_hand_candidate(candidates: Sequence[Candidate], owner: str) -> tuple[Candidate | None, float]:
    normal = [candidate for candidate in candidates if candidate.piece in HAND_PIECE_SET and candidate.color == owner]
    if not normal:
        return None, 0.0
    best = normal[0]
    second_score = normal[1].score if len(normal) > 1 else 0.0
    return best, best.score - second_score


def classify_proposals(
    image: Image.Image,
    proposals: Sequence[PieceProposal],
    model,
    templates,
    crops_dir: Path,
    areas: Sequence[HandArea],
    min_confidence: float,
    ambiguous_margin: float,
    save_crops: bool = True,
) -> tuple[list[RecognizedHandPiece], list[dict]]:
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[RecognizedHandPiece] = []
    unknown: list[dict] = []
    area_by_side = {area.side: area for area in areas}
    for index, proposal in enumerate(proposals, start=1):
        crop_path = crops_dir / f"{proposal.owner}_{proposal.side}_{index:03d}_{proposal.source}.png"
        crop = image.crop(tuple(proposal.rect))
        if save_crops:
            crop.save(crop_path)
        material = piece_material_features(crop)
        if not has_piece_material(material, proposal, area_by_side):
            continue
        if save_crops:
            candidates = classify_piece_cell(crop_path, model, templates)[:5]
        else:
            candidates = classify_piece_crop(crop, model, templates)[:5]
        best, margin = best_hand_candidate(candidates, proposal.owner)
        candidate_dicts = [candidate_to_dict(candidate) for candidate in candidates]
        ambiguous = best is not None and margin < ambiguous_margin
        if best is not None and best.score >= min_confidence:
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
                    "material": material,
                    "candidates": candidate_dicts,
                },
            )
    return suppress_duplicate_pieces(accepted), unknown[:30]


def classify_piece_crop(crop: Image.Image, model, opencv_templates) -> list[Candidate]:
    feature = extract_hog_features(crop.convert("RGB"))
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
        return sorted(hog_candidates, key=lambda candidate: candidate.score, reverse=True)
    # Avoid a public temp file for the common fast path by using HOG directly and keeping OpenCV fallback disabled.
    return merge_hog_and_opencv_candidates(hog_candidates, [])


def suppress_duplicate_pieces(pieces: Sequence[RecognizedHandPiece]) -> list[RecognizedHandPiece]:
    kept: list[RecognizedHandPiece] = []
    for piece in sorted(pieces, key=lambda item: item.confidence, reverse=True):
        if any(existing.owner == piece.owner and existing.side == piece.side and rect_iou(existing.rect, piece.rect) >= 0.35 for existing in kept):
            continue
        kept.append(piece)
    kept.sort(key=lambda item: (item.owner, item.side, item.rect[1], item.rect[0]))
    return kept


def digit_mask(rgb: object) -> object:
    ensure_opencv()
    dark = black_ink_mask(rgb)
    channels = rgb.astype("int16")
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    red_digit = (
        (red > 145)
        & (green < 150)
        & (blue < 150)
        & (red > green + 25)
        & (red > blue + 25)
        & (saturation > 45)
    )
    yellow = (red > 200) & (green > 165) & (blue < 110) & (saturation > 80)
    mask = np.where((dark > 0) | yellow | red_digit, 255, 0).astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype="uint8"))
    return mask


def color_digit_mask(rgb: object, *, strict_yellow: bool = False) -> object:
    ensure_opencv()
    channels = rgb.astype("int16")
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    red_digit = (
        (red > 145)
        & (green < 150)
        & (blue < 150)
        & (red > green + 25)
        & (red > blue + 25)
        & (saturation > 45)
    )
    if strict_yellow:
        # Side-hand count glyphs are much brighter than golden piece edges.
        # Keep this mask narrow so digits do not merge into the piece body.
        yellow = (
            (red > 210)
            & (green > 200)
            & (blue < 150)
            & (np.abs(red - green) < 80)
            & (saturation > 55)
        )
    else:
        yellow = (red > 200) & (green > 165) & (blue < 130) & (saturation > 60)
    mask = np.where(yellow | red_digit, 255, 0).astype("uint8")
    if strict_yellow:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype="uint8"))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype="uint8"))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype="uint8"))


def bright_count_digit_mask(rgb: object) -> object:
    ensure_opencv()
    channels = rgb.astype("int16")
    red = channels[:, :, 0]
    green = channels[:, :, 1]
    blue = channels[:, :, 2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, saturation, value = cv2.split(hsv)
    yellow_count = (
        (red > 220)
        & (green > 205)
        & (blue < 130)
        & (red - green < 85)
        & (saturation > 75)
        & (value > 180)
    )
    mask = np.where(yellow_count, 255, 0).astype("uint8")
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), dtype="uint8"))


def fit_mask_to_canvas(mask: object, size: tuple[int, int] = DIGIT_CANVAS) -> object:
    canvas_w, canvas_h = size
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((canvas_h, canvas_w), dtype="uint8")
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    crop = mask[y1 : y2 + 1, x1 : x2 + 1]
    height, width = crop.shape[:2]
    scale = min((canvas_w - 4) / max(1, width), (canvas_h - 4) / max(1, height))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    resized = np.where(resized > 30, 255, 0).astype("uint8")
    canvas = np.zeros((canvas_h, canvas_w), dtype="uint8")
    left = (canvas_w - resized_w) // 2
    top = (canvas_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def digit_template_font_paths() -> list[Path]:
    return [
        path
        for path in (
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/meiryob.ttc"),
            Path("C:/Windows/Fonts/meiryo.ttc"),
            Path("C:/Windows/Fonts/msgothic.ttc"),
        )
        if path.exists()
    ]


@lru_cache(maxsize=1)
def digit_templates() -> tuple[tuple[int, object], ...]:
    ensure_opencv()
    templates: list[tuple[int, object]] = []
    font_paths = digit_template_font_paths()
    fonts: list[ImageFont.FreeTypeFont | ImageFont.ImageFont] = []
    for font_path in font_paths:
        for size in (34, 40, 46, 54):
            try:
                fonts.append(ImageFont.truetype(str(font_path), size=size))
            except OSError:
                continue
    if not fonts:
        fonts.append(ImageFont.load_default())

    for digit in range(10):
        for font in fonts:
            canvas = Image.new("L", (80, 96), 0)
            draw = ImageDraw.Draw(canvas)
            bbox = draw.textbbox((0, 0), str(digit), font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(((80 - text_w) / 2 - bbox[0], (96 - text_h) / 2 - bbox[1]), str(digit), fill=255, font=font)
            templates.append((digit, fit_mask_to_canvas(np.array(canvas))))
    return tuple(templates)


def binary_similarity(first: object, second: object) -> float:
    first_mask = first > 0
    second_mask = second > 0
    first_count = int(np.count_nonzero(first_mask))
    second_count = int(np.count_nonzero(second_mask))
    if first_count < 8 or second_count < 8:
        return 0.0
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    dice = (2.0 * intersection) / max(1, first_count + second_count)
    iou = intersection / max(1, union)
    return dice * 0.75 + iou * 0.25


def recognize_digit(mask_crop: object) -> tuple[int | None, float]:
    normalized = fit_mask_to_canvas(mask_crop)
    best_digit: int | None = None
    best_score = 0.0
    for digit, template in digit_templates():
        score = binary_similarity(normalized, template)
        if score > best_score:
            best_digit = digit
            best_score = score
    if best_digit is None or best_score < 0.38:
        return None, 0.0
    return best_digit, round(best_score, 4)


def digit_candidates_for_area(
    image: Image.Image,
    area: HandArea,
    cell_w: float,
    cell_h: float,
) -> list[DigitCandidate]:
    ensure_opencv()
    area_rect = tuple(area.rect)
    area_rgb = np.array(image.crop(area_rect).convert("RGB"))
    mask = digit_mask(area_rgb)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    digits: list[DigitCandidate] = []
    append_digit_components(digits, labels, stats, area, cell_w, cell_h, confidence_bonus=0.0, min_score=0.40)
    if area.side in {"left", "right"}:
        bright_mask = bright_count_digit_mask(area_rgb)
        bright_component_count, bright_labels, bright_stats, _ = cv2.connectedComponentsWithStats(bright_mask, 8)
        append_digit_components(
            digits,
            bright_labels,
            bright_stats,
            area,
            cell_w,
            cell_h,
            confidence_bonus=0.08,
            min_score=0.66,
        )
    color_mask = color_digit_mask(area_rgb, strict_yellow=area.side in {"left", "right"})
    _, color_labels, color_stats, _ = cv2.connectedComponentsWithStats(color_mask, 8)
    append_digit_components(
        digits,
        color_labels,
        color_stats,
        area,
        cell_w,
        cell_h,
        confidence_bonus=0.08,
        min_score=0.62,
        count_digits_only=True,
    )
    return suppress_duplicate_digits(digits)


def append_digit_components(
    digits: list[DigitCandidate],
    labels: object,
    stats: object,
    area: HandArea,
    cell_w: float,
    cell_h: float,
    *,
    confidence_bonus: float,
    min_score: float,
    count_digits_only: bool = False,
) -> None:
    component_count = int(stats.shape[0])
    for label in range(1, component_count):
        x, y, width, height, area_size = (int(value) for value in stats[label])
        if area_size < 22:
            continue
        if width < cell_w * 0.08 or width > cell_w * 0.58:
            continue
        if height < cell_h * 0.18 or height > cell_h * 0.78:
            continue
        aspect = width / max(1, height)
        if aspect < 0.16 or aspect > 1.05:
            continue
        component_mask = np.where(labels[y : y + height, x : x + width] == label, 255, 0).astype("uint8")
        digit, score = recognize_digit(component_mask)
        if digit is None or score < min_score:
            continue
        if count_digits_only and digit <= 1:
            continue
        confidence = round(min(0.99, score + confidence_bonus), 4)
        digits.append(
            DigitCandidate(
                rect=[area.rect[0] + x, area.rect[1] + y, area.rect[0] + x + width, area.rect[1] + y + height],
                digit=digit,
                confidence=confidence,
            ),
        )


def suppress_duplicate_digits(digits: Sequence[DigitCandidate]) -> list[DigitCandidate]:
    kept: list[DigitCandidate] = []
    for digit in sorted(digits, key=lambda item: item.confidence, reverse=True):
        if any(rect_iou(existing.rect, digit.rect) >= 0.35 for existing in kept):
            continue
        kept.append(digit)
    kept.sort(key=lambda item: (item.rect[1], item.rect[0]))
    return kept


def digit_piece_association_score(
    piece: RecognizedHandPiece,
    digit: DigitCandidate,
    cell_w: float,
    cell_h: float,
    relaxed: bool = False,
) -> float | None:
    px, py = rect_center(piece.rect)
    digit_x, digit_y = rect_center(digit.rect)
    dx = digit_x - px
    dy = abs(digit_y - py)
    piece_w = max(1, rect_width(piece.rect))
    piece_h = max(1, rect_height(piece.rect))
    piece_large_enough_for_inside_digit = piece_w >= cell_w * 0.58 and piece_h >= cell_h * 0.75
    right_threshold = 0.28 if relaxed else 0.32
    right_limit = 0.72 if relaxed else 0.58
    overlap_limit = 0.18 if relaxed else 0.20
    vertical_limit = 0.60 if relaxed else 0.55
    inside_vertical = 0.45 if relaxed else 0.42
    x_scale = 1.08 if relaxed else 0.95
    y_scale = 0.78 if relaxed else 0.70
    digit_center_inside = (
        piece.rect[0] <= digit_x <= piece.rect[2]
        and piece.rect[1] <= digit_y <= piece.rect[3]
    )
    if digit_center_inside and not piece_large_enough_for_inside_digit:
        return None
    digit_is_right_count = digit_x >= px + piece_w * right_threshold and digit.rect[0] <= piece.rect[2] + cell_w * right_limit
    overlaps_or_right = digit_is_right_count and digit.rect[2] >= piece.rect[0] + cell_w * overlap_limit
    near_vertical = dy <= cell_h * vertical_limit
    if not ((digit_center_inside and dy <= piece_h * inside_vertical) or overlaps_or_right) or not near_vertical or dx < -cell_w * 0.45:
        return None
    distance_score = 1.0 - min(1.0, (abs(dx) / max(1.0, cell_w * x_scale) + dy / max(1.0, cell_h * y_scale)) / 2.0)
    inside_bonus = 0.24 if digit_center_inside else 0.0
    return distance_score * 0.70 + digit.confidence * 0.30 + inside_bonus


def merge_digit_rect(digits: Sequence[DigitCandidate]) -> list[int]:
    return [
        min(digit.rect[0] for digit in digits),
        min(digit.rect[1] for digit in digits),
        max(digit.rect[2] for digit in digits),
        max(digit.rect[3] for digit in digits),
    ]


def digit_sequence_options(
    piece: RecognizedHandPiece,
    digits: Sequence[DigitCandidate],
    used: set[int],
    cell_w: float,
    cell_h: float,
) -> list[tuple[float, tuple[int, ...], DigitCandidate]]:
    single_scored: list[tuple[int, DigitCandidate, float]] = []
    relaxed_scored: list[tuple[int, DigitCandidate, float]] = []
    for index, digit in enumerate(digits):
        if index in used:
            continue
        if digit.confidence >= 0.78:
            score = digit_piece_association_score(piece, digit, cell_w, cell_h)
            if score is not None:
                single_scored.append((index, digit, score))
        if digit.confidence >= 0.72:
            score = digit_piece_association_score(piece, digit, cell_w, cell_h, relaxed=True)
            if score is not None:
                relaxed_scored.append((index, digit, score))

    options: list[tuple[float, tuple[int, ...], DigitCandidate]] = []
    for index, digit, score in single_scored:
        if digit.digit > 1:
            options.append((score, (index,), digit))

    for left_index, left_digit, left_score in relaxed_scored:
        if left_digit.digit != 1:
            continue
        for right_index, right_digit, right_score in relaxed_scored:
            if right_index == left_index:
                continue
            left_x, left_y = rect_center(left_digit.rect)
            right_x, right_y = rect_center(right_digit.rect)
            if right_x <= left_x:
                continue
            if abs(right_y - left_y) > cell_h * 0.38:
                continue
            if right_digit.rect[0] - left_digit.rect[2] > cell_w * 0.38:
                continue
            value = left_digit.digit * 10 + right_digit.digit
            if value < 10 or value > 18:
                continue
            confidence = round(min(left_digit.confidence, right_digit.confidence), 4)
            combined = DigitCandidate(
                rect=merge_digit_rect((left_digit, right_digit)),
                digit=value,
                confidence=confidence,
            )
            score = (left_score + right_score) / 2.0 + 0.18
            options.append((score, (left_index, right_index), combined))
    return sorted(options, key=lambda item: item[0], reverse=True)


def associate_digits(
    pieces: Sequence[RecognizedHandPiece],
    digits: Sequence[DigitCandidate],
    cell_w: float,
    cell_h: float,
) -> list[RecognizedHandPiece]:
    assignments: dict[int, DigitCandidate] = {}
    used: set[int] = set()
    all_options: list[tuple[float, int, tuple[int, ...], DigitCandidate]] = []
    for piece_index, piece in enumerate(pieces):
        if piece.confidence < 0.52:
            continue
        if piece.piece != "FU" and piece.confidence < 0.58:
            continue
        for score, selected_indices, selected_digit in digit_sequence_options(piece, digits, set(), cell_w, cell_h):
            all_options.append((score, piece_index, selected_indices, selected_digit))

    for _, piece_index, selected_indices, selected_digit in sorted(all_options, key=lambda item: item[0], reverse=True):
        if piece_index in assignments:
            continue
        if any(index in used for index in selected_indices):
            continue
        used.update(selected_indices)
        assignments[piece_index] = selected_digit

    updated: list[RecognizedHandPiece] = []
    for piece_index, piece in enumerate(pieces):
        selected_digit = assignments.get(piece_index)
        if selected_digit is None:
            updated.append(piece)
            continue
        updated.append(
            RecognizedHandPiece(
                owner=piece.owner,
                side=piece.side,
                piece=piece.piece,
                rect=piece.rect,
                confidence=piece.confidence,
                ambiguous=piece.ambiguous,
                proposal_source=piece.proposal_source,
                candidates=piece.candidates,
                digit=selected_digit,
            ),
        )
    return updated


def aggregate_hands(pieces: Sequence[RecognizedHandPiece], use_icon_count: bool = False) -> tuple[dict[str, dict[str, int]], list[dict]]:
    hands = {
        "black": empty_hand_counts(),
        "white": empty_hand_counts(),
    }
    grouped: dict[tuple[str, str], dict] = {}
    for piece in pieces:
        key = (piece.owner, piece.piece)
        entry = grouped.setdefault(
            key,
            {
                "owner": piece.owner,
                "piece": piece.piece,
                "count": 0,
                "count_source": "icons",
                "confidence": 0.0,
                "rects": [],
                "digits": [],
                "candidate_sets": [],
                "ambiguous": False,
            },
        )
        count = piece.digit.digit if piece.digit is not None and piece.digit.digit > 1 else 1
        if piece.digit is not None and piece.digit.digit > 1:
            entry["count"] = max(int(entry["count"]), count)
        else:
            entry["count"] = max(int(entry["count"]), count)
        entry["confidence"] = max(entry["confidence"], piece.confidence)
        entry["rects"].append(piece.rect)
        entry["candidate_sets"].append({"rect": piece.rect, "candidates": piece.candidates})
        entry["ambiguous"] = bool(entry["ambiguous"] or piece.ambiguous)
        if piece.digit is not None and piece.digit.digit > 1:
            entry["count_source"] = "digit"
            entry["digits"].append(asdict(piece.digit))

    piece_entries = []
    for (owner, piece), entry in sorted(grouped.items(), key=lambda item: (item[0][0], HAND_PIECES.index(item[0][1]))):
        if use_icon_count:
            count = len(entry["rects"])
            entry["count_source"] = "icons"
        else:
            count = int(entry["count"])
        hands[owner][piece] = count
        entry["count"] = count
        entry["confidence"] = round(entry["confidence"], 4)
        piece_entries.append(entry)
    return hands, piece_entries


def draw_hand_piece_overlay(
    image: Image.Image,
    detection: GridDetection | None,
    areas: Sequence[HandArea],
    pieces: Sequence[RecognizedHandPiece],
    unknown: Sequence[dict],
    digits: Sequence[DigitCandidate],
) -> Image.Image:
    overlay = draw_hand_area_overlay(image, detection, areas)
    draw = ImageDraw.Draw(overlay)
    for item in unknown:
        rect = tuple(item["rect"])
        draw.rectangle(rect, outline=(255, 220, 0), width=3)
        draw.text((rect[0] + 3, rect[1] + 3), f"? {item.get('best_piece') or ''} {item.get('confidence', 0):.2f}", fill=(255, 220, 0))
    for piece in pieces:
        color = (0, 230, 80) if piece.owner == "black" else (255, 70, 70)
        rect = tuple(piece.rect)
        draw.rectangle(rect, outline=color, width=4)
        label = f"{piece.owner}:{piece.piece} {piece.confidence:.2f}"
        if piece.digit is not None:
            label += f" x{piece.digit.digit}"
        draw.text((rect[0] + 4, max(0, rect[1] - 18)), label, fill=color)
    for digit in digits:
        rect = tuple(digit.rect)
        draw.rectangle(rect, outline=(60, 160, 255), width=3)
        draw.text((rect[0] + 3, rect[3] + 2), f"{digit.digit} {digit.confidence:.2f}", fill=(60, 160, 255))
    return overlay


def draw_candidate_grid(
    image: Image.Image,
    pieces: Sequence[RecognizedHandPiece],
    unknown: Sequence[dict],
    out_path: Path,
) -> None:
    items: list[tuple[list[int], str]] = []
    for piece in pieces:
        items.append((piece.rect, f"{piece.owner}:{piece.piece} {piece.confidence:.2f}"))
    for item in unknown:
        items.append((item["rect"], f"?:{item.get('best_piece') or '-'} {item.get('confidence', 0):.2f}"))
    if not items:
        Image.new("RGB", (320, 120), (245, 245, 245)).save(out_path)
        return
    tile_w, tile_h = 150, 150
    cols = min(5, max(1, len(items)))
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (rect, label) in enumerate(items):
        col = index % cols
        row = index // cols
        x = col * tile_w
        y = row * tile_h
        crop = image.crop(tuple(rect)).convert("RGB")
        crop.thumbnail((tile_w - 12, tile_h - 34))
        canvas.paste(crop, (x + (tile_w - crop.width) // 2, y + 6))
        draw.text((x + 6, y + tile_h - 24), label, fill=(20, 20, 20))
    canvas.save(out_path)


def recognize_image(
    image_path: Path,
    out_root: Path,
    template_path: Path,
    min_confidence: float,
    ambiguous_margin: float,
    max_proposals_per_area: int,
    model=None,
    templates=None,
) -> dict:
    ensure_opencv()
    image = Image.open(image_path).convert("RGB")
    detection = detect_grid(image)
    areas = detect_hand_areas(image, detection)
    out_dir = out_root / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if detection is None:
        report = {
            "image": str(image_path),
            "image_size": list(image.size),
            "grid_detected": False,
            "hands": {"black": empty_hand_counts(), "white": empty_hand_counts()},
            "areas": [],
            "pieces": [],
            "unknown": [],
            "digits": [],
            "grid": report_dict(image, detection),
        }
        write_recognition_outputs(image, detection, areas, [], [], [], report, out_dir)
        return report

    cell_w, cell_h = cell_size(detection)
    all_proposals: list[PieceProposal] = []
    all_digits: list[DigitCandidate] = []
    for area in areas:
        all_proposals.extend(proposals_for_area(image, area, cell_w, cell_h, max_proposals_per_area))
        all_digits.extend(digit_candidates_for_area(image, area, cell_w, cell_h))

    resolved_model = model if model is not None else train_hog_svm_from_sprites(template_path)
    resolved_templates = templates if templates is not None else load_opencv_templates(template_path)
    pieces, unknown = classify_proposals(
        image,
        all_proposals,
        resolved_model,
        resolved_templates,
        out_dir / "candidate_crops",
        areas,
        min_confidence,
        ambiguous_margin,
    )
    pieces = associate_digits(pieces, all_digits, cell_w, cell_h)
    hands, piece_entries = aggregate_hands(pieces)
    report = {
        "image": str(image_path),
        "image_size": list(image.size),
        "grid_detected": True,
        "grid": report_dict(image, detection),
        "hands": hands,
        "areas": [
            {
                "owner": area.owner,
                "side": area.side,
                "rect": area.rect,
                "confidence": area.confidence,
                "evidence": area.evidence,
            }
            for area in areas
        ],
        "pieces": piece_entries,
        "raw_pieces": [
            {
                **asdict(piece),
                "digit": asdict(piece.digit) if piece.digit is not None else None,
            }
            for piece in pieces
        ],
        "unknown": unknown,
        "digits": [asdict(digit) for digit in all_digits],
        "summary": {
            "area_count": len(areas),
            "proposal_count": len(all_proposals),
            "recognized_piece_icons": len(pieces),
            "unknown_count": len(unknown),
            "digit_count": len(all_digits),
        },
        "thresholds": {
            "piece_confidence": min_confidence,
            "ambiguous_margin": ambiguous_margin,
        },
    }
    write_recognition_outputs(image, detection, areas, pieces, unknown, all_digits, report, out_dir)
    return report


def write_recognition_outputs(
    image: Image.Image,
    detection: GridDetection | None,
    areas: Sequence[HandArea],
    pieces: Sequence[RecognizedHandPiece],
    unknown: Sequence[dict],
    digits: Sequence[DigitCandidate],
    report: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    area_report = {
        "image": report["image"],
        "image_size": report["image_size"],
        "grid": report.get("grid", report_dict(image, detection)),
        "grid_detected": detection is not None,
        "summary": {
            "area_count": len(areas),
            "sides": [area.side for area in areas],
        },
        "areas": [asdict(area) for area in areas],
    }
    write_json(out_dir / "hand_area_report.json", area_report)
    write_json(out_dir / "hand_pieces_report.json", report)
    draw_hand_area_overlay(image, detection, areas).save(out_dir / "hand_area_overlay.png")
    draw_hand_piece_overlay(image, detection, areas, pieces, unknown, digits).save(out_dir / "hand_piece_overlay.png")
    draw_candidate_grid(image, pieces, unknown, out_dir / "hand_candidate_grid.png")
    areas_dir = out_dir / "areas"
    areas_dir.mkdir(parents=True, exist_ok=True)
    for area in areas:
        image.crop(tuple(area.rect)).save(areas_dir / f"{area.owner}_{area.side}.png")


def recognize_images(
    input_path: Path,
    out_root: Path,
    template_path: Path,
    min_confidence: float,
    ambiguous_margin: float,
    max_proposals_per_area: int,
) -> list[dict]:
    ensure_opencv()
    model = train_hog_svm_from_sprites(template_path)
    templates = load_opencv_templates(template_path)
    reports: list[dict] = []
    for image_path in iter_images(input_path):
        report = recognize_image(
            image_path,
            out_root,
            template_path,
            min_confidence,
            ambiguous_margin,
            max_proposals_per_area,
            model=model,
            templates=templates,
        )
        reports.append(report)
        status = "OK" if report["grid_detected"] else "NG"
        print(
            f"{status}: {image_path} -> {out_root / image_path.stem} "
            f"(areas={len(report['areas'])}, pieces={len(report.get('raw_pieces', []))}, unknown={len(report['unknown'])})",
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize captured shogi pieces in hand areas around a detected board.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("tools/samples/screenshots"),
        help="Screenshot image or directory. Defaults to tools/samples/screenshots.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tools/out/hand_recognition"),
        help="Output directory for hand recognition reports and debug images.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=default_template_path(),
        help="Path to shogi_pieces.png. Defaults to the Android bundled sprite sheet.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum normal-piece score to accept a held piece. Defaults to {DEFAULT_MIN_CONFIDENCE}.",
    )
    parser.add_argument(
        "--ambiguous-margin",
        type=float,
        default=DEFAULT_AMBIGUOUS_MARGIN,
        help=f"Margin below which an accepted piece is flagged ambiguous. Defaults to {DEFAULT_AMBIGUOUS_MARGIN}.",
    )
    parser.add_argument(
        "--max-proposals-per-area",
        type=int,
        default=36,
        help="Maximum candidate crops to classify per detected hand area.",
    )
    args = parser.parse_args()

    reports = recognize_images(
        args.input,
        args.out,
        args.template,
        args.min_confidence,
        args.ambiguous_margin,
        args.max_proposals_per_area,
    )
    ok_count = sum(1 for report in reports if report["grid_detected"])
    recognized_count = sum(len(report.get("raw_pieces", [])) for report in reports)
    print(
        f"OK: analyzed {len(reports)} image(s), grid detected for {ok_count}, "
        f"recognized piece icons={recognized_count}. Output: {args.out}",
    )


if __name__ == "__main__":
    main()
