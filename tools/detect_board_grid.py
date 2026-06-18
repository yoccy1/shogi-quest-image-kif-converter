from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from PIL import Image, ImageDraw

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional for the CLI fallback.
    np = None


LINE_COUNT = 10
GRID_INTERVALS = 9


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class AxisDetection:
    start: int
    step: float
    positions: list[int]
    confidence: float


@dataclass(frozen=True)
class GridDetection:
    board_rect: Rect
    vertical: AxisDetection
    horizontal: AxisDetection
    confidence: float
    method: str


def is_board_pixel(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return (
        red > 115
        and green > 85
        and blue < 175
        and red >= green
        and red > blue * 1.20
        and green > blue * 1.08
        and abs(red - green) < 125
    )


def is_dark_line_pixel(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return max(red, green, blue) < 92 and min(red, green, blue) < 62


def is_brown_grid_line_pixel(rgb: tuple[int, int, int]) -> bool:
    red, green, blue = rgb
    return (
        55 <= red <= 205
        and 35 <= green <= 150
        and blue <= 115
        and red > green * 1.05
        and green > blue * 1.04
        and red - green < 105
    )


def smooth(values: list[float], radius: int) -> list[float]:
    if radius <= 0:
        return values
    smoothed: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


def contiguous_intervals(
    values: list[float],
    threshold: float,
    min_length: int,
    allowed_gap: int = 4,
) -> list[tuple[int, int, float]]:
    intervals: list[tuple[int, int, float]] = []
    start: int | None = None
    last_hit = 0
    for index, value in enumerate(values):
        if value >= threshold:
            if start is None:
                start = index
            last_hit = index
        elif start is not None and index - last_hit > allowed_gap:
            end = last_hit + 1
            if end - start >= min_length:
                intervals.append((start, end, sum(values[start:end]) / (end - start)))
            start = None
    if start is not None:
        end = last_hit + 1
        if end - start >= min_length:
            intervals.append((start, end, sum(values[start:end]) / (end - start)))
    return intervals


def largest_interval(intervals: Iterable[tuple[int, int, float]]) -> tuple[int, int, float] | None:
    return max(intervals, key=lambda item: ((item[1] - item[0]) * item[2], item[1] - item[0]), default=None)


def detect_board_area(image: Image.Image) -> Rect | None:
    if np is not None:
        return detect_board_area_numpy(image)

    width, height = image.size
    pixels = image.load()
    row_ratios: list[float] = []
    for y in range(height):
        hits = 0
        for x in range(width):
            if is_board_pixel(pixels[x, y]):
                hits += 1
        row_ratios.append(hits / width)

    y_interval = largest_interval(
        contiguous_intervals(
            smooth(row_ratios, radius=5),
            threshold=0.42,
            min_length=max(120, height // 8),
            allowed_gap=10,
        ),
    )
    if y_interval is None:
        return None

    top, bottom, _ = y_interval
    col_ratios: list[float] = []
    for x in range(width):
        hits = 0
        total = bottom - top
        for y in range(top, bottom):
            if is_board_pixel(pixels[x, y]):
                hits += 1
        col_ratios.append(hits / max(1, total))

    x_interval = largest_interval(
        contiguous_intervals(
            smooth(col_ratios, radius=3),
            threshold=0.35,
            min_length=max(120, width // 4),
            allowed_gap=10,
        ),
    )
    if x_interval is None:
        return None

    left, right, _ = x_interval
    return Rect(left=left, top=top, right=right, bottom=bottom)


def detect_board_area_numpy(image: Image.Image) -> Rect | None:
    array = image_rgb_array(image)
    mask = board_pixel_mask(array)
    row_ratios = mask.mean(axis=1).tolist()

    height, width = mask.shape
    y_interval = largest_interval(
        contiguous_intervals(
            smooth(row_ratios, radius=5),
            threshold=0.42,
            min_length=max(120, height // 8),
            allowed_gap=10,
        ),
    )
    if y_interval is None:
        return None

    top, bottom, _ = y_interval
    col_ratios = mask[top:bottom, :].mean(axis=0).tolist()
    x_interval = largest_interval(
        contiguous_intervals(
            smooth(col_ratios, radius=3),
            threshold=0.35,
            min_length=max(120, width // 4),
            allowed_gap=10,
        ),
    )
    if x_interval is None:
        return None

    left, right, _ = x_interval
    return Rect(left=left, top=top, right=right, bottom=bottom)


def dark_projection_scores(image: Image.Image, rect: Rect, axis: str) -> list[float]:
    if np is not None:
        return mask_projection_scores(image, rect, axis, dark_line_pixel_mask)
    return projection_scores(image, rect, axis, is_dark_line_pixel)


def brown_projection_scores(image: Image.Image, rect: Rect, axis: str) -> list[float]:
    if np is not None:
        return mask_projection_scores(image, rect, axis, brown_grid_line_pixel_mask)
    return projection_scores(image, rect, axis, is_brown_grid_line_pixel)


def mask_projection_scores(
    image: Image.Image,
    rect: Rect,
    axis: str,
    mask_func,
) -> list[float]:
    array = image_rgb_array(image)
    crop = array[rect.top : rect.bottom, rect.left : rect.right, :]
    mask = mask_func(crop)
    if axis == "x":
        return mask.mean(axis=0).tolist()
    if axis == "y":
        return mask.mean(axis=1).tolist()
    raise ValueError(f"unknown axis: {axis}")


def image_rgb_array(image: Image.Image) -> object:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def board_pixel_mask(array: object) -> object:
    values = array.astype("int16", copy=False)
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return (
        (red > 115)
        & (green > 85)
        & (blue < 175)
        & (red >= green)
        & (red * 100 > blue * 120)
        & (green * 100 > blue * 108)
        & (np.abs(red - green) < 125)
    )


def dark_line_pixel_mask(array: object) -> object:
    values = array.astype("int16", copy=False)
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    return (maximum < 92) & (minimum < 62)


def brown_grid_line_pixel_mask(array: object) -> object:
    values = array.astype("int16", copy=False)
    red = values[:, :, 0]
    green = values[:, :, 1]
    blue = values[:, :, 2]
    return (
        (55 <= red)
        & (red <= 205)
        & (35 <= green)
        & (green <= 150)
        & (blue <= 115)
        & (red * 100 > green * 105)
        & (green * 100 > blue * 104)
        & (red - green < 105)
    )


def projection_scores(
    image: Image.Image,
    rect: Rect,
    axis: str,
    predicate,
) -> list[float]:
    pixels = image.load()
    if axis == "x":
        scores: list[float] = []
        for x in range(rect.left, rect.right):
            hits = 0
            for y in range(rect.top, rect.bottom):
                if predicate(pixels[x, y]):
                    hits += 1
            scores.append(hits / max(1, rect.height))
        return scores
    if axis == "y":
        scores = []
        for y in range(rect.top, rect.bottom):
            hits = 0
            for x in range(rect.left, rect.right):
                if predicate(pixels[x, y]):
                    hits += 1
            scores.append(hits / max(1, rect.width))
        return scores
    raise ValueError(f"unknown axis: {axis}")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    sorted_scores = sorted(scores)
    base = median(sorted_scores)
    strong = sorted_scores[int(len(sorted_scores) * 0.98)]
    scale = max(0.0001, strong - base)
    return [max(0.0, min(1.0, (score - base) / scale)) for score in scores]


def line_score_at(scores: list[float], center: int, radius: int) -> float:
    left = max(0, center - radius)
    right = min(len(scores), center + radius + 1)
    return max(scores[left:right]) if left < right else 0.0


def refine_position(scores: list[float], center: int, radius: int) -> int:
    left = max(0, center - radius)
    right = min(len(scores), center + radius + 1)
    if left >= right:
        return center
    return max(range(left, right), key=lambda index: scores[index])


def detect_axis(scores: list[float]) -> AxisDetection | None:
    length = len(scores)
    if length < 100:
        return None
    normalized = normalize_scores(smooth(scores, radius=1))
    min_step = max(8, int(length * 0.075))
    max_step = max(min_step, int(length * 0.135))
    return detect_axis_with_step_range(normalized, min_step, max_step)


def detect_axis_with_step_range(
    normalized_scores: list[float],
    min_step: int,
    max_step: int,
) -> AxisDetection | None:
    length = len(normalized_scores)
    best: AxisDetection | None = None

    for step in range(min_step, max_step + 1):
        span = step * GRID_INTERVALS
        if span >= length:
            continue
        stride = max(1, step // 10)
        line_radius = max(2, step // 28)
        for start in range(0, length - span, stride):
            raw_positions = [start + index * step for index in range(LINE_COUNT)]
            score = sum(line_score_at(normalized_scores, position, line_radius) for position in raw_positions) / LINE_COUNT
            edge_bonus = (
                line_score_at(normalized_scores, raw_positions[0], line_radius)
                + line_score_at(normalized_scores, raw_positions[-1], line_radius)
            ) / 2
            confidence = max(0.0, min(1.0, score * 0.78 + edge_bonus * 0.22))
            if best is None or confidence > best.confidence:
                refined = [refine_position(normalized_scores, position, line_radius) for position in raw_positions]
                best = AxisDetection(start=start, step=float(step), positions=refined, confidence=confidence)

    return best


def build_grid_detection(
    board_rect: Rect,
    x_scores: list[float],
    y_scores: list[float],
    method: str,
) -> GridDetection | None:
    vertical = detect_axis(x_scores)
    horizontal = detect_axis(y_scores)
    if vertical is None or horizontal is None:
        return None
    confidence = max(0.0, min(1.0, vertical.confidence * horizontal.confidence))
    return GridDetection(
        board_rect=board_rect,
        vertical=AxisDetection(
            start=vertical.start + board_rect.left,
            step=vertical.step,
            positions=[position + board_rect.left for position in vertical.positions],
            confidence=vertical.confidence,
        ),
        horizontal=AxisDetection(
            start=horizontal.start + board_rect.top,
            step=horizontal.step,
            positions=[position + board_rect.top for position in horizontal.positions],
            confidence=horizontal.confidence,
        ),
        confidence=confidence,
        method=method,
    )


def detect_grid(image: Image.Image) -> GridDetection | None:
    board_rect = detect_board_area(image)
    if board_rect is None:
        return detect_grid_from_dark_lines(image)

    candidates = []
    for index, candidate_rect in enumerate(grid_candidate_rects(image, board_rect)):
        suffix = "" if index == 0 else "_expanded"
        candidates.extend(
            [
                build_grid_detection(
                    candidate_rect,
                    dark_projection_scores(image, candidate_rect, "x"),
                    dark_projection_scores(image, candidate_rect, "y"),
                    method=f"dark{suffix}",
                ),
                build_grid_detection(
                    candidate_rect,
                    brown_projection_scores(image, candidate_rect, "x"),
                    brown_projection_scores(image, candidate_rect, "y"),
                    method=f"brown{suffix}",
                ),
            ],
        )
    best = max(
        (candidate for candidate in candidates if candidate is not None),
        key=lambda detection: detection.confidence,
        default=None,
    )
    repaired = repair_grid_from_dark_lines(image, best)
    if best is None:
        return repaired
    if grid_span_ratio(best) < 0.78 or (repaired is not None and repaired.confidence > best.confidence + 0.18):
        best = repaired or best
    if best.confidence < 0.45 and not is_usable_low_confidence_grid(best, image):
        return None
    return best


def grid_candidate_rects(image: Image.Image, board_rect: Rect) -> list[Rect]:
    pad = max(4, round(min(board_rect.width, board_rect.height) * 0.01))
    expanded = Rect(
        left=max(0, board_rect.left - pad),
        top=max(0, board_rect.top - pad),
        right=min(image.width, board_rect.right + pad),
        bottom=min(image.height, board_rect.bottom + pad),
    )
    if expanded == board_rect:
        return [board_rect]
    return [board_rect, expanded]


def is_usable_low_confidence_grid(detection: GridDetection, image: Image.Image) -> bool:
    """Accept clear board-grid candidates whose score is depressed by textured boards.

    Some Shogi Wars boards have strong gradients and piece shadows. The board
    area is still clean, but horizontal projection confidence can fall just
    below the generic 0.45 cutoff. Keep this narrow so unrelated UI lines do
    not become boards.
    """

    grid_width = abs(detection.vertical.positions[-1] - detection.vertical.positions[0])
    grid_height = abs(detection.horizontal.positions[-1] - detection.horizontal.positions[0])
    if grid_width < image.width * 0.60:
        return False
    if grid_height < image.height * 0.28:
        return False
    span = grid_height / max(1, grid_width)
    if not 0.88 <= span <= 1.18:
        return False
    if detection.vertical.confidence < 0.74 or detection.horizontal.confidence < 0.47:
        return False
    return detection.confidence >= 0.38


def grid_span_ratio(detection: GridDetection) -> float:
    x_span = max(1, detection.vertical.positions[-1] - detection.vertical.positions[0])
    y_span = max(1, detection.horizontal.positions[-1] - detection.horizontal.positions[0])
    return y_span / x_span


def repair_grid_from_dark_lines(
    image: Image.Image,
    detection: GridDetection | None,
) -> GridDetection | None:
    if detection is None:
        return detect_grid_from_dark_lines(image)
    ratio = grid_span_ratio(detection)
    if 0.78 <= ratio <= 1.35:
        return detection
    horizontal = detect_horizontal_axis_from_dark_lines(image, detection.vertical)
    if horizontal is None:
        return detection
    return grid_detection_from_axes(detection.vertical, horizontal, "dark_repaired")


def detect_grid_from_dark_lines(image: Image.Image) -> GridDetection | None:
    full_rect = Rect(left=0, top=0, right=image.width, bottom=image.height)
    x_scores = normalize_scores(smooth(dark_projection_scores(image, full_rect, "x"), radius=1))
    vertical = detect_axis_with_step_range(
        x_scores,
        min_step=max(8, int(image.width * 0.055)),
        max_step=max(8, int(image.width * 0.145)),
    )
    if vertical is None:
        return None
    horizontal = detect_horizontal_axis_from_dark_lines(image, vertical)
    if horizontal is None:
        return None
    detection = grid_detection_from_axes(vertical, horizontal, "dark_global")
    return detection if detection.confidence >= 0.45 else None


def detect_horizontal_axis_from_dark_lines(
    image: Image.Image,
    vertical: AxisDetection,
) -> AxisDetection | None:
    left = max(0, min(vertical.positions[0], vertical.positions[-1]))
    right = min(image.width, max(vertical.positions[0], vertical.positions[-1]))
    if right - left < image.width * 0.45:
        return None
    rect = Rect(left=left, top=0, right=right, bottom=image.height)
    y_scores = normalize_scores(smooth(dark_projection_scores(image, rect, "y"), radius=1))
    expected_step = max(8, vertical.step)
    return detect_axis_with_step_range(
        y_scores,
        min_step=max(8, int(expected_step * 0.82)),
        max_step=max(8, int(expected_step * 1.28)),
    )


def grid_detection_from_axes(
    vertical: AxisDetection,
    horizontal: AxisDetection,
    method: str,
) -> GridDetection:
    board_rect = Rect(
        left=min(vertical.positions[0], vertical.positions[-1]),
        top=min(horizontal.positions[0], horizontal.positions[-1]),
        right=max(vertical.positions[0], vertical.positions[-1]),
        bottom=max(horizontal.positions[0], horizontal.positions[-1]),
    )
    confidence = max(0.0, min(1.0, vertical.confidence * horizontal.confidence))
    return GridDetection(
        board_rect=board_rect,
        vertical=vertical,
        horizontal=horizontal,
        confidence=confidence,
        method=method,
    )


def draw_overlay(image: Image.Image, detection: GridDetection | None) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    if detection is None:
        draw.rectangle((0, 0, overlay.width - 1, overlay.height - 1), outline=(255, 0, 0), width=8)
        draw.text((24, 24), "grid detection failed", fill=(255, 0, 0))
        return overlay

    rect = detection.board_rect
    draw.rectangle((rect.left, rect.top, rect.right, rect.bottom), outline=(255, 0, 0), width=4)
    for x in detection.vertical.positions:
        draw.line((x, rect.top, x, rect.bottom), fill=(0, 210, 255), width=3)
    for y in detection.horizontal.positions:
        draw.line((rect.left, y, rect.right, y), fill=(0, 255, 120), width=3)
    draw.text(
        (max(0, rect.left + 8), max(0, rect.top - 32)),
        f"grid confidence={detection.confidence:.3f} method={detection.method}",
        fill=(255, 0, 0),
    )
    return overlay


def save_cells(
    image: Image.Image,
    detection: GridDetection,
    out_dir: Path,
    pad_x_ratio: float = 0.0,
    pad_y_ratio: float = 0.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = detection.vertical.positions
    ys = detection.horizontal.positions
    for row in range(9):
        for col in range(9):
            left = min(xs[col], xs[col + 1])
            right = max(xs[col], xs[col + 1])
            top = min(ys[row], ys[row + 1])
            bottom = max(ys[row], ys[row + 1])
            pad_x = round((right - left) * pad_x_ratio)
            pad_y = round((bottom - top) * pad_y_ratio)
            left = max(0, left - pad_x)
            right = min(image.width, right + pad_x)
            top = max(0, top - pad_y)
            bottom = min(image.height, bottom + pad_y)
            image.crop((left, top, right, bottom)).save(out_dir / f"r{row + 1:02d}_c{col + 1:02d}.png")


def report_dict(image: Image.Image, detection: GridDetection | None) -> dict:
    if detection is None:
        return {"image_size": list(image.size), "detected": False}
    rect = detection.board_rect
    grid_rect = Rect(
        left=detection.vertical.positions[0],
        top=detection.horizontal.positions[0],
        right=detection.vertical.positions[-1],
        bottom=detection.horizontal.positions[-1],
    )
    max_left = max(1, image.width - grid_rect.width)
    max_top = max(1, image.height - grid_rect.height)
    return {
        "image_size": list(image.size),
        "detected": True,
        "confidence": detection.confidence,
        "method": detection.method,
        "board_rect": {
            "left": rect.left,
            "top": rect.top,
            "right": rect.right,
            "bottom": rect.bottom,
            "width": rect.width,
            "height": rect.height,
        },
        "grid_rect": {
            "left": grid_rect.left,
            "top": grid_rect.top,
            "right": grid_rect.right,
            "bottom": grid_rect.bottom,
            "width": grid_rect.width,
            "height": grid_rect.height,
        },
        "cropSelection": {
            "leftRatio": grid_rect.left / max_left,
            "topRatio": grid_rect.top / max_top,
            "widthRatio": grid_rect.width / image.width,
            "heightRatio": grid_rect.height / image.height,
        },
        "vertical_lines": detection.vertical.positions,
        "horizontal_lines": detection.horizontal.positions,
        "vertical_confidence": detection.vertical.confidence,
        "horizontal_confidence": detection.horizontal.confidence,
    }


def process_image(image_path: Path, out_root: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    detection = detect_grid(image)
    image_out = out_root / image_path.stem
    image_out.mkdir(parents=True, exist_ok=True)
    draw_overlay(image, detection).save(image_out / "grid_overlay.png")
    if detection is not None:
        rect = detection.board_rect
        image.crop((rect.left, rect.top, rect.right, rect.bottom)).save(image_out / "board_area.png")
        grid_rect = (
            detection.vertical.positions[0],
            detection.horizontal.positions[0],
            detection.vertical.positions[-1],
            detection.horizontal.positions[-1],
        )
        image.crop(grid_rect).save(image_out / "grid_area.png")
        save_cells(image, detection, image_out / "cells")
    with (image_out / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_dict(image, detection), handle, ensure_ascii=False, indent=2)
    status = "OK" if detection is not None else "NG"
    print(f"{status}: {image_path} -> {image_out}")


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(file for file in path.iterdir() if file.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a 9x9 shogi board grid and write debug images.")
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
        default=Path("tools/out/grid_debug"),
        help="Output directory for overlays, reports, and cell crops.",
    )
    args = parser.parse_args()

    images = iter_images(args.input)
    if not images:
        raise SystemExit(f"No input images found: {args.input}")
    for image_path in images:
        process_image(image_path, args.out)


if __name__ == "__main__":
    main()
