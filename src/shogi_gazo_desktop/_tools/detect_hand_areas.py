from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from detect_board_grid import GridDetection, detect_grid, iter_images, report_dict

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime.
    cv2 = None
    np = None


SIDE_OWNER = {
    "top": "white",
    "left": "white",
    "bottom": "black",
    "right": "black",
}


@dataclass(frozen=True)
class HandArea:
    owner: str
    side: str
    rect: list[int]
    confidence: float
    evidence: str
    components: list[list[int]]


def ensure_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("hand area detection requires numpy and opencv-python. Run: python -m pip install -r tools\\requirements.txt")


def grid_rect(detection: GridDetection) -> tuple[int, int, int, int]:
    return (
        detection.vertical.positions[0],
        detection.horizontal.positions[0],
        detection.vertical.positions[-1],
        detection.horizontal.positions[-1],
    )


def cell_size(detection: GridDetection) -> tuple[float, float]:
    left, top, right, bottom = grid_rect(detection)
    return (right - left) / 9.0, (bottom - top) / 9.0


def clip_rect(rect: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    left, top, right, bottom = rect
    clipped = (
        max(0, min(width, left)),
        max(0, min(height, top)),
        max(0, min(width, right)),
        max(0, min(height, bottom)),
    )
    if clipped[2] - clipped[0] < 8 or clipped[3] - clipped[1] < 8:
        return None
    return clipped


def expand_rect(
    rect: tuple[int, int, int, int],
    pad_x: int,
    pad_y: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = rect
    return clip_rect((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), image_size)


def expand_rect_asymmetric(
    rect: tuple[int, int, int, int],
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = rect
    return clip_rect((left - pad_left, top - pad_top, right + pad_right, bottom + pad_bottom), image_size)


def search_bands(image: Image.Image, detection: GridDetection) -> dict[str, tuple[int, int, int, int]]:
    left, top, right, bottom = grid_rect(detection)
    cell_w, cell_h = cell_size(detection)
    width, height = image.size
    raw = {
        "top": (
            int(left - cell_w * 0.25),
            int(top - cell_h * 1.10),
            int(right + cell_w * 0.25),
            int(top - 2),
        ),
        "bottom": (
            int(left - cell_w * 0.25),
            int(bottom + 2),
            int(right + cell_w * 0.25),
            int(bottom + cell_h * 1.10),
        ),
        "left": (
            int(left - cell_w * 1.55),
            int(top),
            int(left - 2),
            int(bottom + cell_h * 0.18),
        ),
        "right": (
            int(right + 2),
            int(top),
            int(right + cell_w * 1.55),
            int(bottom + cell_h * 0.18),
        ),
    }
    return {
        side: clipped
        for side, rect in raw.items()
        if (clipped := clip_rect(rect, (width, height))) is not None
    }


def board_gutter_areas(image: Image.Image, detection: GridDetection) -> dict[str, tuple[int, int, int, int]]:
    """Return full-width app hand strips when the detected board area includes them.

    Piyo-style screens use a wooden strip directly above and below the grid.
    The board-color detector sees those strips as part of the board rectangle,
    so they are better recovered from board_rect vs. grid_rect than from color
    components.
    """

    left, top, right, bottom = grid_rect(detection)
    cell_w, cell_h = cell_size(detection)
    board = detection.board_rect
    if board.width < image.width * 0.86:
        return {}

    gutters: dict[str, tuple[int, int, int, int]] = {}
    min_height = max(18, int(cell_h * 0.35))
    pad_x = int(cell_w * 0.25)
    if top - board.top >= min_height:
        rect = clip_rect((left - pad_x, max(board.top, int(top - cell_h * 1.10)), right + pad_x, top - 2), image.size)
        if rect is not None:
            gutters["top"] = rect
    if board.bottom - bottom >= min_height:
        rect = clip_rect((left - pad_x, bottom + 2, right + pad_x, min(board.bottom, int(bottom + cell_h * 1.10))), image.size)
        if rect is not None:
            gutters["bottom"] = rect
    return gutters


def layout_fallback_sides(image: Image.Image, detection: GridDetection) -> tuple[str, ...]:
    left, _, right, _ = grid_rect(detection)
    grid_width = right - left
    if grid_width >= image.width * 0.84:
        return ("top", "bottom")
    return ("left", "right")


def bounding_rect(rects: Iterable[list[int]]) -> tuple[int, int, int, int] | None:
    rect_list = list(rects)
    if not rect_list:
        return None
    return (
        min(rect[0] for rect in rect_list),
        min(rect[1] for rect in rect_list),
        max(rect[2] for rect in rect_list),
        max(rect[3] for rect in rect_list),
    )


def intersect_rect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    rect = (
        max(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        min(first[3], second[3]),
    )
    if rect[2] - rect[0] < 8 or rect[3] - rect[1] < 8:
        return None
    return rect


def piece_color_mask(rgb: object) -> object:
    ensure_opencv()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red, green, blue = cv2.split(rgb)
    red_i = red.astype("int16")
    green_i = green.astype("int16")
    blue_i = blue.astype("int16")
    mask = (
        (value > 105)
        & (saturation > 35)
        & (hue >= 4)
        & (hue <= 48)
        & (red_i > green_i - 30)
        & (green_i > blue_i + 12)
    )
    result = np.where(mask, 255, 0).astype("uint8")
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((3, 3), dtype="uint8"))
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((7, 7), dtype="uint8"))


def detect_piece_color_components(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    cell_w: float,
    cell_h: float,
) -> list[list[int]]:
    ensure_opencv()
    left, top, right, bottom = rect
    crop = np.array(image.crop(rect).convert("RGB"))
    mask = piece_color_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[list[int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area < max(120.0, cell_w * cell_h * 0.025):
            continue
        if width < cell_w * 0.20 or height < cell_h * 0.36:
            continue
        if width > cell_w * 1.35 or height > cell_h * 1.60:
            continue
        fill = area / max(1, width * height)
        if fill < 0.22:
            continue
        components.append([left + x, top + y, left + x + width, top + y + height])
    components.sort(key=lambda item: (item[1], item[0]))
    return components


def detect_hand_areas(image: Image.Image, detection: GridDetection | None) -> list[HandArea]:
    if detection is None:
        return []
    cell_w, cell_h = cell_size(detection)
    bands = search_bands(image, detection)
    gutters = board_gutter_areas(image, detection)
    areas: dict[str, HandArea] = {}

    for side, rect in gutters.items():
        areas[side] = HandArea(
            owner=SIDE_OWNER[side],
            side=side,
            rect=list(rect),
            confidence=0.88,
            evidence="board_gutter",
            components=[],
        )

    # Outer band search: for wide-board (top/bottom) layouts, search strictly
    # outside board_rect for hand pieces (handles tatami-style apps like Shoko where
    # the real hand strip is above board_rect.top / below board_rect.bottom).
    board = detection.board_rect
    left_g, top_g, right_g, bottom_g = grid_rect(detection)
    pad_x = int(cell_w * 0.25)
    if board is not None and board.width >= image.width * 0.84:
        outer_candidates: list[tuple[str, tuple[int, int, int, int]]] = []
        if board.top < top_g:
            outer_top = clip_rect(
                (int(left_g - pad_x), max(0, board.top - int(cell_h * 1.5)),
                 int(right_g + pad_x), board.top),
                image.size,
            )
            if outer_top is not None:
                outer_candidates.append(("top", outer_top))
        if board.bottom > bottom_g:
            outer_bottom = clip_rect(
                (int(left_g - pad_x), board.bottom,
                 int(right_g + pad_x), min(image.height, board.bottom + int(cell_h * 1.5))),
                image.size,
            )
            if outer_bottom is not None:
                outer_candidates.append(("bottom", outer_bottom))
        for side, outer_rect in outer_candidates:
            outer_components = detect_piece_color_components(image, outer_rect, cell_w, cell_h)
            existing = areas.get(side)
            if existing is None or existing.evidence == "board_gutter":
                if outer_components:
                    confidence = min(0.96, 0.62 + len(outer_components) * 0.10)
                    evidence = "piece_color_components_outer"
                else:
                    confidence = 0.72
                    evidence = "layout_outer"
                areas[side] = HandArea(
                    owner=SIDE_OWNER[side],
                    side=side,
                    rect=list(outer_rect),
                    confidence=round(confidence, 4),
                    evidence=evidence,
                    components=outer_components,
                )

    for side, rect in bands.items():
        if side in areas:
            continue
        components = detect_piece_color_components(image, rect, cell_w, cell_h)
        if not components:
            continue
        confidence = min(0.96, 0.62 + len(components) * 0.10)
        area_rect = rect
        areas[side] = HandArea(
            owner=SIDE_OWNER[side],
            side=side,
            rect=list(area_rect),
            confidence=round(confidence, 4),
            evidence="piece_color_components",
            components=components,
        )

    for side in layout_fallback_sides(image, detection):
        if side in areas:
            continue
        rect = bands.get(side)
        if rect is None:
            continue
        areas[side] = HandArea(
            owner=SIDE_OWNER[side],
            side=side,
            rect=list(rect),
            confidence=0.36,
            evidence="layout_fallback",
            components=[],
        )

    return [areas[side] for side in ("top", "bottom", "left", "right") if side in areas]


def report_for_image(image_path: Path, image: Image.Image, detection: GridDetection | None, areas: list[HandArea]) -> dict:
    grid_report = report_dict(image, detection)
    return {
        "image": str(image_path),
        "image_size": list(image.size),
        "grid": grid_report,
        "grid_detected": detection is not None,
        "summary": {
            "area_count": len(areas),
            "sides": [area.side for area in areas],
        },
        "areas": [asdict(area) for area in areas],
    }


def draw_hand_area_overlay(
    image: Image.Image,
    detection: GridDetection | None,
    areas: Iterable[HandArea],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    if detection is None:
        draw.rectangle((0, 0, overlay.width - 1, overlay.height - 1), outline=(255, 0, 0), width=8)
        draw.text((24, 24), "grid detection failed", fill=(255, 0, 0))
        return overlay

    grid = grid_rect(detection)
    draw.rectangle(grid, outline=(0, 210, 255), width=4)
    bands = search_bands(image, detection)
    for side, rect in bands.items():
        draw.rectangle(rect, outline=(120, 120, 120), width=2)
        draw.text((rect[0] + 4, rect[1] + 4), f"{side} search", fill=(120, 120, 120))

    colors = {
        "black": (0, 210, 80),
        "white": (245, 80, 80),
    }
    for area in areas:
        color = colors.get(area.owner, (255, 210, 0))
        rect = tuple(area.rect)
        draw.rectangle(rect, outline=color, width=5)
        draw.text(
            (rect[0] + 6, max(0, rect[1] - 24)),
            f"{area.owner}:{area.side} {area.confidence:.2f} {area.evidence}",
            fill=color,
        )
        for component in area.components:
            draw.rectangle(tuple(component), outline=(255, 235, 0), width=3)
    return overlay


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_outputs(
    image: Image.Image,
    detection: GridDetection | None,
    report: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    areas = [HandArea(**area) for area in report["areas"]]
    write_json(out_dir / "hand_area_report.json", report)
    draw_hand_area_overlay(image, detection, areas).save(out_dir / "hand_area_overlay.png")
    areas_dir = out_dir / "areas"
    areas_dir.mkdir(parents=True, exist_ok=True)
    for area in areas:
        image.crop(tuple(area.rect)).save(areas_dir / f"{area.owner}_{area.side}.png")


def process_image(image_path: Path, out_root: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    detection = detect_grid(image)
    areas = detect_hand_areas(image, detection)
    report = report_for_image(image_path, image, detection, areas)
    image_out = out_root / image_path.stem
    write_outputs(image, detection, report, image_out)
    status = "OK" if detection is not None else "NG"
    print(f"{status}: {image_path} -> {image_out} ({len(areas)} hand area(s))")
    return report


def process_images(input_path: Path, out_root: Path) -> list[dict]:
    return [process_image(image_path, out_root) for image_path in iter_images(input_path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect captured-piece hand areas around a detected shogi board.")
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
        default=Path("tools/out/hand_areas"),
        help="Output directory for hand area reports and overlays.",
    )
    args = parser.parse_args()

    reports = process_images(args.input, args.out)
    ok_count = sum(1 for report in reports if report["grid_detected"])
    area_count = sum(len(report["areas"]) for report in reports)
    print(f"OK: analyzed {len(reports)} image(s), grid detected for {ok_count}, hand areas={area_count}. Output: {args.out}")


if __name__ == "__main__":
    main()
