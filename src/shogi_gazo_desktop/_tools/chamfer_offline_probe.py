from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image


FEATURE_WIDTH = 48
FEATURE_HEIGHT = 56
FEATURE_SIZE = FEATURE_WIDTH * FEATURE_HEIGHT
DEFAULT_ANALYSIS_DIR = Path("tools/out/android_device_eval/android_eval_b5_29_clean_component_diagnostics_20260510")
DEFAULT_SCREENSHOTS_DIR = Path("tools/samples/screenshots_by_app_piece_style")
DEFAULT_TEMPLATE_ASSET = Path("app/src/main/assets/app_piece_templates.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def decode_hex_mask(value: str, expected_size: int = FEATURE_SIZE) -> list[bool]:
    output = [False] * expected_size
    bit_index = 0
    for char in value or "":
        try:
            nibble = int(char, 16)
        except ValueError:
            continue
        for shift in range(3, -1, -1):
            if bit_index >= expected_size:
                return output
            output[bit_index] = ((nibble >> shift) & 1) == 1
            bit_index += 1
    return output


def dilate(mask: list[bool], width: int = FEATURE_WIDTH, height: int = FEATURE_HEIGHT) -> list[bool]:
    output = mask.copy()
    for y in range(height):
        for x in range(width):
            if not mask[y * width + x]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        output[ny * width + nx] = True
    return output


def mask_bounds(mask: list[bool], width: int, height: int) -> tuple[int, int, int, int]:
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for index, ink in enumerate(mask):
        if not ink:
            continue
        x = index % width
        y = index // width
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if max_x < min_x or max_y < min_y:
        return 0, 0, width, height
    return min_x, min_y, max_x + 1, max_y + 1


def luma(pixel: tuple[int, int, int, int]) -> int:
    red, green, blue, _alpha = pixel
    return max(0, min(255, (red * 299 + green * 587 + blue * 114) // 1000))


def is_warm_piece_base(red: int, green: int, blue: int) -> bool:
    return red > 135 and green > 95 and blue < 175 and red >= green and green >= blue and red - blue > 35


def source_ink_stats(source_ink: list[bool], source_red: list[bool], width: int, height: int) -> dict[str, int]:
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    ink_pixels = 0
    red_pixels = 0
    for index, ink in enumerate(source_ink):
        if not ink:
            continue
        x = index % width
        y = index // width
        ink_pixels += 1
        if source_red[index]:
            red_pixels += 1
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "ink_pixels": ink_pixels,
        "red_pixels": red_pixels,
    }


def suppress_edge_artifacts(source_ink: list[bool], source_red: list[bool], width: int, height: int) -> None:
    total_ink = sum(source_ink)
    if total_ink <= 0:
        return
    visited = [False] * len(source_ink)
    edge = max(1, round(min(width, height) * 0.025))
    for start, ink in enumerate(source_ink):
        if not ink or visited[start]:
            continue
        component: list[int] = []
        red_pixels = 0
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        touches_edge = False
        visited[start] = True
        queue = [start]
        while queue:
            index = queue.pop()
            component.append(index)
            if source_red[index]:
                red_pixels += 1
            x = index % width
            y = index // width
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            if x <= edge or y <= edge or x >= width - 1 - edge or y >= height - 1 - edge:
                touches_edge = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    nxt = ny * width + nx
                    if not source_ink[nxt] or visited[nxt]:
                        continue
                    visited[nxt] = True
                    queue.append(nxt)
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        red_dominant = red_pixels >= max(4, round(len(component) * 0.35))
        long_border_line = (
            touches_edge
            and len(component) <= max(4, total_ink // 4)
            and (component_width <= 3 or component_height <= 3)
            and (component_width >= width / 4 or component_height >= height / 4)
        )
        small_edge_speck = touches_edge and len(component) <= max(10, total_ink // 18)
        red_edge_marker = (
            touches_edge
            and red_dominant
            and len(component) <= max(16, round(total_ink * 0.45))
            and (component_width >= width * 0.07 or component_height >= height * 0.07)
        )
        if long_border_line or small_edge_speck or red_edge_marker:
            for index in component:
                source_ink[index] = False
                source_red[index] = False


def clean_normalized_mask(mask: list[bool], width: int = FEATURE_WIDTH, height: int = FEATURE_HEIGHT) -> list[bool]:
    total_ink = sum(mask)
    if total_ink <= 0:
        return mask.copy()
    output = [False] * len(mask)
    visited = [False] * len(mask)
    min_area = max(2, round(total_ink * 0.003))
    for start, ink in enumerate(mask):
        if not ink or visited[start]:
            continue
        component: list[int] = []
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        touches_border = False
        visited[start] = True
        queue = [start]
        while queue:
            index = queue.pop()
            component.append(index)
            x = index % width
            y = index // width
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                touches_border = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    nxt = ny * width + nx
                    if not mask[nxt] or visited[nxt]:
                        continue
                    visited[nxt] = True
                    queue.append(nxt)
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        long_border_line = (
            touches_border
            and len(component) <= max(3, total_ink // 5)
            and (component_width <= 2 or component_height <= 2)
            and (component_width >= width / 3 or component_height >= height / 3)
        )
        if len(component) >= min_area and not long_border_line:
            for index in component:
                output[index] = True
    return output if any(output) else mask.copy()


def extract_piece_masks(image: Image.Image) -> dict[str, Any]:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    data = rgba.tobytes()
    pixels = [tuple(data[index : index + 4]) for index in range(0, len(data), 4)]
    lumas: list[int] = []
    visible_pixels = 0
    for pixel in pixels:
        if pixel[3] >= 12:
            visible_pixels += 1
            lumas.append(luma(pixel))
    if not lumas or width <= 0 or height <= 0:
        return empty_source_features()
    lumas.sort()
    transparent_template_like = visible_pixels < width * height * 0.72
    background = lumas[min(len(lumas) - 1, max(0, int(len(lumas) * 0.86)))]
    foreground = lumas[min(len(lumas) - 1, max(0, int(len(lumas) * 0.12)))]
    contrast = max(background - foreground, 35)
    dark_threshold = background - max(30, round(contrast * 0.22))
    border_x = 0 if transparent_template_like else max(1, round(width * 0.035))
    border_y = 0 if transparent_template_like else max(1, round(height * 0.025))
    source_ink = [False] * (width * height)
    source_red = [False] * (width * height)
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    ink_pixels = 0
    red_pixels = 0

    for y in range(height):
        for x in range(width):
            if not transparent_template_like and (x < border_x or x >= width - border_x or y < border_y or y >= height - border_y):
                continue
            index = y * width + x
            red, green, blue, alpha = pixels[index]
            if alpha < 12:
                continue
            pixel_luma = luma(pixels[index])
            red_ink = (
                red > green + 40
                and red > blue + 48
                and green < red * 0.78
                and blue < red * 0.68
                and red > 90
                and pixel_luma < 210
            )
            if transparent_template_like:
                dark_ink = pixel_luma < 170 or (24 <= alpha <= 220 and pixel_luma < 240 and not is_warm_piece_base(red, green, blue))
            else:
                dark_ink = pixel_luma < dark_threshold and pixel_luma < 190
            if not red_ink and not dark_ink:
                continue
            source_ink[index] = True
            source_red[index] = red_ink
            ink_pixels += 1
            red_pixels += 1 if red_ink else 0
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if ink_pixels == 0:
        return empty_source_features()
    suppress_edge_artifacts(source_ink, source_red, width, height)
    stats = source_ink_stats(source_ink, source_red, width, height)
    if stats["ink_pixels"] == 0:
        return empty_source_features()
    min_x = stats["min_x"]
    min_y = stats["min_y"]
    max_x = stats["max_x"]
    max_y = stats["max_y"]
    ink_pixels = stats["ink_pixels"]
    red_pixels = stats["red_pixels"]

    normalized_ink = [False] * FEATURE_SIZE
    normalized_red = [False] * FEATURE_SIZE
    bbox_width = max(max_x - min_x + 1, 1)
    bbox_height = max(max_y - min_y + 1, 1)
    scale = min((FEATURE_WIDTH - 1) / max(bbox_width, 1), (FEATURE_HEIGHT - 1) / max(bbox_height, 1))
    target_width = max((max(bbox_width - 1, 0)) * scale, 0.0)
    target_height = max((max(bbox_height - 1, 0)) * scale, 0.0)
    offset_x = ((FEATURE_WIDTH - 1) - target_width) / 2.0
    offset_y = ((FEATURE_HEIGHT - 1) - target_height) / 2.0
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            source_index = y * width + x
            if not source_ink[source_index]:
                continue
            target_x = min(FEATURE_WIDTH - 1, max(0, round(offset_x + (x - min_x) * scale)))
            target_y = min(FEATURE_HEIGHT - 1, max(0, round(offset_y + (y - min_y) * scale)))
            target_index = target_y * FEATURE_WIDTH + target_x
            normalized_ink[target_index] = True
            if source_red[source_index]:
                normalized_red[target_index] = True
    clean = clean_normalized_mask(normalized_ink)
    ink_mask = dilate(normalized_ink)
    clean_mask = dilate(clean)
    return {
        "ink_mask": ink_mask,
        "clean_mask": clean_mask,
        "red_mask": normalized_red,
        "ink_count": sum(ink_mask),
        "clean_ink_count": sum(clean_mask),
        "original_ink_ratio": ink_pixels / max(1, width * height),
        "red_share": red_pixels / max(1, ink_pixels),
        "bbox_width_ratio": bbox_width / max(1, width),
        "bbox_height_ratio": bbox_height / max(1, height),
        "bbox_center_x": (min_x + max_x + 1) / (2 * max(1, width)),
        "bbox_center_y": (min_y + max_y + 1) / (2 * max(1, height)),
    }


def empty_source_features() -> dict[str, Any]:
    empty = [False] * FEATURE_SIZE
    return {
        "ink_mask": empty,
        "clean_mask": empty,
        "red_mask": empty,
        "ink_count": 0,
        "clean_ink_count": 0,
        "original_ink_ratio": 0.0,
        "red_share": 0.0,
        "bbox_width_ratio": 0.0,
        "bbox_height_ratio": 0.0,
        "bbox_center_x": 0.5,
        "bbox_center_y": 0.5,
    }


def distance_map(mask: list[bool], width: int = FEATURE_WIDTH, height: int = FEATURE_HEIGHT) -> list[float]:
    distances = [math.inf] * (width * height)
    heap: list[tuple[float, int]] = []
    for index, ink in enumerate(mask):
        if ink:
            distances[index] = 0.0
            heap.append((0.0, index))
    if not heap:
        return distances
    heapq.heapify(heap)
    neighbors = [
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    ]
    while heap:
        distance, index = heapq.heappop(heap)
        if distance != distances[index]:
            continue
        x = index % width
        y = index // width
        for dx, dy, weight in neighbors:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            nxt = ny * width + nx
            new_distance = distance + weight
            if new_distance < distances[nxt]:
                distances[nxt] = new_distance
                heapq.heappush(heap, (new_distance, nxt))
    return distances


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def directed_distance_stats(points_mask: list[bool], target_distances: list[float], coverage_distance: float) -> dict[str, float]:
    distances = [target_distances[index] for index, ink in enumerate(points_mask) if ink and math.isfinite(target_distances[index])]
    if not distances:
        return {
            "mean": math.inf,
            "p50": math.inf,
            "p90": math.inf,
            "p95": math.inf,
            "max": math.inf,
            "coverage": 0.0,
            "count": 0,
        }
    return {
        "mean": mean(distances),
        "p50": percentile(distances, 0.50),
        "p90": percentile(distances, 0.90),
        "p95": percentile(distances, 0.95),
        "max": max(distances),
        "coverage": sum(1 for distance in distances if distance <= coverage_distance) / len(distances),
        "count": len(distances),
    }


def chamfer_metrics(source_mask: list[bool], template_mask: list[bool], coverage_distance: float = 1.5) -> dict[str, float]:
    source_distances = distance_map(source_mask)
    template_distances = distance_map(template_mask)
    source_to_template = directed_distance_stats(source_mask, template_distances, coverage_distance)
    template_to_source = directed_distance_stats(template_mask, source_distances, coverage_distance)
    symmetric_mean = (source_to_template["mean"] + template_to_source["mean"]) / 2.0
    symmetric_p90 = (source_to_template["p90"] + template_to_source["p90"]) / 2.0
    symmetric_coverage = (source_to_template["coverage"] + template_to_source["coverage"]) / 2.0
    chamfer_score = 1.0 / (1.0 + symmetric_mean) if math.isfinite(symmetric_mean) else 0.0
    return {
        "source_to_template_mean": source_to_template["mean"],
        "source_to_template_p50": source_to_template["p50"],
        "source_to_template_p90": source_to_template["p90"],
        "source_to_template_p95": source_to_template["p95"],
        "source_to_template_max": source_to_template["max"],
        "source_to_template_coverage": source_to_template["coverage"],
        "source_mask_point_count": source_to_template["count"],
        "template_to_source_mean": template_to_source["mean"],
        "template_to_source_p50": template_to_source["p50"],
        "template_to_source_p90": template_to_source["p90"],
        "template_to_source_p95": template_to_source["p95"],
        "template_to_source_max": template_to_source["max"],
        "template_to_source_coverage": template_to_source["coverage"],
        "template_mask_point_count": template_to_source["count"],
        "symmetric_chamfer_mean": symmetric_mean,
        "symmetric_chamfer_p90": symmetric_p90,
        "symmetric_chamfer_coverage": symmetric_coverage,
        "chamfer_score": chamfer_score,
    }


def template_source_base(source_label: str) -> str:
    text = source_label or ""
    if text.startswith("app_template:"):
        text = text[len("app_template:") :]
    return text.split("+", 1)[0]


def identity_parts(identity: str) -> tuple[str, str]:
    color, _separator, piece = (identity or "").partition(":")
    return color.upper(), piece


def load_templates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    templates: list[dict[str, Any]] = []
    for item in data.get("templates") or []:
        if not isinstance(item, dict):
            continue
        mask = decode_hex_mask(str(item.get("mask") or ""))
        dilated = dilate(mask)
        bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else None
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [parse_int(value) for value in bbox[:4]]
        else:
            x1, y1, x2, y2 = mask_bounds(dilated, FEATURE_WIDTH, FEATURE_HEIGHT)
        templates.append(
            {
                "preset": item.get("preset"),
                "color": item.get("color"),
                "piece": item.get("piece"),
                "source": item.get("source"),
                "row": item.get("row"),
                "col": item.get("col"),
                "mask": dilated,
                "raw_mask": mask,
                "ink_count": sum(dilated),
                "bbox_width": (x2 - x1) / FEATURE_WIDTH,
                "bbox_height": (y2 - y1) / FEATURE_HEIGHT,
                "bbox_center_x": (x1 + x2) / (2 * FEATURE_WIDTH),
                "bbox_center_y": (y1 + y2) / (2 * FEATURE_HEIGHT),
            }
        )
    return templates


def choose_template(row: dict[str, str], templates: list[dict[str, Any]], source_features: dict[str, Any]) -> tuple[dict[str, Any] | None, str, int]:
    color, piece = identity_parts(row.get("candidate_identity", ""))
    source_base = template_source_base(row.get("source", ""))
    preset = "piyo_chick" if row.get("piece_style") == "ひよこ駒" and row.get("app") == "ぴよ将棋" else None
    matches = [
        template
        for template in templates
        if template.get("color") == color
        and template.get("piece") == piece
        and (preset is None or template.get("preset") == preset)
        and template.get("source") == source_base
    ]
    if not matches:
        matches = [
            template
            for template in templates
            if template.get("color") == color
            and template.get("piece") == piece
            and (preset is None or template.get("preset") == preset)
        ]
    if not matches:
        return None, "missing_template", 0

    target_ink_count = parse_int(row.get("target_ink_count"), -1)
    target_bbox_width = parse_float(row.get("target_bbox_width"), -1.0)
    target_bbox_height = parse_float(row.get("target_bbox_height"), -1.0)
    target_bbox_center_x = parse_float(row.get("target_bbox_center_x"), -1.0)
    target_bbox_center_y = parse_float(row.get("target_bbox_center_y"), -1.0)

    def feature_delta(template: dict[str, Any]) -> float:
        return (
            abs(template["ink_count"] - target_ink_count) / 1000.0
            + abs(template["bbox_width"] - target_bbox_width)
            + abs(template["bbox_height"] - target_bbox_height)
            + abs(template["bbox_center_x"] - target_bbox_center_x)
            + abs(template["bbox_center_y"] - target_bbox_center_y)
        )

    best_delta = min(feature_delta(template) for template in matches)
    close = [template for template in matches if feature_delta(template) <= best_delta + 1e-6]
    if len(close) == 1:
        return close[0], "feature_match", len(matches)

    source_mask = source_features.get("clean_mask") or source_features.get("ink_mask") or [False] * FEATURE_SIZE
    scored = [(chamfer_metrics(source_mask, template["mask"])["symmetric_chamfer_mean"], template) for template in close]
    scored.sort(key=lambda pair: pair[0])
    return scored[0][1], "feature_then_chamfer_match", len(matches)


def load_reports_by_sample(analysis_dir: Path) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for report in analysis_dir.glob("*/*/*/piece_report.json"):
        reports[report.parent.name] = report
    return reports


def load_report_cells(report_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in data.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        try:
            cells[(int(cell.get("row")), int(cell.get("col")))] = cell
        except (TypeError, ValueError):
            continue
    return cells


def crop_cell_image(screenshots_dir: Path, row: dict[str, str], cell: dict[str, Any]) -> Image.Image | None:
    rect = ((cell.get("debug") or {}).get("recognition_rect") or {}) if isinstance(cell.get("debug"), dict) else {}
    if not rect:
        return None
    screenshot = screenshots_dir / str(row.get("app")) / str(row.get("piece_style")) / f"{row.get('sample')}.png"
    if not screenshot.exists():
        return None
    image = Image.open(screenshot).convert("RGBA")
    width, height = image.size
    left = round(parse_float(rect.get("left_ratio")) * width)
    top = round(parse_float(rect.get("top_ratio")) * height)
    crop_width = max(1, round(parse_float(rect.get("width_ratio")) * width))
    crop_height = max(1, round(parse_float(rect.get("height_ratio")) * height))
    left = min(max(0, left), width - 1)
    top = min(max(0, top), height - 1)
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


CELL_KEY_FIELDS = ("app", "piece_style", "sample", "square", "row", "col")


def cell_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in CELL_KEY_FIELDS)


def round_float(value: float, digits: int = 4) -> float | str:
    if not math.isfinite(value):
        return ""
    return round(value, digits)


def run_probe(
    analysis_dir: Path,
    screenshots_dir: Path,
    template_asset: Path,
    out_dir: Path,
    top_n: int,
    virtual_chamfer_weight: float,
    coverage_distance: float,
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
    gaps_by_cell = {cell_key(row): row for row in read_csv(gap_path)}
    reports_by_sample = load_reports_by_sample(analysis_dir)
    templates = load_templates(template_asset)
    cells_cache: dict[Path, dict[tuple[int, int], dict[str, Any]]] = {}
    source_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    output_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = cell_key(row)
        if key not in source_cache:
            report_path = reports_by_sample.get(str(row.get("sample")))
            cell: dict[str, Any] | None = None
            if report_path is not None:
                if report_path not in cells_cache:
                    cells_cache[report_path] = load_report_cells(report_path)
                cell = cells_cache[report_path].get((parse_int(row.get("row")), parse_int(row.get("col"))))
            crop = crop_cell_image(screenshots_dir, row, cell or {}) if cell is not None else None
            source_cache[key] = extract_piece_masks(crop) if crop is not None else empty_source_features()
        source_features = source_cache[key]
        template, template_reason, template_match_count = choose_template(row, templates, source_features)
        base_score = parse_float(row.get("score"))
        if template is None:
            metrics = chamfer_metrics([False] * FEATURE_SIZE, [False] * FEATURE_SIZE, coverage_distance)
            template_info: dict[str, Any] = {}
        else:
            metrics = chamfer_metrics(source_features["clean_mask"], template["mask"], coverage_distance)
            template_info = template
        chamfer_score = metrics["chamfer_score"]
        virtual_score = (1.0 - virtual_chamfer_weight) * base_score + virtual_chamfer_weight * chamfer_score
        source_report_clean_count = parse_int(row.get("source_clean_ink_count"), 0)
        output_row = {
            **{field: row.get(field) for field in CELL_KEY_FIELDS},
            "expected": row.get("expected"),
            "predicted_top1": row.get("predicted_top1"),
            "candidate_index": row.get("candidate_index"),
            "candidate_rank": row.get("candidate_rank"),
            "candidate_identity": row.get("candidate_identity"),
            "is_expected": row.get("is_expected"),
            "is_predicted_top1": row.get("is_predicted_top1"),
            "base_score": row.get("score"),
            "source": row.get("source"),
            "template_source_base": template_source_base(row.get("source", "")),
            "template_match_reason": template_reason,
            "template_match_count": template_match_count,
            "selected_template_source": template_info.get("source", ""),
            "selected_template_row": template_info.get("row", ""),
            "selected_template_col": template_info.get("col", ""),
            "source_extracted_ink_count": source_features["ink_count"],
            "source_extracted_clean_ink_count": source_features["clean_ink_count"],
            "source_report_clean_ink_count": source_report_clean_count,
            "source_clean_ink_count_delta": source_features["clean_ink_count"] - source_report_clean_count,
            "template_ink_count": template_info.get("ink_count", ""),
            "target_report_ink_count": row.get("target_ink_count"),
            **{name: round_float(value) for name, value in metrics.items()},
            "virtual_chamfer_weight": virtual_chamfer_weight,
            "virtual_rerank_score": round_float(virtual_score),
        }
        output_rows.append(output_row)

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        grouped[cell_key(row)].append(row)

    cell_summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        for rank, row in enumerate(sorted(rows, key=lambda row: parse_float(row.get("chamfer_score"), -1.0), reverse=True), start=1):
            row["chamfer_rank"] = rank
        for rank, row in enumerate(sorted(rows, key=lambda row: parse_float(row.get("virtual_rerank_score"), -1.0), reverse=True), start=1):
            row["virtual_rerank_rank"] = rank
        base_top = min(rows, key=lambda row: parse_int(row.get("candidate_rank"), 9999))
        chamfer_top = min(rows, key=lambda row: parse_int(row.get("chamfer_rank"), 9999))
        virtual_top = min(rows, key=lambda row: parse_int(row.get("virtual_rerank_rank"), 9999))
        expected = str(base_top.get("expected") or "")
        expected_row = next((row for row in rows if row.get("candidate_identity") == expected), None)
        gap = gaps_by_cell.get(key, {})
        cell_summary_rows.append(
            {
                **{field: base_top.get(field, "") for field in CELL_KEY_FIELDS},
                "expected": expected,
                "base_top1_identity": base_top.get("candidate_identity"),
                "base_top1_score": base_top.get("base_score"),
                "expected_base_rank": expected_row.get("candidate_rank") if expected_row else gap.get("expected_candidate_rank", ""),
                "expected_base_score": expected_row.get("base_score") if expected_row else "",
                "expected_chamfer_rank": expected_row.get("chamfer_rank") if expected_row else "",
                "expected_virtual_rank": expected_row.get("virtual_rerank_rank") if expected_row else "",
                "expected_chamfer_score": expected_row.get("chamfer_score") if expected_row else "",
                "expected_virtual_score": expected_row.get("virtual_rerank_score") if expected_row else "",
                "chamfer_top1_identity": chamfer_top.get("candidate_identity"),
                "chamfer_top1_score": chamfer_top.get("chamfer_score"),
                "virtual_top1_identity": virtual_top.get("candidate_identity"),
                "virtual_top1_score": virtual_top.get("virtual_rerank_score"),
                "virtual_top1_is_expected": virtual_top.get("candidate_identity") == expected,
                "chamfer_top1_is_expected": chamfer_top.get("candidate_identity") == expected,
                "expected_in_topn": expected_row is not None,
                "topn": len(rows),
            }
        )

    base_error_count = len(cell_summary_rows)
    virtual_error_count = sum(1 for row in cell_summary_rows if not row["virtual_top1_is_expected"])
    chamfer_error_count = sum(1 for row in cell_summary_rows if not row["chamfer_top1_is_expected"])
    fixed_by_virtual = sum(1 for row in cell_summary_rows if row["virtual_top1_is_expected"])
    fixed_by_chamfer = sum(1 for row in cell_summary_rows if row["chamfer_top1_is_expected"])
    expected_base_ranks = [parse_float(row.get("expected_base_rank"), math.nan) for row in cell_summary_rows]
    expected_virtual_ranks = [parse_float(row.get("expected_virtual_rank"), math.nan) for row in cell_summary_rows]
    expected_chamfer_ranks = [parse_float(row.get("expected_chamfer_rank"), math.nan) for row in cell_summary_rows]
    expected_base_ranks = [value for value in expected_base_ranks if math.isfinite(value)]
    expected_virtual_ranks = [value for value in expected_virtual_ranks if math.isfinite(value)]
    expected_chamfer_ranks = [value for value in expected_chamfer_ranks if math.isfinite(value)]
    summary_rows = [
        {
            "analysis_dir": str(analysis_dir),
            "top_n": top_n,
            "virtual_chamfer_weight": virtual_chamfer_weight,
            "coverage_distance": coverage_distance,
            "target_error_cells": base_error_count,
            "expected_in_topn_count": sum(1 for row in cell_summary_rows if row["expected_in_topn"]),
            "base_errors_if_top1": base_error_count,
            "virtual_errors_if_top1": virtual_error_count,
            "chamfer_only_errors_if_top1": chamfer_error_count,
            "fixed_by_virtual_count": fixed_by_virtual,
            "fixed_by_chamfer_only_count": fixed_by_chamfer,
            "expected_base_rank_avg": round_float(mean(expected_base_ranks)) if expected_base_ranks else "",
            "expected_virtual_rank_avg": round_float(mean(expected_virtual_ranks)) if expected_virtual_ranks else "",
            "expected_chamfer_rank_avg": round_float(mean(expected_chamfer_ranks)) if expected_chamfer_ranks else "",
        }
    ]

    candidate_fields = [
        *CELL_KEY_FIELDS,
        "expected",
        "predicted_top1",
        "candidate_index",
        "candidate_rank",
        "candidate_identity",
        "is_expected",
        "is_predicted_top1",
        "base_score",
        "source",
        "template_source_base",
        "template_match_reason",
        "template_match_count",
        "selected_template_source",
        "selected_template_row",
        "selected_template_col",
        "source_extracted_ink_count",
        "source_extracted_clean_ink_count",
        "source_report_clean_ink_count",
        "source_clean_ink_count_delta",
        "template_ink_count",
        "target_report_ink_count",
        "source_to_template_mean",
        "source_to_template_p50",
        "source_to_template_p90",
        "source_to_template_p95",
        "source_to_template_max",
        "source_to_template_coverage",
        "source_mask_point_count",
        "template_to_source_mean",
        "template_to_source_p50",
        "template_to_source_p90",
        "template_to_source_p95",
        "template_to_source_max",
        "template_to_source_coverage",
        "template_mask_point_count",
        "symmetric_chamfer_mean",
        "symmetric_chamfer_p90",
        "symmetric_chamfer_coverage",
        "chamfer_score",
        "chamfer_rank",
        "virtual_chamfer_weight",
        "virtual_rerank_score",
        "virtual_rerank_rank",
    ]
    cell_summary_fields = [
        *CELL_KEY_FIELDS,
        "expected",
        "base_top1_identity",
        "base_top1_score",
        "expected_base_rank",
        "expected_base_score",
        "expected_chamfer_rank",
        "expected_virtual_rank",
        "expected_chamfer_score",
        "expected_virtual_score",
        "chamfer_top1_identity",
        "chamfer_top1_score",
        "virtual_top1_identity",
        "virtual_top1_score",
        "virtual_top1_is_expected",
        "chamfer_top1_is_expected",
        "expected_in_topn",
        "topn",
    ]
    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "piece_style_chamfer_candidates.csv", output_rows, candidate_fields)
    write_csv(out_dir / "piece_style_chamfer_cell_summary.csv", cell_summary_rows, cell_summary_fields)
    write_csv(out_dir / "piece_style_chamfer_summary.csv", summary_rows, summary_fields)
    return {
        "candidate_rows": len(output_rows),
        "cell_rows": len(cell_summary_rows),
        "summary": summary_rows[0],
        "out_dir": str(out_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline topN symmetric chamfer rerank probe for Android diagnostic candidates.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--template-assets", type=Path, default=DEFAULT_TEMPLATE_ASSET)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--virtual-chamfer-weight", type=float, default=0.50)
    parser.add_argument("--coverage-distance", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or (args.analysis_dir / "chamfer_offline_probe")
    result = run_probe(
        analysis_dir=args.analysis_dir,
        screenshots_dir=args.screenshots_dir,
        template_asset=args.template_assets,
        out_dir=out_dir,
        top_n=args.top_n,
        virtual_chamfer_weight=args.virtual_chamfer_weight,
        coverage_distance=args.coverage_distance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
