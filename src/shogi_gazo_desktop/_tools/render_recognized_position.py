from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NORMAL_PIECES = ["OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU"]
PROMOTED_PIECES = ["OU", "RY", "UM", None, "NG", "NK", "NY", "TO"]
BOARD_SIZE = 1080
GRID_LINE_COLOR = (119, 74, 30)
GRID_BORDER_COLOR = (95, 50, 20)


@dataclass(frozen=True)
class RenderedOutputs:
    confirmed_path: Path
    candidates_path: Path
    comparison_path: Path | None


def load_report(path: Path) -> dict:
    if path.is_dir():
        for name in ("analysis_report.json", "piece_report.json", "recognized_board.json"):
            candidate = path / name
            if candidate.exists():
                return load_report(candidate)
        raise FileNotFoundError(f"no report json found in: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "piece_recognition" in data:
        return data
    if "cells" in data:
        return {"piece_recognition": data}
    if "rows" in data:
        cells = []
        for row_index, row in enumerate(data["rows"], start=1):
            for col_index, cell in enumerate(row, start=1):
                cells.append({"row": row_index, "col": col_index, **cell, "candidates": []})
        return {"piece_recognition": {"cells": cells, "summary": data.get("summary", {})}}
    raise ValueError(f"unsupported report format: {path}")


def render_outputs(
    report_path: Path,
    out_dir: Path | None = None,
    board_path: Path | None = None,
    pieces_path: Path | None = None,
    screenshot_path: Path | None = None,
) -> RenderedOutputs:
    report = load_report(report_path)
    out_dir = out_dir if out_dir is not None else default_out_dir(report_path)
    board_path = board_path if board_path is not None else default_board_path()
    pieces_path = pieces_path if pieces_path is not None else default_pieces_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = load_piece_templates(pieces_path)
    confirmed = render_board(report, board_path, templates, include_unknown_candidates=False)
    candidates = render_board(report, board_path, templates, include_unknown_candidates=True)

    confirmed_path = out_dir / "position_render_confirmed.png"
    candidates_path = out_dir / "position_render_candidates.png"
    comparison_path = out_dir / "position_comparison.png"
    confirmed.save(confirmed_path)
    candidates.save(candidates_path)

    screenshot = resolve_screenshot_path(report, screenshot_path)
    comparison = render_comparison(report, screenshot, candidates)
    if comparison is not None:
        comparison.save(comparison_path)
    else:
        comparison_path = None

    return RenderedOutputs(
        confirmed_path=confirmed_path,
        candidates_path=candidates_path,
        comparison_path=comparison_path,
    )


def render_board(
    report: dict,
    board_path: Path,
    templates: dict[tuple[str, str], Image.Image],
    include_unknown_candidates: bool,
) -> Image.Image:
    board = Image.open(board_path).convert("RGB").resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.BICUBIC)
    draw_grid(board)
    draw = ImageDraw.Draw(board)
    cell_size = BOARD_SIZE / 9.0

    for cell in recognition_cells(report):
        row = int(cell["row"])
        col = int(cell["col"])
        state = cell.get("state")
        piece_info = piece_for_cell(cell, include_unknown_candidates)
        if piece_info is None:
            continue

        color, piece, tentative = piece_info
        tile = templates.get((color, piece))
        if tile is None:
            continue

        piece_image = resize_piece(tile, int(cell_size))
        if tentative:
            piece_image = with_alpha(piece_image, 128)
        x = round((col - 1) * cell_size + (cell_size - piece_image.width) / 2)
        y = round((row - 1) * cell_size + (cell_size - piece_image.height) / 2)
        board.paste(piece_image, (x, y), piece_image)
        if tentative:
            draw_tentative_label(draw, col, row, piece, cell.get("confidence", 0.0), cell_size)

    return board


def draw_grid(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    for index in range(10):
        position = round(index * BOARD_SIZE / 9)
        width = 5 if index in (0, 9) else 3
        color = GRID_BORDER_COLOR if index in (0, 9) else GRID_LINE_COLOR
        draw.line((position, 0, position, BOARD_SIZE), fill=color, width=width)
        draw.line((0, position, BOARD_SIZE, position), fill=color, width=width)


def piece_for_cell(
    cell: dict,
    include_unknown_candidates: bool,
) -> tuple[str, str, bool] | None:
    if cell.get("state") == "piece" and cell.get("color") and cell.get("piece"):
        return str(cell["color"]), str(cell["piece"]), False
    if not include_unknown_candidates or cell.get("state") != "unknown":
        return None
    candidates = cell.get("candidates") or []
    if not candidates:
        return None
    best = candidates[0]
    return str(best["color"]), str(best["piece"]), True


def load_piece_templates(path: Path) -> dict[tuple[str, str], Image.Image]:
    sheet = Image.open(path).convert("RGBA")
    tile_width = sheet.width // 8
    tile_height = sheet.height // 4
    templates: dict[tuple[str, str], Image.Image] = {}
    for sprite_row in range(4):
        color = "black" if sprite_row < 2 else "white"
        pieces = NORMAL_PIECES if sprite_row % 2 == 0 else PROMOTED_PIECES
        for sprite_col, piece in enumerate(pieces):
            if piece is None:
                continue
            tile = sheet.crop(
                (
                    sprite_col * tile_width,
                    sprite_row * tile_height,
                    (sprite_col + 1) * tile_width,
                    (sprite_row + 1) * tile_height,
                ),
            )
            templates[(color, piece)] = trim_transparent(tile)
    return templates


def trim_transparent(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox is not None else image


def resize_piece(image: Image.Image, cell_size: int) -> Image.Image:
    max_width = int(cell_size * 0.82)
    max_height = int(cell_size * 0.92)
    scale = min(max_width / image.width, max_height / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def with_alpha(image: Image.Image, alpha_value: int) -> Image.Image:
    result = image.copy()
    alpha = result.getchannel("A").point(lambda value: int(value * alpha_value / 255))
    result.putalpha(alpha)
    return result


def draw_tentative_label(
    draw: ImageDraw.ImageDraw,
    col: int,
    row: int,
    piece: str,
    confidence: float,
    cell_size: float,
) -> None:
    left = round((col - 1) * cell_size + 4)
    top = round((row - 1) * cell_size + 4)
    text = f"?{piece} {confidence:.2f}"
    draw.rectangle((left, top, left + 72, top + 20), fill=(255, 255, 255), outline=(190, 105, 0))
    draw.text((left + 3, top + 3), text, fill=(170, 80, 0), font=small_font())


def render_comparison(
    report: dict,
    screenshot_path: Path | None,
    rendered: Image.Image,
) -> Image.Image | None:
    if screenshot_path is None or not screenshot_path.exists():
        return None
    source = source_board_image(report, screenshot_path)
    source = source.resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.BICUBIC)

    header_height = 56
    margin = 24
    comparison = Image.new("RGB", (BOARD_SIZE * 2 + margin * 3, BOARD_SIZE + header_height + margin), (34, 34, 34))
    draw = ImageDraw.Draw(comparison)
    draw.text((margin, 18), "input detected grid area", fill=(255, 255, 255), font=label_font())
    draw.text((BOARD_SIZE + margin * 2, 18), "rendered recognition result", fill=(255, 255, 255), font=label_font())
    comparison.paste(source, (margin, header_height))
    comparison.paste(rendered.convert("RGB"), (BOARD_SIZE + margin * 2, header_height))
    return comparison


def source_board_image(
    report: dict,
    screenshot_path: Path,
) -> Image.Image:
    screenshot = Image.open(screenshot_path).convert("RGB")
    grid = report.get("grid")
    if grid and grid.get("detected") and grid.get("grid_rect"):
        rect = grid["grid_rect"]
        return screenshot.crop((rect["left"], rect["top"], rect["right"], rect["bottom"]))
    return screenshot


def resolve_screenshot_path(
    report: dict,
    override: Path | None,
) -> Path | None:
    if override is not None:
        return override
    image = report.get("image")
    if isinstance(image, str) and image:
        candidate = Path(image)
        if candidate.exists():
            return candidate
        sample_candidate = Path("tools/samples/screenshots") / candidate.name
        if sample_candidate.exists():
            return sample_candidate
    return None


def recognition_cells(report: dict) -> list[dict]:
    cells = report["piece_recognition"]["cells"]
    return sorted(cells, key=lambda cell: (int(cell["row"]), int(cell["col"])))


def small_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def label_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def default_board_path() -> Path:
    candidates = [
        Path("assets/legacy_drawables/shogi_board.png"),
        Path(__file__).resolve().parents[1] / "assets" / "legacy_drawables" / "shogi_board.png",
        Path(__file__).resolve().parents[0] / "assets" / "legacy_drawables" / "shogi_board.png",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_pieces_path() -> Path:
    candidates = [
        Path("assets/legacy_drawables/shogi_pieces.png"),
        Path(__file__).resolve().parents[1] / "assets" / "legacy_drawables" / "shogi_pieces.png",
        Path(__file__).resolve().parents[0] / "assets" / "legacy_drawables" / "shogi_pieces.png",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def default_out_dir(report_path: Path) -> Path:
    return report_path if report_path.is_dir() else report_path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render recognized shogi position using bundled app assets.")
    parser.add_argument("report", type=Path, help="analysis_report.json, piece_report.json, recognized_board.json, or an output directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to the report directory.")
    parser.add_argument("--board", type=Path, default=default_board_path(), help="Board image path.")
    parser.add_argument("--pieces", type=Path, default=default_pieces_path(), help="Piece sprite image path.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Original screenshot path for comparison output.")
    args = parser.parse_args()

    outputs = render_outputs(
        report_path=args.report,
        out_dir=args.out,
        board_path=args.board,
        pieces_path=args.pieces,
        screenshot_path=args.screenshot,
    )
    print(f"OK: confirmed render -> {outputs.confirmed_path}")
    print(f"OK: candidate render -> {outputs.candidates_path}")
    if outputs.comparison_path is not None:
        print(f"OK: comparison -> {outputs.comparison_path}")


if __name__ == "__main__":
    main()
