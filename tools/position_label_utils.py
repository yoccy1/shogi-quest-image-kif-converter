from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


FILES = "987654321"
RANKS = "一二三四五六七八九"
COLORS = ("black", "white")
HAND_PIECES = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
NORMAL_PIECES = ("OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU")
PROMOTED_TO_BASE = {
    "RY": "HI",
    "UM": "KA",
    "NG": "GI",
    "NK": "KE",
    "NY": "KY",
    "TO": "FU",
}
VALID_PIECES = set(NORMAL_PIECES) | set(PROMOTED_TO_BASE)
TOTAL_INVENTORY = {
    "OU": 2,
    "HI": 2,
    "KA": 2,
    "KI": 4,
    "GI": 4,
    "KE": 4,
    "KY": 4,
    "FU": 18,
}


def square_name(row: int, col: int) -> str:
    return f"{FILES[col - 1]}{RANKS[row - 1]}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def empty_hands() -> dict[str, dict[str, int]]:
    return {
        color: {piece: 0 for piece in HAND_PIECES}
        for color in COLORS
    }


def identity(color: str | None, piece: str | None) -> str:
    if color and piece:
        return f"{color}:{piece}"
    return "none"


def base_piece(piece: str) -> str:
    return PROMOTED_TO_BASE.get(piece, piece)


def normalize_label_cell(value: Any, row: int, col: int) -> dict[str, Any]:
    square = square_name(row, col)
    if isinstance(value, str):
        raw = value.strip()
        if raw in {"", ".", "empty"}:
            return {"row": row, "col": col, "square": square, "state": "empty", "color": None, "piece": None}
        if raw in {"unknown", "?"}:
            return {"row": row, "col": col, "square": square, "state": "unknown", "color": None, "piece": None}
        if ":" not in raw:
            raise ValueError(f"{square}: label must be empty or color:piece, got {raw!r}")
        color, piece = raw.split(":", 1)
        color = {"b": "black", "w": "white"}.get(color, color)
        return normalize_piece_cell(square, row, col, color, piece)

    if not isinstance(value, dict):
        raise ValueError(f"{square}: unsupported label cell type: {type(value).__name__}")

    state = value.get("state", "piece" if value.get("piece") else "empty")
    if state == "empty":
        return {"row": row, "col": col, "square": value.get("square", square), "state": "empty", "color": None, "piece": None}
    if state == "unknown":
        return {"row": row, "col": col, "square": value.get("square", square), "state": "unknown", "color": None, "piece": None}
    if state != "piece":
        raise ValueError(f"{square}: invalid state {state!r}")
    return normalize_piece_cell(square, row, col, value.get("color"), value.get("piece"), value.get("square", square))


def normalize_piece_cell(
    expected_square: str,
    row: int,
    col: int,
    color: str | None,
    piece: str | None,
    square: str | None = None,
) -> dict[str, Any]:
    if color not in COLORS:
        raise ValueError(f"{expected_square}: invalid color {color!r}")
    if piece not in VALID_PIECES:
        raise ValueError(f"{expected_square}: invalid piece {piece!r}")
    actual_square = square or expected_square
    if actual_square != expected_square:
        raise ValueError(f"expected {expected_square}, got {actual_square}")
    return {"row": row, "col": col, "square": expected_square, "state": "piece", "color": color, "piece": piece}


def normalize_hands(value: Any, require_hands: bool) -> dict[str, dict[str, int]] | None:
    if value is None:
        if require_hands:
            raise ValueError("hands are required")
        return None
    if not isinstance(value, dict):
        raise ValueError("hands must be an object")
    hands = empty_hands()
    for color in COLORS:
        raw_counts = value.get(color)
        if raw_counts is None:
            if require_hands:
                raise ValueError(f"hands.{color} is required")
            raw_counts = {}
        if not isinstance(raw_counts, dict):
            raise ValueError(f"hands.{color} must be an object")
        for piece in HAND_PIECES:
            count = int(raw_counts.get(piece, 0))
            if count < 0:
                raise ValueError(f"hands.{color}.{piece} must be >= 0")
            hands[color][piece] = count
    return hands


def load_position_label(path: Path, require_hands: bool = False) -> dict[str, Any]:
    data = read_json(path)
    rows = data.get("rows") or data.get("board")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError(f"{path}: expected 9 rows")

    cells = []
    for row_index, row_values in enumerate(rows, start=1):
        if not isinstance(row_values, list) or len(row_values) != 9:
            raise ValueError(f"{path}: row {row_index} must contain 9 cells")
        for col_index, value in enumerate(row_values, start=1):
            cells.append(normalize_label_cell(value, row_index, col_index))

    counts = Counter(cell["state"] for cell in cells)
    hands = normalize_hands(data.get("hands"), require_hands=require_hands)
    return {
        "path": str(path),
        "schema_version": data.get("schema_version", 1),
        "image": data.get("image"),
        "exclude_from_benchmark": bool(data.get("exclude_from_benchmark", False)),
        "exclude_reason": data.get("exclude_reason"),
        "orientation": data.get("orientation", "black_bottom"),
        "cells": cells,
        "hands": hands,
        "summary": {
            "total": len(cells),
            "piece": counts["piece"],
            "empty": counts["empty"],
            "unknown": counts["unknown"],
        },
    }


def inventory_counts(cells: list[dict[str, Any]], hands: dict[str, dict[str, int]] | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for cell in cells:
        if cell["state"] != "piece":
            continue
        counts[base_piece(cell["piece"])] += 1
    if hands:
        for color in COLORS:
            for piece, count in hands[color].items():
                counts[piece] += int(count)
    return counts


def inventory_errors(cells: list[dict[str, Any]], hands: dict[str, dict[str, int]] | None = None) -> list[str]:
    counts = inventory_counts(cells, hands)
    errors: list[str] = []
    for piece, expected in TOTAL_INVENTORY.items():
        actual = counts[piece]
        if actual != expected:
            errors.append(f"{piece}: expected {expected}, got {actual}")
    extra = sorted(piece for piece in counts if piece not in TOTAL_INVENTORY)
    for piece in extra:
        errors.append(f"{piece}: unexpected count {counts[piece]}")
    return errors


def board_inventory_overcount_errors(cells: list[dict[str, Any]]) -> list[str]:
    counts = inventory_counts(cells, None)
    errors: list[str] = []
    for piece, expected in TOTAL_INVENTORY.items():
        actual = counts[piece]
        if actual > expected:
            errors.append(f"{piece}: expected at most {expected} on board, got {actual}")
    extra = sorted(piece for piece in counts if piece not in TOTAL_INVENTORY)
    for piece in extra:
        errors.append(f"{piece}: unexpected count {counts[piece]}")
    return errors


def resolve_label_image_path(label_path: Path, label: dict[str, Any], screenshots_dir: Path) -> Path:
    image_value = label.get("image")
    if not image_value:
        return screenshots_dir / f"{label_path.stem}.png"
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    candidate = (label_path.parent / image_path).resolve()
    if candidate.exists():
        return candidate
    return (screenshots_dir / image_path.name).resolve()


def find_label_path(labels_dir: Path, sample: str, app: str | None = None, piece_style: str | None = None) -> Path:
    if app and piece_style:
        grouped = labels_dir / app / piece_style / f"{sample}.json"
        if grouped.exists():
            return grouped
    if app:
        app_direct = labels_dir / app / f"{sample}.json"
        if app_direct.exists():
            return app_direct
    direct = labels_dir / f"{sample}.json"
    if direct.exists():
        return direct
    matches = sorted(labels_dir.rglob(f"{sample}.json"))
    if app:
        app_matches = [path for path in matches if app in path.parts]
        if piece_style:
            app_style_matches = [path for path in app_matches if piece_style in path.parts]
            if len(app_style_matches) == 1:
                return app_style_matches[0]
        if len(app_matches) == 1:
            return app_matches[0]
    if len(matches) == 1:
        return matches[0]
    return direct
