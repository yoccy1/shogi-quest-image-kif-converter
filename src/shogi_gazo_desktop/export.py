from __future__ import annotations

import json
from typing import Any

from .models import HAND_PIECES, RecognitionResult


PIECE_TO_SFEN = {
    "OU": "K",
    "HI": "R",
    "KA": "B",
    "KI": "G",
    "GI": "S",
    "KE": "N",
    "KY": "L",
    "FU": "P",
    "RY": "+R",
    "UM": "+B",
    "NG": "+S",
    "NK": "+N",
    "NY": "+L",
    "TO": "+P",
}

PIECE_TO_JP = {
    "OU": "玉",
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
    "RY": "龍",
    "UM": "馬",
    "NG": "成銀",
    "NK": "成桂",
    "NY": "成香",
    "TO": "と",
}

PIECE_TO_BOD = {
    "OU": "玉",
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
    "RY": "龍",
    "UM": "馬",
    "NG": "全",
    "NK": "圭",
    "NY": "杏",
    "TO": "と",
}

PROMOTED_TO_BASE = {
    "RY": "HI",
    "UM": "KA",
    "NG": "GI",
    "NK": "KE",
    "NY": "KY",
    "TO": "FU",
}

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

HAND_ORDER = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")
BOARD_FILES = "９８７６５４３２１"
KIF_EMPTY = " ・ "
KANJI_NUMERALS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
    11: "十一",
    12: "十二",
    13: "十三",
    14: "十四",
    15: "十五",
    16: "十六",
    17: "十七",
    18: "十八",
}


class ExportError(ValueError):
    pass


def export_json(result: RecognitionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"


def export_sfen(result: RecognitionResult, side_to_move: str = "black") -> str:
    ensure_exportable(result)
    side = "b" if side_to_move == "black" else "w"
    rows = []
    for row in result.board:
        empty = 0
        parts: list[str] = []
        for cell in row:
            if cell == "empty":
                empty += 1
                continue
            if empty:
                parts.append(str(empty))
                empty = 0
            color, piece = split_piece(cell)
            token = PIECE_TO_SFEN[piece]
            if color == "white":
                token = token.lower()
            parts.append(token)
        if empty:
            parts.append(str(empty))
        rows.append("".join(parts))
    hands = hands_to_sfen(result.hands)
    return f"{'/'.join(rows)} {side} {hands} 1"


def export_kif(result: RecognitionResult, side_to_move: str = "black") -> str:
    ensure_exportable(result)
    white_hand = hands_to_kif(result.hands.get("white", {}))
    black_hand = hands_to_kif(result.hands.get("black", {}))
    lines = [
        "#KIF version=2.0 encoding=UTF-8",
        "手合割：平手",
        f"後手の持駒：{white_hand}",
        "  ９ ８ ７ ６ ５ ４ ３ ２ １",
        "+---------------------------+",
    ]
    for index, row in enumerate(result.board):
        rank = "一二三四五六七八九"[index]
        body = "".join(kif_cell(cell) for cell in row)
        lines.append(f"|{body}|{rank}")
    lines.extend(
        [
            "+---------------------------+",
            f"先手の持駒：{black_hand}",
            "後手番" if side_to_move == "white" else "先手番",
            "手数＝1",
            "手数----指手---------消費時間--",
            "まで0手で中断",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_exportable(result: RecognitionResult) -> None:
    if len(result.board) != 9 or any(len(row) != 9 for row in result.board):
        raise ExportError("board must be 9x9")
    unresolved = ((result.raw_report.get("constraint_postprocess") or {}).get("unresolved") or [])
    if unresolved:
        reasons = [
            f"{item.get('reason', 'unresolved')}:{item.get('square', '?')}"
            for item in unresolved[:8]
            if isinstance(item, dict)
        ]
        raise ExportError("cannot export while board constraints remain unresolved: " + ", ".join(reasons))
    unknown = [
        f"{row + 1},{col + 1}"
        for row, values in enumerate(result.board)
        for col, cell in enumerate(values)
        if cell == "unknown"
    ]
    if unknown:
        raise ExportError("cannot export while unknown cells remain: " + ", ".join(unknown[:12]))
    counts = {piece: 0 for piece in TOTAL_INVENTORY}
    pawns_by_color_file: dict[tuple[str, int], int] = {}
    kings: dict[str, int] = {"black": 0, "white": 0}
    for row_index, row in enumerate(result.board, start=1):
        for col_index, cell in enumerate(row, start=1):
            if cell == "empty":
                continue
            color, piece = split_piece(cell)
            base = PROMOTED_TO_BASE.get(piece, piece)
            counts[base] = counts.get(base, 0) + 1
            if piece == "OU":
                kings[color] += 1
            if piece == "FU":
                pawns_by_color_file[(color, col_index)] = pawns_by_color_file.get((color, col_index), 0) + 1
            if is_immobile(piece, color, row_index):
                raise ExportError(f"immobile piece at row {row_index}: {cell}")
    for color, hand_counts in result.hands.items():
        if color not in {"black", "white"}:
            raise ExportError(f"unsupported hand color: {color!r}")
        for piece, count_value in hand_counts.items():
            if piece not in HAND_PIECES:
                raise ExportError(f"unsupported hand piece: {piece!r}")
            count = int(count_value)
            if count < 0:
                raise ExportError(f"negative hand count: {color}:{piece}={count}")
            counts[piece] = counts.get(piece, 0) + count
    for piece, count in counts.items():
        expected = TOTAL_INVENTORY.get(piece)
        if expected is not None and count > expected:
            raise ExportError(f"too many {piece}: {count} > {expected}")
    if kings != {"black": 1, "white": 1}:
        raise ExportError(f"expected one king per side, got {kings}")
    nifu = [key for key, count in pawns_by_color_file.items() if count > 1]
    if nifu:
        raise ExportError(f"nifu detected: {nifu}")


def split_piece(cell: str) -> tuple[str, str]:
    color, separator, piece = cell.partition(":")
    if separator != ":" or color not in {"black", "white"} or piece not in PIECE_TO_SFEN:
        raise ExportError(f"unsupported cell value: {cell!r}")
    return color, piece


def hands_to_sfen(hands: dict[str, dict[str, int]]) -> str:
    parts: list[str] = []
    for color, transform in (("black", str.upper), ("white", str.lower)):
        counts = hands.get(color, {})
        for piece in HAND_ORDER:
            count = int(counts.get(piece, 0))
            if count <= 0:
                continue
            token = transform(PIECE_TO_SFEN[piece])
            parts.append((str(count) if count > 1 else "") + token)
    return "".join(parts) or "-"


def hands_to_kif(counts: dict[str, Any]) -> str:
    parts = []
    for piece in HAND_PIECES:
        count = int(counts.get(piece, 0))
        if count <= 0:
            continue
        suffix = "" if count == 1 else KANJI_NUMERALS.get(count, str(count))
        parts.append(f"{PIECE_TO_JP[piece]}{suffix}")
    return " ".join(parts) if parts else "なし"


def kif_cell(cell: str) -> str:
    if cell == "empty":
        return KIF_EMPTY
    color, piece = split_piece(cell)
    mark = "v" if color == "white" else " "
    text = PIECE_TO_BOD[piece]
    return mark + text + " "


def is_immobile(piece: str, color: str, row: int) -> bool:
    if color == "black":
        return (piece in {"FU", "KY"} and row == 1) or (piece == "KE" and row <= 2)
    return (piece in {"FU", "KY"} and row == 9) or (piece == "KE" and row >= 8)
