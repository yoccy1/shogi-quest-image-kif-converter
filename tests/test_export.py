from __future__ import annotations

from shogi_gazo_desktop.export import ExportError, export_kif, export_sfen
from shogi_gazo_desktop.models import RecognitionResult, empty_hands


INITIAL_BOARD = [
    ["white:KY", "white:KE", "white:GI", "white:KI", "white:OU", "white:KI", "white:GI", "white:KE", "white:KY"],
    ["empty", "white:HI", "empty", "empty", "empty", "empty", "empty", "white:KA", "empty"],
    ["white:FU", "white:FU", "white:FU", "white:FU", "white:FU", "white:FU", "white:FU", "white:FU", "white:FU"],
    ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    ["black:FU", "black:FU", "black:FU", "black:FU", "black:FU", "black:FU", "black:FU", "black:FU", "black:FU"],
    ["empty", "black:KA", "empty", "empty", "empty", "empty", "empty", "black:HI", "empty"],
    ["black:KY", "black:KE", "black:GI", "black:KI", "black:OU", "black:KI", "black:GI", "black:KE", "black:KY"],
]


def result(board=None, hands=None) -> RecognitionResult:
    return RecognitionResult(
        image="sample.png",
        board=board or INITIAL_BOARD,
        hands=hands or empty_hands(),
        confidence=[],
        raw_report={},
    )


def test_export_initial_position_sfen() -> None:
    assert export_sfen(result()) == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_export_sfen_with_promoted_piece_and_hands() -> None:
    board = [row[:] for row in INITIAL_BOARD]
    board[7][7] = "black:RY"
    board[6][0] = "empty"
    board[6][1] = "empty"
    board[1][7] = "empty"
    hands = empty_hands()
    hands["black"]["FU"] = 2
    hands["white"]["KA"] = 1
    assert "+R" in export_sfen(result(board, hands))
    assert "2Pb" in export_sfen(result(board, hands))


def test_export_kif_contains_position_board() -> None:
    text = export_kif(result(), side_to_move="white")
    assert "後手番" in text
    assert "手合割：その他" in text
    assert "手数＝" not in text
    assert "   1 中断 ( 0:00/00:00:00)" in text
    assert "まで1手で中断" in text
    assert "後手の持駒：なし" in text
    assert "先手の持駒：なし" in text
    assert "+---------------------------+" in text
    board_lines = [line for line in text.splitlines() if line.startswith("|")]
    assert all(len(line.removeprefix("|").split("|")[0].encode("cp932")) == 27 for line in board_lines)


def test_export_rejects_unknown_cells() -> None:
    board = [row[:] for row in INITIAL_BOARD]
    board[0][0] = "unknown"
    try:
        export_sfen(result(board))
    except ExportError:
        return
    raise AssertionError("unknown cells must not export")


def test_export_rejects_negative_hands() -> None:
    hands = empty_hands()
    hands["black"]["FU"] = -1
    try:
        export_sfen(result(hands=hands))
    except ExportError:
        return
    raise AssertionError("negative hand counts must not export")


def test_export_rejects_nifu() -> None:
    board = [row[:] for row in INITIAL_BOARD]
    board[5][0] = "black:FU"
    try:
        export_sfen(result(board))
    except ExportError:
        return
    raise AssertionError("nifu must not export")


def test_export_rejects_missing_king() -> None:
    board = [row[:] for row in INITIAL_BOARD]
    board[8][4] = "empty"
    try:
        export_sfen(result(board))
    except ExportError:
        return
    raise AssertionError("king count errors must not export")


def test_export_rejects_unresolved_constraints() -> None:
    raw_report = {"constraint_postprocess": {"unresolved": [{"reason": "nifu", "square": "5五"}]}}
    item = RecognitionResult(
        image="sample.png",
        board=INITIAL_BOARD,
        hands=empty_hands(),
        confidence=[],
        raw_report=raw_report,
    )
    try:
        export_sfen(item)
    except ExportError:
        return
    raise AssertionError("unresolved board constraints must not export")
