from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from position_label_utils import (
    COLORS,
    HAND_PIECES,
    TOTAL_INVENTORY,
    base_piece,
    find_label_path,
    load_position_label,
)


PIECE_JP = {
    "OU": "玉",
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
}

COLOR_JP = {"black": "先手", "white": "後手"}


def read_report_hands(analysis_dir: Path | None, sample: str) -> dict[str, dict[str, int]] | None:
    if analysis_dir is None:
        return None
    manifest = analysis_dir / "manifest.csv"
    if not manifest.exists():
        return None
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("sample") != sample:
                continue
            report_path = Path(row.get("report") or "")
            if not report_path.exists():
                return None
            report = json.loads(report_path.read_text(encoding="utf-8"))
            hands = report.get("hands")
            return hands if isinstance(hands, dict) else None
    return None


def inventory_detail(path: Path, analysis_dir: Path | None = None) -> dict[str, Any] | None:
    label = load_position_label(path, require_hands=True)
    board_counts: Counter[str] = Counter()
    board_cells_by_base: dict[str, list[str]] = {piece: [] for piece in TOTAL_INVENTORY}
    for cell in label["cells"]:
        if cell["state"] != "piece":
            continue
        piece = base_piece(str(cell["piece"]))
        board_counts[piece] += 1
        board_cells_by_base.setdefault(piece, []).append(
            f"{cell['square']}:{COLOR_JP.get(str(cell['color']), cell['color'])}:{cell['piece']}"
        )

    hand_counts: Counter[str] = Counter()
    hands_by_color = label["hands"]
    for color in COLORS:
        for piece in HAND_PIECES:
            hand_counts[piece] += int(hands_by_color[color].get(piece, 0))

    diffs = {
        piece: board_counts[piece] + hand_counts[piece] - expected
        for piece, expected in TOTAL_INVENTORY.items()
        if board_counts[piece] + hand_counts[piece] != expected
    }
    if not diffs:
        return None

    sample = path.stem
    report_hands = read_report_hands(analysis_dir, sample)
    return {
        "sample": sample,
        "label": str(path),
        "diffs": diffs,
        "board_counts": {piece: board_counts[piece] for piece in TOTAL_INVENTORY},
        "hand_counts": {piece: hand_counts[piece] for piece in HAND_PIECES},
        "hands_by_color": hands_by_color,
        "report_hands": report_hands,
        "board_cells_by_base": board_cells_by_base,
        "suggestion": suggest_action(diffs, hands_by_color, report_hands),
    }


def suggest_action(
    diffs: dict[str, int],
    hands_by_color: dict[str, dict[str, int]],
    report_hands: dict[str, dict[str, int]] | None,
) -> str:
    if any(diff > 0 for diff in diffs.values()):
        return "盤面または持ち駒に余分な駒があります。自動補完せず画像確認が必要です。"
    if all(int(hands_by_color[color].get(piece, 0)) == 0 for color in COLORS for piece in HAND_PIECES):
        return "持ち駒が全て0です。持ち駒ラベル未入力の可能性が高いです。"
    if report_hands:
        candidates = []
        for piece, diff in diffs.items():
            missing = -diff
            recognized = sum(int(report_hands.get(color, {}).get(piece, 0)) for color in COLORS)
            if recognized >= missing:
                candidates.append(PIECE_JP.get(piece, piece))
        if candidates:
            return "認識結果側に不足分の候補があります: " + "、".join(candidates)
    return "不足があります。どちらの持ち駒に属するか確認が必要です。"


def flatten_rows(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in details:
        for piece, diff in item["diffs"].items():
            report_hands = item.get("report_hands") or {}
            rows.append(
                {
                    "sample": item["sample"],
                    "piece": piece,
                    "piece_jp": PIECE_JP.get(piece, piece),
                    "expected_total": TOTAL_INVENTORY[piece],
                    "board_count": item["board_counts"].get(piece, 0),
                    "hand_count": item["hand_counts"].get(piece, 0),
                    "actual_total": item["board_counts"].get(piece, 0) + item["hand_counts"].get(piece, 0),
                    "diff_actual_minus_expected": diff,
                    "black_hand_label": item["hands_by_color"]["black"].get(piece, 0),
                    "white_hand_label": item["hands_by_color"]["white"].get(piece, 0),
                    "black_hand_recognized": report_hands.get("black", {}).get(piece, ""),
                    "white_hand_recognized": report_hands.get("white", {}).get(piece, ""),
                    "suggestion": item["suggestion"],
                    "label": item["label"],
                    "board_cells": " / ".join(item["board_cells_by_base"].get(piece, [])),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["sample"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for item in details:
        rows = []
        for piece, diff in item["diffs"].items():
            cells = "<br>".join(html.escape(cell) for cell in item["board_cells_by_base"].get(piece, []))
            rows.append(
                "<tr>"
                f"<td>{html.escape(PIECE_JP.get(piece, piece))}</td>"
                f"<td>{TOTAL_INVENTORY[piece]}</td>"
                f"<td>{item['board_counts'].get(piece, 0)}</td>"
                f"<td>{item['hand_counts'].get(piece, 0)}</td>"
                f"<td>{item['board_counts'].get(piece, 0) + item['hand_counts'].get(piece, 0)}</td>"
                f"<td>{diff:+d}</td>"
                f"<td>{cells}</td>"
                "</tr>"
            )
        sections.append(
            "<section>"
            f"<h2>{html.escape(item['sample'])}</h2>"
            f"<p>{html.escape(item['suggestion'])}</p>"
            "<table><thead><tr><th>駒</th><th>必要数</th><th>盤上</th><th>持ち駒</th><th>合計</th><th>差分</th><th>盤上セル</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )
    document = (
        "<!doctype html><html lang=\"ja\"><meta charset=\"utf-8\">"
        "<title>教師ラベル物理駒数チェック</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;line-height:1.5}"
        "table{border-collapse:collapse;margin:12px 0 28px;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top}"
        "th{background:#f4f4f4}section{margin-bottom:28px}</style>"
        "<h1>教師ラベル物理駒数チェック</h1>"
        f"<p>不整合サンプル: {len(details)}件</p>"
        + "".join(sections)
        + "</html>"
    )
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit physical inventory consistency of board + hand labels.")
    parser.add_argument("labels_dir", type=Path, nargs="?", default=Path("tools/samples/labels/boards_by_app_piece_style"))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("tools/out/label_inventory_audit"))
    args = parser.parse_args()

    details = [
        detail
        for path in sorted(args.labels_dir.rglob("*.json"))
        if (detail := inventory_detail(path, args.analysis_dir)) is not None
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(details)
    write_csv(args.out_dir / "label_inventory_audit.csv", rows)
    write_html(args.out_dir / "label_inventory_audit.html", details)
    (args.out_dir / "label_inventory_audit.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "labels": len(list(args.labels_dir.rglob("*.json"))),
                "failed_samples": len(details),
                "error_rows": len(rows),
                "csv": str(args.out_dir / "label_inventory_audit.csv"),
                "html": str(args.out_dir / "label_inventory_audit.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
