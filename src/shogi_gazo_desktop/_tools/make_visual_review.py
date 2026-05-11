from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from evaluate_piece_recognition import evaluate_one, load_report
from position_label_utils import HAND_PIECES, find_label_path, load_position_label, square_name


PIECE_TEXT = {
    "OU": "\u7389",
    "HI": "\u98db",
    "KA": "\u89d2",
    "KI": "\u91d1",
    "GI": "\u9280",
    "KE": "\u6842",
    "KY": "\u9999",
    "FU": "\u6b69",
    "RY": "\u9f8d",
    "UM": "\u99ac",
    "NG": "\u6210\u9280",
    "NK": "\u6210\u6842",
    "NY": "\u6210\u9999",
    "TO": "\u3068",
}
COLOR_MARK = {"black": "\u25b2", "white": "\u25b3"}


def report_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    benchmark = path / "benchmark_report.json"
    if benchmark.exists():
        data = json.loads(benchmark.read_text(encoding="utf-8"))
        paths = []
        for result in data.get("results", []):
            report = Path(result.get("report", ""))
            if report.exists():
                paths.append(report)
        if paths:
            return paths
    direct = path / "piece_report.json"
    if direct.exists():
        return [direct]
    return sorted(path.glob("*/piece_report.json"))


def sample_name(report_path: Path, report: dict[str, Any]) -> str:
    image = report.get("image")
    if image:
        return Path(image).stem
    return report_path.parent.name


def evaluate_if_possible(report_path: Path, labels_dir: Path, include_hands: bool) -> dict[str, Any] | None:
    report = load_report(report_path)
    label_path = find_label_path(labels_dir, sample_name(report_path, report))
    if not label_path.exists():
        return None
    try:
        return evaluate_one(report_path, label_path, 0.75, include_hands=include_hands)
    except Exception:
        try:
            return evaluate_one(report_path, label_path, 0.75, include_hands=False)
        except Exception:
            return None


def load_label_if_possible(report_path: Path, labels_dir: Path) -> dict[str, Any] | None:
    report = load_report(report_path)
    label_path = find_label_path(labels_dir, sample_name(report_path, report))
    if not label_path.exists():
        return None
    try:
        return load_position_label(label_path, require_hands=False)
    except Exception:
        return None


def identity_text(color: str | None, piece: str | None) -> str:
    if not color or not piece:
        return ""
    return f"{COLOR_MARK.get(color, '')}{PIECE_TEXT.get(piece, piece)}"


def candidate_text(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return ""
    return identity_text(candidate.get("color"), candidate.get("piece"))


def cell_label(cell: dict[str, Any]) -> str:
    if cell.get("state") == "empty":
        return ""
    if cell.get("state") == "piece":
        return identity_text(cell.get("color"), cell.get("piece"))
    candidates = cell.get("candidates") or []
    return "?" + candidate_text(candidates[0] if candidates else None)


def cell_title(cell: dict[str, Any], error_by_square: dict[str, dict[str, Any]]) -> str:
    parts = [str(cell.get("square") or square_name(int(cell["row"]), int(cell["col"])))]
    parts.append(f"状態={cell.get('state')}")
    parts.append(f"信頼度={cell.get('confidence')}")
    error = error_by_square.get(str(cell.get("square")))
    if error:
        parts.append(f"教師ラベル={error.get('expected')}")
        parts.append(f"予測top1={error.get('predicted_top1')}")
    top = []
    for candidate in (cell.get("candidates") or [])[:3]:
        top.append(f"{candidate_text(candidate)}:{candidate.get('score')}")
    if top:
        parts.append("上位3候補=" + " / ".join(top))
    return " | ".join(parts)


def hands_text(hands: dict[str, Any] | None) -> str:
    if not hands:
        return "-"
    groups = []
    for color in ("black", "white"):
        pieces = []
        for piece in HAND_PIECES:
            count = int((hands.get(color) or {}).get(piece, 0))
            if count:
                suffix = f"x{count}" if count > 1 else ""
                pieces.append(f"{PIECE_TEXT.get(piece, piece)}{suffix}")
        owner = "\u5148\u624b" if color == "black" else "\u5f8c\u624b"
        groups.append(f"{owner}: " + (", ".join(pieces) if pieces else "\u306a\u3057"))
    return " / ".join(groups)


def path_uri(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


def render_board(
    cells: list[dict[str, Any]],
    evaluation: dict[str, Any] | None,
    low_confidence: float,
    *,
    highlight_errors: bool = True,
    highlight_low_confidence: bool = True,
) -> str:
    error_by_square = {item["square"]: item for item in (evaluation or {}).get("errors", [])}
    cell_by_pos = {(int(cell["row"]), int(cell["col"])): cell for cell in cells}
    chunks = ['<div class="board">']
    for row in range(1, 10):
        for col in range(1, 10):
            cell = cell_by_pos.get((row, col), {"row": row, "col": col, "state": "missing", "confidence": 0.0})
            square = str(cell.get("square") or square_name(row, col))
            classes = ["cell"]
            if highlight_errors and square in error_by_square:
                classes.append("error")
            elif cell.get("state") == "unknown":
                classes.append("unknown")
            elif highlight_low_confidence and cell.get("state") == "piece" and float(cell.get("confidence") or 0.0) < low_confidence:
                classes.append("low")
            elif cell.get("state") == "empty":
                classes.append("empty")
            chunks.append(
                '<div class="{classes}" title="{title}">'
                '<span class="sq">{square}</span><span class="piece">{piece}</span>'
                "</div>".format(
                    classes=" ".join(classes),
                    title=html.escape(cell_title(cell, error_by_square), quote=True),
                    square=html.escape(square),
                    piece=html.escape(cell_label(cell)),
                ),
            )
    chunks.append("</div>")
    return "\n".join(chunks)


def render_error_list(evaluation: dict[str, Any] | None) -> str:
    if not evaluation:
        return '<p class="muted">このレポートでは教師ラベルとの比較を利用できませんでした。</p>'
    errors = evaluation.get("errors", [])
    hand_errors = (evaluation.get("hands") or {}).get("error_details", [])
    if not errors and not hand_errors:
        return '<p class="ok">現在の教師ラベル比較では、盤面と持ち駒の不一致はありません。</p>'
    lines = ['<ul class="errors">']
    for item in errors[:24]:
        lines.append(
            "<li>{sq}: 教師ラベル <b>{exp}</b>, 予測top1 <b>{pred}</b>, 確定 {conf}, 信頼度 {score}</li>".format(
                sq=html.escape(str(item.get("square"))),
                exp=html.escape(str(item.get("expected"))),
                pred=html.escape(str(item.get("predicted_top1"))),
                conf=html.escape(str(item.get("confirmed"))),
                score=html.escape(str(item.get("confidence"))),
            ),
        )
    if len(errors) > 24:
        lines.append(f"<li>ほか {len(errors) - 24} 件の盤面不一致があります</li>")
    for item in hand_errors:
        lines.append(
            "<li>持ち駒 {owner}:{piece}: 教師ラベル {expected}, 認識結果 {actual}</li>".format(
                owner=html.escape(str(item.get("owner"))),
                piece=html.escape(str(item.get("piece"))),
                expected=html.escape(str(item.get("expected"))),
                actual=html.escape(str(item.get("actual"))),
            ),
        )
    lines.append("</ul>")
    return "\n".join(lines)


def render_report_card(report_path: Path, labels_dir: Path, include_hands: bool, low_confidence: float) -> str:
    report = load_report(report_path)
    evaluation = evaluate_if_possible(report_path, labels_dir, include_hands)
    label = load_label_if_possible(report_path, labels_dir)
    name = sample_name(report_path, report)
    image_uri = path_uri(report.get("image"))
    metrics = (evaluation or {}).get("metrics", {})
    timing = report.get("timing") or {}
    summary_bits = []
    if metrics:
        summary_bits.append(f"盤面不一致={metrics.get('errors')}")
        summary_bits.append(f"持ち駒不一致={metrics.get('hand_errors')}")
        summary_bits.append(f"駒ありマスのunknown={metrics.get('unknown_on_piece')}")
    if timing:
        summary_bits.append(f"処理時間={timing.get('processing_time_seconds')}秒")
    hand_expected = (label or {}).get("hands") or ((evaluation or {}).get("hands") or {}).get("expected")
    label_board = (
        render_board(label.get("cells") or [], evaluation, low_confidence, highlight_errors=True, highlight_low_confidence=False)
        if label
        else '<p class="muted">教師ラベル盤面を読み込めませんでした。</p>'
    )
    return """
<section class="card">
  <h2>{name}</h2>
  <div class="summary">{summary}</div>
  <div class="review-grid">
    <div>
      <h3>元画像</h3>
      {image}
    </div>
    <div>
      <h3>認識した盤面</h3>
      {board}
    </div>
    <div>
      <h3>教師ラベル（正解盤面）</h3>
      {label_board}
    </div>
    <div>
      <h3>持ち駒</h3>
      <p><b>認識結果</b>: {actual_hands}</p>
      <p><b>教師ラベル（正解）</b>: {expected_hands}</p>
      <h3>優先確認</h3>
      {errors}
    </div>
  </div>
</section>
""".format(
        name=html.escape(name),
        summary=html.escape(" / ".join(summary_bits) if summary_bits else "-"),
        image=f'<img class="shot" src="{html.escape(image_uri, quote=True)}">' if image_uri else '<p class="muted">レポート内に元画像パスがありません。</p>',
        board=render_board(report.get("cells") or [], evaluation, low_confidence),
        label_board=label_board,
        actual_hands=html.escape(hands_text(report.get("hands"))),
        expected_hands=html.escape(hands_text(hand_expected)),
        errors=render_error_list(evaluation),
    )


def render_html(cards: list[str]) -> str:
    template = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>将棋認識 目視確認</title>
<style>
body { margin: 24px; font-family: system-ui, "Yu Gothic", sans-serif; background: #f6f2ea; color: #241f18; }
h1 { margin-bottom: 4px; }
.guide { max-width: 1200px; line-height: 1.7; }
.card { background: #fffaf0; border: 1px solid #d8c8aa; border-radius: 8px; padding: 16px; margin: 18px 0; }
.summary { color: #5f574b; margin-bottom: 12px; }
.review-grid { display: grid; grid-template-columns: minmax(220px, 320px) auto auto minmax(260px, 380px); gap: 16px; align-items: start; }
.shot { width: 100%; max-height: 720px; object-fit: contain; border: 1px solid #c9b897; background: #111; }
.board { display: grid; grid-template-columns: repeat(9, 46px); grid-template-rows: repeat(9, 46px); border: 2px solid #7d532c; background: #e8bd6b; }
.cell { position: relative; width: 46px; height: 46px; border: 1px solid #8a6538; box-sizing: border-box; display: flex; align-items: center; justify-content: center; text-align: center; }
.cell.error { background: #ffdad6; outline: 3px solid #c7342e; z-index: 1; }
.cell.unknown { background: #fff0b8; }
.cell.low { background: #ffe0ad; }
.cell.empty { background: #e8bd6b; }
.sq { position: absolute; top: 2px; left: 3px; font-size: 10px; color: #725a36; }
.piece { font-size: 15px; font-weight: 700; line-height: 1.1; }
.errors { padding-left: 20px; line-height: 1.55; }
.ok { color: #176b3a; font-weight: 700; }
.muted { color: #766c5d; }
@media (max-width: 1100px) {
  .review-grid { grid-template-columns: 1fr; }
  .board { grid-template-columns: repeat(9, 42px); grid-template-rows: repeat(9, 42px); }
  .cell { width: 42px; height: 42px; }
  .piece { font-size: 14px; }
}
</style>
</head>
<body>
<h1>将棋認識 目視確認</h1>
<div class="guide">
  <p>ベンチマークや解析のあと、このページで結果を目視確認します。まず赤いマス（教師ラベルとの不一致）、次に黄色いマス（unknown または低信頼）、最後に持ち駒の数を確認してください。「教師ラベル（正解）」はあなたが提示した正解で、「認識結果」はスクリプトの出力です。このページは確認用であり、推論には教師ラベルを戻しません。</p>
</div>
__CARDS__
</body>
</html>
"""
    return template.replace("__CARDS__", "\n".join(cards))


def write_visual_review(
    reports: Path,
    labels_dir: Path = Path("tools/samples/labels/boards"),
    out_path: Path | None = None,
    include_hands: bool = False,
    low_confidence: float = 0.55,
) -> Path:
    paths = report_paths(reports)
    if not paths:
        raise FileNotFoundError(f"no piece_report.json found under {reports}")
    resolved_out = out_path or (reports if reports.is_dir() else reports.parent) / "visual_review.html"
    cards = [render_report_card(path, labels_dir, include_hands, low_confidence) for path in paths]
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(render_html(cards), encoding="utf-8")
    return resolved_out


def main() -> None:
    parser = argparse.ArgumentParser(description="目視確認用の軽量HTMLページを作成します。")
    parser.add_argument("reports", type=Path, help="ベンチマーク出力フォルダ、解析出力フォルダ、または piece_report.json。")
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards"))
    parser.add_argument("--out", type=Path, help="HTML出力先。既定では <reports>/visual_review.html。")
    parser.add_argument("--include-hands", action="store_true", help="教師ラベルに持ち駒がある場合、持ち駒の数も比較します。")
    parser.add_argument("--low-confidence", type=float, default=0.55)
    args = parser.parse_args()

    out_path = write_visual_review(args.reports, args.labels_dir, args.out, args.include_hands, args.low_confidence)
    print(f"OK: {out_path} を作成しました（{len(report_paths(args.reports))}件のレポート）")


if __name__ == "__main__":
    main()
