from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image

from position_label_utils import HAND_PIECES, find_label_path, load_position_label


PIECE_TEXT = {
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
}
OWNER_TEXT = {"black": "先手", "white": "後手"}
BOX_COLORS = {
    "area": "#2276d2",
    "recognized_ok": "#1d8f4d",
    "recognized_ng": "#d0362f",
    "inventory_removed": "#8b5a2b",
    "unknown": "#e08a00",
    "digit": "#7f44d6",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analysis_report_paths(analysis_dir: Path) -> list[Path]:
    return sorted(analysis_dir.glob("*/*/*/piece_report.json"))


def report_group(analysis_dir: Path, report_path: Path) -> tuple[str, str, str]:
    relative = report_path.relative_to(analysis_dir)
    return relative.parts[0], relative.parts[1], relative.parts[2]


def path_uri(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


def hand_counts(value: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    return {
        owner: {piece: int(((value or {}).get(owner) or {}).get(piece, 0)) for piece in HAND_PIECES}
        for owner in ("black", "white")
    }


def count_diff_rows(expected: dict[str, dict[str, int]], actual: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows = []
    for owner in ("black", "white"):
        for piece in HAND_PIECES:
            expected_count = expected[owner][piece]
            actual_count = actual[owner][piece]
            if expected_count == actual_count:
                continue
            rows.append(
                {
                    "owner": owner,
                    "piece": piece,
                    "expected": expected_count,
                    "actual": actual_count,
                    "diff_actual_minus_expected": actual_count - expected_count,
                }
            )
    return rows


def load_image_size(image_path: str | None) -> tuple[int, int]:
    if not image_path:
        return (1, 1)
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return (1, 1)


def rect_percent(rect: list[int] | tuple[int, int, int, int], width: int, height: int) -> str:
    left, top, right, bottom = [float(value) for value in rect]
    return (
        f"left:{left / max(1, width) * 100:.3f}%;"
        f"top:{top / max(1, height) * 100:.3f}%;"
        f"width:{max(0.0, right - left) / max(1, width) * 100:.3f}%;"
        f"height:{max(0.0, bottom - top) / max(1, height) * 100:.3f}%;"
    )


def rect_key(rect: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        return tuple(int(round(float(value))) for value in rect)
    except Exception:
        return None


def rect_center(rect: Any) -> tuple[float, float] | None:
    key = rect_key(rect)
    if key is None:
        return None
    left, top, right, bottom = key
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def infer_side(owner: str, rects: list[Any], areas: list[dict[str, Any]]) -> str | None:
    owner_areas = [area for area in areas if area.get("owner") == owner and area.get("rect")]
    for rect in rects:
        center = rect_center(rect)
        if center is None:
            continue
        cx, cy = center
        for area in owner_areas:
            left, top, right, bottom = [float(value) for value in area["rect"]]
            if left <= cx <= right and top <= cy <= bottom:
                return str(area.get("side") or "")
    if len(owner_areas) == 1:
        return str(owner_areas[0].get("side") or "")
    return None


def inventory_change_map(hand_report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for change in ((hand_report.get("inventory_sanitization") or {}).get("changes") or []):
        owner = str(change.get("owner") or "")
        piece = str(change.get("piece") or "")
        if owner and piece:
            result[(owner, piece)] = change
    return result


def attached_digit_rect_keys(items: list[dict[str, Any]]) -> set[tuple[int, int, int, int]]:
    keys: set[tuple[int, int, int, int]] = set()
    for item in items:
        for digit in item.get("digits") or []:
            key = rect_key(digit.get("rect"))
            if key is not None:
                keys.add(key)
    return keys


def item_status(owner: str, piece: str, expected: dict[str, dict[str, int]], actual: dict[str, dict[str, int]]) -> str:
    if expected[owner][piece] == actual[owner][piece]:
        return "ok"
    return "ng"


def build_draft(
    analysis_dir: Path,
    report_path: Path,
    labels_dir: Path,
) -> dict[str, Any]:
    app, piece_style, sample = report_group(analysis_dir, report_path)
    report = read_json(report_path)
    label_path = find_label_path(labels_dir, sample)
    label = load_position_label(label_path, require_hands=False) if label_path.exists() else None
    expected = hand_counts((label or {}).get("hands"))
    actual = hand_counts(report.get("hands"))
    hand_report = report.get("hand_recognition") or {}
    image_path = report.get("image")
    width, height = load_image_size(image_path)
    areas = hand_report.get("areas") or []
    changes = inventory_change_map(hand_report)

    items = []
    for entry in hand_report.get("pieces") or []:
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        rects = entry.get("rects") or []
        change = changes.get((owner, piece))
        count = int(entry.get("count") or 0)
        items.append(
            {
                "source": "recognized",
                "owner": owner,
                "owner_text": OWNER_TEXT.get(owner, owner),
                "side": entry.get("side") or infer_side(owner, rects, areas),
                "piece": piece,
                "piece_text": PIECE_TEXT.get(piece, piece),
                "count": count,
                "raw_count_before_sanitize": int(change.get("before", count)) if change else count,
                "expected_count": expected.get(owner, {}).get(piece, 0),
                "actual_count": actual.get(owner, {}).get(piece, 0),
                "status": item_status(owner, piece, expected, actual) if owner in expected and piece in HAND_PIECES else "ng",
                "count_source": entry.get("count_source"),
                "confidence": entry.get("confidence"),
                "ambiguous": bool(entry.get("ambiguous")),
                "rects": rects,
                "digits": entry.get("digits") or [],
                "inventory_sanitization": {"applied": bool(change), **(change or {})},
                "inventory_completion": entry.get("inventory_completion") or [],
                "review_status": "pending",
            }
        )

    diff_rows = count_diff_rows(expected, actual)
    missing_items = [
        {
            "owner": row["owner"],
            "owner_text": OWNER_TEXT.get(row["owner"], row["owner"]),
            "piece": row["piece"],
            "piece_text": PIECE_TEXT.get(row["piece"], row["piece"]),
            "missing_count": -int(row["diff_actual_minus_expected"]),
            "expected": row["expected"],
            "actual": row["actual"],
            "source": "missing_expected",
            "count": -int(row["diff_actual_minus_expected"]),
            "rects": [],
            "digits": [],
            "detection_status": "not_detected",
            "review_status": "needs_box_or_confirm_absent",
        }
        for row in diff_rows
        if int(row["diff_actual_minus_expected"]) < 0
    ]
    extra_items = [
        {
            "owner": row["owner"],
            "owner_text": OWNER_TEXT.get(row["owner"], row["owner"]),
            "piece": row["piece"],
            "piece_text": PIECE_TEXT.get(row["piece"], row["piece"]),
            "extra_count": int(row["diff_actual_minus_expected"]),
            "expected": row["expected"],
            "actual": row["actual"],
            "source": "extra_detected",
            "count": int(row["diff_actual_minus_expected"]),
        }
        for row in diff_rows
        if int(row["diff_actual_minus_expected"]) > 0
    ]

    recognized_keys = {(item["owner"], item["piece"]) for item in items}
    inventory_removed_items = []
    for (owner, piece), change in changes.items():
        if (owner, piece) in recognized_keys:
            continue
        inventory_removed_items.append(
            {
                "source": "inventory_removed",
                "owner": owner,
                "owner_text": OWNER_TEXT.get(owner, owner),
                "side": infer_side(owner, [], areas),
                "piece": piece,
                "piece_text": PIECE_TEXT.get(piece, piece),
                "count": int(change.get("after") or 0),
                "raw_count_before_sanitize": int(change.get("before") or 0),
                "expected_count": expected.get(owner, {}).get(piece, 0),
                "actual_count": actual.get(owner, {}).get(piece, 0),
                "status": "ng" if expected.get(owner, {}).get(piece, 0) else "ok",
                "count_source": "inventory_sanitization",
                "confidence": change.get("confidence"),
                "ambiguous": False,
                "rects": [],
                "digits": [],
                "inventory_sanitization": {"applied": True, **change},
                "review_status": "check_removed_candidate",
            }
        )

    unknown_items = []
    for proposal in hand_report.get("unknown") or []:
        owner = str(proposal.get("owner") or "")
        rect = proposal.get("rect")
        best_piece = proposal.get("best_piece")
        unknown_items.append(
            {
                "source": "unknown_proposal",
                "owner": owner,
                "owner_text": OWNER_TEXT.get(owner, owner),
                "side": proposal.get("side") or infer_side(owner, [rect], areas),
                "piece": None,
                "piece_text": "未確定",
                "best_piece": best_piece,
                "best_piece_text": PIECE_TEXT.get(str(best_piece), str(best_piece)) if best_piece else "",
                "confidence": proposal.get("confidence"),
                "proposal_source": proposal.get("proposal_source"),
                "owner_conflict": bool(proposal.get("owner_conflict")),
                "rects": [rect] if rect else [],
                "digits": [],
                "material": proposal.get("material") or {},
                "candidates": proposal.get("candidates") or [],
                "review_status": "needs_label",
            }
        )

    all_hand_items = items + inventory_removed_items + unknown_items + missing_items
    digit_keys = attached_digit_rect_keys(items)
    unassigned_digits = [
        digit for digit in (hand_report.get("digits") or []) if rect_key(digit.get("rect")) not in digit_keys
    ]

    return {
        "schema_version": 2,
        "sample": sample,
        "app": app,
        "piece_style": piece_style,
        "image": image_path,
        "image_size": [width, height],
        "report": str(report_path),
        "label": str(label_path) if label_path.exists() else None,
        "hands_expected": expected,
        "hands_actual": actual,
        "hands_detected": hand_counts(hand_report.get("hands")),
        "hands_before_inventory_sanitize": hand_counts(hand_report.get("hands_before_inventory_sanitize")),
        "hand_errors": diff_rows,
        "hand_items": all_hand_items,
        "items": items,
        "inventory_removed_items": inventory_removed_items,
        "unknown_items": unknown_items,
        "missing_items": missing_items,
        "extra_items": extra_items,
        "areas": areas,
        "unknown_proposals": hand_report.get("unknown") or [],
        "digits": hand_report.get("digits") or [],
        "unassigned_digits": unassigned_digits,
        "inventory_sanitization": hand_report.get("inventory_sanitization") or {},
        "inventory_completion": hand_report.get("inventory_completion") or {},
    }


def hands_text(hands: dict[str, dict[str, int]]) -> str:
    parts = []
    for owner in ("black", "white"):
        pieces = []
        for piece in HAND_PIECES:
            count = int(hands[owner][piece])
            if count:
                suffix = f"{count}枚" if count > 1 else "1枚"
                pieces.append(f"{PIECE_TEXT[piece]}{suffix}")
        parts.append(f"{OWNER_TEXT[owner]}: " + ("、".join(pieces) if pieces else "なし"))
    return " / ".join(parts)


def render_boxes(draft: dict[str, Any]) -> str:
    width, height = draft.get("image_size") or [1, 1]
    chunks: list[str] = []
    for area in draft.get("areas") or []:
        rect = area.get("rect")
        if not rect:
            continue
        label = f"{OWNER_TEXT.get(area.get('owner'), area.get('owner'))}領域 {area.get('side')} {area.get('evidence')}"
        chunks.append(box_html(rect, width, height, BOX_COLORS["area"], label, "area"))
    for item in draft.get("items") or []:
        color = BOX_COLORS["recognized_ok"] if item.get("status") == "ok" else BOX_COLORS["recognized_ng"]
        if item.get("source") == "inventory_removed":
            color = BOX_COLORS["inventory_removed"]
        label = (
            f"{item.get('owner_text')} {item.get('piece_text')} "
            f"認識{item.get('actual_count')} / 正解{item.get('expected_count')}"
        )
        for rect in item.get("rects") or []:
            chunks.append(box_html(rect, width, height, color, label, "recognized"))
        for digit in item.get("digits") or []:
            rect = digit.get("rect")
            if rect:
                chunks.append(box_html(rect, width, height, BOX_COLORS["digit"], f"数字 {digit.get('digit')}", "digit"))
    for proposal in draft.get("unknown_proposals") or []:
        rect = proposal.get("rect")
        if not rect:
            continue
        label = (
            f"未確定候補 {PIECE_TEXT.get(str(proposal.get('best_piece')), str(proposal.get('best_piece')))} "
            f"{proposal.get('confidence')}"
        )
        chunks.append(box_html(rect, width, height, BOX_COLORS["unknown"], label, "unknown"))
    for digit in draft.get("unassigned_digits") or []:
        rect = digit.get("rect")
        if rect:
            chunks.append(box_html(rect, width, height, BOX_COLORS["digit"], f"未割当数字 {digit.get('digit')}", "digit"))
    return "\n".join(chunks)


def box_html(rect: list[int], width: int, height: int, color: str, label: str, class_name: str) -> str:
    return (
        f'<div class="box {html.escape(class_name)}" '
        f'style="{rect_percent(rect, width, height)}border-color:{html.escape(color)};" '
        f'title="{html.escape(label, quote=True)}">'
        f'<span>{html.escape(label)}</span></div>'
    )


def render_diff_list(draft: dict[str, Any]) -> str:
    if not draft.get("hand_errors"):
        return '<p class="ok">持ち駒は正解ラベルと一致しています。</p>'
    lines = ['<ul class="diffs">']
    for row in draft.get("hand_errors") or []:
        owner = OWNER_TEXT.get(row["owner"], row["owner"])
        piece = PIECE_TEXT.get(row["piece"], row["piece"])
        diff = int(row["diff_actual_minus_expected"])
        if diff < 0:
            note = f"{-diff}枚足りません"
        else:
            note = f"{diff}枚多いです"
        lines.append(
            "<li><b>{owner} {piece}</b>: 正解 {expected}, 認識 {actual}（{note}）</li>".format(
                owner=html.escape(owner),
                piece=html.escape(piece),
                expected=html.escape(str(row["expected"])),
                actual=html.escape(str(row["actual"])),
                note=html.escape(note),
            )
        )
    lines.append("</ul>")
    return "\n".join(lines)


def render_review_item_list(draft: dict[str, Any]) -> str:
    unknown_items = draft.get("unknown_items") or []
    missing_items = draft.get("missing_items") or []
    inventory_removed_items = draft.get("inventory_removed_items") or []
    completion_changes = (draft.get("inventory_completion") or {}).get("changes") or []
    if not unknown_items and not missing_items and not inventory_removed_items and not completion_changes:
        return '<p class="ok">追加確認が必要な持ち駒候補はありません。</p>'
    lines = ['<ul class="review-items">']
    for item in missing_items:
        lines.append(
            "<li><b>未検出</b>: {owner} {piece} が {count}枚足りません。</li>".format(
                owner=html.escape(item.get("owner_text", "")),
                piece=html.escape(item.get("piece_text", "")),
                count=html.escape(str(item.get("missing_count", ""))),
            )
        )
    for item in unknown_items:
        lines.append(
            "<li><b>未確定候補</b>: {owner} 領域、候補 {piece}、信頼度 {confidence}</li>".format(
                owner=html.escape(item.get("owner_text", "")),
                piece=html.escape(item.get("best_piece_text", "")),
                confidence=html.escape(str(item.get("confidence", ""))),
            )
        )
    for item in inventory_removed_items:
        lines.append(
            "<li><b>在庫補正で削除</b>: {owner} {piece} raw {before}枚 -> {after}枚。</li>".format(
                owner=html.escape(item.get("owner_text", "")),
                piece=html.escape(item.get("piece_text", "")),
                before=html.escape(str(item.get("raw_count_before_sanitize", ""))),
                after=html.escape(str(item.get("count", ""))),
            )
        )
    for change in completion_changes:
        owner = OWNER_TEXT.get(change.get("owner"), str(change.get("owner") or ""))
        piece = PIECE_TEXT.get(change.get("piece"), str(change.get("piece") or ""))
        lines.append(
            "<li><b>在庫補完で追加</b>: {owner} {piece} を {count}枚追加。根拠: {source}</li>".format(
                owner=html.escape(owner),
                piece=html.escape(piece),
                count=html.escape(str(change.get("added", ""))),
                source=html.escape(str(change.get("source", ""))),
            )
        )
    lines.append("</ul>")
    return "\n".join(lines)


def render_sample_card(draft: dict[str, Any]) -> str:
    image_uri = path_uri(draft.get("image"))
    error_count = len(draft.get("hand_errors") or [])
    recognized_count = len(draft.get("items") or [])
    unknown_count = len(draft.get("unknown_proposals") or [])
    digit_count = len(draft.get("digits") or [])
    return f"""
<section class="card {'has-error' if error_count else 'ok-card'}">
  <h2>{html.escape(draft['sample'])}</h2>
  <p class="meta">{html.escape(draft['app'])} / {html.escape(draft['piece_style'])} / 持ち駒不一致 {error_count}件 / 認識候補 {recognized_count} / 未確定候補 {unknown_count} / 数字候補 {digit_count}</p>
  <div class="layout">
    <div>
      <h3>画像と検出枠</h3>
      <div class="image-wrap">
        <img src="{html.escape(image_uri, quote=True)}" alt="">
        {render_boxes(draft)}
      </div>
      <div class="legend">
        <span class="area-dot">青: 持ち駒領域</span>
        <span class="ok-dot">緑: 一致候補</span>
        <span class="ng-dot">赤: 不一致候補</span>
        <span class="unknown-dot">橙: 未確定候補</span>
        <span class="digit-dot">紫: 数字</span>
      </div>
    </div>
    <div>
      <h3>持ち駒比較</h3>
      <p><b>正解ラベル</b>: {html.escape(hands_text(draft['hands_expected']))}</p>
      <p><b>認識結果</b>: {html.escape(hands_text(draft['hands_actual']))}</p>
      {render_diff_list(draft)}
      <h3>確認ポイント</h3>
      {render_review_item_list(draft)}
      <h3>JSON下書き</h3>
      <p class="path">{html.escape(str(draft.get('draft_path', '')))}</p>
    </div>
  </div>
</section>
"""


def render_html(drafts: list[dict[str, Any]], only_errors: bool) -> str:
    shown = [draft for draft in drafts if (not only_errors or draft.get("hand_errors"))]
    cards = "\n".join(render_sample_card(draft) for draft in shown)
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>持ち駒認識レビュー</title>
<style>
body {{ margin: 24px; font-family: system-ui, "Yu Gothic", sans-serif; background: #f5f1e9; color: #241f18; }}
h1 {{ margin-bottom: 4px; }}
.lead {{ max-width: 1100px; line-height: 1.7; color: #5f574b; }}
.card {{ background: #fffaf0; border: 1px solid #d8c8aa; border-radius: 8px; padding: 16px; margin: 18px 0; }}
.card.has-error {{ border-left: 6px solid #d0362f; }}
.card.ok-card {{ border-left: 6px solid #1d8f4d; }}
.meta, .path {{ color: #6c6254; }}
.layout {{ display: grid; grid-template-columns: minmax(320px, 560px) minmax(300px, 1fr); gap: 18px; align-items: start; }}
.image-wrap {{ position: relative; width: min(100%, 520px); background: #111; border: 1px solid #c9b897; }}
.image-wrap img {{ display: block; width: 100%; height: auto; }}
.box {{ position: absolute; box-sizing: border-box; border: 3px solid; pointer-events: auto; background: rgba(255,255,255,0.04); }}
.box span {{ position: absolute; left: 0; top: -1.7em; max-width: 220px; font-size: 11px; line-height: 1.2; background: rgba(36,31,24,0.88); color: #fff; padding: 2px 4px; white-space: nowrap; }}
.box.area {{ border-style: dashed; opacity: 0.75; }}
.box.digit {{ border-width: 2px; }}
.diffs {{ line-height: 1.65; padding-left: 1.3em; }}
.ok {{ color: #176b3a; font-weight: 700; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; margin-top: 8px; font-size: 12px; }}
.area-dot {{ color: #2276d2; }} .ok-dot {{ color: #1d8f4d; }} .ng-dot {{ color: #d0362f; }} .unknown-dot {{ color: #e08a00; }} .digit-dot {{ color: #7f44d6; }}
@media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>持ち駒認識レビュー</h1>
<p class="lead">青枠は検出された持ち駒領域、緑/赤枠は認識済みの持ち駒候補、橙枠は未確定候補、紫枠は数字候補です。赤枠や「足りません」と表示される箇所を優先して確認します。ここで作ったJSONは、次の段階で持ち駒専用学習データに変換するための下書きです。</p>
<p class="lead">表示件数: {len(shown)} / 全下書き: {len(drafts)} / エラーのみ表示: {"はい" if only_errors else "いいえ"}</p>
{cards}
</body>
</html>
"""


def build_reviews(args: argparse.Namespace) -> dict[str, Any]:
    report_paths = analysis_report_paths(args.analysis_dir)
    if not report_paths:
        raise FileNotFoundError(f"no reports found under {args.analysis_dir}")
    drafts: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for report_path in report_paths:
        draft = build_draft(args.analysis_dir, report_path, args.labels_dir)
        draft_path = args.out / "drafts" / draft["app"] / draft["piece_style"] / f"{draft['sample']}.json"
        draft["draft_path"] = str(draft_path)
        write_json(draft_path, draft)
        drafts.append(draft)
        summary_rows.append(
            {
                "app": draft["app"],
                "piece_style": draft["piece_style"],
                "sample": draft["sample"],
                "hand_error_count": len(draft["hand_errors"]),
                "missing_item_count": sum(int(item["missing_count"]) for item in draft["missing_items"]),
                "extra_item_count": sum(int(item["extra_count"]) for item in draft["extra_items"]),
                "recognized_item_groups": len(draft["items"]),
                "inventory_removed_groups": len(draft["inventory_removed_items"]),
                "inventory_completion_changes": len((draft.get("inventory_completion") or {}).get("changes") or []),
                "unknown_proposals": len(draft["unknown_proposals"]),
                "digits": len(draft["digits"]),
                "unassigned_digits": len(draft["unassigned_digits"]),
                "draft": str(draft_path),
            }
        )
    write_csv(
        args.out / "hand_items_summary.csv",
        summary_rows,
        [
            "app",
            "piece_style",
            "sample",
            "hand_error_count",
            "missing_item_count",
            "extra_item_count",
            "recognized_item_groups",
            "inventory_removed_groups",
            "inventory_completion_changes",
            "unknown_proposals",
            "digits",
            "unassigned_digits",
            "draft",
        ],
    )
    html_path = args.out / "hand_items_review.html"
    html_path.write_text(render_html(drafts, args.only_errors), encoding="utf-8")
    summary = {
        "analysis_dir": str(args.analysis_dir),
        "labels_dir": str(args.labels_dir),
        "draft_count": len(drafts),
        "samples_with_hand_errors": sum(1 for draft in drafts if draft["hand_errors"]),
        "hand_error_rows": sum(len(draft["hand_errors"]) for draft in drafts),
        "missing_items": sum(sum(int(item["missing_count"]) for item in draft["missing_items"]) for draft in drafts),
        "extra_items": sum(sum(int(item["extra_count"]) for item in draft["extra_items"]) for draft in drafts),
        "html": str(html_path),
        "summary_csv": str(args.out / "hand_items_summary.csv"),
    }
    write_json(args.out / "hand_items_review_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="持ち駒専用の教師データ下書きJSONと目視確認HTMLを作成します。")
    parser.add_argument("analysis_dir", type=Path, help="run_analysis_by_app_piece_style.py の出力フォルダ。")
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards_by_app_piece_style"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/hand_items_review"))
    parser.add_argument("--only-errors", action="store_true", help="HTMLでは持ち駒不一致がある画像だけ表示します。")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = build_reviews(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
