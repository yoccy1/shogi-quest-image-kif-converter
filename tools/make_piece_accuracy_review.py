from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_piece_recognition import (
    candidate_key,
    evaluate_one,
    identity,
    load_report,
    prediction_key,
    top_candidates,
)
from position_label_utils import HAND_PIECES, find_label_path, load_position_label, square_name


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
PIECE_TEXT = {
    "OU": "\u7389",
    "HI": "\u98db",
    "KA": "\u89d2",
    "KI": "\u91d1",
    "GI": "\u9280",
    "KE": "\u6842",
    "KY": "\u9999",
    "FU": "\u6b69",
    "RY": "\u7adc",
    "UM": "\u99ac",
    "NG": "\u6210\u9280",
    "NK": "\u6210\u6842",
    "NY": "\u6210\u9999",
    "TO": "\u3068",
}
COLOR_MARK = {"black": "\u25b2", "white": "\u25b3"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_reports_dir() -> Path:
    return ROOT / "tools" / "out" / "android_device_eval" / "android_eval_full86_20260509_final"


def default_labels_dir() -> Path:
    return ROOT / "tools" / "samples" / "labels" / "boards_by_app_piece_style"


def default_screenshots_dir() -> Path:
    return ROOT / "tools" / "samples" / "screenshots_by_app_piece_style"


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def as_uri(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().as_uri()
    except ValueError:
        return ""


def report_entries(reports_dir: Path) -> list[dict[str, Any]]:
    manifest = reports_dir / "manifest.csv"
    if manifest.exists():
        entries: list[dict[str, Any]] = []
        for row in read_csv(manifest):
            report = Path(row.get("report") or "")
            if report.exists():
                entries.append(
                    {
                        "app": row.get("app") or "",
                        "piece_style": row.get("piece_style") or row.get("glyph") or "",
                        "sample": row.get("sample") or report.parent.name,
                        "image": Path(row["image"]) if row.get("image") else None,
                        "report": report,
                        "seconds": row.get("seconds") or "",
                    },
                )
        return entries

    if reports_dir.is_file():
        return [
            {
                "app": "",
                "piece_style": "",
                "sample": reports_dir.parent.name,
                "image": None,
                "report": reports_dir,
                "seconds": "",
            },
        ]

    paths = sorted(reports_dir.glob("*/*/*/piece_report.json"))
    if not paths:
        direct = reports_dir / "piece_report.json"
        paths = [direct] if direct.exists() else []

    entries = []
    for report in paths:
        sample_dir = report.parent
        style_dir = sample_dir.parent
        app_dir = style_dir.parent
        app = app_dir.name if app_dir != style_dir else ""
        piece_style = style_dir.name if style_dir != sample_dir else ""
        entries.append(
            {
                "app": app,
                "piece_style": piece_style,
                "sample": sample_dir.name,
                "image": None,
                "report": report,
                "seconds": "",
            },
        )
    return entries


def resolve_image_path(
    entry: dict[str, Any],
    label_path: Path | None,
    label: dict[str, Any] | None,
    screenshots_dir: Path,
) -> Path | None:
    raw_image = entry.get("image")
    if isinstance(raw_image, Path) and raw_image.exists():
        return raw_image

    if label_path and label:
        image_value = label.get("image")
        if image_value:
            image_path = Path(str(image_value))
            if image_path.is_absolute() and image_path.exists():
                return image_path
            candidate = (label_path.parent / image_path).resolve()
            if candidate.exists():
                return candidate

    app = str(entry.get("app") or "")
    piece_style = str(entry.get("piece_style") or "")
    sample = str(entry.get("sample") or "")
    base = screenshots_dir / app / piece_style
    for extension in IMAGE_EXTENSIONS:
        candidate = base / f"{sample}{extension}"
        if candidate.exists():
            return candidate
    for candidate in base.glob(f"{sample}.*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def display_identity(value: str | None) -> str:
    if not value or value in {"none", "unknown"}:
        return "?"
    if value == "empty":
        return ""
    color, separator, piece = value.partition(":")
    if not separator:
        return PIECE_TEXT.get(value, value)
    return f"{COLOR_MARK.get(color, '')}{PIECE_TEXT.get(piece, piece)}"


def display_cell(cell: dict[str, Any] | None) -> str:
    if not cell:
        return "?"
    state = cell.get("state")
    if state == "empty":
        return ""
    if state == "piece":
        return display_identity(identity(cell.get("color"), cell.get("piece")))
    candidates = top_candidates(cell)
    if candidates:
        return "?" + display_identity(candidate_key(candidates[0]))
    return "?"


def expected_key(cell: dict[str, Any]) -> str:
    if cell["state"] == "empty":
        return "empty"
    if cell["state"] == "unknown":
        return "unknown"
    return identity(cell.get("color"), cell.get("piece"))


def cell_status(expected: dict[str, Any], actual: dict[str, Any] | None) -> str:
    if expected["state"] == "unknown":
        return "ignored"
    if not actual:
        return "missing"
    if expected["state"] == "empty":
        return "ok-empty" if actual.get("state") == "empty" else "false-piece"
    if actual.get("state") == "empty":
        return "false-empty"
    if actual.get("state") == "unknown":
        candidates = [candidate_key(candidate) for candidate in top_candidates(actual)[:3]]
        return "unknown-top3" if expected_key(expected) in candidates else "unknown"
    if actual.get("state") == "piece" and actual.get("color") == expected.get("color") and actual.get("piece") == expected.get("piece"):
        return "ok-piece"
    if prediction_key(actual) == expected_key(expected):
        return "top1-only"
    candidates = [candidate_key(candidate) for candidate in top_candidates(actual)[:3]]
    if expected_key(expected) in candidates:
        return "top3-only"
    return "wrong"


def status_label(status: str) -> str:
    return {
        "ok-piece": "OK",
        "ok-empty": "empty OK",
        "top1-only": "top1 OK",
        "top3-only": "top3 only",
        "unknown-top3": "unknown/top3",
        "unknown": "unknown",
        "false-empty": "piece->empty",
        "false-piece": "empty->piece",
        "wrong": "wrong",
        "missing": "missing",
        "ignored": "ignored",
    }.get(status, status)


def score_text(actual: dict[str, Any] | None) -> str:
    if not actual:
        return ""
    candidates = top_candidates(actual)
    top = candidates[0] if candidates else None
    score = top.get("score") if top else actual.get("confidence")
    if isinstance(score, (int, float)):
        return f"{score:.3f}"
    return ""


def candidate_summary(actual: dict[str, Any] | None, limit: int = 3) -> str:
    if not actual:
        return ""
    parts = []
    for candidate in top_candidates(actual)[:limit]:
        score = candidate.get("score")
        score_part = f" {score:.3f}" if isinstance(score, (int, float)) else ""
        source = candidate.get("source")
        source_part = f" [{source}]" if source else ""
        parts.append(f"{display_identity(candidate_key(candidate))}{score_part}{source_part}")
    return " / ".join(parts)


def build_cell_rows(
    label: dict[str, Any],
    report: dict[str, Any],
    sample_meta: dict[str, str],
) -> list[dict[str, Any]]:
    actual_by_pos = {(int(cell["row"]), int(cell["col"])): cell for cell in report.get("cells") or []}
    rows = []
    for expected in label["cells"]:
        row = int(expected["row"])
        col = int(expected["col"])
        actual = actual_by_pos.get((row, col))
        status = cell_status(expected, actual)
        rows.append(
            {
                **sample_meta,
                "row": row,
                "col": col,
                "square": expected.get("square") or square_name(row, col),
                "status": status,
                "expected": expected_key(expected),
                "actual_state": actual.get("state") if actual else "missing",
                "confirmed": identity(actual.get("color"), actual.get("piece")) if actual and actual.get("state") == "piece" else (actual.get("state") if actual else "missing"),
                "top1": prediction_key(actual) if actual else "missing",
                "top3": " | ".join(candidate_key(candidate) for candidate in top_candidates(actual or {})[:3]),
                "score": score_text(actual),
                "candidates": candidate_summary(actual),
                "expected_label": display_identity(expected_key(expected)),
                "actual_label": display_cell(actual),
            },
        )
    return rows


def evaluate_entry(
    entry: dict[str, Any],
    labels_dir: Path,
    screenshots_dir: Path,
    include_hands: bool,
    strict_leak_guard: bool,
    require_excluded_source: bool,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    report_path = Path(entry["report"])
    app = str(entry.get("app") or "")
    piece_style = str(entry.get("piece_style") or "")
    sample = str(entry.get("sample") or report_path.parent.name)
    label_path = find_label_path(labels_dir, sample, app or None, piece_style or None)
    label = load_position_label(label_path, require_hands=False) if label_path.exists() else None
    report = load_report(report_path)
    image_path = resolve_image_path(entry, label_path if label_path.exists() else None, label, screenshots_dir)

    evaluation: dict[str, Any] | None = None
    evaluation_error = ""
    if label_path.exists():
        try:
            evaluation = evaluate_one(
                report_path,
                label_path,
                high_confidence_threshold,
                include_hands=include_hands,
                strict_leak_guard=strict_leak_guard,
                forbidden_sources=(sample,),
                require_excluded_source=require_excluded_source,
            )
        except Exception as exc:
            evaluation_error = f"{type(exc).__name__}: {exc}"
            try:
                evaluation = evaluate_one(
                    report_path,
                    label_path,
                    high_confidence_threshold,
                    include_hands=False,
                    strict_leak_guard=strict_leak_guard,
                    forbidden_sources=(sample,),
                    require_excluded_source=require_excluded_source,
                )
            except Exception as fallback_exc:
                evaluation_error = f"{evaluation_error}; fallback={type(fallback_exc).__name__}: {fallback_exc}"
                evaluation = None
    else:
        evaluation_error = f"missing label: {label_path}"

    sample_meta = {
        "app": app,
        "piece_style": piece_style,
        "sample": sample,
    }
    cell_rows = build_cell_rows(label, report, sample_meta) if label else []
    status_counts = Counter(row["status"] for row in cell_rows)
    metrics = evaluation.get("metrics") if evaluation else {}
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    return {
        "app": app,
        "piece_style": piece_style,
        "sample": sample,
        "report_path": report_path,
        "label_path": label_path if label_path.exists() else None,
        "image_path": image_path,
        "seconds": entry.get("seconds") or "",
        "report": report,
        "label": label,
        "evaluation": evaluation,
        "evaluation_error": evaluation_error,
        "metrics": metrics,
        "cell_rows": cell_rows,
        "status_counts": dict(status_counts),
        "model": {
            "excluded_source": model.get("excluded_source"),
            "excluded_sources": model.get("excluded_sources"),
            "no_leak_options": model.get("no_leak_options"),
        },
    }


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return "" if value is None else value


def summarize_entries(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for item in items:
        metrics = item.get("metrics") or {}
        group_key = (item["app"], item["piece_style"])
        totals["samples"] += 1
        groups[group_key]["samples"] += 1
        if not metrics:
            totals["skipped_samples"] += 1
            groups[group_key]["skipped_samples"] += 1
            continue
        true_piece = int(metrics.get("true_piece") or 0)
        true_empty = int(metrics.get("true_empty") or 0)
        totals["true_piece"] += true_piece
        totals["true_empty"] += true_empty
        groups[group_key]["true_piece"] += true_piece
        groups[group_key]["true_empty"] += true_empty
        for key in (
            "errors",
            "hand_errors",
            "leak_errors",
            "unknown_on_piece",
            "false_empty_on_piece",
            "false_piece_on_empty",
            "high_confidence_errors",
            "ignored_unknown",
        ):
            value = int(metrics.get(key) or 0)
            totals[key] += value
            groups[group_key][key] += value
        for key in (
            "confirmed_identity_accuracy",
            "top1_identity_accuracy",
            "top3_contains_identity_accuracy",
            "empty_accuracy",
            "piece_presence_accuracy",
        ):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                weight = true_empty if key == "empty_accuracy" else true_piece
                totals[f"{key}_weighted"] += round(value * weight)
                totals[f"{key}_weight"] += weight
                groups[group_key][f"{key}_weighted"] += round(value * weight)
                groups[group_key][f"{key}_weight"] += weight

    def rates(counter: Counter[str]) -> dict[str, Any]:
        output = dict(counter)
        for key in (
            "confirmed_identity_accuracy",
            "top1_identity_accuracy",
            "top3_contains_identity_accuracy",
            "empty_accuracy",
            "piece_presence_accuracy",
        ):
            weight = counter.get(f"{key}_weight", 0)
            weighted = counter.get(f"{key}_weighted", 0)
            output[key] = round(weighted / weight, 4) if weight else None
            output.pop(f"{key}_weight", None)
            output.pop(f"{key}_weighted", None)
        return output

    return {
        "overall": rates(totals),
        "groups": [
            {"app": app, "piece_style": piece_style, **rates(counter)}
            for (app, piece_style), counter in sorted(groups.items())
        ],
    }


def render_stat(label: str, value: Any, class_name: str = "") -> str:
    return (
        f'<div class="stat {html.escape(class_name)}">'
        f'<span>{html.escape(label)}</span>'
        f"<b>{html.escape(str(value))}</b>"
        "</div>"
    )


def render_group_table(groups: list[dict[str, Any]]) -> str:
    rows = []
    for group in groups:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(group.get('app') or ''))}</td>"
            f"<td>{html.escape(str(group.get('piece_style') or ''))}</td>"
            f"<td>{group.get('samples', 0)}</td>"
            f"<td>{group.get('errors', 0)}</td>"
            f"<td>{group.get('hand_errors', 0)}</td>"
            f"<td>{group.get('leak_errors', 0)}</td>"
            f"<td>{group.get('confirmed_identity_accuracy')}</td>"
            f"<td>{group.get('top1_identity_accuracy')}</td>"
            f"<td>{group.get('top3_contains_identity_accuracy')}</td>"
            "</tr>"
        )
    return (
        '<table class="group-table"><thead><tr>'
        "<th>App</th><th>Style</th><th>Samples</th><th>Board errors</th>"
        "<th>Hand errors</th><th>Leak errors</th><th>Confirmed</th><th>Top1</th><th>Top3</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def render_board(item: dict[str, Any]) -> str:
    row_by_pos = {(int(row["row"]), int(row["col"])): row for row in item["cell_rows"]}
    chunks = ['<div class="board" role="grid">']
    for row_index in range(1, 10):
        for col_index in range(1, 10):
            row = row_by_pos.get((row_index, col_index))
            if not row:
                chunks.append('<div class="cell missing"><span class="piece">?</span></div>')
                continue
            title = (
                f"{row['square']} | status={status_label(row['status'])} | "
                f"expected={row['expected']} | confirmed={row['confirmed']} | "
                f"top1={row['top1']} | top3={row['top3']}"
            )
            chunks.append(
                '<div class="cell {status}" title="{title}">'
                '<span class="sq">{square}</span>'
                '<span class="piece">{actual}</span>'
                '<span class="expected">exp {expected}</span>'
                '<span class="score">{score}</span>'
                "</div>".format(
                    status=html.escape(row["status"]),
                    title=html.escape(title, quote=True),
                    square=html.escape(str(row["square"])),
                    actual=html.escape(str(row["actual_label"])),
                    expected=html.escape(str(row["expected_label"])),
                    score=html.escape(str(row["score"])),
                ),
            )
    chunks.append("</div>")
    return "\n".join(chunks)


def render_error_table(item: dict[str, Any]) -> str:
    rows = [
        row
        for row in item["cell_rows"]
        if row["status"] not in {"ok-piece", "ok-empty", "ignored"}
    ]
    if not rows and int((item.get("metrics") or {}).get("hand_errors") or 0) == 0 and int((item.get("metrics") or {}).get("leak_errors") or 0) == 0:
        return '<p class="ok">No board, hand, or leak errors in this report.</p>'
    chunks = [
        '<table class="error-table"><thead><tr>'
        "<th>Square</th><th>Status</th><th>Expected</th><th>Confirmed</th><th>Top1</th><th>Top3</th><th>Score</th>"
        "</tr></thead><tbody>"
    ]
    for row in rows:
        chunks.append(
            "<tr>"
            f"<td>{html.escape(str(row['square']))}</td>"
            f"<td>{html.escape(status_label(str(row['status'])))}</td>"
            f"<td>{html.escape(str(row['expected']))}</td>"
            f"<td>{html.escape(str(row['confirmed']))}</td>"
            f"<td>{html.escape(str(row['top1']))}</td>"
            f"<td>{html.escape(str(row['top3']))}</td>"
            f"<td>{html.escape(str(row['score']))}</td>"
            "</tr>"
        )
    chunks.append("</tbody></table>")

    hands = (item.get("evaluation") or {}).get("hands") or {}
    hand_errors = hands.get("error_details") or []
    if hand_errors:
        chunks.append('<h4>Hand errors</h4><ul class="compact-list">')
        for error in hand_errors:
            chunks.append(
                "<li>"
                f"{html.escape(str(error.get('owner')))}:{html.escape(str(error.get('piece')))} "
                f"expected={html.escape(str(error.get('expected')))} "
                f"actual={html.escape(str(error.get('actual')))}"
                "</li>"
            )
        chunks.append("</ul>")
    leak_errors = (item.get("evaluation") or {}).get("leak_errors") or []
    if leak_errors:
        chunks.append('<h4>Leak guard errors</h4><ul class="compact-list">')
        for error in leak_errors[:12]:
            chunks.append(f"<li>{html.escape(str(error))}</li>")
        if len(leak_errors) > 12:
            chunks.append(f"<li>... {len(leak_errors) - 12} more</li>")
        chunks.append("</ul>")
    return "\n".join(chunks)


def render_model_details(item: dict[str, Any]) -> str:
    model = item.get("model") or {}
    text = json.dumps(model, ensure_ascii=False, indent=2)
    return f"<details><summary>Model/no-leak metadata</summary><pre>{html.escape(text)}</pre></details>"


def card_classes(item: dict[str, Any]) -> str:
    metrics = item.get("metrics") or {}
    classes = ["sample-card"]
    if int(metrics.get("errors") or 0) > 0:
        classes.append("has-board-error")
    if int(metrics.get("hand_errors") or 0) > 0:
        classes.append("has-hand-error")
    if int(metrics.get("leak_errors") or 0) > 0:
        classes.append("has-leak-error")
    if item.get("evaluation_error") and not item.get("evaluation"):
        classes.append("is-skipped")
    return " ".join(classes)


def render_sample_card(item: dict[str, Any], index: int) -> str:
    metrics = item.get("metrics") or {}
    anchor = f"s{index:03d}"
    app = str(item.get("app") or "")
    style = str(item.get("piece_style") or "")
    sample = str(item.get("sample") or "")
    image_uri = as_uri(item.get("image_path"))
    report_link = as_uri(item.get("report_path"))
    label_link = as_uri(item.get("label_path"))
    image_html = (
        f'<img class="source-image" loading="lazy" src="{html.escape(image_uri, quote=True)}" alt="{html.escape(sample, quote=True)}">'
        if image_uri
        else '<div class="image-missing">image not found</div>'
    )
    eval_error = item.get("evaluation_error") or ""
    warning = f'<p class="warning">{html.escape(eval_error)}</p>' if eval_error else ""
    return f"""
<section id="{anchor}" class="{html.escape(card_classes(item))}" data-app="{html.escape(app, quote=True)}" data-style="{html.escape(style, quote=True)}" data-sample="{html.escape(sample, quote=True)}" data-board-errors="{metric_value(metrics, 'errors') or 0}" data-hand-errors="{metric_value(metrics, 'hand_errors') or 0}" data-leak-errors="{metric_value(metrics, 'leak_errors') or 0}">
  <div class="sample-head">
    <div>
      <h3>{html.escape(sample)}</h3>
      <p>{html.escape(app)} / {html.escape(style)}</p>
    </div>
    <a class="anchor-link" href="#{anchor}">#{index:03d}</a>
  </div>
  <div class="metric-row">
    {render_stat('Board errors', metric_value(metrics, 'errors'), 'danger' if int(metrics.get('errors') or 0) else 'good')}
    {render_stat('Hand errors', metric_value(metrics, 'hand_errors'), 'danger' if int(metrics.get('hand_errors') or 0) else 'good')}
    {render_stat('Leak errors', metric_value(metrics, 'leak_errors'), 'danger' if int(metrics.get('leak_errors') or 0) else 'good')}
    {render_stat('Unknown on piece', metric_value(metrics, 'unknown_on_piece'), 'warn' if int(metrics.get('unknown_on_piece') or 0) else 'good')}
    {render_stat('Confirmed acc.', metric_value(metrics, 'confirmed_identity_accuracy'))}
    {render_stat('Top1 acc.', metric_value(metrics, 'top1_identity_accuracy'))}
    {render_stat('Top3 acc.', metric_value(metrics, 'top3_contains_identity_accuracy'))}
    {render_stat('Seconds', item.get('seconds') or '')}
  </div>
  {warning}
  <div class="sample-layout">
    <div>
      <h4>Source image</h4>
      {image_html}
      <p class="links"><a href="{html.escape(report_link, quote=True)}">report</a> {('<a href="' + html.escape(label_link, quote=True) + '">label</a>') if label_link else ''}</p>
    </div>
    <div>
      <h4>Prediction board</h4>
      {render_board(item)}
    </div>
    <div>
      <h4>Review targets</h4>
      {render_error_table(item)}
      {render_model_details(item)}
    </div>
  </div>
</section>
"""


def render_html(
    reports_dir: Path,
    out_dir: Path,
    items: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    overall = summary["overall"]
    apps = sorted({str(item.get("app") or "") for item in items})
    styles = sorted({str(item.get("piece_style") or "") for item in items})
    app_options = "\n".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in apps)
    style_options = "\n".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in styles)
    cards = "\n".join(render_sample_card(item, index) for index, item in enumerate(items, start=1))
    generated_json = json.dumps(
        {
            "reports_dir": str(reports_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
            "samples": len(items),
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piece Accuracy Review</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f4ed;
  --panel: #fffdf7;
  --ink: #20231f;
  --muted: #65706a;
  --line: #d6d0c2;
  --good: #dff1e6;
  --good-line: #4d9c6c;
  --warn: #fff0bf;
  --danger: #ffd9d5;
  --danger-line: #c53f38;
  --top3: #e5e4ff;
  --empty: #e9c57c;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: system-ui, "Segoe UI", "Yu Gothic UI", sans-serif;
  line-height: 1.45;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 20;
  padding: 16px 24px;
  border-bottom: 1px solid var(--line);
  background: rgba(247, 244, 237, 0.96);
  backdrop-filter: blur(8px);
}}
h1, h2, h3, h4 {{ margin: 0; }}
h1 {{ font-size: 24px; }}
h2 {{ font-size: 18px; margin: 24px 0 10px; }}
h3 {{ font-size: 18px; }}
h4 {{ font-size: 14px; margin-bottom: 8px; color: #3d463f; }}
main {{ padding: 0 24px 32px; }}
.subtle {{ color: var(--muted); margin: 4px 0 0; }}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
  align-items: center;
}}
.toolbar input, .toolbar select {{
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 9px;
  background: white;
  color: var(--ink);
}}
.toolbar input {{ min-width: 260px; }}
.summary-grid, .metric-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 8px;
}}
.summary-grid {{ margin: 18px 0; }}
.stat {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 8px 10px;
  min-height: 58px;
}}
.stat span {{
  display: block;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}}
.stat b {{ display: block; margin-top: 4px; font-size: 18px; }}
.stat.good {{ background: var(--good); border-color: var(--good-line); }}
.stat.warn {{ background: var(--warn); }}
.stat.danger {{ background: var(--danger); border-color: var(--danger-line); }}
.group-table, .error-table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}}
th {{ background: #ece7dc; font-weight: 700; }}
.sample-card {{
  margin: 18px 0;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}}
.sample-card.has-board-error, .sample-card.has-hand-error, .sample-card.has-leak-error {{
  border-color: var(--danger-line);
}}
.sample-head {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 10px;
}}
.sample-head p {{ margin: 3px 0 0; color: var(--muted); }}
.anchor-link {{ color: #2d5b8a; text-decoration: none; font-weight: 700; }}
.sample-layout {{
  display: grid;
  grid-template-columns: minmax(220px, 360px) minmax(500px, auto) minmax(300px, 1fr);
  gap: 14px;
  align-items: start;
  margin-top: 12px;
}}
.source-image {{
  width: 100%;
  max-height: 720px;
  object-fit: contain;
  background: #181818;
  border: 1px solid var(--line);
  border-radius: 6px;
}}
.image-missing {{
  min-height: 220px;
  border: 1px dashed var(--line);
  border-radius: 6px;
  display: grid;
  place-items: center;
  color: var(--muted);
}}
.board {{
  display: grid;
  grid-template-columns: repeat(9, 58px);
  grid-template-rows: repeat(9, 58px);
  border: 2px solid #75552d;
  width: max-content;
  background: var(--empty);
}}
.cell {{
  position: relative;
  width: 58px;
  height: 58px;
  border: 1px solid #8f6b3a;
  background: #e9c57c;
  overflow: hidden;
}}
.cell.ok-piece {{ background: var(--good); }}
.cell.ok-empty {{ background: #e9c57c; }}
.cell.top1-only, .cell.top3-only, .cell.unknown-top3 {{ background: var(--top3); }}
.cell.unknown {{ background: var(--warn); }}
.cell.false-empty, .cell.false-piece, .cell.wrong, .cell.missing {{
  background: var(--danger);
  outline: 3px solid var(--danger-line);
  z-index: 1;
}}
.sq {{
  position: absolute;
  left: 3px;
  top: 2px;
  color: #755d37;
  font-size: 10px;
}}
.piece {{
  position: absolute;
  left: 3px;
  right: 3px;
  top: 15px;
  text-align: center;
  font-size: 18px;
  font-weight: 750;
  line-height: 1;
  white-space: nowrap;
}}
.expected {{
  position: absolute;
  left: 3px;
  bottom: 3px;
  max-width: 52px;
  color: #3e493f;
  font-size: 10px;
  line-height: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.score {{
  position: absolute;
  right: 3px;
  top: 2px;
  font-size: 9px;
  color: #5c675f;
}}
.links a {{ margin-right: 8px; }}
.ok {{ color: #1d7044; font-weight: 700; }}
.warning {{
  border: 1px solid var(--danger-line);
  background: var(--danger);
  border-radius: 6px;
  padding: 8px 10px;
}}
.compact-list {{
  margin: 8px 0;
  padding-left: 18px;
  font-size: 13px;
}}
details {{ margin-top: 10px; }}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: #f1ede2;
  border-radius: 6px;
  padding: 10px;
  font-size: 12px;
}}
.hidden {{ display: none; }}
@media (max-width: 1180px) {{
  .sample-layout {{ grid-template-columns: 1fr; }}
  .board {{
    grid-template-columns: repeat(9, minmax(42px, 1fr));
    grid-template-rows: repeat(9, 50px);
    width: min(100%, 522px);
  }}
  .cell {{ width: auto; height: 50px; }}
  .piece {{ font-size: 15px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Piece Accuracy Review</h1>
  <p class="subtle">Current report: {html.escape(str(reports_dir.resolve()))}</p>
  <div class="toolbar">
    <input id="sampleFilter" type="search" placeholder="Filter sample/app/style">
    <select id="appFilter"><option value="">All apps</option>{app_options}</select>
    <select id="styleFilter"><option value="">All styles</option>{style_options}</select>
    <select id="errorFilter">
      <option value="">All samples</option>
      <option value="board">Board errors only</option>
      <option value="hand">Hand errors only</option>
      <option value="leak">Leak errors only</option>
      <option value="clean">Clean samples only</option>
    </select>
  </div>
</header>
<main>
  <section>
    <h2>Overall</h2>
    <div class="summary-grid">
      {render_stat('Samples', overall.get('samples'))}
      {render_stat('Skipped', overall.get('skipped_samples', 0), 'danger' if int(overall.get('skipped_samples') or 0) else 'good')}
      {render_stat('Board errors', overall.get('errors', 0), 'danger' if int(overall.get('errors') or 0) else 'good')}
      {render_stat('Hand errors', overall.get('hand_errors', 0), 'danger' if int(overall.get('hand_errors') or 0) else 'good')}
      {render_stat('Leak errors', overall.get('leak_errors', 0), 'danger' if int(overall.get('leak_errors') or 0) else 'good')}
      {render_stat('Confirmed acc.', overall.get('confirmed_identity_accuracy'))}
      {render_stat('Top1 acc.', overall.get('top1_identity_accuracy'))}
      {render_stat('Top3 acc.', overall.get('top3_contains_identity_accuracy'))}
    </div>
    {render_group_table(summary['groups'])}
  </section>
  <section>
    <h2>Samples</h2>
    {cards}
  </section>
  <details>
    <summary>Generated metadata</summary>
    <pre>{html.escape(generated_json)}</pre>
  </details>
</main>
<script>
const cards = Array.from(document.querySelectorAll('.sample-card'));
const sampleFilter = document.getElementById('sampleFilter');
const appFilter = document.getElementById('appFilter');
const styleFilter = document.getElementById('styleFilter');
const errorFilter = document.getElementById('errorFilter');
function asNumber(value) {{
  const parsed = Number.parseInt(value || '0', 10);
  return Number.isFinite(parsed) ? parsed : 0;
}}
function refresh() {{
  const text = sampleFilter.value.trim().toLowerCase();
  const app = appFilter.value;
  const style = styleFilter.value;
  const mode = errorFilter.value;
  for (const card of cards) {{
    const haystack = `${{card.dataset.sample}} ${{card.dataset.app}} ${{card.dataset.style}}`.toLowerCase();
    const board = asNumber(card.dataset.boardErrors);
    const hand = asNumber(card.dataset.handErrors);
    const leak = asNumber(card.dataset.leakErrors);
    let visible = true;
    if (text && !haystack.includes(text)) visible = false;
    if (app && card.dataset.app !== app) visible = false;
    if (style && card.dataset.style !== style) visible = false;
    if (mode === 'board' && board === 0) visible = false;
    if (mode === 'hand' && hand === 0) visible = false;
    if (mode === 'leak' && leak === 0) visible = false;
    if (mode === 'clean' && (board > 0 || hand > 0 || leak > 0)) visible = false;
    card.classList.toggle('hidden', !visible);
  }}
}}
[sampleFilter, appFilter, styleFilter, errorFilter].forEach((control) => control.addEventListener('input', refresh));
</script>
</body>
</html>
"""


def detail_page_name(index: int) -> str:
    return f"s{index:03d}.html"


def format_link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'


def status_strip(item: dict[str, Any]) -> str:
    counts = item.get("status_counts") or {}
    order = [
        ("ok-piece", "OK"),
        ("unknown-top3", "U3"),
        ("unknown", "U"),
        ("top3-only", "T3"),
        ("top1-only", "T1"),
        ("wrong", "W"),
        ("ignored", "I"),
    ]
    parts = []
    for status, label in order:
        count = int(counts.get(status) or 0)
        if count:
            parts.append(f'<span class="status-pill {html.escape(status)}">{html.escape(label)} {count}</span>')
    return "".join(parts) or '<span class="status-pill ok-empty">no piece cells</span>'


def render_light_rows(items: list[dict[str, Any]]) -> str:
    rows = []
    for index, item in enumerate(items, start=1):
        metrics = item.get("metrics") or {}
        app = str(item.get("app") or "")
        style = str(item.get("piece_style") or "")
        sample = str(item.get("sample") or "")
        board_errors = int(metrics.get("errors") or 0)
        hand_errors = int(metrics.get("hand_errors") or 0)
        leak_errors = int(metrics.get("leak_errors") or 0)
        rows.append(
            '<tr class="sample-row" '
            f'data-app="{html.escape(app, quote=True)}" '
            f'data-style="{html.escape(style, quote=True)}" '
            f'data-sample="{html.escape(sample, quote=True)}" '
            f'data-board-errors="{board_errors}" '
            f'data-hand-errors="{hand_errors}" '
            f'data-leak-errors="{leak_errors}">'
            f'<td>{format_link("samples/" + detail_page_name(index), f"#{index:03d}")}</td>'
            f"<td>{html.escape(app)}</td>"
            f"<td>{html.escape(style)}</td>"
            f"<td>{html.escape(sample)}</td>"
            f'<td class="num {("bad" if board_errors else "good")}">{board_errors}</td>'
            f'<td class="num {("bad" if hand_errors else "good")}">{hand_errors}</td>'
            f'<td class="num {("bad" if leak_errors else "good")}">{leak_errors}</td>'
            f"<td>{html.escape(str(metric_value(metrics, 'confirmed_identity_accuracy')))}</td>"
            f"<td>{html.escape(str(metric_value(metrics, 'top1_identity_accuracy')))}</td>"
            f"<td>{html.escape(str(metric_value(metrics, 'top3_contains_identity_accuracy')))}</td>"
            f"<td>{status_strip(item)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_light_html(
    reports_dir: Path,
    items: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    overall = summary["overall"]
    apps = sorted({str(item.get("app") or "") for item in items})
    styles = sorted({str(item.get("piece_style") or "") for item in items})
    app_options = "\n".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in apps)
    style_options = "\n".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in styles)
    rows = render_light_rows(items)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piece Accuracy Review - Light</title>
<style>
:root {{
  --bg: #f7f4ed;
  --panel: #fffdf7;
  --ink: #20231f;
  --muted: #65706a;
  --line: #d6d0c2;
  --good: #dff1e6;
  --warn: #fff0bf;
  --danger: #ffd9d5;
  --top3: #e5e4ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: system-ui, "Segoe UI", "Yu Gothic UI", sans-serif;
  line-height: 1.45;
}}
header {{
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(247, 244, 237, 0.97);
}}
main {{ padding: 0 20px 28px; }}
h1 {{ margin: 0; font-size: 22px; }}
h2 {{ margin: 20px 0 10px; font-size: 17px; }}
.subtle {{ color: var(--muted); margin: 4px 0 0; font-size: 13px; }}
.toolbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}}
.toolbar input, .toolbar select {{
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 6px 9px;
  background: white;
  color: var(--ink);
}}
.toolbar input {{ min-width: 260px; }}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px;
  margin: 16px 0;
}}
.stat {{
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  padding: 8px 10px;
}}
.stat span {{ display: block; color: var(--muted); font-size: 12px; }}
.stat b {{ display: block; margin-top: 4px; font-size: 18px; }}
.stat.good {{ background: var(--good); }}
.stat.danger {{ background: var(--danger); }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}}
th {{ background: #ece7dc; position: sticky; top: 114px; z-index: 5; }}
a {{ color: #245b8f; text-decoration: none; font-weight: 700; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.num.good {{ color: #247247; }}
.num.bad {{ color: #a82c27; font-weight: 700; }}
.status-pill {{
  display: inline-block;
  margin: 0 4px 4px 0;
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #f2eee4;
  font-size: 12px;
  white-space: nowrap;
}}
.status-pill.ok-piece, .status-pill.ok-empty {{ background: var(--good); }}
.status-pill.unknown, .status-pill.unknown-top3 {{ background: var(--warn); }}
.status-pill.top3-only, .status-pill.top1-only {{ background: var(--top3); }}
.status-pill.wrong {{ background: var(--danger); }}
.hidden {{ display: none; }}
@media (max-width: 900px) {{
  th, td {{ font-size: 12px; padding: 6px; }}
  .hide-narrow {{ display: none; }}
}}
</style>
</head>
<body>
<header>
  <h1>Piece Accuracy Review - Light</h1>
  <p class="subtle">Current report: {html.escape(str(reports_dir.resolve()))}</p>
  <div class="toolbar">
    <input id="sampleFilter" type="search" placeholder="Filter sample/app/style">
    <select id="appFilter"><option value="">All apps</option>{app_options}</select>
    <select id="styleFilter"><option value="">All styles</option>{style_options}</select>
    <select id="errorFilter">
      <option value="">All samples</option>
      <option value="board">Board errors only</option>
      <option value="hand">Hand errors only</option>
      <option value="leak">Leak errors only</option>
      <option value="clean">Clean samples only</option>
    </select>
  </div>
</header>
<main>
  <section>
    <h2>Overall</h2>
    <div class="summary-grid">
      {render_stat('Samples', overall.get('samples'))}
      {render_stat('Skipped', overall.get('skipped_samples', 0), 'danger' if int(overall.get('skipped_samples') or 0) else 'good')}
      {render_stat('Board errors', overall.get('errors', 0), 'danger' if int(overall.get('errors') or 0) else 'good')}
      {render_stat('Hand errors', overall.get('hand_errors', 0), 'danger' if int(overall.get('hand_errors') or 0) else 'good')}
      {render_stat('Leak errors', overall.get('leak_errors', 0), 'danger' if int(overall.get('leak_errors') or 0) else 'good')}
      {render_stat('Confirmed acc.', overall.get('confirmed_identity_accuracy'))}
      {render_stat('Top1 acc.', overall.get('top1_identity_accuracy'))}
      {render_stat('Top3 acc.', overall.get('top3_contains_identity_accuracy'))}
    </div>
  </section>
  <section>
    <h2>Samples</h2>
    <table>
      <thead>
        <tr>
          <th>Detail</th><th>App</th><th>Style</th><th>Sample</th>
          <th>Board</th><th>Hand</th><th>Leak</th>
          <th class="hide-narrow">Confirmed</th><th class="hide-narrow">Top1</th><th class="hide-narrow">Top3</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</main>
<script>
const rows = Array.from(document.querySelectorAll('.sample-row'));
const sampleFilter = document.getElementById('sampleFilter');
const appFilter = document.getElementById('appFilter');
const styleFilter = document.getElementById('styleFilter');
const errorFilter = document.getElementById('errorFilter');
function asNumber(value) {{
  const parsed = Number.parseInt(value || '0', 10);
  return Number.isFinite(parsed) ? parsed : 0;
}}
function refresh() {{
  const text = sampleFilter.value.trim().toLowerCase();
  const app = appFilter.value;
  const style = styleFilter.value;
  const mode = errorFilter.value;
  for (const row of rows) {{
    const haystack = `${{row.dataset.sample}} ${{row.dataset.app}} ${{row.dataset.style}}`.toLowerCase();
    const board = asNumber(row.dataset.boardErrors);
    const hand = asNumber(row.dataset.handErrors);
    const leak = asNumber(row.dataset.leakErrors);
    let visible = true;
    if (text && !haystack.includes(text)) visible = false;
    if (app && row.dataset.app !== app) visible = false;
    if (style && row.dataset.style !== style) visible = false;
    if (mode === 'board' && board === 0) visible = false;
    if (mode === 'hand' && hand === 0) visible = false;
    if (mode === 'leak' && leak === 0) visible = false;
    if (mode === 'clean' && (board > 0 || hand > 0 || leak > 0)) visible = false;
    row.classList.toggle('hidden', !visible);
  }}
}}
[sampleFilter, appFilter, styleFilter, errorFilter].forEach((control) => control.addEventListener('input', refresh));
</script>
</body>
</html>
"""


def render_detail_page(
    reports_dir: Path,
    out_dir: Path,
    item: dict[str, Any],
    index: int,
) -> str:
    detail_summary = summarize_entries([item])
    document = render_html(reports_dir, out_dir, [item], detail_summary)
    back_link = '<p class="links"><a href="../index.html">Back to lightweight index</a></p>'
    return document.replace("<main>", f"<main>\n{back_link}", 1)


def write_split_pages(
    reports_dir: Path,
    out_dir: Path,
    items: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        detail_path = sample_dir / detail_page_name(index)
        detail_path.write_text(render_detail_page(reports_dir, out_dir, item, index), encoding="utf-8")
    index_path = out_dir / "index.html"
    index_path.write_text(render_light_html(reports_dir, items, summary), encoding="utf-8")
    return index_path


def sample_summary_row(item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metrics") or {}
    return {
        "app": item.get("app") or "",
        "piece_style": item.get("piece_style") or "",
        "sample": item.get("sample") or "",
        "image": str(item.get("image_path") or ""),
        "report": str(item.get("report_path") or ""),
        "label": str(item.get("label_path") or ""),
        "seconds": item.get("seconds") or "",
        "errors": metric_value(metrics, "errors"),
        "hand_errors": metric_value(metrics, "hand_errors"),
        "leak_errors": metric_value(metrics, "leak_errors"),
        "unknown_on_piece": metric_value(metrics, "unknown_on_piece"),
        "false_empty_on_piece": metric_value(metrics, "false_empty_on_piece"),
        "false_piece_on_empty": metric_value(metrics, "false_piece_on_empty"),
        "confirmed_identity_accuracy": metric_value(metrics, "confirmed_identity_accuracy"),
        "top1_identity_accuracy": metric_value(metrics, "top1_identity_accuracy"),
        "top3_contains_identity_accuracy": metric_value(metrics, "top3_contains_identity_accuracy"),
        "evaluation_error": item.get("evaluation_error") or "",
    }


def write_review(
    reports_dir: Path,
    labels_dir: Path,
    screenshots_dir: Path,
    out_dir: Path,
    include_hands: bool,
    strict_leak_guard: bool,
    require_excluded_source: bool,
    high_confidence_threshold: float,
    split_pages: bool = False,
) -> dict[str, Any]:
    entries = report_entries(reports_dir)
    if not entries:
        raise FileNotFoundError(f"no piece_report.json found under {reports_dir}")
    items = [
        evaluate_entry(
            entry,
            labels_dir,
            screenshots_dir,
            include_hands,
            strict_leak_guard,
            require_excluded_source,
            high_confidence_threshold,
        )
        for entry in entries
    ]
    summary = summarize_entries(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_pages:
        index_path = write_split_pages(reports_dir, out_dir, items, summary)
    else:
        index_path = out_dir / "index.html"
        index_path.write_text(render_html(reports_dir, out_dir, items, summary), encoding="utf-8")
    sample_rows = [sample_summary_row(item) for item in items]
    write_csv(
        out_dir / "piece_accuracy_samples.csv",
        sample_rows,
        [
            "app",
            "piece_style",
            "sample",
            "image",
            "report",
            "label",
            "seconds",
            "errors",
            "hand_errors",
            "leak_errors",
            "unknown_on_piece",
            "false_empty_on_piece",
            "false_piece_on_empty",
            "confirmed_identity_accuracy",
            "top1_identity_accuracy",
            "top3_contains_identity_accuracy",
            "evaluation_error",
        ],
    )
    cell_rows = [row for item in items for row in item["cell_rows"]]
    write_csv(
        out_dir / "piece_accuracy_cells.csv",
        cell_rows,
        [
            "app",
            "piece_style",
            "sample",
            "row",
            "col",
            "square",
            "status",
            "expected",
            "actual_state",
            "confirmed",
            "top1",
            "top3",
            "score",
            "candidates",
            "expected_label",
            "actual_label",
        ],
    )
    machine_summary = {
        "reports_dir": str(reports_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "screenshots_dir": str(screenshots_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        **summary,
    }
    write_json(out_dir / "piece_accuracy_summary.json", machine_summary)
    return {
        "index": index_path,
        "samples": len(items),
        "summary": machine_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a visual HTML review for piece recognition accuracy across all sample reports.",
    )
    parser.add_argument("reports_dir", nargs="?", type=Path, default=default_reports_dir())
    parser.add_argument("--labels-dir", type=Path, default=default_labels_dir())
    parser.add_argument("--screenshots-dir", type=Path, default=default_screenshots_dir())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--without-hands", action="store_true")
    parser.add_argument("--no-strict-leak-guard", action="store_true")
    parser.add_argument("--allow-missing-excluded-source", action="store_true")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.75)
    parser.add_argument(
        "--split-pages",
        action="store_true",
        help="Write a lightweight index.html plus one detail HTML per sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports_dir = args.reports_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else reports_dir / "piece_accuracy_review"
    result = write_review(
        reports_dir=reports_dir,
        labels_dir=args.labels_dir.resolve(),
        screenshots_dir=args.screenshots_dir.resolve(),
        out_dir=out_dir,
        include_hands=not args.without_hands,
        strict_leak_guard=not args.no_strict_leak_guard,
        require_excluded_source=not args.allow_missing_excluded_source,
        high_confidence_threshold=args.high_confidence_threshold,
        split_pages=args.split_pages,
    )
    overall = result["summary"]["overall"]
    print(f"OK: wrote {result['index']} for {result['samples']} samples")
    print(
        "summary: "
        f"board_errors={overall.get('errors', 0)} "
        f"hand_errors={overall.get('hand_errors', 0)} "
        f"leak_errors={overall.get('leak_errors', 0)} "
        f"skipped={overall.get('skipped_samples', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
