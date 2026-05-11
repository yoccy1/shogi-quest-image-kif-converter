from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root()
for import_path in (REPO_ROOT / "src", REPO_ROOT / "tools", Path(__file__).resolve().parent):
    value = str(import_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from evaluate_piece_recognition import evaluate_one, load_report  # noqa: E402
from position_label_utils import HAND_PIECES, find_label_path, load_position_label, square_name  # noqa: E402
from shogi_gazo_desktop.export import ExportError, export_kif, export_sfen  # noqa: E402
from shogi_gazo_desktop.models import RecognitionResult, empty_hands  # noqa: E402


PIECE_TEXT = {
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
COLOR_TEXT = {"black": "先手", "white": "後手"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def report_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    benchmark = path / "benchmark_report.json"
    if benchmark.exists():
        data = read_json(benchmark)
        paths: list[Path] = []
        for item in data.get("results", []):
            raw = item.get("report")
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            if candidate.exists():
                paths.append(candidate)
        if paths:
            return paths
    direct = path / "piece_report.json"
    if direct.exists():
        return [direct]
    paths = sorted(path.glob("*/piece_report.json"))
    if not paths:
        raise FileNotFoundError(f"no piece_report.json found under {path}")
    return paths


def default_labels_dir() -> Path:
    preferred = REPO_ROOT / "data" / "samples" / "labels" / "boards_by_app_piece_style"
    if preferred.exists():
        return preferred
    return REPO_ROOT / "tools" / "samples" / "labels" / "boards"


def sample_name(report_path: Path, report: dict[str, Any]) -> str:
    image = report.get("image")
    if image:
        return Path(str(image).replace("\\", "/")).stem
    if report_path.name == "piece_report.json":
        return report_path.parent.name
    return report_path.stem


def path_uri(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.resolve().as_uri()
    except Exception:
        return ""


def resolve_image_path(value: str | None, report_path: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    candidates = [
        (REPO_ROOT / raw).resolve(),
        (Path.cwd() / raw).resolve(),
        (report_path.parent / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_hands(value: Any) -> dict[str, dict[str, int]]:
    hands = empty_hands()
    if not isinstance(value, dict):
        return hands
    for color in ("black", "white"):
        raw_counts = value.get(color) or {}
        if not isinstance(raw_counts, dict):
            continue
        for piece in HAND_PIECES:
            try:
                hands[color][piece] = int(raw_counts.get(piece, 0))
            except Exception:
                hands[color][piece] = 0
    return hands


def load_recognition(report_path: Path) -> dict[str, Any] | None:
    path = report_path.with_name("recognition.json")
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def cells_to_board(cells: list[dict[str, Any]]) -> list[list[str]]:
    board = [["empty" for _ in range(9)] for _ in range(9)]
    for cell in cells:
        try:
            row = int(cell.get("row")) - 1
            col = int(cell.get("col")) - 1
        except Exception:
            continue
        if not (0 <= row < 9 and 0 <= col < 9):
            continue
        state = cell.get("state")
        if state == "piece" and cell.get("color") and cell.get("piece"):
            board[row][col] = f"{cell['color']}:{cell['piece']}"
        elif state == "unknown":
            board[row][col] = "unknown"
        else:
            board[row][col] = "empty"
    return board


def label_rows_to_cells(rows: list[list[Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        for col_index, value in enumerate(row, start=1):
            square = square_name(row_index, col_index)
            if value == "empty":
                cells.append({"row": row_index, "col": col_index, "square": square, "state": "empty"})
            elif value == "unknown":
                cells.append({"row": row_index, "col": col_index, "square": square, "state": "unknown"})
            elif isinstance(value, str) and ":" in value:
                color, piece = value.split(":", 1)
                cells.append(
                    {
                        "row": row_index,
                        "col": col_index,
                        "square": square,
                        "state": "piece",
                        "color": color,
                        "piece": piece,
                    }
                )
            else:
                cells.append({"row": row_index, "col": col_index, "square": square, "state": "empty"})
    return cells


def board_to_label_cells(board: list[list[str]]) -> list[dict[str, Any]]:
    return label_rows_to_cells(board)


def confidence_value(cell: dict[str, Any]) -> float | None:
    value = cell.get("confidence")
    if value is None:
        value = cell.get("score")
    try:
        return float(value)
    except Exception:
        return None


def candidate_identity(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "none"
    color = candidate.get("color")
    piece = candidate.get("piece")
    if color and piece:
        return f"{color}:{piece}"
    value = candidate.get("value") or candidate.get("identity")
    return str(value or "none")


def has_solver_change(cell: dict[str, Any]) -> bool:
    reason = str(cell.get("postprocess_reason") or "")
    if "solver" in reason or "beam" in reason:
        return True
    for item in cell.get("postprocess_history") or []:
        if not isinstance(item, dict):
            continue
        item_reason = str(item.get("reason") or "")
        if "solver" in item_reason or "beam" in item_reason:
            return True
    return False


def compact_cells(cells: list[dict[str, Any]], low_confidence: float) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for cell in cells:
        candidates = []
        for candidate in (cell.get("candidates") or [])[:12]:
            candidates.append(
                {
                    "identity": candidate_identity(candidate),
                    "color": candidate.get("color"),
                    "piece": candidate.get("piece"),
                    "score": candidate.get("score", candidate.get("confidence")),
                    "source": candidate.get("source"),
                }
            )
        conf = confidence_value(cell)
        compact.append(
            {
                "row": cell.get("row"),
                "col": cell.get("col"),
                "square": cell.get("square") or square_name(int(cell.get("row", 1)), int(cell.get("col", 1))),
                "state": cell.get("state"),
                "color": cell.get("color"),
                "piece": cell.get("piece"),
                "identity": f"{cell.get('color')}:{cell.get('piece')}" if cell.get("state") == "piece" else cell.get("state"),
                "confidence": conf,
                "lowConfidence": bool(conf is not None and cell.get("state") == "piece" and conf < low_confidence),
                "solverChanged": has_solver_change(cell),
                "postprocessReason": cell.get("postprocess_reason"),
                "postprocessHistory": cell.get("postprocess_history") or [],
                "bboxRatio": cell.get("bbox_ratio"),
                "candidates": candidates,
            }
        )
    return compact


def load_label(report_path: Path, report: dict[str, Any], labels_dir: Path | None) -> tuple[Path | None, dict[str, Any] | None]:
    if labels_dir is None:
        return None, None
    name = sample_name(report_path, report)
    path = find_label_path(labels_dir, name)
    if not path.exists():
        return path, None
    try:
        return path, load_position_label(path, require_hands=False)
    except Exception:
        return path, None


def load_evaluation_map(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None or not path.exists():
        return {}, None
    data = read_json(path)
    by_sample: dict[str, dict[str, Any]] = {}
    for result in data.get("results") or []:
        sample = result.get("sample")
        if sample:
            by_sample[str(sample)] = result
    return by_sample, data.get("metrics") or data.get("summary")


def evaluate_report(
    report_path: Path,
    label_path: Path | None,
    include_hands: bool,
    evaluation_map: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    if name in evaluation_map:
        return evaluation_map[name]
    if label_path is None or not label_path.exists():
        return None
    try:
        return evaluate_one(report_path, label_path, 0.75, include_hands=include_hands)
    except Exception:
        try:
            return evaluate_one(report_path, label_path, 0.75, include_hands=False)
        except Exception:
            return None


def export_bundle(
    image: str,
    board: list[list[str]],
    hands: dict[str, dict[str, int]],
    raw_report: dict[str, Any],
) -> dict[str, dict[str, str]]:
    result = RecognitionResult(
        image=image,
        board=board,
        hands=hands,
        confidence=[],
        raw_report=raw_report,
    )
    bundle: dict[str, dict[str, str]] = {}
    for side in ("black", "white"):
        entry = {"kif": "", "sfen": "", "error": ""}
        try:
            entry["kif"] = export_kif(result, side_to_move=side)
            entry["sfen"] = export_sfen(result, side_to_move=side)
        except (ExportError, ValueError) as exc:
            entry["error"] = str(exc)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        bundle[side] = entry
    return bundle


def sample_status(metrics: dict[str, Any], cells: list[dict[str, Any]], evaluation: dict[str, Any] | None) -> dict[str, Any]:
    unknown_cells = sum(1 for cell in cells if cell.get("state") == "unknown")
    solver_cells = sum(1 for cell in cells if cell.get("solverChanged"))
    high_conf = len((evaluation or {}).get("high_confidence_errors") or [])
    top3_out = int(metrics.get("top3_outside_errors") or len((evaluation or {}).get("top3_misses") or []))
    return {
        "boardErrors": int(metrics.get("errors") or 0),
        "handErrors": int(metrics.get("hand_errors") or 0),
        "unknown": int(metrics.get("unknown_on_piece") or unknown_cells),
        "unknownCells": unknown_cells,
        "top3Outside": top3_out,
        "highConfidenceErrors": high_conf,
        "solverChanged": solver_cells,
        "needsReview": bool(
            int(metrics.get("errors") or 0)
            or int(metrics.get("hand_errors") or 0)
            or int(metrics.get("unknown_on_piece") or 0)
            or high_conf
        ),
    }


def build_sample(
    report_path: Path,
    labels_dir: Path | None,
    evaluation_map: dict[str, dict[str, Any]],
    include_hands: bool,
    low_confidence: float,
) -> dict[str, Any]:
    report = load_report(report_path)
    name = sample_name(report_path, report)
    recognition = load_recognition(report_path)
    label_path, label = load_label(report_path, report, labels_dir)
    evaluation = evaluate_report(report_path, label_path, include_hands, evaluation_map, name)
    metrics = (evaluation or {}).get("metrics") or {}
    cells = compact_cells(report.get("cells") or [], low_confidence)
    image_path = resolve_image_path(str(report.get("image") or ""), report_path)

    recognized_board = (
        recognition.get("board")
        if isinstance(recognition, dict) and isinstance(recognition.get("board"), list)
        else cells_to_board(report.get("cells") or [])
    )
    recognized_hands = normalize_hands(
        report.get("hands")
        if isinstance(report.get("hands"), dict)
        else (recognition or {}).get("hands") if isinstance(recognition, dict) else None
    )
    label_cells = list((label or {}).get("cells") or [])
    label_board = cells_to_board(label_cells) if label_cells else None
    label_hands = normalize_hands((label or {}).get("hands")) if label and (label.get("hands") is not None) else None
    timing = report.get("timing") or {}
    export_image = str(image_path or report.get("image") or "")

    return {
        "name": name,
        "reportPath": str(report_path),
        "recognitionPath": str(report_path.with_name("recognition.json")),
        "labelPath": str(label_path) if label_path else "",
        "hasLabel": bool(label),
        "imagePath": str(image_path) if image_path else str(report.get("image") or ""),
        "imageUri": path_uri(image_path),
        "grid": report.get("grid") or {},
        "summary": report.get("summary") or {},
        "timing": timing,
        "processingTime": timing.get("processing_time_seconds"),
        "metrics": metrics,
        "status": sample_status(metrics, cells, evaluation),
        "cells": cells,
        "recognizedBoard": recognized_board,
        "recognizedHands": recognized_hands,
        "labelCells": compact_cells(label_cells, low_confidence) if label_cells else [],
        "labelBoard": label_board,
        "labelHands": label_hands,
        "evaluation": {
            "errors": (evaluation or {}).get("errors") or [],
            "handErrors": ((evaluation or {}).get("hands") or {}).get("error_details") or [],
            "top3Misses": (evaluation or {}).get("top3_misses") or [],
            "highConfidenceErrors": (evaluation or {}).get("high_confidence_errors") or [],
            "errorCategories": (evaluation or {}).get("error_categories") or {},
        },
        "solver": {
            "constraintPostprocess": report.get("constraint_postprocess") or {},
            "globalSolver": report.get("global_solver") or {},
            "softInventoryBeam": report.get("soft_inventory_beam") or {},
            "needsReview": bool((recognition or {}).get("needs_review")) if isinstance(recognition, dict) else False,
            "reviewReasons": (recognition or {}).get("review_reasons") if isinstance(recognition, dict) else [],
        },
        "exports": {
            "recognized": export_bundle(export_image, recognized_board, recognized_hands, report),
            "label": export_bundle(export_image, label_board, label_hands or empty_hands(), {}) if label_board else {},
        },
    }


def aggregate(samples: list[dict[str, Any]], evaluation_metrics: dict[str, Any] | None) -> dict[str, Any]:
    totals = Counter()
    processing: list[float] = []
    for sample in samples:
        status = sample["status"]
        totals["samples"] += 1
        totals["needsReview"] += int(status["needsReview"])
        totals["boardErrors"] += int(status["boardErrors"])
        totals["handErrors"] += int(status["handErrors"])
        totals["unknown"] += int(status["unknown"])
        totals["top3Outside"] += int(status["top3Outside"])
        totals["highConfidenceErrors"] += int(status["highConfidenceErrors"])
        totals["solverChangedSamples"] += int(status["solverChanged"] > 0)
        value = sample.get("processingTime")
        if isinstance(value, (int, float)):
            processing.append(float(value))
    summary = dict(totals)
    summary["averageProcessingTime"] = round(sum(processing) / len(processing), 3) if processing else None
    if evaluation_metrics:
        summary["evaluationMetrics"] = evaluation_metrics
    return summary


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__ANALYSIS_DATA__", payload)


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>将棋画像解析ビュー</title>
<style>
:root {
  color-scheme: light;
  --bg: #f4f0e7;
  --panel: #fffaf1;
  --panel-2: #f8efe0;
  --line: #d5c3a6;
  --ink: #241c13;
  --muted: #695f51;
  --accent: #126c5a;
  --bad: #c63b34;
  --warn: #b46b00;
  --unknown: #8b6f00;
  --blue: #255f9b;
  --board: #e8bd6b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: system-ui, "Yu Gothic", "Meiryo", sans-serif;
  background: var(--bg);
  color: var(--ink);
}
button, input, select, textarea {
  font: inherit;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(244, 240, 231, 0.97);
  border-bottom: 1px solid var(--line);
  padding: 12px 18px;
}
.title-row {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 10px;
}
h1 {
  font-size: 20px;
  line-height: 1.2;
  margin: 0;
}
.run-path {
  color: var(--muted);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(8, minmax(86px, 1fr));
  gap: 8px;
}
.metric {
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 6px;
  padding: 7px 9px;
  min-width: 0;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  line-height: 1.2;
}
.metric b {
  font-size: 18px;
  line-height: 1.2;
}
.layout {
  display: grid;
  grid-template-columns: 330px minmax(480px, 1fr) 390px;
  gap: 12px;
  padding: 12px;
}
.panel {
  min-height: calc(100vh - 116px);
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 8px;
  overflow: hidden;
}
.sidebar {
  display: flex;
  flex-direction: column;
}
.filters {
  padding: 10px;
  border-bottom: 1px solid var(--line);
  background: var(--panel-2);
}
.filter-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
  margin-bottom: 8px;
}
.filter-btn, .tab-btn, .small-btn {
  border: 1px solid var(--line);
  background: #fffdf8;
  color: var(--ink);
  border-radius: 6px;
  min-height: 32px;
  cursor: pointer;
}
.filter-btn.active, .tab-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}
.search {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  padding: 8px 9px;
}
.sample-list {
  overflow: auto;
  padding: 8px;
}
.sample-row {
  width: 100%;
  text-align: left;
  border: 1px solid var(--line);
  background: white;
  border-radius: 7px;
  padding: 8px;
  margin-bottom: 7px;
  cursor: pointer;
}
.sample-row.active {
  outline: 3px solid rgba(18, 108, 90, 0.25);
  border-color: var(--accent);
}
.sample-name {
  font-weight: 700;
  font-size: 13px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.badges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 20px;
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 11px;
  background: #ebe2d2;
  color: var(--ink);
}
.badge.bad { background: #f7d7d4; color: #7d1b18; }
.badge.warn { background: #f9e2bd; color: #734700; }
.badge.ok { background: #dceee6; color: #174d3e; }
.viewer {
  display: flex;
  flex-direction: column;
}
.viewer-head {
  border-bottom: 1px solid var(--line);
  padding: 10px 12px;
  background: var(--panel-2);
}
.viewer-title {
  font-size: 17px;
  font-weight: 800;
  margin-bottom: 7px;
  overflow-wrap: anywhere;
}
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.viewer-body {
  flex: 1;
  overflow: auto;
  padding: 12px;
}
.image-wrap {
  position: relative;
  max-width: min(100%, 760px);
  margin: 0 auto;
  background: #111;
  border: 1px solid #111;
}
.image-wrap img {
  display: block;
  width: 100%;
  height: auto;
}
.grid-overlay {
  position: absolute;
  border: 2px solid rgba(255, 255, 255, 0.88);
  pointer-events: none;
}
.grid-overlay .line-v, .grid-overlay .line-h {
  position: absolute;
  background: rgba(255, 255, 255, 0.55);
}
.grid-overlay .line-v { top: 0; bottom: 0; width: 1px; }
.grid-overlay .line-h { left: 0; right: 0; height: 1px; }
.board-wrap {
  display: flex;
  justify-content: center;
}
.board {
  display: grid;
  grid-template-columns: repeat(9, minmax(46px, 1fr));
  grid-template-rows: repeat(9, 1fr);
  width: min(100%, 590px);
  aspect-ratio: 1 / 1;
  border: 2px solid #744d27;
  background: var(--board);
}
.cell {
  position: relative;
  border: 1px solid #87643a;
  background: var(--board);
  color: var(--ink);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 0;
  cursor: pointer;
}
.cell .sq {
  position: absolute;
  top: 2px;
  left: 3px;
  color: rgba(51, 35, 19, 0.72);
  font-size: 10px;
}
.cell .piece {
  font-weight: 800;
  font-size: 17px;
  line-height: 1.05;
  text-align: center;
}
.cell .sub {
  font-size: 10px;
  color: var(--muted);
}
.cell.empty .piece { color: transparent; }
.cell.error { background: #f7cbc6; outline: 3px solid var(--bad); z-index: 2; }
.cell.hand-error { background: #f7d7d4; }
.cell.unknown { background: #fff2b8; }
.cell.low { background: #ffe0b5; }
.cell.solver { box-shadow: inset 0 0 0 3px rgba(37, 95, 155, 0.55); }
.cell.selected { outline: 3px solid #111; z-index: 3; }
.diff-cell {
  gap: 2px;
}
.diff-cell .expected {
  color: var(--accent);
  font-size: 12px;
}
.diff-cell .actual {
  color: var(--bad);
  font-size: 12px;
}
.empty-state {
  max-width: 620px;
  margin: 40px auto;
  color: var(--muted);
  line-height: 1.7;
}
.inspector {
  overflow: auto;
}
.section {
  padding: 12px;
  border-bottom: 1px solid var(--line);
}
.section h2 {
  margin: 0 0 9px;
  font-size: 15px;
}
.kv {
  display: grid;
  grid-template-columns: 116px minmax(0, 1fr);
  gap: 5px 8px;
  font-size: 13px;
}
.kv .k {
  color: var(--muted);
}
.candidate-list {
  display: grid;
  gap: 5px;
}
.candidate {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 8px;
  border: 1px solid var(--line);
  background: white;
  border-radius: 5px;
  padding: 5px 7px;
  font-size: 12px;
}
.hands-grid {
  display: grid;
  grid-template-columns: 54px repeat(7, 1fr);
  gap: 1px;
  border: 1px solid var(--line);
  background: var(--line);
  font-size: 12px;
}
.hands-grid div {
  background: white;
  min-height: 26px;
  padding: 4px;
  text-align: center;
}
.hands-grid .head {
  background: var(--panel-2);
  font-weight: 700;
}
.hands-grid .diff {
  background: #f7d7d4;
  color: #7d1b18;
  font-weight: 700;
}
.export-controls {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 8px;
}
.export-controls select {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  min-height: 32px;
  background: white;
}
.export-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin: 7px 0;
}
.copy-status {
  min-height: 20px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}
.small-btn.copied {
  border-color: var(--accent);
  background: #dceee6;
  color: #174d3e;
}
textarea.kif {
  width: 100%;
  min-height: 210px;
  resize: vertical;
  border: 1px solid var(--line);
  background: #fffefb;
  border-radius: 6px;
  padding: 8px;
  font-family: Consolas, "Yu Gothic", monospace;
  font-size: 12px;
  line-height: 1.45;
}
.sfen {
  border: 1px solid var(--line);
  background: white;
  border-radius: 6px;
  padding: 7px;
  font-family: Consolas, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.error-text {
  color: var(--bad);
  font-weight: 700;
  line-height: 1.5;
}
.muted {
  color: var(--muted);
}
@media (max-width: 1180px) {
  .layout { grid-template-columns: 300px 1fr; }
  .inspector { grid-column: 1 / -1; min-height: auto; }
  .summary-grid { grid-template-columns: repeat(4, 1fr); }
}
@media (max-width: 760px) {
  .layout { grid-template-columns: 1fr; }
  .summary-grid { grid-template-columns: repeat(2, 1fr); }
  .panel { min-height: auto; }
  .board { grid-template-rows: repeat(9, 42px); }
  .cell .piece { font-size: 14px; }
}
</style>
</head>
<body>
<script id="analysis-data" type="application/json">__ANALYSIS_DATA__</script>
<header class="topbar">
  <div class="title-row">
    <h1>将棋画像解析ビュー</h1>
    <div class="run-path" id="runPath"></div>
  </div>
  <div class="summary-grid" id="summaryGrid"></div>
</header>
<main class="layout">
  <aside class="panel sidebar">
    <div class="filters">
      <div class="filter-grid" id="filterGrid"></div>
      <input class="search" id="searchInput" placeholder="サンプル検索">
    </div>
    <div class="sample-list" id="sampleList"></div>
  </aside>
  <section class="panel viewer">
    <div class="viewer-head">
      <div class="viewer-title" id="viewerTitle"></div>
      <div class="tabs" id="tabs"></div>
    </div>
    <div class="viewer-body" id="viewerBody"></div>
  </section>
  <aside class="panel inspector" id="inspector"></aside>
</main>
<script>
const DATA = JSON.parse(document.getElementById("analysis-data").textContent);
const PIECE_TEXT = {OU:"玉",HI:"飛",KA:"角",KI:"金",GI:"銀",KE:"桂",KY:"香",FU:"歩",RY:"龍",UM:"馬",NG:"成銀",NK:"成桂",NY:"成香",TO:"と"};
const COLOR_MARK = {black:"▲", white:"△"};
const COLOR_TEXT = {black:"先手", white:"後手"};
const HAND_PIECES = ["HI","KA","KI","GI","KE","KY","FU"];
const FILTERS = [
  ["all", "すべて"],
  ["review", "要確認"],
  ["board", "盤面差分"],
  ["hand", "持ち駒差分"],
  ["unknown", "unknown"],
  ["top3out", "top3外"],
  ["high", "高信頼誤認"],
  ["solver", "solver変更"]
];
const TABS = [
  ["image", "元画像"],
  ["recognized", "認識盤面"],
  ["label", "正解盤面"],
  ["diff", "差分盤面"]
];
let filter = "all";
let query = "";
let selectedIndex = 0;
let selectedSquare = null;
let tab = "diff";
let exportSource = "recognized";
let exportSide = "black";

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
}
function pieceText(identity) {
  if (!identity || identity === "empty") return "";
  if (identity === "unknown") return "?";
  const [color, piece] = String(identity).split(":");
  return `${COLOR_MARK[color] || ""}${PIECE_TEXT[piece] || piece || ""}`;
}
function cellIdentity(cell) {
  if (!cell) return "empty";
  if (cell.state === "piece" && cell.color && cell.piece) return `${cell.color}:${cell.piece}`;
  if (cell.state === "unknown") return "unknown";
  return "empty";
}
function bySquare(cells) {
  const map = {};
  for (const cell of cells || []) map[cell.square] = cell;
  return map;
}
function boardCell(board, row, col) {
  if (!board || !board[row - 1]) return "empty";
  return board[row - 1][col - 1] || "empty";
}
function squareName(row, col) {
  return "987654321"[col - 1] + "一二三四五六七八九"[row - 1];
}
function statusBadges(sample) {
  const s = sample.status || {};
  const badges = [];
  if (s.boardErrors) badges.push(`<span class="badge bad">盤面 ${s.boardErrors}</span>`);
  if (s.handErrors) badges.push(`<span class="badge bad">持ち駒 ${s.handErrors}</span>`);
  if (s.unknown) badges.push(`<span class="badge warn">unknown ${s.unknown}</span>`);
  if (s.top3Outside) badges.push(`<span class="badge warn">top3外 ${s.top3Outside}</span>`);
  if (s.highConfidenceErrors) badges.push(`<span class="badge bad">高信頼 ${s.highConfidenceErrors}</span>`);
  if (s.solverChanged) badges.push(`<span class="badge">solver ${s.solverChanged}</span>`);
  if (!badges.length) badges.push(`<span class="badge ok">OK</span>`);
  const time = sample.processingTime;
  if (typeof time === "number") badges.push(`<span class="badge">${time.toFixed(2)}秒</span>`);
  return badges.join("");
}
function filterMatches(sample) {
  const s = sample.status || {};
  if (filter === "review" && !s.needsReview) return false;
  if (filter === "board" && !s.boardErrors) return false;
  if (filter === "hand" && !s.handErrors) return false;
  if (filter === "unknown" && !s.unknown) return false;
  if (filter === "top3out" && !s.top3Outside) return false;
  if (filter === "high" && !s.highConfidenceErrors) return false;
  if (filter === "solver" && !s.solverChanged) return false;
  if (query && !sample.name.toLowerCase().includes(query.toLowerCase())) return false;
  return true;
}
function renderSummary() {
  document.getElementById("runPath").textContent = DATA.runDir || "";
  const s = DATA.summary || {};
  const metrics = [
    ["サンプル", s.samples ?? DATA.samples.length],
    ["要確認", s.needsReview ?? 0],
    ["盤面差分", s.boardErrors ?? 0],
    ["持ち駒差分", s.handErrors ?? 0],
    ["unknown", s.unknown ?? 0],
    ["top3外", s.top3Outside ?? 0],
    ["高信頼誤認", s.highConfidenceErrors ?? 0],
    ["平均秒", s.averageProcessingTime ?? "-"]
  ];
  document.getElementById("summaryGrid").innerHTML = metrics.map(([k, v]) => `<div class="metric"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}
function renderFilters() {
  document.getElementById("filterGrid").innerHTML = FILTERS.map(([key, label]) => (
    `<button class="filter-btn ${filter === key ? "active" : ""}" data-filter="${key}">${esc(label)}</button>`
  )).join("");
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      filter = btn.dataset.filter;
      renderAll();
    });
  });
}
function renderList() {
  const list = document.getElementById("sampleList");
  const visible = DATA.samples.map((sample, index) => [sample, index]).filter(([sample]) => filterMatches(sample));
  if (!visible.some(([, index]) => index === selectedIndex) && visible.length) {
    selectedIndex = visible[0][1];
    selectedSquare = null;
  }
  list.innerHTML = visible.map(([sample, index]) => (
    `<button class="sample-row ${index === selectedIndex ? "active" : ""}" data-index="${index}">
      <div class="sample-name">${esc(sample.name)}</div>
      <div class="badges">${statusBadges(sample)}</div>
    </button>`
  )).join("") || `<div class="empty-state">該当するサンプルはありません。</div>`;
  list.querySelectorAll(".sample-row").forEach(row => {
    row.addEventListener("click", () => {
      selectedIndex = Number(row.dataset.index);
      selectedSquare = null;
      renderAll();
    });
  });
}
function renderTabs() {
  document.getElementById("tabs").innerHTML = TABS.map(([key, label]) => (
    `<button class="tab-btn ${tab === key ? "active" : ""}" data-tab="${key}">${esc(label)}</button>`
  )).join("");
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tab = btn.dataset.tab;
      renderSample();
    });
  });
}
function errorBySquare(sample) {
  const map = {};
  for (const item of sample.evaluation?.errors || []) map[item.square] = item;
  return map;
}
function renderImage(sample) {
  if (!sample.imageUri) return `<div class="empty-state">元画像パスを表示できません。</div>`;
  const grid = sample.grid || {};
  const size = grid.image_size || [];
  const rect = grid.grid_rect || grid.board_rect;
  let overlay = "";
  if (rect && size[0] && size[1]) {
    const left = rect.left / size[0] * 100;
    const top = rect.top / size[1] * 100;
    const width = rect.width / size[0] * 100;
    const height = rect.height / size[1] * 100;
    const lines = [];
    for (let i = 1; i < 9; i++) {
      lines.push(`<span class="line-v" style="left:${(i / 9 * 100).toFixed(4)}%"></span>`);
      lines.push(`<span class="line-h" style="top:${(i / 9 * 100).toFixed(4)}%"></span>`);
    }
    overlay = `<div class="grid-overlay" style="left:${left.toFixed(4)}%;top:${top.toFixed(4)}%;width:${width.toFixed(4)}%;height:${height.toFixed(4)}%">${lines.join("")}</div>`;
  }
  return `<div class="image-wrap"><img src="${esc(sample.imageUri)}" alt="">${overlay}</div>`;
}
function renderBoard(sample, mode) {
  const errors = errorBySquare(sample);
  const cells = bySquare(sample.cells || []);
  const labelCells = bySquare(sample.labelCells || []);
  const html = [];
  for (let row = 1; row <= 9; row++) {
    for (let col = 1; col <= 9; col++) {
      const square = squareName(row, col);
      const recCell = cells[square] || {state:"empty", square, row, col};
      const labelCell = labelCells[square] || null;
      const predicted = mode === "label" ? (labelCell ? cellIdentity(labelCell) : boardCell(sample.labelBoard, row, col)) : cellIdentity(recCell);
      const expected = labelCell ? cellIdentity(labelCell) : boardCell(sample.labelBoard, row, col);
      const hasError = Boolean(errors[square]);
      const classes = ["cell"];
      if (selectedSquare === square) classes.push("selected");
      if (mode !== "label" && hasError) classes.push("error");
      if (mode !== "label" && recCell.state === "unknown") classes.push("unknown");
      if (mode !== "label" && recCell.lowConfidence) classes.push("low");
      if (mode !== "label" && recCell.solverChanged) classes.push("solver");
      if ((mode === "label" ? expected : predicted) === "empty") classes.push("empty");
      if (mode === "diff") classes.push("diff-cell");
      let body;
      if (mode === "diff" && hasError) {
        body = `<span class="expected">${esc(pieceText(expected))}</span><span class="actual">${esc(pieceText(predicted))}</span>`;
      } else {
        const value = mode === "label" ? expected : predicted;
        body = `<span class="piece">${esc(pieceText(value))}</span>${recCell.confidence != null && mode !== "label" ? `<span class="sub">${Number(recCell.confidence).toFixed(2)}</span>` : ""}`;
      }
      html.push(`<button class="${classes.join(" ")}" data-square="${square}" title="${esc(square)}"><span class="sq">${square}</span>${body}</button>`);
    }
  }
  return `<div class="board-wrap"><div class="board">${html.join("")}</div></div>`;
}
function renderSample() {
  const sample = DATA.samples[selectedIndex];
  if (!sample) {
    document.getElementById("viewerBody").innerHTML = `<div class="empty-state">サンプルがありません。</div>`;
    return;
  }
  document.getElementById("viewerTitle").textContent = sample.name;
  renderTabs();
  let body = "";
  if (tab === "image") body = renderImage(sample);
  else body = renderBoard(sample, tab);
  document.getElementById("viewerBody").innerHTML = body;
  document.querySelectorAll(".cell").forEach(btn => {
    btn.addEventListener("click", () => {
      selectedSquare = btn.dataset.square;
      renderSample();
      renderInspector();
    });
  });
  renderInspector();
}
function formatHands(hands, color, piece) {
  return Number((hands?.[color] || {})[piece] || 0);
}
function renderHands(sample) {
  const actual = sample.recognizedHands || {};
  const expected = sample.labelHands || {};
  const rows = [];
  rows.push(`<div class="head"></div>${HAND_PIECES.map(p => `<div class="head">${PIECE_TEXT[p]}</div>`).join("")}`);
  for (const color of ["black", "white"]) {
    rows.push(`<div class="head">${COLOR_TEXT[color]}</div>`);
    for (const piece of HAND_PIECES) {
      const a = formatHands(actual, color, piece);
      const e = sample.labelHands ? formatHands(expected, color, piece) : null;
      const diff = e !== null && a !== e;
      rows.push(`<div class="${diff ? "diff" : ""}">${sample.labelHands ? `${a}/${e}` : a}</div>`);
    }
  }
  return `<div class="hands-grid">${rows.join("")}</div>`;
}
function renderCandidates(cell) {
  const list = cell?.candidates || [];
  if (!list.length) return `<div class="muted">候補はありません。</div>`;
  return `<div class="candidate-list">${list.map((candidate, index) => `
    <div class="candidate">
      <b>${index + 1}. ${esc(pieceText(candidate.identity))}</b>
      <span>${candidate.score == null ? "" : Number(candidate.score).toFixed(4)}</span>
      <span>${esc(candidate.source || "")}</span>
    </div>`).join("")}</div>`;
}
function exportEntry(sample) {
  return sample.exports?.[exportSource]?.[exportSide] || {};
}
function renderExport(sample) {
  const hasLabelExport = Boolean(sample.exports?.label?.black || sample.exports?.label?.white);
  if (exportSource === "label" && !hasLabelExport) exportSource = "recognized";
  const entry = exportEntry(sample);
  const disabledLabel = hasLabelExport ? "" : "disabled";
  const body = entry.error
    ? `<div class="error-text">${esc(entry.error)}</div>`
    : `<textarea class="kif" id="kifText" spellcheck="false">${esc(entry.kif || "")}</textarea>
       <div class="sfen" id="sfenText">${esc(entry.sfen || "")}</div>`;
  return `
    <div class="export-controls">
      <select id="exportSource">
        <option value="recognized" ${exportSource === "recognized" ? "selected" : ""}>認識結果</option>
        <option value="label" ${exportSource === "label" ? "selected" : ""} ${disabledLabel}>正解ラベル</option>
      </select>
      <select id="exportSide">
        <option value="black" ${exportSide === "black" ? "selected" : ""}>先手番</option>
        <option value="white" ${exportSide === "white" ? "selected" : ""}>後手番</option>
      </select>
    </div>
    <div class="export-actions">
      <button class="small-btn" id="copyKif">KIFコピー</button>
      <button class="small-btn" id="downloadKif">KIF保存</button>
      <button class="small-btn" id="copySfen">SFENコピー</button>
      <span class="copy-status" id="copyStatus" aria-live="polite"></span>
    </div>
    ${body}`;
}
function bindExport(sample) {
  const source = document.getElementById("exportSource");
  const side = document.getElementById("exportSide");
  if (source) source.addEventListener("change", () => { exportSource = source.value; renderInspector(); });
  if (side) side.addEventListener("change", () => { exportSide = side.value; renderInspector(); });
  const showCopyFeedback = (button, message, failed = false) => {
    const status = document.getElementById("copyStatus");
    const originalText = button ? button.textContent : "";
    if (button) {
      button.classList.toggle("copied", !failed);
      button.textContent = failed ? "失敗" : "コピー済み";
    }
    if (status) {
      status.textContent = message;
      status.style.color = failed ? "var(--bad)" : "var(--accent)";
    }
    window.clearTimeout(showCopyFeedback.timer);
    showCopyFeedback.timer = window.setTimeout(() => {
      if (button) {
        button.classList.remove("copied");
        button.textContent = originalText;
      }
      if (status) status.textContent = "";
    }, 1600);
  };
  const copy = async (text, button, message) => {
    try { await navigator.clipboard.writeText(text || ""); }
    catch (_) {
      const area = document.createElement("textarea");
      area.value = text || "";
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.focus();
      area.select();
      const copied = document.execCommand("copy");
      area.remove();
      if (!copied) { showCopyFeedback(button, "コピーに失敗しました", true); return; }
    }
    showCopyFeedback(button, message);
  };
  const entry = exportEntry(sample);
  document.getElementById("copyKif")?.addEventListener("click", event => copy(entry.kif || "", event.currentTarget, "KIFをコピーしました"));
  document.getElementById("copySfen")?.addEventListener("click", event => copy(entry.sfen || "", event.currentTarget, "SFENをコピーしました"));
  document.getElementById("downloadKif")?.addEventListener("click", () => {
    const blob = new Blob([entry.kif || ""], {type:"text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${sample.name}_${exportSource}_${exportSide}.kif`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
}
function renderInspector() {
  const sample = DATA.samples[selectedIndex];
  const inspector = document.getElementById("inspector");
  if (!sample) {
    inspector.innerHTML = "";
    return;
  }
  const cells = bySquare(sample.cells || []);
  const labels = bySquare(sample.labelCells || []);
  const square = selectedSquare || (sample.evaluation?.errors?.[0]?.square) || "5五";
  selectedSquare = square;
  const cell = cells[square];
  const labelCell = labels[square];
  const error = errorBySquare(sample)[square];
  const cp = sample.solver?.constraintPostprocess || {};
  inspector.innerHTML = `
    <div class="section">
      <h2>セル詳細</h2>
      <div class="kv">
        <div class="k">マス</div><div>${esc(square)}</div>
        <div class="k">認識</div><div>${esc(pieceText(cellIdentity(cell)))}</div>
        <div class="k">正解</div><div>${sample.hasLabel ? esc(pieceText(cellIdentity(labelCell))) : "<span class='muted'>なし</span>"}</div>
        <div class="k">信頼度</div><div>${cell?.confidence == null ? "-" : Number(cell.confidence).toFixed(4)}</div>
        <div class="k">状態</div><div>${esc(cell?.state || "-")}</div>
        <div class="k">補正</div><div>${esc(cell?.postprocessReason || "-")}</div>
        <div class="k">診断</div><div>${esc(error?.diagnostic_category || "-")}</div>
      </div>
    </div>
    <div class="section">
      <h2>top-k候補</h2>
      ${renderCandidates(cell)}
    </div>
    <div class="section">
      <h2>持ち駒</h2>
      ${renderHands(sample)}
      <div class="muted" style="margin-top:6px">表示は 認識/正解 です。</div>
    </div>
    <div class="section">
      <h2>solver / 評価メタ</h2>
      <div class="kv">
        <div class="k">unique</div><div>${esc(cp.unique_solution ?? sample.solver?.globalSolver?.unique_solution ?? "-")}</div>
        <div class="k">2位gap</div><div>${esc(cp.second_best_gap ?? sample.solver?.globalSolver?.second_best_gap ?? "-")}</div>
        <div class="k">timeout</div><div>${esc(cp.solver_timeout ?? sample.solver?.globalSolver?.solver_timeout ?? "-")}</div>
        <div class="k">盤面差分</div><div>${esc(sample.status?.boardErrors ?? 0)}</div>
        <div class="k">持ち駒差分</div><div>${esc(sample.status?.handErrors ?? 0)}</div>
      </div>
    </div>
    <div class="section">
      <h2>KIF / SFEN</h2>
      ${renderExport(sample)}
    </div>`;
  bindExport(sample);
}
function renderAll() {
  renderSummary();
  renderFilters();
  renderList();
  renderSample();
}
document.getElementById("searchInput").addEventListener("input", event => {
  query = event.target.value || "";
  renderAll();
});
renderAll();
</script>
</body>
</html>
"""


def write_image_analysis_html(
    reports: Path,
    labels_dir: Path | None = None,
    out_path: Path | None = None,
    evaluation_path: Path | None = None,
    include_hands: bool = False,
    low_confidence: float = 0.55,
) -> Path:
    paths = report_paths(reports)
    evaluation_map, evaluation_metrics = load_evaluation_map(evaluation_path)
    samples = [
        build_sample(path, labels_dir, evaluation_map, include_hands, low_confidence)
        for path in paths
    ]
    resolved_out = out_path or (reports if reports.is_dir() else reports.parent) / "image_analysis.html"
    payload = {
        "schemaVersion": 1,
        "runDir": str(reports),
        "labelsDir": str(labels_dir) if labels_dir else "",
        "evaluationPath": str(evaluation_path) if evaluation_path else "",
        "summary": aggregate(samples, evaluation_metrics),
        "samples": samples,
    }
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(render_html(payload), encoding="utf-8")
    return resolved_out


def main() -> None:
    parser = argparse.ArgumentParser(description="駒認識結果の解析用HTMLを作成します。")
    parser.add_argument("reports", type=Path, help="runディレクトリ、サンプル出力ディレクトリ、または piece_report.json。")
    parser.add_argument("--labels", "--labels-dir", dest="labels_dir", type=Path, default=default_labels_dir())
    parser.add_argument("--no-labels", action="store_true", help="教師ラベルを読み込まず、認識結果のみでHTMLを作成します。")
    parser.add_argument("--evaluation", type=Path, help="evaluation_summary*.json。指定時は既存の誤認分類を取り込みます。")
    parser.add_argument("--out", type=Path, help="HTML出力先。既定では <reports>/image_analysis.html。")
    parser.add_argument("--include-hands", action="store_true", help="教師ラベルに持ち駒がある場合、持ち駒も比較します。")
    parser.add_argument("--low-confidence", type=float, default=0.55)
    args = parser.parse_args()

    labels_dir = None if args.no_labels else args.labels_dir
    out = write_image_analysis_html(
        args.reports,
        labels_dir=labels_dir,
        out_path=args.out,
        evaluation_path=args.evaluation,
        include_hands=args.include_hands,
        low_confidence=args.low_confidence,
    )
    print(f"OK: {out} を作成しました（{len(report_paths(args.reports))}件）")


if __name__ == "__main__":
    main()
