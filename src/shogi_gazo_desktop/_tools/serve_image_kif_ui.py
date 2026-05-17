from __future__ import annotations

import base64
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root()
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from shogi_gazo_desktop.export import ExportError, export_kif, export_sfen  # noqa: E402
from shogi_gazo_desktop.models import HAND_PIECES, RecognitionOptions, RecognitionResult, empty_hands  # noqa: E402
from shogi_gazo_desktop.paths import (  # noqa: E402
    BUNDLED_MODEL_PATHS_BY_TARGET_HINT,
    DEFAULT_KIF_UI_MODEL_PATH,
    DEFAULT_LABELS_DIR,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SCREENSHOTS_DIR,
)
from shogi_gazo_desktop.recognition import recognize_image  # noqa: E402


MAX_UPLOAD_BYTES = 35 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
TARGET_HINTS = {
    "": "",
    "将棋ウォーズ_一文字": "将棋ウォーズ_一文字",
    "将棋ウォーズ_二文字": "将棋ウォーズ_二文字",
    "将棋クエスト_一文字駒": "将棋クエスト_一文字駒",
    "将棋クエスト_クラシック二文字駒": "将棋クエスト_クラシック二文字駒",
    "ぴよ将棋_ひよこ駒": "ぴよ将棋_ひよこ駒",
    "ぴよ将棋_一文字駒": "ぴよ将棋_一文字駒",
    "ぴよ将棋_二文字駒": "ぴよ将棋_二文字駒",
}
BOARD_PIECES = ("OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU", "RY", "UM", "NG", "NK", "NY", "TO")
BOARD_VALUES = {"empty", "unknown"} | {
    f"{color}:{piece}"
    for color in ("black", "white")
    for piece in BOARD_PIECES
}


@dataclass(frozen=True)
class KifUiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    out_dir: Path = DEFAULT_OUTPUTS_DIR / "kif_ui"
    model_path: Path | None = None
    screenshots_dir: Path = DEFAULT_SCREENSHOTS_DIR
    labels_dir: Path = DEFAULT_LABELS_DIR
    calibration_dir: Path | None = None
    include_hands: bool = True
    train_if_missing: bool = True


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, separator, encoded = data_url.partition(",")
    if separator != "," or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("image payload must be a base64 data URL")
    mime = header[5:].split(";", 1)[0].strip().lower()
    if mime not in MIME_SUFFIX:
        raise ValueError(f"unsupported image type: {mime or 'unknown'}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("image payload is not valid base64") from exc
    if not data:
        raise ValueError("image payload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"image payload is too large: {len(data)} bytes")
    return data, mime


def safe_upload_name(filename: str, mime: str, target_hint: str = "") -> str:
    raw_name = Path(filename or "image").name
    raw_suffix = Path(raw_name).suffix.lower()
    suffix = raw_suffix if raw_suffix in ALLOWED_SUFFIXES else MIME_SUFFIX.get(mime, ".png")
    stem = Path(raw_name).stem
    prefix = TARGET_HINTS.get(target_hint, "")
    stem = f"{prefix}_{stem}" if prefix and prefix not in stem else stem
    stem = re.sub(r"[^\w._-]+", "_", stem, flags=re.UNICODE).strip("._-") or "image"
    stem = stem[:72]
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


def prepare_image_for_recognition(image_path: Path) -> Path:
    with Image.open(image_path) as image:
        width, height = image.size
        scale = max(1, int(max(900 / max(1, width), 1600 / max(1, height)) + 0.9999))
        scale = min(scale, 4)
        if scale <= 1:
            return image_path
        resized = image.convert("RGB").resize((width * scale, height * scale), Image.Resampling.LANCZOS)
    upscaled = image_path.with_name(f"{image_path.stem}_upscaled{image_path.suffix}")
    resized.save(upscaled)
    return upscaled


def model_path_for_target_hint(config: KifUiConfig, target_hint: str) -> Path:
    if config.model_path is not None:
        return config.model_path
    return BUNDLED_MODEL_PATHS_BY_TARGET_HINT.get(target_hint, DEFAULT_KIF_UI_MODEL_PATH)


def export_payload(result: RecognitionResult, side_to_move: str) -> dict[str, str]:
    payload = {"kif": "", "sfen": "", "error": ""}
    try:
        payload["kif"] = export_kif(result, side_to_move=side_to_move)
        payload["sfen"] = export_sfen(result, side_to_move=side_to_move)
    except (ExportError, ValueError) as exc:
        payload["error"] = str(exc)
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


def candidate_identity(candidate: dict[str, Any]) -> str:
    color = candidate.get("color")
    piece = candidate.get("piece")
    if color and piece:
        return f"{color}:{piece}"
    return str(candidate.get("identity") or candidate.get("value") or "none")


def compact_cells(raw_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for cell in raw_cells:
        candidates = []
        for candidate in (cell.get("candidates") or [])[:8]:
            if not isinstance(candidate, dict):
                continue
            candidates.append(
                {
                    "identity": candidate_identity(candidate),
                    "score": candidate.get("score", candidate.get("confidence")),
                    "source": candidate.get("source"),
                }
            )
        if cell.get("state") == "piece" and cell.get("color") and cell.get("piece"):
            identity = f"{cell.get('color')}:{cell.get('piece')}"
        else:
            identity = str(cell.get("state") or "unknown")
        cells.append(
            {
                "row": cell.get("row"),
                "col": cell.get("col"),
                "square": cell.get("square"),
                "state": cell.get("state"),
                "color": cell.get("color"),
                "piece": cell.get("piece"),
                "identity": identity,
                "confidence": cell.get("confidence"),
                "bboxRatio": cell.get("bbox_ratio"),
                "postprocessReason": cell.get("postprocess_reason"),
                "postprocessHistory": cell.get("postprocess_history") or [],
                "candidates": candidates,
            }
        )
    return cells


def compact_rect(rect: Any) -> list[float] | None:
    if not isinstance(rect, list | tuple) or len(rect) != 4:
        return None
    try:
        return [float(value) for value in rect]
    except (TypeError, ValueError):
        return None


def compact_hand_candidates(candidates: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return compact
    for candidate in candidates[:8]:
        if not isinstance(candidate, dict):
            continue
        compact.append(
            {
                "identity": candidate_identity(candidate),
                "score": candidate.get("score", candidate.get("confidence")),
                "source": candidate.get("source"),
            }
        )
    return compact


def compact_hand_recognition(raw: dict[str, Any]) -> dict[str, Any]:
    hand_report = raw.get("hand_recognition") or {}
    if not isinstance(hand_report, dict):
        return {}
    areas = []
    for index, area in enumerate(hand_report.get("areas") or []):
        if not isinstance(area, dict):
            continue
        rect = compact_rect(area.get("rect"))
        if not rect:
            continue
        areas.append(
            {
                "id": f"area-{index}",
                "owner": area.get("owner"),
                "side": area.get("side"),
                "rect": rect,
                "confidence": area.get("confidence"),
                "evidence": area.get("evidence"),
            }
        )
    pieces = []
    for index, piece in enumerate(hand_report.get("pieces") or []):
        if not isinstance(piece, dict):
            continue
        rects = [rect for rect in (compact_rect(item) for item in piece.get("rects") or []) if rect]
        candidate_sets = []
        for set_index, candidate_set in enumerate(piece.get("candidate_sets") or []):
            if not isinstance(candidate_set, dict):
                continue
            rect = compact_rect(candidate_set.get("rect"))
            candidate_sets.append(
                {
                    "id": f"hand-{index}-candidates-{set_index}",
                    "rect": rect,
                    "candidates": compact_hand_candidates(candidate_set.get("candidates")),
                }
            )
        digits = []
        for digit_index, digit in enumerate(piece.get("digits") or []):
            if not isinstance(digit, dict):
                continue
            rect = compact_rect(digit.get("rect"))
            digits.append(
                {
                    "id": f"hand-{index}-digit-{digit_index}",
                    "rect": rect,
                    "digit": digit.get("digit"),
                    "confidence": digit.get("confidence"),
                }
            )
        pieces.append(
            {
                "id": f"hand-{index}",
                "owner": piece.get("owner"),
                "piece": piece.get("piece"),
                "identity": f"{piece.get('owner')}:{piece.get('piece')}" if piece.get("owner") and piece.get("piece") else "unknown",
                "count": piece.get("count"),
                "countSource": piece.get("count_source"),
                "confidence": piece.get("confidence"),
                "rects": rects,
                "digits": digits,
                "candidateSets": candidate_sets,
                "ambiguous": bool(piece.get("ambiguous")),
            }
        )
    unknown = []
    for index, item in enumerate(hand_report.get("unknown") or []):
        if not isinstance(item, dict):
            continue
        rect = compact_rect(item.get("rect"))
        unknown.append(
            {
                "id": f"unknown-hand-{index}",
                "owner": item.get("owner"),
                "rect": rect,
                "candidates": compact_hand_candidates(item.get("candidates")),
            }
        )
    return {
        "hands": hand_report.get("hands") or {},
        "targetFamily": hand_report.get("target_family"),
        "areas": areas,
        "pieces": pieces,
        "unknown": unknown,
        "ownerFlip": hand_report.get("owner_flip") or {},
        "inventorySanitization": hand_report.get("inventory_sanitization") or {},
        "inventoryCompletion": hand_report.get("inventory_completion") or {},
    }


def response_from_result(result: RecognitionResult, side_to_move: str, elapsed_seconds: float) -> dict[str, Any]:
    raw = result.raw_report or {}
    export = export_payload(result, side_to_move)
    return {
        "ok": True,
        "image": result.image,
        "outputPath": result.output_path,
        "reportPath": str(Path(result.output_path).with_name("piece_report.json")) if result.output_path else "",
        "needsReview": result.needs_review,
        "reviewReasons": result.review_reasons,
        "board": result.board,
        "hands": result.hands,
        "confidence": result.confidence,
        "grid": raw.get("grid") or {},
        "cells": compact_cells(raw.get("cells") or []),
        "handRecognition": compact_hand_recognition(raw),
        "summary": raw.get("summary") or {},
        "timing": raw.get("timing") or {},
        "elapsedSeconds": round(elapsed_seconds, 3),
        "export": export,
    }


def recognize_upload(payload: dict[str, Any], config: KifUiConfig) -> dict[str, Any]:
    data, mime = decode_data_url(str(payload.get("dataUrl") or ""))
    filename = str(payload.get("filename") or "image")
    side_to_move = str(payload.get("sideToMove") or "black")
    if side_to_move not in {"black", "white"}:
        raise ValueError("sideToMove must be black or white")
    include_hands = bool(payload.get("includeHands", config.include_hands))
    target_hint = str(payload.get("targetHint") or "")
    if target_hint not in TARGET_HINTS:
        raise ValueError("unsupported targetHint")

    upload_dir = config.out_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / safe_upload_name(filename, mime, target_hint)
    image_path.write_bytes(data)
    recognition_image_path = prepare_image_for_recognition(image_path)

    options = RecognitionOptions(
        model_path=model_path_for_target_hint(config, target_hint),
        screenshots_dir=config.screenshots_dir,
        labels_dir=config.labels_dir,
        calibration_dir=config.calibration_dir or config.screenshots_dir,
        include_hands=include_hands,
        train_if_missing=config.train_if_missing,
        out_dir=config.out_dir / "runs",
    )
    start = time.perf_counter()
    result = recognize_image(recognition_image_path, options)
    return response_from_result(result, side_to_move, time.perf_counter() - start)


def normalize_board_payload(board_payload: Any) -> list[list[str]]:
    if not isinstance(board_payload, list) or len(board_payload) != 9:
        raise ValueError("board must be a 9x9 array")
    board: list[list[str]] = []
    for row in board_payload:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("board must be a 9x9 array")
        normalized_row = []
        for cell in row:
            value = str(cell or "empty")
            if value not in BOARD_VALUES:
                raise ValueError(f"unsupported board cell: {value}")
            normalized_row.append(value)
        board.append(normalized_row)
    return board


def normalize_hands_payload(hands_payload: Any) -> dict[str, dict[str, int]]:
    hands = empty_hands()
    if hands_payload is None:
        return hands
    if not isinstance(hands_payload, dict):
        raise ValueError("hands must be an object")
    for color in ("black", "white"):
        raw_counts = hands_payload.get(color) or {}
        if not isinstance(raw_counts, dict):
            raise ValueError(f"hands.{color} must be an object")
        for piece in HAND_PIECES:
            try:
                count = int(raw_counts.get(piece, 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid hand count: {color}:{piece}") from exc
            if count < 0:
                raise ValueError(f"negative hand count: {color}:{piece}")
            hands[color][piece] = count
    return hands


def export_edited_position(payload: dict[str, Any]) -> dict[str, Any]:
    side_to_move = str(payload.get("sideToMove") or "black")
    if side_to_move not in {"black", "white"}:
        raise ValueError("sideToMove must be black or white")
    result = RecognitionResult(
        image=str(payload.get("image") or ""),
        board=normalize_board_payload(payload.get("board")),
        hands=normalize_hands_payload(payload.get("hands")),
        confidence=[],
        raw_report={},
    )
    return {"ok": True, "export": export_payload(result, side_to_move)}


def json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def make_handler(config: KifUiConfig) -> type[BaseHTTPRequestHandler]:
    class KifUiHandler(BaseHTTPRequestHandler):
        server_version = "ShogiGazoKifUI/1.0"

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self.send_html(HTML)
                return
            if self.path == "/api/health":
                self.send_json({"ok": True})
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path not in {"/api/recognize", "/api/export"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0:
                    raise ValueError("empty request body")
                if length > MAX_UPLOAD_BYTES * 2:
                    raise ValueError("request body is too large")
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/api/recognize":
                    self.send_json(recognize_upload(payload, config))
                else:
                    self.send_json(export_edited_position(payload))
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        def send_html(self, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data: dict[str, Any], status: int = 200) -> None:
            raw = json_bytes(data)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    return KifUiHandler


def serve_kif_ui(config: KifUiConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    url = f"http://{config.host}:{server.server_port}/"
    print(f"Shogi KIF UI: {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>将棋画像 KIF出力</title>
<style>
:root {
  --bg: #f3efe6;
  --panel: #fffaf1;
  --line: #d7c5a8;
  --ink: #251d13;
  --muted: #6f6252;
  --accent: #116c5a;
  --bad: #b9322b;
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
button, input, select, textarea { font: inherit; }
.app {
  display: grid;
  grid-template-rows: auto 1fr;
  min-height: 100vh;
}
.bar {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) auto auto auto auto auto;
  gap: 10px;
  align-items: end;
  padding: 12px;
  border-bottom: 1px solid var(--line);
  background: rgba(243, 239, 230, 0.97);
}
.field {
  display: grid;
  gap: 4px;
}
.field span {
  color: var(--muted);
  font-size: 12px;
}
input[type=file], select {
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  padding: 6px;
}
.paste-btn {
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  color: var(--ink);
  cursor: pointer;
  padding: 0 10px;
}
.paste-btn:active {
  background: #f8efdf;
}
.check {
  display: flex;
  gap: 6px;
  align-items: center;
  min-height: 36px;
  color: var(--muted);
}
.run {
  min-height: 36px;
  border: 1px solid var(--accent);
  border-radius: 6px;
  background: var(--accent);
  color: white;
  padding: 0 14px;
  cursor: pointer;
}
.run:disabled {
  opacity: 0.55;
  cursor: wait;
}
.main {
  display: grid;
  grid-template-columns: minmax(420px, 1.18fr) minmax(480px, 1fr) minmax(330px, 0.78fr);
  gap: 12px;
  padding: 12px;
}
.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  min-height: calc(100vh - 86px);
  overflow: hidden;
}
.panel h2 {
  margin: 0;
  padding: 11px 12px;
  border-bottom: 1px solid var(--line);
  font-size: 15px;
  background: #f8efdf;
}
.content {
  padding: 12px;
}
#result {
  display: flex;
  flex-direction: column;
}
#preview {
  text-align: center;
}
.preview-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
  background: #111;
  border: 1px solid #111;
}
.preview {
  display: block;
  max-width: 100%;
  max-height: calc(100vh - 120px);
  width: auto;
  height: auto;
}
.image-grid {
  position: absolute;
  border: 2px solid rgba(255, 255, 255, 0.86);
  pointer-events: auto;
}
.image-cell {
  position: absolute;
  appearance: none;
  border: 1px solid rgba(255, 255, 255, 0.42);
  background: transparent;
  color: #111;
  cursor: pointer;
  pointer-events: auto;
  padding: 0;
}
.image-cell:hover,
.image-cell.selected {
  background: rgba(255, 246, 175, 0.25);
  outline: 3px solid #fff175;
  z-index: 2;
}
.hand-area {
  position: absolute;
  border: 2px dashed rgba(30, 215, 170, 0.82);
  background: rgba(30, 215, 170, 0.08);
  pointer-events: none;
}
.hand-piece-box {
  position: absolute;
  appearance: none;
  border: 2px solid rgba(255, 184, 77, 0.9);
  background: rgba(255, 184, 77, 0.12);
  color: #111;
  cursor: pointer;
  padding: 0;
}
.hand-piece-box:hover,
.hand-piece-box.selected {
  background: rgba(255, 238, 165, 0.34);
  outline: 3px solid #ffca45;
  z-index: 4;
}
.empty {
  color: var(--muted);
  line-height: 1.7;
}
.status {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}
.badge {
  border-radius: 5px;
  padding: 3px 7px;
  background: #e8ddca;
  font-size: 12px;
}
.badge.bad { background: #f5d2ce; color: #7f1813; }
.badge.ok { background: #d9ece4; color: #164d3e; }
.board {
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  width: min(100%, 430px);
  aspect-ratio: 1 / 1;
  border: 2px solid #744d27;
  background: var(--board);
  margin: 4px auto;
}
.cell {
  position: relative;
  appearance: none;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #87643a;
  background: var(--board);
  color: var(--ink);
  padding: 0;
  min-width: 0;
  font-weight: 800;
  font-size: 17px;
  cursor: pointer;
}
.cell .sq {
  position: absolute;
  top: 2px;
  left: 3px;
  color: rgba(51, 35, 19, 0.72);
  font-size: 10px;
  font-weight: 400;
}
.cell.empty .piece { color: transparent; }
.cell.selected {
  outline: 3px solid #111;
  z-index: 2;
}
.cell.editing {
  outline: 3px solid var(--accent);
  z-index: 3;
}
.piece {
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  padding: 13px 2px 2px;
}
.piece-select {
  width: 100%;
  height: 100%;
  border: 0;
  background: transparent;
  color: var(--ink);
  cursor: pointer;
  font-weight: 800;
  text-align: center;
  text-align-last: center;
  padding: 13px 2px 2px;
}
.piece-select:focus {
  outline: 0;
}
.cell.empty .piece-select {
  color: rgba(51, 35, 19, 0.42);
  font-weight: 700;
}
.hand-summary {
  margin: 5px 0;
}
.hand-owner-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fffdf8;
  overflow: hidden;
}
.hand-owner-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 5px 8px;
  background: #f8efdf;
  border-bottom: 1px solid var(--line);
  font-weight: 800;
}
.hand-owner-total {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.hand-piece-grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 4px;
  padding: 5px;
}
.hand-piece-tile {
  min-height: 52px;
  border: 1px solid #dfcfb4;
  border-radius: 7px;
  background: white;
  color: var(--ink);
  cursor: pointer;
  padding: 4px;
  text-align: center;
}
.hand-piece-tile.zero {
  background: #f6f0e6;
  color: #9a8d7a;
}
.hand-piece-tile.detected {
  border-color: #b06a00;
  box-shadow: inset 0 0 0 2px rgba(255, 184, 77, 0.32);
}
.hand-piece-tile.selected {
  outline: 3px solid rgba(255, 184, 77, 0.55);
  border-color: #b06a00;
}
.hand-piece-name {
  font-size: 11px;
  font-weight: 800;
  line-height: 1.1;
}
.hand-piece-count {
  margin-top: 2px;
  font-size: 16px;
  font-weight: 900;
  line-height: 1;
}
.hand-count-input {
  width: 100%;
  min-height: 24px;
  border: 1px solid #dfcfb4;
  border-radius: 5px;
  background: white;
  color: var(--ink);
  font-size: 15px;
  font-weight: 900;
  line-height: 1;
  text-align: center;
}
.hand-count-input:focus {
  outline: 2px solid rgba(17, 108, 90, 0.35);
  border-color: var(--accent);
}
.hand-piece-meta {
  margin-top: 2px;
  color: var(--muted);
  font-size: 10px;
  min-height: 12px;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 8px;
}
.small {
  min-height: 32px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  cursor: pointer;
  padding: 0 10px;
}
.small.copied {
  border-color: var(--accent);
  background: #d9ece4;
  color: #164d3e;
}
.copy-status {
  min-height: 20px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}
textarea {
  width: 100%;
  min-height: 330px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fffefb;
  padding: 8px;
  font-family: Consolas, "Yu Gothic", monospace;
  font-size: 12px;
  line-height: 1.45;
}
.sfen {
  margin-top: 8px;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: white;
  font-family: Consolas, monospace;
  font-size: 12px;
  overflow-wrap: anywhere;
}
.error {
  color: var(--bad);
  font-weight: 700;
  line-height: 1.6;
}
@media (max-width: 1150px) {
  .bar { grid-template-columns: 1fr 1fr; }
  .main { grid-template-columns: 1fr; }
  .panel { min-height: auto; }
}
@media (max-width: 660px) {
  .bar { grid-template-columns: 1fr; }
  .cell { font-size: 13px; }
  .hand-piece-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<div class="app">
  <header class="bar">
    <label class="field">
      <span>画像</span>
      <input id="file" type="file" accept="image/png,image/jpeg,image/webp">
    </label>
    <label class="field">
      <span>貼り付け</span>
      <button class="paste-btn" id="pasteButton" type="button">Ctrl+V / 貼付</button>
    </label>
    <label class="field">
      <span>手番</span>
      <select id="side">
        <option value="black">先手番</option>
        <option value="white">後手番</option>
      </select>
    </label>
    <label class="field">
      <span>認識スタイル</span>
      <select id="targetHint">
        <option value="将棋クエスト_一文字駒" selected>将棋クエスト 一文字</option>
        <option value="">自動</option>
        <option value="将棋ウォーズ_一文字">将棋ウォーズ 一文字</option>
        <option value="将棋ウォーズ_二文字">将棋ウォーズ 二文字</option>
        <option value="将棋クエスト_クラシック二文字駒">将棋クエスト クラシック二文字</option>
        <option value="ぴよ将棋_ひよこ駒">ぴよ将棋 ひよこ</option>
        <option value="ぴよ将棋_一文字駒">ぴよ将棋 一文字</option>
        <option value="ぴよ将棋_二文字駒">ぴよ将棋 二文字</option>
      </select>
    </label>
    <label class="check">
      <input id="hands" type="checkbox" checked>
      持ち駒も読む
    </label>
    <label class="field">
      <span>表示倍率</span>
      <select id="scale">
        <option value="0.7" selected>70%</option>
        <option value="0.85">85%</option>
        <option value="1">100%</option>
      </select>
    </label>
    <button class="run" id="run" disabled>認識してKIF作成</button>
  </header>
  <main class="main">
    <section class="panel">
      <h2>画像</h2>
      <div class="content" id="preview"><div class="empty">画像を選択してください。</div></div>
    </section>
    <section class="panel">
      <h2>認識結果</h2>
      <div class="content" id="result"><div class="empty">まだ認識していません。</div></div>
    </section>
    <section class="panel">
      <h2>KIF / SFEN</h2>
      <div class="content" id="export"><div class="empty">KIFは認識後に表示されます。</div></div>
    </section>
  </main>
</div>
<script>
const PIECE_TEXT = {OU:"玉",HI:"飛",KA:"角",KI:"金",GI:"銀",KE:"桂",KY:"香",FU:"歩",RY:"龍",UM:"馬",NG:"成銀",NK:"成桂",NY:"成香",TO:"と"};
const COLOR_MARK = {black:"▲", white:"△"};
const COLOR_TEXT = {black:"先手", white:"後手"};
const HAND_PIECES = ["HI","KA","KI","GI","KE","KY","FU"];
const BOARD_PIECES = ["OU","HI","KA","KI","GI","KE","KY","FU","RY","UM","NG","NK","NY","TO"];
const BOARD_VALUES = ["empty", ...BOARD_PIECES.map(piece => `black:${piece}`), ...BOARD_PIECES.map(piece => `white:${piece}`), "unknown"];
const fileInput = document.getElementById("file");
const pasteButton = document.getElementById("pasteButton");
const runButton = document.getElementById("run");
const scaleSelect = document.getElementById("scale");
const sideSelect = document.getElementById("side");
const targetHintSelect = document.getElementById("targetHint");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const exportBox = document.getElementById("export");
let selectedFile = null;
let currentExport = null;
let imageObjectUrl = "";
let lastData = null;
let selectedSquare = null;
let editingSquare = null;
let selectedHandId = null;
let selectedHandKey = null;
let exportRequestId = 0;
document.body.style.zoom = scaleSelect.value;
targetHintSelect.value = "将棋クエスト_一文字駒";
scaleSelect.addEventListener("change", () => {
  document.body.style.zoom = scaleSelect.value;
});
sideSelect.addEventListener("change", () => {
  if (lastData) refreshExport(lastData);
});
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
}
function pieceText(value) {
  if (!value || value === "empty") return "";
  if (value === "unknown") return "?";
  const [color, piece] = String(value).split(":");
  return `${COLOR_MARK[color] || ""}${PIECE_TEXT[piece] || piece || ""}`;
}
function handPieceText(owner, piece, count) {
  const base = `${COLOR_MARK[owner] || ""}${PIECE_TEXT[piece] || piece || "?"}`;
  const n = Number(count || 0);
  return n > 1 ? `${base}x${n}` : base;
}
function ownerText(owner) {
  return COLOR_TEXT[owner] || owner || "?";
}
function handKey(owner, piece) {
  return owner && piece ? `${owner}:${piece}` : "";
}
function boardPieceLabel(value) {
  if (!value || value === "empty") return "・";
  if (value === "unknown") return "?";
  return pieceText(value);
}
function boardPieceOptions(value) {
  const current = BOARD_VALUES.includes(value) ? value : "unknown";
  return BOARD_VALUES.map(option => (
    `<option value="${esc(option)}" ${option === current ? "selected" : ""}>${esc(boardPieceLabel(option))}</option>`
  )).join("");
}
function squareName(row, col) {
  return "987654321"[col - 1] + "一二三四五六七八九"[row - 1];
}
function squareCoords(square) {
  const text = String(square || "");
  const col = "987654321".indexOf(text[0]) + 1;
  const row = "一二三四五六七八九".indexOf(text[1]) + 1;
  return row > 0 && col > 0 ? {row, col} : null;
}
function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
  if (file) setSelectedImage(file);
});
document.addEventListener("paste", event => {
  const file = imageFileFromClipboard(event.clipboardData);
  if (!file) return;
  event.preventDefault();
  setSelectedImage(file);
});
pasteButton.addEventListener("click", async () => {
  const file = await readImageFromClipboard();
  if (file) setSelectedImage(file);
});
function setSelectedImage(file) {
  selectedFile = file;
  runButton.disabled = !selectedFile;
  currentExport = null;
  lastData = null;
  exportRequestId += 1;
  selectedSquare = null;
  editingSquare = null;
  selectedHandId = null;
  selectedHandKey = null;
  if (selectedFile) {
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(selectedFile);
    renderPreview(null);
    result.innerHTML = `<div class="empty">まだ認識していません。</div>`;
    exportBox.innerHTML = `<div class="empty">KIFは認識後に表示されます。</div>`;
  }
}
function imageFileFromClipboard(clipboardData) {
  const items = Array.from(clipboardData?.items || []);
  for (const item of items) {
    if (!item.type || !item.type.startsWith("image/")) continue;
    const file = item.getAsFile();
    if (!file) continue;
    return file.name ? file : new File([file], `clipboard_${Date.now()}.png`, {type: file.type || "image/png"});
  }
  return null;
}
async function readImageFromClipboard() {
  if (!navigator.clipboard?.read) {
    pasteButton.textContent = "Ctrl+Vで貼付";
    window.setTimeout(() => { pasteButton.textContent = "Ctrl+V / 貼付"; }, 1400);
    return null;
  }
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find(value => value.startsWith("image/"));
      if (!type) continue;
      const blob = await item.getType(type);
      return new File([blob], `clipboard_${Date.now()}.${type.includes("jpeg") ? "jpg" : type.split("/")[1] || "png"}`, {type});
    }
  } catch (_) {
    pasteButton.textContent = "Ctrl+Vで貼付";
    window.setTimeout(() => { pasteButton.textContent = "Ctrl+V / 貼付"; }, 1400);
  }
  return null;
}
runButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  runButton.disabled = true;
  runButton.textContent = "認識中...";
  result.innerHTML = `<div class="empty">処理中です。</div>`;
  exportBox.innerHTML = `<div class="empty">処理中です。</div>`;
  try {
    const dataUrl = await fileToDataUrl(selectedFile);
    const response = await fetch("/api/recognize", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        filename: selectedFile.name,
        dataUrl,
        sideToMove: sideSelect.value,
        includeHands: document.getElementById("hands").checked,
        targetHint: targetHintSelect.value
      })
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || "recognition failed");
    lastData = data;
    selectedSquare = firstPieceSquare(data) || "5五";
    editingSquare = null;
    selectedHandId = null;
    selectedHandKey = null;
    renderPreview(data);
    renderResult(data);
    renderExport(data);
  } catch (error) {
    result.innerHTML = `<div class="error">${esc(error.message || error)}</div>`;
    exportBox.innerHTML = `<div class="error">${esc(error.message || error)}</div>`;
  } finally {
    runButton.disabled = !selectedFile;
    runButton.textContent = "認識してKIF作成";
  }
});
function firstPieceSquare(data) {
  for (const cell of data.cells || []) {
    if (cell.state === "piece") return cell.square;
  }
  return null;
}
function cellBySquare(data) {
  const map = {};
  for (const cell of data?.cells || []) {
    if (cell.square) map[cell.square] = cell;
  }
  return map;
}
function boardValue(data, row, col) {
  return data?.board?.[row - 1]?.[col - 1] || "empty";
}
function ensureBoardShape(data) {
  if (!Array.isArray(data.board)) data.board = [];
  for (let row = 0; row < 9; row++) {
    if (!Array.isArray(data.board[row])) data.board[row] = [];
    for (let col = 0; col < 9; col++) {
      if (!data.board[row][col]) data.board[row][col] = "empty";
    }
  }
}
function ensureHandsShape(data) {
  if (!data.hands || typeof data.hands !== "object") data.hands = {};
  for (const color of ["black", "white"]) {
    if (!data.hands[color] || typeof data.hands[color] !== "object") data.hands[color] = {};
    for (const piece of HAND_PIECES) {
      data.hands[color][piece] = Math.max(0, Number.parseInt(data.hands[color][piece] || 0, 10) || 0);
    }
  }
}
function markEdited(data) {
  data.edited = true;
}
function updateCellMetadata(data, square, value) {
  const coords = squareCoords(square);
  if (!coords) return;
  if (!Array.isArray(data.cells)) data.cells = [];
  let cell = data.cells.find(item => item.square === square);
  if (!cell) {
    cell = {row: coords.row, col: coords.col, square};
    data.cells.push(cell);
  }
  cell.state = value === "empty" ? "empty" : (value === "unknown" ? "unknown" : "piece");
  cell.color = null;
  cell.piece = null;
  cell.identity = value;
  cell.confidence = null;
  cell.postprocessReason = "manual";
  cell.candidates = [];
  if (cell.state === "piece") {
    const [color, piece] = value.split(":");
    cell.color = color;
    cell.piece = piece;
  }
}
function setBoardValue(data, square, value) {
  const coords = squareCoords(square);
  if (!coords || !BOARD_VALUES.includes(value)) return;
  ensureBoardShape(data);
  data.board[coords.row - 1][coords.col - 1] = value;
  updateCellMetadata(data, square, value);
  markEdited(data);
}
function setHandCount(data, owner, piece, count) {
  if (!["black", "white"].includes(owner) || !HAND_PIECES.includes(piece)) return;
  ensureHandsShape(data);
  data.hands[owner][piece] = Math.max(0, Number.parseInt(count, 10) || 0);
  markEdited(data);
}
async function refreshExport(data) {
  const requestId = ++exportRequestId;
  try {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        image: data.image || "",
        board: data.board,
        hands: data.hands || {},
        sideToMove: sideSelect.value
      })
    });
    const payload = await response.json();
    if (requestId !== exportRequestId) return;
    if (!payload.ok) throw new Error(payload.error || "export failed");
    data.export = payload.export || {};
    renderExport(data);
  } catch (error) {
    if (requestId !== exportRequestId) return;
    data.export = {error: String(error.message || error)};
    renderExport(data);
  }
}
function renderPreview(data) {
  if (!imageObjectUrl) {
    preview.innerHTML = `<div class="empty">画像を選択してください。</div>`;
    return;
  }
  const overlay = data ? `${renderImageGrid(data)}${renderHandOverlay(data)}` : "";
  preview.innerHTML = `<div class="preview-wrap"><img class="preview" src="${imageObjectUrl}" alt="">${overlay}</div>`;
  bindCompareClicks();
  bindHandClicks();
}
function renderImageGrid(data) {
  const grid = data.grid || {};
  const size = grid.image_size || [];
  const rect = grid.grid_rect || grid.board_rect;
  if (!rect || !size[0] || !size[1]) return "";
  const left = rect.left / size[0] * 100;
  const top = rect.top / size[1] * 100;
  const width = rect.width / size[0] * 100;
  const height = rect.height / size[1] * 100;
  const vertical = Array.isArray(grid.vertical_lines) && grid.vertical_lines.length >= 10 ? grid.vertical_lines : null;
  const horizontal = Array.isArray(grid.horizontal_lines) && grid.horizontal_lines.length >= 10 ? grid.horizontal_lines : null;
  const cells = [];
  for (let row = 1; row <= 9; row++) {
    for (let col = 1; col <= 9; col++) {
      const square = squareName(row, col);
      const value = boardValue(data, row, col);
      const cellLeft = vertical ? ((vertical[col - 1] - rect.left) / rect.width * 100) : ((col - 1) / 9 * 100);
      const cellTop = horizontal ? ((horizontal[row - 1] - rect.top) / rect.height * 100) : ((row - 1) / 9 * 100);
      const cellWidth = vertical ? ((vertical[col] - vertical[col - 1]) / rect.width * 100) : (100 / 9);
      const cellHeight = horizontal ? ((horizontal[row] - horizontal[row - 1]) / rect.height * 100) : (100 / 9);
      cells.push(
        `<button class="image-cell ${selectedSquare === square ? "selected" : ""}" data-square="${square}" ` +
        `style="left:${cellLeft.toFixed(4)}%;top:${cellTop.toFixed(4)}%;` +
        `width:${cellWidth.toFixed(4)}%;height:${cellHeight.toFixed(4)}%" title="${square} ${esc(pieceText(value))}"></button>`
      );
    }
  }
  return `<div class="image-grid" style="left:${left.toFixed(4)}%;top:${top.toFixed(4)}%;width:${width.toFixed(4)}%;height:${height.toFixed(4)}%">${cells.join("")}</div>`;
}
function rectStyle(rect, imageSize) {
  if (!rect || !imageSize?.[0] || !imageSize?.[1]) return "";
  const [x1, y1, x2, y2] = rect;
  const left = x1 / imageSize[0] * 100;
  const top = y1 / imageSize[1] * 100;
  const width = (x2 - x1) / imageSize[0] * 100;
  const height = (y2 - y1) / imageSize[1] * 100;
  return `left:${left.toFixed(4)}%;top:${top.toFixed(4)}%;width:${width.toFixed(4)}%;height:${height.toFixed(4)}%`;
}
function renderHandOverlay(data) {
  const size = data.grid?.image_size || [];
  if (!size[0] || !size[1]) return "";
  const chunks = [];
  for (const area of data.handRecognition?.areas || []) {
    const style = rectStyle(area.rect, size);
    if (!style) continue;
    chunks.push(`<div class="hand-area" style="${style}" title="${esc(ownerText(area.owner))}持ち駒"></div>`);
  }
  for (const piece of data.handRecognition?.pieces || []) {
    const rects = piece.rects || [];
    rects.forEach((rect, index) => {
      const style = rectStyle(rect, size);
      if (!style) return;
      const id = `${piece.id}:${index}`;
      const key = handKey(piece.owner, piece.piece);
      chunks.push(
        `<button class="hand-piece-box ${selectedHandId === id || selectedHandKey === key ? "selected" : ""}" data-hand="${id}" data-hand-key="${esc(key)}" style="${style}" ` +
        `title="${esc(ownerText(piece.owner))} ${esc(handPieceText(piece.owner, piece.piece, piece.count))}"></button>`
      );
    });
  }
  for (const item of data.handRecognition?.unknown || []) {
    const style = rectStyle(item.rect, size);
    if (!style) continue;
    chunks.push(`<button class="hand-piece-box ${selectedHandId === item.id ? "selected" : ""}" data-hand="${item.id}" style="${style}" title="unknown"></button>`);
  }
  return chunks.join("");
}
function bindCompareClicks() {
  document.querySelectorAll(".image-cell[data-square], .cell[data-square]").forEach(element => {
    element.addEventListener("click", event => {
      if (event.target.closest(".piece-select")) return;
      selectedSquare = element.dataset.square;
      editingSquare = null;
      selectedHandId = null;
      selectedHandKey = null;
      if (lastData) {
        renderPreview(lastData);
        renderResult(lastData);
      }
    });
  });
}
function bindBoardEditors() {
  document.querySelectorAll(".cell[data-square]").forEach(element => {
    element.addEventListener("dblclick", event => {
      event.preventDefault();
      selectedSquare = element.dataset.square;
      editingSquare = element.dataset.square;
      selectedHandId = null;
      selectedHandKey = null;
      if (lastData) {
        renderPreview(lastData);
        renderResult(lastData);
      }
    });
  });
  document.querySelectorAll(".piece-select").forEach(element => {
    element.addEventListener("focus", () => {
      selectedSquare = element.dataset.square;
      editingSquare = element.dataset.square;
      selectedHandId = null;
      selectedHandKey = null;
      if (lastData) {
        renderPreview(lastData);
        updateSelectionClasses();
      }
    });
    element.addEventListener("change", () => {
      if (!lastData) return;
      selectedSquare = element.dataset.square;
      editingSquare = null;
      selectedHandId = null;
      selectedHandKey = null;
      setBoardValue(lastData, element.dataset.square, element.value);
      renderPreview(lastData);
      renderResult(lastData);
      refreshExport(lastData);
    });
    element.addEventListener("blur", () => {
      if (!lastData || editingSquare !== element.dataset.square) return;
      editingSquare = null;
      renderResult(lastData);
    });
  });
  const activeEditor = document.querySelector(".piece-select[data-editing='true']");
  if (activeEditor) activeEditor.focus();
}
function bindHandClicks() {
  document.querySelectorAll("[data-hand]").forEach(element => {
    element.addEventListener("click", event => {
      if (event.target.closest(".hand-count-input")) return;
      selectedHandId = element.dataset.hand;
      selectedHandKey = element.dataset.handKey || null;
      selectedSquare = null;
      editingSquare = null;
      if (lastData) {
        renderPreview(lastData);
        renderResult(lastData);
      }
    });
  });
  document.querySelectorAll("[data-hand-key]").forEach(element => {
    if (element.dataset.hand) return;
    element.addEventListener("click", event => {
      if (event.target.closest(".hand-count-input")) return;
      selectedHandId = null;
      selectedHandKey = element.dataset.handKey;
      selectedSquare = null;
      editingSquare = null;
      if (lastData) {
        renderPreview(lastData);
        renderResult(lastData);
      }
    });
  });
}
function bindHandEditors() {
  document.querySelectorAll(".hand-count-input").forEach(element => {
    element.addEventListener("focus", () => {
      selectedHandId = null;
      selectedHandKey = handKey(element.dataset.owner, element.dataset.piece);
      selectedSquare = null;
      editingSquare = null;
      if (lastData) {
        renderPreview(lastData);
        updateSelectionClasses();
      }
    });
    element.addEventListener("change", () => {
      if (!lastData) return;
      selectedHandId = null;
      selectedHandKey = handKey(element.dataset.owner, element.dataset.piece);
      selectedSquare = null;
      editingSquare = null;
      setHandCount(lastData, element.dataset.owner, element.dataset.piece, element.value);
      renderPreview(lastData);
      renderResult(lastData);
      refreshExport(lastData);
    });
  });
}
function updateSelectionClasses() {
  document.querySelectorAll(".image-cell[data-square], .cell[data-square]").forEach(element => {
    element.classList.toggle("selected", selectedSquare === element.dataset.square);
  });
  document.querySelectorAll("[data-hand]").forEach(element => {
    element.classList.toggle("selected", selectedHandId === element.dataset.hand || Boolean(selectedHandKey && selectedHandKey === element.dataset.handKey));
  });
  document.querySelectorAll("[data-hand-key]").forEach(element => {
    if (element.dataset.hand) return;
    element.classList.toggle("selected", Boolean(selectedHandKey && selectedHandKey === element.dataset.handKey));
  });
}
function renderBoard(board) {
  const cells = [];
  for (let row = 1; row <= 9; row++) {
    for (let col = 1; col <= 9; col++) {
      const value = board?.[row - 1]?.[col - 1] || "empty";
      const square = squareName(row, col);
      const editing = editingSquare === square;
      const body = editing
        ? `<select class="piece-select" data-square="${square}" data-editing="true" aria-label="${square}の駒">${boardPieceOptions(value)}</select>`
        : `<span class="piece">${esc(pieceText(value))}</span>`;
      cells.push(
        `<div class="cell ${value === "empty" ? "empty" : ""} ${selectedSquare === square ? "selected" : ""} ${editing ? "editing" : ""}" data-square="${square}">` +
        `<span class="sq">${square}</span>${body}</div>`
      );
    }
  }
  return `<div class="board">${cells.join("")}</div>`;
}
function handItems(data) {
  const items = [];
  for (const piece of data.handRecognition?.pieces || []) {
    (piece.rects || [null]).forEach((rect, index) => {
      items.push({...piece, id: `${piece.id}:${index}`, rectIndex: index, rect});
    });
  }
  for (const item of data.handRecognition?.unknown || []) {
    items.push({id: item.id, owner: item.owner, piece: null, count: 0, confidence: null, rect: item.rect, candidates: item.candidates || [], unknown: true});
  }
  return items;
}
function handStats(data) {
  const stats = {};
  for (const color of ["black", "white"]) {
    stats[color] = {};
    for (const piece of HAND_PIECES) stats[color][piece] = {crops: 0, confidence: []};
  }
  for (const item of handItems(data)) {
    if (!stats[item.owner] || !stats[item.owner][item.piece]) continue;
    stats[item.owner][item.piece].crops += 1;
    if (typeof item.confidence === "number") stats[item.owner][item.piece].confidence.push(item.confidence);
  }
  return stats;
}
function renderHandOwner(data, color) {
  const hands = data.hands || {};
  const stats = handStats(data);
  const total = HAND_PIECES.reduce((sum, piece) => sum + Number((hands[color] || {})[piece] || 0), 0);
  const tiles = HAND_PIECES.map(piece => {
    const count = Number((hands[color] || {})[piece] || 0);
    const item = stats[color][piece];
    const key = handKey(color, piece);
    const avg = item.confidence.length ? item.confidence.reduce((a, b) => a + b, 0) / item.confidence.length : null;
    const classes = ["hand-piece-tile"];
    if (!count) classes.push("zero");
    if (item.crops) classes.push("detected");
    if (selectedHandKey === key) classes.push("selected");
    const meta = item.crops ? `crop ${item.crops}${avg == null ? "" : ` / ${avg.toFixed(2)}`}` : "";
    return `
      <div class="${classes.join(" ")}" data-hand-key="${esc(key)}" role="button" tabindex="0" title="${esc(COLOR_TEXT[color])} ${esc(PIECE_TEXT[piece])}">
        <div class="hand-piece-name">${esc(PIECE_TEXT[piece])}</div>
        <input class="hand-count-input" type="number" min="0" max="18" step="1" value="${count}" data-owner="${esc(color)}" data-piece="${esc(piece)}" aria-label="${esc(COLOR_TEXT[color])} ${esc(PIECE_TEXT[piece])}の枚数">
        <div class="hand-piece-meta">${esc(meta)}</div>
      </div>`;
  }).join("");
  return `
    <div class="hand-summary">
      <section class="hand-owner-card">
        <div class="hand-owner-head"><span>${esc(COLOR_TEXT[color])}の持ち駒</span><span class="hand-owner-total">合計 ${total}</span></div>
        <div class="hand-piece-grid">${tiles}</div>
      </section>
    </div>`;
}
function renderResult(data) {
  const badges = [
    `<span class="badge ${data.needsReview ? "bad" : "ok"}">${data.needsReview ? "要確認" : "OK"}</span>`,
    `<span class="badge">${esc(data.elapsedSeconds)}秒</span>`
  ];
  if (data.edited) badges.push(`<span class="badge">編集済み</span>`);
  result.innerHTML = `
    <div class="status">${badges.join("")}</div>
    ${renderHandOwner(data, "white")}
    ${renderBoard(data.board)}
    ${renderHandOwner(data, "black")}`;
  bindCompareClicks();
  bindBoardEditors();
  bindHandClicks();
  bindHandEditors();
}
function renderExport(data) {
  currentExport = data.export || {};
  if (currentExport.error) {
    exportBox.innerHTML = `<div class="error">${esc(currentExport.error)}</div>`;
    return;
  }
  exportBox.innerHTML = `
    <div class="actions">
      <button class="small" id="copyKif">KIFコピー</button>
      <button class="small" id="downloadKif">KIF保存</button>
      <button class="small" id="copySfen">SFENコピー</button>
    </div>
    <div class="copy-status" id="copyStatus" aria-live="polite"></div>
    <textarea id="kif" spellcheck="false">${esc(currentExport.kif || "")}</textarea>
    <div class="sfen" id="sfen">${esc(currentExport.sfen || "")}</div>`;
  document.getElementById("copyKif").addEventListener("click", event => copyText(currentExport.kif || "", event.currentTarget, "KIFをコピーしました"));
  document.getElementById("copySfen").addEventListener("click", event => copyText(currentExport.sfen || "", event.currentTarget, "SFENをコピーしました"));
  document.getElementById("downloadKif").addEventListener("click", () => {
    const blob = new Blob([currentExport.kif || ""], {type:"text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedFile ? selectedFile.name.replace(/\.[^.]+$/, "") : "position"}.kif`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
}
async function copyText(text, button, message) {
  const status = document.getElementById("copyStatus");
  const originalText = button ? button.textContent : "";
  try {
    await navigator.clipboard.writeText(text);
    showCopyFeedback(button, status, message, originalText);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text || "";
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.focus();
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    showCopyFeedback(button, status, copied ? message : "コピーに失敗しました", originalText, !copied);
  }
}
function showCopyFeedback(button, status, message, originalText, failed = false) {
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
}
</script>
</body>
</html>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Start a local browser UI for image recognition and KIF export.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUTS_DIR / "kif_ui")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()
    serve_kif_ui(
        KifUiConfig(
            host=args.host,
            port=args.port,
            out_dir=args.out,
            model_path=args.model,
            screenshots_dir=args.screenshots_dir,
            labels_dir=args.labels,
            calibration_dir=args.calibration_dir,
            include_hands=not args.no_hands,
            train_if_missing=not args.no_train,
        )
    )


if __name__ == "__main__":
    main()
