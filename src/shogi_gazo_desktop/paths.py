from __future__ import annotations

import sys
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "tools").exists() else Path.cwd()
TOOLS_DIR = ROOT / "tools"
PACKAGED_TOOLS_DIR = Path(__file__).resolve().parent / "_tools"
DATA_DIR = ROOT / "data" / "samples"
DEFAULT_SCREENSHOTS_DIR = DATA_DIR / "screenshots_by_app_piece_style"
DEFAULT_LABELS_DIR = DATA_DIR / "labels" / "boards_by_app_piece_style"
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUTS_DIR / "models" / "piece_model.pkl"
DEFAULT_BOARD_IMAGE = ROOT / "assets" / "legacy_drawables" / "shogi_board.png"
DEFAULT_PIECES_IMAGE = ROOT / "assets" / "legacy_drawables" / "shogi_pieces.png"


def ensure_tools_on_path() -> None:
    tools_path = TOOLS_DIR if TOOLS_DIR.exists() else PACKAGED_TOOLS_DIR
    tools = str(tools_path)
    if tools not in sys.path:
        sys.path.insert(0, tools)


def relative_group(image_path: Path, screenshots_dir: Path) -> tuple[str | None, str | None]:
    try:
        parts = image_path.resolve().relative_to(screenshots_dir.resolve()).parts
    except ValueError:
        return None, None
    app = parts[0] if len(parts) >= 1 else None
    style = parts[1] if len(parts) >= 2 else None
    return app, style
