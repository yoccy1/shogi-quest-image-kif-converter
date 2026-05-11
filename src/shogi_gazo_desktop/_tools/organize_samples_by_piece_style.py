from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from recognize_board_pieces import initial_position_labels


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
HAND_PIECES = ("HI", "KA", "KI", "GI", "KE", "KY", "FU")


STYLE_ORDER = {
    "将棋ウォーズ": ("一文字", "二文字"),
    "将棋クエスト": ("一文字駒", "クラシック二文字駒", "書籍風"),
    "ぴよ将棋": ("ひよこ駒", "二文字駒", "一文字駒", "太文字駒", "昇竜", "昇竜一文字", "風波一文字"),
}


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def empty_hands() -> dict[str, dict[str, int]]:
    return {color: {piece: 0 for piece in HAND_PIECES} for color in ("black", "white")}


def initial_rows() -> list[list[str]]:
    labels = initial_position_labels()
    rows: list[list[str]] = []
    for row in range(1, 10):
        values = []
        for col in range(1, 10):
            color_piece = labels.get((row, col))
            values.append(f"{color_piece[0]}:{color_piece[1]}" if color_piece else "empty")
        rows.append(values)
    return rows


def infer_kind(stem: str) -> str:
    return "初期配置" if "初期配置" in stem else "通常"


def parse_sequence(stem: str) -> str:
    tail = stem.rsplit("_", 1)[-1]
    return tail if tail.isdigit() else "01"


def style_for(app: str, glyph: str, stem: str) -> tuple[str, str]:
    """Return (style, confidence).

    The mapping is intentionally conservative. Shogi Wars stays unchanged as
    requested. Piyo and Quest get a first-pass deterministic split so the
    pipeline can evaluate by style immediately; rows marked "推定" should be
    reviewed visually if the exact app-side theme name matters.
    """
    if app == "将棋ウォーズ":
        return glyph, "確定"

    if app == "将棋クエスト":
        if stem == "将棋クエスト_一文字_初期配置_02":
            return "書籍風", "推定"
        if glyph == "二文字":
            return "クラシック二文字駒", "推定"
        return "一文字駒", "推定"

    if app == "ぴよ将棋":
        number = int(parse_sequence(stem))
        if glyph == "二文字":
            if "初期配置" in stem and number == 2:
                return "太文字駒", "推定"
            if "初期配置" in stem and number == 3:
                return "昇竜", "推定"
            return "二文字駒", "推定"
        if "初期配置" in stem:
            return {
                1: "一文字駒",
                2: "ひよこ駒",
                3: "昇竜一文字",
                4: "風波一文字",
            }.get(number, "一文字駒"), "推定"
        if 1 <= number <= 15:
            return "ひよこ駒", "推定"
        return "一文字駒", "推定"

    return glyph or "未分類", "推定"


def label_lookup(labels_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(labels_dir.rglob("*.json"))}


def image_reference(label_path: Path, image_path: Path) -> str:
    return os.path.relpath(image_path, label_path.parent).replace(os.sep, "/")


def update_label_image(source_label_path: Path, destination_label_path: Path, image_path: Path) -> dict[str, Any]:
    data = json.loads(source_label_path.read_text(encoding="utf-8"))
    data["schema_version"] = max(int(data.get("schema_version", 1)), 2)
    data["image"] = image_reference(destination_label_path, image_path)
    if "hands" not in data:
        data["hands"] = empty_hands()
    metadata = data.setdefault("metadata", {})
    metadata["piece_style_sorted_from"] = str(source_label_path)
    return data


def implicit_initial_label(destination_label_path: Path, image_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "image": image_reference(destination_label_path, image_path),
        "orientation": "black_bottom",
        "rows": initial_rows(),
        "hands": empty_hands(),
        "label_source": "implicit_initial_position",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "app",
        "piece_style",
        "previous_glyph",
        "kind",
        "sequence",
        "classification_status",
        "source_image",
        "style_image",
        "source_label",
        "style_label",
        "label_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def remove_tree_inside(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise SystemExit(f"Refusing to remove path outside output root: {path}")
    if path.exists():
        shutil.rmtree(path)


def copy_samples(
    screenshots_dir: Path,
    labels_dir: Path,
    out_screenshots_dir: Path,
    out_labels_dir: Path,
    manifest_path: Path,
    *,
    clean: bool,
    overwrite_labels: bool,
    allow_label_clean: bool,
) -> None:
    if clean:
        remove_tree_inside(out_screenshots_dir, out_screenshots_dir.parent)
        if allow_label_clean:
            remove_tree_inside(out_labels_dir, out_labels_dir.parent)
    out_screenshots_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for app, styles in STYLE_ORDER.items():
        for style in styles:
            (out_screenshots_dir / app / style).mkdir(parents=True, exist_ok=True)
            (out_labels_dir / app / style).mkdir(parents=True, exist_ok=True)

    labels = label_lookup(labels_dir)
    counters: dict[tuple[str, str, str], int] = {}
    manifest_rows: list[dict[str, str]] = []
    for image_path in iter_images(screenshots_dir):
        rel = image_path.relative_to(screenshots_dir)
        if len(rel.parts) < 3:
            continue
        app, glyph = rel.parts[0], rel.parts[1]
        style, status = style_for(app, glyph, image_path.stem)
        kind = infer_kind(image_path.stem)
        key = (app, style, kind)
        counters[key] = counters.get(key, 0) + 1
        sequence = f"{counters[key]:02d}"
        new_stem = f"{app}_{style}_{kind}_{sequence}"
        destination_image = out_screenshots_dir / app / style / f"{new_stem}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination_image)

        source_label = labels.get(image_path.stem)
        destination_label = out_labels_dir / app / style / f"{new_stem}.json"
        label_status = "教師ラベルなし"
        if destination_label.exists() and not overwrite_labels:
            label_status = "既存ラベル保持"
        elif source_label is not None:
            data = update_label_image(source_label, destination_label, destination_image)
            metadata = data.setdefault("metadata", {})
            metadata["piece_style"] = style
            metadata["previous_sample"] = image_path.stem
            destination_label.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            label_status = "教師ラベルあり"
        elif kind == "初期配置":
            data = implicit_initial_label(destination_label, destination_image)
            metadata = data.setdefault("metadata", {})
            metadata["piece_style"] = style
            metadata["previous_sample"] = image_path.stem
            destination_label.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            label_status = "暗黙の初期配置"

        manifest_rows.append(
            {
                "app": app,
                "piece_style": style,
                "previous_glyph": glyph,
                "kind": kind,
                "sequence": sequence,
                "classification_status": status,
                "source_image": str(image_path),
                "style_image": str(destination_image),
                "source_label": str(source_label or ""),
                "style_label": str(destination_label if destination_label.exists() else ""),
                "label_status": label_status,
            }
        )

    write_manifest(manifest_path, manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a more detailed app/piece-style sample folder without changing the original folders.")
    parser.add_argument("--screenshots-dir", type=Path, default=Path("tools/samples/screenshots_by_app_glyph"))
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards_by_app_glyph"))
    parser.add_argument("--out-screenshots-dir", type=Path, default=Path("tools/samples/screenshots_by_app_piece_style"))
    parser.add_argument("--out-labels-dir", type=Path, default=Path("tools/samples/labels/boards_by_app_piece_style"))
    parser.add_argument("--manifest", type=Path, default=Path("tools/samples/piece_style_manifest.csv"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite-labels", action="store_true", help="Overwrite existing labels in --out-labels-dir.")
    parser.add_argument(
        "--allow-label-clean",
        action="store_true",
        help="Allow --clean to remove --out-labels-dir. Existing labels are preserved unless this is set.",
    )
    args = parser.parse_args()

    copy_samples(
        args.screenshots_dir,
        args.labels_dir,
        args.out_screenshots_dir,
        args.out_labels_dir,
        args.manifest,
        clean=args.clean,
        overwrite_labels=args.overwrite_labels,
        allow_label_clean=args.allow_label_clean,
    )
    print(f"OK: wrote {args.manifest}")
    print(f"OK: screenshots -> {args.out_screenshots_dir}")
    print(f"OK: labels -> {args.out_labels_dir}")


if __name__ == "__main__":
    main()
