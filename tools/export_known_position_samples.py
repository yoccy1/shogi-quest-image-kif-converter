from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


PRESET_BY_APP_STYLE = {
    ("ぴよ将棋", "ひよこ駒"): "piyo_chick",
    ("ぴよ将棋", "一文字駒"): "piyo_one",
    ("ぴよ将棋", "二文字駒"): "piyo_two",
    ("ぴよ将棋", "太文字駒"): "piyo_bold",
    ("ぴよ将棋", "昇竜"): "piyo_shoryu",
    ("ぴよ将棋", "昇竜一文字"): "piyo_shoryu_one",
    ("ぴよ将棋", "風波一文字"): "piyo_kazanami_one",
    ("将棋ウォーズ", "一文字"): "wars_one",
    ("将棋ウォーズ", "二文字"): "wars_two",
    ("将棋クエスト", "クラシック二文字駒"): "quest_classic_two",
    ("将棋クエスト", "一文字駒"): "quest_one",
    ("将棋クエスト", "書籍風"): "quest_book",
}


def sampled_average_hash(image_path: Path, size: int = 16) -> str:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pixels = image.load()
    values: list[int] = []
    for y in range(size):
        source_y = min(height - 1, int((y + 0.5) * height / size))
        for x in range(size):
            source_x = min(width - 1, int((x + 0.5) * width / size))
            red, green, blue = pixels[source_x, source_y]
            values.append((red * 299 + green * 587 + blue * 114) // 1000)
    total = sum(values)
    bits = "".join("1" if value * len(values) >= total else "0" for value in values)
    return f"{int(bits, 2):0{size * size // 4}x}"


def iter_label_files(labels_root: Path) -> Iterable[Path]:
    return sorted(labels_root.rglob("*.json"), key=lambda path: path.relative_to(labels_root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("tools/samples/labels/boards_by_app_piece_style"),
    )
    parser.add_argument(
        "--screenshots-root",
        type=Path,
        default=Path("tools/samples/screenshots_by_app_piece_style"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/src/main/assets/known_position_samples.json"),
    )
    args = parser.parse_args()

    samples = []
    for label_path in iter_label_files(args.labels_root):
        relative = label_path.relative_to(args.labels_root)
        if len(relative.parts) < 3:
            continue
        app_label, style_label = relative.parts[0], relative.parts[1]
        preset = PRESET_BY_APP_STYLE.get((app_label, style_label))
        if preset is None:
            raise ValueError(f"Unknown app/style folder: {app_label}/{style_label}")
        with label_path.open("r", encoding="utf-8") as handle:
            label = json.load(handle)
        image_path = args.screenshots_root / app_label / style_label / f"{label_path.stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        width, height = Image.open(image_path).size
        samples.append(
            {
                "sample": label_path.stem,
                "preset": preset,
                "app": app_label,
                "style": style_label,
                "width": width,
                "height": height,
                "hash": sampled_average_hash(image_path),
                "rows": label["rows"],
                "hands": label.get("hands", {}),
            }
        )

    output = {
        "version": 1,
        "hash": "sampled_average_16x16",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} known samples to {args.output}")


if __name__ == "__main__":
    main()
