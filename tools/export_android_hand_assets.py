from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HAND_PIECES = {"HI", "KA", "KI", "GI", "KE", "KY", "FU"}
DIGIT_WIDTH = 32
DIGIT_HEIGHT = 48


def mask_to_hex(mask: list[bool]) -> str:
    chunks: list[str] = []
    for index in range(0, len(mask), 4):
        value = 0
        for bit in range(4):
            value <<= 1
            if index + bit < len(mask) and mask[index + bit]:
                value |= 1
        chunks.append(format(value, "x"))
    return "".join(chunks)


def font_paths() -> list[Path]:
    windows = Path("C:/Windows/Fonts")
    return [
        windows / "arial.ttf",
        windows / "arialbd.ttf",
        windows / "segoeui.ttf",
        windows / "segoeuib.ttf",
        windows / "meiryo.ttc",
        windows / "meiryob.ttc",
        windows / "msgothic.ttc",
    ]


def render_digit_masks() -> list[dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return seven_segment_digit_masks()

    templates: list[dict[str, Any]] = []
    fonts = [path for path in font_paths() if path.exists()]
    if not fonts:
        return seven_segment_digit_masks()

    for digit in range(2, 19):
        text = str(digit)
        for font_path in fonts:
            for size in (24, 28, 32, 36, 40):
                try:
                    font = ImageFont.truetype(str(font_path), size=size)
                except Exception:
                    continue
                image = Image.new("L", (DIGIT_WIDTH, DIGIT_HEIGHT), 255)
                draw = ImageDraw.Draw(image)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                if text_width <= 0 or text_height <= 0:
                    continue
                x = (DIGIT_WIDTH - text_width) // 2 - bbox[0]
                y = (DIGIT_HEIGHT - text_height) // 2 - bbox[1]
                draw.text((x, y), text, font=font, fill=0)
                pixels = list(image.getdata())
                mask = [pixel < 160 for pixel in pixels]
                if sum(mask) < 8:
                    continue
                templates.append(
                    {
                        "digit": digit,
                        "source": f"font:{font_path.name}:{size}",
                        "width": DIGIT_WIDTH,
                        "height": DIGIT_HEIGHT,
                        "mask": mask_to_hex(mask),
                    }
                )
    return templates or seven_segment_digit_masks()


def seven_segment_digit_masks() -> list[dict[str, Any]]:
    segment_map = {
        "0": "abcfed",
        "1": "bc",
        "2": "abged",
        "3": "abgcd",
        "4": "fgbc",
        "5": "afgcd",
        "6": "afgecd",
        "7": "abc",
        "8": "abcdefg",
        "9": "abfgcd",
    }
    templates: list[dict[str, Any]] = []
    for digit in range(2, 19):
        mask = [False] * (DIGIT_WIDTH * DIGIT_HEIGHT)
        draw_seven_segment(mask, str(digit), segment_map)
        templates.append(
            {
                "digit": digit,
                "source": "seven_segment_fallback",
                "width": DIGIT_WIDTH,
                "height": DIGIT_HEIGHT,
                "mask": mask_to_hex(mask),
            }
        )
    return templates


def draw_seven_segment(mask: list[bool], text: str, segment_map: dict[str, str]) -> None:
    char_width = DIGIT_WIDTH / max(1, len(text))
    for index, char in enumerate(text):
        segments = segment_map.get(char, "")
        left = int(index * char_width + 2)
        right = int((index + 1) * char_width - 3)
        top = 5
        middle = DIGIT_HEIGHT // 2
        bottom = DIGIT_HEIGHT - 6
        xmid = (left + right) // 2
        if "a" in segments:
            fill_rect(mask, left, top, right, top + 3)
        if "b" in segments:
            fill_rect(mask, right - 3, top, right, middle)
        if "c" in segments:
            fill_rect(mask, right - 3, middle, right, bottom)
        if "d" in segments:
            fill_rect(mask, left, bottom - 3, right, bottom)
        if "e" in segments:
            fill_rect(mask, left, middle, left + 3, bottom)
        if "f" in segments:
            fill_rect(mask, left, top, left + 3, middle)
        if "g" in segments:
            fill_rect(mask, left, middle - 2, right, middle + 2)
        if not segments:
            fill_rect(mask, xmid - 1, top, xmid + 1, bottom)


def fill_rect(mask: list[bool], left: int, top: int, right: int, bottom: int) -> None:
    for y in range(max(0, top), min(DIGIT_HEIGHT, bottom)):
        for x in range(max(0, left), min(DIGIT_WIDTH, right)):
            mask[y * DIGIT_WIDTH + x] = True


def select_hand_piece_templates(app_templates: list[dict[str, Any]], limit_per_key: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, str, str], int] = {}
    for template in app_templates:
        piece = template.get("piece")
        if piece not in HAND_PIECES:
            continue
        key = (
            str(template.get("preset") or ""),
            str(template.get("color") or ""),
            str(piece),
        )
        if counts.get(key, 0) >= limit_per_key:
            continue
        selected.append(
            {
                "preset": template.get("preset"),
                "color": template.get("color"),
                "piece": piece,
                "source": template.get("source"),
                "mask": template.get("mask"),
                "darkRatio": template.get("darkRatio"),
                "redShare": template.get("redShare"),
                "bbox": template.get("bbox"),
            }
        )
        counts[key] = counts.get(key, 0) + 1
    return selected


def export_assets(
    app_piece_templates: Path,
    piece_output: Path,
    digit_output: Path,
    layout_output: Path,
    limit_per_key: int,
) -> None:
    data = json.loads(app_piece_templates.read_text(encoding="utf-8"))
    piece_templates = select_hand_piece_templates(data.get("templates") or [], limit_per_key)
    piece_output.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(app_piece_templates),
                "limit_per_preset_color_piece": limit_per_key,
                "templates": piece_templates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    digit_output.write_text(
        json.dumps({"version": 1, "digit_templates": render_digit_masks()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    layout_output.write_text(
        json.dumps(
            {
                "version": 1,
                "layouts": [
                    {
                        "id": "wars_left_right_default",
                        "target_app": "SHOGI_WARS",
                        "areas": [
                            {"color": "WHITE", "side": "left", "rect": [0.0, 0.0, 0.17, 1.0], "confidence": 0.55},
                            {"color": "BLACK", "side": "right", "rect": [0.83, 0.0, 0.17, 1.0], "confidence": 0.55},
                        ],
                    },
                    {
                        "id": "quest_piyo_top_bottom_default",
                        "target_app": "SHOGI_QUEST",
                        "areas": [
                            {"color": "WHITE", "side": "top", "rect": [0.0, 0.0, 1.0, 0.13], "confidence": 0.50},
                            {"color": "BLACK", "side": "bottom", "rect": [0.0, 0.87, 1.0, 0.13], "confidence": 0.50},
                        ],
                    },
                    {
                        "id": "piyo_top_bottom_default",
                        "target_app": "PIYO_SHOGI",
                        "areas": [
                            {"color": "WHITE", "side": "top", "rect": [0.0, 0.0, 1.0, 0.13], "confidence": 0.50},
                            {"color": "BLACK", "side": "bottom", "rect": [0.0, 0.87, 1.0, 0.13], "confidence": 0.50},
                        ],
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(piece_templates)} hand piece templates to {piece_output}")
    print(f"wrote hand digit templates to {digit_output}")
    print(f"wrote hand layouts to {layout_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Android hand recognition assets.")
    parser.add_argument("--app-piece-templates", type=Path, default=Path("app/src/main/assets/app_piece_templates.json"))
    parser.add_argument("--piece-output", type=Path, default=Path("app/src/main/assets/hand_piece_templates.json"))
    parser.add_argument("--digit-output", type=Path, default=Path("app/src/main/assets/hand_digit_templates.json"))
    parser.add_argument("--layout-output", type=Path, default=Path("app/src/main/assets/hand_layouts.json"))
    parser.add_argument("--limit-per-key", type=int, default=6)
    args = parser.parse_args()
    export_assets(
        args.app_piece_templates,
        args.piece_output,
        args.digit_output,
        args.layout_output,
        max(1, args.limit_per_key),
    )


if __name__ == "__main__":
    main()
