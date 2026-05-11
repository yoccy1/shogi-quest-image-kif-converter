from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PRESET_BY_SOURCE_FAMILY = {
    "将棋ウォーズ:一文字": "wars_one",
    "将棋ウォーズ:二文字": "wars_two",
    "将棋クエスト:一文字駒": "quest_one",
    "将棋クエスト:書籍風": "quest_book",
    "将棋クエスト:クラシック二文字駒": "quest_classic_two",
    "ぴよ将棋:一文字駒": "piyo_one",
    "ぴよ将棋:二文字駒": "piyo_two",
    "ぴよ将棋:太文字駒": "piyo_bold",
    "ぴよ将棋:ひよこ駒": "piyo_chick",
    "ぴよ将棋:昇竜": "piyo_shoryu",
    "ぴよ将棋:昇竜一文字": "piyo_shoryu_one",
    "ぴよ将棋:風波一文字": "piyo_kazanami_one",
}

COLOR_NAME = {
    "black": "BLACK",
    "white": "WHITE",
}
NORMALIZED_WIDTH = 48
NORMALIZED_HEIGHT = 56


CURATED_PIECE_TEMPLATE_BLOCKLIST = {
    (
        "piyo_chick",
        "BLACK",
        "KA",
        "ぴよ将棋_ひよこ駒_通常_05",
        8,
        2,
    ),
    (
        "piyo_chick",
        "BLACK",
        "GI",
        "ぴよ将棋_ひよこ駒_通常_02",
        8,
        4,
    ),
}

CURATED_PIECE_TEMPLATE_ALLOWLIST = {
    (
        "piyo_chick",
        "BLACK",
        "GI",
        "ぴよ将棋_ひよこ駒_通常_02",
        6,
        6,
    ),
    (
        "piyo_chick",
        "BLACK",
        "GI",
        "ぴよ将棋_ひよこ駒_通常_03",
        8,
        4,
    ),
    (
        "piyo_chick",
        "BLACK",
        "GI",
        "ぴよ将棋_ひよこ駒_通常_09",
        8,
        2,
    ),
    (
        "piyo_chick",
        "WHITE",
        "GI",
        "ぴよ将棋_ひよこ駒_通常_03",
        6,
        5,
    ),
}


def template_selection_key(
    preset: str,
    color: str,
    piece: str,
    source: str,
    row: int | None,
    col: int | None,
) -> tuple[str, str, str, str, int | None, int | None]:
    return preset, color, piece, source, row, col


def mask_to_hex(mask: bytes) -> str:
    bits = [1 if value else 0 for value in mask]
    padding = (-len(bits)) % 4
    bits.extend([0] * padding)
    chars: list[str] = []
    for index in range(0, len(bits), 4):
        value = 0
        for bit in bits[index : index + 4]:
            value = (value << 1) | bit
        chars.append(format(value, "x"))
    return "".join(chars)


def red_mask_stats(mask: bytes, width: int = NORMALIZED_WIDTH, height: int = NORMALIZED_HEIGHT) -> dict[str, float | str] | None:
    if len(mask) != width * height or not any(mask):
        return None
    red_pixels = 0
    central_pixels = 0
    edge_pixels = 0
    sum_x = 0.0
    sum_y = 0.0
    for index, value in enumerate(mask):
        if not value:
            continue
        x = index % width
        y = index // width
        normalized_x = (x + 0.5) / width
        normalized_y = (y + 0.5) / height
        red_pixels += 1
        sum_x += normalized_x
        sum_y += normalized_y
        if 0.28 <= normalized_x <= 0.72 and 0.18 <= normalized_y <= 0.84:
            central_pixels += 1
        if normalized_x < 0.18 or normalized_x > 0.82 or normalized_y < 0.12 or normalized_y > 0.88:
            edge_pixels += 1
    if red_pixels <= 0:
        return None
    return {
        "redMask": mask_to_hex(mask),
        "redCenterX": round(sum_x / red_pixels, 5),
        "redCenterY": round(sum_y / red_pixels, 5),
        "centralRedShare": round(central_pixels / red_pixels, 5),
        "edgeRedShare": round(edge_pixels / red_pixels, 5),
    }


def export_templates(model_path: Path, output_path: Path, limit_per_label: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learned_piece_recognizer import load_model, source_family

    model = load_model(model_path)
    selected = []
    selected_keys: set[tuple[str, str, str, str, int | None, int | None]] = set()
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    def add_template_item(template) -> None:
        family = source_family(template.source)
        preset = PRESET_BY_SOURCE_FAMILY.get(family)
        if preset is None:
            return
        color = COLOR_NAME.get(template.color)
        if color is None:
            return
        selection_key = template_selection_key(
            preset,
            color,
            template.piece,
            template.source,
            template.row,
            template.col,
        )
        if selection_key in selected_keys or selection_key in CURATED_PIECE_TEMPLATE_BLOCKLIST:
            return
        key = (preset, color, template.piece)
        if counts[key] >= limit_per_label:
            return
        counts[key] += 1
        selected_keys.add(selection_key)
        item = {
            "preset": preset,
            "color": color,
            "piece": template.piece,
            "source": template.source,
            "row": template.row,
            "col": template.col,
            "mask": mask_to_hex(template.mask),
            "darkRatio": round(float(template.dark_ratio), 5),
            "redShare": round(float(template.red_share), 5),
            "bbox": template.bbox,
        }
        stats = red_mask_stats(getattr(template, "red_mask", b""))
        if stats is not None:
            item.update(stats)
        selected.append(item)

    raw_templates = list(model.get("templates", []))
    for template in raw_templates:
        family = source_family(template.source)
        preset = PRESET_BY_SOURCE_FAMILY.get(family)
        color = COLOR_NAME.get(template.color)
        if preset is None or color is None:
            continue
        selection_key = template_selection_key(
            preset,
            color,
            template.piece,
            template.source,
            template.row,
            template.col,
        )
        if selection_key in CURATED_PIECE_TEMPLATE_ALLOWLIST:
            add_template_item(template)
    for template in raw_templates:
        add_template_item(template)

    position_priors = []
    prior_key_pattern = re.compile(r"^(.*):r([1-9]):c([1-9])$")
    for key, counts in sorted((model.get("position_priors") or {}).items()):
        match = prior_key_pattern.match(str(key))
        if match is None:
            continue
        family, row, col = match.groups()
        for label, count in sorted((counts or {}).items()):
            count_int = int(count)
            if count_int <= 0:
                continue
            position_priors.append(
                {
                    "family": family,
                    "row": int(row),
                    "col": int(col),
                    "label": str(label),
                    "count": count_int,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": str(model_path),
                "limit_per_preset_color_piece": limit_per_label,
                "templates": selected,
                "position_priors": position_priors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(selected)} templates and {len(position_priors)} position priors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact Android piece templates from a learned model pickle.")
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/src/main/assets/app_piece_templates.json"),
    )
    parser.add_argument("--limit-per-label", type=int, default=6)
    args = parser.parse_args()
    export_templates(args.model, args.output, args.limit_per_label)


if __name__ == "__main__":
    main()
