from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from chamfer_offline_probe import (  # noqa: E402
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
    chamfer_metrics,
    decode_hex_mask,
    run_probe,
)


def mask_to_hex(mask: list[bool]) -> str:
    bits = [1 if value else 0 for value in mask]
    bits.extend([0] * ((-len(bits)) % 4))
    chars: list[str] = []
    for index in range(0, len(bits), 4):
        value = 0
        for bit in bits[index : index + 4]:
            value = (value << 1) | bit
        chars.append(format(value, "x"))
    return "".join(chars)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ChamferOfflineProbeTest(unittest.TestCase):
    def test_decode_hex_mask_uses_high_bit_first_order(self) -> None:
        self.assertEqual([True, False, True, False, False, True, False, True], decode_hex_mask("a5", 8))

    def test_chamfer_metrics_prefer_overlapping_masks(self) -> None:
        source = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        same = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        far = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        for y in range(10, 40):
            source[y * FEATURE_WIDTH + 20] = True
            same[y * FEATURE_WIDTH + 20] = True
            far[y * FEATURE_WIDTH + 35] = True

        same_metrics = chamfer_metrics(source, same)
        far_metrics = chamfer_metrics(source, far)

        self.assertLess(same_metrics["symmetric_chamfer_mean"], far_metrics["symmetric_chamfer_mean"])
        self.assertGreater(same_metrics["chamfer_score"], far_metrics["chamfer_score"])

    def test_run_probe_outputs_virtual_rank_without_touching_base_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            screenshots = root / "screenshots"
            app = "ぴよ将棋"
            style = "ひよこ駒"
            sample = "toy_sample"
            report_path = analysis / app / style / sample / "piece_report.json"
            report_path.parent.mkdir(parents=True)
            (screenshots / app / style).mkdir(parents=True)

            image = Image.new("RGBA", (100, 100), (210, 178, 118, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((48, 12, 52, 88), fill=(32, 30, 28, 255))
            image.save(screenshots / app / style / f"{sample}.png")
            report_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "row": 1,
                                "col": 1,
                                "debug": {
                                    "recognition_rect": {
                                        "left_ratio": 0.0,
                                        "top_ratio": 0.0,
                                        "width_ratio": 1.0,
                                        "height_ratio": 1.0,
                                    }
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            vertical = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
            horizontal = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
            for y in range(8, 48):
                vertical[y * FEATURE_WIDTH + 24] = True
            for x in range(8, 40):
                horizontal[28 * FEATURE_WIDTH + x] = True
            template_asset = root / "templates.json"
            template_asset.write_text(
                json.dumps(
                    {
                        "templates": [
                            {
                                "preset": "piyo_chick",
                                "color": "WHITE",
                                "piece": "KY",
                                "source": "wrong_template",
                                "mask": mask_to_hex(horizontal),
                            },
                            {
                                "preset": "piyo_chick",
                                "color": "BLACK",
                                "piece": "FU",
                                "source": "right_template",
                                "mask": mask_to_hex(vertical),
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            fields = [
                "app",
                "piece_style",
                "sample",
                "square",
                "row",
                "col",
                "expected",
                "predicted_top1",
                "candidate_index",
                "candidate_rank",
                "candidate_identity",
                "is_expected",
                "is_predicted_top1",
                "score",
                "source",
                "target_ink_count",
                "target_bbox_width",
                "target_bbox_height",
                "target_bbox_center_x",
                "target_bbox_center_y",
                "source_clean_ink_count",
            ]
            rows = [
                {
                    "app": app,
                    "piece_style": style,
                    "sample": sample,
                    "square": "1一",
                    "row": 1,
                    "col": 1,
                    "expected": "black:FU",
                    "predicted_top1": "white:KY",
                    "candidate_index": 1,
                    "candidate_rank": 1,
                    "candidate_identity": "white:KY",
                    "is_expected": False,
                    "is_predicted_top1": True,
                    "score": 0.45,
                    "source": "app_template:wrong_template+1tpl",
                    "target_ink_count": 100,
                    "target_bbox_width": 0.7,
                    "target_bbox_height": 0.1,
                    "target_bbox_center_x": 0.5,
                    "target_bbox_center_y": 0.5,
                    "source_clean_ink_count": 0,
                },
                {
                    "app": app,
                    "piece_style": style,
                    "sample": sample,
                    "square": "1一",
                    "row": 1,
                    "col": 1,
                    "expected": "black:FU",
                    "predicted_top1": "white:KY",
                    "candidate_index": 2,
                    "candidate_rank": 2,
                    "candidate_identity": "black:FU",
                    "is_expected": True,
                    "is_predicted_top1": False,
                    "score": 0.44,
                    "source": "app_template:right_template+1tpl",
                    "target_ink_count": 120,
                    "target_bbox_width": 0.1,
                    "target_bbox_height": 0.7,
                    "target_bbox_center_x": 0.5,
                    "target_bbox_center_y": 0.5,
                    "source_clean_ink_count": 0,
                },
            ]
            write_csv(analysis / "piece_style_board_error_candidates.csv", rows, fields)
            write_csv(
                analysis / "piece_style_board_error_candidate_gaps.csv",
                [
                    {
                        "app": app,
                        "piece_style": style,
                        "sample": sample,
                        "square": "1一",
                        "row": 1,
                        "col": 1,
                        "expected": "black:FU",
                        "expected_candidate_rank": 2,
                    }
                ],
                ["app", "piece_style", "sample", "square", "row", "col", "expected", "expected_candidate_rank"],
            )

            result = run_probe(
                analysis_dir=analysis,
                screenshots_dir=screenshots,
                template_asset=template_asset,
                out_dir=analysis / "probe",
                top_n=30,
                virtual_chamfer_weight=0.5,
                coverage_distance=1.5,
            )

            self.assertEqual(2, result["candidate_rows"])
            self.assertEqual(1, result["summary"]["fixed_by_virtual_count"])
            output_rows = read_output_csv(analysis / "probe" / "piece_style_chamfer_candidates.csv")
            expected = next(row for row in output_rows if row["candidate_identity"] == "black:FU")
            self.assertEqual("2", expected["candidate_rank"])
            self.assertEqual("1", expected["virtual_rerank_rank"])


def read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
