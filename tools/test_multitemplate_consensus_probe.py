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

from multitemplate_consensus_probe import (  # noqa: E402
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
    consensus_metrics,
    load_templates,
    run_probe,
    score_template,
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


def read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class MultiTemplateConsensusProbeTest(unittest.TestCase):
    def test_consensus_metrics_counts_close_support_and_sources(self) -> None:
        rows = [
            {"weighted_template_score": 0.40, "template_source": "a"},
            {"weighted_template_score": 0.392, "template_source": "b"},
            {"weighted_template_score": 0.370, "template_source": "a"},
            {"weighted_template_score": 0.320, "template_source": "c"},
        ]

        metrics = consensus_metrics(rows)

        self.assertEqual(4, metrics["template_count"])
        self.assertEqual(3, metrics["source_count"])
        self.assertEqual(2, metrics["support_count_within_010"])
        self.assertEqual(2, metrics["source_count_ge_036"])
        self.assertGreater(metrics["support_bonus_score"], metrics["best_template_score"])

    def test_score_template_prefers_matching_mask_shape(self) -> None:
        vertical = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        horizontal = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        for y in range(8, 48):
            vertical[y * FEATURE_WIDTH + 24] = True
        for x in range(8, 40):
            horizontal[28 * FEATURE_WIDTH + x] = True
        source = {
            "ink_mask": vertical,
            "clean_mask": vertical,
            "ink_count": sum(vertical),
            "clean_ink_count": sum(vertical),
            "original_ink_ratio": 0.08,
            "red_share": 0.0,
            "bbox_width_ratio": 0.10,
            "bbox_height_ratio": 0.70,
            "bbox_center_x": 0.5,
            "bbox_center_y": 0.5,
            "red_center_x": 0.5,
            "red_center_y": 0.5,
            "central_red_share": 0.0,
            "edge_red_share": 0.0,
        }
        matching = {
            **source,
            "identity": "black:FU",
            "preset": "piyo_chick",
            "piece": "FU",
            "source": "matching",
            "row": 1,
            "col": 1,
        }
        wrong = {
            **source,
            "identity": "black:FU",
            "preset": "piyo_chick",
            "piece": "FU",
            "source": "wrong",
            "row": 1,
            "col": 1,
            "ink_mask": horizontal,
            "clean_mask": horizontal,
            "ink_count": sum(horizontal),
            "clean_ink_count": sum(horizontal),
        }

        matching_score = score_template(source, matching, 5, 5)
        wrong_score = score_template(source, wrong, 5, 5)

        self.assertIsNotNone(matching_score)
        self.assertIsNotNone(wrong_score)
        assert matching_score is not None
        assert wrong_score is not None
        self.assertGreater(matching_score["weighted_template_score"], wrong_score["weighted_template_score"])

    def test_run_probe_outputs_virtual_ranks_without_touching_base_rank(self) -> None:
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
                                "darkRatio": 0.08,
                                "redShare": 0.0,
                                "bbox": [8, 28, 40, 29],
                            },
                            {
                                "preset": "piyo_chick",
                                "color": "BLACK",
                                "piece": "FU",
                                "source": "right_template_a",
                                "mask": mask_to_hex(vertical),
                                "darkRatio": 0.08,
                                "redShare": 0.0,
                                "bbox": [24, 8, 25, 48],
                            },
                            {
                                "preset": "piyo_chick",
                                "color": "BLACK",
                                "piece": "FU",
                                "source": "right_template_b",
                                "mask": mask_to_hex(vertical),
                                "darkRatio": 0.08,
                                "redShare": 0.0,
                                "bbox": [24, 8, 25, 48],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(3, len(load_templates(template_asset)))

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
                "source_ink_ratio",
                "source_bbox_width",
                "source_bbox_height",
                "source_bbox_center_x",
                "source_bbox_center_y",
                "source_red_share",
                "source_central_red_share",
                "source_edge_red_share",
                "source_red_center_x",
                "source_red_center_y",
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
                    "source_ink_ratio": 0.08,
                    "source_bbox_width": 0.10,
                    "source_bbox_height": 0.70,
                    "source_bbox_center_x": 0.5,
                    "source_bbox_center_y": 0.5,
                    "source_red_share": 0.0,
                    "source_central_red_share": 0.0,
                    "source_edge_red_share": 0.0,
                    "source_red_center_x": 0.5,
                    "source_red_center_y": 0.5,
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
                    "source": "app_template:right_template_a+2tpl",
                    "source_ink_ratio": 0.08,
                    "source_bbox_width": 0.10,
                    "source_bbox_height": 0.70,
                    "source_bbox_center_x": 0.5,
                    "source_bbox_center_y": 0.5,
                    "source_red_share": 0.0,
                    "source_central_red_share": 0.0,
                    "source_edge_red_share": 0.0,
                    "source_red_center_x": 0.5,
                    "source_red_center_y": 0.5,
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
            )

            self.assertEqual(2, result["summary"]["candidate_rows"])
            identity_rows = read_output_csv(analysis / "probe" / "piece_style_multitemplate_identity_consensus.csv")
            expected = next(row for row in identity_rows if row["candidate_identity"] == "black:FU")
            self.assertEqual("2", expected["candidate_rank"])
            self.assertEqual("2", expected["template_count"])
            self.assertEqual("1", expected["support_bonus_rank"])


if __name__ == "__main__":
    unittest.main()
