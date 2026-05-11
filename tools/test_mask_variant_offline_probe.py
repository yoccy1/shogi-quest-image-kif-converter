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

from mask_variant_offline_probe import (  # noqa: E402
    FEATURE_HEIGHT,
    FEATURE_WIDTH,
    MASK_VARIANTS,
    build_mask_variants,
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


def read_output_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class MaskVariantOfflineProbeTest(unittest.TestCase):
    def test_mask_variants_prune_red_edge_and_both(self) -> None:
        current = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        red = [False] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        for y in range(10, 20):
            current[y * FEATURE_WIDTH + 20] = True
        for y in range(10, 20):
            current[y * FEATURE_WIDTH + 1] = True
        for y in range(10, 20):
            red[y * FEATURE_WIDTH + 20] = True
        features = {"clean_mask": current, "ink_mask": current, "red_mask": red}

        variants = build_mask_variants(features, edge_band_ratio=0.10, red_dilate_iterations=0)

        self.assertEqual(sum(current), sum(variants["current_clean"]))
        self.assertLess(sum(variants["red_pruned"]), sum(variants["current_clean"]))
        self.assertLess(sum(variants["edge_band_pruned"]), sum(variants["current_clean"]))
        self.assertLess(sum(variants["red_edge_pruned"]), sum(variants["red_pruned"]))
        self.assertLess(sum(variants["red_edge_pruned"]), sum(variants["edge_band_pruned"]))

    def test_interior_only_removes_outer_mask(self) -> None:
        current = [True] * (FEATURE_WIDTH * FEATURE_HEIGHT)
        variants = build_mask_variants({"clean_mask": current, "ink_mask": current, "red_mask": [False] * len(current)})

        self.assertLess(sum(variants["interior_only"]), sum(variants["current_clean"]))
        self.assertGreater(sum(variants["interior_only"]), 0)
        self.assertIn("skeleton_like", variants)
        self.assertGreater(sum(variants["skeleton_like"]), 0)

    def test_run_probe_outputs_required_summary_and_no_leak(self) -> None:
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

            image = Image.new("RGBA", (100, 100), (220, 180, 120, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((48, 15, 52, 85), fill=(30, 30, 28, 255))
            draw.rectangle((2, 12, 8, 88), fill=(210, 40, 30, 255))
            image.save(screenshots / app / style / f"{sample}.png")
            report_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "excluded_source": "leak_source",
                            "excluded_sources": ["leak_source"],
                            "no_leak_options": {"excludedTemplateSource": "leak_source"},
                        },
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
                        ],
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
                                "source": "wrong_source",
                                "mask": mask_to_hex(horizontal),
                                "darkRatio": 0.08,
                                "redShare": 0.0,
                                "bbox": [8, 28, 40, 29],
                            },
                            {
                                "preset": "piyo_chick",
                                "color": "BLACK",
                                "piece": "FU",
                                "source": "right_source",
                                "mask": mask_to_hex(vertical),
                                "darkRatio": 0.08,
                                "redShare": 0.0,
                                "bbox": [24, 8, 25, 48],
                            },
                            {
                                "preset": "piyo_chick",
                                "color": "BLACK",
                                "piece": "FU",
                                "source": "leak_source",
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
                    "source": "app_template:wrong_source+1tpl",
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
                    "source": "app_template:right_source+1tpl",
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
                virtual_weight=0.5,
                edge_band_ratio=0.12,
                interior_x_ratio=0.22,
                interior_y_ratio=0.18,
                red_dilate_iterations=2,
            )

            self.assertEqual(len(MASK_VARIANTS), len(result["summary"]))
            summary_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_summary.csv")
            required = {
                "variant",
                "expected_in_topn_count",
                "variant_errors_if_top1",
                "fixed_by_variant_count",
                "shape_only_variant_errors_if_top1",
                "fixed_by_shape_only_variant_count",
                "expected_variant_rank_avg",
                "avg_variant_retained_ratio",
                "template_score_rows",
                "shape_only_template_score_rows",
                "leak_template_score_rows",
                "shape_only_leak_template_score_rows",
                "candidate_source_leak_rows",
                "candidate_source_empty_rows",
            }
            self.assertTrue(required.issubset(set(summary_rows[0])))
            self.assertTrue(all(row["expected_in_topn_count"] == "1" for row in summary_rows))
            self.assertTrue(all(row["leak_template_score_rows"] == "0" for row in summary_rows))
            self.assertTrue(all(row["shape_only_leak_template_score_rows"] == "0" for row in summary_rows))
            self.assertTrue(all(row["candidate_source_leak_rows"] == "0" for row in summary_rows))
            self.assertTrue(all(row["candidate_source_empty_rows"] == "0" for row in summary_rows))
            by_variant = {row["variant"]: row for row in summary_rows}
            self.assertLess(float(by_variant["red_pruned"]["avg_variant_red_share"]), 0.01)
            self.assertGreaterEqual(
                int(by_variant["red_pruned"]["shape_only_template_score_rows"]),
                int(by_variant["red_pruned"]["template_score_rows"]),
            )
            for row in summary_rows:
                self.assertEqual(1, int(row["variant_errors_if_top1"]) + int(row["fixed_by_variant_count"]))
                self.assertEqual(1, int(row["shape_only_variant_errors_if_top1"]) + int(row["fixed_by_shape_only_variant_count"]))

            identity_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_identity_scores.csv")
            self.assertIn("best_projection_mean_similarity", identity_rows[0])
            self.assertIn("shape_only_variant_score", identity_rows[0])
            self.assertIn("shape_only_variant_rank", identity_rows[0])
            self.assertIn("candidate_source_leak", identity_rows[0])
            expected_rows = [row for row in identity_rows if row["candidate_identity"] == "black:FU"]
            self.assertTrue(expected_rows)
            self.assertTrue(all(row["candidate_rank"] == "2" for row in expected_rows))
            self.assertTrue(all(row["variant_rank"] for row in expected_rows))
            self.assertTrue(all(row["shape_only_variant_rank"] for row in expected_rows))
            cell_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_cell_summary.csv")
            self.assertEqual(len(MASK_VARIANTS), len(cell_rows))
            self.assertIn("base_top1_is_expected", cell_rows[0])
            self.assertIn("fixed_by_variant", cell_rows[0])
            self.assertIn("fixed_by_base_plus_variant", cell_rows[0])
            self.assertIn("fixed_by_shape_only_variant", cell_rows[0])
            self.assertIn("shape_only_variant_top1_identity", cell_rows[0])
            self.assertIn("expected_rank_change_variant", cell_rows[0])
            self.assertTrue(all(row["variant_top1_identity"] for row in cell_rows))
            rank_change_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_rank_change_summary.csv")
            self.assertTrue(rank_change_rows)
            self.assertTrue({"variant", "score_mode", "rank_change", "cell_rows"}.issubset(set(rank_change_rows[0])))
            template_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_template_scores.csv")
            self.assertIn("projection_mean_similarity", template_rows[0])
            self.assertFalse(any(row["template_source"] == "leak_source" for row in template_rows))
            shape_only_template_rows = read_output_csv(analysis / "probe" / "piece_style_mask_variant_shape_only_template_scores.csv")
            self.assertTrue(shape_only_template_rows)
            self.assertIn("projection_mean_similarity", shape_only_template_rows[0])
            self.assertFalse(any(row["template_source"] == "leak_source" for row in shape_only_template_rows))


if __name__ == "__main__":
    unittest.main()
