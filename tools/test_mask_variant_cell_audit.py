from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mask_variant_cell_audit import (  # noqa: E402
    OUTPUT_SCORE_MODES,
    TargetCell,
    run_audit,
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        fieldnames = fields
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class MaskVariantCellAuditTest(unittest.TestCase):
    def test_run_audit_selects_competitors_deltas_summary_and_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            mask_dir = root / "mask"
            out_dir = root / "out"
            screenshots = root / "screenshots"
            template_asset = root / "templates.json"
            template_asset.write_text(json.dumps({"templates": []}), encoding="utf-8")

            base = {
                "app": "ぴよ将棋",
                "piece_style": "ひよこ駒",
                "sample": "toy_sample",
                "square": "1一",
                "row": "1",
                "col": "1",
                "expected": "black:FU",
                "excluded_sources": "leak_src",
            }
            variants = ("current_clean", "interior_only", "skeleton_like")
            identities = ("black:FU", "white:KY", "black:GI")

            write_csv(
                analysis / "piece_style_board_error_candidates.csv",
                [
                    {
                        **base,
                        "candidate_identity": identity,
                        "candidate_rank": str(index + 1),
                        "candidate_source": "safe_src",
                    }
                    for index, identity in enumerate(("white:KY", "black:FU", "black:GI"))
                ],
            )

            write_csv(
                mask_dir / "piece_style_mask_variant_cell_summary.csv",
                [{**base, "variant": variant} for variant in variants],
            )

            mask_rows = []
            for variant, count, red_share in (
                ("current_clean", 100, 0.40),
                ("interior_only", 60, 0.10),
                ("skeleton_like", 20, 0.00),
            ):
                mask_rows.append(
                    {
                        **base,
                        "variant": variant,
                        "current_mask_count": "100",
                        "variant_mask_count": str(count),
                        "variant_red_share": str(red_share),
                        "variant_bbox_width": "0.5",
                        "variant_bbox_height": "0.6",
                        "variant_bbox_center_x": "0.52",
                        "variant_bbox_center_y": "0.48",
                    }
                )
            write_csv(mask_dir / "piece_style_mask_variant_masks.csv", mask_rows)

            identity_rows = []
            for variant in variants:
                for identity in identities:
                    row = {
                        **base,
                        "variant": variant,
                        "candidate_identity": identity,
                        "candidate_source": "safe_src",
                        "candidate_source_leak": "False",
                    }
                    if identity == "black:FU":
                        row.update(
                            {
                                "candidate_rank": "2",
                                "base_score": "0.50",
                                "variant_rank": "1" if variant == "interior_only" else "2",
                                "variant_score": "0.90" if variant == "interior_only" else "0.40",
                                "base_plus_variant_rank": "2",
                                "base_plus_variant_score": "0.70",
                                "shape_only_variant_rank": "1" if variant == "interior_only" else "3",
                                "shape_only_variant_score": "0.95" if variant == "interior_only" else "0.30",
                                "base_plus_shape_only_variant_rank": "3",
                                "base_plus_shape_only_variant_score": "0.60",
                            }
                        )
                    elif identity == "white:KY":
                        row.update(
                            {
                                "candidate_rank": "1",
                                "base_score": "0.60",
                                "variant_rank": "2" if variant == "interior_only" else "1",
                                "variant_score": "0.80" if variant == "interior_only" else "0.80",
                                "base_plus_variant_rank": "1",
                                "base_plus_variant_score": "0.80",
                                "shape_only_variant_rank": "3" if variant == "interior_only" else "2",
                                "shape_only_variant_score": "0.20" if variant == "interior_only" else "0.50",
                                "base_plus_shape_only_variant_rank": "2",
                                "base_plus_shape_only_variant_score": "0.65",
                            }
                        )
                    else:
                        row.update(
                            {
                                "candidate_rank": "3",
                                "base_score": "0.30",
                                "variant_rank": "3",
                                "variant_score": "0.30",
                                "base_plus_variant_rank": "3",
                                "base_plus_variant_score": "0.40",
                                "shape_only_variant_rank": "2" if variant == "interior_only" else "1",
                                "shape_only_variant_score": "0.70" if variant == "interior_only" else "0.90",
                                "base_plus_shape_only_variant_rank": "1",
                                "base_plus_shape_only_variant_score": "0.85",
                            }
                        )
                    identity_rows.append(row)
            write_csv(mask_dir / "piece_style_mask_variant_identity_scores.csv", identity_rows)

            template_rows = []
            shape_rows = []
            for variant in variants:
                for identity, score, source in (
                    ("black:FU", "0.80", "safe_expected"),
                    ("white:KY", "0.70", "safe_ky"),
                    ("black:GI", "0.60", "safe_gi"),
                ):
                    common = {
                        **base,
                        "variant": variant,
                        "identity": identity,
                        "template_source": source,
                        "template_row": "1",
                        "template_col": "1",
                        "template_final_score": score,
                        "template_position_boost": "0.0",
                        "shape": score,
                        "raw_dice": score,
                        "raw_iou": "0.4",
                        "clean_dice": score,
                        "clean_iou": "0.4",
                        "clean_shape": score,
                        "bbox": "0.5",
                        "center": "0.6",
                        "density": "0.7",
                        "red": "0.1",
                        "template_ink_count": "80",
                        "template_clean_ink_count": "80",
                        "template_red_share": "0.1",
                        "projection_x_similarity": "0.8",
                        "projection_y_similarity": "0.9",
                        "projection_mean_similarity": "0.85",
                    }
                    template_rows.append(common)
                    shape_rows.append(
                        {
                            **common,
                            "template_source": f"shape_{source}",
                            "template_final_score": "0.95" if identity == "black:GI" else score,
                            "projection_mean_similarity": "0.95" if identity == "black:GI" else "0.85",
                        }
                    )
            write_csv(mask_dir / "piece_style_mask_variant_template_scores.csv", template_rows)
            write_csv(mask_dir / "piece_style_mask_variant_shape_only_template_scores.csv", shape_rows)

            outputs = run_audit(
                analysis_dir=analysis,
                mask_variant_dir=mask_dir,
                screenshots_dir=screenshots,
                template_asset=template_asset,
                out_dir=out_dir,
                variants=variants,
                targets=(TargetCell("toy_sample", "1一", "black:FU"),),
            )

            identity_output = read_csv(outputs["identity_rows"])
            self.assertEqual(1 * len(variants) * len(OUTPUT_SCORE_MODES) * 2, len(identity_output))

            interior_variant_competitor = next(
                row
                for row in identity_output
                if row["variant"] == "interior_only"
                and row["score_mode"] == "variant"
                and row["role"] == "competitor"
            )
            self.assertEqual("white:KY", interior_variant_competitor["identity"])
            self.assertEqual("2", interior_variant_competitor["rank"])

            interior_shape_competitor = next(
                row
                for row in identity_output
                if row["variant"] == "interior_only"
                and row["score_mode"] == "shape_only_variant"
                and row["role"] == "competitor"
            )
            self.assertEqual("black:GI", interior_shape_competitor["identity"])
            self.assertEqual("2", interior_shape_competitor["rank"])
            self.assertEqual("shape_safe_gi", interior_shape_competitor["best_template_source"])

            pair_rows = read_csv(outputs["pair_deltas"])
            interior_variant_delta = next(
                row
                for row in pair_rows
                if row["variant"] == "interior_only" and row["score_mode"] == "variant"
            )
            self.assertEqual("-1.0", interior_variant_delta["rank_delta_expected_minus_competitor"])
            self.assertEqual("1.0", interior_variant_delta["rank_advantage_expected"])

            summary_rows = read_csv(outputs["cell_summary"])
            self.assertEqual(1, len(summary_rows))
            self.assertIn("variant_top1_fix", summary_rows[0]["interior_only_classification"])
            self.assertIn("base_plus_lost", summary_rows[0]["interior_only_classification"])
            self.assertIn("shape_only_preserved", summary_rows[0]["interior_only_classification"])

            leak_rows = read_csv(outputs["no_leak_audit"])
            self.assertEqual(3, len(leak_rows))
            for row in leak_rows:
                self.assertEqual("0", row["candidate_source_leak_count"])
                self.assertEqual("0", row["candidate_source_empty_count"])
                self.assertEqual("0", row["template_source_leak_count"])
                self.assertEqual("0", row["shape_only_template_source_leak_count"])

            self.assertTrue(outputs["summary_md"].exists())
            self.assertTrue(outputs["summary_html"].exists())


if __name__ == "__main__":
    unittest.main()
