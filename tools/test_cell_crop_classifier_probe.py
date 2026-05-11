from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cell_crop_classifier_probe import (  # noqa: E402
    CellCropExample,
    class_count_rows,
    class_coverage_rows_for_split,
    color_stats,
    classifier_error_summary_rows,
    confidence_margin_audit_rows,
    confidence_margin_quantile_audit_rows,
    confusion_matrix_rows,
    confusion_matrix_rows_by_split,
    expected_rank,
    ranked_predictions,
    residual_summary_rows,
    residual_summary_rows_by_split,
    source_id,
    strict_split_definitions,
    train_and_predict_strict_splits,
    train_and_predict_loso,
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def example(
    sample: str,
    row: int,
    col: int,
    label: str,
    vector: list[float],
    *,
    app: str = "ぴよ将棋",
    style: str = "ひよこ駒",
) -> CellCropExample:
    color, piece = label.split(":", 1)
    return CellCropExample(
        app=app,
        piece_style=style,
        sample=sample,
        source_id=source_id(app, style, sample),
        row=row,
        col=col,
        square=f"{col}{row}",
        label=label,
        color=color,
        piece=piece,
        label_path=Path(f"{sample}.json"),
        image_path=Path(f"{sample}.png"),
        crop_path="",
        gray_crop_path="",
        grid_method="toy",
        grid_confidence=1.0,
        feature=np.asarray(vector, dtype="float32"),
        red_share=0.0,
        black_ink_ratio=0.0,
        red_ink_ratio=0.0,
        edge_red_share=0.0,
        central_red_share=0.0,
        gray_mean=0.0,
        gray_std=1.0,
    )


class CellCropClassifierProbeTest(unittest.TestCase):
    def test_color_stats_separates_red_edge_and_black_ink(self) -> None:
        image = Image.new("RGB", (24, 24), (220, 180, 120))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 23, 3), fill=(210, 30, 25))
        draw.rectangle((9, 9, 14, 14), fill=(20, 20, 20))

        stats = color_stats(image)

        self.assertGreater(stats["red_share"], 0.0)
        self.assertGreater(stats["black_ink_ratio"], 0.0)
        self.assertGreater(stats["edge_red_share"], stats["central_red_share"])

    def test_ranked_predictions_report_expected_rank(self) -> None:
        ranked = ranked_predictions(["a", "b", "c"], [0.1, 0.9, 0.4])

        self.assertEqual([label for _rank, label, _score in ranked], ["b", "c", "a"])
        self.assertEqual(expected_rank(ranked, "c"), 2)
        self.assertIsNone(expected_rank(ranked, "missing"))

    def test_loso_training_keeps_test_source_out_and_predicts_rows(self) -> None:
        examples = []
        for idx in range(4):
            sample = f"sample_{idx}"
            examples.append(example(sample, 1, 1, "black:FU", [0.0 + idx * 0.01, 0.0]))
            examples.append(example(sample, 1, 2, "white:KY", [8.0 + idx * 0.01, 8.0]))

        rows, audit = train_and_predict_loso(examples, "linear_svm")

        self.assertEqual(len(rows), len(examples))
        self.assertEqual(sum(1 for row in audit if row["overlap_source_count"] != 0), 0)
        self.assertTrue(all(row["classifier_top3_contains_expected"] for row in rows))
        self.assertEqual(len(class_count_rows(examples)), 2)

    def test_confusion_matrix_counts_expected_vs_predicted(self) -> None:
        rows = [
            {"expected": "black:FU", "classifier_top1": "black:FU"},
            {"expected": "black:FU", "classifier_top1": "white:KY"},
            {"expected": "white:KY", "classifier_top1": "white:KY"},
        ]

        matrix = confusion_matrix_rows(rows)
        by_expected = {row["expected"]: row for row in matrix}

        self.assertEqual(by_expected["black:FU"]["black:FU"], 1)
        self.assertEqual(by_expected["black:FU"]["white:KY"], 1)
        self.assertEqual(by_expected["white:KY"]["white:KY"], 1)

    def test_split_confusion_and_error_summary_keep_fold_context(self) -> None:
        rows = [
            {
                "split_mode": "split",
                "fold_id": "fold",
                "expected": "black:FU",
                "classifier_top1": "white:KY",
                "classifier_top1_correct": False,
                "classifier_top3_contains_expected": True,
                "classifier_top1_margin": 1.5,
            },
            {
                "split_mode": "split",
                "fold_id": "fold",
                "expected": "black:FU",
                "classifier_top1": "black:FU",
                "classifier_top1_correct": True,
                "classifier_top3_contains_expected": True,
                "classifier_top1_margin": 2.0,
            },
        ]

        matrix, fields = confusion_matrix_rows_by_split(rows)
        errors = classifier_error_summary_rows(rows)

        self.assertIn("split_mode", fields)
        self.assertEqual(matrix[0]["split_mode"], "split")
        self.assertEqual(errors[0]["expected"], "black:FU")
        self.assertEqual(errors[0]["classifier_top1"], "white:KY")
        self.assertEqual(errors[0]["count"], 1)

    def test_residual_summary_joins_gap_rank_and_candidate_leak_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            analysis = Path(temp)
            fields = [
                "app",
                "piece_style",
                "sample",
                "square",
                "row",
                "col",
                "expected",
                "predicted_top1",
                "expected_candidate_rank",
                "top_score",
                "expected_score",
            ]
            gap_row = {
                "app": "ぴよ将棋",
                "piece_style": "ひよこ駒",
                "sample": "target_sample",
                "square": "5五",
                "row": 5,
                "col": 5,
                "expected": "black:FU",
                "predicted_top1": "white:KY",
                "expected_candidate_rank": 12,
                "top_score": 0.5,
                "expected_score": 0.4,
            }
            write_csv(analysis / "piece_style_board_error_candidate_gaps.csv", [gap_row], fields)
            write_csv(
                analysis / "piece_style_board_errors.csv",
                [
                    {
                        "app": "ぴよ将棋",
                        "piece_style": "ひよこ駒",
                        "sample": "target_sample",
                        "square": "5五",
                        "expected": "black:FU",
                        "predicted_top1": "white:KY",
                    }
                ],
                ["app", "piece_style", "sample", "square", "expected", "predicted_top1"],
            )
            write_csv(
                analysis / "piece_style_board_error_candidates.csv",
                [
                    {
                        **gap_row,
                        "candidate_identity": "white:KY",
                        "candidate_rank": 1,
                        "source": "app_template:target_sample+1tpl",
                    }
                ],
                fields + ["candidate_identity", "candidate_rank", "source"],
            )
            predictions = [
                {
                    "app": "ぴよ将棋",
                    "piece_style": "ひよこ駒",
                    "sample": "target_sample",
                    "source_id": "ぴよ将棋/ひよこ駒/target_sample",
                    "row": 5,
                    "col": 5,
                    "expected": "black:FU",
                    "classifier_top1": "black:FU",
                    "classifier_top2": "white:KY",
                    "classifier_top3": "black:KI",
                    "classifier_expected_rank": 1,
                    "classifier_top3_contains_expected": True,
                }
            ]

            rows, audit = residual_summary_rows(analysis, predictions)

            self.assertEqual(rows[0]["baseline_expected_rank"], 12)
            self.assertTrue(rows[0]["classifier_top1_correct"])
            self.assertEqual(rows[0]["classifier_rank_delta_vs_baseline"], -11)
            self.assertEqual(rows[0]["baseline_candidate_source_leak_rows"], 1)
            self.assertTrue(audit[0]["leak"])

    def test_strict_split_holds_out_piyo_chick_normal_and_reports_missing_class(self) -> None:
        examples = [
            example("ぴよ将棋_ひよこ駒_通常_01", 1, 1, "black:FU", [0.0, 0.0]),
            example("ぴよ将棋_ひよこ駒_通常_01", 1, 2, "black:OU", [3.0, 3.0]),
            example("ぴよ将棋_ひよこ駒_初期_01", 1, 1, "white:KY", [8.0, 8.0]),
            example("将棋ウォーズ_通常駒_通常_01", 1, 1, "black:FU", [0.1, 0.1], app="将棋ウォーズ", style="通常駒"),
            example("将棋ウォーズ_通常駒_通常_01", 1, 2, "white:KY", [7.9, 7.9], app="将棋ウォーズ", style="通常駒"),
            example("将棋クエスト_通常駒_通常_01", 1, 1, "black:FU", [0.2, 0.2], app="将棋クエスト", style="通常駒"),
            example("将棋クエスト_通常駒_通常_01", 1, 2, "white:KY", [7.8, 7.8], app="将棋クエスト", style="通常駒"),
        ]

        splits = strict_split_definitions(examples)
        normal_split = next(split for split in splits if split.split_mode == "leave_piyo_chick_normal_out")
        train_sources = {examples[index].source_id for index in normal_split.train_indices}
        test_sources = {examples[index].source_id for index in normal_split.test_indices}

        self.assertFalse(train_sources & test_sources)
        self.assertTrue(all("通常_" in examples[index].sample for index in normal_split.test_indices))
        coverage = class_coverage_rows_for_split(examples, normal_split)
        missing = {row["label"] for row in coverage if row["missing_in_train"]}
        self.assertEqual(missing, {"black:OU"})

        rows, metrics, audit, _coverage = train_and_predict_strict_splits(examples, "linear_svm", [normal_split])
        by_expected = {row["expected"]: row for row in rows}

        self.assertEqual(len(rows), 2)
        self.assertEqual(audit[0]["overlap_source_count"], 0)
        self.assertEqual(audit[0]["train_piyo_chick_normal_count"], 0)
        self.assertEqual(audit[0]["test_piyo_chick_normal_count"], 2)
        self.assertFalse(by_expected["black:OU"]["expected_class_in_train"])
        self.assertEqual(by_expected["black:OU"]["classifier_expected_rank"], "")
        self.assertEqual(metrics[0]["missing_test_class_count"], 1)
        self.assertEqual(metrics[0]["missing_test_example_count"], 1)

    def test_strict_residual_summary_keeps_split_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            analysis = Path(temp)
            fields = [
                "app",
                "piece_style",
                "sample",
                "square",
                "row",
                "col",
                "expected",
                "predicted_top1",
                "expected_candidate_rank",
                "top_score",
                "expected_score",
            ]
            gap_row = {
                "app": "ぴよ将棋",
                "piece_style": "ひよこ駒",
                "sample": "target_sample",
                "square": "5五",
                "row": 5,
                "col": 5,
                "expected": "black:FU",
                "predicted_top1": "white:KY",
                "expected_candidate_rank": 12,
                "top_score": 0.5,
                "expected_score": 0.4,
            }
            write_csv(analysis / "piece_style_board_error_candidate_gaps.csv", [gap_row], fields)
            write_csv(
                analysis / "piece_style_board_errors.csv",
                [{**gap_row}],
                fields,
            )
            write_csv(analysis / "piece_style_board_error_candidates.csv", [], fields + ["candidate_identity", "candidate_rank", "source"])
            predictions = [
                {
                    "split_mode": "leave_piyo_chick_normal_out",
                    "fold_id": "ぴよ将棋/ひよこ駒/通常_*",
                    "app": "ぴよ将棋",
                    "piece_style": "ひよこ駒",
                    "sample": "target_sample",
                    "source_id": "ぴよ将棋/ひよこ駒/target_sample",
                    "row": 5,
                    "col": 5,
                    "expected": "black:FU",
                    "classifier_top1": "black:FU",
                    "classifier_top2": "white:KY",
                    "classifier_top3": "black:KI",
                    "classifier_top1_margin": 2.5,
                    "classifier_expected_rank": 1,
                    "classifier_top3_contains_expected": True,
                    "expected_class_in_train": True,
                }
            ]

            rows, audit = residual_summary_rows_by_split(analysis, predictions)

            self.assertEqual(rows[0]["split_mode"], "leave_piyo_chick_normal_out")
            self.assertEqual(rows[0]["classifier_top1_margin"], 2.5)
            self.assertTrue(rows[0]["expected_class_in_train"])
            self.assertEqual(rows[0]["source_id"], "ぴよ将棋/ひよこ駒/target_sample")
            self.assertEqual(audit[0]["fold_id"], "ぴよ将棋/ひよこ駒/通常_*")

    def test_confidence_margin_audit_counts_residual_and_stable_accepted(self) -> None:
        predictions = [
            {
                "split_mode": "split",
                "fold_id": "fold",
                "source_id": "source_a",
                "row": 1,
                "col": 1,
                "classifier_top1_margin": 2.0,
                "classifier_top1_correct": True,
                "expected_class_in_train": True,
            },
            {
                "split_mode": "split",
                "fold_id": "fold",
                "source_id": "source_b",
                "row": 1,
                "col": 2,
                "classifier_top1_margin": 3.0,
                "classifier_top1_correct": False,
                "expected_class_in_train": True,
            },
            {
                "split_mode": "split",
                "fold_id": "fold",
                "source_id": "source_c",
                "row": 1,
                "col": 3,
                "classifier_top1_margin": 5.0,
                "classifier_top1_correct": True,
                "expected_class_in_train": False,
            },
        ]
        residual = [{**predictions[0], "classifier_top1_correct": True}]
        stable = [{**predictions[1], "classifier_top1_correct": False}]

        rows = confidence_margin_audit_rows(predictions, residual, stable, thresholds=[1.0, 2.5])
        by_threshold = {row["margin_threshold"]: row for row in rows}

        self.assertEqual(by_threshold[1.0]["accepted_example_count"], 2)
        self.assertEqual(by_threshold[1.0]["residual_accepted_top1_fixes"], 1)
        self.assertEqual(by_threshold[1.0]["stable_degradation_accepted"], 1)
        self.assertEqual(by_threshold[2.5]["residual_accepted_top1_fixes"], 0)
        self.assertEqual(by_threshold[2.5]["stable_degradation_accepted"], 1)

        quantile_rows = confidence_margin_quantile_audit_rows(predictions, residual, stable, quantiles=[0.5])
        self.assertEqual(len(quantile_rows), 1)
        self.assertEqual(quantile_rows[0]["split_mode"], "split")
        self.assertEqual(quantile_rows[0]["margin_quantile"], 0.5)
        self.assertGreaterEqual(quantile_rows[0]["accepted_example_count"], 1)


if __name__ == "__main__":
    unittest.main()
