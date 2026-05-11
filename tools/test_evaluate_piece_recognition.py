from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluate_piece_recognition import (  # noqa: E402
    leak_guard_errors,
    load_report,
    main as piece_main,
    source_field_contains_source,
    strict_failure_reasons as piece_strict_failure_reasons,
)
from evaluate_analysis_by_app_piece_style import (  # noqa: E402
    collect_board_error_candidate_gap_rows,
    collect_board_error_candidate_gap_summary_rows,
    collect_board_error_candidate_rows,
    collect_board_error_template_supply_rows,
    evaluate_manifest,
    main as grouped_main,
    strict_failure_reasons as grouped_strict_failure_reasons,
    timing_summary,
)
from run_android_device_eval import (  # noqa: E402
    evaluate as evaluate_android_run,
    is_perfect,
    reports_missing_exclusion_metadata,
)


class LeakGuardTest(unittest.TestCase):
    def test_source_field_contains_nested_forbidden_source_prefixes(self) -> None:
        sample = "sample_001"
        leaking_sources = [
            "learned:sample_001",
            "hand_learned:sample_001",
            "hand_template:app_template:sample_001",
            "known_sample:sample_001",
            "label:sample_001",
            "initial:sample_001",
            "calibration:sample_001",
            "hand_hog:sample_001",
        ]
        for source in leaking_sources:
            with self.subTest(source=source):
                self.assertTrue(source_field_contains_source(source, sample))

        self.assertFalse(source_field_contains_source("position_prior:shogi_wars:single", sample))
        self.assertFalse(source_field_contains_source("learned:sample_001_extra", sample))

    def test_excluded_sources_array_satisfies_strict_guard(self) -> None:
        report = {
            "model": {
                "excluded_sources": ["sample_001", "sample_002"],
                "training_sources": ["sample_003"],
            },
            "cells": [],
        }

        self.assertEqual(
            [],
            leak_guard_errors(report, ["sample_001", "sample_002"], require_excluded_source=True),
        )

    def test_source_prefix_leak_is_reported_even_when_exclusion_metadata_matches(self) -> None:
        report = {
            "model": {"excluded_source": "sample_001"},
            "cells": [{"source": "initial:sample_001"}],
        }

        errors = leak_guard_errors(report, ["sample_001"], require_excluded_source=True)

        self.assertTrue(any("$.cells[0].source leaks forbidden source sample_001" in error for error in errors))

    def test_load_report_uses_android_sample_meta_as_excluded_source_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir)
            report_path = sample_dir / "piece_report.json"
            report_path.write_text(json.dumps({"cells": []}), encoding="utf-8")
            (sample_dir / "sample_meta.txt").write_text(
                "sample=sample_001\nexcluded_template_source=sample_001\n",
                encoding="utf-8",
            )

            report = load_report(report_path)

        self.assertEqual("sample_001", report["model"]["excluded_source"])
        self.assertEqual("android_eval_sample_meta_v1", report["model"]["metadata_schema"])

    def test_allow_missing_excluded_source_does_not_disable_source_leak_scanning(self) -> None:
        report = {
            "model": {
                "training_sources": ["sample_001"],
            },
            "cells": [
                {
                    "source": "label:sample_001",
                },
            ],
        }

        errors = leak_guard_errors(report, ["sample_001"], require_excluded_source=False)

        self.assertFalse(any("excluded_source is missing" in error for error in errors))
        self.assertTrue(any("training_sources" in error for error in errors))
        self.assertTrue(any("$.cells[0].source is label:sample_001" in error for error in errors))

    def test_no_leak_options_excluded_template_source_is_metadata_and_validated(self) -> None:
        report = {
            "model": {
                "excluded_source": "sample_001",
                "excluded_sources": ["sample_001"],
                "no_leak_options": {
                    "excluded_template_source": "sample_001",
                    "excluded_template_sources": ["sample_001"],
                    "excludedTemplateSource": "sample_001",
                    "excludedTemplateSources": ["sample_001"],
                },
            },
            "cells": [],
        }

        self.assertEqual([], leak_guard_errors(report, ["sample_001"], require_excluded_source=True))

        wrong_options = {
            "model": {
                "excluded_source": "sample_001",
                "no_leak_options": {
                    "excludedTemplateSource": "other_sample",
                },
            },
            "cells": [],
        }
        errors = leak_guard_errors(wrong_options, ["sample_001"], require_excluded_source=True)

        self.assertTrue(any("no_leak_options excluded sources do not include expected sample_001" in error for error in errors))
        self.assertTrue(any("not a subset of model excluded sources" in error for error in errors))

    def test_allow_missing_excluded_source_still_rejects_present_wrong_metadata(self) -> None:
        report = {
            "model": {
                "excluded_source": "other_sample",
            },
            "cells": [],
        }

        errors = leak_guard_errors(report, ["sample_001"], require_excluded_source=False)

        self.assertFalse(any("excluded_source is missing" in error for error in errors))
        self.assertTrue(any("model excluded_sources do not include expected sample_001" in error for error in errors))


class EvaluationGateTest(unittest.TestCase):
    def test_piece_require_perfect_rejects_bad_metrics_and_skipped_samples(self) -> None:
        output = {
            "summary": {
                "errors": 1,
                "hand_errors": 2,
                "false_empty_on_piece": 3,
                "false_piece_on_empty": 4,
                "unknown_on_piece": 5,
                "high_confidence_errors": 6,
                "leak_errors": 7,
            },
            "skipped": [{"sample": "sample_001"}],
        }

        reasons = piece_strict_failure_reasons(output, require_perfect=True, fail_on_skipped=False)

        self.assertEqual(
            [
                "skipped_samples=1",
                "errors=1",
                "hand_errors=2",
                "false_empty_on_piece=3",
                "false_piece_on_empty=4",
                "unknown_on_piece=5",
                "high_confidence_errors=6",
                "leak_errors=7",
            ],
            reasons,
        )

    def test_piece_fail_on_skipped_rejects_missing_labels_without_require_perfect(self) -> None:
        output = {
            "summary": {
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 0,
            },
            "skipped": [{"sample": "sample_001"}],
        }

        self.assertEqual(
            ["skipped_samples=1"],
            piece_strict_failure_reasons(output, require_perfect=False, fail_on_skipped=True),
        )

    def test_piece_strict_leak_guard_rejects_leaks_without_require_perfect(self) -> None:
        output = {
            "summary": {
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 2,
            },
            "skipped": [],
        }

        self.assertEqual(
            ["leak_errors=2"],
            piece_strict_failure_reasons(
                output,
                require_perfect=False,
                fail_on_skipped=False,
                fail_on_leak=True,
            ),
        )

    def test_piece_cli_fail_on_skipped_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_dir = root / "sample_without_label"
            labels_dir = root / "labels"
            report_dir.mkdir()
            labels_dir.mkdir()
            (report_dir / "piece_report.json").write_text(json.dumps({"cells": []}), encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "evaluate_piece_recognition.py",
                    str(report_dir),
                    "--labels-dir",
                    str(labels_dir),
                    "--fail-on-skipped",
                ],
            ):
                with self.assertRaises(SystemExit) as raised:
                    piece_main()

        self.assertEqual(2, raised.exception.code)

    def test_timing_summary_counts_missing_timing_instead_of_zero_seconds(self) -> None:
        summary = timing_summary(
            [
                {"sample": "missing", "seconds": "", "app": "app", "glyph": "glyph"},
                {"sample": "invalid", "seconds": "not-a-number", "app": "app", "glyph": "glyph"},
                {"sample": "fast", "seconds": "0.5000", "app": "app", "glyph": "glyph"},
                {"sample": "slow", "seconds": "6.2500", "app": "app", "glyph": "glyph"},
            ],
            max_seconds=5.0,
        )

        self.assertEqual(4, summary["images"])
        self.assertEqual(2, summary["timed_images"])
        self.assertEqual(2, summary["missing_seconds_count"])
        self.assertEqual(1, summary["over_limit_count"])
        self.assertEqual("slow", summary["over_limit_samples"][0]["sample"])
        self.assertEqual("glyph", summary["over_limit_samples"][0]["piece_style"])

    def test_grouped_require_speed_rejects_over_limit_and_missing_timing(self) -> None:
        result = {
            "overall": {
                "skipped_samples": 0,
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 0,
                "timing": {
                    "over_limit_count": 1,
                    "missing_seconds_count": 2,
                },
            },
        }

        self.assertEqual(
            ["over_limit_count=1", "missing_seconds_count=2"],
            grouped_strict_failure_reasons(
                result,
                require_perfect=False,
                fail_on_skipped=False,
                require_speed=True,
                fail_on_missing_timing=False,
            ),
        )

    def test_grouped_fail_on_skipped_rejects_without_require_perfect(self) -> None:
        result = {
            "overall": {
                "skipped_samples": 2,
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 0,
                "timing": {
                    "over_limit_count": 0,
                    "missing_seconds_count": 0,
                },
            },
        }

        self.assertEqual(
            ["skipped_samples=2"],
            grouped_strict_failure_reasons(
                result,
                require_perfect=False,
                fail_on_skipped=True,
                require_speed=False,
                fail_on_missing_timing=False,
            ),
        )

    def test_grouped_fail_on_missing_timing_rejects_without_require_speed(self) -> None:
        result = {
            "overall": {
                "skipped_samples": 0,
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 0,
                "timing": {
                    "over_limit_count": 0,
                    "missing_seconds_count": 3,
                },
            },
        }

        self.assertEqual(
            ["missing_seconds_count=3"],
            grouped_strict_failure_reasons(
                result,
                require_perfect=False,
                fail_on_skipped=False,
                require_speed=False,
                fail_on_missing_timing=True,
            ),
        )

    def test_grouped_strict_leak_guard_rejects_leaks_without_require_perfect(self) -> None:
        result = {
            "overall": {
                "skipped_samples": 0,
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 3,
                "timing": {
                    "over_limit_count": 0,
                    "missing_seconds_count": 0,
                },
            },
        }

        self.assertEqual(
            ["leak_errors=3"],
            grouped_strict_failure_reasons(
                result,
                require_perfect=False,
                fail_on_skipped=False,
                require_speed=False,
                fail_on_missing_timing=False,
                fail_on_leak=True,
            ),
        )

    def test_grouped_cli_require_speed_exits_nonzero_for_missing_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            labels_dir = root / "labels"
            app = "ぴよ将棋"
            style = "ひよこ駒"
            sample = "sample_missing_timing"
            report_path = run_dir / app / style / sample / "piece_report.json"
            label_path = labels_dir / app / style / f"{sample}.json"
            report_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)

            cells = [
                {
                    "row": row,
                    "col": col,
                    "state": "empty",
                    "color": None,
                    "piece": None,
                    "confidence": 1.0,
                    "candidates": [],
                }
                for row in range(1, 10)
                for col in range(1, 10)
            ]
            report_path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
            label_path.write_text(json.dumps({"rows": [["empty"] * 9 for _ in range(9)]}), encoding="utf-8")
            (run_dir / "manifest.csv").write_text(
                "\n".join(
                    [
                        "app,piece_style,sample,label_status,report,seconds",
                        f"{app},{style},{sample},教師ラベルあり,{report_path},",
                    ],
                )
                + "\n",
                encoding="utf-8-sig",
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "evaluate_analysis_by_app_piece_style.py",
                    str(run_dir),
                    "--labels-dir",
                    str(labels_dir),
                    "--require-speed",
                    "--max-seconds",
                    "5",
                ],
            ):
                with self.assertRaises(SystemExit) as raised:
                    grouped_main()

        self.assertEqual(2, raised.exception.code)

    def test_grouped_require_perfect_rejects_leak_high_confidence_and_identity_errors(self) -> None:
        result = {
            "overall": {
                "skipped_samples": 1,
                "errors": 2,
                "hand_errors": 3,
                "false_empty_on_piece": 4,
                "false_piece_on_empty": 5,
                "unknown_on_piece": 6,
                "high_confidence_errors": 7,
                "leak_errors": 8,
                "timing": {
                    "over_limit_count": 0,
                    "missing_seconds_count": 0,
                },
            },
        }

        reasons = grouped_strict_failure_reasons(
            result,
            require_perfect=True,
            fail_on_skipped=False,
            require_speed=False,
            fail_on_missing_timing=False,
        )

        self.assertEqual(
            [
                "skipped_samples=1",
                "errors=2",
                "hand_errors=3",
                "false_empty_on_piece=4",
                "false_piece_on_empty=5",
                "unknown_on_piece=6",
                "high_confidence_errors=7",
                "leak_errors=8",
            ],
            reasons,
        )

    def test_android_is_perfect_rejects_bad_metrics_skips_leaks_and_speed_failures(self) -> None:
        perfect = {
            "overall": {
                "errors": 0,
                "hand_errors": 0,
                "false_empty_on_piece": 0,
                "false_piece_on_empty": 0,
                "unknown_on_piece": 0,
                "high_confidence_errors": 0,
                "leak_errors": 0,
                "skipped_samples": 0,
                "timing": {
                    "over_limit_count": 0,
                    "missing_seconds_count": 0,
                },
            },
        }
        self.assertTrue(is_perfect(perfect, require_speed=True))

        for key in (
            "errors",
            "hand_errors",
            "false_empty_on_piece",
            "false_piece_on_empty",
            "unknown_on_piece",
            "high_confidence_errors",
            "leak_errors",
            "skipped_samples",
        ):
            bad = json.loads(json.dumps(perfect))
            bad["overall"][key] = 1
            with self.subTest(key=key):
                self.assertFalse(is_perfect(bad, require_speed=True))

        for key in ("over_limit_count", "missing_seconds_count"):
            bad = json.loads(json.dumps(perfect))
            bad["overall"]["timing"][key] = 1
            with self.subTest(key=key):
                self.assertFalse(is_perfect(bad, require_speed=True))

    def test_manifest_evaluation_aggregates_real_bad_metrics_from_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            labels_dir = root / "labels"
            app = "ぴよ将棋"
            style = "ひよこ駒"
            sample = "sample_bad_metrics"
            report_path = run_dir / app / style / sample / "piece_report.json"
            label_path = labels_dir / app / style / f"{sample}.json"
            report_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)

            label_rows = [["empty"] * 9 for _ in range(9)]
            label_rows[0][0] = "black:FU"
            label_path.write_text(json.dumps({"rows": label_rows}), encoding="utf-8")

            cells = [
                {
                    "row": row,
                    "col": col,
                    "state": "empty",
                    "color": None,
                    "piece": None,
                    "confidence": 1.0,
                    "candidates": [],
                }
                for row in range(1, 10)
                for col in range(1, 10)
            ]
            cells[0].update({"state": "unknown", "confidence": 0.99})
            cells[1].update(
                {
                    "state": "piece",
                    "color": "white",
                    "piece": "KA",
                    "confidence": 0.99,
                    "candidates": [
                        {
                            "color": "white",
                            "piece": "KA",
                            "score": 0.99,
                            "source": f"label:{sample}",
                        }
                    ],
                }
            )
            report_path.write_text(
                json.dumps({"model": {"excluded_source": sample}, "cells": cells}),
                encoding="utf-8",
            )
            (run_dir / "manifest.csv").write_text(
                "\n".join(
                    [
                        "app,piece_style,sample,label_status,report,seconds",
                        f"{app},{style},{sample},教師ラベルあり,{report_path},1.0000",
                    ],
                )
                + "\n",
                encoding="utf-8-sig",
            )

            result = evaluate_manifest(
                run_dir,
                labels_dir,
                include_hands=False,
                high_confidence_threshold=0.75,
                max_seconds=5.0,
                strict_leak_guard=True,
                require_excluded_source=True,
            )

        self.assertEqual(1, result["overall"]["evaluated_samples"])
        self.assertEqual(2, result["overall"]["errors"])
        self.assertEqual(2, result["overall"]["high_confidence_errors"])
        self.assertEqual(1, result["overall"]["false_piece_on_empty"])
        self.assertEqual(1, result["overall"]["unknown_on_piece"])
        self.assertGreater(result["overall"]["leak_errors"], 0)

    def test_collect_board_error_candidate_rows_uses_diagnostic_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            labels_dir = root / "labels"
            app = "ぴよ将棋"
            style = "ひよこ駒"
            sample = "sample_candidate_diagnostics"
            report_path = run_dir / app / style / sample / "piece_report.json"
            label_path = labels_dir / app / style / f"{sample}.json"
            report_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)

            label_rows = [["empty"] * 9 for _ in range(9)]
            label_rows[0][0] = "black:FU"
            label_path.write_text(json.dumps({"rows": label_rows}), encoding="utf-8")

            cells = [
                {
                    "row": row,
                    "col": col,
                    "state": "empty",
                    "color": None,
                    "piece": None,
                    "confidence": 1.0,
                    "candidates": [],
                }
                for row in range(1, 10)
                for col in range(1, 10)
            ]
            cells[0].update(
                {
                    "state": "unknown",
                    "confidence": 0.46,
                    "candidates": [
                        {
                            "color": "white",
                            "piece": "KY",
                            "score": 0.46,
                            "source": "template_top1",
                        },
                        {
                            "color": "black",
                            "piece": "FU",
                            "score": 0.34,
                            "source": "template_expected",
                        },
                    ],
                    "diagnostic_candidates": [
                        {
                            "rank": 1,
                            "color": "white",
                            "piece": "KY",
                            "score": 0.46,
                            "source": "template_top1",
                            "score_breakdown": {
                                "shape": 0.37,
                                "bbox": 0.89,
                                "position_boost": 0.0,
                                "position_prior_boost": 0.0,
                                "score_without_position_boost": 0.46,
                                "unweighted_template_score": 0.42,
                                "template_weight_contribution": 0.04,
                                "base_template_score": 0.41,
                                "weighted_template_score": 0.45,
                                "exact_template_weight_contribution": 0.04,
                                "score_formula_residual": 0.01,
                                "source_bbox_width": 0.70,
                                "source_bbox_height": 0.80,
                                "target_bbox_width": 0.90,
                                "target_bbox_height": 0.95,
                                "source_bbox_center_x": 0.50,
                                "source_bbox_center_y": 0.60,
                                "target_bbox_center_x": 0.52,
                                "target_bbox_center_y": 0.10,
                                "source_ink_ratio": 0.20,
                                "target_ink_ratio": 0.30,
                                "source_clean_ink_ratio": 0.18,
                                "target_clean_ink_ratio": 0.28,
                                "source_ink_count": 100,
                                "target_ink_count": 140,
                                "source_clean_ink_count": 80,
                                "target_clean_ink_count": 120,
                                "source_connected_component_count": 3,
                                "target_connected_component_count": 1,
                                "source_largest_component_area_ratio": 0.70,
                                "target_largest_component_area_ratio": 0.90,
                                "source_red_dominant_component_area_ratio": 0.20,
                                "target_red_dominant_component_area_ratio": 0.00,
                                "source_edge_touching_component_area_ratio": 0.30,
                                "target_edge_touching_component_area_ratio": 0.10,
                                "source_component_pruned_glyph_bbox_width": 0.55,
                                "source_component_pruned_glyph_bbox_height": 0.66,
                                "target_component_pruned_glyph_bbox_width": 0.21,
                                "target_component_pruned_glyph_bbox_height": 0.82,
                                "source_component_pruned_glyph_bbox_center_x": 0.48,
                                "source_component_pruned_glyph_bbox_center_y": 0.58,
                                "target_component_pruned_glyph_bbox_center_x": 0.50,
                                "target_component_pruned_glyph_bbox_center_y": 0.12,
                                "source_component_pruned_glyph_ink_count": 72,
                                "target_component_pruned_glyph_ink_count": 64,
                                "source_component_pruned_glyph_clean_ink_count": 88,
                                "target_component_pruned_glyph_clean_ink_count": 90,
                            },
                        },
                        {
                            "rank": 7,
                            "color": "black",
                            "piece": "FU",
                            "score": 0.34,
                            "source": "template_expected+position_prior",
                            "score_breakdown": {
                                "shape": 0.31,
                                "bbox": 0.39,
                                "position_boost": 0.02,
                                "position_prior_boost": 0.02,
                                "score_without_position_prior": 0.32,
                                "score_without_position_boost": 0.32,
                                "unweighted_template_score": 0.30,
                                "template_weight_contribution": 0.02,
                                "base_template_score": 0.29,
                                "weighted_template_score": 0.31,
                                "exact_template_weight_contribution": 0.02,
                                "score_formula_residual": 0.01,
                                "source_bbox_width": 0.70,
                                "source_bbox_height": 0.80,
                                "target_bbox_width": 0.40,
                                "target_bbox_height": 0.45,
                                "source_bbox_center_x": 0.50,
                                "source_bbox_center_y": 0.60,
                                "target_bbox_center_x": 0.51,
                                "target_bbox_center_y": 0.58,
                                "source_ink_ratio": 0.20,
                                "target_ink_ratio": 0.22,
                                "source_clean_ink_ratio": 0.18,
                                "target_clean_ink_ratio": 0.19,
                                "source_ink_count": 100,
                                "target_ink_count": 90,
                                "source_clean_ink_count": 80,
                                "target_clean_ink_count": 70,
                                "source_connected_component_count": 3,
                                "target_connected_component_count": 2,
                                "source_largest_component_area_ratio": 0.70,
                                "target_largest_component_area_ratio": 0.40,
                                "source_red_dominant_component_area_ratio": 0.20,
                                "target_red_dominant_component_area_ratio": 0.05,
                                "source_edge_touching_component_area_ratio": 0.30,
                                "target_edge_touching_component_area_ratio": 0.15,
                                "source_component_pruned_glyph_bbox_width": 0.55,
                                "source_component_pruned_glyph_bbox_height": 0.66,
                                "target_component_pruned_glyph_bbox_width": 0.31,
                                "target_component_pruned_glyph_bbox_height": 0.42,
                                "source_component_pruned_glyph_bbox_center_x": 0.48,
                                "source_component_pruned_glyph_bbox_center_y": 0.58,
                                "target_component_pruned_glyph_bbox_center_x": 0.47,
                                "target_component_pruned_glyph_bbox_center_y": 0.52,
                                "source_component_pruned_glyph_ink_count": 72,
                                "target_component_pruned_glyph_ink_count": 44,
                                "source_component_pruned_glyph_clean_ink_count": 88,
                                "target_component_pruned_glyph_clean_ink_count": 50,
                            },
                        },
                    ],
                }
            )
            report_path.write_text(
                json.dumps({"model": {"excluded_source": sample}, "cells": cells}),
                encoding="utf-8",
            )
            (run_dir / "manifest.csv").write_text(
                "\n".join(
                    [
                        "app,piece_style,sample,label_status,report,seconds",
                        f"{app},{style},{sample},教師ラベルあり,{report_path},1.0000",
                    ],
                )
                + "\n",
                encoding="utf-8-sig",
            )

            result = evaluate_manifest(
                run_dir,
                labels_dir,
                include_hands=False,
                high_confidence_threshold=0.75,
                max_seconds=5.0,
                strict_leak_guard=True,
                require_excluded_source=True,
            )
            rows = collect_board_error_candidate_rows(result["results"])
            gap_rows = collect_board_error_candidate_gap_rows(rows)
            summary_rows = collect_board_error_candidate_gap_summary_rows(gap_rows)
            supply_rows = collect_board_error_template_supply_rows(
                gap_rows,
                [
                    {
                        "preset": "piyo_chick",
                        "color": "BLACK",
                        "piece": "FU",
                        "source": "sample_candidate_diagnostics",
                        "row": 4,
                        "col": 4,
                    },
                    {
                        "preset": "piyo_chick",
                        "color": "BLACK",
                        "piece": "FU",
                        "source": "other_sample",
                        "row": 7,
                        "col": 1,
                    },
                    {
                        "preset": "piyo_chick",
                        "color": "WHITE",
                        "piece": "KY",
                        "source": "top_sample",
                        "row": 2,
                        "col": 1,
                    },
                ],
            )

        self.assertEqual(2, len(rows))
        top1, expected = rows
        self.assertEqual("diagnostic_candidates", top1["candidate_set"])
        self.assertEqual("white:KY", top1["candidate_identity"])
        self.assertTrue(top1["is_predicted_top1"])
        self.assertFalse(top1["is_expected"])
        self.assertEqual(7, top1["expected_candidate_rank"])
        self.assertEqual("black:FU", expected["candidate_identity"])
        self.assertTrue(expected["is_expected"])
        self.assertEqual(7, expected["candidate_rank"])
        self.assertEqual(0.02, expected["position_prior_boost"])
        self.assertEqual(0.32, expected["score_without_position_prior"])
        self.assertEqual(0.32, expected["score_without_position_boost"])
        self.assertEqual(0.30, expected["unweighted_template_score"])
        self.assertEqual(0.02, expected["template_weight_contribution"])
        self.assertEqual(0.29, expected["base_template_score"])
        self.assertEqual(0.31, expected["weighted_template_score"])
        self.assertEqual(0.02, expected["exact_template_weight_contribution"])
        self.assertEqual(0.01, expected["score_formula_residual"])
        self.assertEqual(0.70, expected["source_bbox_width"])
        self.assertEqual(0.40, expected["target_bbox_width"])
        self.assertEqual(70, expected["target_clean_ink_count"])
        self.assertEqual(3, expected["source_connected_component_count"])
        self.assertEqual(2, expected["target_connected_component_count"])
        self.assertEqual(0.40, expected["target_largest_component_area_ratio"])
        self.assertEqual(0.31, expected["target_component_pruned_glyph_bbox_width"])
        self.assertEqual(44, expected["target_component_pruned_glyph_ink_count"])
        self.assertEqual(50, expected["target_component_pruned_glyph_clean_ink_count"])
        self.assertEqual(1, len(gap_rows))
        gap = gap_rows[0]
        self.assertEqual("white:KY", gap["top_identity"])
        self.assertEqual("black:FU", gap["expected"])
        self.assertEqual("black:FU", gap["runner_up_identity"])
        self.assertEqual(7, gap["expected_candidate_rank"])
        self.assertEqual(0.12, gap["score_gap_top_minus_expected"])
        self.assertEqual(6.0, gap["rank_gap_top_to_expected"])
        self.assertEqual(0.06, gap["delta_shape"])
        self.assertEqual(-0.02, gap["delta_position_prior_boost"])
        self.assertEqual(0.14, gap["delta_score_without_position_boost"])
        self.assertEqual(0.12, gap["delta_unweighted_template_score"])
        self.assertEqual(0.02, gap["delta_template_weight_contribution"])
        self.assertEqual(0.12, gap["delta_base_template_score"])
        self.assertEqual(0.14, gap["delta_weighted_template_score"])
        self.assertEqual(0.02, gap["delta_exact_template_weight_contribution"])
        self.assertEqual(0.0, gap["delta_score_formula_residual"])
        self.assertEqual(0.0, gap["delta_source_bbox_width"])
        self.assertEqual(0.50, gap["delta_target_bbox_width"])
        self.assertEqual(-0.48, gap["delta_target_bbox_center_y"])
        self.assertEqual(50.0, gap["delta_target_clean_ink_count"])
        self.assertEqual(-1.0, gap["delta_target_connected_component_count"])
        self.assertEqual(0.50, gap["delta_target_largest_component_area_ratio"])
        self.assertEqual(-0.10, gap["delta_target_component_pruned_glyph_bbox_width"])
        self.assertEqual(20.0, gap["delta_target_component_pruned_glyph_ink_count"])
        self.assertEqual(40.0, gap["delta_target_component_pruned_glyph_clean_ink_count"])
        self.assertEqual(1, len(summary_rows))
        summary = summary_rows[0]
        self.assertEqual("white:KY", summary["predicted_top1"])
        self.assertEqual("black:FU", summary["expected"])
        self.assertEqual(1, summary["count"])
        self.assertEqual(0.12, summary["avg_score_gap_top_minus_expected"])
        self.assertEqual(0.06, summary["avg_delta_shape"])
        self.assertEqual(0.14, summary["avg_delta_score_without_position_boost"])
        self.assertEqual(0.12, summary["avg_delta_unweighted_template_score"])
        self.assertEqual(0.02, summary["avg_delta_template_weight_contribution"])
        self.assertEqual(0.12, summary["avg_delta_base_template_score"])
        self.assertEqual(0.14, summary["avg_delta_weighted_template_score"])
        self.assertEqual(0.02, summary["avg_delta_exact_template_weight_contribution"])
        self.assertEqual(0.0, summary["avg_delta_score_formula_residual"])
        self.assertEqual(0.50, summary["avg_delta_target_bbox_width"])
        self.assertEqual(50.0, summary["avg_delta_target_clean_ink_count"])
        self.assertEqual(-1.0, summary["avg_delta_target_connected_component_count"])
        self.assertEqual(0.50, summary["avg_delta_target_largest_component_area_ratio"])
        self.assertEqual(-0.10, summary["avg_delta_target_component_pruned_glyph_bbox_width"])
        self.assertEqual(20.0, summary["avg_delta_target_component_pruned_glyph_ink_count"])
        self.assertEqual(40.0, summary["avg_delta_target_component_pruned_glyph_clean_ink_count"])
        self.assertEqual(7.0, summary["expected_rank_avg"])
        self.assertEqual(1, len(supply_rows))
        supply = supply_rows[0]
        self.assertEqual("piyo_chick", supply["preset"])
        self.assertEqual(2, supply["expected_asset_template_count"])
        self.assertEqual(1, supply["expected_available_template_count"])
        self.assertEqual(1, supply["expected_excluded_template_count"])
        self.assertIn("other_sample:r7:c1", supply["expected_available_template_sources"])
        self.assertIn("sample_candidate_diagnostics:r4:c4", supply["expected_excluded_template_sources"])
        self.assertEqual(1, supply["predicted_available_template_count"])

    def test_mixed_metadata_run_does_not_auto_allow_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            labels_dir = Path(temp_dir) / "labels"
            with_meta = run_dir / "app" / "style" / "sample_with_meta"
            without_meta = run_dir / "app" / "style" / "sample_without_meta"
            with_meta.mkdir(parents=True)
            without_meta.mkdir(parents=True)
            (with_meta / "piece_report.json").write_text(
                json.dumps({"model": {"excluded_source": "sample_with_meta"}, "cells": []}),
                encoding="utf-8",
            )
            (without_meta / "piece_report.json").write_text(json.dumps({"cells": []}), encoding="utf-8")

            missing = reports_missing_exclusion_metadata(run_dir)
            self.assertEqual([without_meta / "piece_report.json"], missing)

            captured: dict[str, list[str]] = {}

            def fake_run(command: list[str], check: bool = True, capture: bool = False):
                captured["command"] = command
                (run_dir / "piece_style_evaluation_summary.json").write_text(
                    json.dumps(
                        {
                            "overall": {
                                "evaluated_samples": 0,
                                "skipped_samples": 0,
                                "errors": 0,
                                "hand_errors": 0,
                                "leak_errors": 0,
                            },
                            "groups": [],
                        },
                    ),
                    encoding="utf-8",
                )
                return type("Completed", (), {"returncode": 0})()

            with mock.patch("run_android_device_eval.run", side_effect=fake_run):
                evaluate_android_run(
                    run_dir,
                    labels_dir,
                    include_hands=False,
                    strict_leak_guard=True,
                    max_seconds=5.0,
                )

            self.assertNotIn("--allow-missing-excluded-source", captured["command"])

            with mock.patch("run_android_device_eval.run", side_effect=fake_run):
                evaluate_android_run(
                    run_dir,
                    labels_dir,
                    include_hands=False,
                    strict_leak_guard=True,
                    max_seconds=5.0,
                    allow_missing_excluded_source=True,
                )

            self.assertIn("--allow-missing-excluded-source", captured["command"])

        leaking_report = {
            "model": {},
            "cells": [{"source": "known_sample:sample_without_meta"}],
        }
        errors = leak_guard_errors(
            leaking_report,
            ["sample_without_meta"],
            require_excluded_source=False,
        )
        self.assertTrue(any("$.cells[0].source is known_sample:sample_without_meta" in error for error in errors))

    def test_grouped_mixed_metadata_allow_missing_only_suppresses_missing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            labels_dir = root / "labels"
            app = "app"
            style = "style"
            wrong_sample = "sample_with_wrong_metadata"
            missing_sample = "sample_without_metadata"
            wrong_report = run_dir / app / style / wrong_sample / "piece_report.json"
            missing_report = run_dir / app / style / missing_sample / "piece_report.json"
            wrong_report.parent.mkdir(parents=True)
            missing_report.parent.mkdir(parents=True)

            def empty_cells() -> list[dict[str, object]]:
                return [
                    {
                        "row": row,
                        "col": col,
                        "state": "empty",
                        "color": None,
                        "piece": None,
                        "confidence": 1.0,
                        "candidates": [],
                    }
                    for row in range(1, 10)
                    for col in range(1, 10)
                ]

            wrong_cells = empty_cells()
            wrong_cells[0]["sources"] = [
                f"known_sample:{wrong_sample}",
                f"label:{wrong_sample}",
                f"learned:{wrong_sample}",
            ]
            wrong_report.write_text(
                json.dumps(
                    {
                        "model": {
                            "excluded_source": "other_sample",
                            "no_leak_options": {
                                "excludedTemplateSource": "other_sample",
                            },
                        },
                        "cells": wrong_cells,
                    },
                ),
                encoding="utf-8",
            )
            missing_report.write_text(json.dumps({"cells": empty_cells()}), encoding="utf-8")

            for sample in (wrong_sample, missing_sample):
                label_path = labels_dir / app / style / f"{sample}.json"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    json.dumps({"rows": [["empty"] * 9 for _ in range(9)]}),
                    encoding="utf-8",
                )

            (run_dir / "manifest.csv").write_text(
                "\n".join(
                    [
                        "app,piece_style,sample,label_status,report,seconds",
                        f"{app},{style},{wrong_sample},教師ラベルあり,{wrong_report},1.0000",
                        f"{app},{style},{missing_sample},教師ラベルあり,{missing_report},1.0000",
                    ],
                )
                + "\n",
                encoding="utf-8-sig",
            )

            result = evaluate_manifest(
                run_dir,
                labels_dir,
                include_hands=False,
                high_confidence_threshold=0.75,
                max_seconds=5.0,
                strict_leak_guard=True,
                require_excluded_source=False,
            )

        self.assertEqual(2, result["overall"]["evaluated_samples"])
        self.assertGreater(result["overall"]["leak_errors"], 0)
        wrong_result = next(item for item in result["results"] if item["sample"] == wrong_sample)
        missing_result = next(item for item in result["results"] if item["sample"] == missing_sample)
        self.assertGreater(wrong_result["metrics"]["leak_errors"], 0)
        wrong_errors = "\n".join(wrong_result["leak_errors"])
        self.assertIn(f"model excluded_sources do not include expected {wrong_sample}", wrong_errors)
        self.assertIn(f"known_sample:{wrong_sample}", wrong_errors)
        self.assertIn(f"label:{wrong_sample}", wrong_errors)
        self.assertIn(f"learned:{wrong_sample}", wrong_errors)
        self.assertEqual(0, missing_result["metrics"]["leak_errors"])


if __name__ == "__main__":
    unittest.main()
