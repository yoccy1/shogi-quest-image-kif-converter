from __future__ import annotations

from pathlib import Path

from shogi_gazo_desktop import cli
from shogi_gazo_desktop.recognition import should_normalize_hands_from_inventory


def test_validate_limit_accepts_positive_and_empty() -> None:
    cli.validate_limit(None)
    cli.validate_limit(1)


def test_validate_limit_rejects_zero_and_negative() -> None:
    for value in (0, -1):
        try:
            cli.validate_limit(value)
        except ValueError as exc:
            assert "--limit" in str(exc)
        else:
            raise AssertionError("non-positive limits must be rejected")


def test_filter_named_paths_by_stem() -> None:
    paths = [Path("a/sample_one.png"), Path("b/sample_two.jpg")]
    assert cli.filter_named_paths(paths, ["sample_two"]) == [Path("b/sample_two.jpg")]


def test_filter_report_paths_by_parent_name() -> None:
    paths = [Path("run/sample_one/piece_report.json"), Path("run/sample_two/piece_report.json")]
    assert cli.filter_report_paths(paths, ["sample_one"]) == [Path("run/sample_one/piece_report.json")]


def test_hand_inventory_normalization_skips_unresolved_or_unknown_board() -> None:
    assert not should_normalize_hands_from_inventory(
        {"constraint_postprocess": {"unresolved": [{"reason": "dead_end"}]}, "cells": []}
    )
    assert should_normalize_hands_from_inventory({"constraint_postprocess": {"unresolved": [{"reason": "nifu"}]}, "cells": []})
    assert not should_normalize_hands_from_inventory({"cells": [{"state": "unknown"}]})
    assert should_normalize_hands_from_inventory({"constraint_postprocess": {"unresolved": []}, "cells": [{"state": "empty"}]})
