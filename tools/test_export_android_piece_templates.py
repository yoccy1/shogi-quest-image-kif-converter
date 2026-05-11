from __future__ import annotations

import unittest
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from export_android_piece_templates import (  # noqa: E402
    CURATED_PIECE_TEMPLATE_ALLOWLIST,
    CURATED_PIECE_TEMPLATE_BLOCKLIST,
    mask_to_hex,
    red_mask_stats,
    template_selection_key,
)


class AndroidTemplateExportTest(unittest.TestCase):
    def test_mask_to_hex_pads_to_nibble(self) -> None:
        self.assertEqual("8", mask_to_hex(bytes([1])))
        self.assertEqual("a", mask_to_hex(bytes([1, 0, 1])))

    def test_red_mask_stats_reports_center_and_edge_shares(self) -> None:
        width = 48
        height = 56
        mask = bytearray(width * height)
        mask[height // 2 * width + width // 2] = 1
        mask[1] = 1

        stats = red_mask_stats(bytes(mask))

        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertIn("redMask", stats)
        self.assertGreater(float(stats["centralRedShare"]), 0.0)
        self.assertGreater(float(stats["edgeRedShare"]), 0.0)
        self.assertGreater(float(stats["redCenterX"]), 0.0)
        self.assertGreater(float(stats["redCenterY"]), 0.0)

    def test_red_mask_stats_omits_empty_mask(self) -> None:
        self.assertIsNone(red_mask_stats(bytes(48 * 56)))

    def test_curated_piyo_chick_template_keys_are_exact(self) -> None:
        blocked = template_selection_key(
            "piyo_chick",
            "BLACK",
            "KA",
            "ぴよ将棋_ひよこ駒_通常_05",
            8,
            2,
        )
        allowed = template_selection_key(
            "piyo_chick",
            "BLACK",
            "GI",
            "ぴよ将棋_ひよこ駒_通常_09",
            8,
            2,
        )
        allowed_black_gi_position = template_selection_key(
            "piyo_chick",
            "BLACK",
            "GI",
            "ぴよ将棋_ひよこ駒_通常_02",
            6,
            6,
        )
        allowed_black_gi_probe = template_selection_key(
            "piyo_chick",
            "BLACK",
            "GI",
            "ぴよ将棋_ひよこ駒_通常_03",
            8,
            4,
        )
        allowed_white_gi = template_selection_key(
            "piyo_chick",
            "WHITE",
            "GI",
            "ぴよ将棋_ひよこ駒_通常_03",
            6,
            5,
        )

        self.assertIn(blocked, CURATED_PIECE_TEMPLATE_BLOCKLIST)
        self.assertIn(allowed, CURATED_PIECE_TEMPLATE_ALLOWLIST)
        self.assertIn(allowed_black_gi_position, CURATED_PIECE_TEMPLATE_ALLOWLIST)
        self.assertIn(allowed_black_gi_probe, CURATED_PIECE_TEMPLATE_ALLOWLIST)
        self.assertIn(allowed_white_gi, CURATED_PIECE_TEMPLATE_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
