from __future__ import annotations

import base64

from PIL import Image

from shogi_gazo_desktop._tools.serve_image_kif_ui import (
    compact_hand_recognition,
    decode_data_url,
    prepare_image_for_recognition,
    safe_upload_name,
)


def test_decode_data_url_accepts_png_payload() -> None:
    payload = base64.b64encode(b"png-bytes").decode("ascii")
    data, mime = decode_data_url(f"data:image/png;base64,{payload}")
    assert data == b"png-bytes"
    assert mime == "image/png"


def test_decode_data_url_rejects_unsupported_mime() -> None:
    payload = base64.b64encode(b"text").decode("ascii")
    try:
        decode_data_url(f"data:text/plain;base64,{payload}")
    except ValueError as exc:
        assert "unsupported image type" in str(exc)
    else:
        raise AssertionError("unsupported uploads must be rejected")


def test_safe_upload_name_keeps_supported_suffix_and_sanitizes_stem() -> None:
    name = safe_upload_name("将棋クエスト 01.png", "image/png")
    assert name.endswith("将棋クエスト_01.png")
    assert "\\" not in name
    assert "/" not in name


def test_safe_upload_name_can_embed_target_hint() -> None:
    name = safe_upload_name("clipboard.png", "image/png", "将棋ウォーズ_一文字")
    assert "将棋ウォーズ_一文字" in name
    assert name.endswith("clipboard.png")


def test_prepare_image_for_recognition_upscales_small_image(tmp_path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (300, 500), (200, 150, 80)).save(image_path)
    prepared = prepare_image_for_recognition(image_path)
    assert prepared != image_path
    with Image.open(prepared) as image:
        assert image.width >= 900
        assert image.height >= 1500


def test_compact_hand_recognition_preserves_rect_and_candidate() -> None:
    report = {
        "hand_recognition": {
            "areas": [{"owner": "black", "side": "bottom", "rect": [1, 2, 3, 4], "confidence": 0.9}],
            "pieces": [
                {
                    "owner": "black",
                    "piece": "FU",
                    "count": 2,
                    "count_source": "digit",
                    "confidence": 0.8,
                    "rects": [[10, 20, 30, 40]],
                    "candidate_sets": [
                        {
                            "rect": [10, 20, 30, 40],
                            "candidates": [{"color": "black", "piece": "FU", "score": 0.8}],
                        }
                    ],
                }
            ],
        }
    }
    compact = compact_hand_recognition(report)
    assert compact["areas"][0]["rect"] == [1.0, 2.0, 3.0, 4.0]
    assert compact["pieces"][0]["identity"] == "black:FU"
    assert compact["pieces"][0]["candidateSets"][0]["candidates"][0]["identity"] == "black:FU"
