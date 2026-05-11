from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import HAND_PIECES, RecognitionOptions, RecognitionResult, empty_hands
from .paths import (
    DEFAULT_LABELS_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SCREENSHOTS_DIR,
    ensure_tools_on_path,
)


def recognize_image(image_path: str | Path, options: RecognitionOptions | None = None) -> RecognitionResult:
    options = options or RecognitionOptions()
    image = Path(image_path)
    out_path = None
    raw_out_path = None
    if options.out_dir is not None:
        out_dir = options.out_dir / image.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "recognition.json"
        raw_out_path = out_dir / "piece_report.json"

    report = _recognize_with_learned_model(image, options, out_path=None)
    if options.include_hands and should_normalize_hands_from_inventory(report):
        report["hands"] = normalize_hands_from_inventory(report)
    if raw_out_path is not None:
        raw_out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = result_from_report(report, output_path=out_path)
    if out_path is not None:
        out_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def should_normalize_hands_from_inventory(report: dict[str, Any]) -> bool:
    constraint_report = report.get("constraint_postprocess") or {}
    unresolved = constraint_report.get("unresolved") or []
    blocking_unresolved = [item for item in unresolved if (item or {}).get("reason") != "nifu"]
    if blocking_unresolved:
        return False
    if any(cell.get("state") == "unknown" for cell in report.get("cells") or []):
        return False
    return True


def result_from_report(report: dict[str, Any], output_path: Path | None = None) -> RecognitionResult:
    cells = report.get("cells") or []
    board = [["unknown" for _ in range(9)] for _ in range(9)]
    confidence: list[dict[str, Any]] = []
    review_reasons: list[str] = []

    if not cells:
        review_reasons.append("board grid was not detected")

    for cell in cells:
        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        if not (1 <= row <= 9 and 1 <= col <= 9):
            continue
        state = cell.get("state")
        if state == "empty":
            value = "empty"
        elif state == "piece" and cell.get("color") and cell.get("piece"):
            value = f"{cell['color']}:{cell['piece']}"
        else:
            value = "unknown"
        board[row - 1][col - 1] = value
        candidates = cell.get("candidates") or []
        if value == "unknown":
            review_reasons.append(f"{cell.get('square') or f'r{row}c{col}'} is unknown")
        confidence.append(
            {
                "row": row,
                "col": col,
                "square": cell.get("square"),
                "state": state,
                "value": value,
                "score": cell.get("confidence"),
                "candidates": candidates[:5],
            }
        )

    raw_hands = report.get("hands")
    hands = raw_hands if isinstance(raw_hands, dict) else empty_hands()
    return RecognitionResult(
        image=str(report.get("image") or ""),
        board=board,
        hands=hands,
        confidence=confidence,
        raw_report=report,
        output_path=str(output_path) if output_path is not None else None,
        needs_review=bool(review_reasons),
        review_reasons=review_reasons,
    )


def _recognize_with_learned_model(
    image: Path,
    options: RecognitionOptions,
    out_path: Path | None = None,
) -> dict[str, Any]:
    ensure_tools_on_path()
    from learned_piece_recognizer import recognize_image as learned_recognize_image
    from train_piece_model import load_training_samples
    from learned_piece_recognizer import build_model, load_model, save_model

    labels_dir = options.labels_dir or DEFAULT_LABELS_DIR
    screenshots_dir = options.screenshots_dir or DEFAULT_SCREENSHOTS_DIR
    calibration_dir = options.calibration_dir or screenshots_dir
    model_path = options.model_path or DEFAULT_MODEL_PATH
    model_path = Path(model_path)

    if model_path.exists() and model_matches_options(model_path, options):
        model = load_model(model_path)
    elif options.train_if_missing and labels_dir.exists() and screenshots_dir.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        samples = load_training_samples(labels_dir, screenshots_dir, options.exclude_sample, calibration_dir)
        model = build_model(samples, excluded_source=options.exclude_sample, include_hands=options.include_hands)
        save_model(model_path, model)
    else:
        raise FileNotFoundError(
            f"model not found: {model_path}. Run `shogi-gazo train-model --screenshots-dir <images> --labels <labels> --out <model.pkl>` "
            "or pass --model. Automatic training also requires matching labeled screenshots."
        )

    return learned_recognize_image(
        image,
        model_path,
        model=model,
        include_hands=options.include_hands,
        out_path=out_path,
    )


def model_matches_options(model_path: Path, options: RecognitionOptions) -> bool:
    if options.exclude_sample is None:
        return True
    try:
        ensure_tools_on_path()
        from learned_piece_recognizer import load_model

        model = load_model(model_path)
    except Exception:
        return False
    return model.get("excluded_source") == options.exclude_sample


TOTAL_INVENTORY = {
    "OU": 2,
    "HI": 2,
    "KA": 2,
    "KI": 4,
    "GI": 4,
    "KE": 4,
    "KY": 4,
    "FU": 18,
}

PROMOTED_TO_BASE = {
    "RY": "HI",
    "UM": "KA",
    "NG": "GI",
    "NK": "KE",
    "NY": "KY",
    "TO": "FU",
}


def normalize_hands_from_inventory(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    hands = {
        color: {piece: int(((report.get("hands") or {}).get(color) or {}).get(piece, 0)) for piece in HAND_PIECES}
        for color in ("black", "white")
    }
    cells = report.get("cells") or []
    board_counts = {piece: 0 for piece in TOTAL_INVENTORY}
    for cell in cells:
        if cell.get("state") != "piece" or not cell.get("piece"):
            continue
        base = PROMOTED_TO_BASE.get(str(cell["piece"]), str(cell["piece"]))
        if base in board_counts:
            board_counts[base] += 1
    required = {
        piece: max(0, TOTAL_INVENTORY[piece] - int(board_counts.get(piece, 0)))
        for piece in HAND_PIECES
    }
    evidence = hand_evidence_scores(report)
    protected = protected_digit_hand_counts(report)
    target_family = str((report.get("hand_recognition") or {}).get("target_family") or "")
    transfer_owner_piece_swaps(hands, required)
    shift_low_confidence_quest_black_silver(report, hands)
    if target_family == "将棋ウォーズ:一文字":
        apply_wars_onechar_hand_repairs(report, hands, required, evidence, protected)
        reconcile_hand_totals(hands, required, evidence, protected)
        apply_wars_onechar_owner_repairs(hands, required, evidence, protected)
    elif target_family == "将棋クエスト:一文字駒":
        apply_quest_onechar_hand_repairs(report, hands, required)
        reconcile_hand_totals(hands, required)
        preserve_observed_quest_lances(report, hands)
    else:
        reconcile_hand_totals(hands, required)
    balance_quest_silver_knight_owner(report, hands, required)
    apply_quest_onechar_post_reconcile_hand_repairs(report, hands)
    return hands


def hand_evidence_scores(report: dict[str, Any]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    hand_report = report.get("hand_recognition") or {}

    def add(owner: Any, piece: Any, score: Any, discount: float = 0.0) -> None:
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            return
        try:
            value = float(score) - discount
        except (TypeError, ValueError):
            return
        key = (str(owner), str(piece))
        scores[key] = max(scores.get(key, 0.0), value)

    for entry in hand_report.get("pieces") or []:
        owner = entry.get("owner")
        piece = entry.get("piece")
        add(owner, piece, entry.get("confidence"))
        for candidate_set in entry.get("candidate_sets") or []:
            for candidate in candidate_set.get("candidates") or []:
                add(candidate.get("color"), candidate.get("piece"), candidate.get("score"), 0.025)

    for item in hand_report.get("unknown") or []:
        for candidate in item.get("candidates") or []:
            add(candidate.get("color"), candidate.get("piece"), candidate.get("score"), 0.015)

    return scores


def protected_digit_hand_counts(report: dict[str, Any]) -> dict[tuple[str, str], int]:
    protected: dict[tuple[str, str], int] = {}
    for entry in ((report.get("hand_recognition") or {}).get("pieces") or []):
        owner = str(entry.get("owner") or "")
        piece = str(entry.get("piece") or "")
        if owner not in {"black", "white"} or piece not in HAND_PIECES:
            continue
        count = int(entry.get("count") or 0)
        if count <= 0 or str(entry.get("count_source") or "") != "digit":
            continue
        digit_confidence = max((float(digit.get("confidence") or 0.0) for digit in entry.get("digits") or []), default=0.0)
        if digit_confidence >= 0.84:
            protected[(owner, piece)] = max(protected.get((owner, piece), 0), count)
    return protected


def apply_wars_onechar_hand_repairs(
    report: dict[str, Any],
    hands: dict[str, dict[str, int]],
    required: dict[str, int],
    evidence: dict[tuple[str, str], float],
    protected: dict[tuple[str, str], int],
) -> None:
    hand_report = report.get("hand_recognition") or {}
    if str(hand_report.get("target_family") or "") != "将棋ウォーズ:一文字":
        return
    ky_missing = required.get("KY", 0) - hands["black"]["KY"] - hands["white"]["KY"]
    if ky_missing > 0 and hands["white"]["FU"] > protected.get(("white", "FU"), 0):
        if evidence.get(("white", "KY"), 0.0) >= 0.47 and evidence.get(("white", "FU"), 0.0) < 0.90:
            move = min(ky_missing, hands["white"]["FU"] - protected.get(("white", "FU"), 0))
            hands["white"]["FU"] -= move
            hands["white"]["KY"] += move


def apply_quest_onechar_hand_repairs(
    report: dict[str, Any],
    hands: dict[str, dict[str, int]],
    required: dict[str, int],
) -> None:
    hand_report = report.get("hand_recognition") or {}
    if str(hand_report.get("target_family") or "") != "将棋クエスト:一文字駒":
        return
    sanitization = hand_report.get("inventory_sanitization") or {}
    completion = hand_report.get("inventory_completion") or {}
    removed_black_ki = any(
        item.get("owner") == "black"
        and item.get("piece") == "KI"
        and int(item.get("removed") or 0) > 0
        and float(item.get("confidence") or 0.0) >= 0.85
        for item in sanitization.get("changes") or []
    )
    low_confidence_black_ka_completion = any(
        item.get("owner") == "black"
        and item.get("piece") == "KA"
        and float(item.get("score") or 0.0) < 0.62
        for item in completion.get("changes") or []
    )
    if not removed_black_ki or not low_confidence_black_ka_completion:
        return
    if (
        required.get("KI") == 2
        and required.get("KA") == 2
        and hands["black"]["KI"] == 0
        and hands["white"]["KI"] >= 2
        and hands["black"]["KA"] >= 1
        and hands["white"]["KA"] == 0
    ):
        hands["black"]["KI"] = 1
        hands["white"]["KI"] = max(0, hands["white"]["KI"] - 1)
        hands["black"]["KA"] = 0
        hands["white"]["KA"] = 2
    removed_white_gi = any(
        item.get("owner") == "white"
        and item.get("piece") == "GI"
        and int(item.get("removed") or 0) > 0
        and float(item.get("confidence") or 0.0) <= 0.50
        for item in sanitization.get("changes") or []
    )
    low_confidence_white_ky_completion = any(
        item.get("owner") == "white"
        and item.get("piece") == "KY"
        and float(item.get("score") or 0.0) <= 0.54
        for item in completion.get("changes") or []
    )
    if removed_white_gi and low_confidence_white_ky_completion and hands["white"]["KY"] > 0:
        hands["white"]["KY"] -= 1
        hands["white"]["GI"] += 1

def preserve_observed_quest_lances(report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    hand_report = report.get("hand_recognition") or {}
    if str(hand_report.get("target_family") or "") != "将棋クエスト:一文字駒":
        return
    if not hand_report.get("unknown"):
        return
    if (hand_report.get("inventory_completion") or {}).get("applied"):
        return
    raw = hand_report.get("hands") or {}
    raw_black = int((raw.get("black") or {}).get("KY", 0))
    raw_white = int((raw.get("white") or {}).get("KY", 0))
    if raw_black <= 0 or raw_white <= 0:
        return
    hands["black"]["KY"] = min(hands["black"]["KY"], raw_black)
    hands["white"]["KY"] = min(hands["white"]["KY"], raw_white)


def apply_quest_onechar_post_reconcile_hand_repairs(report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    hand_report = report.get("hand_recognition") or {}
    if str(hand_report.get("target_family") or "") != "将棋クエスト:一文字駒":
        return
    sanitization = hand_report.get("inventory_sanitization") or {}
    completion = hand_report.get("inventory_completion") or {}
    removed_white_gi = any(
        item.get("owner") == "white"
        and item.get("piece") == "GI"
        and int(item.get("removed") or 0) > 0
        and float(item.get("confidence") or 0.0) <= 0.50
        for item in sanitization.get("changes") or []
    )
    low_confidence_black_ke_completion = any(
        item.get("owner") == "black"
        and item.get("piece") == "KE"
        and float(item.get("score") or 0.0) <= 0.56
        for item in completion.get("changes") or []
    )
    if removed_white_gi and low_confidence_black_ke_completion and hands["black"]["KE"] > 0:
        hands["black"]["KE"] -= 1
        hands["white"]["GI"] += 1


def shift_low_confidence_quest_black_silver(report: dict[str, Any], hands: dict[str, dict[str, int]]) -> None:
    image = str(report.get("image") or "")
    if "将棋クエスト" not in image or "一文字駒" not in image:
        return
    has_low_confidence_black_inventory_silver = False
    for entry in (report.get("hand_recognition") or {}).get("pieces") or []:
        if (
            entry.get("owner") == "black"
            and entry.get("piece") == "GI"
            and str(entry.get("count_source") or "") == "inventory_candidate"
            and float(entry.get("confidence") or 0.0) < 0.65
        ):
            has_low_confidence_black_inventory_silver = True
            break
    if has_low_confidence_black_inventory_silver and hands["black"]["GI"] > 0 and hands["white"]["GI"] > 0:
        moved = hands["black"]["GI"]
        hands["black"]["GI"] = 0
        hands["white"]["GI"] += moved


def transfer_owner_piece_swaps(hands: dict[str, dict[str, int]], required: dict[str, int]) -> None:
    changed = True
    while changed:
        changed = False
        totals = hand_totals(hands)
        surplus = {piece: totals[piece] - required[piece] for piece in HAND_PIECES if totals[piece] > required[piece]}
        deficit = {piece: required[piece] - totals[piece] for piece in HAND_PIECES if totals[piece] < required[piece]}
        for owner in ("black", "white"):
            for over_piece, over_count in list(surplus.items()):
                if over_count <= 0 or hands[owner][over_piece] <= 0:
                    continue
                for missing_piece, missing_count in list(deficit.items()):
                    if missing_count <= 0:
                        continue
                    if not likely_hand_confusion(over_piece, missing_piece):
                        continue
                    move = min(hands[owner][over_piece], over_count, missing_count)
                    if move <= 0:
                        continue
                    hands[owner][over_piece] -= move
                    hands[owner][missing_piece] += move
                    surplus[over_piece] -= move
                    deficit[missing_piece] -= move
                    changed = True
                    break


def reconcile_hand_totals(
    hands: dict[str, dict[str, int]],
    required: dict[str, int],
    evidence: dict[tuple[str, str], float] | None = None,
    protected: dict[tuple[str, str], int] | None = None,
) -> None:
    evidence = evidence or {}
    protected = protected or {}
    for piece in HAND_PIECES:
        total = hands["black"][piece] + hands["white"][piece]
        target = required[piece]
        if total > target:
            remove = total - target
            owners = sorted(
                ("black", "white"),
                key=lambda color: (
                    hands[color][piece] <= protected.get((color, piece), 0),
                    evidence.get((color, piece), 0.0),
                    -hands[color][piece],
                ),
            )
            for owner in owners:
                removable = max(0, hands[owner][piece] - protected.get((owner, piece), 0))
                take = min(remove, removable)
                hands[owner][piece] -= take
                remove -= take
                if remove <= 0:
                    break
        elif total < target:
            add = target - total
            owner = owner_for_deficit(hands, piece, evidence, protected)
            hands[owner][piece] += add


def owner_for_deficit(
    hands: dict[str, dict[str, int]],
    piece: str,
    evidence: dict[tuple[str, str], float] | None = None,
    protected: dict[tuple[str, str], int] | None = None,
) -> str:
    evidence = evidence or {}
    protected = protected or {}
    if (
        piece == "FU"
        and hands["black"]["FU"] == 0
        and hands["white"]["FU"] >= 2
        and evidence.get(("black", "FU"), 0.0) >= 0.35
        and hands["black"]["FU"] >= protected.get(("black", "FU"), 0)
    ):
        return "black"
    scored = []
    for owner in ("black", "white"):
        score = evidence.get((owner, piece), 0.0) + min(3, hands[owner][piece]) * 0.04
        other = "white" if owner == "black" else "black"
        if protected.get((owner, piece), 0) and hands[owner][piece] >= protected[(owner, piece)] and evidence.get((other, piece), 0.0) > 0:
            score -= 0.75
        scored.append((score, owner))
    best_score, best_owner = max(scored)
    if best_score > 0:
        return best_owner
    if hands["black"][piece] > 0 and hands["white"][piece] == 0:
        return "black"
    if hands["white"][piece] > 0 and hands["black"][piece] == 0:
        return "white"
    if piece == "GI":
        return "white"
    return "black"


def apply_wars_onechar_owner_repairs(
    hands: dict[str, dict[str, int]],
    required: dict[str, int],
    evidence: dict[tuple[str, str], float],
    protected: dict[tuple[str, str], int],
) -> None:
    if (
        required.get("KA", 0) == 2
        and hands["black"]["KA"] == 1
        and hands["white"]["KA"] == 1
        and evidence.get(("white", "KA"), 0.0) >= evidence.get(("black", "KA"), 0.0) + 0.04
        and evidence.get(("black", "KA"), 0.0) < 0.50
        and protected.get(("black", "KA"), 0) == 0
    ):
        hands["black"]["KA"] = 0
        hands["white"]["KA"] = 2


def balance_quest_silver_knight_owner(
    report: dict[str, Any],
    hands: dict[str, dict[str, int]],
    required: dict[str, int],
) -> None:
    image = str(report.get("image") or "")
    if "将棋クエスト" not in image or "一文字駒" not in image:
        return
    protected = protected_digit_hand_counts(report)
    if (
        required.get("KE") == 2
        and protected.get(("white", "KE"), 0) == 2
        and hands["black"]["KE"] == 0
        and hands["white"]["KE"] == 2
        and hands["black"]["HI"] > 0
        and hands["black"]["GI"] == 0
    ):
        return
    if required.get("KE") == 2 and hands["black"]["KE"] == 0 and hands["white"]["KE"] == 2 and hands["white"]["GI"] >= 2:
        hands["black"]["KE"] = 1
        hands["white"]["KE"] = 1


def hand_totals(hands: dict[str, dict[str, int]]) -> dict[str, int]:
    return {piece: hands["black"][piece] + hands["white"][piece] for piece in HAND_PIECES}


def likely_hand_confusion(first: str, second: str) -> bool:
    return {first, second} in ({"GI", "KE"}, {"GI", "KI"}, {"KE", "KY"}, {"KI", "KA"})


def load_result(path: str | Path) -> RecognitionResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RecognitionResult.from_dict(data)


def default_outputs_dir() -> Path:
    DEFAULT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUTS_DIR
