from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

from position_label_utils import HAND_PIECES, find_label_path, load_position_label


PIECE_JP = {
    "HI": "飛",
    "KA": "角",
    "KI": "金",
    "GI": "銀",
    "KE": "桂",
    "KY": "香",
    "FU": "歩",
}
OWNER_JP = {"black": "先手", "white": "後手"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_paths(analysis_dir: Path) -> list[Path]:
    return sorted(analysis_dir.glob("*/*/*/piece_report.json"))


def report_group(analysis_dir: Path, report_path: Path) -> tuple[str, str, str]:
    relative = report_path.relative_to(analysis_dir)
    return relative.parts[0], relative.parts[1], relative.parts[2]


def empty_hands() -> dict[str, dict[str, int]]:
    return {owner: {piece: 0 for piece in HAND_PIECES} for owner in ("black", "white")}


def hand_counts(value: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    hands = empty_hands()
    if not isinstance(value, dict):
        return hands
    for owner in ("black", "white"):
        source = value.get(owner) or {}
        for piece in HAND_PIECES:
            hands[owner][piece] = int(source.get(piece) or 0)
    return hands


def crop_rect(image: Image.Image, rect: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(tuple(int(value) for value in rect)).save(out_path)


def top_candidates_text(candidates: list[dict[str, Any]], limit: int = 3) -> str:
    parts = []
    for candidate in candidates[:limit]:
        owner = OWNER_JP.get(candidate.get("color"), str(candidate.get("color") or ""))
        piece = PIECE_JP.get(candidate.get("piece"), str(candidate.get("piece") or ""))
        parts.append(f"{owner}{piece}:{candidate.get('score')}")
    return " / ".join(parts)


def missing_expected_pieces(
    expected: dict[str, dict[str, int]],
    actual: dict[str, dict[str, int]],
    owner: str,
) -> list[str]:
    return [
        piece
        for piece in HAND_PIECES
        if int(expected[owner][piece]) > int(actual[owner][piece])
    ]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    crop_root = args.out / "crops"

    for report_path in report_paths(args.analysis_dir):
        app, piece_style, sample = report_group(args.analysis_dir, report_path)
        report = read_json(report_path)
        label_path = find_label_path(args.labels_dir, sample)
        if not label_path.exists():
            skipped.append({"sample": sample, "reason": "label_missing", "report": str(report_path)})
            continue
        label = load_position_label(label_path, require_hands=False)
        expected = hand_counts(label.get("hands"))
        actual = hand_counts(report.get("hands"))
        image_path = Path(report.get("image") or "")
        if not image_path.exists():
            skipped.append({"sample": sample, "reason": "image_missing", "image": str(image_path)})
            continue
        image = Image.open(image_path).convert("RGB")
        hand_report = report.get("hand_recognition") or {}

        for entry_index, entry in enumerate(hand_report.get("pieces") or [], start=1):
            owner = str(entry.get("owner") or "")
            piece = str(entry.get("piece") or "")
            if owner not in {"black", "white"} or piece not in HAND_PIECES:
                continue
            status = "trusted" if expected[owner][piece] == actual[owner][piece] and actual[owner][piece] > 0 else "needs_review"
            for rect_index, rect in enumerate(entry.get("rects") or [], start=1):
                crop_path = crop_root / app / piece_style / status / owner / piece / f"{sample}_{entry_index:02d}_{rect_index:02d}.png"
                crop_rect(image, rect, crop_path)
                rows.append(
                    {
                        "app": app,
                        "piece_style": piece_style,
                        "sample": sample,
                        "source": "recognized",
                        "status": status,
                        "owner": owner,
                        "owner_jp": OWNER_JP.get(owner, owner),
                        "piece": piece,
                        "piece_jp": PIECE_JP.get(piece, piece),
                        "expected": expected[owner][piece],
                        "actual": actual[owner][piece],
                        "count": entry.get("count"),
                        "count_source": entry.get("count_source"),
                        "confidence": entry.get("confidence"),
                        "rect": json.dumps(rect, ensure_ascii=False),
                        "crop": str(crop_path),
                        "best_piece": "",
                        "top3": top_candidates_text(entry.get("candidates") or []),
                    }
                )

        for unknown_index, proposal in enumerate(hand_report.get("unknown") or [], start=1):
            owner = str(proposal.get("owner") or "")
            rect = proposal.get("rect")
            if owner not in {"black", "white"} or not rect:
                continue
            missing = missing_expected_pieces(expected, actual, owner)
            best_piece = str(proposal.get("best_piece") or "")
            suggested_piece = best_piece if best_piece in missing else (missing[0] if len(missing) == 1 else "")
            status = "suggested_from_missing" if suggested_piece else "needs_review"
            crop_path = crop_root / app / piece_style / status / owner / (suggested_piece or "unknown") / f"{sample}_unknown_{unknown_index:02d}.png"
            crop_rect(image, rect, crop_path)
            rows.append(
                {
                    "app": app,
                    "piece_style": piece_style,
                    "sample": sample,
                    "source": "unknown_proposal",
                    "status": status,
                    "owner": owner,
                    "owner_jp": OWNER_JP.get(owner, owner),
                    "piece": suggested_piece,
                    "piece_jp": PIECE_JP.get(suggested_piece, suggested_piece),
                    "expected": expected[owner].get(suggested_piece, "") if suggested_piece else "",
                    "actual": actual[owner].get(suggested_piece, "") if suggested_piece else "",
                    "count": 1,
                    "count_source": "proposal",
                    "confidence": proposal.get("confidence"),
                    "rect": json.dumps(rect, ensure_ascii=False),
                    "crop": str(crop_path),
                    "best_piece": best_piece,
                    "top3": top_candidates_text(proposal.get("candidates") or []),
                }
            )

    fieldnames = [
        "app",
        "piece_style",
        "sample",
        "source",
        "status",
        "owner",
        "owner_jp",
        "piece",
        "piece_jp",
        "expected",
        "actual",
        "count",
        "count_source",
        "confidence",
        "rect",
        "crop",
        "best_piece",
        "top3",
    ]
    write_csv(args.out / "hand_training_manifest.csv", rows, fieldnames)
    write_csv(args.out / "skipped.csv", skipped, ["sample", "reason", "report", "image"])
    summary = {
        "analysis_dir": str(args.analysis_dir),
        "labels_dir": str(args.labels_dir),
        "rows": len(rows),
        "trusted_rows": sum(row["status"] == "trusted" for row in rows),
        "needs_review_rows": sum(row["status"] == "needs_review" for row in rows),
        "suggested_from_missing_rows": sum(row["status"] == "suggested_from_missing" for row in rows),
        "skipped": len(skipped),
        "manifest": str(args.out / "hand_training_manifest.csv"),
        "crop_root": str(crop_root),
    }
    write_json(args.out / "hand_training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="持ち駒専用分類器のためのクロップ画像と教師データ下書きを作成します。")
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--labels-dir", type=Path, default=Path("tools/samples/labels/boards_by_app_piece_style"))
    parser.add_argument("--out", type=Path, default=Path("tools/out/hand_training_dataset"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
