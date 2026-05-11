from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from position_label_utils import load_position_label, resolve_label_image_path, square_name


FIGURE_TO_PIECE = {
    "PAWN": "FU",
    "BISHOP": "KA",
    "ROOK": "HI",
    "LANCE": "KY",
    "KNIGHT": "KE",
    "SILVER": "GI",
    "GOLD": "KI",
    "KING": "OU",
    "PAWN_PROM": "TO",
    "LANCE_PROM": "NY",
    "KNIGHT_PROM": "NK",
    "SILVER_PROM": "NG",
    "BISHOP_PROM": "UM",
    "ROOK_PROM": "RY",
    "EMPTY": None,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_shogivision_root() -> Path:
    return repo_root() / "ShogiVision-master" / "ShogiVision-master"


def add_shogivision_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path))


def label_context(label_path: Path, labels_dir: Path) -> tuple[str | None, str | None, str]:
    rel = label_path.relative_to(labels_dir)
    sample = label_path.stem
    if len(rel.parts) >= 3:
        return rel.parts[0], rel.parts[1], sample
    if len(rel.parts) >= 2:
        return rel.parts[0], None, sample
    return None, None, sample


def collect_label_paths(labels_dir: Path, app: str | None, style: str | None, limit: int | None) -> list[Path]:
    paths = sorted(labels_dir.rglob("*.json"))
    selected = []
    for path in paths:
        label_app, label_style, _ = label_context(path, labels_dir)
        if app and label_app != app:
            continue
        if style and label_style != style:
            continue
        selected.append(path)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def color_from_direction(direction_name: str, orientation: str) -> str | None:
    black_bottom = orientation != "white_bottom"
    if direction_name == "UP":
        return "black" if black_bottom else "white"
    if direction_name == "DOWN":
        return "white" if black_bottom else "black"
    return None


def identity_from_names(figure_name: str, direction_name: str, orientation: str) -> str:
    piece = FIGURE_TO_PIECE.get(figure_name)
    if piece is None:
        return "empty"
    color = color_from_direction(direction_name, orientation)
    if color is None:
        return f"none:{piece}"
    return f"{color}:{piece}"


def split_identity(value: str) -> tuple[str | None, str | None, str]:
    if value == "empty":
        return None, None, "empty"
    if ":" not in value:
        return None, None, "unknown"
    color, piece = value.split(":", 1)
    if color not in {"black", "white"}:
        return None, piece, "unknown"
    return color, piece, "piece"


def expected_identity(cell: dict[str, Any]) -> str:
    if cell["state"] == "empty":
        return "empty"
    if cell["state"] == "piece":
        return f"{cell['color']}:{cell['piece']}"
    return "unknown"


def top_identity_candidates(
    figure_probs: np.ndarray,
    direction_probs: np.ndarray,
    figure_categories: list[Any],
    direction_categories: list[Any],
    orientation: str,
    top_n: int = 3,
) -> list[tuple[str, float, str, str, float, float]]:
    candidates = []
    for fig_index, figure in enumerate(figure_categories):
        figure_name = figure.name
        for dir_index, direction in enumerate(direction_categories):
            direction_name = direction.name
            identity = identity_from_names(figure_name, direction_name, orientation)
            score = float(figure_probs[fig_index]) * float(direction_probs[dir_index])
            candidates.append(
                (
                    identity,
                    score,
                    figure_name,
                    direction_name,
                    float(figure_probs[fig_index]),
                    float(direction_probs[dir_index]),
                )
            )
    best_by_identity: dict[str, tuple[str, float, str, str, float, float]] = {}
    for candidate in candidates:
        current = best_by_identity.get(candidate[0])
        if current is None or candidate[1] > current[1]:
            best_by_identity[candidate[0]] = candidate
    return sorted(best_by_identity.values(), key=lambda item: item[1], reverse=True)[:top_n]


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    shogivision_root = args.shogivision_root.resolve()
    add_shogivision_path(shogivision_root)

    import torch  # noqa: F401  # Import before PyQt/Ultralytics side effects on Windows.
    import cv2
    from Elements.BoardSplitter import BoardSplitter
    from Elements.CornerDetectors.HardcodedCornerDetector import HardcodedCornerDetector
    from Elements.CornerDetectors.YOLOSegmentationCornerDetector import YOLOSegmentationCornerDetector
    from Elements.ImageGetters import Photo
    from Elements.Recognizers.RecognizerONNX import RecognizerONNX
    from ShogiNeuralNetwork import preprocessing
    from ShogiNeuralNetwork.data_info import CATEGORIES_DIRECTION, CATEGORIES_FIGURE_TYPE
    from config import paths as shogi_paths

    labels_dir = args.labels_dir.resolve()
    screenshots_dir = args.screenshots_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer = RecognizerONNX(str(shogi_paths.MIXED_MODEL_ONNX_PATH))
    corner_detector = YOLOSegmentationCornerDetector()

    cell_rows = []
    board_rows = []
    metrics = Counter()
    group_metrics: dict[tuple[str | None, str | None], Counter[str]] = {}
    error_rows = []
    label_paths = collect_label_paths(labels_dir, args.app, args.style, args.limit)
    started = time.perf_counter()

    for index, label_path in enumerate(label_paths, start=1):
        label = load_position_label(label_path)
        app, style, sample = label_context(label_path, labels_dir)
        image_path = resolve_label_image_path(label_path, label, screenshots_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")

        board_start = time.perf_counter()
        corners = corner_detector.get_corners(image)
        splitter = BoardSplitter(Photo(image), HardcodedCornerDetector(corners))
        cells = splitter.get_board_cells()
        prepared = preprocessing.prepare_cells_imgs(cells, recognizer.image_mode, recognizer.cell_img_size)
        output_names = [output.name for output in recognizer.model.get_outputs()]
        figure_predictions, direction_predictions = recognizer.model.run(
            output_names,
            {recognizer.model.get_inputs()[0].name: prepared},
        )
        elapsed = time.perf_counter() - board_start

        board_metrics = Counter()
        label_cells = {(cell["row"], cell["col"]): cell for cell in label["cells"]}
        for row in range(1, 10):
            for col in range(1, 10):
                flat_index = (row - 1) * 9 + (col - 1)
                expected = label_cells[(row, col)]
                if expected["state"] == "unknown":
                    metrics["ignored_unknown"] += 1
                    board_metrics["ignored_unknown"] += 1
                    continue

                candidates = top_identity_candidates(
                    figure_predictions[flat_index],
                    direction_predictions[flat_index],
                    CATEGORIES_FIGURE_TYPE,
                    CATEGORIES_DIRECTION,
                    label["orientation"],
                    top_n=3,
                )
                top1 = candidates[0]
                top1_key = top1[0]
                expected_key = expected_identity(expected)
                pred_color, pred_piece, pred_state = split_identity(top1_key)
                correct_identity = top1_key == expected_key
                top3_keys = [candidate[0] for candidate in candidates]
                expected_in_top3 = expected_key in top3_keys

                metrics["total"] += 1
                board_metrics["total"] += 1
                if expected["state"] == "empty":
                    metrics["true_empty"] += 1
                    board_metrics["true_empty"] += 1
                    if pred_state == "empty":
                        metrics["empty_correct"] += 1
                        board_metrics["empty_correct"] += 1
                    else:
                        metrics["false_piece_on_empty"] += 1
                        board_metrics["false_piece_on_empty"] += 1
                elif expected["state"] == "piece":
                    metrics["true_piece"] += 1
                    board_metrics["true_piece"] += 1
                    if pred_state == "piece":
                        metrics["piece_presence_correct"] += 1
                        board_metrics["piece_presence_correct"] += 1
                    else:
                        metrics["false_empty_on_piece"] += 1
                        board_metrics["false_empty_on_piece"] += 1
                    if correct_identity:
                        metrics["top1_identity_correct"] += 1
                        board_metrics["top1_identity_correct"] += 1
                    if expected_in_top3:
                        metrics["top3_contains_identity"] += 1
                        board_metrics["top3_contains_identity"] += 1

                if not correct_identity:
                    metrics["errors"] += 1
                    board_metrics["errors"] += 1
                    error_rows.append(
                        {
                            "app": app,
                            "style": style,
                            "sample": sample,
                            "square": square_name(row, col),
                            "row": row,
                            "col": col,
                            "expected": expected_key,
                            "actual": top1_key,
                            "top1_score": round(top1[1], 6),
                            "top3": "|".join(top3_keys),
                        }
                    )

                cell_rows.append(
                    {
                        "app": app,
                        "style": style,
                        "sample": sample,
                        "label_path": str(label_path),
                        "image_path": str(image_path),
                        "orientation": label["orientation"],
                        "square": square_name(row, col),
                        "row": row,
                        "col": col,
                        "expected": expected_key,
                        "actual": top1_key,
                        "actual_color": pred_color,
                        "actual_piece": pred_piece,
                        "actual_state": pred_state,
                        "top1_score": round(top1[1], 6),
                        "top1_figure": top1[2],
                        "top1_direction": top1[3],
                        "top1_figure_prob": round(top1[4], 6),
                        "top1_direction_prob": round(top1[5], 6),
                        "top3": "|".join(top3_keys),
                        "expected_in_top3": expected_in_top3,
                        "correct_identity": correct_identity,
                    }
                )

        board_rows.append(
            {
                "index": index,
                "app": app,
                "style": style,
                "sample": sample,
                "label_path": str(label_path),
                "image_path": str(image_path),
                "elapsed_seconds": round(elapsed, 4),
                "total": board_metrics["total"],
                "true_piece": board_metrics["true_piece"],
                "true_empty": board_metrics["true_empty"],
                "errors": board_metrics["errors"],
                "top1_identity_accuracy": rate(board_metrics["top1_identity_correct"], board_metrics["true_piece"]),
                "top3_identity_accuracy": rate(board_metrics["top3_contains_identity"], board_metrics["true_piece"]),
                "empty_accuracy": rate(board_metrics["empty_correct"], board_metrics["true_empty"]),
                "piece_presence_accuracy": rate(board_metrics["piece_presence_correct"], board_metrics["true_piece"]),
                "corners": json.dumps([[int(x), int(y)] for x, y in corners], ensure_ascii=False),
            }
        )
        group_key = (app, style)
        if group_key not in group_metrics:
            group_metrics[group_key] = Counter()
        for metric_key, metric_value in board_metrics.items():
            group_metrics[group_key][metric_key] += metric_value
        group_metrics[group_key]["samples"] += 1

    write_csv(out_dir / "shogivision_cell_predictions.csv", cell_rows)
    write_csv(out_dir / "shogivision_board_summary.csv", board_rows)
    write_csv(out_dir / "shogivision_errors.csv", error_rows)
    group_rows = []
    for (app, style), group in sorted(group_metrics.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")):
        group_rows.append(
            {
                "app": app,
                "style": style,
                "samples": group["samples"],
                "total": group["total"],
                "true_piece": group["true_piece"],
                "true_empty": group["true_empty"],
                "errors": group["errors"],
                "false_piece_on_empty": group["false_piece_on_empty"],
                "false_empty_on_piece": group["false_empty_on_piece"],
                "top1_identity_accuracy": rate(group["top1_identity_correct"], group["true_piece"]),
                "top3_identity_accuracy": rate(group["top3_contains_identity"], group["true_piece"]),
                "empty_accuracy": rate(group["empty_correct"], group["true_empty"]),
                "piece_presence_accuracy": rate(group["piece_presence_correct"], group["true_piece"]),
            }
        )
    write_csv(out_dir / "shogivision_group_summary.csv", group_rows)

    summary = {
        "labels_dir": str(labels_dir),
        "screenshots_dir": str(screenshots_dir),
        "shogivision_root": str(shogivision_root),
        "out_dir": str(out_dir),
        "sample_count": len(label_paths),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "metrics": {
            "total": metrics["total"],
            "true_piece": metrics["true_piece"],
            "true_empty": metrics["true_empty"],
            "errors": metrics["errors"],
            "false_piece_on_empty": metrics["false_piece_on_empty"],
            "false_empty_on_piece": metrics["false_empty_on_piece"],
            "top1_identity_correct": metrics["top1_identity_correct"],
            "top3_contains_identity": metrics["top3_contains_identity"],
            "empty_correct": metrics["empty_correct"],
            "piece_presence_correct": metrics["piece_presence_correct"],
            "top1_identity_accuracy": rate(metrics["top1_identity_correct"], metrics["true_piece"]),
            "top3_identity_accuracy": rate(metrics["top3_contains_identity"], metrics["true_piece"]),
            "empty_accuracy": rate(metrics["empty_correct"], metrics["true_empty"]),
            "piece_presence_accuracy": rate(metrics["piece_presence_correct"], metrics["true_piece"]),
        },
        "outputs": {
            "cell_predictions": str(out_dir / "shogivision_cell_predictions.csv"),
            "board_summary": str(out_dir / "shogivision_board_summary.csv"),
            "errors": str(out_dir / "shogivision_errors.csv"),
            "group_summary": str(out_dir / "shogivision_group_summary.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run ShogiVision on existing labelled screenshots.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=root / "tools" / "samples" / "labels" / "boards_by_app_piece_style",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        default=root / "tools" / "samples" / "screenshots_by_app_piece_style",
    )
    parser.add_argument("--shogivision-root", type=Path, default=default_shogivision_root())
    parser.add_argument("--out-dir", type=Path, default=root / "tools" / "out" / "shogivision_offline_probe")
    parser.add_argument("--app", type=str)
    parser.add_argument("--style", type=str)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    summary = run_probe(parse_args())
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote: {summary['out_dir']}")


if __name__ == "__main__":
    main()
