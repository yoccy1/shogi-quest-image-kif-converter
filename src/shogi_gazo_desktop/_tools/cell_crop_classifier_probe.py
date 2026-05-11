from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from detect_board_grid import GridDetection, detect_grid
from position_label_utils import find_label_path, identity, load_position_label, resolve_label_image_path, square_name
from recognize_board_pieces import (
    black_ink_mask,
    crop_calibration_cell,
    extract_hog_features,
    red_ink_mask,
)


DEFAULT_SCREENSHOTS_DIR = Path("tools/samples/screenshots_by_app_piece_style")
DEFAULT_LABELS_DIR = Path("tools/samples/labels/boards_by_app_piece_style")
DEFAULT_ANALYSIS_DIR = Path("tools/out/android_device_eval/android_eval_b5_29_clean_component_diagnostics_20260510")
DEFAULT_OUT_DIR = Path("tools/out/cell_crop_classifier_probe_b5_29_20260510")
DEFAULT_REPORTS_DIR = Path("tools/out/android_device_eval/android_eval_b5_29_clean_component_diagnostics_20260510")

GRAY_SIZE = (32, 32)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CELL_KEY_FIELDS = ("app", "piece_style", "sample", "square", "row", "col")
STRICT_MARGIN_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 8.0, 16.0, 32.0)


@dataclass(frozen=True)
class NamedSplit:
    split_mode: str
    fold_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


@dataclass(frozen=True)
class CellCropExample:
    app: str
    piece_style: str
    sample: str
    source_id: str
    row: int
    col: int
    square: str
    label: str
    color: str
    piece: str
    label_path: Path
    image_path: Path
    crop_path: str
    gray_crop_path: str
    grid_method: str
    grid_confidence: float
    feature: np.ndarray
    red_share: float
    black_ink_ratio: float
    red_ink_ratio: float
    edge_red_share: float
    central_red_share: float
    gray_mean: float
    gray_std: float


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def sanitize_path_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.\-一-龯ぁ-んァ-ヶー]+", "_", value).strip("_") or "unknown"


def source_id(app: str, piece_style: str, sample: str) -> str:
    return f"{app}/{piece_style}/{sample}"


def parse_app_style_from_label(label_path: Path, labels_dir: Path) -> tuple[str, str]:
    rel = label_path.relative_to(labels_dir)
    parts = rel.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "", ""


def normalized_gray(crop: Image.Image) -> tuple[np.ndarray, Image.Image, float, float]:
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    resized = cv2.resize(gray, GRAY_SIZE, interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    std = float(resized.std())
    normalized = (resized - mean) / max(std, 1.0)
    preview = np.clip((normalized * 32.0) + 128.0, 0, 255).astype("uint8")
    return normalized.reshape(-1).astype("float32"), Image.fromarray(preview, mode="L"), mean, std


def color_stats(crop: Image.Image) -> dict[str, float]:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    red_mask = red_ink_mask(rgb) > 0
    black_mask = black_ink_mask(rgb) > 0
    red_count = int(np.count_nonzero(red_mask))
    black_count = int(np.count_nonzero(black_mask))
    ink_count = red_count + black_count
    height, width = red_mask.shape
    yy, xx = np.indices(red_mask.shape)
    central = (
        (xx >= width * 0.28)
        & (xx <= width * 0.72)
        & (yy >= height * 0.18)
        & (yy <= height * 0.84)
    )
    edge = (
        (xx < width * 0.18)
        | (xx > width * 0.82)
        | (yy < height * 0.12)
        | (yy > height * 0.88)
    )
    red_mean = rgb[:, :, 0].mean()
    green_mean = rgb[:, :, 1].mean()
    blue_mean = rgb[:, :, 2].mean()
    red_std = rgb[:, :, 0].std()
    green_std = rgb[:, :, 1].std()
    blue_std = rgb[:, :, 2].std()
    return {
        "red_share": red_count / max(1, ink_count),
        "black_ink_ratio": black_count / max(1, red_mask.size),
        "red_ink_ratio": red_count / max(1, red_mask.size),
        "edge_red_share": int(np.count_nonzero(red_mask & edge)) / max(1, red_count),
        "central_red_share": int(np.count_nonzero(red_mask & central)) / max(1, red_count),
        "red_mean": float(red_mean) / 255.0,
        "green_mean": float(green_mean) / 255.0,
        "blue_mean": float(blue_mean) / 255.0,
        "red_std": float(red_std) / 255.0,
        "green_std": float(green_std) / 255.0,
        "blue_std": float(blue_std) / 255.0,
    }


def feature_vector(crop: Image.Image) -> tuple[np.ndarray, Image.Image, dict[str, float]]:
    gray_vector, gray_image, gray_mean, gray_std = normalized_gray(crop)
    hog = extract_hog_features(crop)
    stats = color_stats(crop)
    extras = np.asarray(
        [
            stats["red_share"],
            stats["black_ink_ratio"],
            stats["red_ink_ratio"],
            stats["edge_red_share"],
            stats["central_red_share"],
            stats["red_mean"],
            stats["green_mean"],
            stats["blue_mean"],
            stats["red_std"],
            stats["green_std"],
            stats["blue_std"],
            gray_mean / 255.0,
            gray_std / 255.0,
            hog.red_share,
            hog.ink_density,
            hog.edge_density,
        ],
        dtype="float32",
    )
    vector = np.concatenate([gray_vector, np.asarray(hog.vector, dtype="float32"), extras]).astype("float32")
    stats.update({"gray_mean": gray_mean, "gray_std": gray_std})
    return vector, gray_image, stats


def crop_output_paths(out_dir: Path, label: str, source: str, row: int, col: int) -> tuple[Path, Path]:
    safe_label = sanitize_path_part(label.replace(":", "_"))
    safe_source = sanitize_path_part(source.replace("/", "__"))
    filename = f"{safe_source}_r{row:02d}_c{col:02d}.png"
    return (
        out_dir / "dataset" / "crops" / safe_label / filename,
        out_dir / "dataset" / "gray32" / safe_label / filename,
    )


def build_dataset(
    labels_dir: Path,
    screenshots_dir: Path,
    out_dir: Path,
    *,
    write_crops: bool = True,
) -> tuple[list[CellCropExample], list[dict[str, Any]]]:
    examples: list[CellCropExample] = []
    issues: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.rglob("*.json")):
        app, piece_style = parse_app_style_from_label(label_path, labels_dir)
        if not app or not piece_style:
            issues.append({"severity": "warning", "source": label_path.stem, "issue": "unstructured_label_path", "path": str(label_path)})
            continue
        label = load_position_label(label_path)
        if label.get("exclude_from_benchmark"):
            issues.append({"severity": "info", "source": label_path.stem, "issue": "exclude_from_benchmark", "path": str(label_path)})
            continue
        image_path = resolve_label_image_path(label_path, label, screenshots_dir)
        if not image_path.exists():
            issues.append({"severity": "error", "source": label_path.stem, "issue": "missing_image", "path": str(image_path)})
            continue
        image = Image.open(image_path).convert("RGB")
        detection = detect_grid(image)
        if detection is None:
            issues.append({"severity": "error", "source": label_path.stem, "issue": "grid_detection_failed", "path": str(image_path)})
            continue
        source = source_id(app, piece_style, label_path.stem)
        for cell in label["cells"]:
            if cell["state"] != "piece":
                continue
            row = int(cell["row"])
            col = int(cell["col"])
            crop = crop_calibration_cell(image, detection, row, col)
            label_identity = identity(cell["color"], cell["piece"])
            crop_path, gray_path = crop_output_paths(out_dir, label_identity, source, row, col)
            vector, gray_image, stats = feature_vector(crop)
            if write_crops:
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                gray_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
                gray_image.save(gray_path)
            examples.append(
                CellCropExample(
                    app=app,
                    piece_style=piece_style,
                    sample=label_path.stem,
                    source_id=source,
                    row=row,
                    col=col,
                    square=cell.get("square") or square_name(row, col),
                    label=label_identity,
                    color=cell["color"],
                    piece=cell["piece"],
                    label_path=label_path,
                    image_path=image_path,
                    crop_path=str(crop_path),
                    gray_crop_path=str(gray_path),
                    grid_method=detection.method,
                    grid_confidence=float(detection.confidence),
                    feature=vector,
                    red_share=stats["red_share"],
                    black_ink_ratio=stats["black_ink_ratio"],
                    red_ink_ratio=stats["red_ink_ratio"],
                    edge_red_share=stats["edge_red_share"],
                    central_red_share=stats["central_red_share"],
                    gray_mean=stats["gray_mean"],
                    gray_std=stats["gray_std"],
                )
            )
    return examples, issues


def dataset_manifest_rows(examples: Sequence[CellCropExample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ex in enumerate(examples):
        rows.append(
            {
                "example_index": index,
                "app": ex.app,
                "piece_style": ex.piece_style,
                "sample": ex.sample,
                "source_id": ex.source_id,
                "square": ex.square,
                "row": ex.row,
                "col": ex.col,
                "label": ex.label,
                "color": ex.color,
                "piece": ex.piece,
                "crop_path": ex.crop_path,
                "gray_crop_path": ex.gray_crop_path,
                "image_path": str(ex.image_path),
                "label_path": str(ex.label_path),
                "grid_method": ex.grid_method,
                "grid_confidence": round_float(ex.grid_confidence),
                "red_share": round_float(ex.red_share),
                "black_ink_ratio": round_float(ex.black_ink_ratio),
                "red_ink_ratio": round_float(ex.red_ink_ratio),
                "edge_red_share": round_float(ex.edge_red_share),
                "central_red_share": round_float(ex.central_red_share),
                "gray_mean": round_float(ex.gray_mean),
                "gray_std": round_float(ex.gray_std),
            }
        )
    return rows


def make_classifier(name: str, random_state: int = 20260510) -> Any:
    if name == "logistic":
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1200,
            solver="lbfgs",
            random_state=random_state,
        )
    elif name == "linear_svm":
        estimator = SGDClassifier(
            loss="hinge",
            alpha=0.001,
            class_weight="balanced",
            max_iter=60,
            tol=None,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"unsupported classifier: {name}")
    return make_pipeline(StandardScaler(), estimator)


def class_scores(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.asarray(model.classes_)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
    else:
        scores = model.predict_proba(x)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    if scores.shape[1] != len(classes) and len(classes) == 2:
        scores = np.vstack([-scores[:, 0], scores[:, 0]]).T
    return classes, scores


def ranked_predictions(classes: Sequence[str], scores: Sequence[float]) -> list[tuple[int, str, float]]:
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    return [(rank, str(classes[index]), float(score)) for rank, (index, score) in enumerate(ranked, start=1)]


def expected_rank(ranked: Sequence[tuple[int, str, float]], expected: str) -> int | None:
    for rank, label, _score in ranked:
        if label == expected:
            return rank
    return None


def prediction_rows_for_indices(
    examples: Sequence[CellCropExample],
    indices: Sequence[int],
    model: Any,
    x_matrix: np.ndarray,
    *,
    split_mode: str,
    fold_id: str,
    train_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    classes, scores = class_scores(model, x_matrix[list(indices)])
    available_labels = set(str(label) for label in classes.tolist()) if train_labels is None else train_labels
    rows: list[dict[str, Any]] = []
    for output_index, example_index in enumerate(indices):
        ex = examples[example_index]
        ranked = ranked_predictions(classes, scores[output_index])
        rank = expected_rank(ranked, ex.label)
        top1 = ranked[0][1] if ranked else ""
        top3 = [item[1] for item in ranked[:3]]
        top1_margin = ranked[0][2] - ranked[1][2] if len(ranked) >= 2 else 0.0
        rows.append(
            {
                "split_mode": split_mode,
                "fold_id": fold_id,
                "example_index": example_index,
                "app": ex.app,
                "piece_style": ex.piece_style,
                "sample": ex.sample,
                "source_id": ex.source_id,
                "square": ex.square,
                "row": ex.row,
                "col": ex.col,
                "expected": ex.label,
                "classifier_top1": top1,
                "classifier_top1_score": round_float(ranked[0][2]) if ranked else "",
                "classifier_top2": ranked[1][1] if len(ranked) >= 2 else "",
                "classifier_top2_score": round_float(ranked[1][2]) if len(ranked) >= 2 else "",
                "classifier_top3": ranked[2][1] if len(ranked) >= 3 else "",
                "classifier_top3_score": round_float(ranked[2][2]) if len(ranked) >= 3 else "",
                "classifier_expected_rank": rank if rank is not None else "",
                "classifier_top3_contains_expected": ex.label in top3,
                "classifier_top1_correct": top1 == ex.label,
                "classifier_top1_margin": round_float(top1_margin),
                "expected_class_in_train": ex.label in available_labels,
                "trained_class_count": len(available_labels),
            }
        )
    return rows


def train_and_predict_loso(
    examples: Sequence[CellCropExample],
    classifier_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x_matrix = np.vstack([ex.feature for ex in examples]).astype("float32")
    y = np.asarray([ex.label for ex in examples])
    sources = np.asarray([ex.source_id for ex in examples])
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    unique_sources = sorted(set(sources.tolist()))
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for heldout in unique_sources:
        train_indices = np.flatnonzero(sources != heldout)
        test_indices = np.flatnonzero(sources == heldout)
        train_sources = set(sources[train_indices].tolist())
        test_sources = set(sources[test_indices].tolist())
        overlap = sorted(train_sources & test_sources)
        audit_rows.append(
            {
                "split_mode": "leave_one_source_out",
                "fold_id": heldout,
                "train_example_count": len(train_indices),
                "test_example_count": len(test_indices),
                "train_source_count": len(train_sources),
                "test_source_count": len(test_sources),
                "overlap_source_count": len(overlap),
                "overlap_sources": ";".join(overlap),
                "leak": bool(overlap),
            }
        )
        model = make_classifier(classifier_name)
        model.fit(x_matrix[train_indices], y[train_indices])
        train_labels = set(str(label) for label in y[train_indices].tolist())
        prediction_rows.extend(
            prediction_rows_for_indices(
                examples,
                test_indices.tolist(),
                model,
                x_matrix,
                split_mode="leave_one_source_out",
                fold_id=heldout,
                train_labels=train_labels,
            )
        )
    return prediction_rows, audit_rows


def run_group_kfold_summary(
    examples: Sequence[CellCropExample],
    classifier_name: str,
    n_splits: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x_matrix = np.vstack([ex.feature for ex in examples]).astype("float32")
    y = np.asarray([ex.label for ex in examples])
    groups = np.asarray([ex.source_id for ex in examples])
    source_count = len(set(groups.tolist()))
    n_splits = max(2, min(n_splits, source_count))
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(GroupKFold(n_splits=n_splits).split(x_matrix, y, groups), start=1):
        train_sources = set(groups[train_indices].tolist())
        test_sources = set(groups[test_indices].tolist())
        overlap = sorted(train_sources & test_sources)
        model = make_classifier(classifier_name)
        model.fit(x_matrix[train_indices], y[train_indices])
        train_labels = set(str(label) for label in y[train_indices].tolist())
        rows = prediction_rows_for_indices(
            examples,
            test_indices.tolist(),
            model,
            x_matrix,
            split_mode="group_kfold",
            fold_id=f"fold_{fold_index}",
            train_labels=train_labels,
        )
        y_true = [row["expected"] for row in rows]
        y_pred = [row["classifier_top1"] for row in rows]
        top3 = sum(1 for row in rows if row["classifier_top3_contains_expected"])
        summary_rows.append(
            {
                "split_mode": "group_kfold",
                "fold_id": f"fold_{fold_index}",
                "train_example_count": len(train_indices),
                "test_example_count": len(test_indices),
                "train_source_count": len(train_sources),
                "test_source_count": len(test_sources),
                "top1_accuracy": round_float(accuracy_score(y_true, y_pred)),
                "top3_accuracy": round_float(top3 / max(1, len(rows))),
            }
        )
        audit_rows.append(
            {
                "split_mode": "group_kfold",
                "fold_id": f"fold_{fold_index}",
                "train_example_count": len(train_indices),
                "test_example_count": len(test_indices),
                "train_source_count": len(train_sources),
                "test_source_count": len(test_sources),
                "overlap_source_count": len(overlap),
                "overlap_sources": ";".join(overlap),
                "leak": bool(overlap),
            }
        )
    return summary_rows, audit_rows


def is_piyo_chick_normal(ex: CellCropExample) -> bool:
    return ex.app == "ぴよ将棋" and ex.piece_style == "ひよこ駒" and "_通常_" in ex.sample


def is_piyo_chick(ex: CellCropExample) -> bool:
    return ex.app == "ぴよ将棋" and ex.piece_style == "ひよこ駒"


def strict_split_definitions(examples: Sequence[CellCropExample]) -> list[NamedSplit]:
    all_indices = set(range(len(examples)))
    splits: list[NamedSplit] = []
    by_app_style: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, ex in enumerate(examples):
        by_app_style[(ex.app, ex.piece_style)].append(index)
    for (app, style), test_indices in sorted(by_app_style.items()):
        test_set = set(test_indices)
        train_indices = tuple(sorted(all_indices - test_set))
        splits.append(
            NamedSplit(
                split_mode="leave_app_style_out",
                fold_id=f"{app}/{style}",
                train_indices=train_indices,
                test_indices=tuple(sorted(test_indices)),
            )
        )

    normal_indices = [index for index, ex in enumerate(examples) if is_piyo_chick_normal(ex)]
    if normal_indices:
        test_set = set(normal_indices)
        splits.append(
            NamedSplit(
                split_mode="leave_piyo_chick_normal_out",
                fold_id="ぴよ将棋/ひよこ駒/通常_*",
                train_indices=tuple(sorted(all_indices - test_set)),
                test_indices=tuple(sorted(normal_indices)),
            )
        )

    chick_indices = [index for index, ex in enumerate(examples) if is_piyo_chick(ex)]
    if chick_indices:
        test_set = set(chick_indices)
        splits.append(
            NamedSplit(
                split_mode="leave_piyo_chick_out",
                fold_id="ぴよ将棋/ひよこ駒",
                train_indices=tuple(sorted(all_indices - test_set)),
                test_indices=tuple(sorted(chick_indices)),
            )
        )
    return splits


def split_source_overlap(examples: Sequence[CellCropExample], split: NamedSplit) -> tuple[set[str], set[str], list[str]]:
    train_sources = {examples[index].source_id for index in split.train_indices}
    test_sources = {examples[index].source_id for index in split.test_indices}
    return train_sources, test_sources, sorted(train_sources & test_sources)


def class_coverage_rows_for_split(examples: Sequence[CellCropExample], split: NamedSplit) -> list[dict[str, Any]]:
    train_counts = Counter(examples[index].label for index in split.train_indices)
    test_counts = Counter(examples[index].label for index in split.test_indices)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(train_counts) | set(test_counts)):
        rows.append(
            {
                "split_mode": split.split_mode,
                "fold_id": split.fold_id,
                "label": label,
                "train_count": train_counts[label],
                "test_count": test_counts[label],
                "missing_in_train": test_counts[label] > 0 and train_counts[label] == 0,
            }
        )
    return rows


def train_and_predict_strict_splits(
    examples: Sequence[CellCropExample],
    classifier_name: str,
    splits: Sequence[NamedSplit] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    x_matrix = np.vstack([ex.feature for ex in examples]).astype("float32")
    y = np.asarray([ex.label for ex in examples])
    selected_splits = list(splits) if splits is not None else strict_split_definitions(examples)
    prediction_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for split in selected_splits:
        train_indices = np.asarray(split.train_indices, dtype=int)
        test_indices = np.asarray(split.test_indices, dtype=int)
        train_sources, test_sources, overlap = split_source_overlap(examples, split)
        test_app_styles = sorted({(examples[index].app, examples[index].piece_style) for index in split.test_indices})
        heldout_app_styles = ";".join(f"{app}/{style}" for app, style in test_app_styles)
        train_same_app_style_count = sum(
            1 for index in split.train_indices if (examples[index].app, examples[index].piece_style) in test_app_styles
        )
        test_same_app_style_count = sum(
            1 for index in split.test_indices if (examples[index].app, examples[index].piece_style) in test_app_styles
        )
        train_piyo_chick_normal_count = sum(1 for index in split.train_indices if is_piyo_chick_normal(examples[index]))
        test_piyo_chick_normal_count = sum(1 for index in split.test_indices if is_piyo_chick_normal(examples[index]))
        train_piyo_chick_count = sum(1 for index in split.train_indices if is_piyo_chick(examples[index]))
        test_piyo_chick_count = sum(1 for index in split.test_indices if is_piyo_chick(examples[index]))
        train_labels = set(str(label) for label in y[train_indices].tolist())
        test_labels = set(str(label) for label in y[test_indices].tolist())
        coverage = class_coverage_rows_for_split(examples, split)
        coverage_rows.extend(coverage)
        missing_labels = sorted(test_labels - train_labels)
        missing_example_count = sum(1 for index in test_indices if examples[int(index)].label in missing_labels)
        audit_rows.append(
            {
                "split_mode": split.split_mode,
                "fold_id": split.fold_id,
                "heldout_app_styles": heldout_app_styles,
                "train_example_count": len(train_indices),
                "test_example_count": len(test_indices),
                "train_source_count": len(train_sources),
                "test_source_count": len(test_sources),
                "train_same_app_style_count": train_same_app_style_count,
                "test_same_app_style_count": test_same_app_style_count,
                "train_piyo_chick_normal_count": train_piyo_chick_normal_count,
                "test_piyo_chick_normal_count": test_piyo_chick_normal_count,
                "train_piyo_chick_count": train_piyo_chick_count,
                "test_piyo_chick_count": test_piyo_chick_count,
                "overlap_source_count": len(overlap),
                "overlap_sources": ";".join(overlap),
                "leak": bool(overlap),
            }
        )
        if len(train_indices) == 0 or len(test_indices) == 0 or len(train_labels) < 2:
            metrics_rows.append(
                {
                    "split_mode": split.split_mode,
                    "fold_id": split.fold_id,
                    "train_example_count": len(train_indices),
                    "test_example_count": len(test_indices),
                    "train_source_count": len(train_sources),
                    "test_source_count": len(test_sources),
                    "train_class_count": len(train_labels),
                    "test_class_count": len(test_labels),
                    "missing_test_class_count": len(missing_labels),
                    "missing_test_classes": ";".join(missing_labels),
                    "missing_test_example_count": missing_example_count,
                    "top1_accuracy": "",
                    "top3_accuracy": "",
                    "covered_class_example_count": 0,
                    "covered_class_top1_accuracy": "",
                    "covered_class_top3_accuracy": "",
                    "skipped": True,
                }
            )
            continue
        model = make_classifier(classifier_name)
        model.fit(x_matrix[train_indices], y[train_indices])
        rows = prediction_rows_for_indices(
            examples,
            test_indices.tolist(),
            model,
            x_matrix,
            split_mode=split.split_mode,
            fold_id=split.fold_id,
            train_labels=train_labels,
        )
        prediction_rows.extend(rows)
        covered_rows = [row for row in rows if row.get("expected_class_in_train") is True]
        metrics_rows.append(
            {
                "split_mode": split.split_mode,
                "fold_id": split.fold_id,
                "train_example_count": len(train_indices),
                "test_example_count": len(test_indices),
                "train_source_count": len(train_sources),
                "test_source_count": len(test_sources),
                "train_class_count": len(train_labels),
                "test_class_count": len(test_labels),
                "missing_test_class_count": len(missing_labels),
                "missing_test_classes": ";".join(missing_labels),
                "missing_test_example_count": missing_example_count,
                "top1_accuracy": round_float(sum(1 for row in rows if row["classifier_top1_correct"]) / max(1, len(rows))),
                "top3_accuracy": round_float(sum(1 for row in rows if row["classifier_top3_contains_expected"]) / max(1, len(rows))),
                "covered_class_example_count": len(covered_rows),
                "covered_class_top1_accuracy": (
                    round_float(sum(1 for row in covered_rows if row["classifier_top1_correct"]) / max(1, len(covered_rows)))
                    if covered_rows
                    else ""
                ),
                "covered_class_top3_accuracy": (
                    round_float(sum(1 for row in covered_rows if row["classifier_top3_contains_expected"]) / max(1, len(covered_rows)))
                    if covered_rows
                    else ""
                ),
                "skipped": False,
            }
        )
    return prediction_rows, metrics_rows, audit_rows, coverage_rows


def confusion_matrix_rows(prediction_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels: set[str] = set()
    for row in prediction_rows:
        expected = str(row["expected"])
        predicted = str(row["classifier_top1"])
        counts[expected][predicted] += 1
        labels.add(expected)
        labels.add(predicted)
    sorted_labels = sorted(labels)
    rows: list[dict[str, Any]] = []
    for expected in sorted_labels:
        row: dict[str, Any] = {"expected": expected, "total": sum(counts[expected].values())}
        for predicted in sorted_labels:
            row[predicted] = counts[expected][predicted]
        rows.append(row)
    return rows


def confusion_matrix_rows_by_split(prediction_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    labels = sorted({str(row["expected"]) for row in prediction_rows} | {str(row["classifier_top1"]) for row in prediction_rows})
    rows: list[dict[str, Any]] = []
    for (split_mode, fold_id), group_rows in sorted(predictions_by_split(prediction_rows).items()):
        for row in confusion_matrix_rows(group_rows):
            rows.append({"split_mode": split_mode, "fold_id": fold_id, **row})
    return rows, ["split_mode", "fold_id", "expected", "total"] + labels


def classifier_error_summary_rows(prediction_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("classifier_top1_correct") is True:
            continue
        key = (
            str(row.get("split_mode", "")),
            str(row.get("fold_id", "")),
            str(row.get("expected", "")),
            str(row.get("classifier_top1", "")),
        )
        grouped[key].append(row)
    rows: list[dict[str, Any]] = []
    for (split_mode, fold_id, expected, predicted), group_rows in sorted(grouped.items()):
        margins = [parse_float(row.get("classifier_top1_margin")) for row in group_rows]
        top3_contains = sum(1 for row in group_rows if row.get("classifier_top3_contains_expected") is True)
        rows.append(
            {
                "split_mode": split_mode,
                "fold_id": fold_id,
                "expected": expected,
                "classifier_top1": predicted,
                "count": len(group_rows),
                "top3_contains_expected_count": top3_contains,
                "avg_margin": round_float(sum(margins) / max(1, len(margins))),
                "min_margin": round_float(min(margins) if margins else 0.0),
                "max_margin": round_float(max(margins) if margins else 0.0),
            }
        )
    return rows


def aggregate_metrics(prediction_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(prediction_rows)
    top1 = sum(1 for row in prediction_rows if row["classifier_top1_correct"])
    top3 = sum(1 for row in prediction_rows if row["classifier_top3_contains_expected"])
    by_app_style: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_app_style[(str(row["app"]), str(row["piece_style"]))].append(row)
        by_label[str(row["expected"])].append(row)
    return {
        "total_examples": total,
        "top1_correct": top1,
        "top1_accuracy": top1 / max(1, total),
        "top3_contains": top3,
        "top3_accuracy": top3 / max(1, total),
        "by_app_style": {
            f"{app}/{style}": {
                "examples": len(rows),
                "top1_accuracy": sum(1 for row in rows if row["classifier_top1_correct"]) / max(1, len(rows)),
                "top3_accuracy": sum(1 for row in rows if row["classifier_top3_contains_expected"]) / max(1, len(rows)),
            }
            for (app, style), rows in sorted(by_app_style.items())
        },
        "by_label": {
            label: {
                "examples": len(rows),
                "top1_accuracy": sum(1 for row in rows if row["classifier_top1_correct"]) / max(1, len(rows)),
                "top3_accuracy": sum(1 for row in rows if row["classifier_top3_contains_expected"]) / max(1, len(rows)),
            }
            for label, rows in sorted(by_label.items())
        },
    }


def prediction_lookup(prediction_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in prediction_rows:
        lookup[(str(row["source_id"]), parse_int(row["row"]), parse_int(row["col"]))] = row
    return lookup


def cell_key_from_row(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in CELL_KEY_FIELDS)  # type: ignore[return-value]


def baseline_candidate_source_leak(row: dict[str, str]) -> bool:
    source = row.get("source", "")
    sample = row.get("sample", "")
    if not source or not sample:
        return False
    return sample in source


def residual_summary_rows(
    analysis_dir: Path,
    predictions: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = read_csv(analysis_dir / "piece_style_board_errors.csv")
    gaps = {cell_key_from_row(row): row for row in read_csv(analysis_dir / "piece_style_board_error_candidate_gaps.csv")}
    gaps_by_square = {
        (row.get("app", ""), row.get("piece_style", ""), row.get("sample", ""), row.get("square", "")): row
        for row in gaps.values()
    }
    candidates = read_csv(analysis_dir / "piece_style_board_error_candidates.csv")
    candidate_leaks = Counter()
    candidate_rows = Counter()
    for row in candidates:
        key = cell_key_from_row(row)
        candidate_rows[key] += 1
        if baseline_candidate_source_leak(row):
            candidate_leaks[key] += 1
    lookup = prediction_lookup(predictions)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for error in errors:
        app = error.get("app", "")
        style = error.get("piece_style", "")
        sample = error.get("sample", "")
        square = error.get("square", "")
        gap = gaps_by_square.get((app, style, sample, square), {})
        row_num = parse_int(error.get("row") or gap.get("row"))
        col_num = parse_int(error.get("col") or gap.get("col"))
        source = source_id(app, style, sample)
        pred = lookup.get((source, row_num, col_num), {})
        expected = error.get("expected") or gap.get("expected", "")
        classifier_rank = pred.get("classifier_expected_rank", "")
        baseline_rank = parse_int(gap.get("expected_candidate_rank") or error.get("expected_candidate_rank"), 0)
        full_key = cell_key_from_row({**error, "row": str(row_num), "col": str(col_num)})
        rows.append(
            {
                "app": app,
                "piece_style": style,
                "sample": sample,
                "source_id": source,
                "square": square or gap.get("square", ""),
                "row": row_num,
                "col": col_num,
                "expected": expected,
                "baseline_top1": error.get("predicted_top1") or gap.get("predicted_top1", ""),
                "baseline_expected_rank": baseline_rank if baseline_rank else "",
                "baseline_top_score": gap.get("top_score", ""),
                "baseline_expected_score": gap.get("expected_score", ""),
                "classifier_top1": pred.get("classifier_top1", ""),
                "classifier_top2": pred.get("classifier_top2", ""),
                "classifier_top3": pred.get("classifier_top3", ""),
                "classifier_top1_margin": pred.get("classifier_top1_margin", ""),
                "classifier_expected_rank": classifier_rank,
                "classifier_top1_correct": pred.get("classifier_top1") == expected,
                "classifier_top3_contains_expected": pred.get("classifier_top3_contains_expected", ""),
                "expected_class_in_train": pred.get("expected_class_in_train", ""),
                "classifier_rank_delta_vs_baseline": (
                    parse_int(classifier_rank, 9999) - baseline_rank
                    if classifier_rank != "" and baseline_rank
                    else ""
                ),
                "classifier_rank_improved": (
                    parse_int(classifier_rank, 9999) < baseline_rank
                    if classifier_rank != "" and baseline_rank
                    else ""
                ),
                "baseline_candidate_rows": candidate_rows[full_key],
                "baseline_candidate_source_leak_rows": candidate_leaks[full_key],
            }
        )
        audit_rows.append(
            {
                "app": app,
                "piece_style": style,
                "sample": sample,
                "source_id": source,
                "row": row_num,
                "col": col_num,
                "classifier_prediction_found": bool(pred),
                "baseline_candidate_rows": rows[-1]["baseline_candidate_rows"],
                "baseline_candidate_source_leak_rows": rows[-1]["baseline_candidate_source_leak_rows"],
                "leak": rows[-1]["baseline_candidate_source_leak_rows"] > 0,
            }
        )
    return rows, audit_rows


def predictions_by_split(predictions: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row.get("split_mode", "")), str(row.get("fold_id", "")))].append(row)
    return grouped


def residual_summary_rows_by_split(
    analysis_dir: Path,
    predictions: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for (split_mode, fold_id), group_rows in sorted(predictions_by_split(predictions).items()):
        residual_rows, residual_audit = residual_summary_rows(analysis_dir, group_rows)
        for row, audit in zip(residual_rows, residual_audit):
            if not audit.get("classifier_prediction_found"):
                continue
            with_split = {"split_mode": split_mode, "fold_id": fold_id, **row}
            rows.append(with_split)
            audit_rows.append({"split_mode": split_mode, "fold_id": fold_id, **audit})
    return rows, audit_rows


def report_identity(cell: dict[str, Any]) -> str:
    if cell.get("state") == "piece" and cell.get("color") and cell.get("piece"):
        return identity(str(cell["color"]), str(cell["piece"]))
    return str(cell.get("state") or "unknown")


def load_report_cells(report_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in data.get("cells") or []:
        output[(parse_int(cell.get("row")), parse_int(cell.get("col")))] = cell
    return output


def stable_degradation_rows(
    analysis_dir: Path,
    labels_dir: Path,
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = prediction_lookup(predictions)
    rows: list[dict[str, Any]] = []
    for report_path in sorted(analysis_dir.rglob("piece_report.json")):
        try:
            rel = report_path.relative_to(analysis_dir)
        except ValueError:
            continue
        if len(rel.parts) < 4:
            continue
        app, style, sample = rel.parts[0], rel.parts[1], rel.parts[2]
        label_path = find_label_path(labels_dir, sample, app=app, piece_style=style)
        if not label_path.exists():
            continue
        label = load_position_label(label_path)
        report_cells = load_report_cells(report_path)
        source = source_id(app, style, sample)
        for cell in label["cells"]:
            if cell["state"] != "piece":
                continue
            row = int(cell["row"])
            col = int(cell["col"])
            expected = identity(cell["color"], cell["piece"])
            baseline = report_cells.get((row, col), {})
            baseline_top1 = report_identity(baseline)
            pred = lookup.get((source, row, col), {})
            if baseline_top1 != expected:
                continue
            if not pred or pred.get("classifier_top1") == expected:
                continue
            rows.append(
                {
                    "app": app,
                    "piece_style": style,
                    "sample": sample,
                    "source_id": source,
                    "square": cell.get("square") or square_name(row, col),
                    "row": row,
                    "col": col,
                    "expected": expected,
                    "baseline_top1": baseline_top1,
                    "baseline_confidence": baseline.get("confidence", ""),
                    "baseline_source": baseline.get("source", ""),
                    "classifier_top1": pred.get("classifier_top1", ""),
                    "classifier_top2": pred.get("classifier_top2", ""),
                    "classifier_top3": pred.get("classifier_top3", ""),
                    "classifier_top1_margin": pred.get("classifier_top1_margin", ""),
                    "classifier_expected_rank": pred.get("classifier_expected_rank", ""),
                    "classifier_top3_contains_expected": pred.get("classifier_top3_contains_expected", ""),
                    "expected_class_in_train": pred.get("expected_class_in_train", ""),
                }
            )
    return rows


def stable_degradation_rows_by_split(
    analysis_dir: Path,
    labels_dir: Path,
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split_mode, fold_id), group_rows in sorted(predictions_by_split(predictions).items()):
        for row in stable_degradation_rows(analysis_dir, labels_dir, group_rows):
            rows.append({"split_mode": split_mode, "fold_id": fold_id, **row})
    return rows


def class_count_rows(examples: Sequence[CellCropExample]) -> list[dict[str, Any]]:
    counts = Counter(ex.label for ex in examples)
    return [{"label": label, "examples": count} for label, count in sorted(counts.items())]


def source_count_rows(examples: Sequence[CellCropExample]) -> list[dict[str, Any]]:
    by_source: dict[str, list[CellCropExample]] = defaultdict(list)
    for ex in examples:
        by_source[ex.source_id].append(ex)
    rows: list[dict[str, Any]] = []
    for source, source_examples in sorted(by_source.items()):
        first = source_examples[0]
        rows.append(
            {
                "source_id": source,
                "app": first.app,
                "piece_style": first.piece_style,
                "sample": first.sample,
                "occupied_examples": len(source_examples),
                "classes": len({ex.label for ex in source_examples}),
            }
        )
    return rows


def split_cell_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row.get("split_mode", "")),
        str(row.get("fold_id", "")),
        str(row.get("source_id", "")),
        parse_int(row.get("row")),
        parse_int(row.get("col")),
    )


def confidence_margin_audit_rows(
    predictions: Sequence[dict[str, Any]],
    residual_rows: Sequence[dict[str, Any]],
    stable_rows: Sequence[dict[str, Any]],
    thresholds: Sequence[float] = STRICT_MARGIN_THRESHOLDS,
) -> list[dict[str, Any]]:
    residual_keys = {split_cell_key(row): row for row in residual_rows}
    stable_keys = {split_cell_key(row): row for row in stable_rows}
    rows: list[dict[str, Any]] = []
    for (split_mode, fold_id), group_rows in sorted(predictions_by_split(predictions).items()):
        residual_total = sum(1 for key in residual_keys if key[0] == split_mode and key[1] == fold_id)
        stable_total = sum(1 for key in stable_keys if key[0] == split_mode and key[1] == fold_id)
        for threshold in thresholds:
            accepted = [
                row
                for row in group_rows
                if row.get("expected_class_in_train") is True
                and parse_float(row.get("classifier_top1_margin"), -1.0e9) >= threshold
            ]
            accepted_keys = {split_cell_key(row) for row in accepted}
            residual_accepted = [residual_keys[key] for key in accepted_keys if key in residual_keys]
            stable_accepted = [stable_keys[key] for key in accepted_keys if key in stable_keys]
            rows.append(
                {
                    "split_mode": split_mode,
                    "fold_id": fold_id,
                    "margin_threshold": threshold,
                    "test_example_count": len(group_rows),
                    "accepted_example_count": len(accepted),
                    "accepted_coverage": round_float(len(accepted) / max(1, len(group_rows))),
                    "accepted_top1_accuracy": (
                        round_float(sum(1 for row in accepted if row.get("classifier_top1_correct") is True) / max(1, len(accepted)))
                        if accepted
                        else ""
                    ),
                    "accepted_error_count": sum(1 for row in accepted if row.get("classifier_top1_correct") is not True),
                    "residual_total": residual_total,
                    "residual_accepted_count": len(residual_accepted),
                    "residual_accepted_top1_fixes": sum(1 for row in residual_accepted if row.get("classifier_top1_correct") is True),
                    "residual_accepted_wrong": sum(1 for row in residual_accepted if row.get("classifier_top1_correct") is not True),
                    "stable_degradation_total": stable_total,
                    "stable_degradation_accepted": len(stable_accepted),
                }
            )
    return rows


def confidence_margin_quantile_audit_rows(
    predictions: Sequence[dict[str, Any]],
    residual_rows: Sequence[dict[str, Any]],
    stable_rows: Sequence[dict[str, Any]],
    quantiles: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split_mode, fold_id), group_rows in sorted(predictions_by_split(predictions).items()):
        margins = [
            parse_float(row.get("classifier_top1_margin"), 0.0)
            for row in group_rows
            if row.get("expected_class_in_train") is True
        ]
        if not margins:
            continue
        for quantile in quantiles:
            threshold = float(np.quantile(np.asarray(margins, dtype="float32"), quantile))
            audit = confidence_margin_audit_rows(
                [row for row in predictions if str(row.get("split_mode", "")) == split_mode and str(row.get("fold_id", "")) == fold_id],
                [row for row in residual_rows if str(row.get("split_mode", "")) == split_mode and str(row.get("fold_id", "")) == fold_id],
                [row for row in stable_rows if str(row.get("split_mode", "")) == split_mode and str(row.get("fold_id", "")) == fold_id],
                thresholds=[threshold],
            )
            if not audit:
                continue
            rows.append({"margin_quantile": quantile, **audit[0]})
    return rows


def production_gate_candidate_rows(
    margin_rows: Sequence[dict[str, Any]],
    metrics_rows: Sequence[dict[str, Any]],
    no_leak_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_lookup = {(str(row.get("split_mode", "")), str(row.get("fold_id", ""))): row for row in metrics_rows}
    leak_counts = Counter(
        (str(row.get("split_mode", "")), str(row.get("fold_id", "")))
        for row in no_leak_rows
        if str(row.get("leak")).lower() == "true" or row.get("leak") is True
    )
    rows: list[dict[str, Any]] = []
    for row in margin_rows:
        key = (str(row.get("split_mode", "")), str(row.get("fold_id", "")))
        metric = metric_lookup.get(key, {})
        residual_total = parse_int(row.get("residual_total"))
        residual_fixes = parse_int(row.get("residual_accepted_top1_fixes"))
        stable_accepted = parse_int(row.get("stable_degradation_accepted"))
        accepted_error_count = parse_int(row.get("accepted_error_count"))
        missing_count = parse_int(metric.get("missing_test_class_count"))
        leak_count = leak_counts[key]
        candidate = (
            residual_total > 0
            and residual_fixes == residual_total
            and stable_accepted == 0
            and accepted_error_count == 0
            and missing_count == 0
            and leak_count == 0
        )
        blocking_reasons = []
        if residual_total == 0:
            blocking_reasons.append("no_residual_cells_in_fold")
        elif residual_fixes != residual_total:
            blocking_reasons.append("does_not_fix_all_residual_cells")
        if stable_accepted:
            blocking_reasons.append("stable_degradation_accepted")
        if accepted_error_count:
            blocking_reasons.append("accepted_errors")
        if missing_count:
            blocking_reasons.append("missing_test_classes")
        if leak_count:
            blocking_reasons.append("leak")
        rows.append(
            {
                "split_mode": key[0],
                "fold_id": key[1],
                "margin_threshold": row.get("margin_threshold", ""),
                "production_gate_candidate": candidate,
                "blocking_reasons": ";".join(blocking_reasons) if blocking_reasons else "",
                "accepted_example_count": row.get("accepted_example_count", ""),
                "accepted_coverage": row.get("accepted_coverage", ""),
                "accepted_top1_accuracy": row.get("accepted_top1_accuracy", ""),
                "accepted_error_count": accepted_error_count,
                "residual_total": residual_total,
                "residual_accepted_top1_fixes": residual_fixes,
                "stable_degradation_accepted": stable_accepted,
                "missing_test_class_count": missing_count,
                "no_leak_failures": leak_count,
            }
        )
    return rows


def write_strict_markdown_summary(
    out_dir: Path,
    classifier_name: str,
    metrics_rows: Sequence[dict[str, Any]],
    residual_rows: Sequence[dict[str, Any]],
    stable_rows: Sequence[dict[str, Any]],
    gate_rows: Sequence[dict[str, Any]],
) -> None:
    residual_by_split: Counter[tuple[str, str]] = Counter()
    residual_fixed_by_split: Counter[tuple[str, str]] = Counter()
    for row in residual_rows:
        key = (str(row.get("split_mode", "")), str(row.get("fold_id", "")))
        residual_by_split[key] += 1
        if row.get("classifier_top1_correct") is True:
            residual_fixed_by_split[key] += 1
    stable_by_split: Counter[tuple[str, str]] = Counter((str(row.get("split_mode", "")), str(row.get("fold_id", ""))) for row in stable_rows)
    candidate_count = sum(1 for row in gate_rows if row.get("production_gate_candidate") is True)
    lines = [
        "# Cell Crop Classifier Strict Split Probe",
        "",
        f"- classifier: `{classifier_name}`",
        "- production status: offline diagnostic only, not approved for production",
        f"- production gate candidate rows: {candidate_count}",
        "",
        "## Strict Split Metrics",
        "",
        "| split | fold | test | missing classes | top1 | top3 | residual fixed | stable degradation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        key = (str(row.get("split_mode", "")), str(row.get("fold_id", "")))
        residual_total = residual_by_split[key]
        residual_fixed = residual_fixed_by_split[key]
        lines.append(
            "| {split} | {fold} | {test} | {missing} | {top1} | {top3} | {fixed}/{total} | {stable} |".format(
                split=key[0],
                fold=key[1],
                test=row.get("test_example_count", ""),
                missing=row.get("missing_test_class_count", ""),
                top1=row.get("top1_accuracy", ""),
                top3=row.get("top3_accuracy", ""),
                fixed=residual_fixed,
                total=residual_total,
                stable=stable_by_split[key],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These rows are stricter offline holdouts. They do not change Android recognition.",
            "- Offline top1/top3 values are not Android recognition metrics.",
            "- A production gate candidate requires no leak, no missing held-out class, all residual cells fixed under the gate, no accepted stable degradation, and no accepted classifier errors.",
            "",
            "WAIT_QA",
        ]
    )
    (out_dir / "cell_crop_classifier_strict_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(
    out_dir: Path,
    metrics: dict[str, Any],
    residual_rows: Sequence[dict[str, Any]],
    stable_rows: Sequence[dict[str, Any]],
    no_leak_rows: Sequence[dict[str, Any]],
    classifier_name: str,
) -> None:
    residual_fixed = sum(1 for row in residual_rows if row.get("classifier_top1_correct") is True)
    residual_top3 = sum(1 for row in residual_rows if row.get("classifier_top3_contains_expected") is True)
    leak_count = sum(1 for row in no_leak_rows if str(row.get("leak")).lower() == "true" or row.get("leak") is True)
    lines = [
        "# Cell Crop Classifier Offline Probe",
        "",
        f"- classifier: `{classifier_name}`",
        f"- split: leave-one-source-out grouped by `app/piece_style/sample`",
        f"- examples: {metrics['total_examples']}",
        f"- top1 accuracy: {metrics['top1_accuracy']:.4f}",
        f"- top3 accuracy: {metrics['top3_accuracy']:.4f}",
        f"- B5-29 residual top1 fixes: {residual_fixed}/{len(residual_rows)}",
        f"- B5-29 residual top3 contains expected: {residual_top3}/{len(residual_rows)}",
        f"- stable baseline-correct cells worsened by classifier top1: {len(stable_rows)}",
        f"- no-leak audit leak rows: {leak_count}",
        "",
        "## B5-29 Residual Cells",
        "",
        "| sample | square | expected | baseline top1/rank | classifier top1/rank/top3 |",
        "|---|---:|---|---|---|",
    ]
    for row in residual_rows:
        classifier_top3 = "/".join(str(row.get(field, "")) for field in ("classifier_top1", "classifier_top2", "classifier_top3"))
        lines.append(
            "| {sample} | {square} | {expected} | {baseline_top1} / {baseline_expected_rank} | {top3} / {rank} |".format(
                sample=row.get("sample", ""),
                square=row.get("square", ""),
                expected=row.get("expected", ""),
                baseline_top1=row.get("baseline_top1", ""),
                baseline_expected_rank=row.get("baseline_expected_rank", ""),
                top3=classifier_top3,
                rank=row.get("classifier_expected_rank", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Production",
            "",
            "This is an offline diagnostic only. Do not promote it to production without AI-2 QA and an explicit productionization plan.",
            "",
            "WAIT_QA",
        ]
    )
    (out_dir / "cell_crop_classifier_probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    labels_dir: Path,
    screenshots_dir: Path,
    analysis_dir: Path,
    out_dir: Path,
    classifier_name: str = "linear_svm",
    group_folds: int = 5,
    write_crops: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples, dataset_issues = build_dataset(labels_dir, screenshots_dir, out_dir, write_crops=write_crops)
    if len(examples) < 2:
        raise RuntimeError("not enough occupied cell examples")

    manifest_rows = dataset_manifest_rows(examples)
    write_csv(
        out_dir / "cell_crop_dataset_manifest.csv",
        manifest_rows,
        [
            "example_index",
            "app",
            "piece_style",
            "sample",
            "source_id",
            "square",
            "row",
            "col",
            "label",
            "color",
            "piece",
            "crop_path",
            "gray_crop_path",
            "image_path",
            "label_path",
            "grid_method",
            "grid_confidence",
            "red_share",
            "black_ink_ratio",
            "red_ink_ratio",
            "edge_red_share",
            "central_red_share",
            "gray_mean",
            "gray_std",
        ],
    )
    write_csv(out_dir / "cell_crop_dataset_class_counts.csv", class_count_rows(examples), ["label", "examples"])
    write_csv(out_dir / "cell_crop_dataset_source_counts.csv", source_count_rows(examples), ["source_id", "app", "piece_style", "sample", "occupied_examples", "classes"])
    write_csv(out_dir / "cell_crop_dataset_issues.csv", dataset_issues, ["severity", "source", "issue", "path"])

    loso_predictions, loso_audit = train_and_predict_loso(examples, classifier_name)
    write_csv(
        out_dir / "cell_crop_classifier_loso_predictions.csv",
        loso_predictions,
        [
            "split_mode",
            "fold_id",
            "example_index",
            "app",
            "piece_style",
            "sample",
            "source_id",
            "square",
            "row",
            "col",
            "expected",
            "classifier_top1",
            "classifier_top1_score",
            "classifier_top2",
            "classifier_top2_score",
            "classifier_top3",
            "classifier_top3_score",
            "classifier_expected_rank",
            "classifier_top3_contains_expected",
            "classifier_top1_correct",
            "classifier_top1_margin",
            "expected_class_in_train",
            "trained_class_count",
        ],
    )
    group_summary, group_audit = run_group_kfold_summary(examples, classifier_name, group_folds)
    write_csv(
        out_dir / "cell_crop_classifier_group_split_summary.csv",
        group_summary,
        [
            "split_mode",
            "fold_id",
            "train_example_count",
            "test_example_count",
            "train_source_count",
            "test_source_count",
            "top1_accuracy",
            "top3_accuracy",
        ],
    )

    confusion = confusion_matrix_rows(loso_predictions)
    confusion_fields = ["expected", "total"] + sorted({row["expected"] for row in loso_predictions} | {row["classifier_top1"] for row in loso_predictions})
    write_csv(out_dir / "cell_crop_classifier_confusion_matrix.csv", confusion, confusion_fields)

    metrics = aggregate_metrics(loso_predictions)
    write_json(out_dir / "cell_crop_classifier_metrics_summary.json", metrics)

    residual_rows, residual_audit = residual_summary_rows(analysis_dir, loso_predictions)
    write_csv(
        out_dir / "cell_crop_classifier_b5_29_residual_summary.csv",
        residual_rows,
        [
            "app",
            "piece_style",
            "sample",
            "source_id",
            "square",
            "row",
            "col",
            "expected",
            "baseline_top1",
            "baseline_expected_rank",
            "baseline_top_score",
            "baseline_expected_score",
            "classifier_top1",
            "classifier_top2",
            "classifier_top3",
            "classifier_top1_margin",
            "classifier_expected_rank",
            "classifier_top1_correct",
            "classifier_top3_contains_expected",
            "expected_class_in_train",
            "classifier_rank_delta_vs_baseline",
            "classifier_rank_improved",
            "baseline_candidate_rows",
            "baseline_candidate_source_leak_rows",
        ],
    )

    stable_rows = stable_degradation_rows(analysis_dir, labels_dir, loso_predictions)
    write_csv(
        out_dir / "cell_crop_classifier_stable_cell_degradation_audit.csv",
        stable_rows,
        [
            "app",
            "piece_style",
            "sample",
            "source_id",
            "square",
            "row",
            "col",
            "expected",
            "baseline_top1",
            "baseline_confidence",
            "baseline_source",
            "classifier_top1",
            "classifier_top2",
            "classifier_top3",
            "classifier_top1_margin",
            "classifier_expected_rank",
            "classifier_top3_contains_expected",
            "expected_class_in_train",
        ],
    )

    no_leak_rows = loso_audit + group_audit + residual_audit
    write_csv(
        out_dir / "cell_crop_classifier_no_leak_audit.csv",
        no_leak_rows,
        [
            "split_mode",
            "fold_id",
            "app",
            "piece_style",
            "sample",
            "source_id",
            "row",
            "col",
            "classifier_prediction_found",
            "train_example_count",
            "test_example_count",
            "train_source_count",
            "test_source_count",
            "overlap_source_count",
            "overlap_sources",
            "baseline_candidate_rows",
            "baseline_candidate_source_leak_rows",
            "leak",
        ],
    )

    strict_predictions, strict_metrics, strict_audit, strict_coverage = train_and_predict_strict_splits(examples, classifier_name)
    strict_prediction_fields = [
        "split_mode",
        "fold_id",
        "example_index",
        "app",
        "piece_style",
        "sample",
        "source_id",
        "square",
        "row",
        "col",
        "expected",
        "classifier_top1",
        "classifier_top1_score",
        "classifier_top2",
        "classifier_top2_score",
        "classifier_top3",
        "classifier_top3_score",
        "classifier_expected_rank",
        "classifier_top3_contains_expected",
        "classifier_top1_correct",
        "classifier_top1_margin",
        "expected_class_in_train",
        "trained_class_count",
    ]
    write_csv(out_dir / "cell_crop_classifier_strict_split_predictions.csv", strict_predictions, strict_prediction_fields)
    strict_confusion_rows, strict_confusion_fields = confusion_matrix_rows_by_split(strict_predictions)
    write_csv(out_dir / "cell_crop_classifier_strict_split_confusion_matrix.csv", strict_confusion_rows, strict_confusion_fields)
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_error_summary.csv",
        classifier_error_summary_rows(strict_predictions),
        [
            "split_mode",
            "fold_id",
            "expected",
            "classifier_top1",
            "count",
            "top3_contains_expected_count",
            "avg_margin",
            "min_margin",
            "max_margin",
        ],
    )
    strict_metrics_fields = [
        "split_mode",
        "fold_id",
        "train_example_count",
        "test_example_count",
        "train_source_count",
        "test_source_count",
        "train_class_count",
        "test_class_count",
        "missing_test_class_count",
        "missing_test_classes",
        "missing_test_example_count",
        "top1_accuracy",
        "top3_accuracy",
        "covered_class_example_count",
        "covered_class_top1_accuracy",
        "covered_class_top3_accuracy",
        "skipped",
    ]
    write_csv(out_dir / "cell_crop_classifier_strict_split_metrics_summary.csv", strict_metrics, strict_metrics_fields)
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_no_leak_audit.csv",
        strict_audit,
        [
            "split_mode",
            "fold_id",
            "heldout_app_styles",
            "train_example_count",
            "test_example_count",
            "train_source_count",
            "test_source_count",
            "train_same_app_style_count",
            "test_same_app_style_count",
            "train_piyo_chick_normal_count",
            "test_piyo_chick_normal_count",
            "train_piyo_chick_count",
            "test_piyo_chick_count",
            "overlap_source_count",
            "overlap_sources",
            "leak",
        ],
    )
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_class_coverage.csv",
        strict_coverage,
        ["split_mode", "fold_id", "label", "train_count", "test_count", "missing_in_train"],
    )
    missing_summary_rows = []
    for row in strict_metrics:
        missing_summary_rows.append(
            {
                "split_mode": row.get("split_mode", ""),
                "fold_id": row.get("fold_id", ""),
                "missing_test_class_count": row.get("missing_test_class_count", ""),
                "missing_test_classes": row.get("missing_test_classes", ""),
                "missing_test_example_count": row.get("missing_test_example_count", ""),
            }
        )
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_missing_class_summary.csv",
        missing_summary_rows,
        ["split_mode", "fold_id", "missing_test_class_count", "missing_test_classes", "missing_test_example_count"],
    )

    strict_residual_rows, strict_residual_audit = residual_summary_rows_by_split(analysis_dir, strict_predictions)
    strict_residual_fields = [
        "split_mode",
        "fold_id",
        "app",
        "piece_style",
        "sample",
        "source_id",
        "square",
        "row",
        "col",
        "expected",
        "baseline_top1",
        "baseline_expected_rank",
        "baseline_top_score",
        "baseline_expected_score",
        "classifier_top1",
        "classifier_top2",
        "classifier_top3",
        "classifier_top1_margin",
        "classifier_expected_rank",
        "classifier_top1_correct",
        "classifier_top3_contains_expected",
        "expected_class_in_train",
        "classifier_rank_delta_vs_baseline",
        "classifier_rank_improved",
        "baseline_candidate_rows",
        "baseline_candidate_source_leak_rows",
    ]
    write_csv(out_dir / "cell_crop_classifier_strict_split_b5_29_residual_summary.csv", strict_residual_rows, strict_residual_fields)

    strict_stable_rows = stable_degradation_rows_by_split(analysis_dir, labels_dir, strict_predictions)
    strict_stable_fields = [
        "split_mode",
        "fold_id",
        "app",
        "piece_style",
        "sample",
        "source_id",
        "square",
        "row",
        "col",
        "expected",
        "baseline_top1",
        "baseline_confidence",
        "baseline_source",
        "classifier_top1",
        "classifier_top2",
        "classifier_top3",
        "classifier_top1_margin",
        "classifier_expected_rank",
        "classifier_top3_contains_expected",
        "expected_class_in_train",
    ]
    write_csv(out_dir / "cell_crop_classifier_strict_split_stable_degradation_audit.csv", strict_stable_rows, strict_stable_fields)

    strict_margin_rows = confidence_margin_audit_rows(strict_predictions, strict_residual_rows, strict_stable_rows)
    strict_margin_fields = [
        "split_mode",
        "fold_id",
        "margin_threshold",
        "test_example_count",
        "accepted_example_count",
        "accepted_coverage",
        "accepted_top1_accuracy",
        "accepted_error_count",
        "residual_total",
        "residual_accepted_count",
        "residual_accepted_top1_fixes",
        "residual_accepted_wrong",
        "stable_degradation_total",
        "stable_degradation_accepted",
    ]
    write_csv(out_dir / "cell_crop_classifier_strict_split_margin_audit.csv", strict_margin_rows, strict_margin_fields)

    strict_margin_quantile_rows = confidence_margin_quantile_audit_rows(strict_predictions, strict_residual_rows, strict_stable_rows)
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_margin_quantile_audit.csv",
        strict_margin_quantile_rows,
        ["margin_quantile"] + strict_margin_fields,
    )

    strict_gate_rows = production_gate_candidate_rows(strict_margin_rows, strict_metrics, strict_audit)
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_production_gate_summary.csv",
        strict_gate_rows,
        [
            "split_mode",
            "fold_id",
            "margin_threshold",
            "production_gate_candidate",
            "blocking_reasons",
            "accepted_example_count",
            "accepted_coverage",
            "accepted_top1_accuracy",
            "accepted_error_count",
            "residual_total",
            "residual_accepted_top1_fixes",
            "stable_degradation_accepted",
            "missing_test_class_count",
            "no_leak_failures",
        ],
    )
    write_csv(
        out_dir / "cell_crop_classifier_strict_split_residual_no_leak_audit.csv",
        strict_residual_audit,
        [
            "split_mode",
            "fold_id",
            "app",
            "piece_style",
            "sample",
            "source_id",
            "row",
            "col",
            "classifier_prediction_found",
            "baseline_candidate_rows",
            "baseline_candidate_source_leak_rows",
            "leak",
        ],
    )
    write_strict_markdown_summary(out_dir, classifier_name, strict_metrics, strict_residual_rows, strict_stable_rows, strict_gate_rows)

    write_markdown_summary(out_dir, metrics, residual_rows, stable_rows, no_leak_rows, classifier_name)

    summary = {
        "out_dir": str(out_dir),
        "classifier": classifier_name,
        "examples": len(examples),
        "sources": len({ex.source_id for ex in examples}),
        "classes": len({ex.label for ex in examples}),
        "dataset_issues": len(dataset_issues),
        "loso_top1_accuracy": metrics["top1_accuracy"],
        "loso_top3_accuracy": metrics["top3_accuracy"],
        "b5_29_residual_rows": len(residual_rows),
        "b5_29_residual_top1_fixes": sum(1 for row in residual_rows if row.get("classifier_top1_correct") is True),
        "b5_29_residual_top3_contains": sum(1 for row in residual_rows if row.get("classifier_top3_contains_expected") is True),
        "stable_cell_degradation_rows": len(stable_rows),
        "no_leak_rows": len(no_leak_rows),
        "no_leak_failures": sum(1 for row in no_leak_rows if row.get("leak") is True),
        "strict_split_prediction_rows": len(strict_predictions),
        "strict_split_rows": len(strict_metrics),
        "strict_split_b5_29_residual_rows": len(strict_residual_rows),
        "strict_split_stable_degradation_rows": len(strict_stable_rows),
        "strict_split_no_leak_failures": sum(1 for row in strict_audit if row.get("leak") is True),
        "strict_split_gate_candidates": sum(1 for row in strict_gate_rows if row.get("production_gate_candidate") is True),
    }
    write_json(out_dir / "cell_crop_classifier_run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline no-leak occupied cell crop classifier probe.")
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--classifier", choices=("linear_svm", "logistic"), default="linear_svm")
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--no-write-crops", action="store_true")
    args = parser.parse_args()

    summary = run_probe(
        labels_dir=args.labels_dir,
        screenshots_dir=args.screenshots_dir,
        analysis_dir=args.analysis_dir,
        out_dir=args.out_dir,
        classifier_name=args.classifier,
        group_folds=args.group_folds,
        write_crops=not args.no_write_crops,
    )
    print(
        "OK: "
        f"examples={summary['examples']} "
        f"sources={summary['sources']} "
        f"classes={summary['classes']} "
        f"loso_top1={summary['loso_top1_accuracy']:.4f} "
        f"loso_top3={summary['loso_top3_accuracy']:.4f} "
        f"residual_top1_fixes={summary['b5_29_residual_top1_fixes']}/{summary['b5_29_residual_rows']} "
        f"out={summary['out_dir']}"
    )


if __name__ == "__main__":
    main()
