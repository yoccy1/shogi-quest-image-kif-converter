from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learned_piece_recognizer import hand_crop_feature_vector, load_model, save_model
from position_label_utils import HAND_PIECES


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_training_rows(manifest: Path, statuses: set[str]) -> list[dict[str, str]]:
    rows = []
    for row in read_manifest(manifest):
        piece = row.get("piece") or ""
        crop = Path(row.get("crop") or "")
        if row.get("status") not in statuses:
            continue
        if piece not in HAND_PIECES or not crop.exists():
            continue
        rows.append(row)
    return rows


def extract_dataset(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    vectors = []
    labels = []
    skipped = []
    for row in rows:
        crop_path = Path(row["crop"])
        try:
            with Image.open(crop_path) as image:
                vector = hand_crop_feature_vector(image.convert("RGB"))
        except Exception as exc:
            skipped.append({**row, "skip_reason": f"{type(exc).__name__}: {exc}"})
            continue
        if vector is None:
            skipped.append({**row, "skip_reason": "feature_extract_failed"})
            continue
        vectors.append(vector)
        labels.append(row["piece"])
    if not vectors:
        raise ValueError("no usable hand crop training rows")
    return np.vstack(vectors).astype("float32"), np.asarray(labels), skipped


def train_classifier(vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    label_set = sorted(set(labels.tolist()))
    if len(label_set) < 2:
        raise ValueError("hand crop classifier needs at least two piece classes")
    knn = KNeighborsClassifier(n_neighbors=min(5, len(labels)), weights="distance", metric="cosine", n_jobs=1)
    knn.fit(vectors, labels)
    sgd = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="modified_huber",
            alpha=0.0003,
            max_iter=1400,
            random_state=20260508,
            tol=1e-3,
        ),
    )
    sgd.fit(vectors, labels)
    warm = vectors[:1]
    knn.predict_proba(warm)
    sgd.predict_proba(warm)
    return {
        "enabled": True,
        "method": "hand_crop_knn_sgd",
        "knn": knn,
        "sgd": sgd,
        "labels": label_set,
        "feature_size": int(vectors.shape[1]),
        "training_rows": int(vectors.shape[0]),
        "label_counts": dict(Counter(labels.tolist())),
    }


def self_check(classifier: dict[str, Any], vectors: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = []
    for estimator_name in ("knn", "sgd"):
        estimator = classifier[estimator_name]
        predicted = estimator.predict(vectors)
        correct = int(np.count_nonzero(predicted == labels))
        predictions.append(
            {
                "estimator": estimator_name,
                "correct": correct,
                "total": int(len(labels)),
                "accuracy": round(correct / max(1, len(labels)), 4),
            }
        )
    return {"train_self_check": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description="trusted持ち駒クロップから軽量分類器を学習し、既存モデルへ埋め込みます。")
    parser.add_argument("--manifest", type=Path, default=Path("tools/out/hand_training_dataset_20260508/hand_training_manifest.csv"))
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument(
        "--enabled-family",
        action="append",
        default=[],
        help="Enable this classifier only for the given source family, e.g. 将棋クエスト:クラシック二文字駒.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=["trusted"],
        help="Training row status to include. Repeatable. Default: trusted.",
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    statuses = set(args.status)
    rows = load_training_rows(args.manifest, statuses)
    vectors, labels, skipped = extract_dataset(rows)
    classifier = train_classifier(vectors, labels)
    classifier["training_statuses"] = sorted(statuses)
    classifier["enabled_families"] = args.enabled_family

    model = load_model(args.base_model)
    model["hand_crop_classifier"] = classifier
    save_model(args.out_model, model)

    summary = {
        "manifest": str(args.manifest),
        "base_model": str(args.base_model),
        "out_model": str(args.out_model),
        "statuses": sorted(statuses),
        "enabled_families": args.enabled_family,
        "input_rows": len(rows),
        "trained_rows": int(vectors.shape[0]),
        "skipped_rows": len(skipped),
        "label_counts": classifier["label_counts"],
        **self_check(classifier, vectors, labels),
    }
    summary_path = args.summary or args.out_model.with_suffix(".hand_crop_summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
