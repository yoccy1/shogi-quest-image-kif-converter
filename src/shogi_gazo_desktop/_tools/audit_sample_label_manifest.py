from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PATH_COLUMNS = (
    "source_image",
    "style_image",
    "sorted_image",
    "source_label",
    "style_label",
    "sorted_label",
    "image",
    "report",
    "label_path",
)


SampleKey = tuple[str, str, str]


def key_text(key: SampleKey | None) -> str:
    if key is None:
        return ""
    return "/".join(key)


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def path_from_manifest(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue: str,
    key: SampleKey | None = None,
    *,
    path: Path | None = None,
    expected: str = "",
    actual: str = "",
    detail: str = "",
    manifest: Path | None = None,
    row_number: int | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "issue": issue,
            "app": key[0] if key else "",
            "style": key[1] if key else "",
            "sample": key[2] if key else "",
            "key": key_text(key),
            "path": repo_path(path) if path else "",
            "expected": expected,
            "actual": actual,
            "detail": detail,
            "manifest": repo_path(manifest) if manifest else "",
            "row": row_number or "",
        }
    )


def sample_key_for_path(path: Path, root: Path) -> SampleKey | None:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        return None
    return relative.parts[0], relative.parts[1], path.stem


def collect_images(root: Path, issues: list[dict[str, Any]]) -> dict[SampleKey, Path]:
    images: dict[SampleKey, Path] = {}
    for path in sorted(root.rglob("*")) if root.exists() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = sample_key_for_path(path, root)
        if key is None:
            add_issue(issues, "warning", "image_outside_app_style_layout", path=path)
            continue
        if key in images:
            add_issue(
                issues,
                "error",
                "duplicate_image_key",
                key,
                path=path,
                actual=repo_path(images[key]),
            )
            continue
        images[key] = path
    return images


def collect_labels(root: Path, issues: list[dict[str, Any]]) -> dict[SampleKey, Path]:
    labels: dict[SampleKey, Path] = {}
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        key = sample_key_for_path(path, root)
        if key is None:
            add_issue(issues, "warning", "label_outside_app_style_layout", path=path)
            continue
        if key in labels:
            add_issue(
                issues,
                "error",
                "duplicate_label_key",
                key,
                path=path,
                actual=repo_path(labels[key]),
            )
            continue
        labels[key] = path
    return labels


def is_initial_sample(sample: str) -> bool:
    return "初期配置" in sample


def resolve_label_image_path(label_path: Path, label: dict[str, Any], screenshots_root: Path, key: SampleKey) -> Path:
    image_value = label.get("image")
    if not image_value:
        return screenshots_root / key[0] / key[1] / f"{key[2]}.png"

    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path

    candidates = [
        (label_path.parent / image_path).resolve(),
        (screenshots_root / key[0] / key[1] / image_path.name).resolve(),
        (screenshots_root / image_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def key_for_existing_image(path: Path, screenshots_root: Path) -> SampleKey | None:
    try:
        relative = path.resolve().relative_to(screenshots_root.resolve())
    except ValueError:
        return None
    if len(relative.parts) < 3:
        return None
    return relative.parts[0], relative.parts[1], path.stem


def audit_label_json(
    labels: dict[SampleKey, Path],
    screenshots_root: Path,
    issues: list[dict[str, Any]],
) -> None:
    for key, label_path in sorted(labels.items()):
        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception as exc:
            add_issue(
                issues,
                "error",
                "label_json_invalid",
                key,
                path=label_path,
                detail=f"{type(exc).__name__}: {exc}",
            )
            continue

        image_path = resolve_label_image_path(label_path, data, screenshots_root, key)
        if not image_path.exists():
            add_issue(
                issues,
                "error",
                "label_image_missing",
                key,
                path=label_path,
                expected=repo_path(image_path),
            )
        else:
            image_key = key_for_existing_image(image_path, screenshots_root)
            if image_key is None:
                add_issue(
                    issues,
                    "warning",
                    "label_image_outside_sample_root",
                    key,
                    path=label_path,
                    actual=repo_path(image_path),
                )
            elif image_key != key:
                add_issue(
                    issues,
                    "error",
                    "label_image_key_mismatch",
                    key,
                    path=label_path,
                    expected=key_text(key),
                    actual=key_text(image_key),
                )

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        piece_style = metadata.get("piece_style")
        if piece_style and piece_style != key[1]:
            add_issue(
                issues,
                "warning",
                "metadata_piece_style_mismatch",
                key,
                path=label_path,
                expected=key[1],
                actual=str(piece_style),
            )


def audit_sample_label_pairs(
    images: dict[SampleKey, Path],
    labels: dict[SampleKey, Path],
    issues: list[dict[str, Any]],
    *,
    strict_initial_labels: bool,
) -> None:
    for key, image_path in sorted(images.items()):
        if key in labels:
            continue
        if is_initial_sample(key[2]) and not strict_initial_labels:
            add_issue(issues, "warning", "missing_initial_label", key, path=image_path)
        else:
            add_issue(issues, "error", "missing_label", key, path=image_path)

    for key, label_path in sorted(labels.items()):
        if key not in images:
            add_issue(issues, "error", "orphan_label", key, path=label_path)


def audit_duplicate_stems(
    images: dict[SampleKey, Path],
    labels: dict[SampleKey, Path],
    issues: list[dict[str, Any]],
) -> None:
    by_image_stem: dict[str, list[SampleKey]] = defaultdict(list)
    by_label_stem: dict[str, list[SampleKey]] = defaultdict(list)
    for key in images:
        by_image_stem[key[2]].append(key)
    for key in labels:
        by_label_stem[key[2]].append(key)

    for stem, keys in sorted(by_image_stem.items()):
        if len(keys) > 1:
            add_issue(
                issues,
                "warning",
                "duplicate_sample_stem",
                keys[0],
                actual=", ".join(key_text(key) for key in sorted(keys)),
            )
    for stem, keys in sorted(by_label_stem.items()):
        if len(keys) > 1:
            add_issue(
                issues,
                "warning",
                "duplicate_label_stem",
                keys[0],
                actual=", ".join(key_text(key) for key in sorted(keys)),
            )


def manifest_key(row: dict[str, str]) -> SampleKey | None:
    app = row.get("app") or row.get("app_name") or ""
    style = row.get("piece_style") or row.get("glyph") or row.get("glyph_count_label") or ""
    sample = row.get("sample") or ""
    if not sample:
        image_value = row.get("style_image") or row.get("sorted_image") or row.get("image") or ""
        if image_value:
            sample = Path(image_value).stem
    if app and style and sample:
        return app, style, sample
    return None


def status_says_labeled(status: str) -> bool:
    return status in {"教師ラベルあり", "初期配置ラベル", "暗黙の初期配置", "教師ラベル候補あり"}


def status_says_missing(status: str) -> bool:
    return status in {"教師ラベル未作成", "教師ラベルなし", "label_missing"}


def audit_manifest(
    manifest_path: Path,
    images: dict[SampleKey, Path],
    labels: dict[SampleKey, Path],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if not manifest_path.exists():
        add_issue(issues, "error", "manifest_missing", manifest=manifest_path)
        return {"manifest": repo_path(manifest_path), "rows": 0}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    seen: set[SampleKey] = set()
    for index, row in enumerate(rows, start=2):
        key = manifest_key(row)
        if key is None:
            add_issue(issues, "error", "manifest_key_missing", manifest=manifest_path, row_number=index)
        else:
            if key in seen:
                add_issue(issues, "error", "manifest_duplicate_key", key, manifest=manifest_path, row_number=index)
            seen.add(key)

            if key not in images:
                add_issue(issues, "error", "manifest_sample_missing_image", key, manifest=manifest_path, row_number=index)

            status = row.get("label_status") or ""
            has_label = (row.get("has_label") or "").strip().lower()
            exact_label_exists = key in labels
            if status_says_labeled(status) and not exact_label_exists:
                add_issue(
                    issues,
                    "error",
                    "manifest_labeled_but_exact_label_missing",
                    key,
                    expected="exact label under labels_root/app/style/sample.json",
                    actual=status,
                    manifest=manifest_path,
                    row_number=index,
                )
            if status_says_missing(status) and exact_label_exists:
                add_issue(
                    issues,
                    "warning",
                    "manifest_label_status_stale",
                    key,
                    expected="label_status should reflect existing exact label",
                    actual=status,
                    manifest=manifest_path,
                    row_number=index,
                )
            if has_label == "true" and not exact_label_exists:
                add_issue(
                    issues,
                    "error",
                    "manifest_has_label_but_exact_label_missing",
                    key,
                    manifest=manifest_path,
                    row_number=index,
                )

        for column in PATH_COLUMNS:
            value = (row.get(column) or "").strip()
            if not value:
                continue
            path = path_from_manifest(value)
            if not path.exists():
                add_issue(
                    issues,
                    "error",
                    "manifest_path_missing",
                    key,
                    path=path,
                    detail=f"{column} does not exist",
                    manifest=manifest_path,
                    row_number=index,
                )

    return {"manifest": repo_path(manifest_path), "rows": len(rows)}


def audit_skipped(analysis_dir: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    skipped_path = analysis_dir / "piece_style_evaluation_skipped.json"
    if not skipped_path.exists():
        return {"analysis_dir": repo_path(analysis_dir), "skipped": None}
    try:
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_issue(
            issues,
            "error",
            "skipped_json_invalid",
            path=skipped_path,
            detail=f"{type(exc).__name__}: {exc}",
        )
        return {"analysis_dir": repo_path(analysis_dir), "skipped": None}
    for item in skipped:
        key = (
            str(item.get("app") or ""),
            str(item.get("piece_style") or item.get("glyph") or ""),
            str(item.get("sample") or ""),
        )
        reason = str(item.get("reason") or "")
        severity = "error" if reason == "label_missing" else "warning"
        add_issue(
            issues,
            severity,
            "evaluation_skipped_sample",
            key if all(key) else None,
            path=skipped_path,
            detail=reason,
        )
    return {"analysis_dir": repo_path(analysis_dir), "skipped": len(skipped)}


def write_csv(path: Path, issues: list[dict[str, Any]]) -> None:
    fields = [
        "severity",
        "issue",
        "app",
        "style",
        "sample",
        "key",
        "path",
        "expected",
        "actual",
        "detail",
        "manifest",
        "row",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: issue.get(field, "") for field in fields} for issue in issues)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sample image, label, and manifest key consistency.")
    parser.add_argument("--screenshots-root", type=Path, default=ROOT / "tools" / "samples" / "screenshots_by_app_piece_style")
    parser.add_argument("--labels-root", type=Path, default=ROOT / "tools" / "samples" / "labels" / "boards_by_app_piece_style")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--analysis-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tools" / "out" / "sample_label_manifest_audit")
    parser.add_argument("--strict-initial-labels", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    issues: list[dict[str, Any]] = []
    screenshots_root = args.screenshots_root.resolve()
    labels_root = args.labels_root.resolve()

    if not screenshots_root.exists():
        add_issue(issues, "error", "screenshots_root_missing", path=screenshots_root)
    if not labels_root.exists():
        add_issue(issues, "error", "labels_root_missing", path=labels_root)

    images = collect_images(screenshots_root, issues)
    labels = collect_labels(labels_root, issues)
    audit_sample_label_pairs(images, labels, issues, strict_initial_labels=args.strict_initial_labels)
    audit_label_json(labels, screenshots_root, issues)
    audit_duplicate_stems(images, labels, issues)

    manifest_paths = args.manifest
    if not manifest_paths:
        default_manifest = ROOT / "tools" / "samples" / "piece_style_manifest.csv"
        if default_manifest.exists():
            manifest_paths = [default_manifest]
    manifest_summaries = [audit_manifest(path, images, labels, issues) for path in manifest_paths]
    skipped_summaries = [audit_skipped(path, issues) for path in args.analysis_dir]

    summary = {
        "screenshots_root": repo_path(screenshots_root),
        "labels_root": repo_path(labels_root),
        "images": len(images),
        "labels": len(labels),
        "manifests": manifest_summaries,
        "analysis_dirs": skipped_summaries,
        "issue_count": len(issues),
        "by_severity": dict(sorted(Counter(issue["severity"] for issue in issues).items())),
        "by_issue": dict(sorted(Counter(issue["issue"] for issue in issues).items())),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "issues": issues}
    (args.out_dir / "sample_label_manifest_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "sample_label_manifest_audit.csv", issues)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_error and summary["by_severity"].get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
