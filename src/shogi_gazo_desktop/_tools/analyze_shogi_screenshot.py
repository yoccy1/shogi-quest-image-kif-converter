from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from detect_board_grid import detect_grid, draw_overlay, iter_images, report_dict, save_cells
from recognize_board_pieces import (
    default_board_labels_dir,
    default_calibration_dir,
    default_template_path,
    recognize_cells,
    write_recognition_outputs,
)
from render_recognized_position import default_board_path, render_outputs


def analyze_image(
    image_path: Path,
    out_root: Path,
    template_path: Path,
    board_path: Path,
    recognition_method: str,
    calibration_dir: Path,
    board_labels_dir: Path,
    exclude_self_calibration_source: bool = False,
    apply_label_corrections: bool = True,
    write_debug_images: bool = True,
    render_position: bool = True,
    fast_recognition: bool = False,
    label_oracle_baseline: bool = False,
) -> dict:
    started_ns = time.perf_counter_ns()
    image = Image.open(image_path).convert("RGB")
    detection = detect_grid(image)
    image_out = out_root / image_path.stem
    image_out.mkdir(parents=True, exist_ok=True)

    grid_report = report_dict(image, detection)
    if write_debug_images:
        draw_overlay(image, detection).save(image_out / "grid_overlay.png")
    write_json(image_out / "report.json", grid_report)

    piece_report = None
    if detection is not None:
        rect = detection.board_rect
        if write_debug_images:
            image.crop((rect.left, rect.top, rect.right, rect.bottom)).save(image_out / "board_area.png")
        grid_rect = (
            detection.vertical.positions[0],
            detection.horizontal.positions[0],
            detection.vertical.positions[-1],
            detection.horizontal.positions[-1],
        )
        if write_debug_images:
            image.crop(grid_rect).save(image_out / "grid_area.png")
        cells_dir = image_out / "cells"
        save_cells(image, detection, cells_dir)
        recognition_cells_dir = image_out / "recognition_cells"
        save_cells(image, detection, recognition_cells_dir, pad_x_ratio=0.08, pad_y_ratio=0.18)
        piece_report = recognize_cells(
            recognition_cells_dir,
            template_path,
            method=recognition_method,
            empty_cells_dir=cells_dir,
            calibration_dir=calibration_dir,
            calibration_source_hint=image_path.stem,
            board_labels_dir=board_labels_dir,
            exclude_self_calibration_source=exclude_self_calibration_source,
            apply_label_corrections=apply_label_corrections,
            fast_recognition=fast_recognition,
            label_oracle_baseline=label_oracle_baseline,
        )
        write_recognition_outputs(
            piece_report,
            image_out,
            Path(piece_report["cells_dir"]),
            write_debug_images=write_debug_images,
        )

    analysis = {
        "image": str(image_path),
        "output_dir": str(image_out),
        "grid": grid_report,
        "piece_recognition": piece_report,
        "cells": piece_report["cells"] if piece_report else [],
    }
    analysis["timing"] = timing_report(started_ns)
    write_json(image_out / "analysis_report.json", analysis)
    if piece_report is not None and render_position:
        outputs = render_outputs(
            report_path=image_out / "analysis_report.json",
            out_dir=image_out,
            board_path=board_path,
            pieces_path=template_path,
            screenshot_path=image_path,
        )
        analysis["rendered_outputs"] = {
            "confirmed": str(outputs.confirmed_path),
            "candidates": str(outputs.candidates_path),
            "comparison": str(outputs.comparison_path) if outputs.comparison_path else None,
        }
        analysis["timing"] = timing_report(started_ns)
        write_json(image_out / "analysis_report.json", analysis)
    elif piece_report is not None:
        analysis["rendered_outputs"] = {
            "confirmed": None,
            "candidates": None,
            "comparison": None,
            "skipped": True,
        }
        analysis["timing"] = timing_report(started_ns)
        write_json(image_out / "analysis_report.json", analysis)
    return analysis


def analyze_images(
    input_path: Path,
    out_root: Path,
    template_path: Path,
    board_path: Path,
    recognition_method: str,
    calibration_dir: Path,
    board_labels_dir: Path,
    exclude_self_calibration_source: bool = False,
    apply_label_corrections: bool = True,
    write_debug_images: bool = True,
    render_position: bool = True,
    fast_recognition: bool = False,
    label_oracle_baseline: bool = False,
) -> list[dict]:
    reports: list[dict] = []
    for image_path in iter_images(input_path):
        report = analyze_image(
            image_path,
            out_root,
            template_path,
            board_path,
            recognition_method,
            calibration_dir,
            board_labels_dir,
            exclude_self_calibration_source,
            apply_label_corrections,
            write_debug_images,
            render_position,
            fast_recognition,
            label_oracle_baseline,
        )
        reports.append(report)
        timing = report.get("timing") or {}
        elapsed_ms = timing.get("processing_time_ms")
        if isinstance(elapsed_ms, (int, float)):
            print(f"{image_path.name}: {elapsed_ms:.1f} ms", flush=True)
    return reports


def timing_report(started_ns: int) -> dict:
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "processing_time_ns": elapsed_ns,
        "processing_time_ms": round(elapsed_ns / 1_000_000, 3),
        "processing_time_seconds": round(elapsed_ns / 1_000_000_000, 3),
    }


def write_json(
    path: Path,
    data: dict,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a shogi board grid and recognize pieces in one pass.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("tools/samples/screenshots"),
        help="Screenshot image or directory. Defaults to tools/samples/screenshots.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tools/out/analysis"),
        help="Output directory for grid and piece recognition reports.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=default_template_path(),
        help="Path to shogi_pieces.png. Defaults to the Android bundled sprite sheet.",
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=default_board_path(),
        help="Path to shogi_board.png. Defaults to the Android bundled board image.",
    )
    parser.add_argument(
        "--method",
        choices=("hog_svm", "opencv", "legacy"),
        default="hog_svm",
        help="Piece recognition method. Defaults to hog_svm.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=default_calibration_dir(),
        help="Initial-position screenshots used as extra recognition templates. Defaults to tools/samples/screenshots/初期配置.",
    )
    parser.add_argument(
        "--board-labels-dir",
        type=Path,
        default=default_board_labels_dir(),
        help="Labeled board screenshots used as extra recognition templates. Defaults to tools/samples/labels/boards.",
    )
    parser.add_argument(
        "--exclude-self-calibration-source",
        action="store_true",
        help="Exclude templates from the same screenshot source during recognition. Use this for holdout evaluation.",
    )
    parser.add_argument(
        "--no-label-corrections",
        action="store_true",
        help="Do not apply exact board labels as final corrections even when a matching label JSON exists.",
    )
    parser.add_argument(
        "--no-debug-images",
        action="store_true",
        help="Write JSON reports only; skip debug overlays, preview PNGs, and candidate grids.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendered position and comparison images.",
    )
    parser.add_argument(
        "--fast-output",
        action="store_true",
        help="Shortcut for --no-debug-images --no-render. Keeps JSON reports and cell crops.",
    )
    parser.add_argument(
        "--fast-recognition",
        action="store_true",
        help="Use the faster recognizer path that skips slow full OpenCV fallback matching.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for --fast-output --fast-recognition.",
    )
    parser.add_argument(
        "--label-oracle-baseline",
        action="store_true",
        help="Diagnostic only: allow teacher labels to be used as oracle output/corrections.",
    )
    args = parser.parse_args()

    reports = analyze_images(
        args.input,
        args.out,
        args.template,
        args.board,
        args.method,
        args.calibration_dir,
        args.board_labels_dir,
        args.exclude_self_calibration_source,
        not args.no_label_corrections,
        not (args.no_debug_images or args.fast_output or args.fast),
        not (args.no_render or args.fast_output or args.fast),
        args.fast_recognition or args.fast,
        args.label_oracle_baseline,
    )
    ok_count = sum(1 for report in reports if report["grid"].get("detected"))
    print(f"OK: analyzed {len(reports)} image(s), grid detected for {ok_count}. Output: {args.out}")


if __name__ == "__main__":
    main()
