from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

from evaluate_piece_recognition import model_excluded_sources, read_key_value_metadata
from position_label_utils import find_label_path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "com.example.shogigazo"
TEST_RUNNER = f"{PACKAGE_NAME}.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "com.example.shogigazo.eval.RecognitionDeviceEvalTest"
REMOTE_ROOT = f"/sdcard/Android/data/{PACKAGE_NAME}/files/recognition_eval"
INTERNAL_INPUT_DIR = f"/data/data/{PACKAGE_NAME}/files/recognition_eval/input"
INTERNAL_INPUT_REL = "files/recognition_eval/input"
INTERNAL_REPORTS_REL = "files/recognition_eval/reports"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MANIFEST_FIELDS = [
    "app",
    "piece_style",
    "sample",
    "kind",
    "label_status",
    "label_match",
    "label_path",
    "image",
    "report",
    "seconds",
    "piece",
    "empty",
    "unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Android in-app recognition over sample screenshots and evaluate the reports.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=ROOT / "tools" / "samples" / "screenshots_by_app_piece_style",
        help="Screenshot sample root grouped as app/style/images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "tools" / "samples" / "labels" / "boards_by_app_piece_style",
        help="Board/hand label root used by the Python evaluator.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "tools" / "out" / "android_device_eval",
        help="Local output root for pulled reports and evaluation summaries.",
    )
    parser.add_argument("--run-id", default="", help="Report run id. Defaults to android_eval_<timestamp>.")
    parser.add_argument("--serial", default="", help="ADB serial. Defaults to a connected physical device, then emulator.")
    parser.add_argument("--adb", type=Path, default=None, help="Path to adb.exe. Defaults to Android SDK adb.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build APKs before running.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install APKs before running.")
    parser.add_argument("--skip-push", action="store_true", help="Reuse samples already pushed to the device.")
    parser.add_argument("--skip-instrumentation", action="store_true", help="Reuse reports already generated on the device.")
    parser.add_argument("--skip-pull", action="store_true", help="Reuse reports already pulled locally.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Only generate/pull reports; do not run Python evaluation.")
    parser.add_argument("--without-hands", action="store_true", help="Evaluate board cells only.")
    parser.add_argument("--app", default="", help="Only run images whose path includes this app label.")
    parser.add_argument("--style", default="", help="Only run images whose path includes this piece style label.")
    parser.add_argument("--sample-contains", default="", help="Only run images whose sample stem contains this text.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to run on the device.")
    parser.add_argument("--no-known-samples", action="store_true", help="Disable known_position_samples matching inside the Android app.")
    parser.add_argument("--no-initial-position-pattern", action="store_true", help="Disable initial-position pattern rescue inside the Android app.")
    parser.add_argument("--position-prior-scale", type=float, default=1.0, help="Scale Android position/template priors from 0.0 to 1.0.")
    parser.add_argument("--no-board-ocr", action="store_true", help="Skip Android board OCR merge after template analysis.")
    parser.add_argument(
        "--collect-candidate-diagnostics",
        action="store_true",
        help="Include extended per-cell board candidate diagnostics in Android reports.",
    )
    parser.add_argument("--strict-leak-guard", action="store_true", help="Fail evaluation when report sources contain each sample name.")
    parser.add_argument(
        "--allow-missing-excluded-source",
        action="store_true",
        help="With --strict-leak-guard, allow old reports that lack model.excluded_source metadata.",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0, help="Per-image timing threshold used by evaluation summaries.")
    parser.add_argument("--require-speed", action="store_true", help="When used with --require-perfect, also require no timing over-limit samples.")
    parser.add_argument(
        "--require-perfect",
        action="store_true",
        help="Return exit code 2 unless evaluated metrics are perfect.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(quote_arg(part) for part in command)
    print(f"$ {printable}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        env=env,
    )
    if capture and completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def run_binary(
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    cwd: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    printable = " ".join(quote_arg(part) for part in command)
    print(f"$ {printable}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout.decode("utf-8", errors="replace"), end="")
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def run_capture_binary(
    command: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    printable = " ".join(quote_arg(part) for part in command)
    print(f"$ {printable}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr:
        print(completed.stderr.decode("utf-8", errors="replace"), end="")
    if check and completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout[:1000].decode("utf-8", errors="replace"), end="")
        raise SystemExit(completed.returncode)
    return completed


def quote_arg(value: str) -> str:
    if not value:
        return '""'
    if any(char.isspace() for char in value) or any(char in value for char in '"&()[]{};'):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def find_adb(explicit: Path | None) -> str:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if android_home:
        candidates.append(Path(android_home) / "platform-tools" / "adb.exe")
        candidates.append(Path(android_home) / "platform-tools" / "adb")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "adb"


def adb_cmd(adb: str, serial: str, *args: str) -> list[str]:
    command = [adb]
    if serial:
        command += ["-s", serial]
    command += list(args)
    return command


def list_devices(adb: str) -> list[tuple[str, str]]:
    completed = run([adb, "devices", "-l"], capture=True, check=True)
    devices: list[tuple[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, status = parts[0], parts[1]
        if status == "device":
            devices.append((serial, line))
    return devices


def choose_device(adb: str, requested_serial: str) -> str:
    devices = list_devices(adb)
    if requested_serial:
        if any(serial == requested_serial for serial, _ in devices):
            return requested_serial
        lines = "\n".join(line for _, line in devices) or "(no authorized devices)"
        raise SystemExit(f"Requested device {requested_serial!r} is not available.\n{lines}")
    if not devices:
        raise SystemExit("No authorized Android devices found by adb devices -l.")
    physical = [item for item in devices if not item[0].startswith("emulator-")]
    serial, line = (physical or devices)[0]
    print(f"Using device: {line}", flush=True)
    return serial


def gradle_env() -> dict[str, str]:
    env = os.environ.copy()
    bundled_jbr = Path(r"C:\Program Files\Android\Android Studio\jbr")
    if not env.get("JAVA_HOME") and bundled_jbr.exists():
        env["JAVA_HOME"] = str(bundled_jbr)
    return env


def build_apks(env: dict[str, str]) -> None:
    run([str(ROOT / "gradlew.bat"), "assembleDebug", "assembleDebugAndroidTest"], env=env)


def install_apks(adb: str, serial: str) -> None:
    app_apk = ROOT / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
    test_apk = ROOT / "app" / "build" / "outputs" / "apk" / "androidTest" / "debug" / "app-debug-androidTest.apk"
    for apk in [app_apk, test_apk]:
        if not apk.exists():
            raise SystemExit(f"APK not found: {apk}")
        run(adb_cmd(adb, serial, "install", "-r", str(apk)))


def shell_quote_remote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


def prepare_remote_input(adb: str, serial: str, sample_dir: Path) -> None:
    if not sample_dir.exists():
        raise SystemExit(f"Sample directory not found: {sample_dir}")
    run_as(adb, serial, "rm", "-rf", INTERNAL_INPUT_REL)
    run_as(adb, serial, "mkdir", "-p", f"{INTERNAL_INPUT_REL}/images")
    rows = ["path\tapp\tstyle\tsample\tsource_relative_path"]
    images = sorted(
        path
        for path in sample_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for index, image in enumerate(images, start=1):
        relative = image.relative_to(sample_dir)
        parts = relative.parts
        if len(parts) < 3:
            continue
        app, style, sample = parts[0], parts[1], image.stem
        remote_name = f"images/{index:04d}{image.suffix.lower()}"
        write_app_internal_file(
            adb,
            serial,
            f"{INTERNAL_INPUT_REL}/{remote_name}",
            image.read_bytes(),
        )
        rows.append(
            "\t".join(
                [
                    remote_name,
                    app,
                    style,
                    sample,
                    relative.as_posix(),
                ],
            ),
        )
    write_app_internal_file(
        adb,
        serial,
        f"{INTERNAL_INPUT_REL}/input_manifest.tsv",
        ("\n".join(rows) + "\n").encode("utf-8"),
    )
    print(f"Prepared {len(rows) - 1} input images in app internal storage.", flush=True)


def run_as(adb: str, serial: str, *args: str) -> None:
    run(adb_cmd(adb, serial, "shell", "run-as", PACKAGE_NAME, *args))


def write_app_internal_file(adb: str, serial: str, relative_path: str, content: bytes) -> None:
    command = adb_cmd(
        adb,
        serial,
        "exec-in",
        "run-as",
        PACKAGE_NAME,
        "sh",
        "-c",
        f"cat > {shell_quote_remote(relative_path)}",
    )
    run_binary(command, input_bytes=content)


def clear_remote_reports(adb: str, serial: str, run_id: str) -> None:
    run_as(adb, serial, "rm", "-rf", f"{INTERNAL_REPORTS_REL}/{run_id}")
    run_as(adb, serial, "mkdir", "-p", f"{INTERNAL_REPORTS_REL}/{run_id}")


def run_instrumentation(
    adb: str,
    serial: str,
    run_id: str,
    app: str,
    style: str,
    sample_contains: str,
    limit: int,
    use_known_samples: bool,
    use_initial_position_pattern: bool,
    position_prior_scale: float,
    use_board_ocr: bool,
    collect_candidate_diagnostics: bool,
) -> int:
    args = [
        "shell",
        "am",
        "instrument",
        "-w",
        "-e",
        "class",
        TEST_CLASS,
        "-e",
        "evalRunId",
        run_id,
        "-e",
        "evalInputDir",
        INTERNAL_INPUT_DIR,
        "-e",
        "evalUseKnownSamples",
        str(use_known_samples).lower(),
        "-e",
        "evalUseInitialPositionPattern",
        str(use_initial_position_pattern).lower(),
        "-e",
        "evalPositionPriorScale",
        f"{position_prior_scale:.4f}",
        "-e",
        "evalUseBoardOcr",
        str(use_board_ocr).lower(),
        "-e",
        "evalCollectCandidateDiagnostics",
        str(collect_candidate_diagnostics).lower(),
    ]
    if app:
        args += ["-e", "evalApp", app]
    if style:
        args += ["-e", "evalStyle", style]
    if sample_contains:
        args += ["-e", "evalSampleContains", sample_contains]
    if limit > 0:
        args += ["-e", "evalLimit", str(limit)]
    args.append(TEST_RUNNER)
    completed = run(
        adb_cmd(adb, serial, *args),
        check=False,
        capture=True,
    )
    return completed.returncode


def remove_tree_inside(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise SystemExit(f"Refusing to remove path outside output root: {path}")
    if path.exists():
        shutil.rmtree(path)


def pull_reports(adb: str, serial: str, out_root: Path, run_id: str) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    local_run = out_root / run_id
    remove_tree_inside(local_run, out_root)
    completed = run_capture_binary(
        adb_cmd(
            adb,
            serial,
            "exec-out",
            "run-as",
            PACKAGE_NAME,
            "sh",
            "-c",
            f"cd {shell_quote_remote(INTERNAL_REPORTS_REL)} && tar -cf - {shell_quote_remote(run_id)}",
        ),
    )
    safe_extract_tar(completed.stdout, out_root)
    if not local_run.exists():
        raise SystemExit(f"Pulled report directory not found: {local_run}")
    return local_run


def safe_extract_tar(data: bytes, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"Refusing unsafe tar member: {member.name}")
            target = (destination / member_path).resolve()
            if target != destination and destination not in target.parents:
                raise SystemExit(f"Refusing tar member outside output root: {member.name}")
        archive.extractall(destination)


def read_elapsed_seconds(meta_path: Path) -> float | None:
    if not meta_path.exists():
        return None
    for line in meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key == "elapsed_seconds":
            try:
                return float(value)
            except ValueError:
                return None
    return None


def report_has_exclusion_metadata(report_path: Path) -> bool:
    companion = read_key_value_metadata(report_path.with_name("sample_meta.txt"))
    if companion.get("excluded_template_source"):
        return True
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    report = data.get("piece_recognition") if isinstance(data, dict) and "piece_recognition" in data else data
    model = report.get("model") if isinstance(report, dict) else None
    excluded_sources, _ = model_excluded_sources(model)
    return bool(excluded_sources)


def reports_missing_exclusion_metadata(run_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for _, _, _, report_path in iter_report_dirs(run_dir):
        if not report_has_exclusion_metadata(report_path):
            missing.append(report_path)
    return missing


def iter_report_dirs(run_dir: Path) -> Iterable[tuple[str, str, str, Path]]:
    for report in sorted(run_dir.glob("*/*/*/piece_report.json")):
        sample_dir = report.parent
        style_dir = sample_dir.parent
        app_dir = style_dir.parent
        if app_dir.name.startswith("_"):
            continue
        yield app_dir.name, style_dir.name, sample_dir.name, report


def image_path_for_sample(sample_dir: Path, app: str, style: str, sample: str) -> Path | None:
    base = sample_dir / app / style
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = base / f"{sample}{extension}"
        if candidate.exists():
            return candidate
    for candidate in base.glob(f"{sample}.*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def count_states(report_path: Path) -> tuple[int, int, int]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0, 0
    piece = empty = unknown = 0
    for cell in report.get("cells") or []:
        state = cell.get("state")
        if state == "piece":
            piece += 1
        elif state == "empty":
            empty += 1
        else:
            unknown += 1
    return piece, empty, unknown


def exact_label_path(labels_dir: Path, sample: str, app: str, style: str) -> Path:
    return labels_dir / app / style / f"{sample}.json"


def label_info(labels_dir: Path, sample: str, app: str, style: str) -> tuple[str, str, str]:
    exact = exact_label_path(labels_dir, sample, app, style)
    if exact.exists():
        return "教師ラベルあり", "exact_app_style_sample", str(exact.resolve())

    fallback = find_label_path(labels_dir, sample, app, style)
    if fallback.exists():
        return "教師ラベル候補あり", "fallback_not_exact", str(fallback.resolve())
    return "教師ラベル未作成", "missing", ""


def write_manifest(run_dir: Path, sample_dir: Path, labels_dir: Path) -> Path:
    rows: list[dict[str, str]] = []
    for app, style, sample, report in iter_report_dirs(run_dir):
        image = image_path_for_sample(sample_dir, app, style, sample)
        label_status, label_match, label_path = label_info(labels_dir, sample, app, style)
        seconds = read_elapsed_seconds(report.parent / "device_eval_meta.txt")
        piece, empty, unknown = count_states(report)
        rows.append(
            {
                "app": app,
                "piece_style": style,
                "sample": sample,
                "kind": image.suffix.lower().lstrip(".") if image else "",
                "label_status": label_status,
                "label_match": label_match,
                "label_path": label_path,
                "image": str(image.resolve()) if image else "",
                "report": str(report.resolve()),
                "seconds": f"{seconds:.4f}" if seconds is not None else "",
                "piece": str(piece),
                "empty": str(empty),
                "unknown": str(unknown),
            },
        )
    manifest = run_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest with {len(rows)} reports: {manifest}", flush=True)
    return manifest


def evaluate(
    run_dir: Path,
    labels_dir: Path,
    include_hands: bool,
    strict_leak_guard: bool,
    max_seconds: float,
    allow_missing_excluded_source: bool = False,
) -> dict[str, object] | None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "evaluate_analysis_by_app_piece_style.py"),
        str(run_dir),
        "--labels-dir",
        str(labels_dir),
    ]
    if include_hands:
        command.append("--include-hands")
    if strict_leak_guard:
        command.append("--strict-leak-guard")
        missing_metadata = reports_missing_exclusion_metadata(run_dir)
        if missing_metadata and allow_missing_excluded_source:
            command.append("--allow-missing-excluded-source")
            print(
                "Strict leak guard is explicitly allowing missing excluded_source metadata for "
                f"{len(missing_metadata)} existing report(s). "
                "Add Android report model.excluded_source metadata to enforce this gate.",
                flush=True,
            )
        elif missing_metadata:
            print(
                "Strict leak guard found "
                f"{len(missing_metadata)} report(s) missing excluded_source metadata. "
                "Evaluation will fail unless report metadata is fixed or "
                "--allow-missing-excluded-source is explicitly provided for legacy reports.",
                flush=True,
            )
    command += ["--max-seconds", f"{max_seconds:.4f}"]
    completed = run(command, check=False, capture=True)
    summary_path = run_dir / "piece_style_evaluation_summary.json"
    if not summary_path.exists():
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = summary.get("overall") or {}
    print("\nAndroid recognition summary:", flush=True)
    interesting_keys = [
        "evaluated_samples",
        "skipped_samples",
        "errors",
        "hand_errors",
        "false_empty_on_piece",
        "false_piece_on_empty",
        "unknown_on_piece",
        "high_confidence_errors",
        "leak_errors",
        "confirmed_identity_accuracy",
        "empty_accuracy",
        "top1_accuracy",
        "top3_accuracy",
    ]
    for key in interesting_keys:
        if key in overall:
            print(f"  {key}: {overall[key]}", flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return summary


def is_perfect(summary: dict[str, object] | None, require_speed: bool = False) -> bool:
    if not summary:
        return False
    overall = summary.get("overall") or {}
    zero_keys = [
        "errors",
        "hand_errors",
        "false_empty_on_piece",
        "false_piece_on_empty",
        "unknown_on_piece",
        "high_confidence_errors",
        "leak_errors",
        "skipped_samples",
    ]
    if not all(int(overall.get(key) or 0) == 0 for key in zero_keys):
        return False
    if require_speed:
        timing = overall.get("timing") or {}
        if int(timing.get("over_limit_count") or 0) != 0:
            return False
        if int(timing.get("missing_seconds_count") or 0) != 0:
            return False
    return True


def main() -> int:
    args = parse_args()
    adb = find_adb(args.adb)
    run_id = args.run_id or time.strftime("android_eval_%Y%m%d_%H%M%S")
    sample_dir = args.sample_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    out_root = args.out_root.resolve()
    serial = choose_device(adb, args.serial)
    env = gradle_env()

    if not args.skip_build:
        build_apks(env)
    if not args.skip_install:
        install_apks(adb, serial)
    if not args.skip_push:
        prepare_remote_input(adb, serial, sample_dir)
    if not args.skip_instrumentation:
        clear_remote_reports(adb, serial, run_id)
        instrumentation_code = run_instrumentation(
            adb,
            serial,
            run_id,
            app=args.app,
            style=args.style,
            sample_contains=args.sample_contains,
            limit=args.limit,
            use_known_samples=not args.no_known_samples,
            use_initial_position_pattern=not args.no_initial_position_pattern,
            position_prior_scale=args.position_prior_scale,
            use_board_ocr=not args.no_board_ocr,
            collect_candidate_diagnostics=args.collect_candidate_diagnostics,
        )
    else:
        instrumentation_code = 0
    if args.skip_pull:
        run_dir = out_root / run_id
        if not run_dir.exists():
            raise SystemExit(f"Local run directory does not exist: {run_dir}")
    else:
        run_dir = pull_reports(adb, serial, out_root, run_id)
    write_manifest(run_dir, sample_dir, labels_dir)

    summary = None
    if not args.skip_evaluate:
        summary = evaluate(
            run_dir,
            labels_dir,
            include_hands=not args.without_hands,
            strict_leak_guard=args.strict_leak_guard,
            max_seconds=args.max_seconds,
            allow_missing_excluded_source=args.allow_missing_excluded_source,
        )

    if instrumentation_code != 0:
        print(f"Instrumentation failed with exit code {instrumentation_code}. Reports were still pulled if present.", flush=True)
        return instrumentation_code
    if args.require_perfect and not is_perfect(summary, require_speed=args.require_speed):
        print("Metrics are not perfect yet.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
