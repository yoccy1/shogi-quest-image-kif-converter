from __future__ import annotations

import argparse
import json
from pathlib import Path

from position_label_utils import inventory_errors, load_position_label


def validate_label(path: Path, require_hands: bool) -> dict:
    result = {
        "label": str(path),
        "ok": True,
        "errors": [],
    }
    try:
        label = load_position_label(path, require_hands=require_hands)
        if require_hands:
            errors = inventory_errors(label["cells"], label["hands"])
            if errors:
                result["ok"] = False
                result["errors"].extend(errors)
        result["summary"] = label["summary"]
        result["has_hands"] = label["hands"] is not None
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate board and hand teacher labels.")
    parser.add_argument(
        "labels_dir",
        nargs="?",
        type=Path,
        default=Path("tools/samples/labels/boards"),
        help="Directory containing board label JSON files.",
    )
    parser.add_argument("--require-hands", action="store_true", help="Require schema v2 hands for each label.")
    parser.add_argument("--out", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    paths = sorted(args.labels_dir.rglob("*.json"))
    results = [validate_label(path, args.require_hands) for path in paths]
    summary = {
        "labels_dir": str(args.labels_dir),
        "label_count": len(results),
        "ok_count": sum(1 for result in results if result["ok"]),
        "failed_count": sum(1 for result in results if not result["ok"]),
        "require_hands": args.require_hands,
    }
    report = {"summary": summary, "results": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for result in results:
        if not result["ok"]:
            print(f"NG: {result['label']}")
            for error in result["errors"][:8]:
                print(f"  {error}")
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
