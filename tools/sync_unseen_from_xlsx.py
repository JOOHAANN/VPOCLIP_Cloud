#!/usr/bin/env python
"""Sync xlsx action names/descriptions into a CLIPGCN ZSL split metadata file.

Default behavior keeps the existing 50 seen class IDs fixed. If the xlsx has a
split marker column, the rows marked unseen become the new webcam unseen
candidates. Feature arrays are not rebuilt; this is intended for realtime
zero-shot text-candidate experiments.
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

CLIPGCN_ROOT = Path(__file__).resolve().parents[1]
if str(CLIPGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIPGCN_ROOT))

from action_label_utils import (  # noqa: E402
    DEFAULT_NAME_COLUMNS,
    choose_display_name,
    find_unseen_labels_from_records,
    read_action_records,
)
from train import get_path_from_config, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CLIPGCN_ROOT / "config_50_5.yaml"))
    parser.add_argument("--split-dir", default=None, help="Defaults to config data.train.data_dir.")
    parser.add_argument("--xlsx", default=None, help="Defaults to config data.text.xlsx.")
    parser.add_argument("--split-column", default="split", help="Column whose values are seen/unseen.")
    parser.add_argument("--unseen-count", type=int, default=5)
    parser.add_argument(
        "--name-columns",
        nargs="+",
        default=list(DEFAULT_NAME_COLUMNS),
        help="First non-empty column becomes display_name in metadata.",
    )
    parser.add_argument("--allow-seen-change", action="store_true", help="Allow changing the fixed seen IDs too.")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--report", default=None, help="Optional CSV report path.")
    return parser.parse_args()


def class_name_maps(records, labels, name_columns):
    names = {}
    descriptions = {}
    for label in labels:
        record = records.get(int(label), {})
        names[str(int(label))] = choose_display_name(record, name_columns) or f"class {int(label)}"
        descriptions[str(int(label))] = str(record.get("global_description", "")).strip()
    return names, descriptions


def backup_file(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def write_report(path, records, seen_labels, unseen_labels, name_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unseen_set = {int(label) for label in unseen_labels}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "xlsx_ID", "split", "display_name", "action_description", "global_description"],
        )
        writer.writeheader()
        for label in sorted(int(x) for x in set(seen_labels) | set(unseen_labels)):
            record = records.get(label, {})
            writer.writerow(
                {
                    "label": label,
                    "xlsx_ID": record.get("ID", label + 1),
                    "split": "unseen" if label in unseen_set else "seen",
                    "display_name": choose_display_name(record, name_columns) or "",
                    "action_description": record.get("action_description", ""),
                    "global_description": record.get("global_description", ""),
                }
            )


def main():
    args = parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_config(config_path)
    text_config = config["data"]["text"]

    split_dir = Path(args.split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"]))
    split_dir = split_dir if split_dir.is_absolute() else Path(get_path_from_config(config_path, split_dir))
    metadata_path = split_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    xlsx_path = Path(args.xlsx or get_path_from_config(config_path, text_config["xlsx"]))
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_seen = sorted(int(label) for label in metadata["seen_classes"])
    original_unseen = sorted(int(label) for label in metadata["unseen_classes"])

    records = read_action_records(
        xlsx_path,
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
    )
    missing = [label for label in original_seen + original_unseen if label not in records]
    if missing:
        raise ValueError(f"xlsx is missing rows for labels: {missing}")

    marked_seen, marked_unseen = find_unseen_labels_from_records(records, args.split_column)
    if marked_unseen:
        new_unseen = marked_unseen
        if args.allow_seen_change:
            new_seen = marked_seen or sorted(label for label in records if label not in set(new_unseen))
        else:
            new_seen = original_seen
    else:
        new_seen = original_seen
        new_unseen = original_unseen
        print(
            f"Column {args.split_column!r} has no unseen markers; using existing metadata unseen labels: {new_unseen}"
        )

    if len(new_unseen) != args.unseen_count:
        raise ValueError(f"Expected {args.unseen_count} unseen classes, got {len(new_unseen)}: {new_unseen}")

    overlap = sorted(set(new_seen) & set(new_unseen))
    if overlap:
        raise ValueError(f"Labels cannot be both seen and unseen: {overlap}")

    if marked_seen and not args.allow_seen_change and sorted(marked_seen) != original_seen:
        raise ValueError(
            "xlsx split markers changed the fixed 50 seen IDs.\n"
            f"metadata seen: {original_seen}\n"
            f"xlsx seen:     {sorted(marked_seen)}\n"
            "Keep the 50 seen rows fixed, or pass --allow-seen-change if you really want to change them."
        )

    all_labels = sorted(set(new_seen) | set(new_unseen))
    all_names, all_descriptions = class_name_maps(records, all_labels, args.name_columns)
    seen_names, _seen_descriptions = class_name_maps(records, new_seen, args.name_columns)
    unseen_names, _unseen_descriptions = class_name_maps(records, new_unseen, args.name_columns)

    metadata["seen_classes"] = new_seen
    metadata["unseen_classes"] = new_unseen
    metadata["seen_class_count"] = len(new_seen)
    metadata["unseen_class_count"] = len(new_unseen)
    metadata.setdefault("seen", {})["class_ids"] = new_seen
    metadata.setdefault("unseen", {})["class_ids"] = new_unseen
    metadata["action_display_names"] = all_names
    metadata["action_global_descriptions"] = all_descriptions
    metadata["seen_action_display_names"] = seen_names
    metadata["unseen_action_display_names"] = unseen_names
    metadata["xlsx_sync"] = {
        "xlsx": str(xlsx_path),
        "split_column": args.split_column,
        "name_columns": args.name_columns,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed_seen_classes": not args.allow_seen_change,
        "note": (
            "Text candidates are read from xlsx at runtime. By default this script keeps the existing "
            "50 seen IDs fixed and updates webcam unseen candidates from rows marked unseen."
        ),
    }

    if not args.no_backup:
        backup_path = backup_file(metadata_path)
        print(f"Backup: {backup_path}")

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = Path(args.report) if args.report else split_dir / "action_label_report.csv"
    write_report(report_path, records, new_seen, new_unseen, args.name_columns)

    print(f"Updated: {metadata_path}")
    print(f"Report: {report_path}")
    print(f"seen_classes: {new_seen}")
    print(f"unseen_classes: {new_unseen}")
    print("unseen names:")
    for label in new_unseen:
        print(f"  {label}: {unseen_names[str(label)]}")


if __name__ == "__main__":
    main()
