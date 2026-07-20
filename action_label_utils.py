"""Utilities for action labels stored in the CLIPGCN xlsx file."""

from pathlib import Path

from model import _read_xlsx_rows
from train import get_path_from_config


DEFAULT_NAME_COLUMNS = (
    "display_name",
    "action_name",
    "class_name",
    "name",
    "action_description",
    "global_description",
)


def normalize_label_value(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def label_from_record(record, row_idx, id_column="ID", label_offset=1):
    if id_column in record and record[id_column] != "":
        return int(float(record[id_column])) - int(label_offset)
    return int(row_idx)


def read_action_records(xlsx_path, id_column="ID", label_offset=1):
    rows = _read_xlsx_rows(str(xlsx_path))
    records = {}
    for row_idx, record in enumerate(rows):
        label = label_from_record(record, row_idx, id_column=id_column, label_offset=label_offset)
        records[label] = dict(record)
    return records


def choose_display_name(record, preferred_columns=DEFAULT_NAME_COLUMNS):
    for column in preferred_columns:
        value = str(record.get(column, "")).strip()
        if value:
            return value
    return None


def load_action_display_names(config, config_path, preferred_columns=DEFAULT_NAME_COLUMNS):
    text_config = config["data"]["text"]
    records = read_action_records(
        get_path_from_config(config_path, text_config["xlsx"]),
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
    )
    names = {}
    for label, record in records.items():
        display_name = choose_display_name(record, preferred_columns=preferred_columns)
        if display_name:
            names[int(label)] = display_name
    return names


def find_unseen_labels_from_records(records, split_column):
    unseen = []
    seen = []
    for label, record in records.items():
        value = normalize_label_value(record.get(split_column, ""))
        if value in {"unseen", "u", "zsl", "holdout", "held_out", "heldout", "1", "true", "yes"}:
            unseen.append(int(label))
        elif value in {"seen", "s", "train", "0", "false", "no"}:
            seen.append(int(label))
    return sorted(seen), sorted(unseen)


def resolve_config_path(path):
    return str(Path(path).expanduser().resolve())
