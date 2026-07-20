#!/usr/bin/env python
"""Realtime webcam inference for CLIPGCN.

The trained CLIPGCN model expects pre-extracted video, pose, object and joint
features. This script reads frames from a local webcam, then runs X3D, YOLO,
MediaPipe/CTR-GCN, and CLIPGCN online.
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from action_label_utils import load_action_display_names
from test import apply_unseen_score_scale, load_split_classes, load_split_metadata, logits_to_unit_cosine_scores
from test_raw_end_to_end import (
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    ctrgcn_pose_from_model,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
    x3d_features_from_model,
    x3d_tensor_from_frames,
)
from train import get_device, get_path_from_config, load_config, print_device_info


SCRIPT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_ROOT.parent
UNKNOWN_LABEL = -1
DEFAULT_ACTION_DURATION_TYPES = {
    0: "L",
    1: "L",
    2: "L",
    3: "L",
    4: "L",
    5: "S",
    6: "L",
    7: "L",
    8: "S",
    9: "S",
    10: "S",
    11: "S",
    12: "S",
    13: "L",
    14: "L",
    15: "S",
    16: "S",
    17: "L",
    18: "L",
    19: "L",
    20: "L",
    21: "L",
    22: "S",
    23: "S",
    24: "S",
    25: "S",
    26: "L",
    27: "S",
    28: "L",
    29: "L",
    30: "S",
    31: "S",
    32: "S",
    33: "S",
    34: "S",
    35: "S",
    36: "S",
    37: "S",
    38: "S",
    39: "S",
    40: "S",
    41: "S",
    42: "S",
    43: "S",
    44: "S",
    45: "S",
    46: "L",
    47: "S",
    48: "S",
    49: "S",
    50: "S",
    51: "L",
    52: "S",
    53: "L",
    54: "L",
}


def resolve_existing_path(path_value, *, search_roots=None, label="path"):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return str(path)

    roots = []
    if search_roots:
        roots.extend(Path(root) for root in search_roots)
    roots.extend([Path.cwd(), SCRIPT_ROOT, WORKSPACE_ROOT])

    seen = set()
    for root in roots:
        candidate = (root / path).resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate)

    tried = ", ".join(str((root / path).resolve()) for root in roots)
    raise FileNotFoundError(f"{label} not found: {path}. Tried: {tried}")


def default_yolo_repo():
    candidates = [
        WORKSPACE_ROOT / "yolov5",
        SCRIPT_ROOT / "yolov5",
        Path("/workspace/yolov5"),
    ]
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return str(candidate)
    return str(candidates[0])


def default_yolo_weights():
    candidates = [
        SCRIPT_ROOT / "local_models" / "yolov5m.pt",
        WORKSPACE_ROOT / "yolov5" / "yolov5m.pt",
        SCRIPT_ROOT / "yolov5" / "yolov5m.pt",
        Path("/workspace/yolov5/yolov5m.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def default_mediapipe_model_asset():
    env_path = os.environ.get("MEDIAPIPE_POSE_MODEL")
    if env_path and Path(env_path).expanduser().exists():
        return env_path

    names = [
        "pose_landmarker_full.task",
        "pose_landmarker_lite.task",
        "pose_landmarker_heavy.task",
    ]
    roots = [SCRIPT_ROOT, WORKSPACE_ROOT, Path("/workspace"), Path.cwd()]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    return env_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLIPGCN realtime inference from a local webcam.")
    parser.add_argument(
        "--config",
        default=str(SCRIPT_ROOT / "config_50_5.yaml"),
        help="Path to the CLIPGCN YAML config.",
    )
    parser.add_argument(
        "--class-split-dir",
        default=None,
        help="Directory whose metadata.json defines seen/unseen classes. Defaults to config data.train.data_dir.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["unseen", "seen", "all"],
        default="all",
        help="Which action text labels are valid predictions.",
    )
    parser.add_argument(
        "--class-config",
        default=None,
        help=(
            "Optional editable class-selection file (.csv/.tsv/.yaml/.json). "
            "Each class can set enabled true/false and split seen/unseen."
        ),
    )
    parser.add_argument(
        "--all-classes-seen",
        action="store_true",
        help="Override split metadata and treat every label from the xlsx/class config as seen with no unseen classes.",
    )
    parser.add_argument(
        "--seen-classes",
        default=None,
        help=(
            "Optional seen class override, e.g. '0-57' or 'A01,A02,A58'. "
            "Numeric IDs are zero-based; Axx IDs are converted to zero-based labels."
        ),
    )
    parser.add_argument(
        "--unseen-classes",
        default=None,
        help=(
            "Optional unseen class override, e.g. '9,10,11,17,49' or 'A10,A11,A12,A18,A50'. "
            "If only unseen is set, all other labels 0..54 become seen."
        ),
    )
    parser.add_argument(
        "--exclude-classes",
        default=None,
        help=(
            "Optional class IDs to remove from the candidate pool entirely, e.g. 'A01,A02' or '0,1'. "
            "Excluded labels are removed after seen/unseen overrides are applied."
        ),
    )
    parser.add_argument(
        "--unseen-score-scale",
        type=float,
        default=1.3,
        help="Multiplier applied to unseen class confidence scores before top-k prediction.",
    )
    parser.add_argument("--clipgcn-checkpoint", default=None, help="Optional CLIPGCN checkpoint override.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=13, help="Rolling frame window. Must match training.")
    parser.add_argument("--predict-every", type=int, default=13, help="Run recognition once every N captured frames.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--display-filter-window",
        type=int,
        default=10,
        help=(
            "Smooth displayed top-k labels by majority vote over the latest N predictions at each rank. "
            "Use 1 to show raw predictions."
        ),
    )
    parser.add_argument(
        "--disable-unknown",
        action="store_true",
        help="Disable entropy-based UNKNOWN display.",
    )
    parser.add_argument(
        "--unknown-entropy-threshold",
        type=float,
        default=0.90,
        help="Show UNKNOWN when normalized entropy is at least this value and top1 score is low enough.",
    )
    parser.add_argument(
        "--unknown-top1-threshold",
        type=float,
        default=0.55,
        help="Entropy-based UNKNOWN also requires top1 cosine score to be at most this value.",
    )
    parser.add_argument(
        "--unknown-temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax over 0-1 cosine scores when computing UNKNOWN entropy.",
    )
    parser.add_argument(
        "--temporal-strategy",
        choices=["short", "short-long", "uniform3s", "last13"],
        default="short-long",
        help=(
            "short/short-long/uniform3s sample 13 frames from time windows. "
            "last13 uses the latest contiguous 13 captured frames like the old realtime demo."
        ),
    )
    parser.add_argument("--short-window-seconds", type=float, default=2.0, help="Seconds covered by 13 frames for S actions.")
    parser.add_argument("--long-window-seconds", type=float, default=4.0, help="Seconds covered by 13 frames for L actions.")
    parser.add_argument(
        "--uniform-window-seconds",
        type=float,
        default=3.0,
        help="Seconds covered by 13 frames when --temporal-strategy uniform3s is used.",
    )
    parser.add_argument(
        "--long-rerank-top-k",
        type=int,
        default=5,
        help="Run the long-window pass only if an L action appears in this many short-window candidates.",
    )
    parser.add_argument(
        "--pose-source",
        choices=["mediapipe", "zero"],
        default="mediapipe",
        help="Realtime pose source. Use zero only for debugging or when MediaPipe is unavailable.",
    )
    parser.add_argument("--mediapipe-model-complexity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--mediapipe-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--mediapipe-min-visibility", type=float, default=0.2)
    parser.add_argument(
        "--mediapipe-model-asset",
        default=default_mediapipe_model_asset(),
        help=(
            "Path to a MediaPipe Tasks pose_landmarker .task model. Required when the installed "
            "mediapipe package exposes tasks.vision.PoseLandmarker instead of the legacy solutions API. "
            "Can also be set with MEDIAPIPE_POSE_MODEL."
        ),
    )
    parser.add_argument("--window-name", default="CLIPGCN realtime")
    parser.add_argument("--headless", action="store_true", help="Print predictions without opening a display window.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when the config requests CUDA.")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn benchmark for fixed-size realtime inference on CUDA.",
    )

    parser.add_argument("--x3d-root", default=str(X3D_ROOT))
    parser.add_argument(
        "--x3d-config",
        default=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"),
    )
    parser.add_argument(
        "--x3d-checkpoint",
        default=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"),
    )
    parser.add_argument("--x3d-layer", default="s5")

    # Kept for compatibility with helpers imported from test_raw_end_to_end.py.
    parser.add_argument("--ctrgcn-root", default=str(CTRGCN_ROOT))
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"))
    parser.add_argument("--ctrgcn-hook-layer", default="l4")

    parser.add_argument("--yolo-repo", default=default_yolo_repo())
    parser.add_argument("--yolo-weights", default=default_yolo_weights())
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true", help="Run YOLO in FP16 on CUDA.")
    parser.add_argument(
        "--yolo-detect-every",
        type=int,
        default=1,
        help="Run YOLO once every N predictions and reuse the previous object map in between.",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Use zero object maps instead of YOLO.")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


def validate_args(args):
    if args.frames != 13:
        raise ValueError(
            "This CLIPGCN checkpoint expects 13-frame features. "
            "Keep --frames 13 unless you retrain/update the fusion model."
        )
    if args.predict_every <= 0:
        raise ValueError("--predict-every must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.display_filter_window <= 0:
        raise ValueError("--display-filter-window must be positive.")
    if not 0.0 <= args.unknown_entropy_threshold <= 1.0:
        raise ValueError("--unknown-entropy-threshold must be in [0, 1].")
    if not 0.0 <= args.unknown_top1_threshold <= 1.0:
        raise ValueError("--unknown-top1-threshold must be in [0, 1].")
    if args.unknown_temperature <= 0:
        raise ValueError("--unknown-temperature must be positive.")
    if args.unseen_score_scale <= 0:
        raise ValueError("--unseen-score-scale must be positive.")
    temporal_strategy = getattr(args, "temporal_strategy", None)
    if temporal_strategy == "uniform3s":
        if getattr(args, "uniform_window_seconds", 0) <= 0:
            raise ValueError("--uniform-window-seconds must be positive.")
    elif temporal_strategy != "last13":
        if getattr(args, "short_window_seconds", 0) <= 0:
            raise ValueError("--short-window-seconds must be positive.")
        if getattr(args, "long_window_seconds", 0) <= 0:
            raise ValueError("--long-window-seconds must be positive.")
        if getattr(args, "long_window_seconds", 0) < getattr(args, "short_window_seconds", 0):
            raise ValueError("--long-window-seconds must be greater than or equal to --short-window-seconds.")
        if getattr(args, "long_rerank_top_k", 0) <= 0:
            raise ValueError("--long-rerank-top-k must be positive.")


def parse_class_spec(spec, *, label_min=0, label_max=54):
    if spec is None:
        return None

    value = str(spec).strip()
    if value == "" or value.lower() in {"none", "empty", "[]"}:
        return []

    labels = []
    for raw_token in value.replace(";", ",").replace(" ", ",").split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_token, end_token = [part.strip() for part in token.split("-", 1)]
            start = parse_single_class_id(start_token)
            end = parse_single_class_id(end_token)
            if end < start:
                raise ValueError(f"Invalid descending class range: {token}")
            labels.extend(range(start, end + 1))
        else:
            labels.append(parse_single_class_id(token))

    unique_labels = sorted(set(labels))
    invalid = [label for label in unique_labels if label < label_min or label > label_max]
    if invalid:
        raise ValueError(
            f"Class IDs must be in {label_min}..{label_max} or A{label_min + 1:02d}..A{label_max + 1:02d}. "
            f"Invalid: {invalid}"
        )
    return unique_labels


def parse_single_class_id(token):
    token = str(token).strip()
    if not token:
        raise ValueError("Empty class ID in class list.")
    if token[0].lower() == "a":
        return int(token[1:]) - 1
    return int(token)


def parse_config_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y", "on", "enable", "enabled", "active"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disable", "disabled", "inactive"}:
        return False
    raise ValueError(f"Invalid boolean value in class config: {value!r}")


def parse_config_float(value, default=0.0):
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def normalize_class_split(value, default="seen"):
    if value is None:
        return default
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text == "":
        return default
    if text in {"seen", "s", "train", "trained"}:
        return "seen"
    if text in {"unseen", "u", "zsl", "holdout", "held_out", "heldout"}:
        return "unseen"
    if text in {"disabled", "disable", "excluded", "exclude", "remove", "removed", "ignore", "ignored", "off"}:
        return "disabled"
    raise ValueError(f"Invalid split value in class config: {value!r}")


def first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] not in {None, ""}:
            return mapping[key]
    return default


def load_class_config(path):
    resolved_path = resolve_existing_path(path, search_roots=[SCRIPT_ROOT, WORKSPACE_ROOT], label="class config")
    suffix = Path(resolved_path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(resolved_path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
        return {"path": resolved_path, "classes": rows}

    with open(resolved_path, "r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(handle) or {}
        else:
            raise ValueError("Class config must be .csv, .tsv, .yaml, .yml, or .json.")

    if isinstance(data, list):
        data = {"classes": data}
    if not isinstance(data, dict):
        raise ValueError("YAML/JSON class config must be a mapping or a list of class records.")
    data["path"] = resolved_path
    return data


def iter_class_config_records(classes):
    if classes is None:
        return []
    if isinstance(classes, dict):
        records = []
        for class_id, value in classes.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("id", class_id)
            else:
                record = {"id": class_id, "split": value}
            records.append(record)
        return records
    if isinstance(classes, list):
        records = []
        for item in classes:
            if isinstance(item, dict):
                records.append(dict(item))
            else:
                records.append({"id": item})
        return records
    raise ValueError("Class config 'classes' must be a list or mapping.")


def labels_from_class_config(path, universe):
    config = load_class_config(path)
    default_enabled = parse_config_bool(first_present(config, ["default_enabled", "enabled"], True), default=True)
    default_split = normalize_class_split(first_present(config, ["default_split", "split"], "seen"), default="seen")
    if default_split == "disabled":
        default_enabled = False
        default_split = "seen"

    state = {
        int(label): {
            "enabled": default_enabled,
            "split": default_split,
            "score_scale": 1.0,
            "score_bias": 0.0,
        }
        for label in universe
    }

    for record in iter_class_config_records(config.get("classes")):
        class_id = first_present(record, ["id", "ID", "action_id", "class_id", "label", "label_id"])
        if class_id is None:
            raise ValueError(f"Class config record is missing an id/label field: {record}")

        class_id_text = str(class_id).strip()
        if class_id_text.upper() in {"DEFAULT", "*", "ALL"}:
            default_enabled = parse_config_bool(
                first_present(record, ["enabled", "enable", "active"], default_enabled),
                default=default_enabled,
            )
            default_split = normalize_class_split(
                first_present(record, ["split", "status", "type"], default_split),
                default=default_split,
            )
            if default_split == "disabled":
                default_enabled = False
                default_split = "seen"
            for label in universe:
                current = state[int(label)]
                state[int(label)] = {
                    "enabled": default_enabled,
                    "split": default_split,
                    "score_scale": parse_config_float(
                        first_present(record, ["score_scale", "scale"], current["score_scale"]),
                        default=current["score_scale"],
                    ),
                    "score_bias": parse_config_float(
                        first_present(record, ["score_bias", "bias"], current["score_bias"]),
                        default=current["score_bias"],
                    )
                    - parse_config_float(first_present(record, ["score_penalty", "penalty"], 0.0), default=0.0),
                }
            continue

        label = parse_single_class_id(class_id_text)
        if label not in state:
            raise ValueError(f"Class config label out of range: {class_id}")

        current = state[label]
        split = normalize_class_split(first_present(record, ["split", "status", "type"], current["split"]), default=current["split"])
        enabled = parse_config_bool(
            first_present(record, ["enabled", "enable", "active"], current["enabled"]),
            default=current["enabled"],
        )
        score_scale = parse_config_float(
            first_present(record, ["score_scale", "scale"], current["score_scale"]),
            default=current["score_scale"],
        )
        score_bias = parse_config_float(
            first_present(record, ["score_bias", "bias"], current["score_bias"]),
            default=current["score_bias"],
        ) - parse_config_float(first_present(record, ["score_penalty", "penalty"], 0.0), default=0.0)
        if split == "disabled":
            enabled = False
            split = current["split"]
        state[label] = {
            "enabled": enabled,
            "split": split,
            "score_scale": score_scale,
            "score_bias": score_bias,
        }

    seen_labels = []
    unseen_labels = []
    excluded_labels = []
    score_adjustments = {}
    for label in sorted(state):
        item = state[label]
        if not item["enabled"]:
            excluded_labels.append(label)
        elif item["split"] == "unseen":
            unseen_labels.append(label)
        else:
            seen_labels.append(label)
        if item["score_scale"] != 1.0 or item["score_bias"] != 0.0:
            score_adjustments[label] = {
                "scale": float(item["score_scale"]),
                "bias": float(item["score_bias"]),
            }

    return {
        "seen_labels": seen_labels,
        "unseen_labels": unseen_labels,
        "excluded_labels": excluded_labels,
        "score_adjustments": score_adjustments,
        "path": config["path"],
    }


def select_candidate_labels(seen_labels, unseen_labels, candidate_scope):
    if candidate_scope == "seen":
        return sorted(seen_labels)
    if candidate_scope == "unseen":
        return sorted(unseen_labels)
    if candidate_scope == "all":
        return sorted(set(seen_labels) | set(unseen_labels))
    raise ValueError(f"Unsupported candidate scope: {candidate_scope}")


def resolve_realtime_class_labels(args, class_split_dir, *, all_labels=range(55)):
    all_labels = sorted(int(label) for label in all_labels)
    metadata = load_split_metadata(class_split_dir)
    metadata_seen = [int(label) for label in metadata.get("seen_classes", all_labels)]
    metadata_unseen = [int(label) for label in metadata.get("unseen_classes", [])]
    universe = sorted(set(metadata_seen) | set(metadata_unseen) | set(all_labels))
    label_min = min(universe)
    label_max = max(universe)

    seen_override = parse_class_spec(getattr(args, "seen_classes", None), label_min=label_min, label_max=label_max)
    unseen_override = parse_class_spec(getattr(args, "unseen_classes", None), label_min=label_min, label_max=label_max)
    cli_exclude_labels = set(
        parse_class_spec(getattr(args, "exclude_classes", None), label_min=label_min, label_max=label_max) or []
    )
    class_config_path = getattr(args, "class_config", None)
    class_config_selection = labels_from_class_config(class_config_path, universe) if class_config_path else None

    if getattr(args, "all_classes_seen", False):
        seen_labels = list(universe)
        unseen_labels = []
        base_excluded_labels = set()
        score_adjustments = {}
    elif seen_override is None and unseen_override is None:
        if class_config_selection is not None:
            seen_labels = class_config_selection["seen_labels"]
            unseen_labels = class_config_selection["unseen_labels"]
            base_excluded_labels = set(class_config_selection["excluded_labels"])
            score_adjustments = dict(class_config_selection.get("score_adjustments", {}))
        else:
            seen_labels = metadata_seen
            unseen_labels = metadata_unseen
            base_excluded_labels = set()
            score_adjustments = {}
    elif seen_override is None:
        unseen_set = set(unseen_override)
        seen_labels = [label for label in universe if label not in unseen_set]
        unseen_labels = list(unseen_override)
        base_excluded_labels = set()
        score_adjustments = {}
    elif unseen_override is None:
        seen_set = set(seen_override)
        seen_labels = list(seen_override)
        unseen_labels = [label for label in universe if label not in seen_set]
        base_excluded_labels = set()
        score_adjustments = {}
    else:
        seen_labels = list(seen_override)
        unseen_labels = list(unseen_override)
        base_excluded_labels = set()
        score_adjustments = {}

    overlap = sorted(set(seen_labels) & set(unseen_labels))
    if overlap:
        raise ValueError(f"Seen and unseen classes overlap: {overlap}")

    exclude_labels = base_excluded_labels | cli_exclude_labels
    if exclude_labels:
        seen_labels = [label for label in seen_labels if label not in exclude_labels]
        unseen_labels = [label for label in unseen_labels if label not in exclude_labels]
        score_adjustments = {
            label: value for label, value in score_adjustments.items() if label not in exclude_labels
        }

    candidate_labels = select_candidate_labels(seen_labels, unseen_labels, args.candidate_scope)
    if not candidate_labels:
        raise ValueError(
            "No candidate classes remain after applying --candidate-scope, --seen-classes, "
            "--unseen-classes, and --exclude-classes."
        )

    class_selection = {
        "seen_labels": sorted(seen_labels),
        "unseen_labels": sorted(unseen_labels),
        "excluded_labels": sorted(exclude_labels),
        "score_adjustments": score_adjustments,
        "candidate_labels": candidate_labels,
        "source": class_config_selection["path"]
        if class_config_selection is not None
        and not getattr(args, "all_classes_seen", False)
        and seen_override is None
        and unseen_override is None
        else (
            "override"
            if getattr(args, "all_classes_seen", False)
            or seen_override is not None
            or unseen_override is not None
            or cli_exclude_labels
            else "metadata"
        ),
    }
    return class_selection


def open_camera(args):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    return cap


def has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def maybe_enable_headless(args):
    if args.headless or has_display():
        return args
    print("Warning: no GUI display detected, enabling --headless automatically.")
    args.headless = True
    return args


def import_mediapipe_solutions_pose_module():
    try:
        from mediapipe.python.solutions import pose as mp_pose

        return mp_pose
    except Exception:
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise AttributeError("module 'mediapipe' has no attribute 'solutions'")
        return solutions.pose


def import_mediapipe_tasks_pose_api():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    if not hasattr(vision, "PoseLandmarker"):
        raise AttributeError("mediapipe.tasks.python.vision has no PoseLandmarker")
    return mp, mp_tasks, vision


def resolve_mediapipe_model_asset(args):
    path = args.mediapipe_model_asset or os.environ.get("MEDIAPIPE_POSE_MODEL")
    if not path:
        return None
    return resolve_existing_path(
        path,
        search_roots=[SCRIPT_ROOT, WORKSPACE_ROOT, Path("/workspace"), Path.cwd()],
        label="MediaPipe pose model asset",
    )


def mediapipe_diagnostics():
    lines = []
    spec = importlib.util.find_spec("mediapipe")
    if spec is None:
        return "mediapipe import spec: not found"

    lines.append(f"mediapipe import origin: {spec.origin}")
    lines.append(f"mediapipe search locations: {list(spec.submodule_search_locations or [])}")
    try:
        import mediapipe as mp

        lines.append(f"mediapipe module file: {getattr(mp, '__file__', None)}")
        lines.append(f"mediapipe has solutions: {hasattr(mp, 'solutions')}")
        lines.append(f"mediapipe has tasks: {hasattr(mp, 'tasks')}")
        lines.append(f"mediapipe sample attrs: {sorted(name for name in dir(mp) if not name.startswith('_'))[:30]}")
    except Exception as exc:
        lines.append(f"mediapipe import error: {exc}")
    return "\n".join(lines)


def resolve_runtime_pose_source(args):
    if args.pose_source != "mediapipe":
        return args.pose_source
    try:
        import_mediapipe_solutions_pose_module()
        return "mediapipe"
    except Exception as solutions_exc:
        try:
            import_mediapipe_tasks_pose_api()
            model_asset_path = resolve_mediapipe_model_asset(args)
            if model_asset_path is None:
                raise FileNotFoundError(
                    "MediaPipe Tasks PoseLandmarker requires a local pose_landmarker .task model. "
                    "Pass --mediapipe-model-asset or set MEDIAPIPE_POSE_MODEL."
                )
            return "mediapipe"
        except Exception as tasks_exc:
            model_asset_hint = (
                "Set --mediapipe-model-asset /path/to/pose_landmarker_full.task "
                "or MEDIAPIPE_POSE_MODEL when using the MediaPipe Tasks-only package."
            )
            if not (args.mediapipe_model_asset or os.environ.get("MEDIAPIPE_POSE_MODEL")):
                model_asset_hint = (
                    "The installed mediapipe package exposes the Tasks API, which requires a local "
                    "pose_landmarker .task model file. " + model_asset_hint
                )
            raise RuntimeError(
                "MediaPipe Pose API is required because --pose-source mediapipe was requested, "
                "but it could not be loaded.\n"
                f"Legacy solutions error: {solutions_exc}\n"
                f"Tasks PoseLandmarker error: {tasks_exc}\n"
                f"{model_asset_hint}\n"
                f"{mediapipe_diagnostics()}"
            ) from tasks_exc


class MediaPipePoseSource:
    """Converts MediaPipe's 33 landmarks into the NTU/Kinect 25-joint layout."""

    def __init__(self, args):
        self.min_visibility = float(args.mediapipe_min_visibility)
        self.backend = None
        self.mp = None
        self.last_timestamp_ms = 0
        try:
            mp_pose = import_mediapipe_solutions_pose_module()
            self.backend = "solutions"
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=args.mediapipe_model_complexity,
                enable_segmentation=False,
                min_detection_confidence=args.mediapipe_min_detection_confidence,
                min_tracking_confidence=args.mediapipe_min_tracking_confidence,
            )
        except Exception as solutions_exc:
            try:
                mp, mp_tasks, vision = import_mediapipe_tasks_pose_api()
                model_asset_path = resolve_mediapipe_model_asset(args)
                if model_asset_path is None:
                    raise FileNotFoundError(
                        "MediaPipe Tasks PoseLandmarker requires a local pose_landmarker .task model. "
                        "Pass --mediapipe-model-asset or set MEDIAPIPE_POSE_MODEL."
                    )
                self.backend = "tasks"
                self.mp = mp
                options = vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=model_asset_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=args.mediapipe_min_detection_confidence,
                    min_pose_presence_confidence=args.mediapipe_min_detection_confidence,
                    min_tracking_confidence=args.mediapipe_min_tracking_confidence,
                    output_segmentation_masks=False,
                )
                self.pose = vision.PoseLandmarker.create_from_options(options)
            except Exception as tasks_exc:
                raise RuntimeError(
                    "Failed to initialize MediaPipe pose source.\n"
                    f"Legacy solutions error: {solutions_exc}\n"
                    f"Tasks PoseLandmarker error: {tasks_exc}\n"
                    f"{mediapipe_diagnostics()}"
                ) from tasks_exc
        self.last_joints_3d = np.zeros((25, 3), dtype=np.float32)
        self.last_joint_xy = np.zeros((25, 2), dtype=np.float32)

    def close(self):
        self.pose.close()

    def process(self, frame_rgb):
        if self.backend == "solutions":
            results = self.pose.process(frame_rgb)
            pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
        else:
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            results = self.pose.detect_for_video(image, timestamp_ms)
            pose_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None

        if not pose_landmarks:
            return {
                "joints_3d": self.last_joints_3d.copy(),
                "joint_xy": self.last_joint_xy.copy(),
                "detected": False,
            }

        landmarks = np.asarray(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    getattr(landmark, "visibility", getattr(landmark, "presence", 1.0)),
                ]
                for landmark in pose_landmarks
            ],
            dtype=np.float32,
        )
        joints_3d, joint_xy = mediapipe_landmarks_to_ntu25(landmarks, self.min_visibility)
        self.last_joints_3d = joints_3d
        self.last_joint_xy = joint_xy
        return {
            "joints_3d": joints_3d,
            "joint_xy": joint_xy,
            "detected": True,
        }


def mediapipe_landmarks_to_ntu25(landmarks, min_visibility):
    def point(index):
        if landmarks[index, 3] < min_visibility:
            return None
        x = landmarks[index, 0] * 2.0 - 1.0
        y = landmarks[index, 1] * 2.0 - 1.0
        z = landmarks[index, 2]
        if not np.isfinite([x, y, z]).all():
            return None
        return np.asarray([x, y, z], dtype=np.float32)

    def average(*indices):
        values = [point(index) for index in indices]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)

    def midpoint(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return ((a + b) * 0.5).astype(np.float32)

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    spine_mid = midpoint(shoulder_center, hip_center)

    ntu_points = [
        hip_center,  # 1 spine base
        spine_mid,  # 2 spine mid
        shoulder_center,  # 3 neck
        average(0, 7, 8),  # 4 head
        left_shoulder,
        point(13),
        point(15),
        point(19),
        right_shoulder,
        point(14),
        point(16),
        point(20),
        left_hip,
        point(25),
        point(27),
        point(31),
        right_hip,
        point(26),
        point(28),
        point(32),
        shoulder_center,  # 21 spine shoulder
        point(19),
        point(21),
        point(20),
        point(22),
    ]

    joints_3d = np.zeros((25, 3), dtype=np.float32)
    for index, value in enumerate(ntu_points):
        if value is not None:
            joints_3d[index] = value
    joint_xy = np.clip(joints_3d[:, :2], -1.0, 1.0).astype(np.float32)
    return joints_3d, joint_xy


def build_zero_pose_inputs(batch_size, frames, device):
    # Laptop webcam RGB has no Kinect/ETRI 25-joint skeleton stream.
    pose = torch.zeros(batch_size, 2, 64, frames, 25, dtype=torch.float32, device=device)
    joint_xy = torch.zeros(batch_size, frames, 25, 2, dtype=torch.float32, device=device)
    return pose, joint_xy


def build_mediapipe_skeleton_inputs(skeleton_buffer, device):
    joints_3d = np.stack([sample["joints_3d"] for sample in skeleton_buffer], axis=0).astype(np.float32)
    joint_xy = np.stack([sample["joint_xy"] for sample in skeleton_buffer], axis=0).astype(np.float32)

    skeleton = np.zeros((3, joints_3d.shape[0], 25, 2), dtype=np.float32)
    skeleton[:, :, :, 0] = joints_3d.transpose(2, 0, 1)
    detected_frames = sum(1 for sample in skeleton_buffer if sample["detected"])

    return (
        torch.from_numpy(skeleton).unsqueeze(0).to(device=device, dtype=torch.float32),
        torch.from_numpy(joint_xy).unsqueeze(0).to(device=device, dtype=torch.float32),
        detected_frames,
    )


def action_duration_type(label, args):
    return getattr(args, "action_duration_types", DEFAULT_ACTION_DURATION_TYPES).get(int(label), "S")


def is_long_action(label, args):
    return action_duration_type(label, args).upper() == "L"


def history_duration_seconds(history_buffer):
    if len(history_buffer) < 2:
        return 0.0
    return max(0.0, float(history_buffer[-1]["timestamp"] - history_buffer[0]["timestamp"]))


def trim_history_buffer(history_buffer, args, now):
    if args.temporal_strategy == "uniform3s":
        max_window = args.uniform_window_seconds
    else:
        max_window = max(args.short_window_seconds, args.long_window_seconds)
    keep_seconds = max_window + 1.0
    while len(history_buffer) > args.frames and now - history_buffer[0]["timestamp"] > keep_seconds:
        history_buffer.popleft()


def history_window_ready(history_buffer, window_seconds, args):
    return len(history_buffer) >= args.frames and history_duration_seconds(history_buffer) >= window_seconds


def active_window_seconds(args):
    if args.temporal_strategy == "uniform3s":
        return args.uniform_window_seconds
    return args.short_window_seconds


def warmup_frame_count(history_buffer, args):
    if not history_buffer:
        return 0
    progress = min(1.0, history_duration_seconds(history_buffer) / active_window_seconds(args))
    return min(args.frames, max(1, int(round(progress * args.frames))))


def sample_history_window(history_buffer, window_seconds, frame_count):
    samples = list(history_buffer)
    if not samples:
        raise ValueError("Cannot sample an empty realtime history.")

    end_time = samples[-1]["timestamp"]
    start_time = end_time - window_seconds
    timestamps = [sample["timestamp"] for sample in samples]
    targets = np.linspace(start_time, end_time, num=frame_count)

    selected = []
    cursor = 0
    last_index = len(samples) - 1
    for target in targets:
        while cursor < last_index and timestamps[cursor] < target:
            cursor += 1
        if cursor > 0:
            previous_index = cursor - 1
            if abs(timestamps[previous_index] - target) <= abs(timestamps[cursor] - target):
                selected.append(samples[previous_index])
                continue
        selected.append(samples[cursor])

    frames = [sample["frame_rgb"] for sample in selected]
    skeletons = [sample["skeleton"] for sample in selected if sample.get("skeleton") is not None]
    return frames, skeletons


def latest_history_ready(history_buffer, args):
    if len(history_buffer) < args.frames:
        return False
    if args.runtime_pose_source == "zero":
        return True
    return all(sample.get("skeleton") is not None for sample in list(history_buffer)[-args.frames :])


def latest_history_window(history_buffer, frame_count):
    samples = list(history_buffer)[-frame_count:]
    frames = [sample["frame_rgb"] for sample in samples]
    skeletons = [sample["skeleton"] for sample in samples if sample.get("skeleton") is not None]
    return frames, skeletons


def finish_end_to_end_timer(start, device):
    """Finish timing after all asynchronous accelerator work is complete."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def score_prediction_window(
    frames,
    skeleton_samples,
    args,
    device,
    x3d_cfg,
    x3d_model,
    x3d_captured,
    ctrgcn_model,
    ctrgcn_captured,
    object_map_runner,
    clipgcn_model,
    candidate_labels,
    unseen_labels,
    use_amp,
):
    x3d_clip = x3d_tensor_from_frames(
        frames,
        int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
        x3d_cfg.TRANSFORM.MEAN,
        x3d_cfg.TRANSFORM.STD,
    ).unsqueeze(0).to(device, non_blocking=True)
    yolo_frame = frames[len(frames) // 2]
    detected_pose_frames = 0
    if args.runtime_pose_source == "mediapipe":
        skeletons, joint_xy, detected_pose_frames = build_mediapipe_skeleton_inputs(skeleton_samples, device)
    else:
        pose_features, joint_xy = build_zero_pose_inputs(batch_size=1, frames=args.frames, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        object_maps = object_map_runner([yolo_frame])
        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clip, args)
        if args.runtime_pose_source == "mediapipe":
            pose_features = ctrgcn_pose_from_model(ctrgcn_model, ctrgcn_captured, skeletons, args)
        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xy)
        cosine_scores = logits_to_unit_cosine_scores(clipgcn_model, logits)
        prediction_scores, used_scale = apply_unseen_score_scale(
            clipgcn_model,
            logits,
            candidate_labels,
            unseen_labels=unseen_labels,
            unseen_score_scale=args.unseen_score_scale,
        )
        prediction_scores, cosine_scores, used_class_adjustments = apply_class_score_adjustments(
            prediction_scores,
            cosine_scores,
            candidate_labels,
            args,
            used_unseen_scale=used_scale,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "prediction_scores": prediction_scores[0].detach().cpu(),
        "cosine_scores": cosine_scores[0].detach().cpu(),
        "elapsed": elapsed,
        "used_scale": used_scale,
        "used_class_score_adjustments": used_class_adjustments,
        "detected_pose_frames": detected_pose_frames,
        "yolo_watched_detections": [dict(item) for item in object_map_runner.last_watched_detections],
    }


def apply_class_score_adjustments(prediction_scores, cosine_scores, candidate_labels, args, used_unseen_scale=False):
    adjustments = getattr(args, "class_score_adjustments", {}) or {}
    active = {
        int(label): value
        for label, value in adjustments.items()
        if int(label) in {int(candidate_label) for candidate_label in candidate_labels}
    }
    if not active:
        return prediction_scores, cosine_scores, False

    adjusted_cosine = cosine_scores.clone()
    adjusted_prediction = prediction_scores.clone() if used_unseen_scale else cosine_scores.clone()
    for index, label in enumerate(candidate_labels):
        item = active.get(int(label))
        if not item:
            continue
        scale = float(item.get("scale", 1.0))
        bias = float(item.get("bias", 0.0))
        adjusted_cosine[:, index] = (adjusted_cosine[:, index] * scale + bias).clamp(0.0, 1.0)
        adjusted_prediction[:, index] = adjusted_prediction[:, index] * scale + bias
        if not used_unseen_scale:
            adjusted_prediction[:, index] = adjusted_cosine[:, index]

    return adjusted_prediction, adjusted_cosine, True


def build_prediction_from_scores(score_result, args, candidate_labels, temporal_mode):
    top_k = min(args.top_k, len(candidate_labels))
    top_scores, top_indices = torch.topk(score_result["prediction_scores"], k=top_k)
    labels = [int(candidate_labels[index]) for index in top_indices.detach().cpu().tolist()]
    scores = [float(score_result["cosine_scores"][index].detach().cpu()) for index in top_indices.tolist()]
    ranking_scores = [float(score) for score in top_scores.detach().cpu().tolist()]
    unknown_info = evaluate_unknown_condition(score_result, args, scores[0] if scores else 0.0)
    display_labels = labels
    display_scores = scores
    display_ranking_scores = ranking_scores
    if unknown_info["is_unknown"]:
        keep_count = max(0, top_k - 1)
        display_labels = [UNKNOWN_LABEL] + labels[:keep_count]
        display_scores = [unknown_info["entropy"]] + scores[:keep_count]
        display_ranking_scores = [unknown_info["entropy"]] + ranking_scores[:keep_count]
    raw_labels = labels[:top_k]
    raw_scores = scores[:top_k]
    raw_ranking_scores = ranking_scores[:top_k]
    return {
        "labels": display_labels,
        "scores": display_scores,
        "ranking_scores": display_ranking_scores,
        "raw_labels": raw_labels,
        "raw_scores": raw_scores,
        "raw_ranking_scores": raw_ranking_scores,
        "unknown": unknown_info["is_unknown"],
        "unknown_reason": "entropy" if unknown_info["is_unknown"] else None,
        "unknown_entropy": unknown_info["entropy"],
        "unknown_top1_score": unknown_info["top1_score"],
        "elapsed": score_result["elapsed"],
        "used_scale": score_result["used_scale"],
        "used_class_score_adjustments": score_result.get("used_class_score_adjustments", False),
        "detected_pose_frames": score_result["detected_pose_frames"],
        "yolo_watched_detections": [dict(item) for item in score_result.get("yolo_watched_detections", [])],
        "temporal_mode": temporal_mode,
    }


def evaluate_unknown_condition(score_result, args, top1_score):
    cosine_scores = score_result["cosine_scores"].float()
    if getattr(args, "disable_unknown", False) or cosine_scores.numel() <= 1:
        return {
            "is_unknown": False,
            "entropy": 0.0,
            "top1_score": float(top1_score),
        }

    temperature = max(float(args.unknown_temperature), 1e-6)
    probs = torch.softmax(cosine_scores / temperature, dim=0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    normalized_entropy = float((entropy / torch.log(torch.tensor(float(probs.numel())))).detach().cpu())
    top1_score = float(top1_score)
    is_unknown = (
        normalized_entropy >= float(args.unknown_entropy_threshold)
        and top1_score <= float(args.unknown_top1_threshold)
    )
    return {
        "is_unknown": bool(is_unknown),
        "entropy": normalized_entropy,
        "top1_score": top1_score,
    }


def merge_short_long_scores(short_result, long_result, args, candidate_labels):
    prediction_scores = short_result["prediction_scores"].clone()
    cosine_scores = short_result["cosine_scores"].clone()
    for index, label in enumerate(candidate_labels):
        if is_long_action(label, args):
            prediction_scores[index] = long_result["prediction_scores"][index]
            cosine_scores[index] = long_result["cosine_scores"][index]
    watched_by_name = {}
    for result in (short_result, long_result):
        for detection in result.get("yolo_watched_detections", []):
            name = str(detection["name"])
            confidence = float(detection["confidence"])
            watched_by_name[name] = max(watched_by_name.get(name, 0.0), confidence)
    watched_detections = [
        {"name": name, "confidence": watched_by_name[name]}
        for name in ("stove", "biscuits", "pot")
        if name in watched_by_name
    ]
    return {
        "prediction_scores": prediction_scores,
        "cosine_scores": cosine_scores,
        "elapsed": short_result["elapsed"] + long_result["elapsed"],
        "used_scale": short_result["used_scale"],
        "detected_pose_frames": long_result["detected_pose_frames"],
        "yolo_watched_detections": watched_detections,
    }


def run_prediction(
    frame_buffer,
    skeleton_buffer,
    args,
    device,
    x3d_cfg,
    x3d_model,
    x3d_captured,
    ctrgcn_model,
    ctrgcn_captured,
    object_map_runner,
    clipgcn_model,
    candidate_labels,
    unseen_labels,
    use_amp,
):
    score_result = score_prediction_window(
        list(frame_buffer),
        list(skeleton_buffer),
        args,
        device,
        x3d_cfg,
        x3d_model,
        x3d_captured,
        ctrgcn_model,
        ctrgcn_captured,
        object_map_runner,
        clipgcn_model,
        candidate_labels,
        unseen_labels,
        use_amp,
    )
    return build_prediction_from_scores(score_result, args, candidate_labels, temporal_mode="13f")


def run_last13_history_prediction(
    history_buffer,
    args,
    device,
    x3d_cfg,
    x3d_model,
    x3d_captured,
    ctrgcn_model,
    ctrgcn_captured,
    object_map_runner,
    clipgcn_model,
    candidate_labels,
    unseen_labels,
    use_amp,
):
    frames, skeletons = latest_history_window(history_buffer, args.frames)
    return run_prediction(
        frames,
        skeletons,
        args,
        device,
        x3d_cfg,
        x3d_model,
        x3d_captured,
        ctrgcn_model,
        ctrgcn_captured,
        object_map_runner,
        clipgcn_model,
        candidate_labels,
        unseen_labels,
        use_amp,
    )


def run_temporal_prediction(
    history_buffer,
    args,
    device,
    x3d_cfg,
    x3d_model,
    x3d_captured,
    ctrgcn_model,
    ctrgcn_captured,
    object_map_runner,
    clipgcn_model,
    candidate_labels,
    unseen_labels,
    use_amp,
):
    if args.temporal_strategy == "uniform3s":
        frames, skeletons = sample_history_window(
            history_buffer,
            args.uniform_window_seconds,
            args.frames,
        )
        result = score_prediction_window(
            frames,
            skeletons,
            args,
            device,
            x3d_cfg,
            x3d_model,
            x3d_captured,
            ctrgcn_model,
            ctrgcn_captured,
            object_map_runner,
            clipgcn_model,
            candidate_labels,
            unseen_labels,
            use_amp,
        )
        return build_prediction_from_scores(result, args, candidate_labels, temporal_mode="3s")

    short_frames, short_skeletons = sample_history_window(
        history_buffer,
        args.short_window_seconds,
        args.frames,
    )
    short_result = score_prediction_window(
        short_frames,
        short_skeletons,
        args,
        device,
        x3d_cfg,
        x3d_model,
        x3d_captured,
        ctrgcn_model,
        ctrgcn_captured,
        object_map_runner,
        clipgcn_model,
        candidate_labels,
        unseen_labels,
        use_amp,
    )
    short_probe = build_prediction_from_scores(short_result, args, candidate_labels, temporal_mode="2s")
    rerank_count = min(args.long_rerank_top_k, len(short_probe["labels"]))
    needs_long_window = (
        args.temporal_strategy == "short-long"
        and any(is_long_action(label, args) for label in short_probe["labels"][:rerank_count])
        and history_window_ready(history_buffer, args.long_window_seconds, args)
    )
    if not needs_long_window:
        return short_probe

    long_frames, long_skeletons = sample_history_window(
        history_buffer,
        args.long_window_seconds,
        args.frames,
    )
    long_result = score_prediction_window(
        long_frames,
        long_skeletons,
        args,
        device,
        x3d_cfg,
        x3d_model,
        x3d_captured,
        ctrgcn_model,
        ctrgcn_captured,
        object_map_runner,
        clipgcn_model,
        candidate_labels,
        unseen_labels,
        use_amp,
    )
    merged_result = merge_short_long_scores(short_result, long_result, args, candidate_labels)
    return build_prediction_from_scores(merged_result, args, candidate_labels, temporal_mode="2s+4s")


def truncate_text(value, max_chars=46):
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def prediction_label_text(label, args, max_chars=46):
    label = int(label)
    if label == UNKNOWN_LABEL:
        return "UNKNOWN"
    name = getattr(args, "label_display_names", {}).get(label)
    suffix = " (unseen)" if is_unseen_label(label, args) else ""
    if not name:
        return f"class {label}{suffix}"
    return f"{label} {truncate_text(name, max_chars=max_chars)}{suffix}"


def is_unseen_label(label, args):
    if int(label) == UNKNOWN_LABEL:
        return False
    return int(label) in getattr(args, "unseen_label_set", set())


def prediction_color(label, rank, args):
    if int(label) == UNKNOWN_LABEL:
        return (80, 180, 255)
    if is_unseen_label(label, args):
        return (70, 210, 255) if rank == 1 else (120, 230, 255)
    return (80, 255, 120) if rank == 1 else (230, 230, 230)


def prediction_score_text(label, score, prediction):
    if int(label) == UNKNOWN_LABEL:
        entropy = float(prediction.get("unknown_entropy", score))
        top1_score = float(prediction.get("unknown_top1_score", 0.0))
        return f"H={entropy:.2f} top1={top1_score:.2f}"
    return f"{score:.4f}"


def vote_ranked_prediction(prediction_history, args):
    if not prediction_history:
        return None

    latest = prediction_history[-1]
    window = max(1, int(getattr(args, "display_filter_window", 1)))
    recent = list(prediction_history)[-window:]
    if window <= 1 or len(recent) <= 1:
        return latest

    max_rows = min(
        int(getattr(args, "top_k", len(latest.get("labels", [])))),
        max(len(item.get("labels", [])) for item in recent),
    )
    voted_labels = []
    voted_scores = []
    voted_ranking_scores = []
    used_labels = set()

    for rank_index in range(max_rows):
        stats = {}
        for history_index, item in enumerate(recent):
            labels = item.get("labels", [])
            if rank_index >= len(labels):
                continue
            label = int(labels[rank_index])
            score = float(item.get("scores", [0.0] * len(labels))[rank_index])
            ranking_scores = item.get("ranking_scores", item.get("scores", []))
            ranking_score = float(ranking_scores[rank_index]) if rank_index < len(ranking_scores) else score
            entry = stats.setdefault(
                label,
                {
                    "count": 0,
                    "score_sum": 0.0,
                    "ranking_score_sum": 0.0,
                    "latest_seen": -1,
                },
            )
            entry["count"] += 1
            entry["score_sum"] += score
            entry["ranking_score_sum"] += ranking_score
            entry["latest_seen"] = history_index

        if not stats:
            continue

        ranked = sorted(
            stats.items(),
            key=lambda item: (
                item[1]["count"],
                item[1]["ranking_score_sum"] / max(1, item[1]["count"]),
                item[1]["latest_seen"],
            ),
            reverse=True,
        )
        selected_label, selected_stats = next(
            ((label, data) for label, data in ranked if label not in used_labels),
            ranked[0],
        )
        used_labels.add(selected_label)
        count = max(1, selected_stats["count"])
        voted_labels.append(int(selected_label))
        voted_scores.append(float(selected_stats["score_sum"] / count))
        voted_ranking_scores.append(float(selected_stats["ranking_score_sum"] / count))

    filtered = dict(latest)
    filtered["labels"] = voted_labels
    filtered["scores"] = voted_scores
    filtered["ranking_scores"] = voted_ranking_scores
    filtered["unknown"] = UNKNOWN_LABEL in voted_labels
    if filtered["unknown"]:
        reason_counts = {}
        for item in recent:
            if UNKNOWN_LABEL not in [int(label) for label in item.get("labels", [])]:
                continue
            reason = item.get("unknown_reason") or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason_counts:
            filtered["unknown_reason"] = max(reason_counts.items(), key=lambda item: item[1])[0]
    filtered["display_filter_window"] = window
    filtered["display_filter_count"] = len(recent)
    filtered["raw_temporal_mode"] = latest.get("temporal_mode", "13f")
    filtered["temporal_mode"] = f"{latest.get('temporal_mode', '13f')} vote{len(recent)}/{window}"
    return filtered


def draw_overlay(frame_bgr, prediction, frame_count, args):
    overlay = frame_bgr.copy()
    shown_rows = 1 if prediction is None else len(prediction.get("labels", []))
    watched_detections = [] if prediction is None else prediction.get("yolo_watched_detections", [])
    detected_names = {str(item["name"]) for item in watched_detections}
    watched_flags_text = "  ".join(
        f"{name}={int(name in detected_names)}" for name in ("stove", "biscuits", "pot")
    )
    overlay_height = max(175, 135 + shown_rows * 25)
    overlay_width = min(frame_bgr.shape[1], 920)
    cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    cv2.putText(
        frame_bgr,
        "CLIPGCN realtime",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if args.temporal_strategy == "last13":
        window_text = f"latest_contiguous={args.frames}"
    elif args.temporal_strategy == "uniform3s":
        window_text = f"window={args.uniform_window_seconds:g}s"
    else:
        window_text = f"window={args.short_window_seconds:g}s/{args.long_window_seconds:g}s"
    cv2.putText(
        frame_bgr,
        f"frames={args.frames}  {window_text}  pose={args.runtime_pose_source}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if prediction is None:
        cv2.putText(
            frame_bgr,
            f"YOLO objects: {watched_flags_text}",
            (18, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
        text = f"warming up: {frame_count}/{args.frames}"
        cv2.putText(frame_bgr, text, (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        return frame_bgr

    core_latency_ms = prediction["elapsed"] * 1000.0
    end_to_end_latency_ms = prediction.get("end_to_end_elapsed", prediction["elapsed"]) * 1000.0
    entropy_text = (
        f"H={float(prediction.get('unknown_entropy', 0.0)):.2f} "
        f"top1={float(prediction.get('unknown_top1_score', 0.0)):.2f}"
    )
    cv2.putText(
        frame_bgr,
        (
            f"E2E {end_to_end_latency_ms:.1f} ms  core {core_latency_ms:.1f} ms  "
            f"pose_frames {prediction['detected_pose_frames']}/{args.frames}  "
            f"{entropy_text}  mode={prediction.get('temporal_mode', '13f')}"
        ),
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"YOLO objects: {watched_flags_text}",
        (18, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    ranking_base_y = 117
    for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1):
        y = ranking_base_y + rank * 24
        color = prediction_color(label, rank, args)
        cv2.putText(
            frame_bgr,
            f"{rank}. {prediction_label_text(label, args, max_chars=62)}: {prediction_score_text(label, score, prediction)}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def print_prediction(prediction, args):
    if prediction is None:
        return
    parts = [
        f"{rank}. {prediction_label_text(label, args)}={prediction_score_text(label, score, prediction)}"
        for rank, (label, score) in enumerate(zip(prediction["labels"], prediction["scores"]), start=1)
    ]
    mode = prediction.get("temporal_mode", "13f")
    core_latency_ms = prediction["elapsed"] * 1000.0
    end_to_end_latency_ms = prediction.get("end_to_end_elapsed", prediction["elapsed"]) * 1000.0
    watched_detections = prediction.get("yolo_watched_detections", [])
    detected_names = {str(item["name"]) for item in watched_detections}
    watched_text = " | YOLO " + ", ".join(
        f"{name}={int(name in detected_names)}" for name in ("stove", "biscuits", "pot")
    )
    print(
        f"[E2E {end_to_end_latency_ms:.1f} ms | core {core_latency_ms:.1f} ms | {mode}] "
        + " | ".join(parts)
        + watched_text,
        flush=True,
    )


def main():
    args = parse_args()
    validate_args(args)
    args.action_duration_types = DEFAULT_ACTION_DURATION_TYPES
    args = maybe_enable_headless(args)
    args.runtime_pose_source = resolve_runtime_pose_source(args)

    config_path = resolve_existing_path(args.config, label="config")
    config = load_config(config_path)
    args.label_display_names = load_action_display_names(config, config_path)
    args.x3d_root = resolve_existing_path(args.x3d_root, label="X3D root")
    args.x3d_config = resolve_existing_path(
        args.x3d_config,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, SCRIPT_ROOT],
        label="X3D config",
    )
    args.x3d_checkpoint = resolve_existing_path(
        args.x3d_checkpoint,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, SCRIPT_ROOT],
        label="X3D checkpoint",
    )
    args.ctrgcn_root = resolve_existing_path(args.ctrgcn_root, label="CTR-GCN root")
    args.ctrgcn_config = resolve_existing_path(
        args.ctrgcn_config,
        search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, SCRIPT_ROOT],
        label="CTR-GCN config",
    )
    args.ctrgcn_weights = resolve_existing_path(
        args.ctrgcn_weights,
        search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, SCRIPT_ROOT],
        label="CTR-GCN weights",
    )
    if not args.no_yolo:
        try:
            args.yolo_repo = resolve_existing_path(args.yolo_repo, label="YOLO repo")
        except FileNotFoundError as exc:
            print(f"Warning: {exc}. Falling back to zero object maps.")
            args.no_yolo = True
        else:
            try:
                args.yolo_weights = resolve_existing_path(
                    args.yolo_weights,
                    search_roots=[Path(args.yolo_repo), Path(args.yolo_repo).parent, SCRIPT_ROOT],
                    label="YOLO weights",
                )
            except FileNotFoundError as exc:
                print(f"Warning: {exc}. Falling back to zero object maps.")
                args.no_yolo = True

    requested_device = str(config["runtime"].get("device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "The config requests CUDA, but torch.cuda.is_available() is False. "
            "Pass --allow-cpu intentionally, or run in an environment with GPU access."
        )
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_split_dir = args.class_split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    class_split_dir = get_path_from_config(config_path, class_split_dir)
    class_selection = resolve_realtime_class_labels(args, class_split_dir, all_labels=args.label_display_names.keys())
    unseen_labels = class_selection["unseen_labels"]
    args.unseen_label_set = set(unseen_labels)
    args.class_score_adjustments = class_selection.get("score_adjustments", {})
    candidate_labels = class_selection["candidate_labels"]

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model = None
    ctrgcn_captured = None
    ctrgcn_hook = None
    if args.runtime_pose_source == "mediapipe":
        ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    yolo_model = load_yolo_model(args, device)
    object_map_runner = ObjectMapRunner(yolo_model, device, args)
    clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
        args,
        config,
        config_path,
        device,
        candidate_labels,
    )
    candidate_labels = [int(label) for label in candidate_labels]
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"

    print("Realtime CLIPGCN webcam inference")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  class_split_dir: {class_split_dir}")
    print(f"  class_selection: {class_selection['source']}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {candidate_labels}")
    print(f"  seen_labels: {class_selection['seen_labels']}")
    print(f"  unseen_labels: {unseen_labels}")
    if class_selection["excluded_labels"]:
        print(f"  excluded_labels: {class_selection['excluded_labels']}")
    if args.class_score_adjustments:
        print(f"  class_score_adjustments: {args.class_score_adjustments}")
    print(f"  unseen_score_scale: {args.unseen_score_scale:g}")
    print(f"  display_filter_window: {args.display_filter_window}")
    print(
        "  unknown_rule: "
        f"{'off' if args.disable_unknown else 'on'} "
        f"entropy>={args.unknown_entropy_threshold:g} top1<={args.unknown_top1_threshold:g}"
    )
    print(f"  pose_source: {args.pose_source}")
    print(f"  runtime_pose_source: {args.runtime_pose_source}")
    print(f"  temporal_strategy: {args.temporal_strategy}")
    if args.temporal_strategy == "last13":
        print(f"  latest_contiguous_frames: {args.frames}")
    elif args.temporal_strategy == "uniform3s":
        print(f"  uniform_window_seconds: {args.uniform_window_seconds:g}")
    else:
        print(f"  short_window_seconds: {args.short_window_seconds:g}")
        print(f"  long_window_seconds: {args.long_window_seconds:g}")
    print(f"  camera_index: {args.camera_index}")
    print("  YOLO overlay targets: stove, biscuits, pot")
    print("  latency: E2E=newest raw frame to voted prediction; core=model section only")
    print("  quit: press q or ESC")

    pose_source = MediaPipePoseSource(args) if args.runtime_pose_source == "mediapipe" else None
    cap = open_camera(args)
    history_buffer = deque()
    prediction = None
    display_prediction = None
    prediction_history = deque(maxlen=args.display_filter_window)
    captured_frames = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            # Input acquisition is excluded; everything from the newly read raw
            # frame through pose extraction and the final voted prediction is E2E.
            frame_processing_start = time.perf_counter()
            captured_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = time.monotonic()
            skeleton_sample = None
            if pose_source is not None:
                skeleton_sample = pose_source.process(frame_rgb)
            history_buffer.append(
                {
                    "timestamp": timestamp,
                    "frame_rgb": frame_rgb,
                    "skeleton": skeleton_sample,
                }
            )
            trim_history_buffer(history_buffer, args, timestamp)

            if args.temporal_strategy == "last13":
                frame_count = min(len(history_buffer), args.frames)
                should_predict = (
                    latest_history_ready(history_buffer, args)
                    and captured_frames % args.predict_every == 0
                )
            else:
                frame_count = warmup_frame_count(history_buffer, args)
                pose_ready = args.runtime_pose_source == "zero" or all(
                    sample.get("skeleton") is not None for sample in history_buffer
                )
                should_predict = (
                    history_window_ready(history_buffer, active_window_seconds(args), args)
                    and pose_ready
                    and captured_frames % args.predict_every == 0
                )
            if should_predict:
                if args.temporal_strategy == "last13":
                    prediction = run_last13_history_prediction(
                        history_buffer,
                        args,
                        device,
                        x3d_cfg,
                        x3d_model,
                        x3d_captured,
                        ctrgcn_model,
                        ctrgcn_captured,
                        object_map_runner,
                        clipgcn_model,
                        candidate_labels,
                        unseen_labels,
                        use_amp,
                    )
                else:
                    prediction = run_temporal_prediction(
                        history_buffer,
                        args,
                        device,
                        x3d_cfg,
                        x3d_model,
                        x3d_captured,
                        ctrgcn_model,
                        ctrgcn_captured,
                        object_map_runner,
                        clipgcn_model,
                        candidate_labels,
                        unseen_labels,
                        use_amp,
                    )
                prediction_history.append(prediction)
                display_prediction = vote_ranked_prediction(prediction_history, args)
                end_to_end_elapsed = finish_end_to_end_timer(frame_processing_start, device)
                prediction["end_to_end_elapsed"] = end_to_end_elapsed
                display_prediction["end_to_end_elapsed"] = end_to_end_elapsed
                if args.headless:
                    print_prediction(display_prediction, args)

            if not args.headless:
                draw_overlay(frame_bgr, display_prediction, frame_count, args)
                cv2.imshow(args.window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        x3d_hook.remove()
        if ctrgcn_hook is not None:
            ctrgcn_hook.remove()
        if pose_source is not None:
            pose_source.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
