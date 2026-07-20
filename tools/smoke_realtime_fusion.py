#!/usr/bin/env python
"""Smoke-test the realtime CLIPGCN fusion path without a webcam.

This runs synthetic RGB frames and synthetic 25-joint skeletons through the
same X3D, CTR-GCN, YOLO, and CLIPGCN fusion function used by webcam_realtime.py.
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

CLIPGCN_ROOT = Path(__file__).resolve().parents[1]
if str(CLIPGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIPGCN_ROOT))

from test import load_split_classes, load_split_metadata
from test_raw_end_to_end import (
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_x3d_model,
    load_yolo_model,
)
from train import get_device, get_path_from_config, load_config, print_device_info
from webcam_realtime import (
    resolve_existing_path,
    run_prediction,
    validate_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CLIPGCN_ROOT / "config_50_5.yaml"))
    parser.add_argument("--class-split-dir", default=str(CLIPGCN_ROOT / "data" / "contrastive_zsl_splits" / "50_5"))
    parser.add_argument("--candidate-scope", choices=["unseen", "seen", "all"], default="all")
    parser.add_argument("--unseen-score-scale", type=float, default=1.3)
    parser.add_argument("--clipgcn-checkpoint", default=None)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260701)

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

    parser.add_argument("--ctrgcn-root", default=str(CTRGCN_ROOT))
    parser.add_argument(
        "--ctrgcn-config",
        default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"),
    )
    parser.add_argument(
        "--ctrgcn-weights",
        default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"),
    )
    parser.add_argument("--ctrgcn-hook-layer", default="l4")

    parser.add_argument("--yolo-repo", default="/workspace/yolov5")
    parser.add_argument("--yolo-weights", default="/workspace/yolov5/yolov5m.pt")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--yolo-detect-every", type=int, default=1)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


def synthetic_frames(num_frames, height, width, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    frames = []
    for frame_idx in range(num_frames):
        noise = rng.normal(0.0, 3.0, size=(height, width)).astype(np.float32)
        red = (x[None, :] + frame_idx * 7 + noise).clip(0, 255)
        green = (y + frame_idx * 5).clip(0, 255)
        blue = ((red * 0.35 + green * 0.65) + frame_idx * 3).clip(0, 255)
        frame = np.stack([red, np.broadcast_to(green, red.shape), blue], axis=-1).astype(np.uint8)
        cv2.circle(
            frame,
            (int(width * (0.25 + 0.03 * frame_idx)), int(height * 0.5)),
            max(8, height // 12),
            (245, 245, 245),
            thickness=-1,
        )
        frames.append(frame)
    return frames


def synthetic_skeleton_buffer(num_frames):
    buffer = deque(maxlen=num_frames)
    joint_offsets = np.linspace(-0.35, 0.35, 25, dtype=np.float32)
    for frame_idx in range(num_frames):
        phase = frame_idx / max(num_frames - 1, 1)
        joints = np.zeros((25, 3), dtype=np.float32)
        joints[:, 0] = joint_offsets + 0.06 * np.sin(phase * np.pi * 2.0)
        joints[:, 1] = np.linspace(-0.55, 0.55, 25, dtype=np.float32)
        joints[:, 2] = 0.2 + 0.04 * np.cos(phase * np.pi * 2.0 + joint_offsets)
        joints[0] = (0.0, 0.1, 0.25)
        joints[1] = (0.0, -0.05, 0.25)
        joints[2] = (0.0, -0.25, 0.25)
        joints[3] = (0.0, -0.45, 0.2)
        buffer.append(
            {
                "joints_3d": joints,
                "joint_xy": np.clip(joints[:, :2], -1.0, 1.0).astype(np.float32),
                "detected": True,
            }
        )
    return buffer


def main():
    args = parse_args()
    args.pose_source = "mediapipe"
    args.runtime_pose_source = "mediapipe"
    args.predict_every = args.frames
    validate_args(args)

    config_path = resolve_existing_path(args.config, label="config")
    config = load_config(config_path)
    args.x3d_root = resolve_existing_path(args.x3d_root, label="X3D root")
    args.x3d_config = resolve_existing_path(
        args.x3d_config,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, CLIPGCN_ROOT],
        label="X3D config",
    )
    args.x3d_checkpoint = resolve_existing_path(
        args.x3d_checkpoint,
        search_roots=[Path(args.x3d_root), Path(args.x3d_root).parent, CLIPGCN_ROOT],
        label="X3D checkpoint",
    )
    args.ctrgcn_root = resolve_existing_path(args.ctrgcn_root, label="CTR-GCN root")
    args.ctrgcn_config = resolve_existing_path(
        args.ctrgcn_config,
        search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, CLIPGCN_ROOT],
        label="CTR-GCN config",
    )
    args.ctrgcn_weights = resolve_existing_path(
        args.ctrgcn_weights,
        search_roots=[Path(args.ctrgcn_root), Path(args.ctrgcn_root).parent, CLIPGCN_ROOT],
        label="CTR-GCN weights",
    )

    if not args.no_yolo:
        args.yolo_repo = resolve_existing_path(args.yolo_repo, label="YOLO repo")
        args.yolo_weights = resolve_existing_path(
            args.yolo_weights,
            search_roots=[Path(args.yolo_repo), Path(args.yolo_repo).parent, CLIPGCN_ROOT],
            label="YOLO weights",
        )

    requested_device = str(config["runtime"].get("device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("The config requests CUDA, but torch.cuda.is_available() is False.")
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    split_metadata = load_split_metadata(class_split_dir)
    unseen_labels = [int(label) for label in split_metadata.get("unseen_classes", [])]
    candidate_labels = load_split_classes(class_split_dir, args.candidate_scope)

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
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

    frame_buffer = deque(synthetic_frames(args.frames, args.height, args.width, args.seed), maxlen=args.frames)
    skeleton_buffer = synthetic_skeleton_buffer(args.frames)
    try:
        prediction = run_prediction(
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
        )
    finally:
        x3d_hook.remove()
        ctrgcn_hook.remove()

    print("Realtime fusion smoke test passed")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  pose_branch: CTR-GCN synthetic skeleton")
    print(f"  object_branch: {'YOLO' if yolo_model is not None else 'zero object map'}")
    print(f"  video_branch: X3D")
    print(f"  top_labels: {prediction['labels']}")
    print(f"  top_scores: {[round(score, 6) for score in prediction['scores']]}")
    print(f"  elapsed_ms: {prediction['elapsed'] * 1000.0:.2f}")


if __name__ == "__main__":
    main()
