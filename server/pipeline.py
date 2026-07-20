"""CLIPGCN tri-modal inference behind the REST service.

Everything heavy (X3D, CTR-GCN, YOLO, the CLIP text encoder, MediaPipe pose) is
loaded once in ActionRecognizer.__init__ and shared by all requests. Most of the
loading/scoring code is lifted from webcam_realtime.py and test_raw_end_to_end.py.

If the checkpoints or their imports aren't around we start in mock mode:
recognize() returns a random class and add_class() just keeps the prompt text.
Lets you poke at the API on a machine without the weights.
"""

import logging
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

CLIPGCN_ROOT = Path(__file__).resolve().parents[1]
if str(CLIPGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIPGCN_ROOT))

log = logging.getLogger("vpoclip.pipeline")

NUM_FRAMES = 13
CUSTOM_LABEL_BASE = 1000
DEFAULT_CONFIG = CLIPGCN_ROOT / "config_50_5.yaml"
# tasks-only mediapipe builds need this asset and won't find it by themselves
POSE_TASK_ASSET = CLIPGCN_ROOT / "local_models" / "pose_landmarker_full.task"
# optional per-class on/off table; if it's there we keep only the enabled rows
CLASS_CONFIG = CLIPGCN_ROOT / "realtime_class_config.csv"

# mock-mode last resort, when even the xlsx vocabulary can't be read
FALLBACK_ACTIONS = [
    "eating food with a fork",
    "pouring water into a cup",
    "taking medicine",
    "washing hands",
    "reading a book",
    "talking on the phone",
    "sitting down on a chair",
    "standing up",
]

try:
    import torch

    from action_label_utils import load_action_display_names
    from model import load_action_descriptions
    from test import logits_to_unit_cosine_scores
    from test_raw_end_to_end import (
        CTRGCN_ROOT,
        X3D_ROOT,
        ctrgcn_pose_from_model,
        load_clipgcn_model,
        load_ctrgcn_model,
        load_x3d_model,
        load_yolo_model,
        object_maps_from_yolo,
        x3d_features_from_model,
        x3d_tensor_from_frames,
    )
    from train import get_device, get_path_from_config, load_config
    from webcam_realtime import (
        MediaPipePoseSource,
        build_mediapipe_skeleton_inputs,
        build_zero_pose_inputs,
        default_mediapipe_model_asset,
        default_yolo_repo,
        default_yolo_weights,
        labels_from_class_config,
    )

    IMPORT_ERROR = None
except Exception as exc:  # torch or one of the model repos is missing
    IMPORT_ERROR = exc


def build_pipeline_args():
    """Same defaults webcam_realtime.py uses, just without the CLI."""
    return SimpleNamespace(
        frames=NUM_FRAMES,
        pose_source="mediapipe",
        runtime_pose_source="mediapipe",
        x3d_root=str(X3D_ROOT),
        x3d_config=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"),
        x3d_checkpoint=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"),
        x3d_layer="s5",
        ctrgcn_root=str(CTRGCN_ROOT),
        ctrgcn_config=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"),
        ctrgcn_weights=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"),
        ctrgcn_hook_layer="l4",
        yolo_repo=default_yolo_repo(),
        yolo_weights=default_yolo_weights(),
        yolo_size=640,
        yolo_conf=0.25,
        yolo_iou=0.45,
        yolo_half=False,
        yolo_detect_every=1,
        no_yolo=False,
        object_grid_size=6,
        object_value="presence",
        object_max_distance_weight=10.0,
        clipgcn_checkpoint=None,
        mediapipe_model_complexity=1,
        mediapipe_min_detection_confidence=0.5,
        mediapipe_min_tracking_confidence=0.5,
        mediapipe_min_visibility=0.2,
        mediapipe_model_asset=(
            str(POSE_TASK_ASSET) if POSE_TASK_ASSET.exists() else default_mediapipe_model_asset()
        ),
        class_config=str(CLASS_CONFIG) if CLASS_CONFIG.exists() else None,
    )


class ActionRecognizer:
    """Open-vocabulary recognizer over the frozen CLIPGCN model.

    self.classes is the vocabulary and the source of truth: row i of the model's
    text feature bank is the prototype for self.classes[i].
    """

    def __init__(self, config_path=None):
        self.lock = threading.Lock()
        self.mock_mode = False
        self.device_name = "cpu"
        self.classes = []  # each item: {"label", "name", "prompts", "split"}
        self.unseen_labels = []
        # unseen/zero-shot classes score lower than the trained ones, so bump
        # them up a bit to keep them in the race (same trick as the offline eval)
        self.unseen_score_scale = 1.3
        self.pose_source = None
        try:
            self._load_models(config_path or str(DEFAULT_CONFIG))
            log.info("Loaded CLIPGCN pipeline on %s with %d classes", self.device_name, len(self.classes))
        except Exception:
            log.warning(
                "Could not load the CLIPGCN pipeline, falling back to mock mode "
                "(recognize returns random predictions).",
                exc_info=True,
            )
            self.mock_mode = True
            self._load_mock_vocabulary()

    def _load_models(self, config_path):
        if IMPORT_ERROR is not None:
            raise IMPORT_ERROR

        config = load_config(config_path)
        self.args = build_pipeline_args()
        self.device = get_device(None)  # cuda:0 if we have it, otherwise cpu
        self.device_name = str(self.device)
        self.use_amp = bool(config["runtime"].get("amp", False)) and self.device.type == "cuda"

        self.x3d_model, self.x3d_captured, self._x3d_hook, self.x3d_cfg = load_x3d_model(self.args, self.device)
        self.ctrgcn_model, self.ctrgcn_captured, self._ctrgcn_hook = load_ctrgcn_model(self.args, self.device)
        try:
            self.yolo_model = load_yolo_model(self.args, self.device)
        except Exception as yolo_exc:
            log.warning("YOLO unavailable (%s), using zero object maps.", yolo_exc)
            self.yolo_model = None

        # full vocabulary from the xlsx, keyed by label
        text_config = config["data"]["text"]
        all_labels, all_texts, _records = load_action_descriptions(
            xlsx_path=get_path_from_config(config_path, text_config["xlsx"]),
            text_column=text_config.get("text_column", "global_description"),
            id_column=text_config.get("id_column", "ID"),
            label_offset=text_config.get("label_offset", 1),
            prompt_template=text_config.get("prompt_template", "{global_description}"),
        )
        label_to_text = {int(label): text for label, text in zip(all_labels, all_texts)}
        display_names = load_action_display_names(config, config_path)

        candidate_labels, self.unseen_labels = self._resolve_class_config(all_labels)

        self.clipgcn, checkpoint_path, labels = load_clipgcn_model(
            self.args, config, config_path, self.device, candidate_labels=candidate_labels
        )
        log.info("CLIPGCN checkpoint: %s", checkpoint_path)

        try:
            self.pose_source = MediaPipePoseSource(self.args)
        except Exception as pose_exc:
            log.warning("MediaPipe pose unavailable (%s), using zero pose features.", pose_exc)
            self.pose_source = None

        # one prototype row per class, in the same order as the model text bank
        unseen = set(self.unseen_labels)
        self.classes = []
        for label in labels:
            label = int(label)
            self.classes.append({
                "label": label,
                "name": display_names.get(label, label_to_text[label]),
                "prompts": [label_to_text[label]],
                "split": "unseen" if label in unseen else "seen",
            })
        self._prototypes = self.clipgcn.text_features.detach().clone()
        seen = sum(1 for c in self.classes if c["split"] == "seen")
        log.info("Vocabulary: %d classes (%d seen, %d unseen)", len(self.classes), seen, len(self.classes) - seen)

    def _resolve_class_config(self, all_labels):
        """Read realtime_class_config.csv if it exists. Returns
        (candidate_labels or None, unseen_labels); None means keep everything."""
        path = self.args.class_config
        if not path or not Path(path).exists():
            return None, []
        universe = sorted(int(label) for label in all_labels)
        selection = labels_from_class_config(path, universe)
        candidate = sorted(set(selection["seen_labels"]) | set(selection["unseen_labels"]))
        log.info("Class config %s: keeping %d of %d classes", Path(path).name, len(candidate), len(universe))
        return candidate, sorted(selection["unseen_labels"])

    def _load_mock_vocabulary(self):
        try:
            from model import _read_xlsx_rows

            rows = _read_xlsx_rows(str(CLIPGCN_ROOT / "ntu55_global_descriptions.xlsx"))
            names = [row.get("action_description") or row.get("global_description") for row in rows]
            names = [name for name in names if name]
        except Exception:
            names = list(FALLBACK_ACTIONS)
        self.classes = [{"label": i, "name": name, "prompts": [name]} for i, name in enumerate(names)]

    def list_classes(self):
        with self.lock:
            return [c["name"] for c in self.classes]

    def unseen_classes(self):
        with self.lock:
            return [c["name"] for c in self.classes if c.get("split") == "unseen"]

    @property
    def vocabulary_size(self):
        return len(self.classes)

    def _find_class(self, name):
        wanted = name.strip().lower()
        for index, c in enumerate(self.classes):
            if c["name"].strip().lower() == wanted:
                return index
        return None

    def add_class(self, name, descriptions):
        """Embed the description(s) with the CLIP text encoder and add or replace
        the class prototype. No retraining involved."""
        descriptions = [t.strip() for t in descriptions if t and t.strip()]
        if not descriptions:
            raise ValueError("add_class needs at least one non-empty description")

        with self.lock:
            existing = self._find_class(name)
            if not self.mock_mode:
                embeddings = self.clipgcn.text_encoder(descriptions)
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                prototype = embeddings.mean(dim=0, keepdim=True).to(self._prototypes.device)

            if existing is not None:
                self.classes[existing]["prompts"] = descriptions
                if not self.mock_mode:
                    self._prototypes[existing] = prototype[0]
            else:
                label = max((c["label"] for c in self.classes), default=-1)
                label = max(label + 1, CUSTOM_LABEL_BASE)
                # anything added at runtime is zero-shot, so unseen
                self.classes.append(
                    {"label": label, "name": name.strip(), "prompts": descriptions, "split": "unseen"}
                )
                if not self.mock_mode:
                    self._prototypes = torch.cat([self._prototypes, prototype], dim=0)

            if not self.mock_mode:
                self._refresh_text_bank()
        log.info("Vocabulary now has %d classes (added %r)", len(self.classes), name)

    def remove_class(self, name):
        with self.lock:
            index = self._find_class(name)
            if index is None:
                return False
            del self.classes[index]
            if not self.mock_mode:
                keep = [i for i in range(self._prototypes.shape[0]) if i != index]
                self._prototypes = self._prototypes[keep]
                self._refresh_text_bank()
        return True

    def _refresh_text_bank(self):
        self.clipgcn.set_text_features(self._prototypes, [c["label"] for c in self.classes])

    def recognize(self, frames, top_k=5):
        """Tri-modal inference on a clip of BGR frames. Any length >= 1 works;
        frames get resampled to the 13-frame window the checkpoint expects."""
        if not frames:
            raise ValueError("recognize() needs at least one frame")
        start = time.perf_counter()
        with self.lock:
            if not self.classes:
                raise RuntimeError("The vocabulary is empty, add a class first")
            names = [c["name"] for c in self.classes]
            splits = [c.get("split", "seen") for c in self.classes]
            scores = self._mock_scores() if self.mock_mode else self._score_clip(frames)

        # give the unseen classes the same little boost the offline eval uses
        if self.unseen_score_scale != 1.0:
            for i, split in enumerate(splits):
                if split == "unseen":
                    scores[i] = min(1.0, scores[i] * self.unseen_score_scale)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        topk = []
        for i in order[:min(top_k, len(order))]:
            topk.append({"action": names[i], "confidence": round(float(scores[i]), 4)})
        return {
            "action": topk[0]["action"],
            "confidence": topk[0]["confidence"],
            "topk": topk,
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 1),
        }

    def _score_clip(self, frames):
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if len(frames_rgb) != NUM_FRAMES:
            indices = np.linspace(0, len(frames_rgb) - 1, NUM_FRAMES).round().astype(int)
            frames_rgb = [frames_rgb[i] for i in indices]

        clip = x3d_tensor_from_frames(
            frames_rgb,
            int(self.x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
            self.x3d_cfg.TRANSFORM.MEAN,
            self.x3d_cfg.TRANSFORM.STD,
        ).unsqueeze(0).to(self.device)

        if self.pose_source is not None:
            samples = [self.pose_source.process(frame) for frame in frames_rgb]
            skeletons, joint_xy, _detected = build_mediapipe_skeleton_inputs(samples, self.device)
        else:
            pose_features, joint_xy = build_zero_pose_inputs(1, NUM_FRAMES, self.device)

        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            object_maps = object_maps_from_yolo(
                self.yolo_model, [frames_rgb[len(frames_rgb) // 2]], self.device, self.args
            )
            video_features = x3d_features_from_model(self.x3d_model, self.x3d_captured, clip, self.args)
            if self.pose_source is not None:
                pose_features = ctrgcn_pose_from_model(self.ctrgcn_model, self.ctrgcn_captured, skeletons, self.args)
            logits = self.clipgcn(video_features, pose_features, object_maps, joint_xy)
            scores = logits_to_unit_cosine_scores(self.clipgcn, logits)
        return scores[0].float().cpu().tolist()

    def _mock_scores(self):
        scores = [random.uniform(0.05, 0.45) for _ in self.classes]
        scores[random.randrange(len(scores))] = random.uniform(0.55, 0.9)
        return scores
