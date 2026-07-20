#!/usr/bin/env python
"""Webcam client for the VPOCLIP action recognition service.

Grabs webcam frames, sends a short clip to the server a couple of times a
second and draws whatever prediction comes back. All the model inference lives
on the server, so here we only need opencv + requests.

Keys in the video window:
    a       add a new action (you type the name/description in the terminal)
    r       ask for a caregiver report (printed in the terminal)
    q/ESC   quit
"""

import argparse
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam client for the VPOCLIP service.")
    parser.add_argument(
        "--server",
        default=os.environ.get("VPOCLIP_SERVER", "http://127.0.0.1:8000"),
        help="Base URL of the service (or set VPOCLIP_SERVER).",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=13, help="Frames per clip, must match the server model.")
    parser.add_argument("--window", type=float, default=2.0, help="Seconds of video covered by one clip.")
    parser.add_argument("--interval", type=float, default=0.5, help="Min seconds between recognize requests.")
    parser.add_argument("--send-width", type=int, default=640, help="Resize frames to this width before upload (0 = keep).")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=5.0, help="Request timeout in seconds.")
    return parser.parse_args()


class RecognizeWorker(threading.Thread):
    """Sends the newest clip to /recognize on a background thread so the video
    window never blocks on the network."""

    def __init__(self, args, frame_buffer, buffer_lock):
        super().__init__(daemon=True)
        self.args = args
        self.frame_buffer = frame_buffer
        self.buffer_lock = buffer_lock
        self.prediction = None
        self.status = "connecting..."
        self.running = True

    def run(self):
        while self.running:
            clip = self.sample_clip()
            if clip is None:
                time.sleep(0.1)
                continue
            started = time.time()
            try:
                files = [
                    ("files", (f"frame{i}.jpg", data, "image/jpeg"))
                    for i, data in enumerate(clip)
                ]
                response = requests.post(
                    f"{self.args.server}/recognize", files=files, timeout=self.args.timeout
                )
                response.raise_for_status()
                self.prediction = response.json()
                self.prediction["rtt_ms"] = (time.time() - started) * 1000.0
                self.status = "ok"
            except requests.RequestException as exc:
                self.status = f"server error: {type(exc).__name__}"
                print(f"Warning: recognize request failed: {exc}")
                time.sleep(1.0)
            # don't hammer the server, at most one request per interval
            remaining = self.args.interval - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)

    def sample_clip(self):
        """Pick args.frames evenly spaced frames from the last window seconds."""
        with self.buffer_lock:
            samples = list(self.frame_buffer)
        if len(samples) < self.args.frames:
            return None
        if samples[-1][0] - samples[0][0] < self.args.window * 0.8:
            return None
        cutoff = samples[-1][0] - self.args.window
        recent = [item for item in samples if item[0] >= cutoff]
        indices = np.linspace(0, len(recent) - 1, self.args.frames).round().astype(int)
        encoded = []
        for i in indices:
            frame = recent[i][1]
            if self.args.send_width > 0 and frame.shape[1] > self.args.send_width:
                height = int(frame.shape[0] * self.args.send_width / frame.shape[1])
                frame = cv2.resize(frame, (self.args.send_width, height))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
            if not ok:
                return None
            encoded.append(jpeg.tobytes())
        return encoded


def add_action_dialog(args):
    print("\n--- Add a new action (video is paused) ---")
    name = input("Action name: ").strip()
    if not name:
        print("Cancelled.")
        return
    description = input("Casual description (optional, any language): ").strip()
    body = {"name": name}
    if description:
        body["casual_description"] = description
    try:
        response = requests.post(f"{args.server}/add_action", json=body, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"add_action failed: {exc}")
        return
    data = response.json()
    print(f"Added {data['added']!r} (source: {data['source']}, vocabulary: {data['vocabulary_size']})")
    for prompt in data["prompts"]:
        print(f"  - {prompt}")


def report_dialog(args):
    print("\n--- Requesting caregiver report ---")
    try:
        response = requests.post(f"{args.server}/report", json={"minutes": 10, "language": "en"}, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"report failed: {exc}")
        return
    data = response.json()
    print(f"Report ({data['events_analyzed']} events, source: {data['source']}):")
    print(f"  {data['report']}")


def draw_overlay(frame, prediction, status):
    # dark banner across the top so the text stays readable
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(frame.shape[1], 640), 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "VPOCLIP client  [a]dd  [r]eport  [q]uit", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if prediction is None:
        cv2.putText(frame, status, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
        return
    cv2.putText(
        frame,
        f"{prediction['action']}  ({prediction['confidence']:.2f})",
        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 255, 120), 2, cv2.LINE_AA,
    )
    for rank, entry in enumerate(prediction.get("topk", [])[1:3], start=2):
        cv2.putText(
            frame,
            f"{rank}. {entry['action']} ({entry['confidence']:.2f})",
            (12, 62 + (rank - 1) * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"server {prediction['latency_ms']:.0f} ms  rtt {prediction.get('rtt_ms', 0):.0f} ms  [{status}]",
        (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1, cv2.LINE_AA,
    )


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera_index}")

    frame_buffer = deque()
    buffer_lock = threading.Lock()
    worker = RecognizeWorker(args, frame_buffer, buffer_lock)
    worker.start()
    print(f"Streaming to {args.server} (q or ESC to quit)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam")
            now = time.monotonic()
            with buffer_lock:
                frame_buffer.append((now, frame.copy()))
                # drop frames older than the clip window (plus a second of slack)
                while frame_buffer and now - frame_buffer[0][0] > args.window + 1.0:
                    frame_buffer.popleft()

            draw_overlay(frame, worker.prediction, worker.status)
            cv2.imshow("VPOCLIP client", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                add_action_dialog(args)
            elif key == ord("r"):
                report_dialog(args)
    finally:
        worker.running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
