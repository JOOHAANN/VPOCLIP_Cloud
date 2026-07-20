# Small in-memory log of the actions we recognized recently. /report reads it.

import threading
import time
from collections import deque


class ActionLog:
    def __init__(self, maxlen=500):
        self.entries = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def add(self, action, confidence):
        entry = {"timestamp": time.time(), "action": action, "confidence": float(confidence)}
        with self.lock:
            self.entries.append(entry)

    def recent(self, minutes):
        cutoff = time.time() - minutes * 60
        with self.lock:
            return [dict(e) for e in self.entries if e["timestamp"] >= cutoff]

    def __len__(self):
        with self.lock:
            return len(self.entries)
