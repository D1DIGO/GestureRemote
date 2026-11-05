"""Bounded FPS/latency summaries and optional per-frame CSV (no images)."""

from collections import deque
import csv
from pathlib import Path
from statistics import fmean


class PerformanceMonitor:
    def __init__(self, window_size: int = 120):
        self.timestamps: deque[float] = deque(maxlen=window_size)
        self.inference_ms: deque[float] = deque(maxlen=window_size)
        self.frames = 0
        self.no_hand_frames = 0

    def record(self, timestamp: float, inference_ms: float, hand_detected: bool) -> None:
        self.timestamps.append(timestamp)
        self.inference_ms.append(inference_ms)
        self.frames += 1
        self.no_hand_frames += int(not hand_detected)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2 or self.timestamps[-1] <= self.timestamps[0]:
            return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])

    @property
    def mean_inference_ms(self) -> float:
        return fmean(self.inference_ms) if self.inference_ms else 0.0


class MetricsRecorder:
    FIELDS = ["timestamp", "gesture", "confidence", "hand_detected", "active", "event", "action",
              "command_success", "command_ms", "inference_ms", "frame_age_ms", "fps", "preprocessing", "dry_run"]

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of earlier trials.
        self._file = path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._rows = 0

    def write(self, **row) -> None:
        self._writer.writerow(row)
        self._rows += 1
        if row.get("event") or self._rows % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.close()
