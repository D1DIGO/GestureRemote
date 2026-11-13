"""Bounded FPS/latency summaries and optional per-frame CSV (no images)."""

# Keep a bounded history of recent processing timestamps.
from collections import deque
# Read or write tabular performance measurements.
import csv
# Handle filesystem paths without platform-specific separators.
from pathlib import Path
# Import fmean from statistics for use in this module.
from statistics import fmean


# Keep lightweight session counts and bounded rolling timing samples.
class PerformanceMonitor:
    # Choose how many recent frames contribute to rolling statistics.
    def __init__(self, window_size: int = 120):
        # Discard old completion timestamps automatically when the window fills.
        self.timestamps: deque[float] = deque(maxlen=window_size)
        # Keep processing durations in a matching bounded sample buffer.
        self.inference_ms: deque[float] = deque(maxlen=window_size)
        # Initialize the total number of processed frames.
        self.frames = 0
        # Count frames without a detected hand separately from total frames.
        self.no_hand_frames = 0

    # Record one completed frame and its recognition measurements.
    def record(self, timestamp: float, inference_ms: float, hand_detected: bool) -> None:
        # Use completion times so command delays affect measured FPS.
        self.timestamps.append(timestamp)
        # Store preprocessing-plus-recognition duration in milliseconds.
        self.inference_ms.append(inference_ms)
        # Increment the session's frame count.
        self.frames += 1
        # Add one only when no hand was detected in this observation.
        self.no_hand_frames += int(not hand_detected)

    # Expose this computed value through normal attribute access.
    @property
    # Expose recent completed-frame throughput as a computed property.
    def fps(self) -> float:
        # Avoid division by zero before enough distinct samples exist.
        if len(self.timestamps) < 2 or self.timestamps[-1] <= self.timestamps[0]:
            # Report zero when FPS cannot yet be calculated.
            return 0.0
        # Divide frame intervals by elapsed time, not the raw number of samples.
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])

    # Expose this computed value through normal attribute access.
    @property
    # Expose the arithmetic mean processing duration of recent frames.
    def mean_inference_ms(self) -> float:
        # Return zero before the first timing sample arrives.
        return fmean(self.inference_ms) if self.inference_ms else 0.0


# Write per-frame metadata to CSV without storing camera images.
class MetricsRecorder:
    # Define timestamp, recognition, control-state, and command columns.
    FIELDS = ["timestamp", "gesture", "confidence", "hand_detected", "active", "event", "action",
              # Add outcome, latency, FPS, and runtime-mode columns.
              "command_success", "command_ms", "inference_ms", "frame_age_ms", "fps", "preprocessing", "dry_run"]

    # Create a new trial file and its CSV writer.
    def __init__(self, path: Path):
        # Create the trial directory when necessary.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of earlier trials.
        # Use exclusive creation so an existing trial cannot be overwritten.
        self._file = path.open("x", newline="", encoding="utf-8")
        # Map each row dictionary to the declared column order.
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        # Write column names before any measurements.
        self._writer.writeheader()
        # Initialize the row counter used for periodic flushing.
        self._rows = 0

    # Append one measurement row supplied as keyword arguments.
    def write(self, **row) -> None:
        # Serialize the values as a CSV record.
        self._writer.writerow(row)
        # Track how many data rows have been written.
        self._rows += 1
        # Flush important state events immediately and ordinary rows every 30 frames.
        if row.get("event") or self._rows % 30 == 0:
            # Push Python's buffered text to the file without closing the recording.
            self._file.flush()

    # Finish the trial and release its file handle.
    def close(self) -> None:
        # Flush remaining buffered records when closing the file.
        self._file.close()
