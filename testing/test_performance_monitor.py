# Read or write tabular performance measurements.
import csv

# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Measure processing performance and record optional CSV metrics.
from testing.performance_monitor import MetricsRecorder, PerformanceMonitor


# Test: fps includes time between processed frames.
def test_fps_includes_time_between_processed_frames():
    # Use a small rolling window so eviction behavior is visible in four samples.
    monitor = PerformanceMonitor(window_size=3)
    # FPS is unknown before there are at least two distinct timestamps.
    assert monitor.fps == 0
    # Simulate four frames completed 100ms apart.
    for t in [0, 0.1, 0.2, 0.3]:
        # Record 10ms processing time and no detected hand for each frame.
        monitor.record(t, 10, False)
    # The rolling rate should be ten completed frames per second.
    assert monitor.fps == pytest.approx(10)
    # The average processing duration should remain ten milliseconds.
    assert monitor.mean_inference_ms == 10
    # Only the three newest timestamp samples should be retained.
    assert len(monitor.timestamps) == 3
    # Session totals must still include samples evicted from the rolling window.
    assert monitor.frames == monitor.no_hand_frames == 4


# Test: metrics do not overwrite an existing trial.
def test_metrics_do_not_overwrite_an_existing_trial(tmp_path):
    # Choose a temporary CSV destination for a single trial.
    path = tmp_path / "trial.csv"
    # Create a new recorder and its header.
    recorder = MetricsRecorder(path)
    # Write one activation event to make the file content recognizable.
    recorder.write(timestamp=1, event="activated")
    # Flush and close the first recorder before testing replacement protection.
    recorder.close()
    # Opening the same trial path again must fail rather than truncate it.
    with pytest.raises(FileExistsError):
        # Attempt to create a second recorder at the existing path.
        MetricsRecorder(path)
    # Reopen the original file for verification.
    with path.open() as file:
        # Confirm the first trial's activation record survived unchanged.
        assert list(csv.DictReader(file))[0]["event"] == "activated"
