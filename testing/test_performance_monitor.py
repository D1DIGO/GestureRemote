import csv

import pytest

from testing.performance_monitor import MetricsRecorder, PerformanceMonitor


def test_fps_includes_time_between_processed_frames():
    monitor = PerformanceMonitor(window_size=3)
    assert monitor.fps == 0
    for t in [0, 0.1, 0.2, 0.3]:
        monitor.record(t, 10, False)
    assert monitor.fps == pytest.approx(10)
    assert monitor.mean_inference_ms == 10
    assert len(monitor.timestamps) == 3
    assert monitor.frames == monitor.no_hand_frames == 4


def test_metrics_do_not_overwrite_an_existing_trial(tmp_path):
    path = tmp_path / "trial.csv"
    recorder = MetricsRecorder(path)
    recorder.write(timestamp=1, event="activated")
    recorder.close()
    with pytest.raises(FileExistsError):
        MetricsRecorder(path)
    with path.open() as file:
        assert list(csv.DictReader(file))[0]["event"] == "activated"
