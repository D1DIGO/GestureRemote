from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from recognition.gesture_recognition import GestureRecognizer
from settings import RecognitionConfig


@pytest.fixture
def recognizer(monkeypatch, tmp_path):
    path = tmp_path / "model.task"
    path.touch()
    native = Mock()
    monkeypatch.setattr("recognition.gesture_recognition.vision.GestureRecognizer.create_from_options", lambda options: native)
    return GestureRecognizer(RecognitionConfig(model_path=str(path))), native


def test_missing_model_has_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        GestureRecognizer(RecognitionConfig(model_path=str(tmp_path / "absent.task")))


def test_returns_category_confidence_and_timestamp_and_rgb(recognizer):
    wrapper, native = recognizer
    native.recognize_for_video.return_value = SimpleNamespace(
        hand_landmarks=[[1]], gestures=[[SimpleNamespace(category_name="Thumb_Up", score=0.93)]])
    frame = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    event = wrapper.recognize(frame, 1.0001)
    assert (event.gesture, event.confidence, event.timestamp) == ("Thumb_Up", 0.93, 1.0001)
    image, timestamp = native.recognize_for_video.call_args.args
    assert image.numpy_view()[0, 0].tolist() == [30, 20, 10]
    assert timestamp == 1000
    wrapper.recognize(frame, 1.0002)
    assert native.recognize_for_video.call_args.args[1] == 1001
    wrapper.close()
    native.close.assert_called_once()


@pytest.mark.parametrize("hands,name,expected_hand", [(0, None, False), (2, "Victory", True), (1, "None", True)])
def test_no_hand_unknown_or_ambiguous_results(recognizer, hands, name, expected_hand):
    wrapper, native = recognizer
    native.recognize_for_video.return_value = SimpleNamespace(
        hand_landmarks=[[1]] * hands,
        gestures=[[SimpleNamespace(category_name=name, score=0.99)]] * hands)
    observation = wrapper.recognize(np.zeros((8, 8, 3), np.uint8), 1)
    assert observation.gesture is None
    assert observation.hand_detected is expected_hand


@pytest.mark.model
def test_real_cpu_model_on_blank_video_frames():
    path = Path(__file__).resolve().parents[1] / "models/gesture_recognizer.task"
    if not path.exists():
        pytest.skip("Download the official model to run the real CPU smoke test")
    with GestureRecognizer(RecognitionConfig(model_path=str(path))) as recognizer:
        for timestamp in (0.0, 0.033, 0.067):
            result = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), timestamp)
            assert result.gesture is None
            assert not result.hand_detected
