# Handle filesystem paths without platform-specific separators.
from pathlib import Path
# Build lightweight objects that mimic recognition results in tests.
from types import SimpleNamespace
# Replace hardware or network dependencies with controllable test doubles.
from unittest.mock import Mock

# Work with image arrays and numerical lookup tables.
import numpy as np
# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Convert camera images into MediaPipe gesture observations.
from recognition.gesture_recognition import GestureRecognizer
# Load validated configuration types, defaults, or configuration helpers.
from settings import RecognitionConfig


# Provide reusable test setup with a fresh instance for each test.
@pytest.fixture
# Build the real adapter around a fake native recognizer.
def recognizer(monkeypatch, tmp_path):
    # Choose an isolated model path for the adapter's existence check.
    path = tmp_path / "model.task"
    # Create an empty placeholder because native model loading is mocked here.
    path.touch()
    # Track calls to the native recognition and cleanup methods.
    native = Mock()
    # Replace MediaPipe task creation so this unit test needs no real model.
    monkeypatch.setattr("recognition.gesture_recognition.vision.GestureRecognizer.create_from_options", lambda options: native)
    # Return both the adapter under test and its inspectable native mock.
    return GestureRecognizer(RecognitionConfig(model_path=str(path))), native


# Test: missing model has download instructions.
def test_missing_model_has_download_instructions(tmp_path):
    # Require the missing-file error to mention how to download the model.
    with pytest.raises(FileNotFoundError, match="download_model.py"):
        # Attempt construction with a path that deliberately does not exist.
        GestureRecognizer(RecognitionConfig(model_path=str(tmp_path / "absent.task")))


# Test: returns category confidence and timestamp and rgb.
def test_returns_category_confidence_and_timestamp_and_rgb(recognizer):
    # Unpack the adapter and its fake native backend.
    wrapper, native = recognizer
    # Prepare a native-shaped result with a known category and score.
    native.recognize_for_video.return_value = SimpleNamespace(
        # Represent exactly one detected hand classified as Thumb_Up.
        hand_landmarks=[[1]], gestures=[[SimpleNamespace(category_name="Thumb_Up", score=0.93)]])
    # Use distinct BGR channel values so incorrect channel order is visible.
    frame = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    # Recognize one frame at a known fractional-second timestamp.
    event = wrapper.recognize(frame, 1.0001)
    # Require the adapter to preserve category, score, and capture time.
    assert (event.gesture, event.confidence, event.timestamp) == ("Thumb_Up", 0.93, 1.0001)
    # Inspect the actual image and SDK timestamp passed into MediaPipe.
    image, timestamp = native.recognize_for_video.call_args.args
    # The RGB buffer must reverse the original blue and red channel values.
    assert image.numpy_view()[0, 0].tolist() == [30, 20, 10]
    # Convert seconds to the first integer-millisecond SDK timestamp.
    assert timestamp == 1000
    # Submit another frame whose time rounds to the same millisecond.
    wrapper.recognize(frame, 1.0002)
    # The adapter must bump the SDK timestamp instead of submitting a duplicate.
    assert native.recognize_for_video.call_args.args[1] == 1001
    # Explicitly close the adapter to test native resource cleanup.
    wrapper.close()
    # The underlying recognizer must be closed exactly once.
    native.close.assert_called_once()


# Cover no hand, two-hand ambiguity, and a single unclassified hand.
@pytest.mark.parametrize("hands,name,expected_hand", [(0, None, False), (2, "Victory", True), (1, "None", True)])
# Test: no hand unknown or ambiguous results.
def test_no_hand_unknown_or_ambiguous_results(recognizer, hands, name, expected_hand):
    # Unpack the adapter and its native mock for this scenario.
    wrapper, native = recognizer
    # Prepare the result shape expected from the selected hand-count case.
    native.recognize_for_video.return_value = SimpleNamespace(
        # Supply one landmark group per reported hand.
        hand_landmarks=[[1]] * hands,
        # Supply matching category lists, including unknown labels when requested.
        gestures=[[SimpleNamespace(category_name=name, score=0.99)]] * hands)
    # Run the adapter on a small synthetic image.
    observation = wrapper.recognize(np.zeros((8, 8, 3), np.uint8), 1)
    # None of these cases should produce an accepted named gesture.
    assert observation.gesture is None
    # Preserve the distinction between no hand and an ambiguous visible hand.
    assert observation.hand_detected is expected_hand


# Mark the smoke test that actually loads the downloaded CPU model.
@pytest.mark.model
# Test: real cpu model on blank video frames.
def test_real_cpu_model_on_blank_video_frames():
    # Locate the official model relative to this test's project directory.
    path = Path(__file__).resolve().parents[1] / "models/gesture_recognizer.task"
    # A fresh checkout may not have downloaded the binary model yet.
    if not path.exists():
        # Report a skip explicitly instead of pretending real inference was tested.
        pytest.skip("Download the official model to run the real CPU smoke test")
    # Create and clean up a real CPU recognizer around the smoke test.
    with GestureRecognizer(RecognitionConfig(model_path=str(path))) as recognizer:
        # Exercise increasing timestamps across several blank video frames.
        for timestamp in (0.0, 0.033, 0.067):
            # Use a black 640x480 frame to check the native inference pipeline.
            result = recognizer.recognize(np.zeros((480, 640, 3), np.uint8), timestamp)
            # A blank image should not produce a gesture category.
            assert result.gesture is None
            # A blank image should not report a detected hand.
            assert not result.hand_detected
