# Define structured records or create modified copies of configuration records.
from dataclasses import replace
# Replace hardware or network dependencies with controllable test doubles.
from unittest.mock import Mock
# Coordinate background capture with the foreground processing loop.
import threading

# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Work with image arrays and numerical lookup tables.
import numpy as np
# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Access the latest-frame camera wrapper and its error types.
from camera.camera import Camera, CameraError
# Apply the optional image-enhancement pipeline.
from camera.image_processing import ImageProcessor
# Load validated configuration types, defaults, or configuration helpers.
from settings import CameraConfig, PreprocessingConfig


# Test: disabled processing returns original and never modifies it.
def test_disabled_processing_returns_original_and_never_modifies_it():
    # Build a small dark BGR frame with known pixel values.
    frame = np.full((24, 32, 3), 10, dtype=np.uint8)
    # Configure an enhancement but leave the processing switch disabled.
    processor = ImageProcessor(PreprocessingConfig(brightness=50))
    # Disabled processing must return the exact original array object.
    assert processor.process(frame) is frame
    # Comparison-only forcing should apply the configured brightness offset.
    assert np.all(processor.process(frame, force=True) == 60)
    # Neither call should change the input array's pixel values.
    assert np.all(frame == 10)


# Test: negative brightness clips instead of reflecting.
def test_negative_brightness_clips_instead_of_reflecting():
    # Enable an offset that would push every input value below zero.
    processor = ImageProcessor(PreprocessingConfig(enabled=True, brightness=-30))
    # Negative output must clamp to black rather than reflect to positive values.
    assert not processor.process(np.full((10, 10, 3), 20, dtype=np.uint8)).any()


# Test: gamma brightens and preserves image format.
def test_gamma_brightens_and_preserves_image_format():
    # Create a dark, uniform image to make gamma's effect easy to measure.
    frame = np.full((32, 32, 3), 50, dtype=np.uint8)
    # Apply a brightening gamma value with processing enabled.
    result = ImageProcessor(PreprocessingConfig(enabled=True, gamma=2)).process(frame)
    # Enhancement must preserve image dimensions and channel count.
    assert result.shape == frame.shape
    # Keep the byte-pixel dtype expected by OpenCV and MediaPipe.
    assert result.dtype == np.uint8
    # Gamma above one should increase the mean intensity for this dark image.
    assert result.mean() > frame.mean()


# Test: clahe and denoise keep shape and input.
def test_clahe_and_denoise_keep_shape_and_input():
    # Use a fixed random seed for a reproducible noisy test frame.
    frame = np.random.default_rng(0).integers(0, 100, (32, 48, 3), dtype=np.uint8)
    # Keep a copy to detect accidental mutation of the original image.
    original = frame.copy()
    # Exercise Gaussian smoothing and luminance CLAHE together.
    result = ImageProcessor(PreprocessingConfig(enabled=True, clahe=True, denoise=True)).process(frame)
    # Both filters must preserve image dimensions.
    assert result.shape == frame.shape
    # Both filters must return byte-valued pixels.
    assert result.dtype == np.uint8
    # Compare every original pixel to verify processing did not mutate the input.
    np.testing.assert_array_equal(frame, original)


# Test: missing camera releases all failed backends.
def test_missing_camera_releases_all_failed_backends(monkeypatch):
    # Create a fake capture device without accessing camera hardware.
    capture = Mock()
    # Simulate a native handle that never opens successfully.
    capture.isOpened.return_value = False
    # Return that failed handle each time a backend is tried.
    factory = Mock(return_value=capture)
    # Intercept hardware capture construction.
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    # Force Windows backend selection even when pytest runs on Linux.
    monkeypatch.setattr("camera.camera.platform.system", lambda: "Windows")
    # Require a descriptive camera-open failure.
    with pytest.raises(CameraError, match="Cannot open camera"):
        # Trigger the real initialization and automatic backend fallback.
        with Camera(CameraConfig()):
            # The body should not execute because context entry must fail.
            pass
    # Require DirectShow first, followed by the default backend.
    assert [call.args[1] for call in factory.call_args_list] == [cv2.CAP_DSHOW, cv2.CAP_ANY]
    # Each failed handle must be released before proceeding.
    assert capture.release.call_count == 2


# Test: capture read failure and cleanup.
def test_capture_read_failure_and_cleanup(monkeypatch):
    # Create a fake capture device for the frame-reading scenario.
    capture = Mock()
    # Let the fake native device open successfully.
    capture.isOpened.return_value = True
    # Provide numeric driver properties for startup logging.
    capture.get.return_value = 30
    # Simulate a failed capture read with no image buffer.
    capture.read.return_value = (False, None)
    # Use the fake handle instead of opening a physical webcam.
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    # Exercise the actual capture thread and context-manager cleanup.
    with Camera(CameraConfig(backend="any")) as camera:
        # Expect the worker error to reach the foreground reader.
        with pytest.raises(CameraError, match="stopped delivering"):
            # Wait for the worker's frame or failure notification.
            camera.read()
    # The capture handle must be released exactly once.
    capture.release.assert_called_once()


# Test: newest frame replaces old frames.
def test_newest_frame_replaces_old_frames(monkeypatch):
    # Create a fake capture device for the shutdown scenario.
    capture = Mock()
    # Allow the fake camera to initialize normally.
    capture.isOpened.return_value = True
    # Return numeric driver properties for the startup log.
    capture.get.return_value = 30
    # Signal when the worker has published all three test frames.
    ready = threading.Event()
    # Keep the fake reader blocked until the test explicitly releases it.
    finish = threading.Event()
    # Create three distinct image objects to identify which frame is returned.
    frames = [np.full((8, 8, 3), n, np.uint8) for n in range(3)]
    # Start the fake capture sequence at its first frame.
    index = 0

    # Implement a finite burst followed by a controllable blocked read.
    def read():
        # Update the frame index stored in the enclosing test scope.
        nonlocal index
        # Deliver each of the three distinct images once.
        if index < 3:
            # Choose the next image from the test burst.
            frame = frames[index]
            # Advance the fake device to the following frame.
            index += 1
            # Report a successful native read while burst images remain.
            return True, frame
        # Tell the foreground test that the newest frame is already available.
        ready.set()  # all three frames have already been published
        # Wait for cleanup permission, bounded to avoid hanging the test suite.
        finish.wait(2)
        # End the fake capture stream after the blocking interval.
        return False, None

    # Have each mock read execute the controlled burst function.
    capture.read.side_effect = read
    # Redirect OpenCV camera creation to the fake capture handle.
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    # Choose an explicit backend to keep this test platform-neutral.
    camera = Camera(CameraConfig(backend="any"))
    # Run the real camera object's lifecycle around the threaded test.
    with camera:
        # Handle failures from the operations in this block.
        try:
            # Wait until all burst frames have been published.
            assert ready.wait(2)
            # Read the camera's single retained frame.
            latest = camera.read()
            # Require the newest frame rather than the first item of a backlog.
            assert latest.image is frames[2]
            # Shorten the reader timeout for the no-new-frame check.
            camera.config = replace(camera.config, read_timeout=0.01)
            # Expect a timeout when only the already-processed frame exists.
            with pytest.raises(CameraError, match="Timed out"):
                # Reject the retained frame by passing its timestamp as the lower bound.
                camera.read(after=latest.timestamp)
        # Run cleanup whether the preceding operations succeeded or failed.
        finally:
            # Always unblock the worker, even if an assertion failed above.
            finish.set()
    # Verify the native capture handle is released only once.
    capture.release.assert_called_once()
