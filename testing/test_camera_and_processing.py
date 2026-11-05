from dataclasses import replace
from unittest.mock import Mock
import threading

import cv2
import numpy as np
import pytest

from camera.camera import Camera, CameraError
from camera.image_processing import ImageProcessor
from settings import CameraConfig, PreprocessingConfig


def test_disabled_processing_returns_original_and_never_modifies_it():
    frame = np.full((24, 32, 3), 10, dtype=np.uint8)
    processor = ImageProcessor(PreprocessingConfig(brightness=50))
    assert processor.process(frame) is frame
    assert np.all(processor.process(frame, force=True) == 60)
    assert np.all(frame == 10)


def test_negative_brightness_clips_instead_of_reflecting():
    processor = ImageProcessor(PreprocessingConfig(enabled=True, brightness=-30))
    assert not processor.process(np.full((10, 10, 3), 20, dtype=np.uint8)).any()


def test_gamma_brightens_and_preserves_image_format():
    frame = np.full((32, 32, 3), 50, dtype=np.uint8)
    result = ImageProcessor(PreprocessingConfig(enabled=True, gamma=2)).process(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert result.mean() > frame.mean()


def test_clahe_and_denoise_keep_shape_and_input():
    frame = np.random.default_rng(0).integers(0, 100, (32, 48, 3), dtype=np.uint8)
    original = frame.copy()
    result = ImageProcessor(PreprocessingConfig(enabled=True, clahe=True, denoise=True)).process(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(frame, original)


def test_missing_camera_releases_all_failed_backends(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = False
    factory = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    monkeypatch.setattr("camera.camera.platform.system", lambda: "Windows")
    with pytest.raises(CameraError, match="Cannot open camera"):
        with Camera(CameraConfig()):
            pass
    assert [call.args[1] for call in factory.call_args_list] == [cv2.CAP_DSHOW, cv2.CAP_ANY]
    assert capture.release.call_count == 2


def test_capture_read_failure_and_cleanup(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 30
    capture.read.return_value = (False, None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    with Camera(CameraConfig(backend="any")) as camera:
        with pytest.raises(CameraError, match="stopped delivering"):
            camera.read()
    capture.release.assert_called_once()


def test_newest_frame_replaces_old_frames(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 30
    ready = threading.Event()
    finish = threading.Event()
    frames = [np.full((8, 8, 3), n, np.uint8) for n in range(3)]
    index = 0

    def read():
        nonlocal index
        if index < 3:
            frame = frames[index]
            index += 1
            return True, frame
        ready.set()  # all three frames have already been published
        finish.wait(2)
        return False, None

    capture.read.side_effect = read
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    camera = Camera(CameraConfig(backend="any"))
    with camera:
        try:
            assert ready.wait(2)
            latest = camera.read()
            assert latest.image is frames[2]
            camera.config = replace(camera.config, read_timeout=0.01)
            with pytest.raises(CameraError, match="Timed out"):
                camera.read(after=latest.timestamp)
        finally:
            finish.set()
    capture.release.assert_called_once()
