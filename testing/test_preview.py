# Replace hardware or network dependencies with controllable test doubles.
from unittest.mock import Mock
# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Work with image arrays and numerical lookup tables.
import numpy as np

# Render recognition status and control feedback in a window.
from preview import Preview
# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation


# Test: closing native window is a clean exit.
def test_closing_native_window_is_a_clean_exit(monkeypatch):
    # Intercept display calls so the test opens no real window.
    monkeypatch.setattr(cv2, "imshow", Mock())
    # Simulate no keyboard exit request.
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    # Reproduce Qt's error when a closed window has already lost its receiver.
    monkeypatch.setattr(cv2, "getWindowProperty", Mock(side_effect=cv2.error("NULL guiReceiver")))
    # Create the Python object without running the native window constructor.
    preview = Preview.__new__(Preview)
    # Require show() to return a clean stop signal instead of raising.
    assert not preview.show(np.zeros((20, 20, 3), np.uint8), None, GestureObservation(None, 0, 1, False),
                            # Provide ordinary disabled-session status for the rendered header.
                            active=False, last_action="none", status="VLC connected", fps=10,
                            # Include the processing flags and activation guidance used in the real UI.
                            preprocessing=False, dry_run=False, feedback="OFF - hold Victory to enable")
