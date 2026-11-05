from unittest.mock import Mock
import cv2
import numpy as np

from preview import Preview
from recognition.observation import GestureObservation


def test_closing_native_window_is_a_clean_exit(monkeypatch):
    monkeypatch.setattr(cv2, "imshow", Mock())
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", Mock(side_effect=cv2.error("NULL guiReceiver")))
    preview = Preview.__new__(Preview)
    assert not preview.show(np.zeros((20, 20, 3), np.uint8), None, GestureObservation(None, 0, 1, False),
                            active=False, last_action="none", status="VLC connected", fps=10,
                            preprocessing=False, dry_run=False, feedback="OFF - hold Victory to enable")
