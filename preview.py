"""Small OpenCV status window; never used in no-preview mode."""

import cv2
import numpy as np

from recognition.observation import GestureObservation

WINDOW = "GestureRemote - Q / Esc to quit"


class Preview:
    def __init__(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1100, 720)

    def show(self, original: np.ndarray, enhanced: np.ndarray | None, observation: GestureObservation,
             *, active: bool, last_action: str, status: str, fps: float,
             preprocessing: bool, dry_run: bool, feedback: str = "") -> bool:
        def fit(frame):
            return cv2.resize(frame, (640, max(1, round(frame.shape[0] * 640 / frame.shape[1]))))

        feed = fit(original)
        if enhanced is not None:
            feed = np.hstack((feed, fit(enhanced)))
        # Separate header keeps labels readable even with 320x240 capture.
        canvas = np.zeros((feed.shape[0] + 183, feed.shape[1], 3), dtype=np.uint8)
        canvas[183:] = feed
        lines = [
            f"Control: {'ON' if active else 'OFF'}   {'DRY RUN' if dry_run else 'VLC'}   FPS: {fps:.1f}",
            f"Gesture: {observation.gesture or 'None / ambiguous'}   Confidence: {observation.confidence:.2f}",
            feedback,
            f"Last command: {last_action}",
            status,
            f"Input: {'enhanced' if preprocessing else 'original'} | " + ("Left: original / Right: enhanced" if enhanced is not None else "Hold Victory to toggle control"),
        ]
        for i, line in enumerate(lines):
            color = (100, 240, 100) if i == 0 and active else (230, 230, 230)
            cv2.putText(canvas, line[:100], (12, 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            # Qt can destroy its receiver as soon as the close button is clicked.
            return False

    def close(self) -> None:
        cv2.destroyAllWindows()
