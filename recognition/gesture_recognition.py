"""Official MediaPipe Gesture Recognizer in synchronous VIDEO mode on CPU."""

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from recognition.observation import GestureObservation
from settings import RecognitionConfig


class GestureRecognizer:
    def __init__(self, config: RecognitionConfig):
        if not Path(config.model_path).is_file():
            raise FileNotFoundError(f"Model missing: {config.model_path}. Run: python scripts/download_model.py")
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=config.model_path, delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            # Detect two to reject ambiguity rather than switch between two users.
            num_hands=2,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1

    def recognize(self, frame: np.ndarray, timestamp: float) -> GestureObservation:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Frame timestamp must be finite and nonnegative")
        # Millisecond rounding must not produce duplicate MediaPipe timestamps.
        timestamp_ms = max(self._last_ms + 1, int(timestamp * 1000))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        self._last_ms = timestamp_ms
        hands = len(result.hand_landmarks)
        if hands != 1 or not result.gestures or not result.gestures[0]:
            return GestureObservation(None, 0.0, timestamp, hand_detected=hands > 0)
        category = max(result.gestures[0], key=lambda item: item.score)
        name = category.category_name
        return GestureObservation(None if name in (None, "None", "Unknown") else name, float(category.score), timestamp)

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
