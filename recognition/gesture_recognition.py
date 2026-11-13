"""Official MediaPipe Gesture Recognizer in synchronous VIDEO mode on CPU."""

# Validate finite numbers before using them in timing or volume calculations.
import math
# Handle filesystem paths without platform-specific separators.
from pathlib import Path

# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Access MediaPipe image and recognition APIs.
import mediapipe as mp
# Work with image arrays and numerical lookup tables.
import numpy as np
# Access the MediaPipe task interfaces used by the recognizer.
from mediapipe.tasks import python
# Configure MediaPipe tasks and CPU inference options.
from mediapipe.tasks.python import vision

# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation
# Load validated configuration types, defaults, or configuration helpers.
from settings import RecognitionConfig


# Adapt MediaPipe's native results to the application's observation type.
class GestureRecognizer:
    # Load the official model and configure CPU video inference.
    def __init__(self, config: RecognitionConfig):
        # Check for the model before asking the native SDK to open it.
        if not Path(config.model_path).is_file():
            # Point the user to the downloader when the asset is missing.
            raise FileNotFoundError(f"Model missing: {config.model_path}. Run: python scripts/download_model.py")
        # Build the options consumed by MediaPipe's Gesture Recognizer.
        options = vision.GestureRecognizerOptions(
            # Select the local model file and explicitly request CPU execution.
            base_options=python.BaseOptions(model_asset_path=config.model_path, delegate=python.BaseOptions.Delegate.CPU),
            # Use VIDEO mode to preserve temporal tracking between frames.
            running_mode=vision.RunningMode.VIDEO,
            # Detect two to reject ambiguity rather than switch between two users.
            # Detect up to two hands so ambiguous multi-hand input can be rejected.
            num_hands=2,
            # Set the minimum confidence for detecting a hand.
            min_hand_detection_confidence=config.detection_confidence,
            # Set the minimum confidence that landmarks represent a present hand.
            min_hand_presence_confidence=config.presence_confidence,
            # Set the minimum tracking confidence between frames.
            min_tracking_confidence=config.tracking_confidence,
        )
        # Allocate the native recognizer with these options.
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        # Initialize the last SDK timestamp below the first valid millisecond.
        self._last_ms = -1

    # Convert one BGR frame and return a timestamped gesture observation.
    def recognize(self, frame: np.ndarray, timestamp: float) -> GestureObservation:
        # Reject invalid or negative capture times.
        if not math.isfinite(timestamp) or timestamp < 0:
            # Explain the required timestamp domain to the caller.
            raise ValueError("Frame timestamp must be finite and nonnegative")
        # Millisecond rounding must not produce duplicate MediaPipe timestamps.
        # Keep SDK millisecond timestamps increasing even after rounding.
        timestamp_ms = max(self._last_ms + 1, int(timestamp * 1000))
        # Convert OpenCV's BGR channel order to MediaPipe's RGB order.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap a contiguous RGB buffer as a MediaPipe image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # Run synchronous recognition on this video frame.
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        # Remember the timestamp accepted by the native recognizer.
        self._last_ms = timestamp_ms
        # Count detected hands using the returned landmark groups.
        hands = len(result.hand_landmarks)
        # Reject missing categories and frames containing zero or multiple hands.
        if hands != 1 or not result.gestures or not result.gestures[0]:
            # Distinguish no hand from an ambiguous hand without accepting a command pose.
            return GestureObservation(None, 0.0, timestamp, hand_detected=hands > 0)
        # Select the highest-scoring category for the single detected hand.
        category = max(result.gestures[0], key=lambda item: item.score)
        # Read the official category name, such as Thumb_Up or Victory.
        name = category.category_name
        # Normalize unknown labels while preserving confidence and capture time.
        return GestureObservation(None if name in (None, "None", "Unknown") else name, float(category.score), timestamp)

    # Release model resources held by the native SDK.
    def close(self) -> None:
        # Close the underlying MediaPipe task.
        self._recognizer.close()

    # Allow the recognizer to be managed by a with block.
    def __enter__(self):
        # Expose this already-initialized recognizer to the caller.
        return self

    # Release native resources when the with block exits.
    def __exit__(self, *args) -> None:
        # Use the same cleanup path for success and exceptions.
        self.close()
