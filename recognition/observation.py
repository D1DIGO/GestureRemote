# Define structured records or create modified copies of configuration records.
from dataclasses import dataclass


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Represent one recognition result independently of OpenCV and MediaPipe.
class GestureObservation:
    # Use None when no supported pose can be identified.
    gesture: str | None
    # Keep the classifier score separate from command acceptance.
    confidence: float
    # Use monotonic capture time for hold durations and cooldowns.
    timestamp: float  # monotonic seconds, not wall clock
    # Distinguish an uncertain visible hand from a frame with no detected hand.
    hand_detected: bool = True
