from dataclasses import dataclass


@dataclass(frozen=True)
class GestureObservation:
    gesture: str | None
    confidence: float
    timestamp: float  # monotonic seconds, not wall clock
    hand_detected: bool = True
