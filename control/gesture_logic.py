"""Elapsed-time state machine; no camera, HTTP, or native dependencies."""

from dataclasses import dataclass
import math

from recognition.observation import GestureObservation
from settings import GestureConfig


@dataclass(frozen=True)
class Decision:
    kind: str  # activated, deactivated, timeout, command
    timestamp: float
    action: str | None = None


class GestureStateMachine:
    """A confirmed hold fires once, then remains latched until a stable release.

    Unknown/low-confidence hands cancel confirmation but do not count as a
    release. A no-hand interval or a confident different gesture does. Gaps
    in observation reset timing, never serve as proof of a release.
    """

    def __init__(self, config: GestureConfig):
        self.config = config
        self.active = False
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._latched: str | None = None
        self._release_since: float | None = None
        self._last_timestamp: float | None = None
        self._last_activity = 0.0
        self._last_actions: dict[str, float] = {}

    def deactivate(self) -> None:
        """Fail closed after an unsuccessful command; preserve the hold latch."""
        self.active = False
        self._candidate = None
        self._release_since = None

    def update(self, observation: GestureObservation) -> Decision | None:
        t = observation.timestamp
        if not math.isfinite(t) or (self._last_timestamp is not None and t <= self._last_timestamp):
            raise ValueError("Observation timestamps must be finite and strictly increasing")
        if self._last_timestamp is not None and t - self._last_timestamp > self.config.max_frame_gap:
            self._candidate = None
            self._release_since = None
        self._last_timestamp = t

        if self.active and t - self._last_activity >= self.config.inactivity_timeout:
            self.deactivate()
            return Decision("timeout", t)

        valid = (observation.hand_detected
                 and math.isfinite(observation.confidence)
                 and self.config.confidence_threshold <= observation.confidence <= 1
                 and observation.gesture in ({"Victory"} | self.config.mappings.keys()))
        gesture = observation.gesture if valid else None

        if self._latched is not None:
            released = not observation.hand_detected or (gesture is not None and gesture != self._latched)
            if released:
                if self._release_since is None:
                    self._release_since = t
                if t - self._release_since >= self.config.release_duration:
                    self._latched = None
                    self._release_since = None
            else:
                self._release_since = None

        if gesture is None:
            self._candidate = None
            return None
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_since = t
        if self._latched is not None:
            return None

        duration = self.config.activation_duration if gesture == "Victory" else self.config.confirmation_duration
        if t - self._candidate_since < duration:
            return None
        if gesture == "Victory":
            self._latched = gesture
            self.active = not self.active
            self._last_activity = t
            return Decision("activated" if self.active else "deactivated", t)
        if not self.active:
            return None
        action = self.config.mappings[gesture]
        if t - self._last_actions.get(action, -math.inf) < self.config.cooldown:
            return None
        self._latched = gesture
        self._last_actions[action] = t
        self._last_activity = t
        return Decision("command", t, action)
