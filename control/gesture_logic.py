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
    """Confirmed toggles fire once; configured volume/seek actions repeat.

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
        self.feedback = "OFF - hold Victory to enable"

    def deactivate(self) -> None:
        """Fail closed after an unsuccessful command; preserve the hold latch."""
        self.active = False
        self._candidate = None
        self._release_since = None
        self.feedback = "OFF - release, then hold Victory to enable"

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

        threshold = self.config.confidence_thresholds.get(observation.gesture, self.config.confidence_threshold)
        valid = (observation.hand_detected
                 and math.isfinite(observation.confidence)
                 and threshold <= observation.confidence <= 1
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
            if not self.active:
                self.feedback = "OFF - hold Victory to enable"
            elif observation.gesture and observation.confidence < threshold:
                self.feedback = f"Low confidence: {observation.confidence:.2f} < {threshold:.2f}"
            else:
                self.feedback = "Show one clear hand; waiting for a recognized pose"
            return None
        if gesture != self._candidate:
            self._candidate = gesture
            self._candidate_since = t
        action = self.config.mappings.get(gesture)
        repeating = self.active and self._latched == gesture and action in self.config.repeat_actions
        if self._latched is not None and not repeating:
            self.feedback = "Already sent - release or change gesture" if self.active or gesture == "Victory" else "OFF - hold Victory to enable"
            return None

        duration = self.config.activation_duration if gesture == "Victory" else self.config.confirmation_durations.get(gesture, self.config.confirmation_duration)
        if not self.active and gesture != "Victory":
            self.feedback = "OFF - hold Victory to enable"
            return None
        if t - self._candidate_since < duration:
            self.feedback = f"Hold {gesture}: {t - self._candidate_since:.1f}/{duration:.1f}s"
            return None
        if gesture == "Victory":
            self._latched = gesture
            self.active = not self.active
            self._last_activity = t
            self.feedback = "ON - show a command gesture" if self.active else "OFF - release Victory before reactivating"
            return Decision("activated" if self.active else "deactivated", t)
        if not self.active:
            return None
        interval = max(self.config.cooldown, self.config.repeat_interval) if repeating else self.config.cooldown
        remaining = interval - (t - self._last_actions.get(action, -math.inf))
        if remaining > 0:
            self.feedback = f"{'Repeating' if repeating else 'Cooldown'} {action}: next in {remaining:.1f}s"
            return None
        self._latched = gesture
        self._last_actions[action] = t
        self._last_activity = t
        self.feedback = f"Sent {action}; " + ("keep holding to repeat" if action in self.config.repeat_actions else "release to use again")
        return Decision("command", t, action)
