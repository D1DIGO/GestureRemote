"""Elapsed-time state machine; no camera, HTTP, or native dependencies."""

# Define structured records or create modified copies of configuration records.
from dataclasses import dataclass
# Validate finite numbers before using them in timing or volume calculations.
import math

# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation
# Load validated configuration types, defaults, or configuration helpers.
from settings import GestureConfig


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Describe a state transition or an accepted media action.
class Decision:
    # Identify activation, deactivation, inactivity timeout, or command execution.
    kind: str  # activated, deactivated, timeout, command
    # Attach the observation time that caused this decision.
    timestamp: float
    # Only command decisions need a media-action name.
    action: str | None = None


# Decide when gestures may act, without depending on devices or HTTP.
class GestureStateMachine:
    """Confirmed toggles fire once; configured volume/seek actions repeat.

    Unknown/low-confidence hands cancel confirmation but do not count as a
    release. A no-hand interval or a confident different gesture does. Gaps
    in observation reset timing, never serve as proof of a release.
    """

    # Initialize all state for a new gesture-control session.
    def __init__(self, config: GestureConfig):
        # Retain the configured mappings, confidence limits, and timing rules.
        self.config = config
        # Require deliberate activation before any media command.
        self.active = False
        # No pose is currently accumulating confirmation time.
        self._candidate: str | None = None
        # Store when the current candidate pose first appeared.
        self._candidate_since = 0.0
        # Remember a fired pose to prevent accidental toggle repetition.
        self._latched: str | None = None
        # Track when evidence of releasing the fired pose began.
        self._release_since: float | None = None
        # No observation timestamp has been accepted yet.
        self._last_timestamp: float | None = None
        # Track the most recent activation or accepted media command.
        self._last_activity = 0.0
        # Remember execution times separately for each media action.
        self._last_actions: dict[str, float] = {}
        # Explain the initial disabled state to the preview.
        self.feedback = "OFF - hold Victory to enable"

    # Disable commands while preserving protection against a still-held pose.
    def deactivate(self) -> None:
        """Fail closed after an unsuccessful command; preserve the hold latch."""
        # Stop accepting media commands.
        self.active = False
        # Discard any partially confirmed candidate pose.
        self._candidate = None
        # Discard partial release evidence as well.
        self._release_since = None
        # Explain that a fresh deliberate activation is required.
        self.feedback = "OFF - release, then hold Victory to enable"

    # Consume one observation and emit at most one decision.
    def update(self, observation: GestureObservation) -> Decision | None:
        # Use the frame's timestamp rather than sampling another clock here.
        t = observation.timestamp
        # Reject invalid times and repeated or out-of-order observations.
        if not math.isfinite(t) or (self._last_timestamp is not None and t <= self._last_timestamp):
            # Make the timestamp contract explicit to callers and tests.
            raise ValueError("Observation timestamps must be finite and strictly increasing")
        # Do not count an unobserved time gap as continuous pose stability.
        if self._last_timestamp is not None and t - self._last_timestamp > self.config.max_frame_gap:
            # Restart confirmation after a gap that exceeds the limit.
            self._candidate = None
            # Do not treat the gap as proof that a fired pose was released.
            self._release_since = None
        # Remember this observation time for the next gap check.
        self._last_timestamp = t

        # Disable control if no accepted command has refreshed activity in time.
        if self.active and t - self._last_activity >= self.config.inactivity_timeout:
            # Reset active confirmation while retaining the fired-pose latch.
            self.deactivate()
            # Tell the integration loop why control was disabled.
            return Decision("timeout", t)

        # Use a pose-specific confidence limit when configured, otherwise the default.
        threshold = self.config.confidence_thresholds.get(observation.gesture, self.config.confidence_threshold)
        # A valid command pose requires a detected hand.
        valid = (observation.hand_detected
                 # Reject nonfinite confidence values rather than trusting their comparisons.
                 and math.isfinite(observation.confidence)
                 # Require confidence to meet the selected threshold and remain at most one.
                 and threshold <= observation.confidence <= 1
                 # Accept only Victory or a category with a configured media mapping.
                 and observation.gesture in ({"Victory"} | self.config.mappings.keys()))
        # Normalize rejected observations to no accepted pose.
        gesture = observation.gesture if valid else None

        # Check release evidence only when a fired pose is still latched.
        if self._latched is not None:
            # A missing hand or a confident different pose can establish release.
            released = not observation.hand_detected or (gesture is not None and gesture != self._latched)
            # Accumulate release time while that evidence remains present.
            if released:
                # Start timing release only on its first observed frame.
                if self._release_since is None:
                    # Record the start of the release interval.
                    self._release_since = t
                # Require the release evidence to persist long enough.
                if t - self._release_since >= self.config.release_duration:
                    # Allow a fresh command after a confirmed release.
                    self._latched = None
                    # Clear the completed release timer.
                    self._release_since = None
            # The hand no longer provides reliable release evidence.
            else:
                # Restart release timing rather than combining disconnected intervals.
                self._release_since = None

        # An unknown or rejected pose cannot confirm or repeat a command.
        if gesture is None:
            # Cancel any unfinished hold immediately.
            self._candidate = None
            # Prioritize activation guidance when control is disabled.
            if not self.active:
                # Explain that a recognized media pose cannot act while control is OFF.
                self.feedback = "OFF - hold Victory to enable"
            # Distinguish a recognized but insufficiently confident category.
            elif observation.gesture and observation.confidence < threshold:
                # Show both the observed score and its required threshold.
                self.feedback = f"Low confidence: {observation.confidence:.2f} < {threshold:.2f}"
            # Handle absent, ambiguous, or unsupported hands with a general prompt.
            else:
                # Ask for one clear recognizable hand.
                self.feedback = "Show one clear hand; waiting for a recognized pose"
            # Produce no action from a rejected observation.
            return None
        # A change of accepted pose starts a new confirmation interval.
        if gesture != self._candidate:
            # Remember the new candidate category.
            self._candidate = gesture
            # Measure its stability from this frame onward.
            self._candidate_since = t
        # Look up the media action; Victory intentionally has no media mapping.
        action = self.config.mappings.get(gesture)
        # Allow a latched pose to repeat only for enabled repeatable actions.
        repeating = self.active and self._latched == gesture and action in self.config.repeat_actions
        # Keep toggles blocked until release, even after their cooldown has elapsed.
        if self._latched is not None and not repeating:
            # Explain whether the user needs to release a pose or enable control.
            self.feedback = "Already sent - release or change gesture" if self.active or gesture == "Victory" else "OFF - hold Victory to enable"
            # Do not emit another toggle from a held gesture.
            return None

        # Victory has a separate hold duration; media poses may have overrides.
        duration = self.config.activation_duration if gesture == "Victory" else self.config.confirmation_durations.get(gesture, self.config.confirmation_duration)
        # Ignore media poses while disabled, even if they have been held steadily.
        if not self.active and gesture != "Victory":
            # Keep activation instructions visible.
            self.feedback = "OFF - hold Victory to enable"
            # Return without requesting a media command.
            return None
        # Wait until the current pose has remained acceptable for its full hold time.
        if t - self._candidate_since < duration:
            # Display elapsed hold time beside the required duration.
            self.feedback = f"Hold {gesture}: {t - self._candidate_since:.1f}/{duration:.1f}s"
            # Do not act on a partially completed hold.
            return None
        # Handle the reserved activation pose before media-action logic.
        if gesture == "Victory":
            # Latch Victory so a long hold cannot repeatedly toggle control.
            self._latched = gesture
            # Switch between enabled and disabled control.
            self.active = not self.active
            # Start the inactivity timer from this deliberate activation decision.
            self._last_activity = t
            # Explain the new state and whether Victory must be released.
            self.feedback = "ON - show a command gesture" if self.active else "OFF - release Victory before reactivating"
            # Emit a state transition rather than any VLC command.
            return Decision("activated" if self.active else "deactivated", t)
        # Defensively reject media commands whenever control is disabled.
        if not self.active:
            # No media action is allowed in the disabled state.
            return None
        # Repeats must satisfy both the repeat interval and per-action cooldown.
        interval = max(self.config.cooldown, self.config.repeat_interval) if repeating else self.config.cooldown
        # Calculate remaining cooldown; an unseen action is immediately eligible.
        remaining = interval - (t - self._last_actions.get(action, -math.inf))
        # Delay the action if its minimum spacing has not elapsed.
        if remaining > 0:
            # Show a countdown for either repetition or cooldown.
            self.feedback = f"{'Repeating' if repeating else 'Cooldown'} {action}: next in {remaining:.1f}s"
            # Wait for another observation instead of blocking the processing loop.
            return None
        # Latch this accepted pose before the integration loop sends its command.
        self._latched = gesture
        # Record the action time so alternate mappings cannot bypass cooldown.
        self._last_actions[action] = t
        # Refresh inactivity only after accepting a command.
        self._last_activity = t
        # Tell the user whether to hold for repeats or release for another toggle.
        self.feedback = f"Sent {action}; " + ("keep holding to repeat" if action in self.config.repeat_actions else "release to use again")
        # Return the accepted media action to the integration loop.
        return Decision("command", t, action)
