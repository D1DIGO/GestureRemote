# Define structured records or create modified copies of configuration records.
from dataclasses import replace

# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Apply gesture confirmation, activation, and repetition rules.
from control.gesture_logic import GestureStateMachine
# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation
# Load validated configuration types, defaults, or configuration helpers.
from settings import DEFAULT_MAPPINGS, GestureConfig


# Simulate timestamped camera observations without waiting in real time.
class Timeline:
    # Build an isolated gesture machine with optional settings for this scenario.
    def __init__(self, **overrides):
        # Apply test-specific overrides to the normal gesture configuration.
        self.machine = GestureStateMachine(replace(GestureConfig(), **overrides))
        # Start one frame before zero so the first simulated timestamp is zero.
        self.t = -0.05

    # Deliver one observation; jump controls the simulated time between frames.
    def frame(self, gesture, confidence=0.95, hand=None, jump=0.05):
        # Advance simulated time while avoiding accumulated floating-point noise.
        self.t = round(self.t + jump, 6)
        # Pass the simulated category, confidence, and timestamp to the decision logic.
        return self.machine.update(GestureObservation(gesture, confidence, self.t,
                                                      # Infer hand presence from the category unless the test explicitly overrides it.
                                                      gesture is not None if hand is None else hand))

    # Repeat a pose at 20 simulated frames per second and collect emitted events.
    def hold(self, gesture, duration, confidence=0.95, hand=None):
        # Collect only events produced during this simulated interval.
        events = []
        # Include both endpoints of the requested hold interval.
        for _ in range(round(duration / 0.05) + 1):
            # Feed the next observation with unchanged pose and confidence.
            result = self.frame(gesture, confidence, hand)
            # Ignore frames that have not produced an event.
            if result:
                # Record the event for later assertions about timing and action.
                events.append(result)
        # Expose all events produced during the hold to the test.
        return events

    # Provide the standard deliberate activation and release sequence.
    def activate(self):
        # Hold Victory long enough to exceed the default activation duration.
        events = self.hold("Victory", 1.3)
        # Verify the exact state-transition sequence is ['activated'].
        assert [e.kind for e in events] == ["activated"]
        # Release the activation pose before testing media gestures.
        self.hold(None, 0.4)


# Test: disabled by default and media cannot activate.
def test_disabled_by_default_and_media_cannot_activate():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Verify gesture control remains disabled.
    assert not t.machine.active
    # Check every default media gesture while control is disabled.
    for gesture in DEFAULT_MAPPINGS:
        # Verify this interval produces no additional control events.
        assert t.hold(gesture, 2) == []
    # Verify gesture control remains disabled.
    assert not t.machine.active


# Test: activation duration and no repeat until release.
def test_activation_duration_and_no_repeat_until_release():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Verify this interval produces no additional control events.
    assert t.hold("Victory", 1.1) == []
    # Verify the exact state-transition sequence is ['activated'].
    assert [e.kind for e in t.hold("Victory", 0.3)] == ["activated"]
    # Verify this interval produces no additional control events.
    assert t.hold("Victory", 5) == []
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold(None, 0.4)
    # Verify the exact state-transition sequence is ['deactivated'].
    assert [e.kind for e in t.hold("Victory", 1.3)] == ["deactivated"]
    # Verify this interval produces no additional control events.
    assert t.hold("Victory", 2) == []


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("score", [0.0, 0.74, float("nan"), float("inf"), 1.1])
# Test: low or invalid confidence cannot activate.
def test_low_or_invalid_confidence_cannot_activate(score):
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Verify this interval produces no additional control events.
    assert t.hold("Victory", 2, confidence=score) == []


# Test: threshold is inclusive.
def test_threshold_is_inclusive():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Verify the exact state-transition sequence is ['activated'].
    assert [e.kind for e in t.hold("Victory", 1.3, confidence=0.75)] == ["activated"]


# Test: confirmation is elapsed time and requires consistency.
def test_confirmation_is_elapsed_time_and_requires_consistency():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 0.3) == []
    # Inject one observation to simulate a confidence change or tracking gap.
    t.frame("Open_Palm", confidence=0.6)
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 0.3) == []
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.3)] == ["play_pause"]


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("step", [0.02, 0.1, 0.2])
# Test: activation independent of frame rate.
def test_activation_independent_of_frame_rate(step):
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Collect only events produced during this simulated interval.
    events = []
    # Simulate roughly two seconds at the selected frame interval.
    for _ in range(round(2 / step)):
        # Feed Victory observations at the selected simulated frame rate.
        event = t.frame("Victory", jump=step)
        # Record only observations that produce an activation event.
        if event:
            # Collect the event to check that activation occurred exactly once.
            events.append(event)
    # Verify the exact state-transition sequence is ['activated'].
    assert [e.kind for e in events] == ["activated"]


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("gesture,action", DEFAULT_MAPPINGS.items())
# Test: gesture action mapping.
def test_gesture_action_mapping(gesture, action):
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify the accepted media-action sequence is [action].
    assert [e.action for e in t.hold(gesture, 0.7)] == [action]
    # Continue holding the same gesture to test its repetition policy.
    repeats = t.hold(gesture, 3)
    # Distinguish repeatable adjustments from one-shot toggle commands.
    if action in t.machine.config.repeat_actions:
        # Verify the accepted media-action sequence is [action] * 3.
        assert [e.action for e in repeats] == [action] * 3
        # Verify every consecutive repeat respects the configured minimum interval.
        assert all(b.timestamp - a.timestamp >= 1 for a, b in zip(repeats, repeats[1:]))
    # Handle the alternative to the preceding condition.
    else:
        # Verify this interval produces no additional control events.
        assert repeats == []


# Test: custom mapping.
def test_custom_mapping():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(mappings={"Open_Palm": "seek_backward"})
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify the accepted media-action sequence is ['seek_backward'].
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["seek_backward"]
    # Verify this interval produces no additional control events.
    assert t.hold("Thumb_Up", 2) == []


# Test: brief no hand does not rearm but deliberate release does.
def test_brief_no_hand_does_not_rearm_but_deliberate_release_does():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 0.7)
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold(None, 0.1)
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 2) == []
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold(None, 0.4)
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]


# Test: unknown or low confidence hands never rearm latch.
def test_unknown_or_low_confidence_hands_never_rearm_latch():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 0.7)
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold(None, 1, hand=True)
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Thumb_Up", 1, confidence=0.5)
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 1) == []


# Test: stable change releases previous gesture.
def test_stable_change_releases_previous_gesture():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]
    # Verify the accepted media-action sequence is ['volume_up'].
    assert [e.action for e in t.hold("Thumb_Up", 0.7)] == ["volume_up"]
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]


# Test: cooldown applies to action even when two gestures share it.
def test_cooldown_applies_to_action_even_when_two_gestures_share_it():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(cooldown=3, mappings={"Open_Palm": "play_pause", "Thumb_Up": "play_pause"})
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 0.7)
    # Verify this interval produces no additional control events.
    assert t.hold("Thumb_Up", 0.8) == []
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Thumb_Up", 3)] == ["play_pause"]


# Test: tracking gap cancels candidate and does not prove release.
def test_tracking_gap_cancels_candidate_and_does_not_prove_release():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 0.3)
    # Verify this isolated observation cannot trigger an event.
    assert t.frame("Open_Palm", jump=2) is None
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 0.3) == []
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.3)] == ["play_pause"]
    # Verify this isolated observation cannot trigger an event.
    assert t.frame(None, jump=2) is None
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 1) == []


# Test: long gap cannot count toward release.
def test_long_gap_cannot_count_toward_release():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 0.7)
    # Inject one observation to simulate a confidence change or tracking gap.
    t.frame(None)
    # Inject one observation to simulate a confidence change or tracking gap.
    t.frame(None, jump=1)
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 1) == []


# Test: timeout ignores low confidence activity.
def test_timeout_ignores_low_confidence_activity():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(inactivity_timeout=3)
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Keep an uncertain palm visible beyond the inactivity timeout.
    events = t.hold("Open_Palm", 4, confidence=0.2)
    # Verify the exact state-transition sequence is ['timeout'].
    assert [e.kind for e in events] == ["timeout"]
    # Verify gesture control remains disabled.
    assert not t.machine.active


# Test: accepted command refreshes timeout but held command does not.
def test_accepted_command_refreshes_timeout_but_held_command_does_not():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(inactivity_timeout=3)
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold(None, 1.3)
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 1) == []
    # Verify accepted activity keeps gesture control enabled.
    assert t.machine.active
    # Verify the exact state-transition sequence is ['timeout'].
    assert [e.kind for e in t.hold("Open_Palm", 3)] == ["timeout"]


# Test: held victory cannot reactivate after timeout.
def test_held_victory_cannot_reactivate_after_timeout():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(inactivity_timeout=3)
    # Keep Victory held through activation and the later idle timeout.
    events = t.hold("Victory", 8)
    # Verify the exact state-transition sequence is ['activated', 'timeout'].
    assert [e.kind for e in events] == ["activated", "timeout"]


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("bad_time", [0, -1, float("nan"), float("inf")])
# Test: nonincreasing or nonfinite timestamps rejected.
def test_nonincreasing_or_nonfinite_timestamps_rejected(bad_time):
    # Create a fresh machine for validating observation timestamps.
    m = GestureStateMachine(GestureConfig())
    # Establish zero as the last accepted timestamp.
    m.update(GestureObservation(None, 0, 0, False))
    # Require the enclosed operation to raise the specified validation error.
    with pytest.raises(ValueError, match="timestamps"):
        # Submit an invalid timestamp and require validation to reject it.
        m.update(GestureObservation("Victory", 1, bad_time))


# Test: repeats stop on low confidence and need a new stable hold.
def test_repeats_stop_on_low_confidence_and_need_a_new_stable_hold():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Thumb_Up", 1.7)
    # Verify this interval produces no additional control events.
    assert t.hold("Thumb_Up", 2, confidence=0.4) == []
    # Verify the preview feedback explains the current control state.
    assert "Low confidence" in t.machine.feedback
    # Verify this interval produces no additional control events.
    assert t.hold("Thumb_Up", 0.3) == []
    # Verify the accepted media-action sequence is ['volume_up'].
    assert [e.action for e in t.hold("Thumb_Up", 0.3)] == ["volume_up"]


# Test: repeats stop when hand disappears and after deactivation.
def test_repeats_stop_when_hand_disappears_and_after_deactivation():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Thumb_Up", 0.7)
    # Verify this interval produces no additional control events.
    assert t.hold(None, 1.5) == []
    # Verify the exact state-transition sequence is ['deactivated'].
    assert [e.kind for e in t.hold("Victory", 1.3)] == ["deactivated"]
    # Verify this interval produces no additional control events.
    assert t.hold("Thumb_Up", 3) == []


# Test: repeat interval and cooldown both apply.
def test_repeat_interval_and_cooldown_both_apply():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(cooldown=2, repeat_interval=0.5)
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Hold volume-up long enough to observe several rate-limited repeats.
    events = t.hold("Thumb_Up", 5)
    # Verify the hold produces exactly 3 command events.
    assert len(events) == 3
    # Verify every consecutive repeat respects the configured minimum interval.
    assert all(b.timestamp - a.timestamp >= 2 for a, b in zip(events, events[1:]))


# Test: repeat can be disabled.
def test_repeat_can_be_disabled():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(repeat_actions=[])
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify the hold produces exactly 1 command events.
    assert len(t.hold("Thumb_Up", 4)) == 1


# Test: palm threshold override and longer hold do not weaken victory.
def test_palm_threshold_override_and_longer_hold_do_not_weaken_victory():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline(confidence_thresholds={"Open_Palm": 0.5}, confirmation_durations={"Open_Palm": 0.8})
    # Verify this interval produces no additional control events.
    assert t.hold("Victory", 2, confidence=0.6) == []
    # Activate control and release Victory before testing a media gesture.
    t.activate()
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 0.6, confidence=0.55) == []
    # Verify the preview feedback explains the current control state.
    assert "Hold Open_Palm" in t.machine.feedback
    # Verify the accepted media-action sequence is ['play_pause'].
    assert [e.action for e in t.hold("Open_Palm", 0.3, confidence=0.55)] == ["play_pause"]
    # Verify this interval produces no additional control events.
    assert t.hold("Open_Palm", 3, confidence=0.55) == []


# Test: disabled feedback explains recognized but inactive pose.
def test_disabled_feedback_explains_recognized_but_inactive_pose():
    # Create a fresh timeline with the settings needed for this scenario.
    t = Timeline()
    # Advance the scenario by holding this pose for the given number of seconds.
    t.hold("Open_Palm", 1)
    # Verify the preview feedback explains the current control state.
    assert "OFF" in t.machine.feedback
    # Verify the preview feedback explains the current control state.
    assert "Victory" in t.machine.feedback
