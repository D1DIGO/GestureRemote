from dataclasses import replace

import pytest

from control.gesture_logic import GestureStateMachine
from recognition.observation import GestureObservation
from settings import DEFAULT_MAPPINGS, GestureConfig


class Timeline:
    def __init__(self, **overrides):
        self.machine = GestureStateMachine(replace(GestureConfig(), **overrides))
        self.t = -0.05

    def frame(self, gesture, confidence=0.95, hand=None, jump=0.05):
        self.t = round(self.t + jump, 6)
        return self.machine.update(GestureObservation(gesture, confidence, self.t,
                                                      gesture is not None if hand is None else hand))

    def hold(self, gesture, duration, confidence=0.95, hand=None):
        events = []
        for _ in range(round(duration / 0.05) + 1):
            result = self.frame(gesture, confidence, hand)
            if result:
                events.append(result)
        return events

    def activate(self):
        events = self.hold("Victory", 1.3)
        assert [e.kind for e in events] == ["activated"]
        self.hold(None, 0.4)


def test_disabled_by_default_and_media_cannot_activate():
    t = Timeline()
    assert not t.machine.active
    for gesture in DEFAULT_MAPPINGS:
        assert t.hold(gesture, 2) == []
    assert not t.machine.active


def test_activation_duration_and_no_repeat_until_release():
    t = Timeline()
    assert t.hold("Victory", 1.1) == []
    assert [e.kind for e in t.hold("Victory", 0.3)] == ["activated"]
    assert t.hold("Victory", 5) == []
    t.hold(None, 0.4)
    assert [e.kind for e in t.hold("Victory", 1.3)] == ["deactivated"]
    assert t.hold("Victory", 2) == []


@pytest.mark.parametrize("score", [0.0, 0.74, float("nan"), float("inf"), 1.1])
def test_low_or_invalid_confidence_cannot_activate(score):
    t = Timeline()
    assert t.hold("Victory", 2, confidence=score) == []


def test_threshold_is_inclusive():
    t = Timeline()
    assert [e.kind for e in t.hold("Victory", 1.3, confidence=0.75)] == ["activated"]


def test_confirmation_is_elapsed_time_and_requires_consistency():
    t = Timeline()
    t.activate()
    assert t.hold("Open_Palm", 0.3) == []
    t.frame("Open_Palm", confidence=0.6)
    assert t.hold("Open_Palm", 0.3) == []
    assert [e.action for e in t.hold("Open_Palm", 0.3)] == ["play_pause"]


@pytest.mark.parametrize("step", [0.02, 0.1, 0.2])
def test_activation_independent_of_frame_rate(step):
    t = Timeline()
    events = []
    for _ in range(round(2 / step)):
        event = t.frame("Victory", jump=step)
        if event:
            events.append(event)
    assert [e.kind for e in events] == ["activated"]


@pytest.mark.parametrize("gesture,action", DEFAULT_MAPPINGS.items())
def test_gesture_action_mapping(gesture, action):
    t = Timeline()
    t.activate()
    assert [e.action for e in t.hold(gesture, 0.7)] == [action]
    repeats = t.hold(gesture, 3)
    if action in t.machine.config.repeat_actions:
        assert [e.action for e in repeats] == [action] * 3
        assert all(b.timestamp - a.timestamp >= 1 for a, b in zip(repeats, repeats[1:]))
    else:
        assert repeats == []


def test_custom_mapping():
    t = Timeline(mappings={"Open_Palm": "seek_backward"})
    t.activate()
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["seek_backward"]
    assert t.hold("Thumb_Up", 2) == []


def test_brief_no_hand_does_not_rearm_but_deliberate_release_does():
    t = Timeline()
    t.activate()
    t.hold("Open_Palm", 0.7)
    t.hold(None, 0.1)
    assert t.hold("Open_Palm", 2) == []
    t.hold(None, 0.4)
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]


def test_unknown_or_low_confidence_hands_never_rearm_latch():
    t = Timeline()
    t.activate()
    t.hold("Open_Palm", 0.7)
    t.hold(None, 1, hand=True)
    t.hold("Thumb_Up", 1, confidence=0.5)
    assert t.hold("Open_Palm", 1) == []


def test_stable_change_releases_previous_gesture():
    t = Timeline()
    t.activate()
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]
    assert [e.action for e in t.hold("Thumb_Up", 0.7)] == ["volume_up"]
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]


def test_cooldown_applies_to_action_even_when_two_gestures_share_it():
    t = Timeline(cooldown=3, mappings={"Open_Palm": "play_pause", "Thumb_Up": "play_pause"})
    t.activate()
    t.hold("Open_Palm", 0.7)
    assert t.hold("Thumb_Up", 0.8) == []
    assert [e.action for e in t.hold("Thumb_Up", 3)] == ["play_pause"]


def test_tracking_gap_cancels_candidate_and_does_not_prove_release():
    t = Timeline()
    t.activate()
    t.hold("Open_Palm", 0.3)
    assert t.frame("Open_Palm", jump=2) is None
    assert t.hold("Open_Palm", 0.3) == []
    assert [e.action for e in t.hold("Open_Palm", 0.3)] == ["play_pause"]
    assert t.frame(None, jump=2) is None
    assert t.hold("Open_Palm", 1) == []


def test_long_gap_cannot_count_toward_release():
    t = Timeline()
    t.activate()
    t.hold("Open_Palm", 0.7)
    t.frame(None)
    t.frame(None, jump=1)
    assert t.hold("Open_Palm", 1) == []


def test_timeout_ignores_low_confidence_activity():
    t = Timeline(inactivity_timeout=3)
    t.activate()
    events = t.hold("Open_Palm", 4, confidence=0.2)
    assert [e.kind for e in events] == ["timeout"]
    assert not t.machine.active


def test_accepted_command_refreshes_timeout_but_held_command_does_not():
    t = Timeline(inactivity_timeout=3)
    t.activate()
    t.hold(None, 1.3)
    assert [e.action for e in t.hold("Open_Palm", 0.7)] == ["play_pause"]
    assert t.hold("Open_Palm", 1) == []
    assert t.machine.active
    assert [e.kind for e in t.hold("Open_Palm", 3)] == ["timeout"]


def test_held_victory_cannot_reactivate_after_timeout():
    t = Timeline(inactivity_timeout=3)
    events = t.hold("Victory", 8)
    assert [e.kind for e in events] == ["activated", "timeout"]


@pytest.mark.parametrize("bad_time", [0, -1, float("nan"), float("inf")])
def test_nonincreasing_or_nonfinite_timestamps_rejected(bad_time):
    m = GestureStateMachine(GestureConfig())
    m.update(GestureObservation(None, 0, 0, False))
    with pytest.raises(ValueError, match="timestamps"):
        m.update(GestureObservation("Victory", 1, bad_time))


def test_repeats_stop_on_low_confidence_and_need_a_new_stable_hold():
    t = Timeline()
    t.activate()
    t.hold("Thumb_Up", 1.7)
    assert t.hold("Thumb_Up", 2, confidence=0.4) == []
    assert "Low confidence" in t.machine.feedback
    assert t.hold("Thumb_Up", 0.3) == []
    assert [e.action for e in t.hold("Thumb_Up", 0.3)] == ["volume_up"]


def test_repeats_stop_when_hand_disappears_and_after_deactivation():
    t = Timeline()
    t.activate()
    t.hold("Thumb_Up", 0.7)
    assert t.hold(None, 1.5) == []
    assert [e.kind for e in t.hold("Victory", 1.3)] == ["deactivated"]
    assert t.hold("Thumb_Up", 3) == []


def test_repeat_interval_and_cooldown_both_apply():
    t = Timeline(cooldown=2, repeat_interval=0.5)
    t.activate()
    events = t.hold("Thumb_Up", 5)
    assert len(events) == 3
    assert all(b.timestamp - a.timestamp >= 2 for a, b in zip(events, events[1:]))


def test_repeat_can_be_disabled():
    t = Timeline(repeat_actions=[])
    t.activate()
    assert len(t.hold("Thumb_Up", 4)) == 1


def test_palm_threshold_override_and_longer_hold_do_not_weaken_victory():
    t = Timeline(confidence_thresholds={"Open_Palm": 0.5}, confirmation_durations={"Open_Palm": 0.8})
    assert t.hold("Victory", 2, confidence=0.6) == []
    t.activate()
    assert t.hold("Open_Palm", 0.6, confidence=0.55) == []
    assert "Hold Open_Palm" in t.machine.feedback
    assert [e.action for e in t.hold("Open_Palm", 0.3, confidence=0.55)] == ["play_pause"]
    assert t.hold("Open_Palm", 3, confidence=0.55) == []


def test_disabled_feedback_explains_recognized_but_inactive_pose():
    t = Timeline()
    t.hold("Open_Palm", 1)
    assert "OFF" in t.machine.feedback
    assert "Victory" in t.machine.feedback
