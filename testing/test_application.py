import csv
from unittest.mock import Mock

import numpy as np
import pytest

from camera.camera import CapturedFrame, CameraError
from control.media_controller import CommandResult
import main
from recognition.observation import GestureObservation


@pytest.fixture
def runtime(monkeypatch):
    """Exercise the actual integration loop with a deterministic camera timeline."""
    clock = [0.0]
    script = ["Victory"] * 30 + [None] * 8 + ["Open_Palm"] * 16 + ["Closed_Fist"] * 16
    state = {"index": 0, "gesture": None, "camera_closed": False, "recognizer_closed": False}
    controller = Mock()
    controller.check_connection.return_value = CommandResult(True, "connected")
    controller.execute.return_value = CommandResult(True, "acknowledged")
    factory = Mock(return_value=controller)

    class FakeCamera:
        def __init__(self, config): pass
        def __enter__(self): return self
        def __exit__(self, *args): state["camera_closed"] = True
        def read(self, after):
            state["gesture"] = script[state["index"]]
            state["index"] += 1
            clock[0] = round(clock[0] + 0.05, 6)
            return CapturedFrame(np.zeros((8, 8, 3), np.uint8), clock[0])

    class FakeRecognizer:
        def __init__(self, config): pass
        def __enter__(self): return self
        def __exit__(self, *args): state["recognizer_closed"] = True
        def recognize(self, frame, timestamp):
            return GestureObservation(state["gesture"], 0.95, timestamp, state["gesture"] is not None)

    monkeypatch.setattr("camera.camera.Camera", FakeCamera)
    monkeypatch.setattr("recognition.gesture_recognition.GestureRecognizer", FakeRecognizer)
    monkeypatch.setattr("control.media_controller.VLCController", factory)
    monkeypatch.setattr("preview.Preview", Mock(side_effect=AssertionError("No GUI should be used")))
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    return controller, factory, state, ["--no-preview", "--duration", "3.499"]


def test_integrated_gestures_trigger_vlc_and_resources_close(runtime, tmp_path):
    controller, factory, state, args = runtime
    metrics = tmp_path / "trial.csv"
    assert main.main(args + ["--metrics", str(metrics)]) == 0
    assert [c.args[0] for c in controller.execute.call_args_list] == ["play_pause", "mute"]
    assert state["camera_closed"] and state["recognizer_closed"]
    controller.close.assert_called_once()
    rows = list(csv.DictReader(metrics.open()))
    assert [r["action"] for r in rows if r["event"] == "command"] == ["play_pause", "mute"]
    assert all(r["command_success"] == "True" for r in rows if r["event"] == "command")


def test_dry_run_never_constructs_or_contacts_vlc(runtime):
    controller, factory, state, args = runtime
    assert main.main(args + ["--dry-run"]) == 0
    factory.assert_not_called()
    assert state["camera_closed"] and state["recognizer_closed"]


def test_command_failure_disables_further_commands(runtime):
    controller, factory, state, args = runtime
    controller.execute.return_value = CommandResult(False, "connection lost")
    assert main.main(args) == 0
    controller.execute.assert_called_once_with("play_pause")


def test_vlc_unavailable_rejects_activation(runtime):
    controller, factory, state, args = runtime
    controller.check_connection.return_value = CommandResult(False, "unavailable")
    assert main.main(args) == 0
    controller.execute.assert_not_called()
    assert controller.check_connection.call_count == 2


def test_camera_failure_exits_cleanly(runtime, monkeypatch):
    controller, factory, state, args = runtime
    monkeypatch.setattr("camera.camera.Camera.read", Mock(side_effect=CameraError("unplugged")))
    assert main.main(args) == 1
    assert state["camera_closed"] and state["recognizer_closed"]
    controller.close.assert_called_once()


def test_keyboard_interrupt_cleans_up(runtime, monkeypatch):
    controller, factory, state, args = runtime
    monkeypatch.setattr("camera.camera.Camera.read", Mock(side_effect=KeyboardInterrupt))
    assert main.main(args) == 0
    assert state["camera_closed"] and state["recognizer_closed"]
    controller.close.assert_called_once()


def test_stale_inference_cannot_activate(runtime, monkeypatch):
    controller, factory, state, args = runtime
    # Shift captured timestamp far behind the integration clock.
    camera_class = __import__("camera.camera", fromlist=["Camera"]).Camera
    read = camera_class.read

    def old_frame(self, after):
        current = read(self, after)
        return CapturedFrame(current.image, current.timestamp - 1)

    monkeypatch.setattr(camera_class, "read", old_frame)
    assert main.main(args) == 0
    controller.execute.assert_not_called()
    assert controller.check_connection.call_count == 1


def test_config_check_needs_no_camera_or_password(monkeypatch):
    monkeypatch.delenv("GESTUREREMOTE_VLC_PASSWORD", raising=False)
    assert main.main(["--check-config"]) == 0


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_duration(duration):
    with pytest.raises(SystemExit) as error:
        main.parse_args(["--duration", duration])
    assert error.value.code == 2
