# Read or write tabular performance measurements.
import csv
# Replace hardware or network dependencies with controllable test doubles.
from unittest.mock import Mock

# Work with image arrays and numerical lookup tables.
import numpy as np
# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Access the latest-frame camera wrapper and its error types.
from camera.camera import CapturedFrame, CameraError
# Send accepted media actions through the VLC backend.
from control.media_controller import CommandResult
# Exercise the real application entry point with isolated dependencies.
import main
# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation


# Provide reusable test setup with a fresh instance for each test.
@pytest.fixture
# Provide a deterministic session fixture without camera, GUI, or HTTP access.
def runtime(monkeypatch):
    """Exercise the actual integration loop with a deterministic camera timeline."""
    # Use a mutable clock so nested fake devices can advance application time.
    clock = [0.0]
    # Script activation, release, play/pause, and mute poses at 20 observations/s.
    script = ["Victory"] * 30 + [None] * 8 + ["Open_Palm"] * 16 + ["Closed_Fist"] * 16
    # Track the current scripted pose and whether resources were cleaned up.
    state = {"index": 0, "gesture": None, "camera_closed": False, "recognizer_closed": False}
    # Replace the media backend with an inspectable mock object.
    controller = Mock()
    # Make startup and activation connectivity probes succeed by default.
    controller.check_connection.return_value = CommandResult(True, "connected")
    # Make media commands succeed unless a particular test overrides them.
    controller.execute.return_value = CommandResult(True, "acknowledged")
    # Track whether the integration loop ever constructs a VLC controller.
    factory = Mock(return_value=controller)

    # Stand in for a webcam while retaining the real camera interface.
    class FakeCamera:
        # Accept the usual config argument without opening hardware.
        def __init__(self, config): pass
        # Behave like the real camera's context manager on entry.
        def __enter__(self): return self
        # Record cleanup so tests can check exceptional exit paths.
        def __exit__(self, *args): state["camera_closed"] = True
        # Return one scripted frame and advance the fake clock.
        def read(self, after):
            # Select the pose associated with this frame index.
            state["gesture"] = script[state["index"]]
            # Move forward so every read consumes a new scripted observation.
            state["index"] += 1
            # Advance time by 50ms and avoid cumulative floating-point drift.
            clock[0] = round(clock[0] + 0.05, 6)
            # Return a tiny black frame with a deterministic capture time.
            return CapturedFrame(np.zeros((8, 8, 3), np.uint8), clock[0])

    # Replace native recognition while preserving the observation contract.
    class FakeRecognizer:
        # Accept model settings without loading a native recognizer.
        def __init__(self, config): pass
        # Expose the fake recognizer through a with block.
        def __enter__(self): return self
        # Record recognizer cleanup independently of camera cleanup.
        def __exit__(self, *args): state["recognizer_closed"] = True
        # Translate the current scripted pose into an observation.
        def recognize(self, frame, timestamp):
            # Use high confidence and report no hand for scripted None entries.
            return GestureObservation(state["gesture"], 0.95, timestamp, state["gesture"] is not None)

    # Redirect camera construction to the scripted fake.
    monkeypatch.setattr("camera.camera.Camera", FakeCamera)
    # Redirect model construction to the scripted recognizer.
    monkeypatch.setattr("recognition.gesture_recognition.GestureRecognizer", FakeRecognizer)
    # Intercept controller construction with the mock factory.
    monkeypatch.setattr("control.media_controller.VLCController", factory)
    # Fail the test if no-preview mode accidentally attempts to create a GUI.
    monkeypatch.setattr("preview.Preview", Mock(side_effect=AssertionError("No GUI should be used")))
    # Make the application read simulated time instead of the real clock.
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    # Return all inspectable fakes and stop just before the script runs out.
    return controller, factory, state, ["--no-preview", "--duration", "3.499"]


# Test: integrated gestures trigger vlc and resources close.
def test_integrated_gestures_trigger_vlc_and_resources_close(runtime, tmp_path):
    # Unpack the reusable runtime fixture and its command-line arguments.
    controller, factory, state, args = runtime
    # Use a temporary CSV path isolated from other test runs.
    metrics = tmp_path / "trial.csv"
    # Run the real application entry point and require a clean exit.
    assert main.main(args + ["--metrics", str(metrics)]) == 0
    # Require exactly one play/pause and one mute command in the scripted order.
    assert [c.args[0] for c in controller.execute.call_args_list] == ["play_pause", "mute"]
    # Check both context-managed devices were released.
    assert state["camera_closed"] and state["recognizer_closed"]
    # Verify normal shutdown closes the controller exactly once.
    controller.close.assert_called_once()
    # Read the written telemetry as dictionaries keyed by CSV column names.
    rows = list(csv.DictReader(metrics.open()))
    # Ensure recorded command events match actual backend calls.
    assert [r["action"] for r in rows if r["event"] == "command"] == ["play_pause", "mute"]
    # Ensure those real-mode command attempts are marked successful in CSV.
    assert all(r["command_success"] == "True" for r in rows if r["event"] == "command")


# Test: dry run never constructs or contacts vlc.
def test_dry_run_never_constructs_or_contacts_vlc(runtime):
    # Retrieve the mocked controller, factory, state log, and CLI options.
    controller, factory, state, args = runtime
    # Run the same timeline without enabling real media control.
    assert main.main(args + ["--dry-run"]) == 0
    # Verify dry-run mode never creates a real media controller.
    factory.assert_not_called()
    # Dry-run mode must still release camera and recognition resources.
    assert state["camera_closed"] and state["recognizer_closed"]


# Test: command failure disables further commands.
def test_command_failure_disables_further_commands(runtime):
    # Retrieve the isolated application setup for the failure scenario.
    controller, factory, state, args = runtime
    # Simulate losing VLC when the first media command is attempted.
    controller.execute.return_value = CommandResult(False, "connection lost")
    # A backend failure should disable control without crashing the process.
    assert main.main(args) == 0
    # The first failure must prevent the later scripted mute command.
    controller.execute.assert_called_once_with("play_pause")


# Test: vlc unavailable rejects activation.
def test_vlc_unavailable_rejects_activation(runtime):
    # Retrieve the mocks and CLI options used by this scenario.
    controller, factory, state, args = runtime
    # Simulate VLC being unavailable for both connectivity checks.
    controller.check_connection.return_value = CommandResult(False, "unavailable")
    # Require main.main(args) == 0 for this scenario to pass.
    assert main.main(args) == 0
    # Verify a failed activation probe prevents media commands.
    controller.execute.assert_not_called()
    # Expect only the startup probe and the deliberate activation probe.
    assert controller.check_connection.call_count == 2


# Test: camera failure exits cleanly.
def test_camera_failure_exits_cleanly(runtime, monkeypatch):
    # Retrieve the mocked runtime to test exit and cleanup behavior.
    controller, factory, state, args = runtime
    # Simulate a camera disappearing on the next frame read.
    monkeypatch.setattr("camera.camera.Camera.read", Mock(side_effect=CameraError("unplugged")))
    # The application should report a failure exit code for a camera error.
    assert main.main(args) == 1
    # Require state['camera_closed'] and state['recognizer_closed'] for this scenario to pass.
    assert state["camera_closed"] and state["recognizer_closed"]
    # Verify cleanup closes the controller even when the loop exits early.
    controller.close.assert_called_once()


# Test: keyboard interrupt cleans up.
def test_keyboard_interrupt_cleans_up(runtime, monkeypatch):
    # Retrieve the runtime dependencies for the exception scenario.
    controller, factory, state, args = runtime
    # Simulate Ctrl+C while the main loop waits for a frame.
    monkeypatch.setattr("camera.camera.Camera.read", Mock(side_effect=KeyboardInterrupt))
    # A deliberate user interruption should still return success.
    assert main.main(args) == 0
    # Require state['camera_closed'] and state['recognizer_closed'] for this scenario to pass.
    assert state["camera_closed"] and state["recognizer_closed"]
    # Verify exceptional shutdown also closes the HTTP controller.
    controller.close.assert_called_once()


# Test: stale inference cannot activate.
def test_stale_inference_cannot_activate(runtime, monkeypatch):
    # Retrieve the runtime mocks before replacing the captured timestamps.
    controller, factory, state, args = runtime
    # Shift captured timestamp far behind the integration clock.
    # Look up the patched fake camera class currently used by the application.
    camera_class = __import__("camera.camera", fromlist=["Camera"]).Camera
    # Keep its original read method so the wrapper can reuse the scripted timeline.
    read = camera_class.read

    # Produce an otherwise valid frame with an intentionally stale timestamp.
    def old_frame(self, after):
        # Obtain the next normal scripted frame first.
        current = read(self, after)
        # Move its capture time one second into the past.
        return CapturedFrame(current.image, current.timestamp - 1)

    # Replace reads with the stale-frame wrapper for this test.
    monkeypatch.setattr(camera_class, "read", old_frame)
    # Require main.main(args) == 0 for this scenario to pass.
    assert main.main(args) == 0
    # Verify stale frames cannot send playback commands.
    controller.execute.assert_not_called()
    # Stale frames must never activate control and cause a second connectivity probe.
    assert controller.check_connection.call_count == 1


# Test: config check needs no camera or password.
def test_config_check_needs_no_camera_or_password(monkeypatch):
    # Ensure this diagnostic does not accidentally depend on a real secret.
    monkeypatch.delenv("GESTUREREMOTE_VLC_PASSWORD", raising=False)
    # Validate settings without requiring a camera or password.
    assert main.main(["--check-config"]) == 0


# Exercise zero, negative, NaN, and infinite session durations.
@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
# Test: invalid duration.
def test_invalid_duration(duration):
    # argparse should reject these values by raising a usage-error exit.
    with pytest.raises(SystemExit) as error:
        # Parse the intentionally invalid duration without starting the app.
        main.parse_args(["--duration", duration])
    # Exit code 2 denotes a command-line usage error.
    assert error.value.code == 2
