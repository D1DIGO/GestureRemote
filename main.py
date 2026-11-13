"""GestureRemote CLI and integration loop. Run python main.py --help."""

# Parse command-line flags and display usage errors.
import argparse
# Register cleanup callbacks for resources acquired during startup.
from contextlib import ExitStack
# Define structured records or create modified copies of configuration records.
from dataclasses import replace
# Report runtime status and errors through the standard logging system.
import logging
# Validate finite numbers before using them in timing or volume calculations.
import math
# Access process environment variables and operating-system helpers.
import os
# Handle filesystem paths without platform-specific separators.
from pathlib import Path
# Handle termination requests so the application can clean up.
import signal
# Access process arguments, exit behavior, or platform information.
import sys
# Measure monotonic elapsed time and perform short waits.
import time

# Apply gesture confirmation, activation, and repetition rules.
from control.gesture_logic import GestureStateMachine
# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation
# Load validated configuration types, defaults, or configuration helpers.
from settings import AppConfig, load_config

# Use one logger for application-level status messages.
log = logging.getLogger("gestureremote")
# Resolve default paths relative to this script, not the terminal directory.
ROOT = Path(__file__).resolve().parent


# Validate a positive time limit supplied on the command line.
def positive_seconds(value: str) -> float:
    # Convert the command-line string to seconds.
    result = float(value)
    # Reject NaN, infinity, zero, and negative durations.
    if not math.isfinite(result) or result <= 0:
        # Let argparse display the validation error as a CLI usage error.
        raise argparse.ArgumentTypeError("duration must be a finite positive number")
    # Return the validated number of seconds.
    return result


# Define the available command-line options and parse the supplied arguments.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Create the command-line parser and its help description.
    parser = argparse.ArgumentParser(description="Control VLC using deliberate, held hand gestures.")
    # Allow a custom JSON configuration path.
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    # Provide a mode that runs without opening a preview window.
    parser.add_argument("--no-preview", action="store_true", help="run without an OpenCV window; Ctrl+C exits")
    # Allow side-by-side original and enhanced previews.
    parser.add_argument("--compare", action="store_true", help="show original and enhanced frames side by side")
    # Prevent contradictory preprocessing flags in the same command.
    group = parser.add_mutually_exclusive_group()
    # Override configuration to enable image enhancement.
    group.add_argument("--preprocess", action="store_true", help="use configured enhancement for inference")
    # Override configuration to use the original frames.
    group.add_argument("--no-preprocess", action="store_true", help="use raw frames for inference")
    # Allow recognition tests without sending any VLC requests.
    parser.add_argument("--dry-run", action="store_true", help="recognize and log decisions without contacting VLC")
    # Optionally record per-frame measurements in a new CSV file.
    parser.add_argument("--metrics", type=Path, help="write per-frame measurements to a NEW CSV file")
    # Validate and store an optional session time limit.
    parser.add_argument("--duration", type=positive_seconds, help="exit after this many seconds")
    # Allow only one standalone diagnostic check at a time.
    checks = parser.add_mutually_exclusive_group()
    # Offer configuration validation without loading the vision libraries.
    checks.add_argument("--check-config", action="store_true", help="validate configuration without loading native dependencies")
    # Offer a read-only VLC connectivity check.
    checks.add_argument("--check-vlc", action="store_true", help="read VLC status once and exit; no playback changes")
    # Offer CPU model inference without requiring a webcam.
    checks.add_argument("--check-model", action="store_true", help="run CPU inference on a blank frame; no camera or VLC")
    # Return parsed options; None means read the actual command line.
    return parser.parse_args(argv)


# Connect capture, recognition, decisions, playback control, and the preview.
def run(config: AppConfig, args: argparse.Namespace) -> int:
    # Access the latest-frame camera wrapper and its error types.
    from camera.camera import Camera
    # Apply the optional image-enhancement pipeline.
    from camera.image_processing import ImageProcessor
    # Send accepted media actions through the VLC backend.
    from control.media_controller import VLCController
    # Render recognition status and control feedback in a window.
    from preview import Preview
    # Convert camera images into MediaPipe gesture observations.
    from recognition.gesture_recognition import GestureRecognizer
    # Measure processing performance and record optional CSV metrics.
    from testing.performance_monitor import MetricsRecorder, PerformanceMonitor

    # Create an initially disabled gesture-control state machine.
    machine = GestureStateMachine(config.gestures)
    # Prepare reusable enhancement resources, such as the gamma lookup table.
    processor = ImageProcessor(config.preprocessing)
    # Track processing speed and inference latency over recent frames.
    monitor = PerformanceMonitor()
    # Start with the preview preference from configuration.
    preview_enabled = config.preview
    # Detect a Linux session without display-related environment variables.
    if preview_enabled and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        # Explain why this session will continue without a window.
        log.warning("No display session detected; continuing without preview (Ctrl+C to exit)")
        # Skip GUI creation when no desktop session was detected.
        preview_enabled = False
    # Register resource cleanup so it also runs after errors or interruption.
    with ExitStack() as resources:
        # Use no media backend unless real playback control is enabled.
        controller = None
        # Initialize the status message for dry-run sessions.
        status = "Dry run: media commands are only logged"
        # Create the HTTP controller only when commands may reach VLC.
        if not args.dry_run:
            # Build the controller using validated connection settings.
            controller = VLCController(config.vlc)
            # Close the HTTP session when leaving the resource scope.
            resources.callback(controller.close)
            # Check VLC availability without changing playback.
            result = controller.check_connection()
            # Keep the connection result for display in the preview.
            status = result.message
            # Log successful probes normally and failures as warnings.
            (log.info if result.success else log.warning)(status)
        # Create a CSV recorder only if a metrics path was requested.
        recorder = MetricsRecorder(args.metrics) if args.metrics else None
        # Register cleanup only when a recorder exists.
        if recorder:
            # Ensure buffered CSV data is closed on every exit path.
            resources.callback(recorder.close)
        # Create the CPU recognizer and register its context-manager cleanup.
        recognizer = resources.enter_context(GestureRecognizer(config.recognition))
        # Open the webcam and register capture cleanup.
        camera = resources.enter_context(Camera(config.camera))
        # Create a preview only when this session allows one.
        preview = Preview() if preview_enabled else None
        # Register GUI cleanup only when a window was created.
        if preview:
            # Destroy preview windows when the session ends.
            resources.callback(preview.close)
        # There has not yet been a successfully executed media command.
        last_action = "none"
        # Accept the first camera frame regardless of its monotonic timestamp.
        previous_frame = -1.0
        # Record a clock value unaffected by wall-clock adjustments.
        start = time.monotonic()
        # Schedule the first periodic status report five seconds later.
        next_report = start + 5
        # Tell the user how to activate control and exit the application.
        log.info("Control OFF. Hold Victory %.1fs to enable. Exit: Q/Esc in preview, or Ctrl+C in terminal.", config.gestures.activation_duration)
        # Process frames until the optional session duration has elapsed.
        while args.duration is None or time.monotonic() - start < args.duration:
            # Request a frame newer than the one already processed.
            captured = camera.read(after=previous_frame)
            # Remember this frame so it cannot be processed twice.
            previous_frame = captured.timestamp
            # Start measuring preprocessing plus recognition time.
            inference_start = time.monotonic()
            # Produce an enhanced image when enabled or needed for visual comparison.
            enhanced = processor.process(captured.image, force=config.compare_preview)
            # Choose exactly one input image for recognition.
            inference_frame = enhanced if config.preprocessing.enabled else captured.image
            # Recognize the pose and preserve the frame's capture timestamp.
            observation = recognizer.recognize(inference_frame, captured.timestamp)
            # Sample the clock after recognition has finished.
            now = time.monotonic()
            # Convert the processing duration from seconds to milliseconds.
            inference_ms = (now - inference_start) * 1000
            # Measure how old the captured frame has become.
            frame_age_ms = (now - captured.timestamp) * 1000
            # Reject decisions based on frames older than the configured safety limit.
            if frame_age_ms > config.gestures.max_frame_gap * 1000:
                # An old frame cannot confirm a current gesture or prove release.
                # Cancel pose confirmation without treating a stale frame as a hand release.
                observation = GestureObservation(None, 0, captured.timestamp, hand_detected=True)
                # Explain the stale-frame rejection to the user.
                status = "Frame too old; lower resolution or disable enhancement"
            # Ask the state machine whether this observation permits an action.
            decision = machine.update(observation)
            # Leave command success blank when no real HTTP command is attempted.
            command_success: bool | str = ""
            # Leave HTTP timing blank when there is no command request.
            command_ms: float | str = ""
            # Handle only frames that produced a state transition or command.
            if decision is not None:
                # Check the backend again when the user activates gesture control.
                if decision.kind == "activated":
                    # Dry-run sessions do not need a VLC connectivity check.
                    if controller:
                        # Confirm VLC is reachable before accepting media commands.
                        result = controller.check_connection()
                        # Show the latest connectivity result.
                        status = result.message
                        # Reject activation if VLC is unavailable or authentication fails.
                        if not result.success:
                            # Return to the disabled state while preserving release protection.
                            machine.deactivate()
                            # Explain the deliberate gesture needed to retry activation.
                            status += "; control OFF, release and hold Victory to retry"
                    # Log the final state after the connectivity check.
                    log.info("Control %s: %s", "ON" if machine.active else "OFF", status)
                # Handle either deliberate deactivation or an inactivity timeout.
                elif decision.kind in ("deactivated", "timeout"):
                    # Record why control was disabled.
                    status = "Control OFF: " + decision.kind
                    # Report this transition immediately, without waiting for the periodic log.
                    log.info(status)
                # Dispatch an accepted media action.
                elif decision.kind == "command":
                    # In dry-run mode, describe the action instead of sending it.
                    if controller is None:
                        # Label simulated commands so they cannot be mistaken for playback changes.
                        last_action = f"would execute {decision.action}"
                        # Make the absence of an HTTP request explicit in the preview.
                        status = "Dry run: nothing sent to VLC"
                        # Log the simulated command for testing.
                        log.info(last_action)
                    # A real controller exists, so execute the accepted command.
                    else:
                        # Start measuring the command's HTTP round-trip work.
                        command_start = time.monotonic()
                        # Send the action through the media-controller interface.
                        result = controller.execute(decision.action)
                        # Measure request and verification time in milliseconds.
                        command_ms = (time.monotonic() - command_start) * 1000
                        # Record the backend's success flag for the CSV output.
                        command_success = result.success
                        # Use the backend's message rather than assuming playback changed.
                        status = result.message
                        # Update the last successful command only after backend success.
                        if result.success:
                            # Remember the acknowledged or verified action for the preview.
                            last_action = decision.action
                            # Log the executed action and its result.
                            log.info("Command %s: %s", decision.action, status)
                        # On failure, require a new deliberate activation before further commands.
                        else:
                            # Disable command acceptance instead of silently retrying an uncertain action.
                            machine.deactivate()
                            # Keep the disabled state visible alongside the error.
                            status += "; control OFF"
                            # Tell the user which action failed and how to retry safely.
                            log.warning("Command %s: %s. Release and hold Victory to retry.", decision.action, status)
            # Include command delays in the completed-frame FPS measurement.
            monitor.record(time.monotonic(), inference_ms, observation.hand_detected)
            # Write measurements only when a CSV recorder was requested.
            if recorder:
                # Record frame time and the raw recognized category, if any.
                recorder.write(timestamp=captured.timestamp, gesture=observation.gesture or "",
                               # Record confidence separately from whether a hand was detected.
                               confidence=observation.confidence, hand_detected=observation.hand_detected,
                               # Store the final control state and the state-machine event.
                               active=machine.active, event=decision.kind if decision else "",
                               # Store the requested action and the real HTTP outcome, if applicable.
                               action=decision.action if decision else "", command_success=command_success,
                               # Keep HTTP time, processing time, and frame age as distinct measurements.
                               command_ms=command_ms, inference_ms=inference_ms, frame_age_ms=frame_age_ms,
                               # Record FPS and the processing mode to make trial comparisons possible.
                               fps=monitor.fps, preprocessing=config.preprocessing.enabled, dry_run=args.dry_run)
            # Show the current frame and stop if the preview requests exit.
            if preview and not preview.show(
                # Use the original image on the left in comparison mode.
                captured.image if config.compare_preview else inference_frame,
                # Supply the enhanced image only for a two-image preview.
                enhanced if config.compare_preview else None, observation,
                # Pass current control state, command feedback, and processing speed.
                active=machine.active, last_action=last_action, status=status, fps=monitor.fps,
                # Label preprocessing and dry-run mode accurately in the UI.
                preprocessing=config.preprocessing.enabled, dry_run=args.dry_run,
                # Explain why the current pose was accepted, delayed, or rejected.
                feedback=machine.feedback,
            ):
                # Leave the frame loop when the preview closes or an exit key is pressed.
                break
            # Limit periodic status logging to once every five seconds.
            if time.monotonic() >= next_report:
                # Build a compact runtime status message.
                log.info("Control %s | %s %.2f | %.1f FPS | processing %.1f ms | last: %s | %s",
                         # Include the active state and current recognized label and confidence.
                         "ON" if machine.active else "OFF", observation.gesture or "None", observation.confidence,
                         # Include FPS, average processing time, and decision feedback.
                         monitor.fps, monitor.mean_inference_ms, last_action, machine.feedback)
                # Schedule the next report relative to the current monotonic time.
                next_report = time.monotonic() + 5
        # Summarize processed frames without claiming labeled recognition accuracy.
        log.info("Stopped after %d frames; no-hand frames: %d (not a labeled accuracy measurement)",
                 # Report total frames and frames without a detected hand.
                 monitor.frames, monitor.no_hand_frames)
    # Return a successful process exit code after orderly cleanup.
    return 0


# Handle CLI modes and convert expected failures into clear exit codes.
def main(argv: list[str] | None = None) -> int:
    # Parse command-line options before configuring the application.
    args = parse_args(argv)
    # Set a readable default log format with timestamps and severity levels.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Handle failures from the operations in this block.
    try:
        # Load and validate JSON before opening devices or contacting VLC.
        config = load_config(args.config)
        # Enable detailed logging only when debug mode is configured.
        logging.getLogger().setLevel(logging.DEBUG if config.debug else logging.INFO)
        # Apply the no-preview override without mutating the original dataclass.
        config = replace(config, preview=config.preview and not args.no_preview,
                         # Enable comparison if either JSON or the CLI requested it.
                         compare_preview=config.compare_preview or args.compare)
        # Override preprocessing only when an explicit CLI flag was supplied.
        if args.preprocess or args.no_preprocess:
            # Replace nested immutable settings with a modified copy.
            config = replace(config, preprocessing=replace(config.preprocessing, enabled=args.preprocess))
        # Finish here for a configuration-only diagnostic run.
        if args.check_config:
            # Report that all configuration checks succeeded.
            print("Configuration valid")
            # Report successful completion to the calling process.
            return 0
        # Run a read-only VLC probe instead of the camera loop.
        if args.check_vlc:
            # Send accepted media actions through the VLC backend.
            from control.media_controller import VLCController
            # Create the backend; a missing password produces a clear setup error.
            controller = VLCController(config.vlc)
            # Handle failures from the operations in this block.
            try:
                # Request status without any playback command parameters.
                result = controller.check_connection()
                # Print the probe's human-readable result.
                print(result.message)
                # Use exit code 1 when the connection probe failed.
                return 0 if result.success else 1
            # Run cleanup whether the preceding operations succeeded or failed.
            finally:
                # Release the HTTP session even if the diagnostic request failed.
                controller.close()
        # Run the downloaded model on synthetic input without a camera.
        if args.check_model:
            # Work with image arrays and numerical lookup tables.
            import numpy as np
            # Convert camera images into MediaPipe gesture observations.
            from recognition.gesture_recognition import GestureRecognizer
            # Close the native recognizer automatically after the smoke test.
            with GestureRecognizer(config.recognition) as recognizer:
                # Use a black 640x480 image; this checks inference, not hand accuracy.
                observation = recognizer.recognize(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
            # Print the returned observation as evidence that inference completed.
            print(f"CPU inference completed: {observation}")
            # Report successful completion to the calling process.
            return 0
        # Start the normal integrated camera-and-control session.
        return run(config, args)
    # Treat Ctrl+C and the translated termination signal as clean exits.
    except KeyboardInterrupt:
        # Confirm that the session was stopped by the user.
        log.info("Stopped by user")
        # Report successful completion to the calling process.
        return 0
    # Handle missing Python packages without an unhelpful full traceback.
    except ImportError as exc:
        # Explain how to install the missing or incompatible dependency.
        log.error("Missing/incompatible dependency: %s. Install requirements.txt in your virtual environment.", exc)
        # Report failure with a nonzero process exit status.
        return 1
    # Handle common setup, file, configuration, and device errors.
    except (OSError, ValueError, RuntimeError) as exc:
        # Show the original error message for troubleshooting.
        log.error("%s", exc)
        # Report failure with a nonzero process exit status.
        return 1
    # Catch unexpected SDK failures after resource cleanup has run.
    except Exception:
        # Native SDK exceptions are not consistent across versions. Always clean up.
        # Include a traceback for failures that were not anticipated above.
        log.exception("Unexpected application error")
        # Report failure with a nonzero process exit status.
        return 1


# Translate a termination signal into the normal interruption path.
def _stop(signum, frame):
    # Raise the same exception handled for Ctrl+C.
    raise KeyboardInterrupt


# Run the CLI only when this module is executed as a script.
if __name__ == "__main__":
    # Make SIGTERM use the same cleanup path as keyboard interruption.
    signal.signal(signal.SIGTERM, _stop)
    # Return the application's exit code to the operating system.
    raise SystemExit(main())
