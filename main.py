"""GestureRemote CLI and integration loop. Run python main.py --help."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import logging
import math
import os
from pathlib import Path
import signal
import sys
import time

from control.gesture_logic import GestureStateMachine
from recognition.observation import GestureObservation
from settings import AppConfig, load_config

log = logging.getLogger("gestureremote")
ROOT = Path(__file__).resolve().parent


def positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("duration must be a finite positive number")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control VLC using deliberate, held hand gestures.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--no-preview", action="store_true", help="run without an OpenCV window; Ctrl+C exits")
    parser.add_argument("--compare", action="store_true", help="show original and enhanced frames side by side")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preprocess", action="store_true", help="use configured enhancement for inference")
    group.add_argument("--no-preprocess", action="store_true", help="use raw frames for inference")
    parser.add_argument("--dry-run", action="store_true", help="recognize and log decisions without contacting VLC")
    parser.add_argument("--metrics", type=Path, help="write per-frame measurements to a NEW CSV file")
    parser.add_argument("--duration", type=positive_seconds, help="exit after this many seconds")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-config", action="store_true", help="validate configuration without loading native dependencies")
    checks.add_argument("--check-vlc", action="store_true", help="read VLC status once and exit; no playback changes")
    checks.add_argument("--check-model", action="store_true", help="run CPU inference on a blank frame; no camera or VLC")
    return parser.parse_args(argv)


def run(config: AppConfig, args: argparse.Namespace) -> int:
    from camera.camera import Camera
    from camera.image_processing import ImageProcessor
    from control.media_controller import VLCController
    from preview import Preview
    from recognition.gesture_recognition import GestureRecognizer
    from testing.performance_monitor import MetricsRecorder, PerformanceMonitor

    machine = GestureStateMachine(config.gestures)
    processor = ImageProcessor(config.preprocessing)
    monitor = PerformanceMonitor()
    preview_enabled = config.preview
    if preview_enabled and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log.warning("No display session detected; continuing without preview (Ctrl+C to exit)")
        preview_enabled = False
    with ExitStack() as resources:
        controller = None
        status = "Dry run: media commands are only logged"
        if not args.dry_run:
            controller = VLCController(config.vlc)
            resources.callback(controller.close)
            result = controller.check_connection()
            status = result.message
            (log.info if result.success else log.warning)(status)
        recorder = MetricsRecorder(args.metrics) if args.metrics else None
        if recorder:
            resources.callback(recorder.close)
        recognizer = resources.enter_context(GestureRecognizer(config.recognition))
        camera = resources.enter_context(Camera(config.camera))
        preview = Preview() if preview_enabled else None
        if preview:
            resources.callback(preview.close)
        last_action = "none"
        previous_frame = -1.0
        start = time.monotonic()
        next_report = start + 5
        log.info("Control OFF. Hold Victory %.1fs to enable. Exit: Q/Esc in preview, or Ctrl+C in terminal.", config.gestures.activation_duration)
        while args.duration is None or time.monotonic() - start < args.duration:
            captured = camera.read(after=previous_frame)
            previous_frame = captured.timestamp
            inference_start = time.monotonic()
            enhanced = processor.process(captured.image, force=config.compare_preview)
            inference_frame = enhanced if config.preprocessing.enabled else captured.image
            observation = recognizer.recognize(inference_frame, captured.timestamp)
            now = time.monotonic()
            inference_ms = (now - inference_start) * 1000
            frame_age_ms = (now - captured.timestamp) * 1000
            if frame_age_ms > config.gestures.max_frame_gap * 1000:
                # An old frame cannot confirm a current gesture or prove release.
                observation = GestureObservation(None, 0, captured.timestamp, hand_detected=True)
                status = "Frame too old; lower resolution or disable enhancement"
            decision = machine.update(observation)
            command_success: bool | str = ""
            command_ms: float | str = ""
            if decision is not None:
                if decision.kind == "activated":
                    if controller:
                        result = controller.check_connection()
                        status = result.message
                        if not result.success:
                            machine.deactivate()
                            status += "; control OFF, release and hold Victory to retry"
                    log.info("Control %s: %s", "ON" if machine.active else "OFF", status)
                elif decision.kind in ("deactivated", "timeout"):
                    status = "Control OFF: " + decision.kind
                    log.info(status)
                elif decision.kind == "command":
                    if controller is None:
                        last_action = f"would execute {decision.action}"
                        status = "Dry run: nothing sent to VLC"
                        log.info(last_action)
                    else:
                        command_start = time.monotonic()
                        result = controller.execute(decision.action)
                        command_ms = (time.monotonic() - command_start) * 1000
                        command_success = result.success
                        status = result.message
                        if result.success:
                            last_action = decision.action
                            log.info("Command %s: %s", decision.action, status)
                        else:
                            machine.deactivate()
                            status += "; control OFF"
                            log.warning("Command %s: %s. Release and hold Victory to retry.", decision.action, status)
            monitor.record(time.monotonic(), inference_ms, observation.hand_detected)
            if recorder:
                recorder.write(timestamp=captured.timestamp, gesture=observation.gesture or "",
                               confidence=observation.confidence, hand_detected=observation.hand_detected,
                               active=machine.active, event=decision.kind if decision else "",
                               action=decision.action if decision else "", command_success=command_success,
                               command_ms=command_ms, inference_ms=inference_ms, frame_age_ms=frame_age_ms,
                               fps=monitor.fps, preprocessing=config.preprocessing.enabled, dry_run=args.dry_run)
            if preview and not preview.show(
                captured.image if config.compare_preview else inference_frame,
                enhanced if config.compare_preview else None, observation,
                active=machine.active, last_action=last_action, status=status, fps=monitor.fps,
                preprocessing=config.preprocessing.enabled, dry_run=args.dry_run,
            ):
                break
            if time.monotonic() >= next_report:
                log.info("Control %s | %s %.2f | %.1f FPS | processing %.1f ms | last: %s",
                         "ON" if machine.active else "OFF", observation.gesture or "None", observation.confidence,
                         monitor.fps, monitor.mean_inference_ms, last_action)
                next_report = time.monotonic() + 5
        log.info("Stopped after %d frames; no-hand frames: %d (not a labeled accuracy measurement)",
                 monitor.frames, monitor.no_hand_frames)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
        logging.getLogger().setLevel(logging.DEBUG if config.debug else logging.INFO)
        config = replace(config, preview=config.preview and not args.no_preview,
                         compare_preview=config.compare_preview or args.compare)
        if args.preprocess or args.no_preprocess:
            config = replace(config, preprocessing=replace(config.preprocessing, enabled=args.preprocess))
        if args.check_config:
            print("Configuration valid")
            return 0
        if args.check_vlc:
            from control.media_controller import VLCController
            controller = VLCController(config.vlc)
            try:
                result = controller.check_connection()
                print(result.message)
                return 0 if result.success else 1
            finally:
                controller.close()
        if args.check_model:
            import numpy as np
            from recognition.gesture_recognition import GestureRecognizer
            with GestureRecognizer(config.recognition) as recognizer:
                observation = recognizer.recognize(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
            print(f"CPU inference completed: {observation}")
            return 0
        return run(config, args)
    except KeyboardInterrupt:
        log.info("Stopped by user")
        return 0
    except ImportError as exc:
        log.error("Missing/incompatible dependency: %s. Install requirements.txt in your virtual environment.", exc)
        return 1
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    except Exception:
        # Native SDK exceptions are not consistent across versions. Always clean up.
        log.exception("Unexpected application error")
        return 1


def _stop(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _stop)
    raise SystemExit(main())
