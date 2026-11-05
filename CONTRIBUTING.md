# Contributing as a six-person team

Work in small branches and review changes to shared interfaces with the module
owner. Run `python -m pytest -q` and `python -m pip check` before submitting.
Do not commit passwords, `.env`, local configuration, recordings, downloaded
models, or virtual environments. Use Python type hints and keep stateful device
operations separate from decision logic.

| Developer | Ownership | Required collaboration / checks |
| --- | --- | --- |
| 1 — Project lead | `main.py`, `settings.py`, configuration, dependencies, integration | Review contracts; verify startup/cleanup, dry-run, no-preview, and failure behavior |
| 2 — Camera | `camera/`, camera portions of tests | Verify actual resolution/backends on Windows and Linux; measure enhancement against baseline |
| 3 — Computer vision | `recognition/`, model downloader | Use only official pretrained MediaPipe; preserve RGB conversion, CPU delegate, timestamps and ambiguity behavior |
| 4 — Gesture logic | `control/gesture_logic.py`, `test_gestures.py` | Own activation, release, cooldown and timeout invariants; test with deterministic time |
| 5 — Media controller | `control/media_controller.py`, HTTP tests | Verify commands on real VLC, error messages, credentials and no-retry behavior |
| 6 — Testing | `testing/`, CI, `docs/PHYSICAL_TESTING.md`, verification record | Coordinate cross-platform and labeled physical trials; distinguish goals, mocks, inference checks, and measurements |

## Contracts

- `Camera.read(after)` returns a `CapturedFrame(image, timestamp)` newer than
  `after`, or raises `CameraError`. Images are uint8 BGR arrays. Capture keeps
  one most recent frame; it does not queue frames for future recognition.
- `GestureRecognizer.recognize(frame, timestamp)` returns an immutable
  `GestureObservation`. Times are monotonic seconds. `hand_detected=False`
  means no detected hand; `gesture=None, hand_detected=True` means an unknown or
  ambiguous hand, not a confirmed release. Do not conflate these states.
- `GestureStateMachine.update(observation)` returns zero or one `Decision`.
  Input timestamps must increase strictly. It owns no devices, threads, HTTP,
  or clocks. An accepted command consumes the hold before the HTTP call; a
  failed request must never be implicitly replayed.
- `MediaController` exposes `execute(action)`, `check_connection()`, and
  `close()`. Both operations return `CommandResult(success, message)`. A backend
  must not log secrets or use OS key simulation. Requests are bounded and
  relative/toggle actions must not be retried automatically. A future browser
  backend must implement the same protocol; add a factory only when needed.
- `main.py` owns integration and cleanup. A failed HTTP command deactivates
  control. Subsequent Victory activation checks the connection. The application
  does not send actions while disabled, in dry-run, or from stale frames.
- Defaults live in dataclasses; `config.json` exposes the full example. Keep
  them consistent. Adding a field requires validation, documentation, and an
  appropriate behavioral test. Relative paths resolve from the config file.

## State-machine review guide

`OFF → Victory held → ON → Victory held or inactivity → OFF`.
Inside either state, confirmation tracks one candidate pose and its start time.
Firing latches that pose. Only a sustained confident change or no-hand interval
releases the latch. Low confidence cancels confirmation but cannot prove
release; frame gaps reset timers and preserve the latch. Cooldown is per
**action**, preventing two differently mapped gestures from bypassing it.
Activation is independent of media mappings and cannot send a media action.

Review edge cases using synthetic timelines, not `sleep()` tests. New logic
must preserve protection against flickering detections, duplicate timestamps,
holding Victory through inactivity, and long processing/network gaps.

## Physical evidence

Follow `docs/PHYSICAL_TESTING.md`. Record OS, Python/dependency versions, camera,
CPU, VLC version, actual resolution, configuration, distance and lighting.
Keep unsuccessful trials. Report confidence separately from accuracy. Record
actual observed player effects separately from HTTP acknowledgment. Never
label a mocked or still-image test as physical webcam validation.
