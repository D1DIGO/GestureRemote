# Reproducible physical validation

These tests remain to be run by the team. The application has no measured claim
of reliable 1–2m control or dim-light performance. Do not change thresholds
mid-run or discard unsuccessful trials.

## 1. Equipment and environment record

Use the laptop's built-in webcam and CPU; no external camera or GPU is required.
Record date, tester, OS/version, Python, CPU, camera model, VLC version,
`python -m pip freeze`, application revision, local JSON configuration, requested
and reported capture resolution/FPS, and whether preview/comparison was enabled.
Describe lamp positions, backlighting, screen brightness and camera exposure
settings. If a light meter is already available, record lux at the hand, but
do not make additional hardware a prerequisite.

Mark distances of **0.7m, 1.5m and 2.0m** from camera lens to the plane of the
hand. Add a **1.0m** baseline to evaluate the specific 90% goal. Keep seating,
hand position, background, visible hand size and video screen brightness as
consistent as possible. Record how much of the hand is clipped or too small.

Repeat the matrix for **normal room lighting** and **dim movie-watching
lighting**, on Windows and Linux. Fedora and Ubuntu each need a smoke run;
do not infer performance on an untested laptop from another machine.

## 2. Setup and safety checks

1. Run automated tests, `--check-config`, `--check-model`, then `--check-vlc`.
2. Open a disposable local, seekable video with audio in VLC; start at a safe
   volume and away from the start/end. Keep the HTTP interface on loopback.
3. Run `python main.py --dry-run --duration 60` at 0.7m under normal lighting.
   Check the detected labels, entire hand visibility, and actual camera size.
4. Run normal mode. Verify that gestures while OFF have no playback effect.
   Hold Victory at least 1.2s, then verify one deliberate command. Repeat with
   VLC minimized or another window focused. Mute twice and confirm both the
   sound and VLC's mute indicator; seek forward/backward and inspect the time.
5. Verify preview Q/Esc/close and terminal Ctrl+C release the webcam. Verify
   `--no-preview` continues to control VLC and logs state changes.

## 3. Trial design

For each distance × lighting condition, perform **10 repetitions of every pose**
(seven poses including Victory). Randomize the command order and rotate testers
to avoid tuning only to one person's hand. Between trials lower the hand out of
view for at least 0.5s; leave at least one second between accepted commands.
Keep control active for command trials, using Victory when necessary. Test
Victory activation and deactivation separately, 10 repetitions of each.

For each command trial record:

| Trial field | Meaning |
| --- | --- |
| Trial ID, distance, lighting, preprocessing | Condition and configuration |
| Expected pose/action | Ground truth specified before presenting the hand |
| Stable-pose onset | When the complete intended pose becomes visible |
| Recognition outcome | Correct label, wrong label, no hand, or uncertain/ambiguous |
| Control decision | Correct action, wrong action, duplicate, or none |
| HTTP outcome | Acknowledged, failed, or timeout/outcome unknown |
| Observed VLC effect | Correct change, wrong change, duplicate, or no effect |
| Response latency | Stable-pose onset to visible/audible VLC effect |
| Notes | Occlusion, movement, exposure shifts, video boundary, or other failure |

Use a stopwatch/observer or a built-in desktop screen recording of the preview
beside VLC to establish pose onset and response. If screen recording is used,
record its overhead and delete it after evaluation as appropriate. These are
evaluation observations, not a training dataset. The application itself never
stores frames. For frame-level label accuracy, manually mark stable-pose time
intervals and compare their expected labels against the corresponding CSV rows;
exclude the intentional transition interval consistently across all trials.

Allow up to three seconds for a normal command trial (four for Victory), then
mark failure if nothing happens. Still include failed/late trials in success
rates. For latency summaries, report the number of successful trials separately
so failures do not disappear from the result. Reset video position as needed
so seek tests do not hit end-of-file; do not count unseekable-stream limitations
as recognition errors.

## 4. Record telemetry and compare preprocessing

Create a local configuration with a specific enhancement candidate, e.g. gamma
1.5 or CLAHE enabled. Neutral defaults do not alter the image. Start with raw
frames, then repeat the same trials with just **one** enhancement change:

```bash
python main.py --config config.local.json --no-preprocess --duration 300 --metrics measurements/raw-15m-dim-01.csv
python main.py --config config.local.json --preprocess --duration 300 --metrics measurements/gamma-15m-dim-01.csv
```

Use `--dry-run` for recognition-only comparisons. Compare the two images with
`--compare`, but leave comparison consistently enabled/disabled in performance
runs. Alternate raw/enhanced run order across testers to reduce practice and
lighting drift. Reject enhancement if command accuracy or response latency
worsens even if the image looks brighter.

CSV timestamps use the host monotonic clock, not calendar time. Record the
first timestamp for alignment with manually annotated trials. `inference_ms`
includes preprocessing; `command_ms` is only HTTP round-trip time, **not** full
gesture response latency. `frame_age_ms` excludes exposure and unobservable
driver buffering. `fps` measures completed processing including HTTP stalls.
The CSV records HTTP acknowledgment, not actual player effects.
`event` is the gesture decision; `active` is the state after the connectivity
check or command attempt. An `activated` event with `active=False` means the
connection check rejected activation, not that control became usable.

## 5. Metrics and targets

| Metric | Definition | Initial goal |
| --- | --- | --- |
| Label accuracy | Correct recognized labels / labeled stable-pose frames; count missing detections as incorrect | Report by pose and condition, plus confusion counts |
| Command success | Trials with exactly one correct observed VLC effect / all intended command trials | ≥90% at 1.0m normal; ≥80% at 1.5m dim |
| Response latency | Stable-pose onset to observed VLC effect; report median, p95 and failures | Normal commands about ≤1s; Victory intentionally ≥1.2s |
| Processing FPS | Completed frames/time, excluding startup; report median and low percentile | ≥15 on a typical laptop |
| False activation | OFF→ON transitions with no intentional Victory | Report count and count/hour; aim for zero |
| Unintended commands | Any command without an intended pose, including duplicates | Aim for zero; report count/hour |
| Detection failures | Labeled hand-visible frames with no detected hand / hand-visible frames; also trials never recognized | Report by condition |

Confidence is a model output, not measured accuracy. An undetected hand outside
a trial is normal; do not treat all no-hand frames in a session as failures.
Report activation success separately from command success. Do not mix dry-run
decisions with actual playback success.

## 6. Negative and failure trials

- Watch video normally for **10 minutes per lighting condition while OFF**:
  talk, eat, reposition and move hands naturally without intentional Victory.
  Record false activations and any unintended commands. Repeat 10 minutes
  after intentional activation (reactivate after timeout as needed), recording
  the actual active exposure time as the denominator for false-command rate.
- Hold each media pose for five seconds. Expect only one command. Briefly
  occlude/reveal the hand for less than 0.3s; expect no duplicate. Lower it for
  0.5s and repeat deliberately; expect one new command.
- Hold Victory for longer than the 20s inactivity interval. It may activate
  once and later time out; it must not repeatedly toggle or reactivate.
- Show low-confidence/partial hands and two hands. Expect no accidental
  commands; note that the model may fail to detect one of two visible hands.
- Verify no media command for 20s causes control to turn OFF. Presenting an
  unrecognized hand must not keep it ON indefinitely.
- Close VLC during an active session. The next request should fail, control
  should turn OFF, and there must be no command replay when VLC restarts.
  Restore VLC, release and hold Victory again, then issue a fresh gesture.
- Use an intentionally wrong password. Verify an authentication error and
  disabled control. Restore the password in the environment and restart.
- Start with an invalid camera index, then revoke camera access/disconnect it
  during a session where possible. Expect a clear error and orderly cleanup.
- Repeat a short run at 1280×720 and 640×480. Check whether lower processing
  cost is outweighed by fewer hand pixels at 1.5–2m.

## 7. Report limitations honestly

Create a table with one row per machine/condition, counts of trials, successes,
wrong labels/actions, duplicates, FPS, latency, detection failures and false
activations. Include raw CSV paths and exact configuration. Report a range of
performance across testers rather than only the best run.

If goals fail, first inspect lighting, motion blur, hand pixel size and camera
exposure. Try moderate diffuse front lighting or a closer hand; compare camera
resolutions and a single preprocessing method. Document the practical maximum
distance/lighting that works. Raising a confidence threshold may reduce false
commands but increase misses; lowering it may have the opposite effect. Do not
promise that software can restore detail the webcam never captured.
