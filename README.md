# GestureRemote

GestureRemote controls VLC from a laptop webcam using deliberate hand gestures.
It uses OpenCV, the official MediaPipe Gesture Recognizer on **CPU**, and VLC's
localhost HTTP interface. No keyboard focus, GPU, custom model, or training is
required. Windows 10/11 and Linux share the same core code.

**Verification:** the implementation and hardware-independent tests run on
Fedora 44 with Python 3.14.8. Real CPU inference with the official model has been
tested. Subsequent laptop checks opened the webcam at approximately 10 FPS and
verified commands against VLC 3.0.24. User testing at 0.5m identified low-confidence
palm/thumb detections; formal accuracy and dim-light trials remain outstanding.
Windows/Ubuntu hardware operation also needs validation. See [verification](docs/VERIFICATION.md) and
the [physical test procedure](docs/PHYSICAL_TESTING.md). Performance targets are
goals, not measured capabilities.

## Install

Use a **64-bit x86 Python 3.11–3.14** installation. Python 3.14 is verified locally;
the CI workflow covers 3.11 and 3.14 on Windows and Ubuntu when run. ARM and
32-bit systems are outside the verified MVP dependency setup. Install from this
directory. Use a virtual environment; do not install into system Python.

### Windows 10/11 (PowerShell)

Install 64-bit Python from [python.org](https://www.python.org/downloads/windows/)
and VLC from [VideoLAN](https://www.videolan.org/vlc/).

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe main.py --check-model
```

Use `py -3.11` instead if that is your installed version. Explicit Python paths
avoid needing to change PowerShell's execution policy to activate the venv.
Allow desktop apps to access the camera in Windows privacy settings and close
other camera applications. The automatic backend tries DirectShow, then OpenCV's
default backend. If capture fails, try `"backend": "msmf"` in `config.local.json`.

### Fedora

```bash
sudo dnf install python3 python3-pip mesa-libGL glib2 portaudio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/download_model.py
python main.py --check-model
```

Install VLC using `sudo dnf install vlc`. If your enabled repositories do not
provide the required VLC/codec packages, follow the distribution-specific
[VideoLAN Fedora instructions](https://www.videolan.org/vlc/download-fedora.html)
and [RPM Fusion configuration guide](https://rpmfusion.org/Configuration).
Use a native VLC package for the simplest HTTP setup.

### Ubuntu

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip libgl1 libglib2.0-0 libportaudio2 vlc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/download_model.py
python main.py --check-model
```

Use Ubuntu with Python 3.11 or newer (for example Ubuntu 24.04's Python 3.12).
If your default Python is older, install a supported interpreter first and use
its versioned command for `venv`. On Linux, auto capture tries V4L2 then the
default backend. Check `/dev/video*`, your session's camera permissions, the
hardware privacy shutter, and whether another application owns the webcam.
Do not run GestureRemote as root to bypass camera permissions.

For runtime-only installations, use `requirements.txt`; `requirements-dev.txt`
adds pytest. MediaPipe depends on `opencv-contrib-python`, so this project uses
that wheel alone. Do not also install `opencv-python` or a headless wheel in the
same environment: they all provide `cv2` and can conflict. For Linux preview
issues, use a working desktop session/XWayland or `--no-preview`; VLC control
does not depend on X11 input automation.

## Configure VLC once

These instructions target VLC **3.x** with its Lua HTTP interface installed.
Open a local video with audible audio and seekable content before testing.

1. Open **Tools → Preferences**, select **Show settings: All**.
2. Under **Interface → Main interfaces**, enable **Web**.
3. Under **Interface → Main interfaces → Lua**, set a nonempty **Lua HTTP
   password**. Choose your own password; never put it in this repository.
4. Save and completely exit VLC. Relaunch it bound to loopback (password is
   already saved in VLC's own preferences):

   Linux:

   ```bash
   vlc --extraintf=http --http-host=127.0.0.1 --http-port=8080
   ```

   Windows PowerShell:

   ```powershell
   & "C:\Program Files\VideoLAN\VLC\vlc.exe" --extraintf=http --http-host=127.0.0.1 --http-port=8080
   ```

   Fully exit any existing VLC instance first so it cannot ignore startup
   options. Keep the bind address on loopback; no firewall opening is needed.

5. Set the same password in the terminal that will run GestureRemote. Avoid
   putting a literal password in command history:

   Bash:

   ```bash
   read -rsp 'VLC HTTP password: ' GESTUREREMOTE_VLC_PASSWORD
   export GESTUREREMOTE_VLC_PASSWORD
   echo
   ```

   PowerShell:

   ```powershell
   $vlcSecret = Read-Host 'VLC HTTP password' -AsSecureString
   $env:GESTUREREMOTE_VLC_PASSWORD = [System.Net.NetworkCredential]::new('', $vlcSecret).Password
   Remove-Variable vlcSecret
   ```

   The process needs the password in memory; no application password is saved
   to JSON or logs. The default HTTP username is empty, as required by VLC.

6. Check the connection without changing playback:

   ```bash
   python main.py --check-vlc
   ```

   On Windows, substitute `.\.venv\Scripts\python.exe` for `python` throughout.
   You can also open [VLC's local interface](http://127.0.0.1:8080/) and enter a
   blank username and your VLC password. Authentication failure usually means
   the password differs or VLC was not restarted. Connection failure usually
   means the Web interface is disabled, the port differs, or VLC is closed.

The controller sends requests to `/requests/status.json` using Requests query
parameters (so relative `+` values are encoded correctly):

| Action | HTTP query |
| --- | --- |
| Play / pause | Read state, then `pl_forcepause`, `pl_forceresume`, or `pl_play` |
| Mute / unmute | `command=volume&val=0` / remembered previous volume |
| Volume up / down | Read volume, then set its new absolute value in steps of 13 |
| Seek forward / backward | `command=seek&val=+10` / `-10` |

Volume uses VLC units: 256 corresponds to 100%, so 13 is approximately five
percentage points. Repeated volume-up stops at `vlc.max_volume` (256, or 100%, by
default); volume-down stops at zero. Play/pause verifies the resulting state and
reports when no media is selected. Mute sets volume to zero and remembers the
previous value in memory; the next fist restores it, capped at `max_volume`.
If started at zero with no remembered value, `unmute_volume` defaults to 128
(50%). This is volume-based muting, not VLC's native mute flag; independently
muted VLC/system audio must be unmuted there. Playback and volume changes are
checked against returned status, with short read-only polling for delayed updates.
An unchanged HTTP 200 response is not reported as successful mute/playback.

Command syntax was checked against the [VLC request documentation](https://github.com/videolan/vlc/blob/3.0.x/share/lua/http/requests/README.txt),
the [VLC 3.x HTTP handler](https://github.com/videolan/vlc/blob/3.0.x/share/lua/intf/modules/httprequests.lua),
and its [action definitions](https://github.com/videolan/vlc/blob/3.0.x/src/misc/actions.c).
An HTTP acknowledgment does not prove a visible media change (for example,
seeking an unseekable stream); confirm actual effects during physical testing.

## Run

```bash
python main.py --check-config
python main.py --dry-run              # camera + recognition, no VLC/password needed
python main.py                        # normal webcam preview + VLC control
python main.py --no-preview           # controls continue without a window
```

Control starts **OFF** every time. Show only one hand, with the full hand inside
the camera view. Hold Victory for about 1.2 seconds to enable control, then
change to a command gesture and hold it for about 0.5 seconds:

| Pose | Official category | Default action |
| --- | --- | --- |
| Victory / V sign | `Victory` | Enable / disable control (reserved) |
| Open palm | `Open_Palm` | Play / pause |
| Closed fist | `Closed_Fist` | Mute / unmute |
| Thumb up | `Thumb_Up` | Volume up |
| Thumb down | `Thumb_Down` | Volume down |
| Index finger up | `Pointing_Up` | Seek forward 10 seconds |
| I love you | `ILoveYou` | Seek backward 10 seconds |

Lower the hand out of view for at least 0.3 seconds, or show a different confident
mapped gesture, before repeating play/pause or mute. **Hold volume or seek
gestures to repeat once per second**; repeats stop on a missing, changed, or
low-confidence pose and require a stable hold again after interruption.
Play/pause, mute, and Victory fire once per hold so they cannot toggle back and
forth. Set `repeat_actions` to `[]` to disable repetition, or remove selected
volume/seek actions. The interval is the greater of `repeat_interval` and
`cooldown`. Victory never sends a media command. After 20 seconds without an
accepted command, control turns off; repeated accepted commands refresh that
timer, but simply holding a nonrepeating pose does not.

The preview shows recognition, confidence, control state, last acknowledged
command, and processed FPS. A separate feedback line explains OFF status,
low confidence, hold progress, repeat countdown, and release requirements.
A recognized label alone does not mean its confidence/hold requirements passed.
**Q**, **Esc**, closing the preview, or **Ctrl+C** in
the terminal exits. Without preview, use Ctrl+C; status is logged every five
seconds and commands/state changes are logged immediately. SIGTERM also cleans
up. A broken camera ends the session instead of replaying old observations.

VLC requests have a bounded timeout and are never automatically retried. A
timeout can mean that VLC executed the command but its response was lost.
Any failed command disables gesture control. Restore VLC, release the gesture,
then hold Victory again; activation checks connectivity before accepting media
commands. Recognition remains available if VLC closes.

## Configuration and preprocessing

Copy `config.json` to the ignored `config.local.json` and run:

```bash
python main.py --config config.local.json
```

Every omitted field uses its default; provided mappings replace the entire
mapping table. Unknown keys, wrong types, invalid ranges, non-loopback hosts,
and attempts to map Victory are rejected. Model paths are relative to the
configuration file, not the shell working directory. Restart after edits.

| Section | Settings |
| --- | --- |
| `camera` | Index, width/height (1280×720), requested FPS, backend, read timeout |
| `recognition` | Model path, detection/presence/tracking thresholds (0.5) |
| `gestures` | Category confidence (0.75), confirmation (0.5s), activation (1.2s), cooldown (1s), release (0.3s), inactivity (20s), max frame gap/age (0.3s), mappings |
| `preprocessing` | Enabled, brightness offset, contrast multiplier, gamma, CLAHE/clip limit, light Gaussian denoising |
| `vlc` | Loopback host, port, username, password environment variable name, timeout, volume step, seek seconds |
| Top level | `preview`, `compare_preview`, `debug` |

`gestures.confidence_thresholds` and `confirmation_durations` accept per-category
overrides, for example `{"Open_Palm": 0.5, "Thumb_Up": 0.6}` and
`{"Open_Palm": 0.8, "Thumb_Up": 0.6}` respectively. These values were selected
for the reported laptop's scores, not validated as universal defaults. Lower
thresholds can accept more incorrect poses; evaluate them with the physical
test procedure. Victory retains the global threshold and activation duration.
`repeat_actions` contains only volume/seek actions and `repeat_interval` is in
seconds. `vlc.max_volume` caps automatic increases and `unmute_volume` defines
the fallback when there is no remembered pre-mute volume.

Processing is **off by default**. The initial enhancement values are neutral;
to evaluate an actual change, set (for example) `gamma` to `1.5` in your local
configuration. Gamma greater than one brightens, smaller than one darkens.
Brightness is added with clipping; contrast multiplies values. Optional CLAHE
acts on luminance, and denoising uses a small 3×3 Gaussian blur. The order is
denoise → contrast/brightness → gamma → CLAHE. There is no claim that these
settings improve recognition; blurred detail and amplified noise can hurt it.

```bash
python main.py --config config.local.json --dry-run --compare --no-preprocess
python main.py --config config.local.json --dry-run --compare --preprocess
```

Comparison displays original (left) and enhanced (right). **Only the selected
input is recognized**: this avoids doubling inference work. Run matched trials
with `--no-preprocess` and `--preprocess` to compare recognition accuracy and
latency, not just appearance. `--compare` does not enable preprocessing for
inference. Headless comparison adds processing cost but cannot show images.

```bash
python main.py --dry-run --duration 60 --metrics measurements/baseline.csv
```

Each metrics path must be new, to avoid overwriting trials. CSV contains
monotonic timestamps, recognized labels/confidence, hand presence, control
events, HTTP outcomes/time, processing time (preprocessing + recognition),
frame age, rolling processed FPS, and preprocessing/dry-run flags. It contains
no images or credentials. Dry-run command success is blank, never a claimed
playback success. FPS includes command delays; it is not the camera's requested
FPS. Per-frame logging and comparison preview themselves have some overhead.

## Reliability and practical limits

- A low-confidence result or unknown hand cancels an unfinished hold. A brief
  detection loss cannot rearm a fired gesture. Two detected hands are treated
  as ambiguous; use one visible hand. A second hand can still be missed by the
  model, so this is not an identity or authorization system.
- A frame gap over 0.3 seconds resets hold/release timing but preserves the
  latch. Frames already older than that threshold after processing cannot
  trigger actions. If inference is too slow, lower resolution (960×540 or
  640×480), disable enhancement/comparison, then measure again. Raising the
  frame-gap threshold trades stale-input protection for tolerance of low FPS.
- Capture continuously drains the webcam in one thread and keeps only the
  latest frame. HTTP requests run synchronously and may briefly delay the
  processing loop; they cannot build a queue of old gestures. Camera timestamps
  are host receipt times; camera exposure/driver buffering is not measurable
  through this API and adds latency.
- Exposure, sensor quality, hand size in pixels, motion blur, backlighting,
  and distance can dominate accuracy. A little diffuse front lighting, a
  closer/larger visible hand, and a clean camera lens may help more than image
  enhancement. 1–2m in dim light is explicitly an unverified target.
- Some camera drivers can block inside native `read()`. Reads presented to
  the main loop time out; shutdown waits at most one second for the capture
  thread and leaves a stuck daemon to process exit. It never calls native
  `release()` concurrently with `read()`.

## Tests and architecture

```bash
python -m pytest -q
python -m pip check
python main.py --check-model
```

Tests use synthetic observations, mocked HTTP, fake cameras and the actual
integration loop. The `model` test loads the official bundle on CPU when it is
present; otherwise it is explicitly skipped. `--check-model` does not validate
hand recognition, camera performance, or VLC. CI is configured but has not been
run on a hosted Windows/Ubuntu runner during this delivery.

```text
main.py                         CLI, resource cleanup, application loop
settings.py + config.json       Validation and defaults; no secrets
camera/camera.py                Latest-frame capture and driver errors
camera/image_processing.py      Optional deterministic enhancement
recognition/observation.py      Gesture, confidence, monotonic timestamp
recognition/gesture_recognition.py  MediaPipe VIDEO mode, CPU, result adapter
control/gesture_logic.py        Pure elapsed-time state machine
control/media_controller.py     MediaController protocol and VLC HTTP backend
preview.py                      Minimal OpenCV status window
scripts/download_model.py       Atomic model download and SHA256 verification
testing/                        pytest suite and performance recording
docs/                           Physical tests and verification record
CONTRIBUTING.md                 Six-person ownership and interface contracts
```

The official [MediaPipe Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python)
defines the VIDEO inference API. Model millisecond timestamps are strictly
increasing; gesture decisions use monotonic seconds. MediaPipe internally uses
LiteRT/TFLite for CPU inference; the project installs no separate TensorFlow
training framework. Camera frames are not saved or transmitted by GestureRemote;
only media commands go to localhost. The model download requires internet once.

See [CONTRIBUTING.md](CONTRIBUTING.md) for module contracts and team ownership.
YouTube/browser support is an extension point through `MediaController`, not
part of this MVP.
