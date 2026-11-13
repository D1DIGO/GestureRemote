# Configuration reading guide

The source files and tests contain short English comments before each meaningful
code line. Start with `main.py`, follow the camera and recognizer into
`control/gesture_logic.py`, then read `control/media_controller.py` to see how
accepted gestures become VLC requests. Test comments explain the simulated
inputs and expected outcomes. Closing brackets and existing docstrings do not
need duplicate explanations.

`config.json` stays valid JSON, which does not support comments. This table
explains each setting in the same terms as the comments in `settings.py`.
All durations are in seconds unless stated otherwise. Missing settings use
application defaults; a supplied mappings object replaces the whole mapping.
Relative model paths are resolved against the configuration file's directory.

| Setting | Repository default | Explanation |
| --- | --- | --- |
| `camera.index` | `0` | Use the first camera unless the user selects another device index. |
| `camera.width` | `1280` | Request the capture width in pixels; the driver may negotiate another size. |
| `camera.height` | `720` | Request the capture height in pixels. |
| `camera.fps` | `30` | Request a frame rate, without assuming the driver will deliver it. |
| `camera.backend` | `"auto"` | Select platform-aware backend defaults when set to auto. |
| `camera.read_timeout` | `3.0` | Bound how long the foreground loop waits for a fresh frame, in seconds. |
| `recognition.model_path` | `"models/gesture_recognizer.task"` | Resolve this relative model path against the loaded configuration file. |
| `recognition.detection_confidence` | `0.5` | Set the minimum hand-detector confidence accepted by MediaPipe. |
| `recognition.presence_confidence` | `0.5` | Set the minimum hand-presence confidence in landmark estimation. |
| `recognition.tracking_confidence` | `0.5` | Set the confidence threshold for tracking across video frames. |
| `gestures.confidence_threshold` | `0.75` | Require this category score unless a media gesture has an override. |
| `gestures.confidence_thresholds` | `{}` | Override the minimum accepted confidence for individual mapped gestures. |
| `gestures.confirmation_duration` | `0.5` | Measure normal pose confirmation in seconds, not frame counts. |
| `gestures.confirmation_durations` | `{}` | Allow longer or shorter holds for individual media gestures. |
| `gestures.cooldown` | `1.0` | Enforce a minimum interval between executions of the same action. |
| `gestures.activation_duration` | `1.2` | Use a separate deliberate hold duration for Victory activation. |
| `gestures.inactivity_timeout` | `20.0` | Disable active control after this many seconds without an accepted command. |
| `gestures.release_duration` | `0.3` | Require sustained release evidence before rearming a toggle. |
| `gestures.max_frame_gap` | `0.3` | Reject stale frames and discontinuous hold timing beyond this gap. |
| `gestures.repeat_actions` | `["volume_up", "volume_down", "seek_forward", "seek_backward"]` | Repeat volume and seek actions while held, but never playback/mute toggles. |
| `gestures.repeat_interval` | `1.0` | Space held-gesture repeats in seconds, also respecting cooldown. |
| `gestures.mappings` | See gesture table below | Map official gesture categories to supported media actions. |
| `preprocessing.enabled` | `false` | Keep raw input as the default recognition source. |
| `preprocessing.brightness` | `0` | Add this signed brightness offset before clamping pixel values. |
| `preprocessing.contrast` | `1.0` | Multiply pixel intensities by this contrast factor. |
| `preprocessing.gamma` | `1.0` | Use one for no gamma change; values above one brighten the image. |
| `preprocessing.clahe` | `false` | Optionally enhance local luminance contrast using CLAHE. |
| `preprocessing.clahe_clip_limit` | `2.0` | Limit how strongly CLAHE amplifies local contrast. |
| `preprocessing.denoise` | `false` | Optionally apply a small Gaussian blur before other enhancements. |
| `vlc.host` | `"127.0.0.1"` | Use IPv4 loopback by default instead of a network-accessible host. |
| `vlc.port` | `8080` | Connect to VLC's usual HTTP port. |
| `vlc.username` | `""` | VLC normally expects an empty HTTP Basic authentication username. |
| `vlc.password_env` | `"GESTUREREMOTE_VLC_PASSWORD"` | Store only the environment-variable name, never the password itself. |
| `vlc.timeout` | `0.7` | Bound connection/read waiting for each HTTP request, in seconds. |
| `vlc.volume_step` | `13` | Use roughly five percentage points per volume step: 256 units is 100%. |
| `vlc.seek_seconds` | `10` | Seek by this many seconds for each accepted forward/backward command. |
| `vlc.max_volume` | `256` | Cap automatic volume increases and mute restoration at this level. |
| `vlc.unmute_volume` | `128` | Restore this fallback level when starting at zero without remembered volume. |
| `preview` | `true` | Display the OpenCV preview unless configuration or CLI disables it. |
| `compare_preview` | `false` | Show two image panels only when comparison is requested. |
| `debug` | `false` | Keep normal logging concise unless debug mode is enabled. |

## Default gesture mappings

| Category | Action | While held |
| --- | --- | --- |
| `Victory` | Enable or disable control; reserved outside the mapping table | Once per hold |
| `Open_Palm` | `play_pause` | Once per hold |
| `Closed_Fist` | `mute` | Once per hold; the next released-and-confirmed fist restores volume |
| `Thumb_Up` | `volume_up` | Repeat |
| `Thumb_Down` | `volume_down` | Repeat |
| `Pointing_Up` | `seek_forward` | Repeat |
| `ILoveYou` | `seek_backward` | Repeat |

Repeats require a stable, confident pose and obey both `repeat_interval` and
`cooldown`. An empty `repeat_actions` list disables repeats. The per-gesture
confidence and confirmation overrides apply only to mapped media gestures;
Victory keeps the global confidence threshold and `activation_duration`.

VLC volume uses 256 units for 100%. `password_env` names an environment variable;
it is not the password itself. The JSON file contains no password.

For setup, validation rules, and practical behavior, see the
[README](../README.md#configuration-and-preprocessing).
