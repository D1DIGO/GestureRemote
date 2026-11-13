"""Validated JSON settings. Secrets are read separately by the VLC controller."""

# Define structured records or create modified copies of configuration records.
from dataclasses import dataclass, field, fields
# Import ipaddress for use in this module.
import ipaddress
# Read or write JSON configuration data.
import json
# Validate finite numbers before using them in timing or volume calculations.
import math
# Handle filesystem paths without platform-specific separators.
from pathlib import Path
# Describe flexible status values and the media-controller interface with type hints.
from typing import Any

# List supported media-control categories; Victory is reserved for activation.
GESTURES = {"Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down", "Pointing_Up", "ILoveYou"}
# Restrict mappings to operations implemented by the media-controller interface.
ACTIONS = {"play_pause", "mute", "volume_up", "volume_down", "seek_forward", "seek_backward"}
# Pair each default category with its corresponding media action.
DEFAULT_MAPPINGS = dict(zip(
    # Keep official MediaPipe category names in their mapping order.
    ["Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down", "Pointing_Up", "ILoveYou"],
    # Assign one default media action to each category above.
    ["play_pause", "mute", "volume_up", "volume_down", "seek_forward", "seek_backward"],
))


# Use a dedicated exception for invalid or unreadable configuration.
class ConfigError(ValueError):
    """A setting could not be loaded or is unsafe/invalid."""


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Collect webcam preferences independently of recognition settings.
class CameraConfig:
    # Use the first camera unless the user selects another device index.
    index: int = 0
    # Request the capture width in pixels; the driver may negotiate another size.
    width: int = 1280
    # Request the capture height in pixels.
    height: int = 720
    # Request a frame rate, without assuming the driver will deliver it.
    fps: int = 30
    # Select platform-aware backend defaults when set to auto.
    backend: str = "auto"
    # Bound how long the foreground loop waits for a fresh frame, in seconds.
    read_timeout: float = 3.0


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Keep model location and MediaPipe's hand-tracking thresholds together.
class RecognitionConfig:
    # Resolve this relative model path against the loaded configuration file.
    model_path: str = "models/gesture_recognizer.task"
    # Set the minimum hand-detector confidence accepted by MediaPipe.
    detection_confidence: float = 0.5
    # Set the minimum hand-presence confidence in landmark estimation.
    presence_confidence: float = 0.5
    # Set the confidence threshold for tracking across video frames.
    tracking_confidence: float = 0.5


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Define gesture acceptance, repetition, activation, and release behavior.
class GestureConfig:
    # Require this category score unless a media gesture has an override.
    confidence_threshold: float = 0.75
    # Give each configuration its own dictionary of per-gesture confidence limits.
    confidence_thresholds: dict[str, float] = field(default_factory=dict)
    # Measure normal pose confirmation in seconds, not frame counts.
    confirmation_duration: float = 0.5
    # Allow longer or shorter holds for individual media gestures.
    confirmation_durations: dict[str, float] = field(default_factory=dict)
    # Enforce a minimum interval between executions of the same action.
    cooldown: float = 1.0
    # Use a separate deliberate hold duration for Victory activation.
    activation_duration: float = 1.2
    # Disable active control after this many seconds without an accepted command.
    inactivity_timeout: float = 20.0
    # Require sustained release evidence before rearming a toggle.
    release_duration: float = 0.3
    # Reject stale frames and discontinuous hold timing beyond this gap.
    max_frame_gap: float = 0.3
    # Repeat volume and seek actions while held, but never playback/mute toggles.
    repeat_actions: list[str] = field(default_factory=lambda: ["volume_up", "volume_down", "seek_forward", "seek_backward"])
    # Space held-gesture repeats in seconds, also respecting cooldown.
    repeat_interval: float = 1.0
    # Copy the mapping table so configuration instances do not share mutations.
    mappings: dict[str, str] = field(default_factory=lambda: DEFAULT_MAPPINGS.copy())


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Configure optional image enhancement without enabling it by default.
class PreprocessingConfig:
    # Keep raw input as the default recognition source.
    enabled: bool = False
    # Add this signed brightness offset before clamping pixel values.
    brightness: float = 0
    # Multiply pixel intensities by this contrast factor.
    contrast: float = 1.0
    # Use one for no gamma change; values above one brighten the image.
    gamma: float = 1.0
    # Optionally enhance local luminance contrast using CLAHE.
    clahe: bool = False
    # Limit how strongly CLAHE amplifies local contrast.
    clahe_clip_limit: float = 2.0
    # Optionally apply a small Gaussian blur before other enhancements.
    denoise: bool = False


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Describe VLC's local HTTP endpoint and audio/seek step sizes.
class VLCConfig:
    # Use IPv4 loopback by default instead of a network-accessible host.
    host: str = "127.0.0.1"
    # Connect to VLC's usual HTTP port.
    port: int = 8080
    # VLC normally expects an empty HTTP Basic authentication username.
    username: str = ""
    # Store only the environment-variable name, never the password itself.
    password_env: str = "GESTUREREMOTE_VLC_PASSWORD"
    # Bound connection/read waiting for each HTTP request, in seconds.
    timeout: float = 0.7
    # Use roughly five percentage points per volume step: 256 units is 100%.
    volume_step: int = 13
    # Seek by this many seconds for each accepted forward/backward command.
    seek_seconds: int = 10
    # Cap automatic volume increases and mute restoration at this level.
    max_volume: int = 256
    # Restore this fallback level when starting at zero without remembered volume.
    unmute_volume: int = 128


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Group all validated application settings into one immutable container.
class AppConfig:
    # Create independent default camera settings for each application config.
    camera: CameraConfig = field(default_factory=CameraConfig)
    # Create independent default model and tracking settings.
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    # Create independent gesture timing, threshold, and mapping settings.
    gestures: GestureConfig = field(default_factory=GestureConfig)
    # Create independent optional enhancement settings.
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    # Create independent VLC connection and action settings.
    vlc: VLCConfig = field(default_factory=VLCConfig)
    # Display the OpenCV preview unless configuration or CLI disables it.
    preview: bool = True
    # Show two image panels only when comparison is requested.
    compare_preview: bool = False
    # Keep normal logging concise unless debug mode is enabled.
    debug: bool = False


# Validate numeric type, finiteness, range, and optional integer-only values.
def _number(name: str, value: Any, low: float, high: float, *, integer: bool = False) -> None:
    # Reject booleans explicitly because Python otherwise treats them as integers.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            # Reject NaN, infinity, and values outside the allowed inclusive interval.
            or not math.isfinite(value) or not low <= value <= high
            # Enforce an actual integer where fractions are not meaningful.
            or (integer and not isinstance(value, int))):
        # Report the setting name and its allowed numeric range.
        raise ConfigError(f"{name} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")


# Build a configuration dataclass while rejecting misspelled or extra keys.
def _construct(cls: type, data: Any, name: str) -> Any:
    # Each settings section must be a JSON object, not a list or scalar.
    if not isinstance(data, dict):
        # Identify the section whose structure is invalid.
        raise ConfigError(f"{name} must be a JSON object")
    # Compare supplied keys with the dataclass's declared field names.
    unknown = data.keys() - {f.name for f in fields(cls)}
    # Do not silently ignore unknown settings.
    if unknown:
        # List unknown keys in a stable order for a useful error message.
        raise ConfigError(f"Unknown settings in {name}: {', '.join(sorted(unknown))}")
    # Construct the dataclass; omitted fields retain their declared defaults.
    return cls(**data)


# Check every configuration section before devices and networking are used.
def validate(config: AppConfig) -> None:
    # Validate the top-level runtime flags together.
    for name in ("preview", "compare_preview", "debug"):
        # Require real JSON booleans rather than truthy strings or integers.
        if type(getattr(config, name)) is not bool:
            # Name the incorrectly typed flag.
            raise ConfigError(f"{name} must be true or false")
    # Use a short local reference while validating camera settings.
    c = config.camera
    # Define supported integer ranges for device index, dimensions, and FPS.
    for name, lo, hi in [("index", 0, 99), ("width", 160, 7680), ("height", 120, 4320), ("fps", 1, 240)]:
        # Validate each camera value using the shared numeric checker.
        _number(f"camera.{name}", getattr(c, name), lo, hi, integer=True)
    # Reject backend names the camera module cannot translate.
    if c.backend not in ("auto", "any", "dshow", "msmf", "v4l2"):
        # Show the accepted backend identifiers.
        raise ConfigError("camera.backend must be auto, any, dshow, msmf or v4l2")
    # Require a bounded positive wait for camera frames.
    _number("camera.read_timeout", c.read_timeout, 0.1, 30)
    # Validate model settings next.
    r = config.recognition
    # Require a nonempty textual model path.
    if not isinstance(r.model_path, str) or not r.model_path.strip():
        # Report a missing or invalid path before loading the native model.
        raise ConfigError("recognition.model_path must be a nonempty path")
    # Check each MediaPipe confidence setting consistently.
    for name in ("detection_confidence", "presence_confidence", "tracking_confidence"):
        # Confidence values must lie between zero and one.
        _number(f"recognition.{name}", getattr(r, name), 0, 1)
    # Validate decision-state-machine settings separately from MediaPipe settings.
    g = config.gestures
    # Check the shared gesture-category confidence threshold.
    _number("gestures.confidence_threshold", g.confidence_threshold, 0, 1)
    # Assign different allowed ranges to per-pose scores and hold durations.
    for name, low, high in (("confidence_thresholds", 0, 1), ("confirmation_durations", 0.01, 30)):
        # Read the currently selected override dictionary.
        overrides = getattr(g, name)
        # Require overrides to map category names to numeric values.
        if not isinstance(overrides, dict):
            # Explain which override table has the wrong type.
            raise ConfigError(f"gestures.{name} must be an object")
        # Validate every supplied category override.
        for gesture, value in overrides.items():
            # Only known media poses may override these settings; Victory remains reserved.
            if gesture not in GESTURES:
                # Reject unknown categories or attempts to override activation this way.
                raise ConfigError(f"gestures.{name}: unknown or reserved gesture {gesture}")
            # Check this override against its threshold or duration bounds.
            _number(f"gestures.{name}.{gesture}", value, low, high)
    # Define actions that can repeat safely without toggling themselves back.
    repeatable = {"volume_up", "volume_down", "seek_forward", "seek_backward"}
    # Require a list containing only supported volume and seek actions.
    if not isinstance(g.repeat_actions, list) or any(not isinstance(a, str) or a not in repeatable for a in g.repeat_actions):
        # Explain why play/pause and mute cannot be configured as repeating toggles.
        raise ConfigError("gestures.repeat_actions may only contain volume and seek actions; toggles cannot repeat")
    # Bound repeat spacing to a positive, practical interval.
    _number("gestures.repeat_interval", g.repeat_interval, 0.1, 60)
    # Apply common positive-duration validation to hold and release timers.
    for name in ("confirmation_duration", "activation_duration", "release_duration", "max_frame_gap"):
        # Reject zero or excessive timing values.
        _number(f"gestures.{name}", getattr(g, name), 0.01, 30)
    # Allow zero cooldown but reject negative spacing.
    _number("gestures.cooldown", g.cooldown, 0, 60)
    # Bound the inactivity timer independently of confirmation times.
    _number("gestures.inactivity_timeout", g.inactivity_timeout, 1, 3600)
    # Ensure active control lasts longer than any configured pose confirmation.
    if g.inactivity_timeout <= max(g.activation_duration, g.confirmation_duration, *g.confirmation_durations.values()):
        # Explain a contradictory timeout and hold-duration configuration.
        raise ConfigError("gestures.inactivity_timeout must exceed hold durations")
    # Require mappings to be a JSON object.
    if not isinstance(g.mappings, dict):
        # Reject a list or scalar used in place of gesture/action mappings.
        raise ConfigError("gestures.mappings must be an object")
    # Validate both sides of every category-to-action mapping.
    for gesture, action in g.mappings.items():
        # Reject unknown categories, nonstring actions, and unsupported operations.
        if gesture not in GESTURES or not isinstance(action, str) or action not in ACTIONS:
            # Show the invalid pair and remind the user that Victory is reserved.
            raise ConfigError(f"Invalid gesture mapping: {gesture!r} -> {action!r}; Victory is reserved")
    # Validate optional enhancement settings next.
    p = config.preprocessing
    # Group the three enhancement on/off flags.
    for name in ("enabled", "clahe", "denoise"):
        # Accept only actual booleans for these flags.
        if type(getattr(p, name)) is not bool:
            # Report the incorrectly typed enhancement option.
            raise ConfigError(f"preprocessing.{name} must be true or false")
    # Define bounds for brightness, contrast, gamma, and CLAHE strength.
    for name, lo, hi in [("brightness", -255, 255), ("contrast", 0.1, 4), ("gamma", 0.1, 5), ("clahe_clip_limit", 0.1, 20)]:
        # Check the selected enhancement parameter against its bounds.
        _number(f"preprocessing.{name}", getattr(p, name), lo, hi)
    # Validate VLC connection and command settings last.
    v = config.vlc
    # Require a textual host before checking its address type.
    if not isinstance(v.host, str):
        # Explain the accepted localhost/loopback-only destination policy.
        raise ConfigError("vlc.host must be localhost or a loopback IP address")
    # Accept the conventional localhost name without parsing it as an IP literal.
    if v.host != "localhost":
        # Handle failures from the operations in this block.
        try:
            # Parse a literal address and require it to refer to this machine.
            if not ipaddress.ip_address(v.host).is_loopback:
                # Send non-loopback addresses through the same configuration error path.
                raise ValueError
        # Handle both malformed IP addresses and valid but non-loopback addresses.
        except ValueError as exc:
            # Reject remote destinations before any credentials can be sent to them.
            raise ConfigError("vlc.host must be localhost or a loopback IP address") from exc
    # Require an integer port in the standard TCP port range.
    _number("vlc.port", v.port, 1, 65535, integer=True)
    # Require a short positive timeout for each HTTP request.
    _number("vlc.timeout", v.timeout, 0.05, 10)
    # Bound volume step size in VLC's native units.
    _number("vlc.volume_step", v.volume_step, 1, 256, integer=True)
    # Require a positive integer number of seek seconds.
    _number("vlc.seek_seconds", v.seek_seconds, 1, 3600, integer=True)
    # Bound the configured maximum volume, including optional amplification.
    _number("vlc.max_volume", v.max_volume, 1, 512, integer=True)
    # Keep the unmute fallback positive and no higher than the maximum.
    _number("vlc.unmute_volume", v.unmute_volume, 1, v.max_volume, integer=True)
    # Reject nonstring HTTP usernames.
    if not isinstance(v.username, str):
        # Explain the required username type.
        raise ConfigError("vlc.username must be a string")
    # Require a nonempty identifier for the password environment variable.
    if not isinstance(v.password_env, str) or not v.password_env.isidentifier():
        # Reject values that cannot serve as the expected variable name.
        raise ConfigError("vlc.password_env must name an environment variable")


# Read JSON, construct defaults, validate settings, and resolve the model path.
def load_config(path: str | Path) -> AppConfig:
    # Normalize the config path so relative model paths have a stable base.
    path = Path(path).resolve()
    # Handle failures from the operations in this block.
    try:
        # Read UTF-8 text and parse it as JSON.
        data = json.loads(path.read_text(encoding="utf-8"))
    # Handle filesystem failures and malformed JSON uniformly.
    except (OSError, ValueError) as exc:
        # Include the failing path while preserving the underlying exception chain.
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    # Require the top-level JSON document to be an object.
    if not isinstance(data, dict):
        # Reject a list, null, or scalar as the configuration root.
        raise ConfigError("Configuration root must be an object")
    # Associate each JSON section with its corresponding dataclass.
    sections = {
        # Map capture and model settings to their typed containers.
        "camera": CameraConfig, "recognition": RecognitionConfig,
        # Map decision, image-processing, and VLC settings to their containers.
        "gestures": GestureConfig, "preprocessing": PreprocessingConfig, "vlc": VLCConfig,
    }
    # Copy raw values before replacing section dictionaries with dataclasses.
    values = dict(data)
    # Construct every known section, even if it was omitted from JSON.
    for name, cls in sections.items():
        # Use defaults for missing sections and reject unknown fields within each one.
        values[name] = _construct(cls, data.get(name, {}), name)
    # Build the complete application configuration and reject unknown top-level keys.
    config = _construct(AppConfig, values, "configuration")
    # Run all type, range, and cross-field checks before returning.
    validate(config)
    # A config can be used from any working directory.
    # Define structured records or create modified copies of configuration records.
    from dataclasses import replace
    # Turn the model-path string into a platform-independent path object.
    model_path = Path(config.recognition.model_path)
    # Leave already-absolute model paths unchanged.
    if not model_path.is_absolute():
        # Resolve relative model paths beside the configuration file.
        model_path = path.parent / model_path
    # Return a new immutable config containing the resolved model path.
    return replace(config, recognition=replace(config.recognition, model_path=str(model_path)))
