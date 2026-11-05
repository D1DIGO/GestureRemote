"""Validated JSON settings. Secrets are read separately by the VLC controller."""

from dataclasses import dataclass, field, fields
import ipaddress
import json
import math
from pathlib import Path
from typing import Any

GESTURES = {"Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down", "Pointing_Up", "ILoveYou"}
ACTIONS = {"play_pause", "mute", "volume_up", "volume_down", "seek_forward", "seek_backward"}
DEFAULT_MAPPINGS = dict(zip(
    ["Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down", "Pointing_Up", "ILoveYou"],
    ["play_pause", "mute", "volume_up", "volume_down", "seek_forward", "seek_backward"],
))


class ConfigError(ValueError):
    """A setting could not be loaded or is unsafe/invalid."""


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"
    read_timeout: float = 3.0


@dataclass(frozen=True)
class RecognitionConfig:
    model_path: str = "models/gesture_recognizer.task"
    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5


@dataclass(frozen=True)
class GestureConfig:
    confidence_threshold: float = 0.75
    confidence_thresholds: dict[str, float] = field(default_factory=dict)
    confirmation_duration: float = 0.5
    confirmation_durations: dict[str, float] = field(default_factory=dict)
    cooldown: float = 1.0
    activation_duration: float = 1.2
    inactivity_timeout: float = 20.0
    release_duration: float = 0.3
    max_frame_gap: float = 0.3
    repeat_actions: list[str] = field(default_factory=lambda: ["volume_up", "volume_down", "seek_forward", "seek_backward"])
    repeat_interval: float = 1.0
    mappings: dict[str, str] = field(default_factory=lambda: DEFAULT_MAPPINGS.copy())


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = False
    brightness: float = 0
    contrast: float = 1.0
    gamma: float = 1.0
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    denoise: bool = False


@dataclass(frozen=True)
class VLCConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    username: str = ""
    password_env: str = "GESTUREREMOTE_VLC_PASSWORD"
    timeout: float = 0.7
    volume_step: int = 13
    seek_seconds: int = 10
    max_volume: int = 256
    unmute_volume: int = 128


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vlc: VLCConfig = field(default_factory=VLCConfig)
    preview: bool = True
    compare_preview: bool = False
    debug: bool = False


def _number(name: str, value: Any, low: float, high: float, *, integer: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high
            or (integer and not isinstance(value, int))):
        raise ConfigError(f"{name} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")


def _construct(cls: type, data: Any, name: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a JSON object")
    unknown = data.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown settings in {name}: {', '.join(sorted(unknown))}")
    return cls(**data)


def validate(config: AppConfig) -> None:
    for name in ("preview", "compare_preview", "debug"):
        if type(getattr(config, name)) is not bool:
            raise ConfigError(f"{name} must be true or false")
    c = config.camera
    for name, lo, hi in [("index", 0, 99), ("width", 160, 7680), ("height", 120, 4320), ("fps", 1, 240)]:
        _number(f"camera.{name}", getattr(c, name), lo, hi, integer=True)
    if c.backend not in ("auto", "any", "dshow", "msmf", "v4l2"):
        raise ConfigError("camera.backend must be auto, any, dshow, msmf or v4l2")
    _number("camera.read_timeout", c.read_timeout, 0.1, 30)
    r = config.recognition
    if not isinstance(r.model_path, str) or not r.model_path.strip():
        raise ConfigError("recognition.model_path must be a nonempty path")
    for name in ("detection_confidence", "presence_confidence", "tracking_confidence"):
        _number(f"recognition.{name}", getattr(r, name), 0, 1)
    g = config.gestures
    _number("gestures.confidence_threshold", g.confidence_threshold, 0, 1)
    for name, low, high in (("confidence_thresholds", 0, 1), ("confirmation_durations", 0.01, 30)):
        overrides = getattr(g, name)
        if not isinstance(overrides, dict):
            raise ConfigError(f"gestures.{name} must be an object")
        for gesture, value in overrides.items():
            if gesture not in GESTURES:
                raise ConfigError(f"gestures.{name}: unknown or reserved gesture {gesture}")
            _number(f"gestures.{name}.{gesture}", value, low, high)
    repeatable = {"volume_up", "volume_down", "seek_forward", "seek_backward"}
    if not isinstance(g.repeat_actions, list) or any(not isinstance(a, str) or a not in repeatable for a in g.repeat_actions):
        raise ConfigError("gestures.repeat_actions may only contain volume and seek actions; toggles cannot repeat")
    _number("gestures.repeat_interval", g.repeat_interval, 0.1, 60)
    for name in ("confirmation_duration", "activation_duration", "release_duration", "max_frame_gap"):
        _number(f"gestures.{name}", getattr(g, name), 0.01, 30)
    _number("gestures.cooldown", g.cooldown, 0, 60)
    _number("gestures.inactivity_timeout", g.inactivity_timeout, 1, 3600)
    if g.inactivity_timeout <= max(g.activation_duration, g.confirmation_duration, *g.confirmation_durations.values()):
        raise ConfigError("gestures.inactivity_timeout must exceed hold durations")
    if not isinstance(g.mappings, dict):
        raise ConfigError("gestures.mappings must be an object")
    for gesture, action in g.mappings.items():
        if gesture not in GESTURES or not isinstance(action, str) or action not in ACTIONS:
            raise ConfigError(f"Invalid gesture mapping: {gesture!r} -> {action!r}; Victory is reserved")
    p = config.preprocessing
    for name in ("enabled", "clahe", "denoise"):
        if type(getattr(p, name)) is not bool:
            raise ConfigError(f"preprocessing.{name} must be true or false")
    for name, lo, hi in [("brightness", -255, 255), ("contrast", 0.1, 4), ("gamma", 0.1, 5), ("clahe_clip_limit", 0.1, 20)]:
        _number(f"preprocessing.{name}", getattr(p, name), lo, hi)
    v = config.vlc
    if not isinstance(v.host, str):
        raise ConfigError("vlc.host must be localhost or a loopback IP address")
    if v.host != "localhost":
        try:
            if not ipaddress.ip_address(v.host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ConfigError("vlc.host must be localhost or a loopback IP address") from exc
    _number("vlc.port", v.port, 1, 65535, integer=True)
    _number("vlc.timeout", v.timeout, 0.05, 10)
    _number("vlc.volume_step", v.volume_step, 1, 256, integer=True)
    _number("vlc.seek_seconds", v.seek_seconds, 1, 3600, integer=True)
    _number("vlc.max_volume", v.max_volume, 1, 512, integer=True)
    _number("vlc.unmute_volume", v.unmute_volume, 1, v.max_volume, integer=True)
    if not isinstance(v.username, str):
        raise ConfigError("vlc.username must be a string")
    if not isinstance(v.password_env, str) or not v.password_env.isidentifier():
        raise ConfigError("vlc.password_env must name an environment variable")


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be an object")
    sections = {
        "camera": CameraConfig, "recognition": RecognitionConfig,
        "gestures": GestureConfig, "preprocessing": PreprocessingConfig, "vlc": VLCConfig,
    }
    values = dict(data)
    for name, cls in sections.items():
        values[name] = _construct(cls, data.get(name, {}), name)
    config = _construct(AppConfig, values, "configuration")
    validate(config)
    # A config can be used from any working directory.
    from dataclasses import replace
    model_path = Path(config.recognition.model_path)
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    return replace(config, recognition=replace(config.recognition, model_path=str(model_path)))
