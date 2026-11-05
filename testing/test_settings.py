import json
from pathlib import Path

import pytest

from settings import ConfigError, load_config


def test_shipped_configuration_and_relative_model_path():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.json")
    assert config.gestures.confidence_threshold == 0.75
    assert config.recognition.model_path == str(root / "models/gesture_recognizer.task")


def test_missing_fields_use_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera":{"width":640}}')
    config = load_config(path)
    assert config.camera.width == 640
    assert config.camera.height == 720


@pytest.mark.parametrize("data", [
    [], {"unknown": 1}, {"debug": "yes"}, {"preview": 1},
    {"camera": []}, {"camera": {"width": True}}, {"camera": {"fps": 0}},
    {"camera": {"backend": "x11"}}, {"camera": {"index": -1}},
    {"recognition": {"model_path": ""}}, {"recognition": {"tracking_confidence": 2}},
    {"gestures": {"confidence_threshold": -0.1}}, {"gestures": {"cooldown": float("nan")}},
    {"gestures": {"activation_duration": 0}}, {"gestures": {"inactivity_timeout": 1}},
    {"gestures": {"mappings": {"Victory": "play_pause"}}},
    {"gestures": {"mappings": {"Open_Palm": "launch_rocket"}}},
    {"gestures": {"mappings": {"Open_Palm": []}}},
    {"vlc": {"host": "example.com"}}, {"vlc": {"host": "192.168.1.4"}},
    {"vlc": {"port": 65536}}, {"vlc": {"password": "never-track-this"}},
    {"vlc": {"password_env": ""}}, {"preprocessing": {"gamma": 0}},
    {"preprocessing": {"enabled": 1}},
    {"gestures": {"repeat_actions": ["mute"]}},
    {"gestures": {"repeat_actions": ["play_pause"]}},
    {"gestures": {"repeat_actions": "volume_up"}},
    {"gestures": {"repeat_interval": 0}},
    {"gestures": {"confidence_thresholds": {"Victory": 0.1}}},
    {"gestures": {"confidence_thresholds": {"Open_Palm": -1}}},
    {"gestures": {"confirmation_durations": {"Open_Palm": 25}}},
    {"vlc": {"max_volume": 600}}, {"vlc": {"unmute_volume": 300}},
])
def test_invalid_configuration_has_meaningful_error(tmp_path, data):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert str(error.value)


@pytest.mark.parametrize("text", ["{", "null", "1"])
def test_malformed_json(tmp_path, text):
    path = tmp_path / "settings.json"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "absent.json")
