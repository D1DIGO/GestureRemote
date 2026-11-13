# Read or write JSON configuration data.
import json
# Handle filesystem paths without platform-specific separators.
from pathlib import Path

# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Load validated configuration types, defaults, or configuration helpers.
from settings import ConfigError, load_config


# Test: shipped configuration and relative model path.
def test_shipped_configuration_and_relative_model_path():
    # Locate the project without assuming pytest's working directory.
    root = Path(__file__).resolve().parents[1]
    # Load the same configuration shipped to users.
    config = load_config(root / "config.json")
    # Check that the documented default confidence threshold has not drifted.
    assert config.gestures.confidence_threshold == 0.75
    # Require relative model paths to resolve from the configuration directory.
    assert config.recognition.model_path == str(root / "models/gesture_recognizer.task")


# Test: missing fields use defaults.
def test_missing_fields_use_defaults(tmp_path):
    # Create an isolated partial settings file.
    path = tmp_path / "settings.json"
    # Override only width and leave all other fields absent.
    path.write_text('{"camera":{"width":640}}')
    # Load the partial configuration through the public loader.
    config = load_config(path)
    # Preserve the supplied width override.
    assert config.camera.width == 640
    # Fill the missing height from the dataclass default.
    assert config.camera.height == 720


# Run the validation test once for each malformed settings example.
@pytest.mark.parametrize("data", [
    # Reject a nonobject root, an unknown key, and incorrectly typed booleans.
    [], {"unknown": 1}, {"debug": "yes"}, {"preview": 1},
    # Reject malformed camera sections, boolean dimensions, and zero FPS.
    {"camera": []}, {"camera": {"width": True}}, {"camera": {"fps": 0}},
    # Reject an unsupported capture backend and a negative camera index.
    {"camera": {"backend": "x11"}}, {"camera": {"index": -1}},
    # Reject empty model paths and confidence scores outside zero to one.
    {"recognition": {"model_path": ""}}, {"recognition": {"tracking_confidence": 2}},
    # Reject a negative confidence threshold and a nonfinite cooldown.
    {"gestures": {"confidence_threshold": -0.1}}, {"gestures": {"cooldown": float("nan")}},
    # Reject zero activation holds and inactivity shorter than confirmation.
    {"gestures": {"activation_duration": 0}}, {"gestures": {"inactivity_timeout": 1}},
    # Prevent the reserved activation gesture from sending a media action.
    {"gestures": {"mappings": {"Victory": "play_pause"}}},
    # Reject actions that no media backend supports.
    {"gestures": {"mappings": {"Open_Palm": "launch_rocket"}}},
    # Require a mapping's action value to be a string.
    {"gestures": {"mappings": {"Open_Palm": []}}},
    # Restrict HTTP destinations to this machine's loopback addresses.
    {"vlc": {"host": "example.com"}}, {"vlc": {"host": "192.168.1.4"}},
    # Reject invalid ports and password literals in tracked configuration.
    {"vlc": {"port": 65536}}, {"vlc": {"password": "never-track-this"}},
    # Reject an empty secret-variable name and an invalid gamma exponent.
    {"vlc": {"password_env": ""}}, {"preprocessing": {"gamma": 0}},
    # Do not accept integers as enhancement on/off flags.
    {"preprocessing": {"enabled": 1}},
    # Do not allow mute to toggle repeatedly during one hold.
    {"gestures": {"repeat_actions": ["mute"]}},
    # Do not allow play/pause to toggle repeatedly during one hold.
    {"gestures": {"repeat_actions": ["play_pause"]}},
    # Require repeat_actions to be a list rather than a single string.
    {"gestures": {"repeat_actions": "volume_up"}},
    # Reject zero spacing between repeated commands.
    {"gestures": {"repeat_interval": 0}},
    # Do not weaken activation through a per-media-gesture override.
    {"gestures": {"confidence_thresholds": {"Victory": 0.1}}},
    # Validate the numeric range of per-gesture confidence overrides.
    {"gestures": {"confidence_thresholds": {"Open_Palm": -1}}},
    # Reject a per-pose hold longer than the configured inactivity timeout.
    {"gestures": {"confirmation_durations": {"Open_Palm": 25}}},
    # Validate the volume ceiling and keep the unmute fallback within it.
    {"vlc": {"max_volume": 600}}, {"vlc": {"unmute_volume": 300}},
])
# Test: invalid configuration has meaningful error.
def test_invalid_configuration_has_meaningful_error(tmp_path, data):
    # Create a separate settings file for each invalid input case.
    path = tmp_path / "settings.json"
    # Serialize this case so the actual JSON loader and validator are exercised.
    path.write_text(json.dumps(data))
    # Expect a configuration-specific error, not an unrelated exception.
    with pytest.raises(ConfigError) as error:
        # Attempt to load the intentionally invalid settings.
        load_config(path)
    # Require an explanatory error message rather than an empty exception.
    assert str(error.value)


# Cover malformed JSON syntax and valid JSON with invalid root types.
@pytest.mark.parametrize("text", ["{", "null", "1"])
# Test: malformed json.
def test_malformed_json(tmp_path, text):
    # Choose an isolated path for the malformed document.
    path = tmp_path / "settings.json"
    # Write the case exactly, without JSON serialization repairing its syntax.
    path.write_text(text)
    # Both parsing and root-type failures should be reported as ConfigError.
    with pytest.raises(ConfigError):
        # Run the malformed document through the real configuration loader.
        load_config(path)


# Test: missing file.
def test_missing_file(tmp_path):
    # Require a useful read-error message for a nonexistent file.
    with pytest.raises(ConfigError, match="Cannot read"):
        # Attempt to load a path that has not been created.
        load_config(tmp_path / "absent.json")
