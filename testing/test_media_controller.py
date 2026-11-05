from dataclasses import replace
from unittest.mock import Mock

import pytest
import requests

from control.media_controller import VLCController
from settings import VLCConfig


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setenv("GESTUREREMOTE_VLC_PASSWORD", "test-only-secret")
    session = Mock(spec=requests.Session)
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {"state": "paused", "volume": 128}
    return VLCController(VLCConfig(), session=session)


@pytest.mark.parametrize("action,params", [
    ("play_pause", {"command": "pl_pause"}),
    ("mute", {"command": "key", "val": "volume-mute"}),
    ("volume_up", {"command": "volume", "val": "+13"}),
    ("volume_down", {"command": "volume", "val": "-13"}),
    ("seek_forward", {"command": "seek", "val": "+10"}),
    ("seek_backward", {"command": "seek", "val": "-10"}),
])
def test_actual_vlc_commands(controller, action, params):
    assert controller.execute(action).success
    controller.session.get.assert_called_once_with(
        "http://127.0.0.1:8080/requests/status.json", params=params, timeout=0.7, allow_redirects=False)
    assert controller.session.auth == ("", "test-only-secret")
    assert not controller.session.trust_env


def test_status_probe_does_not_change_playback(controller):
    assert controller.check_connection().success
    assert controller.session.get.call_args.kwargs["params"] == {}


@pytest.mark.parametrize("status", [401, 403, 404, 500, 302])
def test_http_errors(controller, status):
    controller.session.get.return_value.status_code = status
    result = controller.execute("play_pause")
    assert not result.success
    assert "test-only-secret" not in result.message
    if status in (401, 403):
        assert "authentication" in result.message


@pytest.mark.parametrize("exception", [requests.Timeout("secret"), requests.ConnectionError("secret"), requests.RequestException("secret")])
def test_connection_failures_are_safe_and_never_retried(controller, exception):
    controller.session.get.side_effect = exception
    result = controller.execute("mute")
    assert not result.success
    assert "secret" not in result.message
    assert controller.session.get.call_count == 1


@pytest.mark.parametrize("payload", [[], {}, {"state": "nonsense"}, {"state": "playing", "error": "failed"}])
def test_invalid_payload(controller, payload):
    controller.session.get.return_value.json.return_value = payload
    assert not controller.execute("volume_up").success


def test_bad_json(controller):
    controller.session.get.return_value.json.side_effect = requests.exceptions.JSONDecodeError("bad", "", 0)
    assert "malformed JSON" in controller.execute("mute").message


def test_unsupported_action_makes_no_request(controller):
    assert not controller.execute("bad").success
    controller.session.get.assert_not_called()


def test_missing_password(monkeypatch):
    monkeypatch.delenv("GESTUREREMOTE_VLC_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="GESTUREREMOTE_VLC_PASSWORD"):
        VLCController(VLCConfig())


def test_ipv6_url_and_custom_steps(monkeypatch):
    monkeypatch.setenv("CUSTOM_VLC_SECRET", "test-password")
    session = Mock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {"state": "playing"}
    c = VLCController(replace(VLCConfig(), host="::1", port=9000, volume_step=26,
                              seek_seconds=20, password_env="CUSTOM_VLC_SECRET"), session=session)
    c.execute("seek_backward")
    assert c.url == "http://[::1]:9000/requests/status.json"
    assert session.get.call_args.kwargs["params"]["val"] == "-20"
    c.close()
    session.close.assert_called_once()


def test_requests_encodes_relative_plus():
    request = requests.Request("GET", "http://127.0.0.1:8080/requests/status.json",
                               params={"command": "seek", "val": "+10"}).prepare()
    assert "val=%2B10" in request.url
