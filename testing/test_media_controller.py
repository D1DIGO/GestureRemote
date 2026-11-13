# Define structured records or create modified copies of configuration records.
from dataclasses import replace
# Replace hardware or network dependencies with controllable test doubles.
from unittest.mock import Mock

# Define test fixtures, parameterized cases, and expected failures.
import pytest
# Build HTTP requests and handle transport failures.
import requests

# Send accepted media actions through the VLC backend.
from control.media_controller import VLCController
# Load validated configuration types, defaults, or configuration helpers.
from settings import VLCConfig


# Provide reusable test setup with a fresh instance for each test.
@pytest.fixture
# Create a VLC controller backed by a fake HTTP session for each test.
def controller(monkeypatch):
    # Supply a temporary test password that pytest restores after the test.
    monkeypatch.setenv("GESTUREREMOTE_VLC_PASSWORD", "test-only-secret")
    # Restrict the fake session to attributes available on a real requests session.
    session = Mock(spec=requests.Session)
    # Simulate an HTTP response accepted by the transport layer.
    session.get.return_value.status_code = 200
    # Simulate selected, paused media at half of the configured volume ceiling.
    session.get.return_value.json.return_value = {"state": "paused", "volume": 128, "length": 60, "currentplid": 1}
    # Inject the fake session so the tests never contact a real VLC instance.
    return VLCController(VLCConfig(), session=session)


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("action,params,after", [
    # Resuming paused media must use the explicit resume command.
    ("play_pause", {"command": "pl_forceresume"}, {"state": "playing"}),
    # Muting must set volume to zero and observe the new volume.
    ("mute", {"command": "volume", "val": "0"}, {"volume": 0}),
    # Volume-up adds the default step of 13 to the initial volume of 128.
    ("volume_up", {"command": "volume", "val": "141"}, {"volume": 141}),
    # Volume-down subtracts the default step of 13 from the initial volume.
    ("volume_down", {"command": "volume", "val": "115"}, {"volume": 115}),
    # Forward seeking uses a positive relative offset in seconds.
    ("seek_forward", {"command": "seek", "val": "+10"}, {"time": 15}),
    # Backward seeking uses a negative relative offset in seconds.
    ("seek_backward", {"command": "seek", "val": "-10"}, {"time": 0}),
])
# Test: actual vlc commands.
def test_actual_vlc_commands(controller, action, params, after):
    # Capture the player state returned by the initial read-only status request.
    before = controller.session.get.return_value.json.return_value
    # Return the initial state followed by the expected state after the command.
    controller.session.get.return_value.json.side_effect = [before, {**before, **after}]
    # Verify the controller confirms successful completion.
    assert controller.execute(action).success
    # Check the exact endpoint, parameters, timeout, and redirect policy used.
    controller.session.get.assert_called_with(
        # Expect the configured local status endpoint and explicit request options.
        "http://127.0.0.1:8080/requests/status.json", params=params, timeout=0.7, allow_redirects=False)
    # Verify exactly 2 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 2
    # Verify HTTP authentication uses the configured username and test password.
    assert controller.session.auth == ("", "test-only-secret")
    # Verify environment proxy settings cannot redirect local VLC requests.
    assert not controller.session.trust_env


# Test: status probe does not change playback.
def test_status_probe_does_not_change_playback(controller):
    # Verify the controller confirms successful completion.
    assert controller.check_connection().success
    # Verify the exact VLC command parameters and request sequence.
    assert controller.session.get.call_args.kwargs["params"] == {}


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("status", [401, 403, 404, 500, 302])
# Test: http errors.
def test_http_errors(controller, status):
    # Simulate the selected HTTP error status from VLC.
    controller.session.get.return_value.status_code = status
    # Attempt playback control and retain the reported failure.
    result = controller.execute("play_pause")
    # Verify the controller reports failure instead of claiming a successful command.
    assert not result.success
    # Verify the error message does not expose the test password or exception secret.
    assert "test-only-secret" not in result.message
    # Authentication failures should receive a more specific user-facing explanation.
    if status in (401, 403):
        # Verify the returned message explains the expected result.
        assert "authentication" in result.message


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("exception", [requests.Timeout("secret"), requests.ConnectionError("secret"), requests.RequestException("secret")])
# Test: connection failures are safe and never retried.
def test_connection_failures_are_safe_and_never_retried(controller, exception):
    # Raise the chosen transport exception instead of returning a response.
    controller.session.get.side_effect = exception
    # Attempt mute and inspect the controller's safe error message.
    result = controller.execute("mute")
    # Verify the controller reports failure instead of claiming a successful command.
    assert not result.success
    # Verify the error message does not expose the test password or exception secret.
    assert "secret" not in result.message
    # Verify exactly 1 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 1


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("payload", [[], {}, {"state": "nonsense"}, {"state": "playing", "error": "failed"}])
# Test: invalid payload.
def test_invalid_payload(controller, payload):
    # Return a structurally invalid or explicitly unsuccessful status payload.
    controller.session.get.return_value.json.return_value = payload
    # Verify the controller reports failure instead of claiming a successful command.
    assert not controller.execute("volume_up").success


# Test: bad json.
def test_bad_json(controller):
    # Simulate a response body that cannot be decoded as JSON.
    controller.session.get.return_value.json.side_effect = requests.exceptions.JSONDecodeError("bad", "", 0)
    # Verify the returned message explains the expected result.
    assert "malformed JSON" in controller.execute("mute").message


# Test: unsupported action makes no request.
def test_unsupported_action_makes_no_request(controller):
    # Verify the controller reports failure instead of claiming a successful command.
    assert not controller.execute("bad").success
    # Verify unsupported actions are rejected before any network request.
    controller.session.get.assert_not_called()


# Test: missing password.
def test_missing_password(monkeypatch):
    # Remove the password variable to exercise startup validation.
    monkeypatch.delenv("GESTUREREMOTE_VLC_PASSWORD", raising=False)
    # Require the enclosed operation to raise the specified validation error.
    with pytest.raises(ValueError, match="GESTUREREMOTE_VLC_PASSWORD"):
        # Constructing the controller must reject the missing password.
        VLCController(VLCConfig())


# Test: ipv6 url and custom steps.
def test_ipv6_url_and_custom_steps(monkeypatch):
    # Provide a temporary password under a custom environment-variable name.
    monkeypatch.setenv("CUSTOM_VLC_SECRET", "test-password")
    # Create a fake HTTP session for custom connection settings.
    session = Mock()
    # Simulate an HTTP response accepted by the transport layer.
    session.get.return_value.status_code = 200
    # Simulate playing media with a known duration so seeking is allowed.
    session.get.return_value.json.return_value = {"state": "playing", "length": 60}
    # Use IPv6 loopback, a non-default port, and a custom volume step.
    c = VLCController(replace(VLCConfig(), host="::1", port=9000, volume_step=26,
                              # Customize the seek interval and password source, using the fake session.
                              seek_seconds=20, password_env="CUSTOM_VLC_SECRET"), session=session)
    # Request a backward seek through the customized controller.
    c.execute("seek_backward")
    # Verify IPv6 loopback uses brackets and the custom HTTP port.
    assert c.url == "http://[::1]:9000/requests/status.json"
    # Verify the exact VLC command parameters and request sequence.
    assert session.get.call_args.kwargs["params"]["val"] == "-20"
    # Release the session owned by this test controller.
    c.close()
    # Verify controller cleanup closes its HTTP session exactly once.
    session.close.assert_called_once()


# Test: requests encodes relative plus.
def test_requests_encodes_relative_plus():
    # Prepare a real requests URL without sending any network traffic.
    request = requests.Request("GET", "http://127.0.0.1:8080/requests/status.json",
                               # Encode the positive seek offset as a query parameter.
                               params={"command": "seek", "val": "+10"}).prepare()
    # Verify the plus sign is URL-encoded rather than interpreted as a space.
    assert "val=%2B10" in request.url


# Test: mute restores the volume observed before muting.
def test_mute_restores_the_volume_observed_before_muting(controller):
    # Keep playback metadata fixed while simulating different volume readings.
    base = {"state": "playing", "length": 60}
    # Simulate the status readings before and after muting and unmuting.
    controller.session.get.return_value.json.side_effect = [{**base, "volume": v} for v in (91, 0, 0, 91)]
    # Verify the returned message explains the expected result.
    assert "Muted" in controller.execute("mute").message
    # Verify the returned message explains the expected result.
    assert "Unmuted" in controller.execute("mute").message
    # Extract only state-changing requests, excluding read-only status polls.
    commands = [c.kwargs["params"] for c in controller.session.get.call_args_list if c.kwargs["params"]]
    # Verify the exact VLC command parameters and request sequence.
    assert commands == [{"command": "volume", "val": "0"}, {"command": "volume", "val": "91"}]


# Test: mute http 200 without volume change is failure.
def test_mute_http_200_without_volume_change_is_failure(controller):
    # Verify the controller reports failure instead of claiming a successful command.
    assert not controller.execute("mute").success
    # Extract only state-changing requests, excluding read-only status polls.
    commands = [c.kwargs["params"] for c in controller.session.get.call_args_list if c.kwargs["params"]]
    # Verify the exact VLC command parameters and request sequence.
    assert commands == [{"command": "volume", "val": "0"}]


# Test: status poll confirms delayed change without replaying command.
def test_status_poll_confirms_delayed_change_without_replaying_command(controller):
    # Capture the player state returned by the initial read-only status request.
    before = controller.session.get.return_value.json.return_value
    # Delay the volume change until a read-only confirmation poll.
    controller.session.get.return_value.json.side_effect = [before, before, {**before, "volume": 0}]
    # Verify the controller confirms successful completion.
    assert controller.execute("mute").success
    # Verify the exact VLC command parameters and request sequence.
    assert [c.kwargs["params"] for c in controller.session.get.call_args_list] == [{}, {"command": "volume", "val": "0"}, {}]


# Test: play pause requires selected media.
def test_play_pause_requires_selected_media(controller):
    # Simulate an empty player with no currently selected playlist item.
    controller.session.get.return_value.json.return_value = {"state": "stopped", "currentplid": -1}
    # Attempt play/pause when there is no media to control.
    result = controller.execute("play_pause")
    # Verify the controller reports failure instead of claiming a successful command.
    assert not result.success
    # Verify the returned message explains the expected result.
    assert "open a video" in result.message
    # Verify exactly 1 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 1


# Test: play pause confirms paused state.
def test_play_pause_confirms_paused_state(controller):
    # Simulate playback transitioning from playing to paused.
    controller.session.get.return_value.json.side_effect = [{"state": "playing"}, {"state": "paused"}]
    # Verify the controller confirms successful completion.
    assert controller.execute("play_pause").success
    # Verify the exact VLC command parameters and request sequence.
    assert controller.session.get.call_args.kwargs["params"] == {"command": "pl_forcepause"}


# Run this test once for each supplied input or input combination.
@pytest.mark.parametrize("volume,action", [(256, "volume_up"), (0, "volume_down")])
# Test: volume stops at limits.
def test_volume_stops_at_limits(controller, volume, action):
    # Place the player at the selected upper or lower volume boundary.
    controller.session.get.return_value.json.return_value = {"state": "playing", "volume": volume}
    # Attempt the adjustment that would move beyond that boundary.
    result = controller.execute(action)
    # Verify reaching a volume boundary is reported as a successful no-op.
    assert result.success and "limit reached" in result.message
    # Verify exactly 1 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 1


# Test: zero volume start has bounded unmute fallback.
def test_zero_volume_start_has_bounded_unmute_fallback(controller):
    # Start muted and simulate restoration to the configured fallback volume.
    controller.session.get.return_value.json.side_effect = [{"state": "playing", "volume": v} for v in (0, 128)]
    # Verify the controller confirms successful completion.
    assert controller.execute("mute").success
    # Verify the exact VLC command parameters and request sequence.
    assert controller.session.get.call_args.kwargs["params"]["val"] == "128"


# Test: timeout after sending mute is not replayed.
def test_timeout_after_sending_mute_is_not_replayed(controller):
    # Keep a successful preflight response before simulating a command timeout.
    response = controller.session.get.return_value
    # Let preflight succeed, then raise a timeout while sending the command.
    controller.session.get.side_effect = [response, requests.Timeout()]
    # Verify the controller reports failure instead of claiming a successful command.
    assert not controller.execute("mute").success
    # Verify exactly 2 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 2


# Test: vlc lua floating point volume is supported.
def test_vlc_lua_floating_point_volume_is_supported(controller):
    # Return floating-point volumes, matching VLC Lua's numeric representation.
    controller.session.get.return_value.json.side_effect = [{"state": "playing", "volume": v} for v in (156.0, 0.0, 0.0, 156.0)]
    # Verify the controller confirms successful completion.
    assert controller.execute("mute").success
    # Verify the controller confirms successful completion.
    assert controller.execute("mute").success


# Test: volume up does not reduce externally amplified volume.
def test_volume_up_does_not_reduce_externally_amplified_volume(controller):
    # Simulate volume amplified above this application's configured ceiling.
    controller.session.get.return_value.json.return_value = {"state": "playing", "volume": 300.0}
    # Verify the controller confirms successful completion.
    assert controller.execute("volume_up").success
    # Verify exactly 1 HTTP requests were made, with no extra retries.
    assert controller.session.get.call_count == 1
