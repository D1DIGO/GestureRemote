"""VLC HTTP backend with verified playback/volume changes and no command retries."""

# Define structured records or create modified copies of configuration records.
from dataclasses import dataclass, field
# Validate finite numbers before using them in timing or volume calculations.
import math
# Access process environment variables and operating-system helpers.
import os
# Measure monotonic elapsed time and perform short waits.
import time
# Describe flexible status values and the media-controller interface with type hints.
from typing import Any
# Describe flexible status values and the media-controller interface with type hints.
from typing import Protocol

# Build HTTP requests and handle transport failures.
import requests

# Load validated configuration types, defaults, or configuration helpers.
from settings import VLCConfig


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Return a media operation's outcome and optional observed VLC state.
class CommandResult:
    # Indicate whether the operation met this backend's success criteria.
    success: bool
    # Provide a user-facing message without embedding credentials.
    message: str
    # Keep selected status fields out of the object's default representation.
    status: dict[str, Any] | None = field(default=None, repr=False, compare=False)


# Define the interface a future media backend must implement.
class MediaController(Protocol):
    # Require a method that executes one named media action.
    def execute(self, action: str) -> CommandResult: ...
    # Require a read-only connection probe.
    def check_connection(self) -> CommandResult: ...
    # Require explicit cleanup for any backend resources.
    def close(self) -> None: ...


# Implement the media interface through VLC's local HTTP endpoint.
class VLCController:
    # Accept an optional injected HTTP session to enable isolated tests.
    def __init__(self, config: VLCConfig, *, session: requests.Session | None = None):
        # Read the password from the configured environment variable.
        password = os.environ.get(config.password_env, "")
        # Reject an empty password before making any HTTP request.
        if not password:
            # Name the missing environment variable without displaying a secret.
            raise ValueError(f"Set {config.password_env} to the password configured in VLC")
        # Store connection and action settings for later requests.
        self.config = config
        # Reuse an injected test session or create a real persistent HTTP session.
        self.session = session if session is not None else requests.Session()
        # Avoid routing localhost through environment proxies or loading .netrc.
        self.session.trust_env = False  # do not proxy localhost or read .netrc
        # Configure HTTP Basic authentication without adding credentials to the URL.
        self.session.auth = (config.username, password)
        # Wrap IPv6 addresses in brackets when building an HTTP URL.
        host = f"[{config.host}]" if ":" in config.host else config.host
        # Use VLC's JSON status endpoint for status and command requests.
        self.url = f"http://{host}:{config.port}/requests/status.json"
        # Choose a fallback restore level until a real pre-mute volume is observed.
        self._restore_volume = config.unmute_volume

    # Make one bounded HTTP request and validate its status response.
    def _request(self, params: dict[str, str]) -> CommandResult:
        # Handle failures from the operations in this block.
        try:
            # Never retry a toggle/relative command: a timeout may follow success.
            # Send encoded query parameters; do not follow redirects or retry commands.
            response = self.session.get(self.url, params=params, timeout=self.config.timeout, allow_redirects=False)
            # Identify authentication or access failures separately.
            if response.status_code in (401, 403):
                # Explain the password problem without leaking its value.
                return CommandResult(False, "VLC authentication failed; check the HTTP password")
            # Reject any remaining non-200 HTTP response.
            if response.status_code != 200:
                # Return the status code to help diagnose endpoint failures.
                return CommandResult(False, f"VLC returned HTTP {response.status_code}")
            # Decode VLC's response as JSON.
            payload = response.json()
            # Require a status object with a recognized playback state.
            if not isinstance(payload, dict) or payload.get("state") not in ("playing", "paused", "stopped"):
                # Do not treat an unrelated HTTP 200 response as valid VLC status.
                return CommandResult(False, "VLC returned an invalid status response")
            # Respect a command error explicitly reported by VLC.
            if payload.get("error"):
                # Report the command error as an unsuccessful operation.
                return CommandResult(False, "VLC reported a command error")
            # Retain only playback fields needed for command verification.
            status = {key: payload[key] for key in ("state", "volume", "time", "length", "currentplid") if key in payload}
            # An HTTP acknowledgment alone does not yet verify every action's effect.
            return CommandResult(True, "VLC acknowledged request", status)
        # Handle timeouts without replaying a possibly successful command.
        except requests.Timeout:
            # Tell the user the actual outcome is uncertain.
            return CommandResult(False, "VLC request timed out; command outcome is unknown (not retried)")
        # Handle invalid JSON, including Requests' JSON decoding exception.
        except ValueError:
            # Distinguish malformed responses from connectivity failures.
            return CommandResult(False, "VLC returned malformed JSON")
        # Handle refused connections and other HTTP transport errors.
        except requests.RequestException:
            # Explain how to restore the unavailable VLC interface.
            return CommandResult(False, "VLC unavailable; start VLC and enable its HTTP interface")

    # Probe connectivity without a command parameter.
    def check_connection(self) -> CommandResult:
        # Read status without intentionally changing playback.
        return self._request({})

    # Confirm a state change using read-only polling, never command replay.
    def _verify(self, result: CommandResult, key: str, target: Any, message: str) -> CommandResult:
        """Poll status only: VLC can apply a command after producing its response."""
        # Inspect the initial response and at most three follow-up status responses.
        for attempt in range(4):
            # Stop verification immediately if a request failed.
            if not result.success:
                # Preserve the original transport or response error.
                return result
            # Check the requested field against the expected target value.
            if result.status and result.status.get(key) == target:
                # Report success only after the requested state was observed.
                return CommandResult(True, message, result.status)
            # Skip sleeping and polling after the last permitted check.
            if attempt < 3:
                # Give VLC a short time to apply a delayed state change.
                time.sleep(0.05)
                # Poll status without resending the action.
                result = self._request({})
        # Report that the expected effect was not confirmed.
        return CommandResult(False, f"VLC did not confirm {message.lower()}; no command retry", result.status)

    # Translate one application action into VLC commands and verification.
    def execute(self, action: str) -> CommandResult:
        # Define relative seek requests using the configured number of seconds.
        seeks = {
            # A leading plus sign requests a forward seek relative to current time.
            "seek_forward": {"command": "seek", "val": f"+{self.config.seek_seconds}"},
            # A leading minus sign requests a backward seek relative to current time.
            "seek_backward": {"command": "seek", "val": f"-{self.config.seek_seconds}"},
        }
        # Reject unsupported action names before contacting VLC.
        if action not in {"play_pause", "mute", "volume_up", "volume_down"} | seeks.keys():
            # Return a clear error for an unknown backend operation.
            return CommandResult(False, f"Unsupported action: {action}")
        # Read current playback state and volume before deciding how to change them.
        before = self._request({})
        # Do not send a command if its initial status request failed.
        if not before.success:
            # Return that failure without attempting a media operation.
            return before
        # Use the validated selected fields from the status response.
        state = before.status or {}
        # Handle relative seeking independently of playback and volume toggles.
        if action in seeks:
            # Reject seeking when playback is stopped or no positive duration is known.
            if state.get("state") == "stopped" or state.get("length", 0) <= 0:
                # Ask the user to select suitable media before retrying.
                return CommandResult(False, "Open a seekable video in VLC first")
            # Send one seek request; this path checks acknowledgment, not final position.
            return self._request(seeks[action])
        # Choose an explicit target for the play/pause toggle.
        if action == "play_pause":
            # Distinguish an empty player from one with a selected playlist item.
            if state.get("state") == "stopped" and state.get("currentplid", -1) < 0:
                # Explain why play/pause cannot start media that was never selected.
                return CommandResult(False, "No media selected in VLC; open a video first")
            # Pause playing media; otherwise aim to start or resume playback.
            target = "paused" if state["state"] == "playing" else "playing"
            # Choose an explicit pause, resume, or play operation from the current state.
            command = "pl_forcepause" if target == "paused" else "pl_forceresume" if state["state"] == "paused" else "pl_play"
            # Verify that VLC actually reports the intended playback state.
            return self._verify(self._request({"command": command}), "state", target, f"Playback {target}")

        # Read the player's current volume before any audio adjustment.
        volume = state.get("volume")
        # Reject missing, boolean, nonnumeric, nonfinite, or negative volume values.
        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume < 0:
            # Do not calculate a volume target from unusable audio status.
            return CommandResult(False, "VLC did not report a usable audio volume")
        # Normalize Lua's integral floating-point volume representation.
        volume = round(volume)  # VLC's Lua JSON may encode volume as 156.0.
        # Implement mute as zero volume plus a remembered restore level.
        if action == "mute":
            # A nonzero volume means this command should mute.
            if volume > 0:
                # Remember the current level, bounded by the configured restore limit.
                self._restore_volume = min(volume, self.config.max_volume)
                # Request silence by setting the volume to zero.
                target = 0
            # A zero volume means this command should restore the remembered level.
            else:
                # Restore the remembered value or the initial fallback.
                target = self._restore_volume
            # Describe the resulting mute state using VLC's 256-units-per-100% scale.
            message = "Muted (volume 0)" if target == 0 else f"Unmuted (volume {target / 256:.0%})"
        # Handle ordinary volume increase or decrease.
        else:
            # Limit increases without lowering a level already raised externally.
            target = (max(volume, min(self.config.max_volume, volume + self.config.volume_step))
                      # For a decrease, subtract one step without going below zero.
                      if action == "volume_up" else max(0, volume - self.config.volume_step))
            # Format the target volume as a percentage for user feedback.
            message = f"Volume {target / 256:.0%}"
            # Avoid redundant volume requests when a limit has already been reached.
            if target == volume:
                # Report the limit without claiming an additional volume change.
                return CommandResult(True, message + " (limit reached)", state)
        # Explicit volume zero works even when a VLC build ignores HTTP hotkeys.
        # Set an absolute volume and verify VLC reports that target.
        return self._verify(self._request({"command": "volume", "val": str(target)}), "volume", target, message)

    # Release connections associated with this controller.
    def close(self) -> None:
        # Close the Requests session.
        self.session.close()
