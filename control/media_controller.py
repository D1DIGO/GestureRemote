"""VLC HTTP backend with verified playback/volume changes and no command retries."""

from dataclasses import dataclass, field
import math
import os
import time
from typing import Any
from typing import Protocol

import requests

from settings import VLCConfig


@dataclass(frozen=True)
class CommandResult:
    success: bool
    message: str
    status: dict[str, Any] | None = field(default=None, repr=False, compare=False)


class MediaController(Protocol):
    def execute(self, action: str) -> CommandResult: ...
    def check_connection(self) -> CommandResult: ...
    def close(self) -> None: ...


class VLCController:
    def __init__(self, config: VLCConfig, *, session: requests.Session | None = None):
        password = os.environ.get(config.password_env, "")
        if not password:
            raise ValueError(f"Set {config.password_env} to the password configured in VLC")
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.session.trust_env = False  # do not proxy localhost or read .netrc
        self.session.auth = (config.username, password)
        host = f"[{config.host}]" if ":" in config.host else config.host
        self.url = f"http://{host}:{config.port}/requests/status.json"
        self._restore_volume = config.unmute_volume

    def _request(self, params: dict[str, str]) -> CommandResult:
        try:
            # Never retry a toggle/relative command: a timeout may follow success.
            response = self.session.get(self.url, params=params, timeout=self.config.timeout, allow_redirects=False)
            if response.status_code in (401, 403):
                return CommandResult(False, "VLC authentication failed; check the HTTP password")
            if response.status_code != 200:
                return CommandResult(False, f"VLC returned HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("state") not in ("playing", "paused", "stopped"):
                return CommandResult(False, "VLC returned an invalid status response")
            if payload.get("error"):
                return CommandResult(False, "VLC reported a command error")
            status = {key: payload[key] for key in ("state", "volume", "time", "length", "currentplid") if key in payload}
            return CommandResult(True, "VLC acknowledged request", status)
        except requests.Timeout:
            return CommandResult(False, "VLC request timed out; command outcome is unknown (not retried)")
        except ValueError:
            return CommandResult(False, "VLC returned malformed JSON")
        except requests.RequestException:
            return CommandResult(False, "VLC unavailable; start VLC and enable its HTTP interface")

    def check_connection(self) -> CommandResult:
        return self._request({})

    def _verify(self, result: CommandResult, key: str, target: Any, message: str) -> CommandResult:
        """Poll status only: VLC can apply a command after producing its response."""
        for attempt in range(4):
            if not result.success:
                return result
            if result.status and result.status.get(key) == target:
                return CommandResult(True, message, result.status)
            if attempt < 3:
                time.sleep(0.05)
                result = self._request({})
        return CommandResult(False, f"VLC did not confirm {message.lower()}; no command retry", result.status)

    def execute(self, action: str) -> CommandResult:
        seeks = {
            "seek_forward": {"command": "seek", "val": f"+{self.config.seek_seconds}"},
            "seek_backward": {"command": "seek", "val": f"-{self.config.seek_seconds}"},
        }
        if action not in {"play_pause", "mute", "volume_up", "volume_down"} | seeks.keys():
            return CommandResult(False, f"Unsupported action: {action}")
        before = self._request({})
        if not before.success:
            return before
        state = before.status or {}
        if action in seeks:
            if state.get("state") == "stopped" or state.get("length", 0) <= 0:
                return CommandResult(False, "Open a seekable video in VLC first")
            return self._request(seeks[action])
        if action == "play_pause":
            if state.get("state") == "stopped" and state.get("currentplid", -1) < 0:
                return CommandResult(False, "No media selected in VLC; open a video first")
            target = "paused" if state["state"] == "playing" else "playing"
            command = "pl_forcepause" if target == "paused" else "pl_forceresume" if state["state"] == "paused" else "pl_play"
            return self._verify(self._request({"command": command}), "state", target, f"Playback {target}")

        volume = state.get("volume")
        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume < 0:
            return CommandResult(False, "VLC did not report a usable audio volume")
        volume = round(volume)  # VLC's Lua JSON may encode volume as 156.0.
        if action == "mute":
            if volume > 0:
                self._restore_volume = min(volume, self.config.max_volume)
                target = 0
            else:
                target = self._restore_volume
            message = "Muted (volume 0)" if target == 0 else f"Unmuted (volume {target / 256:.0%})"
        else:
            target = (max(volume, min(self.config.max_volume, volume + self.config.volume_step))
                      if action == "volume_up" else max(0, volume - self.config.volume_step))
            message = f"Volume {target / 256:.0%}"
            if target == volume:
                return CommandResult(True, message + " (limit reached)", state)
        # Explicit volume zero works even when a VLC build ignores HTTP hotkeys.
        return self._verify(self._request({"command": "volume", "val": str(target)}), "volume", target, message)

    def close(self) -> None:
        self.session.close()
