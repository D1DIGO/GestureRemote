"""VLC HTTP backend. HTTP hotkeys are internal VLC actions, not OS keystrokes."""

from dataclasses import dataclass
import os
from typing import Protocol

import requests

from settings import VLCConfig


@dataclass(frozen=True)
class CommandResult:
    success: bool
    message: str


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
            return CommandResult(True, "VLC acknowledged request")
        except requests.Timeout:
            return CommandResult(False, "VLC request timed out; command outcome is unknown (not retried)")
        except ValueError:
            return CommandResult(False, "VLC returned malformed JSON")
        except requests.RequestException:
            return CommandResult(False, "VLC unavailable; start VLC and enable its HTTP interface")

    def check_connection(self) -> CommandResult:
        return self._request({})

    def execute(self, action: str) -> CommandResult:
        commands = {
            "play_pause": {"command": "pl_pause"},
            "mute": {"command": "key", "val": "volume-mute"},
            "volume_up": {"command": "volume", "val": f"+{self.config.volume_step}"},
            "volume_down": {"command": "volume", "val": f"-{self.config.volume_step}"},
            "seek_forward": {"command": "seek", "val": f"+{self.config.seek_seconds}"},
            "seek_backward": {"command": "seek", "val": f"-{self.config.seek_seconds}"},
        }
        if action not in commands:
            return CommandResult(False, f"Unsupported action: {action}")
        return self._request(commands[action])

    def close(self) -> None:
        self.session.close()
