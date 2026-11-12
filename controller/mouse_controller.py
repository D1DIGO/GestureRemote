import subprocess
import json


def get_screen_size():
    monitors = json.loads(
        subprocess.check_output(["hyprctl", "monitors", "-j"], text=True)
    )

    if not monitors:
        raise RuntimeError("No Hyprland monitor found")

    monitor = next((m for m in monitors if m.get("focused")), monitors[0])

    return monitor["width"], monitor["height"]


class MouseController:

    def __init__(self):
        self.width, self.height = get_screen_size()

    def _run(self, *args):
        subprocess.run(
            ["ydotool", *map(str, args)], check=True, stdout=subprocess.DEVNULL
        )

    def move_to(self, x, y):
        x = max(0, min(int(x), self.width - 1))
        y = max(0, min(int(y), self.height - 1))

        self._run("mousemove", "--absolute", x, y)

    def click(self):
        self._run("click", "0xC0")

    def mouse_down(self):
        self._run("click", "0x40")

    def mouse_up(self):
        self._run("click", "0x80")
