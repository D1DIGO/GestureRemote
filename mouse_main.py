#!/usr/bin/env python3
"""
gesture_mouse_ydotool.py

Live webcam gesture control using ydotool (works on Hyprland / any Wayland compositor):
    camera frame -> MediaPipe Hands -> landmarks -> trained sklearn MLP -> gesture class
    -> command, with the class shown on the camera window.

The cursor moves RELATIVELY, like a touchpad: when your open palm moves right, the cursor moves
right by an amount proportional to how far your hand moved; when your hand stops, the cursor stops.
Your absolute hand position does not matter, so you can "clutch": change to another gesture or
drop the hand, bring it back to a comfortable spot and keep going from where the cursor is.

Default gesture commands:
    palm   -> move the cursor by moving your palm
    fist   -> short fist = left click (the click happens when you open the hand again)
              fist held longer than --drag-hold seconds = press and hold the left button (drag);
              while dragging, moving your fist moves the cursor; opening the hand releases the button
    peace  -> right click          (hold the gesture ~0.8 s)
    ok     -> double click
    like   -> volume up key   dislike -> volume down key   mute -> mute key
              (key events: they trigger whatever your compositor binds do for XF86AudioRaiseVolume etc.)
    one    -> scroll up            two_up  -> scroll down
(Edit ONE_SHOT_ACTIONS below to change or add commands. Classes your model was not trained on are never triggered.)

Requirements:
    pip install mediapipe opencv-python joblib scikit-learn numpy pandas
    ydotool + its daemon:   sudo pacman -S ydotool     (or your distro's package)
        - ydotoold must be running (e.g. in hyprland.conf:  exec-once = ydotoold)
        - ydotoold needs write access to /dev/uinput. Either run it as root, or give your user access:
              sudo modprobe uinput
              echo uinput | sudo tee /etc/modules-load.d/uinput.conf
              echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.rules
              sudo usermod -aG input $USER
              sudo udevadm control --reload-rules && sudo udevadm trigger
              (log out and back in so the group change applies)
        - If ydotoold uses a non-default socket, export YDOTOOL_SOCKET=<path> before starting this script.

Files needed in the same folder:
    train_mlp_sklearn.py   (provides the exact feature preprocessing used in training)
    gesture_mlp.joblib     (produced by train_mlp_sklearn.py)

Usage:
    python gesture_mouse_ydotool.py
    python gesture_mouse_ydotool.py --sensitivity 1500 --drag-hold 0.9 --start-enabled

Tuning the feel:
    --sensitivity   cursor pixels per "camera frame width" of hand movement (default 2000).
                    Lower = slower, more precise. Higher = faster.
    --smoothing     0..1, weight of the newest hand position (default 0.5). Lower = smoother but laggier.
    --deadzone      hand movements smaller than this many cursor pixels are ignored (hand tremor / tracking noise).
    Note: ydotool's relative motion passes through your compositor's pointer acceleration. If the speed feels
    uneven, try a flat acceleration profile (in Hyprland: input { accel_profile = flat }).

Keys in the camera window:  C = toggle mouse control ON/OFF (starts OFF),  Q / ESC = quit.
IMPORTANT: with focus-follows-mouse, the camera window loses keyboard focus as soon as the cursor
moves, so those keys may not reach it. Add binds to your compositor config instead (Hyprland example):
    bind = $mainMod SHIFT, G, exec, pkill -USR1 -f gesture_mouse_ydotool.py   # toggle control ON/OFF
    bind = $mainMod SHIFT, X, exec, pkill -f gesture_mouse_ydotool.py         # quit (releases the button)
"""

import argparse
import math
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from train_mlp import predict_gestures

# Palm base landmarks (wrist + the 4 finger knuckles). They barely move when you close your hand,
# so the cursor does not jump when you go from palm to fist.
PALM_IDS = (0, 5, 9, 13, 17)

# ydotool mouse button codes: base button, optionally OR-ed with a press/release mask
BUTTON_CODES = {"left": 0x00, "right": 0x01, "middle": 0x02}
PRESS_ONLY = 0x40
RELEASE_ONLY = 0x80
PRESS_AND_RELEASE = 0xC0

# Linux input event key codes (linux/input-event-codes.h)
KEY_MUTE = 113
KEY_VOLUMEDOWN = 114
KEY_VOLUMEUP = 115

# gesture class name -> (description shown on screen, function(backend) to run)
ONE_SHOT_ACTIONS = {
    "peace": ("RIGHT CLICK", lambda b: b.click("right")),
    "ok": ("DOUBLE CLICK", lambda b: b.double_click()),
    "like": ("VOLUME UP", lambda b: b.tap_key(KEY_VOLUMEUP)),
    "dislike": ("VOLUME DOWN", lambda b: b.tap_key(KEY_VOLUMEDOWN)),
    "mute": ("MUTE", lambda b: b.tap_key(KEY_MUTE)),
    "one": ("SCROLL UP", lambda b: b.scroll(5)),
    "two_up": ("SCROLL DOWN", lambda b: b.scroll(-5)),
}

SIGNAL_STATE = {"toggle_requested": False}


def on_sigusr1(signum, frame):
    SIGNAL_STATE["toggle_requested"] = True


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


# --------------------------------------------------------------------------- #
# ydotool backend
# --------------------------------------------------------------------------- #
class YdotoolBackend:
    def __init__(self):
        self._warned = False

        if shutil.which("ydotool") is None:
            sys.exit("'ydotool' was not found in PATH.")

        # "click 0x00" selects the left button but does nothing, so it is a harmless connectivity test.
        ok, message = self._run_ydotool(["click", "0x00"])
        if not ok:
            sys.exit(
                "ydotool could not send input. Make sure ydotoold is running and can write to "
                "/dev/uinput (see the setup notes at the top of this file).\n"
                f"ydotool said: {message or '(no output)'}"
            )

    def _run_ydotool(self, args):
        try:
            proc = subprocess.run(
                ["ydotool", *args], capture_output=True, text=True, timeout=3
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            return False, str(exc)
        message = (proc.stderr or proc.stdout or "").strip()
        return proc.returncode == 0, message

    def _ydotool(self, args):
        ok, message = self._run_ydotool(args)
        if not ok and not self._warned:
            self._warned = True
            print(f"[warn] ydotool {' '.join(args)} failed: {message or '(no output)'}")
        return ok

    @staticmethod
    def _button_arg(button, mask):
        return f"0x{BUTTON_CODES[button] | mask:02X}"

    def move_relative(self, dx, dy):
        """Move the cursor by (dx, dy) pixels (positive = right / down)."""
        self._ydotool(["mousemove", "-x", str(int(dx)), "-y", str(int(dy))])

    def mouse_down(self, button="left"):
        self._ydotool(["click", self._button_arg(button, PRESS_ONLY)])

    def mouse_up(self, button="left"):
        self._ydotool(["click", self._button_arg(button, RELEASE_ONLY)])

    def click(self, button="left"):
        self._ydotool(["click", self._button_arg(button, PRESS_AND_RELEASE)])

    def double_click(self):
        self._ydotool(
            ["click", "--repeat", "2", self._button_arg("left", PRESS_AND_RELEASE)]
        )

    def scroll(self, notches):
        """Positive = scroll up, negative = scroll down."""
        # ydotool's wheel axis is inverted: a positive value scrolls down.
        self._ydotool(["mousemove", "--wheel", "-x", "0", "-y", str(-int(notches))])

    def tap_key(self, key_code):
        self._ydotool(["key", f"{key_code}:1", f"{key_code}:0"])

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Smoothing of the per-frame predictions
# --------------------------------------------------------------------------- #
class GestureStabilizer:
    """Majority vote over the last few frames so single misclassified frames are ignored."""

    def __init__(self, window, min_votes):
        self.history = deque(maxlen=window)
        self.min_votes = min_votes
        self.stable = None

    def update(self, gesture):
        self.history.append(gesture)
        label, votes = Counter(self.history).most_common(1)[0]
        if votes >= self.min_votes:
            self.stable = label
        return self.stable


class OneShotTrigger:
    """Fires an action once after a gesture has been held for `hold` seconds (then waits for a gesture change)."""

    def __init__(self, hold, cooldown):
        self.hold = hold
        self.cooldown = cooldown
        self.current = None
        self.since = 0.0
        self.fired = False
        self.last_fire = 0.0

    def update(self, gesture, now):
        if gesture != self.current:
            self.current = gesture
            self.since = now
            self.fired = False
            return None
        if (
            gesture in ONE_SHOT_ACTIONS
            and not self.fired
            and now - self.since >= self.hold
            and now - self.last_fire >= self.cooldown
        ):
            self.fired = True
            self.last_fire = now
            return gesture
        return None


# --------------------------------------------------------------------------- #
# Mouse state machine
# --------------------------------------------------------------------------- #
class MouseController:
    def __init__(
        self,
        backend,
        move_gestures,
        click_gesture,
        drag_hold,
        sensitivity,
        smoothing,
        deadzone,
        max_step,
        lost_grace,
    ):
        self.backend = backend
        self.move_gestures = set(move_gestures)
        self.click_gesture = click_gesture
        self.drag_hold = drag_hold
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self.deadzone = deadzone
        self.max_step = max_step
        self.lost_grace = lost_grace

        self.enabled = False
        self.moving = False
        self.smooth = None  # smoothed hand position (frame-width units)
        self.anchor = None  # hand position at the last cursor movement
        self.carry = (0.0, 0.0)  # sub-pixel remainder that was not sent yet
        self.fist_start = (
            None  # time when the current fist started (waiting for click/drag decision)
        )
        self.dragging = False
        self.lost_since = None
        self.flash_text = ""
        self.flash_until = 0.0

    # ---- helpers ----------------------------------------------------------
    def flash(self, text, now, duration=0.8):
        self.flash_text = text
        self.flash_until = now + duration

    def fist_progress(self, now):
        if self.dragging:
            return 1.0
        if self.fist_start is not None:
            return min(1.0, (now - self.fist_start) / self.drag_hold)
        return 0.0

    def _reset_tracking(self):
        self.smooth = None
        self.anchor = None
        self.carry = (0.0, 0.0)

    def _follow(self, hand_xy):
        """Move the cursor by the (smoothed) distance the hand moved since the last cursor movement."""
        if self.smooth is None:
            # first frame of a movement segment: just remember where the hand is (no jump)
            self.smooth = hand_xy
            self.anchor = hand_xy
            self.carry = (0.0, 0.0)
            return

        a = self.smoothing
        self.smooth = (
            self.smooth[0] + a * (hand_xy[0] - self.smooth[0]),
            self.smooth[1] + a * (hand_xy[1] - self.smooth[1]),
        )

        dx = (self.smooth[0] - self.anchor[0]) * self.sensitivity
        dy = (self.smooth[1] - self.anchor[1]) * self.sensitivity
        dist = math.hypot(dx, dy)
        if dist < self.deadzone:
            return  # tiny movement / tracking noise: keep the anchor so slow real movement still accumulates

        if (
            dist > self.max_step
        ):  # guard against tracking glitches (e.g. the detector switching hands)
            factor = self.max_step / dist
            dx *= factor
            dy *= factor

        total_x = dx + self.carry[0]
        total_y = dy + self.carry[1]
        ix, iy = int(total_x), int(total_y)  # truncate toward zero
        self.carry = (total_x - ix, total_y - iy)
        if ix != 0 or iy != 0:
            self.backend.move_relative(ix, iy)
        self.anchor = self.smooth

    def release_all(self):
        """Make sure the mouse button is never left pressed."""
        if self.dragging:
            self.backend.mouse_up("left")
        self.dragging = False
        self.fist_start = None
        self.lost_since = None
        self.moving = False
        self._reset_tracking()

    # ---- main update ------------------------------------------------------
    def update(self, gesture, hand_xy, now):
        """
        gesture: stable gesture name or None
        hand_xy: (x, y) palm position in frame-width units (x = 0..1, y = 0..height/width), or None if no hand
        Returns a short status text for the overlay.
        """
        if not self.enabled:
            self.release_all()
            return "CONTROL OFF"

        is_fist = gesture == self.click_gesture
        is_move = gesture in self.move_gestures and hand_xy is not None

        # --- currently dragging ---
        if self.dragging:
            if is_fist:
                self.lost_since = None
                if hand_xy is not None:
                    self._follow(hand_xy)
                return "DRAGGING"
            if gesture is None:  # hand lost / low confidence: give a short grace period
                if self.lost_since is None:
                    self.lost_since = now
                if now - self.lost_since < self.lost_grace:
                    return "DRAGGING (hand lost)"
            self.backend.mouse_up("left")
            self.dragging = False
            self.lost_since = None
            self.moving = False
            self._reset_tracking()
            self.flash("DROP", now)
            # fall through so a palm can start moving immediately

        # --- fist: wait to decide between click and drag ---
        if is_fist:
            if self.fist_start is None:
                self.fist_start = now
                self.moving = False
            if now - self.fist_start >= self.drag_hold:
                self.backend.mouse_down("left")
                self.dragging = True
                self.fist_start = None
                self.lost_since = None
                self._reset_tracking()  # drag movement is measured from where the fist is now
                self.flash("DRAG START", now)
                return "DRAGGING"
            return "FIST (keep holding to drag)"

        if self.fist_start is not None:
            # The fist ended before the drag threshold -> it was a click,
            # unless the hand simply disappeared (then cancel without clicking).
            self.fist_start = None
            if gesture is not None:
                self.backend.click("left")
                self.flash("CLICK", now)

        # --- open palm: move the cursor relative to the hand movement ---
        if is_move:
            if not self.moving:
                self.moving = True
                self._reset_tracking()
            self._follow(hand_xy)
            return "MOVING"

        self.moving = False
        return "IDLE"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def draw_overlay(frame, raw_gesture, conf, stable, status, progress, mouse, fps, now):
    h, w = frame.shape[:2]

    raw_text = f"{raw_gesture} ({conf:.2f})" if raw_gesture else f"- ({conf:.2f})"
    put_text(
        frame, f"Gesture: {stable if stable else '-'}", (10, 32), 0.95, (0, 255, 255)
    )
    put_text(frame, f"Raw: {raw_text}", (10, 62), 0.6)
    put_text(
        frame,
        f"Action: {status}",
        (10, 92),
        0.7,
        (0, 255, 0) if mouse.enabled else (0, 165, 255),
    )
    put_text(
        frame,
        f"Mouse control: {'ON' if mouse.enabled else 'OFF'}   [C] toggle   [Q] quit",
        (10, h - 40),
        0.55,
    )
    put_text(frame, f"FPS: {fps:.0f}", (10, h - 15), 0.55)

    if progress > 0.0:
        bar_x, bar_y = int(w * 0.25), h - 80
        bar_w, bar_h = int(w * 0.5), 14
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1
        )
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + int(bar_w * progress), bar_y + bar_h),
            (0, 140, 255),
            -1,
        )
        put_text(frame, "hold to drag", (bar_x, bar_y - 8), 0.5)

    if now < mouse.flash_until:
        (tw, th), _ = cv2.getTextSize(
            mouse.flash_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        )
        put_text(frame, mouse.flash_text, ((w - tw) // 2, h // 2), 1.4, (0, 255, 0), 3)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Control the mouse with hand gestures "
        "(MediaPipe + sklearn MLP + ydotool, relative movement)."
    )
    p.add_argument(
        "--model",
        default="gesture_mlp.joblib",
        help="Model bundle from train_mlp_sklearn.py",
    )
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--move-gestures",
        nargs="+",
        default=["palm"],
        help="Gesture class name(s) that move the cursor",
    )
    p.add_argument(
        "--click-gesture", default="fist", help="Gesture class for click / drag"
    )
    p.add_argument(
        "--drag-hold",
        type=float,
        default=0.7,
        help="Seconds the fist must be held before it becomes a drag instead of a click",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Predictions below this probability are ignored",
    )
    p.add_argument(
        "--sensitivity",
        type=float,
        default=2000.0,
        help="Cursor pixels per camera-frame-width of hand movement (lower = slower / more precise)",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Weight of the newest hand position, 0..1 (smaller = smoother but laggier)",
    )
    p.add_argument(
        "--deadzone",
        type=float,
        default=1.5,
        help="Hand movements smaller than this many cursor pixels are ignored (tremor / noise)",
    )
    p.add_argument(
        "--max-step",
        type=float,
        default=200.0,
        help="Maximum cursor pixels moved per frame (guards against tracking glitches)",
    )
    p.add_argument(
        "--lost-grace",
        type=float,
        default=0.5,
        help="Seconds a drag survives when the hand is briefly lost",
    )
    p.add_argument(
        "--window", type=int, default=5, help="Frames used for majority-vote smoothing"
    )
    p.add_argument(
        "--min-votes",
        type=int,
        default=3,
        help="Votes needed inside the window to change gesture",
    )
    p.add_argument(
        "--action-hold",
        type=float,
        default=0.8,
        help="Seconds to hold a gesture before its one-shot action fires",
    )
    p.add_argument(
        "--action-cooldown",
        type=float,
        default=1.5,
        help="Minimum seconds between one-shot actions",
    )
    p.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1],
        default=1,
        help="MediaPipe hand model: 1 = more accurate (matches training), 0 = faster",
    )
    p.add_argument(
        "--start-enabled",
        action="store_true",
        help="Start with mouse control already ON",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not hasattr(mp, "solutions"):
        sys.exit(
            "This mediapipe version does not provide mp.solutions. "
            "Install one that does: pip install mediapipe==0.10.14"
        )

    try:
        bundle = joblib.load(args.model)
    except FileNotFoundError:
        sys.exit(f"Model file not found: {args.model} (run train_mlp_sklearn.py first)")

    classes = list(bundle["classes"])
    for name in list(args.move_gestures) + [args.click_gesture]:
        if name not in classes:
            sys.exit(f"Gesture '{name}' is not one of the model's classes: {classes}")

    backend = YdotoolBackend()
    mouse = MouseController(
        backend=backend,
        move_gestures=args.move_gestures,
        click_gesture=args.click_gesture,
        drag_hold=args.drag_hold,
        sensitivity=args.sensitivity,
        smoothing=args.smoothing,
        deadzone=args.deadzone,
        max_step=args.max_step,
        lost_grace=args.lost_grace,
    )
    mouse.enabled = args.start_enabled
    stabilizer = GestureStabilizer(args.window, args.min_votes)
    one_shot = OneShotTrigger(args.action_hold, args.action_cooldown)
    reserved = set(args.move_gestures) | {args.click_gesture}

    signal.signal(signal.SIGUSR1, on_sigusr1)
    signal.signal(signal.SIGTERM, on_sigterm)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS

    print(f"[info] classes: {classes}")
    print(
        f"[info] move with: {args.move_gestures} | click/drag with: '{args.click_gesture}'"
    )
    print(
        f"[info] relative movement: sensitivity={args.sensitivity:.0f}px/frame-width, "
        f"smoothing={args.smoothing}, deadzone={args.deadzone}px"
    )
    print(
        "[info] mouse control starts "
        + ("ON" if mouse.enabled else "OFF (press C, or send SIGUSR1)")
    )
    print(
        "[info] toggle from a compositor bind:  pkill -USR1 -f gesture_mouse_ydotool.py"
    )
    print("[info] quit from a compositor bind:    pkill -f gesture_mouse_ydotool.py")

    window_name = "Gesture Mouse"
    fps = 0.0
    last_time = time.time()

    try:
        while True:
            if SIGNAL_STATE["toggle_requested"]:
                SIGNAL_STATE["toggle_requested"] = False
                mouse.enabled = not mouse.enabled
                if not mouse.enabled:
                    mouse.release_all()
                print(
                    f"[info] mouse control {'ON' if mouse.enabled else 'OFF'} (signal)"
                )

            ok, frame = cap.read()
            if not ok:
                print("[warn] camera frame not available, stopping")
                break

            frame = cv2.flip(
                frame, 1
            )  # mirror view so moving your hand right moves the cursor right
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            raw_gesture = None
            conf = 0.0
            hand_xy = None

            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label

                coords = np.array(
                    [[[p.x, p.y, p.z] for p in hand_lms.landmark]], dtype=np.float32
                )
                labels, confs = predict_gestures(bundle, coords, [w], [h], [handedness])
                conf = float(confs[0])
                if conf >= args.min_confidence:
                    raw_gesture = labels[0]

                xs = [hand_lms.landmark[i].x for i in PALM_IDS]
                ys = [hand_lms.landmark[i].y for i in PALM_IDS]
                # Both axes in frame-width units, so the same physical hand movement gives the same
                # cursor distance horizontally and vertically.
                hand_xy = (sum(xs) / len(xs), (sum(ys) / len(ys)) * h / w)

                drawing.draw_landmarks(frame, hand_lms, connections)

            stable = stabilizer.update(raw_gesture)
            now = time.time()

            status = mouse.update(stable, hand_xy, now)

            fired = one_shot.update(stable, now)
            if (
                fired is not None
                and fired not in reserved
                and mouse.enabled
                and not mouse.dragging
            ):
                description, action = ONE_SHOT_ACTIONS[fired]
                try:
                    action(backend)
                    mouse.flash(description, now)
                except Exception as exc:
                    print(f"[warn] action '{description}' failed: {exc}")

            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt

            draw_overlay(
                frame,
                raw_gesture,
                conf,
                stable,
                status,
                mouse.fist_progress(now),
                mouse,
                fps,
                now,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                mouse.enabled = not mouse.enabled
                if not mouse.enabled:
                    mouse.release_all()
                print(f"[info] mouse control {'ON' if mouse.enabled else 'OFF'}")

    except KeyboardInterrupt:
        print("[info] stopping")
    finally:
        try:
            mouse.release_all()  # never leave the button pressed
        except Exception:
            pass
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        backend.close()


if __name__ == "__main__":
    main()
