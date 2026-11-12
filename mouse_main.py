import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from controller.mouse_controller import MouseController

# ---------------- CONFIGURATION ----------------

mouse = MouseController()

MODEL_PATH = "models/hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Region of the camera image used for mouse movement
MOUSE_MARGIN = 0.15

# Time before a fist becomes a drag
DRAG_DELAY = 0.45

# Smoothing: lower = smoother, higher = faster
SMOOTHING = 0.35

# Set this after testing palm orientation
PALM_NORMAL_SIGN = 1

SCREEN_WIDTH, SCREEN_HEIGHT = mouse.width, mouse.height


# ---------------- MEDIAPIPE ----------------

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

detector = HandLandmarker.create_from_options(options)


# ---------------- STATE ----------------

previous_mouse = None

fist_start_time = None

dragging = False
mouse_down = False

last_timestamp = 0


# ---------------- GEOMETRY ----------------


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_landmarks(hand):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def is_finger_extended(points, tip, pip):
    """
    Compare the fingertip and PIP joint
    relative to the wrist.

    This is a basic heuristic, not a
    universal anatomical rule.
    """
    wrist = points[0]

    tip_dist = distance(points[tip], wrist)
    pip_dist = distance(points[pip], wrist)

    return tip_dist > pip_dist * 1.15


def get_finger_states(points):
    """
    Returns states for:
    index, middle, ring, pinky.

    Thumb is handled separately.
    """

    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]

    states = [is_finger_extended(points, tip, pip) for tip, pip in fingers]

    return states


def is_open_hand(points):
    states = get_finger_states(points)

    return all(states)


def is_closed_hand(points):
    states = get_finger_states(points)

    return not any(states)


def palm_faces_camera(points):
    """
    Estimate whether the palm plane faces
    approximately toward the camera.

    Uses the palm normal from landmarks
    0 (wrist), 5 (index MCP), 17 (pinky MCP).

    The sign depends on hand orientation
    and camera mirroring. Calibrate it.
    """

    wrist = points[0]
    index_mcp = points[5]
    pinky_mcp = points[17]

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return False

    normal = normal / norm

    # z is the depth direction
    # The sign must be calibrated.
    return PALM_NORMAL_SIGN * normal[2] > 0.65


# ---------------- MOUSE MOVEMENT ----------------


def move_mouse(points):
    global previous_mouse

    # Use index fingertip
    x = float(points[8][0])
    y = float(points[8][1])

    # Ignore edges of camera frame
    x = (x - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)
    y = (y - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)

    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    # Mirror horizontally for intuitive control
    target_x = x * SCREEN_WIDTH
    target_y = y * SCREEN_HEIGHT

    target = np.array([target_x, target_y])

    if previous_mouse is None:
        previous_mouse = target
    else:
        previous_mouse = previous_mouse * (1 - SMOOTHING) + target * SMOOTHING

    mouse.move_to(int(previous_mouse[0]), int(previous_mouse[1]))


# ---------------- ACTIONS ----------------


def handle_fist():
    global fist_start_time
    global dragging
    global mouse_down

    current_time = time.monotonic()

    if fist_start_time is None:
        fist_start_time = current_time

    duration = current_time - fist_start_time

    if duration >= DRAG_DELAY and not dragging:
        mouse.mouse_down()
        mouse_down = True
        dragging = True

    if dragging:
        return "DRAGGING"

    return "FIST"


def release_fist():
    global fist_start_time
    global dragging
    global mouse_down

    if dragging:
        if mouse_down:
            mouse.mouse_up()

    elif fist_start_time is not None:
        duration = time.monotonic() - fist_start_time

        # Quick fist = click
        if duration < DRAG_DELAY:
            mouse.click()

    fist_start_time = None
    dragging = False
    mouse_down = False


# ---------------- MAIN LOOP ----------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Hand mouse started.")
print("Press Q to quit.")

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Mirror the camera image
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.monotonic() * 1000)

        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(mp_image, timestamp)

        status = "NO HAND"

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            points = get_landmarks(hand)

            # Draw hand landmarks
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for connection in connections:
                start = connection.start
                end = connection.end

                h, w = frame.shape[:2]

                p1 = (int(points[start][0] * w), int(points[start][1] * h))

                p2 = (int(points[end][0] * w), int(points[end][1] * h))

                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            open_hand = is_open_hand(points)
            closed_hand = is_closed_hand(points)
            palm_facing = palm_faces_camera(points)

            if open_hand and palm_facing:

                release_fist()
                move_mouse(points)

                status = "OPEN PALM - MOVING"

            elif closed_hand:

                status = handle_fist()

            else:

                release_fist()
                status = "OTHER GESTURE"

        else:
            release_fist()

        cv2.putText(
            frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("Hand Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always release a held mouse button
    if mouse_down:
        mouse.mouse_up()

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
