from collections import deque
import time
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from controller.mouse_controller import MouseController

# ---------------- CONFIGURATION ----------------

mouse = MouseController()

MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/asl_classifier.joblib"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MOUSE_MARGIN = 0.15
DRAG_DELAY = 0.45
SMOOTHING = 0.35
PALM_NORMAL_SIGN = 1

# ASL Classifier Settings
ASL_CONFIDENCE_THRESHOLD = 0.6
ASL_HOLD_FRAMES = 8  # Consecutive frames required to confirm a sign
ASL_COOLDOWN = 1.0  # Seconds before the same action can fire again
FLIP_X_FOR_ASL = True  # Un-mirrors X so a right hand on a flipped webcam matches right-hand training data

SCREEN_WIDTH, SCREEN_HEIGHT = mouse.width, mouse.height


# ---------------- LOAD ASL MODEL & MEDIAPIPE ----------------

asl_model = joblib.load(CLASSIFIER_PATH)

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

asl_history = deque(maxlen=ASL_HOLD_FRAMES)
last_asl_trigger_time = 0.0


# ---------------- GEOMETRY ----------------


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_landmarks(hand):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def is_finger_extended(points, tip, pip):
    wrist = points[0]
    tip_dist = distance(points[tip], wrist)
    pip_dist = distance(points[pip], wrist)
    return tip_dist > pip_dist * 1.15


def get_finger_states(points):
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return [is_finger_extended(points, tip, pip) for tip, pip in fingers]


def is_open_hand(points):
    return all(get_finger_states(points))


def palm_faces_camera(points):
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
    return PALM_NORMAL_SIGN * normal[2] > 0.65


# ---------------- MOUSE MOVEMENT & FIST ----------------


def move_mouse(points):
    global previous_mouse

    x = (points[0][0] - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)
    y = (points[0][1] - MOUSE_MARGIN) / (1 - 2 * MOUSE_MARGIN)

    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    target = np.array([x * SCREEN_WIDTH, y * SCREEN_HEIGHT])

    if previous_mouse is None:
        previous_mouse = target
    else:
        previous_mouse = previous_mouse * (1 - SMOOTHING) + target * SMOOTHING

    mouse.move_to(int(previous_mouse[0]), int(previous_mouse[1]))


def handle_fist():
    global fist_start_time, dragging, mouse_down

    current_time = time.monotonic()
    if fist_start_time is None:
        fist_start_time = current_time

    if (current_time - fist_start_time) >= DRAG_DELAY and not dragging:
        mouse.mouse_down()
        mouse_down = True
        dragging = True

    return "DRAGGING" if dragging else "FIST"


def release_fist():
    global fist_start_time, dragging, mouse_down

    if dragging:
        if mouse_down:
            mouse.mouse_up()
    elif fist_start_time is not None:
        if (time.monotonic() - fist_start_time) < DRAG_DELAY:
            mouse.click()

    fist_start_time = None
    dragging = False
    mouse_down = False


# ---------------- ASL CLASSIFICATION & TASKS ----------------

FIST_SIGNS = {
    "A",
    "E",
    "S",
    "T",
}
FIST_CONFIRM_FRAMES = 2  # Brief debounce to prevent accidental clicks/drops
fist_confirm_count = 0


def extract_asl_features(points):
    if points.shape != (21, 3):
        return None

    norm_points = points - points[0]

    if FLIP_X_FOR_ASL:
        norm_points[:, 0] = -norm_points[:, 0]

    scale = np.linalg.norm(norm_points[9])
    if scale < 1e-6:
        return None

    norm_points /= scale
    return norm_points.flatten().reshape(1, -1)


def execute_asl_task(letter):
    """
    One-shot actions for non-dragging ASL signs.
    """
    print(f"[TASK TRIGGERED] ASL Letter: {letter}")

    if letter == "C":
        print("CCCCCC")
    elif letter == "V":
        print("VVVVVVVVVV")
    elif letter == "W":
        print("WWWWWWWW")
    elif letter == "Y":
        print("YYYYYYYYYYY")


def process_asl_gesture(points):
    global last_asl_trigger_time, fist_confirm_count

    features = extract_asl_features(points)
    if features is None:
        asl_history.clear()
        fist_confirm_count = 0
        release_fist()
        return "ASL: INVALID"

    probabilities = asl_model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    predicted_letter = asl_model.classes_[best_index]

    # 1. CONTINUOUS CLICK / DRAG STATE ("A" or "E")
    if predicted_letter in FIST_SIGNS and confidence >= ASL_CONFIDENCE_THRESHOLD:
        asl_history.clear()
        fist_confirm_count += 1

        if fist_confirm_count >= FIST_CONFIRM_FRAMES:
            fist_status = handle_fist()
            return f"{fist_status} ({predicted_letter}: {confidence:.0%})"

        return (
            f"HOLDING {predicted_letter} ({fist_confirm_count}/{FIST_CONFIRM_FRAMES})"
        )

    # Not holding A/E anymore -> release any active click/drag
    fist_confirm_count = 0
    release_fist()

    # 2. ONE-SHOT ASL ACTIONS (C, V, W, Y, etc.)
    if confidence < ASL_CONFIDENCE_THRESHOLD:
        asl_history.clear()
        return f"ASL: {predicted_letter}? ({confidence:.0%})"

    asl_history.append(predicted_letter)

    if len(asl_history) == ASL_HOLD_FRAMES and len(set(asl_history)) == 1:
        now = time.monotonic()
        if (now - last_asl_trigger_time) >= ASL_COOLDOWN:
            execute_asl_task(predicted_letter)
            last_asl_trigger_time = now
            asl_history.clear()
            return f"ACTION: {predicted_letter} ({confidence:.0%})"

    return f"ASL: {predicted_letter} ({confidence:.0%}) [{len(asl_history)}/{ASL_HOLD_FRAMES}]"


# ---------------- MAIN LOOP ----------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Hand mouse + ASL classifier started. Press Q to quit.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(mp_image, timestamp)
        status = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            points = get_landmarks(hand)

            # Draw hand landmarks
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            h, w = frame.shape[:2]
            for connection in connections:
                p1 = (
                    int(points[connection.start][0] * w),
                    int(points[connection.start][1] * h),
                )
                p2 = (
                    int(points[connection.end][0] * w),
                    int(points[connection.end][1] * h),
                )
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            open_hand = is_open_hand(points)
            palm_facing = palm_faces_camera(points)

            if open_hand and palm_facing:
                fist_confirm_count = 0
                release_fist()
                asl_history.clear()
                status = "OPEN PALM - MOVING"
                move_mouse(points)
            else:
                # Let process_asl_gesture manage when to hold or release the mouse button
                move_mouse(points)
                status = process_asl_gesture(points)

        else:
            fist_confirm_count = 0
            release_fist()
            asl_history.clear()

        cv2.putText(
            frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Hand Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if mouse_down:
        mouse.mouse_up()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
