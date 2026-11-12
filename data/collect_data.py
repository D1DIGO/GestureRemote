import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT = "dataset/gestures.csv"

LABELS = {
    ord("1"): "OPEN_PALM",
    ord("2"): "FIST",
    ord("3"): "VICTORY",
    ord("4"): "POINTING",
    ord("5"): "THUMB_UP",
}

os.makedirs("dataset", exist_ok=True)

BaseOptions = python.BaseOptions

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

detector = vision.HandLandmarker.create_from_options(options)


def extract_features(hand):

    points = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    # Translation normalization
    points = points - points[0]

    # Scale normalization
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        return None

    points = points / scale

    return points.flatten()


rows = []

if os.path.exists(OUTPUT):
    old = pd.read_csv(OUTPUT)
    rows = old.values.tolist()

cap = cv2.VideoCapture(0)

last_timestamp = 0

print("Dataset collector started.")
print("Press 1-5 to capture a sample.")
print("Press Q to quit.")

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.monotonic() * 1000)
        timestamp = max(timestamp, last_timestamp + 1)
        last_timestamp = timestamp

        result = detector.detect_for_video(image, timestamp)

        status = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            status = "HAND DETECTED"

            features = extract_features(hand)

            key = cv2.waitKey(1) & 0xFF

            if key in LABELS and features is not None:

                label = LABELS[key]

                rows.append([label] + features.tolist())

                columns = ["label"] + [f"f{i}" for i in range(63)]

                pd.DataFrame(rows, columns=columns).to_csv(OUTPUT, index=False)

                print(f"Saved {label}. " f"Total samples: {len(rows)}")

        else:
            key = cv2.waitKey(1) & 0xFF

        cv2.putText(
            frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            "1: Palm  2: Fist  3: Victory",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Dataset Collector", frame)

        if key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
