import argparse
import os
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset/asl_alphabet_train/asl_alphabet_train")

LANDMARKER_PATH = Path("models/hand_landmarker.task")

FEATURES_PATH = Path("dataset/asl_landmarks.csv")

CLASSIFIER_PATH = Path("models/asl_classifier.joblib")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================


def create_detector():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_features(hand_landmarks):
    """
    Convert 21 MediaPipe landmarks into 63 normalized features.

    Landmark format:
        x, y, z

    Normalization:
        1. Move wrist to origin.
        2. Normalize by hand size.
        3. Flatten into a 63-dimensional vector.
    """

    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype=np.float32,
    )

    if points.shape != (21, 3):
        return None

    # Wrist becomes the origin.
    points = points - points[0]

    # Distance from wrist to middle finger MCP.
    scale = np.linalg.norm(points[9])

    if scale < 1e-6:
        return None

    points /= scale

    return points.flatten()


def process_image(image_path, detector):
    """
    Read an image, run MediaPipe and return hand features.

    Returns None if no hand is detected.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Cannot read: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    return extract_features(hand)


# ============================================================
# DATASET EXTRACTION
# ============================================================


def extract_dataset(force=False):
    """
    Read images from class folders.

    Example:
        dataset/.../A/A1.jpg -> label A
        dataset/.../B/B1.jpg -> label B

    Save extracted landmarks into a CSV file.
    """

    if FEATURES_PATH.exists() and not force:
        print(f"Dataset already exists: {FEATURES_PATH}")
        print("Use --force to extract it again.")
        return

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    total_images = 0
    detected_images = 0
    skipped_images = 0

    class_dirs = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.is_dir()
        and len(path.name) == 1
        and path.name.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    if not class_dirs:
        raise RuntimeError("No alphabet class folders found.")

    print(f"Found {len(class_dirs)} classes.")

    detector = create_detector()

    try:
        for class_dir in class_dirs:
            label = class_dir.name.upper()

            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"\nProcessing {label}: " f"{len(image_paths)} images")

            for index, image_path in enumerate(image_paths, start=1):
                total_images += 1

                features = process_image(image_path, detector)

                if features is None:
                    skipped_images += 1
                    continue

                row = {"label": label}

                for feature_index, value in enumerate(features):
                    row[f"f{feature_index}"] = float(value)

                rows.append(row)
                detected_images += 1

                if index % 500 == 0:
                    print(f"  Processed {index}/" f"{len(image_paths)}")

    finally:
        detector.close()

    if not rows:
        raise RuntimeError("No hands were detected in the dataset.")

    df = pd.DataFrame(rows)

    df.to_csv(FEATURES_PATH, index=False)

    print("\n========== EXTRACTION RESULTS ==========")
    print(f"Total images:     {total_images}")
    print(f"Detected hands:   {detected_images}")
    print(f"Skipped images:   {skipped_images}")
    print(f"Saved features:   {FEATURES_PATH}")
    print(f"Feature count:    {NUM_FEATURES}")


# ============================================================
# DATASET LOADING
# ============================================================


def load_dataset():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature dataset not found. Run extract first.")

    df = pd.read_csv(FEATURES_PATH)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a label column.")

    feature_columns = [f"f{i}" for i in range(NUM_FEATURES)]

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains invalid values.")

    return X, y


# ============================================================
# MODEL TRAINING
# ============================================================


def train_model():
    X, y = load_dataset()

    classes, counts = np.unique(y, return_counts=True)

    print("\n========== DATASET ==========")
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {list(classes)}")

    print("\nSamples per class:")

    for label, count in zip(classes, counts):
        print(f"{label}: {count}")

    if np.min(counts) < 2:
        raise ValueError(
            "Every class needs at least two samples "
            "for a stratified train/test split."
        )

    # Split dataset into training and testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n========== MODEL CONFIGURATION ==========")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(256, 256),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    batch_size=256,
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=15,
                    alpha=0.0001,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("\nStarting training...")

    model.fit(X_train, y_train)

    print("\nTraining completed.")

    # Evaluate on data that was not used for fitting.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Iterations: {model.named_steps['classifier'].n_iter_}")

    print("\nClassification report:")

    print(classification_report(y_test, predictions, labels=classes, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=classes))

    # Save the complete pipeline.
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, CLASSIFIER_PATH)

    print(f"\nModel saved to: {CLASSIFIER_PATH}")


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================


def predict_image(image_path, expected_label=None):
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError("Trained classifier not found. Run train first.")

    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_PATH}")

    model = joblib.load(CLASSIFIER_PATH)

    detector = create_detector()

    try:
        features = process_image(image_path, detector)
    finally:
        detector.close()

    if features is None:
        print("\nNo hand detected in the image.")
        return

    features = features.reshape(1, -1)

    # Predict letter.
    prediction = model.predict(features)[0]

    # Get classifier confidence scores.
    probabilities = model.predict_proba(features)[0]

    classifier = model.named_steps["classifier"]

    best_index = np.argmax(probabilities)

    confidence = float(probabilities[best_index])

    print("\n========== PREDICTION ==========")
    print(f"Predicted letter: {prediction}")
    print(f"Confidence score: {confidence:.2%}")

    # Show top 3 predictions.
    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\nTop predictions:")

    for index in top_indices:
        print(f"{classifier.classes_[index]}: " f"{probabilities[index]:.2%}")

    if expected_label is not None:
        expected_label = expected_label.upper()

        print("\n========== LABEL VERIFICATION ==========")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {prediction}")

        if prediction == expected_label:
            print("RESULT: CORRECT")
        else:
            print("RESULT: INCORRECT")


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="ASL Alphabet Recognition using MediaPipe + sklearn"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extract command.
    extract_parser = subparsers.add_parser(
        "extract", help="Extract MediaPipe landmarks from dataset images."
    )

    extract_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract dataset even if CSV already exists.",
    )

    # Train command.
    subparsers.add_parser("train", help="Train and evaluate the classifier.")

    # Predict command.
    predict_parser = subparsers.add_parser(
        "predict", help="Predict the ASL letter in an image."
    )

    predict_parser.add_argument("image", type=str, help="Path to the image.")

    predict_parser.add_argument(
        "--expected",
        type=str,
        default=None,
        help="Expected letter to verify the prediction.",
    )

    args = parser.parse_args()

    if args.command == "extract":
        extract_dataset(force=args.force)

    elif args.command == "train":
        train_model()

    elif args.command == "predict":
        predict_image(args.image, expected_label=args.expected)


if __name__ == "__main__":
    main()
