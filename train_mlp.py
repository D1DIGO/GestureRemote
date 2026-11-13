#!/usr/bin/env python3
"""
train_mlp_sklearn.py

Trains a small MLP gesture classifier with scikit-learn (CPU only, no torch) on
the landmarks written by extract_landmarks.py.

The HaGRID annotation files (ann_train_val/{class}.json) are used to look up the
anonymized user_id of every image, so the train / val / test split is done BY
PERSON. This stops the model from "cheating" by seeing the same person in both
training and test data.

Usage:
    pip install numpy pandas scikit-learn joblib
    python train_mlp_sklearn.py
    python train_mlp_sklearn.py --csv landmarks.csv --ann-dir dataset/hagrid-sample-30k-384p/ann_train_val

Outputs:
    gesture_mlp.joblib     dict with the model, scaler and class names
    gesture_classes.json   class names in output order
    confusion_matrix.csv   test-set confusion matrix

IMPORTANT: at inference time (e.g. webcam) build the model input with
landmarks_to_points() below (or simply call predict_gestures()), using the raw
MediaPipe landmarks (normalized x, y, z), the frame width/height in pixels, and
the handedness label MediaPipe reports ("Left" / "Right").

Loading the model elsewhere:
    import joblib
    bundle = joblib.load("gesture_mlp.joblib")
    labels, confidences = predict_gestures(bundle, coords, img_w, img_h, handedness)
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

NUM_LANDMARKS = 21
NUM_POINTS = NUM_LANDMARKS - 1  # wrist is removed after normalization (always 0,0,0)


# --------------------------------------------------------------------------- #
# Feature engineering (reuse this exact function at inference time)
# --------------------------------------------------------------------------- #
def landmarks_to_points(coords, img_w, img_h, handedness):
    """
    coords:      float array (N, 21, 3), raw MediaPipe x, y (0..1) and z
    img_w/img_h: arrays (N,), image size in pixels
    handedness:  array (N,) of strings, "Left" or "Right" (as reported by MediaPipe)

    Returns float32 array (N, 20, 3):
      - x/y/z converted to a consistent pixel-proportional scale
      - translated so the wrist is the origin
      - "Left" hands mirrored so all hands look like the same hand
      - divided by the hand size (max 2D distance from the wrist)
      - wrist row removed
    Orientation is NOT normalized on purpose, because gestures such as
    like/dislike or peace/peace_inverted differ only by rotation.
    """
    pts = np.asarray(coords, dtype=np.float32).copy()
    img_w = np.asarray(img_w, dtype=np.float32)
    img_h = np.asarray(img_h, dtype=np.float32)
    handedness = np.asarray(handedness)

    pts[:, :, 0] *= img_w[:, None]
    pts[:, :, 1] *= img_h[:, None]
    pts[:, :, 2] *= img_w[:, None]  # MediaPipe z uses roughly the same scale as x

    pts = pts - pts[:, :1, :]  # wrist -> origin

    left = handedness == "Left"
    pts[left, :, 0] = -pts[left, :, 0]  # mirror left hands

    scale = np.linalg.norm(pts[:, :, :2], axis=2).max(axis=1)
    pts = pts / np.maximum(scale, 1e-6)[:, None, None]

    return pts[:, 1:, :].astype(np.float32)


def predict_gestures(bundle, coords, img_w, img_h, handedness):
    """
    Convenience inference helper.
    bundle: dict loaded from gesture_mlp.joblib
    coords: (N, 21, 3) raw MediaPipe landmarks; img_w, img_h, handedness: length-N sequences
    Returns (list of predicted class names, array of confidences).
    """
    points = landmarks_to_points(coords, img_w, img_h, handedness)
    features = bundle["scaler"].transform(points.reshape(len(points), -1))
    proba = bundle["model"].predict_proba(features)
    idx = proba.argmax(axis=1)
    return [bundle["classes"][i] for i in idx], proba.max(axis=1)


# --------------------------------------------------------------------------- #
# Data loading and splitting
# --------------------------------------------------------------------------- #
def load_landmark_csv(csv_path):
    df = pd.read_csv(csv_path)
    coord_cols = [
        f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
    ]
    required = ["id", "class", "img_w", "img_h", "handedness"] + coord_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"CSV is missing columns: {missing[:6]}{' ...' if len(missing) > 6 else ''}"
        )
    df = df.dropna(subset=required).reset_index(drop=True)
    return df, coord_cols


def load_user_ids(ann_dir):
    """Map image id -> anonymized user_id using the HaGRID annotation files."""
    mapping = {}
    ann_dir = Path(ann_dir)
    if not ann_dir.is_dir():
        print(f"[warn] annotation folder not found: {ann_dir}")
        return mapping
    for ann_path in sorted(ann_dir.glob("*.json")):
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        for img_id, info in annotations.items():
            user_id = info.get("user_id")
            if user_id is not None:
                mapping[img_id] = user_id
    return mapping


def split_by_group(groups, val_frac, test_frac, seed):
    """Random split over unique groups (users). Returns boolean masks (train, val, test)."""
    rng = np.random.default_rng(seed)
    users = np.array(sorted(set(groups)))
    rng.shuffle(users)
    n_users = len(users)
    n_test = int(round(n_users * test_frac))
    n_val = int(round(n_users * val_frac))
    test_users = set(users[:n_test])
    val_users = set(users[n_test : n_test + n_val])

    is_test = np.array([g in test_users for g in groups])
    is_val = np.array([g in val_users for g in groups])
    is_train = ~(is_test | is_val)
    return is_train, is_val, is_test


# --------------------------------------------------------------------------- #
# Augmentation (numpy, regenerated every epoch)
# --------------------------------------------------------------------------- #
def augment_points(points, rng, max_rot_deg, scale_jitter, noise_std):
    """points: (N, 20, 3). Small in-plane rotation, scale jitter, gaussian noise."""
    n = points.shape[0]
    theta = rng.uniform(-1.0, 1.0, size=n) * math.radians(max_rot_deg)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]

    xs, ys, zs = points[..., 0], points[..., 1], points[..., 2]
    new_x = xs * cos_t - ys * sin_t
    new_y = xs * sin_t + ys * cos_t

    scale = 1.0 + rng.uniform(-1.0, 1.0, size=(n, 1)) * scale_jitter
    out = np.stack([new_x * scale, new_y * scale, zs * scale], axis=-1)
    out = out + rng.normal(0.0, noise_std, size=out.shape)
    return out.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a scikit-learn MLP gesture classifier on MediaPipe landmarks."
    )
    p.add_argument(
        "--csv", default="landmarks.csv", help="CSV from extract_landmarks.py"
    )
    p.add_argument(
        "--ann-dir",
        default="dataset/hagrid-sample-30k-384p/ann_train_val",
        help="HaGRID annotation folder (used for user_id based splitting)",
    )
    p.add_argument(
        "--out", default="gesture_mlp.joblib", help="Model bundle output path"
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--alpha", type=float, default=1e-4, help="L2 regularization strength"
    )
    p.add_argument(
        "--hidden", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes"
    )
    p.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help="Freshly augmented copies of the training set added every epoch (0 = no augmentation)",
    )
    p.add_argument(
        "--max-rot",
        type=float,
        default=10.0,
        help="Max rotation augmentation in degrees (keep small: orientation matters)",
    )
    p.add_argument("--scale-jitter", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience (epochs)"
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- load data ---------------------------------------------------------
    df, coord_cols = load_landmark_csv(args.csv)
    classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_class_ids = np.arange(len(classes))
    print(f"[info] {len(df)} samples, {len(classes)} classes: {classes}")

    user_map = load_user_ids(args.ann_dir)
    df["group"] = df["id"].map(user_map)
    n_no_user = int(df["group"].isna().sum())
    if n_no_user:
        print(
            f"[warn] {n_no_user} samples have no user_id in the annotations; "
            f"each is treated as its own group"
        )
        df["group"] = df["group"].fillna(df["id"])

    coords = df[coord_cols].to_numpy(dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3)
    points = landmarks_to_points(
        coords,
        df["img_w"].to_numpy(),
        df["img_h"].to_numpy(),
        df["handedness"].to_numpy(),
    )  # (N, 20, 3)
    labels = df["class"].map(class_to_idx).to_numpy(dtype=np.int64)

    # ---- split by user -----------------------------------------------------
    is_train, is_val, is_test = split_by_group(
        df["group"].to_numpy(), args.val_frac, args.test_frac, args.seed
    )
    print(
        f"[info] split sizes -> train: {is_train.sum()}, val: {is_val.sum()}, test: {is_test.sum()}"
    )
    if is_train.sum() == 0 or is_val.sum() == 0 or is_test.sum() == 0:
        sys.exit(
            "One of the splits is empty. Extract more data or adjust --val-frac / --test-frac."
        )

    pts_train, y_train = points[is_train], labels[is_train]
    pts_val, y_val = points[is_val], labels[is_val]
    pts_test, y_test = points[is_test], labels[is_test]

    # Scaler is fitted on the (un-augmented) training features only.
    scaler = StandardScaler()
    scaler.fit(pts_train.reshape(len(pts_train), -1))

    def featurize(pts):
        return scaler.transform(pts.reshape(len(pts), -1))

    x_val = featurize(pts_val)
    x_test = featurize(pts_test)

    # ---- model -------------------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden),
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.lr,
        shuffle=True,
        random_state=args.seed,
    )

    # ---- training loop (one epoch per partial_fit call) --------------------
    best_model = None
    best_val_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if args.augment_copies > 0:
            parts = [pts_train]
            label_parts = [y_train]
            for _ in range(args.augment_copies):
                parts.append(
                    augment_points(
                        pts_train, rng, args.max_rot, args.scale_jitter, args.noise
                    )
                )
                label_parts.append(y_train)
            epoch_pts = np.concatenate(parts, axis=0)
            epoch_y = np.concatenate(label_parts, axis=0)
        else:
            epoch_pts, epoch_y = pts_train, y_train

        model.partial_fit(featurize(epoch_pts), epoch_y, classes=all_class_ids)

        val_proba = model.predict_proba(x_val)
        val_pred = val_proba.argmax(axis=1)
        val_acc = float((val_pred == y_val).mean())
        val_loss = log_loss(y_val, val_proba, labels=all_class_ids)
        print(
            f"epoch {epoch:03d} | train loss {model.loss_:.4f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"[info] early stopping (no val improvement for {args.patience} epochs)"
                )
                break

    print(f"[info] best val acc {best_val_acc:.4f} at epoch {best_epoch}")
    model = best_model

    # ---- test evaluation ---------------------------------------------------
    test_proba = model.predict_proba(x_test)
    test_pred = test_proba.argmax(axis=1)
    test_acc = float((test_pred == y_test).mean())
    test_loss = log_loss(y_test, test_proba, labels=all_class_ids)
    print(f"\nTEST accuracy: {test_acc:.4f} (log loss {test_loss:.4f})\n")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=list(all_class_ids),
            target_names=classes,
            digits=3,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, test_pred, labels=list(all_class_ids))
    cm_df = pd.DataFrame(
        cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    )
    cm_df.to_csv("confusion_matrix.csv")

    # ---- save --------------------------------------------------------------
    bundle = {
        "model": model,
        "scaler": scaler,
        "classes": classes,
        "hidden": list(args.hidden),
    }
    joblib.dump(bundle, args.out)
    with open("gesture_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"Saved model bundle -> {args.out}")
    print("Saved class names -> gesture_classes.json")
    print("Saved confusion matrix -> confusion_matrix.csv")


if __name__ == "__main__":
    main()
