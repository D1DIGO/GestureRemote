#!/usr/bin/env python3
"""
extract_landmarks.py

Runs HaGRID images through MediaPipe Hands and writes the 21 hand landmarks of
the annotated gesture hand to a CSV file.

Expected folder layout:
    <root>/hagrid_30k/train_val_{class}/{ID}.jpg
    <root>/ann_train_val/{class}.json

Usage:
    pip install mediapipe opencv-python
    python extract_landmarks.py
    python extract_landmarks.py --root dataset/hagrid-sample-30k-384p --out landmarks.csv
    python extract_landmarks.py --limit-per-class 200 --out test_landmarks.csv   # quick test

CSV columns:
    id, class, img_w, img_h, handedness, handedness_score, x0, y0, z0, ..., x20, y20, z20
x and y are MediaPipe's normalized image coordinates (0..1), z is MediaPipe's
relative depth (wrist = 0). img_w / img_h are stored so the training script can
convert back to pixel proportions (avoids distorting the hand shape on
non-square images).
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

NUM_LANDMARKS = 21
# The detected hand's landmark centroid must lie inside the annotated gesture
# bounding box, enlarged by this fraction on each side.
BBOX_PADDING = 0.25

_hands = None  # one MediaPipe Hands instance per worker process


def init_worker():
    global _hands
    _hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def process_task(task):
    """Returns (status, class_name, csv_row_or_None)."""
    img_path, cls, img_id, bbox = task

    image = cv2.imread(img_path)
    if image is None:
        return ("unreadable", cls, None)
    img_h, img_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = _hands.process(rgb)
    if not result.multi_hand_landmarks:
        return ("no_hand_detected", cls, None)

    # HaGRID bboxes are normalized [x_top_left, y_top_left, width, height]
    bx, by, bw, bh = bbox
    box_cx = bx + bw / 2.0
    box_cy = by + bh / 2.0
    x_min = bx - bw * BBOX_PADDING
    x_max = bx + bw * (1.0 + BBOX_PADDING)
    y_min = by - bh * BBOX_PADDING
    y_max = by + bh * (1.0 + BBOX_PADDING)

    best = None
    best_dist = float("inf")
    for hand_lms, handedness in zip(
        result.multi_hand_landmarks, result.multi_handedness
    ):
        mean_x = sum(p.x for p in hand_lms.landmark) / NUM_LANDMARKS
        mean_y = sum(p.y for p in hand_lms.landmark) / NUM_LANDMARKS
        if not (x_min <= mean_x <= x_max and y_min <= mean_y <= y_max):
            continue
        dist = (mean_x - box_cx) ** 2 + (mean_y - box_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = (hand_lms, handedness.classification[0])

    if best is None:
        return ("hand_not_in_gesture_bbox", cls, None)

    hand_lms, hand_info = best
    row = [img_id, cls, img_w, img_h, hand_info.label, round(float(hand_info.score), 4)]
    for p in hand_lms.landmark:
        row.extend([p.x, p.y, p.z])
    return ("ok", cls, row)


def csv_header():
    header = ["id", "class", "img_w", "img_h", "handedness", "handedness_score"]
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def build_tasks(root, limit_per_class, include_no_gesture):
    ann_dir = root / "ann_train_val"
    img_root = root / "hagrid_30k"

    if not ann_dir.is_dir():
        sys.exit(f"Annotation folder not found: {ann_dir}")
    if not img_root.is_dir():
        sys.exit(f"Image folder not found: {img_root}")

    tasks = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        cls = ann_path.stem
        if cls == "no_gesture" and not include_no_gesture:
            continue

        folder = img_root / f"train_val_{cls}"
        if not folder.is_dir():
            print(f"[warn] image folder missing for class '{cls}': {folder}")
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        available = {p.stem: p for p in folder.glob("*.jpg")}
        count = 0
        for img_id, info in annotations.items():
            img_file = available.get(img_id)
            if img_file is None:
                continue
            labels = info.get("labels", [])
            bboxes = info.get("bboxes", [])
            matches = [i for i, label in enumerate(labels) if label == cls]
            if not matches or matches[0] >= len(bboxes):
                continue
            tasks.append((str(img_file), cls, img_id, bboxes[matches[0]]))
            count += 1
            if limit_per_class and count >= limit_per_class:
                break

        print(f"[info] {cls}: {count} images queued")

    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from HaGRID images."
    )
    parser.add_argument(
        "--root",
        default="dataset/hagrid-sample-30k-384p",
        help="Dataset root containing hagrid_30k/ and ann_train_val/",
    )
    parser.add_argument("--out", default="landmarks.csv", help="Output CSV path")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Max images per class (0 = all). Useful for a quick test run.",
    )
    parser.add_argument(
        "--include-no-gesture",
        action="store_true",
        help="Also process the 'no_gesture' class if its folder exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not hasattr(mp, "solutions"):
        sys.exit(
            "This mediapipe version does not provide mp.solutions. "
            "Install one that does: pip install mediapipe==0.10.14"
        )

    tasks = build_tasks(Path(args.root), args.limit_per_class, args.include_no_gesture)
    if not tasks:
        sys.exit("No images found. Check --root and the folder layout.")

    attempted = Counter(t[1] for t in tasks)
    kept = Counter()
    dropped = Counter()

    out_path = Path(args.out)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] processing {len(tasks)} images with {args.workers} workers")
    start = time.time()

    with open(out_path, "w", newline="", encoding="utf-8") as f, multiprocessing.Pool(
        processes=args.workers, initializer=init_worker
    ) as pool:
        writer = csv.writer(f)
        writer.writerow(csv_header())

        for done, (status, cls, row) in enumerate(
            pool.imap_unordered(process_task, tasks, chunksize=32), start=1
        ):
            if status == "ok":
                writer.writerow(row)
                kept[cls] += 1
            else:
                dropped[status] += 1

            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.time() - start
                print(f"[progress] {done}/{len(tasks)} images ({elapsed:.0f}s elapsed)")

    print("\nDone. Landmarks kept per class:")
    for cls in sorted(attempted):
        print(f"  {cls:<22} {kept[cls]:>6} / {attempted[cls]:<6}")
    print(f"\nTotal rows written: {sum(kept.values())} -> {out_path}")
    if dropped:
        print("Dropped images by reason:")
        for reason, n in dropped.most_common():
            print(f"  {reason}: {n}")


if __name__ == "__main__":
    main()
