"""
Create synthetic defective samples from a single good image.
Outputs multiple defect types for testing/demo purposes.
"""
import os
import argparse
from pathlib import Path
import random
import csv
import re

import numpy as np
import cv2
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img: np.ndarray, path: str) -> None:
    Image.fromarray(img).save(path)


def compute_object_mask(img: np.ndarray) -> np.ndarray:
    """Estimate object mask by separating it from the background."""
    h, w = img.shape[:2]
    pad_h = max(5, h // 20)
    pad_w = max(5, w // 20)
    corners = [
        img[:pad_h, :pad_w],
        img[:pad_h, -pad_w:],
        img[-pad_h:, :pad_w],
        img[-pad_h:, -pad_w:],
    ]
    bg = np.mean([c.reshape(-1, 3) for c in corners], axis=0).mean(axis=0)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    bg_lab = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_RGB2LAB)[0, 0]
    diff = np.linalg.norm(lab.astype(np.float32) - bg_lab.astype(np.float32), axis=2)

    mask = (diff > 15).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    return mask


def sample_point_in_mask(mask: np.ndarray, max_tries: int = 50) -> tuple:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return w // 2, h // 2
    for _ in range(max_tries):
        idx = random.randint(0, len(xs) - 1)
        return int(xs[idx]), int(ys[idx])
    idx = random.randint(0, len(xs) - 1)
    return int(xs[idx]), int(ys[idx])


def sample_circle_in_mask(mask: np.ndarray, min_r: int, max_r: int, max_tries: int = 50) -> tuple:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = int(dist.max())
    if max_dist <= 0:
        x, y = sample_point_in_mask(mask)
        return (x, y), min_r

    min_r = min(min_r, max_dist)
    max_r = min(max_r, max_dist)
    for _ in range(max_tries):
        r = random.randint(min_r, max_r)
        ys, xs = np.where(dist >= r)
        if len(xs) == 0:
            break
        idx = random.randint(0, len(xs) - 1)
        return (int(xs[idx]), int(ys[idx])), r

    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    r = min(max_r, int(dist[y, x]))
    return (int(x), int(y)), max(r, 1)


def sample_rect_in_mask(mask: np.ndarray, min_w: int, max_w: int, min_h: int, max_h: int, max_tries: int = 50) -> tuple:
    h, w = mask.shape[:2]
    for _ in range(max_tries):
        rect_w = random.randint(min_w, min(max_w, w - 2))
        rect_h = random.randint(min_h, min(max_h, h - 2))
        x = random.randint(1, w - rect_w - 1)
        y = random.randint(1, h - rect_h - 1)
        roi = mask[y:y+rect_h, x:x+rect_w]
        if roi.size > 0 and (roi.mean() > 0.9):
            return x, y, x + rect_w, y + rect_h

    x, y = sample_point_in_mask(mask)
    rect_w = min_w
    rect_h = min_h
    x1 = max(0, x - rect_w // 2)
    y1 = max(0, y - rect_h // 2)
    x2 = min(w - 1, x1 + rect_w)
    y2 = min(h - 1, y1 + rect_h)
    return x1, y1, x2, y2


def add_scratch(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(12):
        x1, y1 = sample_point_in_mask(mask)
        x2, y2 = sample_point_in_mask(mask)
        color = (0, 0, 0)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness=random.randint(4, 8))
    return out


def add_crack(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    points = [sample_point_in_mask(mask) for _ in range(12)]
    for i in range(len(points) - 1):
        cv2.line(out, points[i], points[i+1], (0, 0, 0), thickness=6)
    return out


def add_dent(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    min_r = max(12, min(h, w) // 10)
    max_r = max(min_r + 5, min(h, w) // 5)
    center, radius = sample_circle_in_mask(mask, min_r, max_r)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    blurred = cv2.GaussianBlur(out, (31, 31), 0)
    out[mask == 255] = blurred[mask == 255]
    cv2.circle(out, center, radius, (0, 0, 0), 6)
    return out


def add_missing_label(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    x1, y1, x2, y2 = sample_rect_in_mask(
        mask,
        min_w=max(20, w // 8),
        max_w=max(40, w // 3),
        min_h=max(20, h // 10),
        max_h=max(40, h // 4),
    )
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(out, (x1-10, y1-10), (x2+10, y2+10), (0, 0, 0), 4)
    return out


def add_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([17, 19, 21])
    return cv2.GaussianBlur(img, (k, k), 0)


def add_color_shift(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    shift = np.array([random.uniform(-60, 60), random.uniform(-50, 50), random.uniform(-50, 50)])
    scale = np.array([random.uniform(0.7, 1.4), random.uniform(0.7, 1.4), random.uniform(0.7, 1.4)])
    out = np.clip(out * scale + shift, 0, 255).astype(np.uint8)
    # Add a strong tint overlay
    tint = np.full_like(out, (30, 0, 120), dtype=np.uint8)
    out = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
    return out


def add_asymmetry(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    left_w = w // 2
    right_w = w - left_w
    left = out[:, :left_w]
    right = cv2.flip(left, 1)
    if right.shape[1] != right_w:
        right = cv2.resize(right, (right_w, h), interpolation=cv2.INTER_LINEAR)
    out[:, left_w:] = right
    # Introduce a small local change on one side
    cx, cy = sample_point_in_mask(mask)
    cv2.circle(out, (cx, cy), 30, (0, 0, 0), -1)
    x1, y1, x2, y2 = sample_rect_in_mask(
        mask,
        min_w=max(30, w // 6),
        max_w=max(50, w // 3),
        min_h=max(30, h // 6),
        max_h=max(60, h // 3),
    )
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return out


def generate_defects(img: np.ndarray) -> dict:
    mask = compute_object_mask(img)
    return {
        "defect_scratch": add_scratch(img, mask),
        "defect_crack": add_crack(img, mask),
        "defect_dent": add_dent(img, mask),
        "defect_missing_label": add_missing_label(img, mask),
        "defect_blur": add_blur(img),
        "defect_color_shift": add_color_shift(img),
        "defect_asymmetric": add_asymmetry(img, mask),
    }


def main():
    parser = argparse.ArgumentParser(description="Create synthetic defective samples from a good image")
    parser.add_argument("--input", type=str, default="data/water_bottles/train/good/001.jpeg",
                        help="Path to a good image (will also be copied as GOOD into test)")
    parser.add_argument("--append", action="store_true",
                        help="Do not delete existing test images; append new defects")
    parser.add_argument("--add-defects", type=int, default=6,
                        help="Number of defective images to create")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Fallback: use the first image in the training good folder
        train_good_dir = Path("data/water_bottles/train/good")
        candidates = sorted(list(train_good_dir.glob("*.jpg")) + list(train_good_dir.glob("*.jpeg")) + list(train_good_dir.glob("*.png")))
        if candidates:
            input_path = candidates[0]
            print(f"Input not found. Using: {input_path}")
        else:
            raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir = Path("data/water_bottles/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(str(input_path))
    labels_path = output_dir / "labels.csv"
    labels = []

    if args.append:
        if labels_path.exists():
            with open(labels_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        labels.append((row[0], int(row[1])))
    else:
        # Clean test folder so only the new images remain
        for existing in output_dir.glob("*.jpg"):
            existing.unlink(missing_ok=True)
        for existing in output_dir.glob("*.jpeg"):
            existing.unlink(missing_ok=True)
        for existing in output_dir.glob("*.png"):
            existing.unlink(missing_ok=True)

        # Save 4 good samples (same image, different names)
        for i in range(1, 5):
            good_out_path = output_dir / f"GOOD_{i:02d}.jpg"
            save_image(img, str(good_out_path))
            print(f"Saved: {good_out_path}")
            labels.append((good_out_path.name, 0))

    # Save defective samples (strong, distinct defects)
    defects = generate_defects(img)
    defect_order = [
        "defect_crack",
        "defect_dent",
        "defect_missing_label",
        "defect_scratch",
        "defect_color_shift",
        "defect_asymmetric",
    ]
    existing_files = {p.name for p in output_dir.glob("*.jpg")}

    next_index = {}
    for name in defect_order:
        pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.jpg$", re.IGNORECASE)
        max_idx = 0
        for fname in existing_files:
            match = pattern.match(fname)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        next_index[name] = max_idx + 1

    count = max(1, args.add_defects)
    for i in range(count):
        name = defect_order[i % len(defect_order)]
        defect_img = defects[name]
        idx = next_index[name]
        next_index[name] += 1
        out_path = output_dir / f"{name}_{idx:02d}.jpg"
        save_image(defect_img, str(out_path))
        print(f"Saved: {out_path}")
        labels.append((out_path.name, 1))

    # Write labels.csv
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in labels:
            writer.writerow(row)
    print(f"Saved labels: {labels_path}")

    print("\nDone. Defective samples created.")


if __name__ == "__main__":
    main()
