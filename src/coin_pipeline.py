from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import canny
from sklearn.metrics import classification_report
from sklearn.svm import SVC

COIN_LABELS = {
    "1kr_obverse": 0,
    "1kr_reverse": 1,
    "5kr_obverse": 2,
    "5kr_reverse": 3,
    "10kr_obverse": 4,
    "10kr_reverse": 5,
    "25kr_obverse": 6,
    "25kr_reverse": 7,
    "50kr_obverse": 8,
    "50kr_reverse": 9,
    "1TL_obverse": 10,
    "1TL_reverse": 11,
}
ID_TO_LABEL = {value: key for key, value in COIN_LABELS.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def hough_circle_transform(edge_map: np.ndarray, min_radius: int, max_radius: int, threshold: int, angle_step: int = 10, radius_step: int = 3) -> list[tuple[int, int, int, int]]:
    if min_radius <= 0 or max_radius <= min_radius:
        raise ValueError("Expected 0 < min_radius < max_radius")
    y_edges, x_edges = np.nonzero(edge_map)
    if len(x_edges) == 0:
        return []
    angles = np.deg2rad(np.arange(0, 360, angle_step))
    cos_t, sin_t = np.cos(angles), np.sin(angles)
    height, width = edge_map.shape
    candidates: list[tuple[int, int, int, int]] = []
    for radius in range(min_radius, max_radius + 1, radius_step):
        accumulator = np.zeros((height, width), dtype=np.uint16)
        for x, y in zip(x_edges, y_edges):
            centers_x = np.rint(x - radius * cos_t).astype(int)
            centers_y = np.rint(y - radius * sin_t).astype(int)
            valid = (centers_x >= 0) & (centers_x < width) & (centers_y >= 0) & (centers_y < height)
            np.add.at(accumulator, (centers_y[valid], centers_x[valid]), 1)
        peaks_y, peaks_x = np.where(accumulator >= threshold)
        for x, y in zip(peaks_x, peaks_y):
            candidates.append((int(x), int(y), radius, int(accumulator[y, x])))
    return non_maximum_suppression(candidates)


def non_maximum_suppression(candidates: list[tuple[int, int, int, int]], center_distance_factor: float = 0.55, radius_distance_factor: float = 0.35) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: item[3], reverse=True):
        x, y, radius, _ = candidate
        duplicate = False
        for kx, ky, kr, _ in kept:
            center_distance = np.hypot(x - kx, y - ky)
            radius_distance = abs(radius - kr)
            if center_distance < center_distance_factor * min(radius, kr) and radius_distance < radius_distance_factor * min(radius, kr):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_circles(image: np.ndarray, min_radius: int, max_radius: int, threshold: int, sigma: float = 2.3) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edge_map = canny(gray.astype(np.float32) / 255.0, sigma=sigma)
    return hough_circle_transform(edge_map, min_radius, max_radius, threshold)


def draw_circles(image: np.ndarray, circles: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = image.copy()
    for x, y, radius, votes in circles:
        cv2.circle(out, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(out, str(votes), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def hog_descriptor(image: np.ndarray, size: int = 128, cell_size: int = 8, block_cells: int = 2, bins: int = 9) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180.0
    cells_y = size // cell_size
    cells_x = size // cell_size
    cell_hist = np.zeros((cells_y, cells_x, bins), dtype=np.float32)
    bin_width = 180.0 / bins
    for cy in range(cells_y):
        for cx in range(cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            mag = magnitude[y0:y1, x0:x1].ravel()
            ori = orientation[y0:y1, x0:x1].ravel()
            bin_pos = ori / bin_width
            lower = np.floor(bin_pos).astype(int) % bins
            upper = (lower + 1) % bins
            upper_weight = bin_pos - np.floor(bin_pos)
            lower_weight = 1.0 - upper_weight
            np.add.at(cell_hist[cy, cx], lower, mag * lower_weight)
            np.add.at(cell_hist[cy, cx], upper, mag * upper_weight)
    blocks: list[np.ndarray] = []
    eps = 1e-6
    for by in range(cells_y - block_cells + 1):
        for bx in range(cells_x - block_cells + 1):
            block = cell_hist[by:by + block_cells, bx:bx + block_cells].ravel()
            block /= np.sqrt(np.dot(block, block) + eps * eps)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + eps * eps)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def label_from_filename(path: Path) -> int | None:
    stem = path.stem
    for class_name, label_id in COIN_LABELS.items():
        if stem.startswith(class_name + "_") or stem == class_name:
            return label_id
    return None


def load_classification_folder(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_filename(path)
        if label is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        features.append(hog_descriptor(image))
        labels.append(label)
    if not features:
        raise ValueError(f"No labeled images found in {folder}")
    return np.vstack(features), np.asarray(labels)


def classify(train_dir: Path, test_dir: Path) -> None:
    x_train, y_train = load_classification_folder(train_dir)
    x_test, y_test = load_classification_folder(test_dir)
    clf = SVC(kernel="linear", class_weight="balanced")
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    present_labels = sorted(set(y_test.tolist()) | set(predictions.tolist()))
    names = [ID_TO_LABEL[i] for i in present_labels]
    print(classification_report(y_test, predictions, labels=present_labels, target_names=names, zero_division=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coin detection and recognition")
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect")
    detect.add_argument("--image", type=Path, required=True)
    detect.add_argument("--output", type=Path, default=Path("outputs/detected.jpg"))
    detect.add_argument("--min-radius", type=int, required=True)
    detect.add_argument("--max-radius", type=int, required=True)
    detect.add_argument("--threshold", type=int, default=45)
    detect.add_argument("--sigma", type=float, default=2.3)
    classify_parser = sub.add_parser("classify")
    classify_parser.add_argument("--train", type=Path, required=True)
    classify_parser.add_argument("--test", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "classify":
        classify(args.train, args.test)
        return
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    circles = detect_circles(image, args.min_radius, args.max_radius, args.threshold, args.sigma)
    rendered = draw_circles(image, circles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        raise OSError(f"Could not write {args.output}")
    print(f"Detected {len(circles)} circles; saved {args.output}")


if __name__ == "__main__":
    main()
