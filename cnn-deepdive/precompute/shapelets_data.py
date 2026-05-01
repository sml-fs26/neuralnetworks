"""Procedural shapelet image generators.

6 classes: cross, L, vertical_line, horizontal_line, circle, triangle.
28x28 grayscale, pixels in [0, 1].

Deterministic given a seeded numpy.random.RandomState.
"""

import math
import numpy as np

CLASSES = ['cross', 'L', 'vertical_line', 'horizontal_line', 'circle', 'triangle']
H = W = 28


def _rotate_point(x, y, cx, cy, theta):
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def _draw_line(img, x0, y0, x1, y1, val=1.0, thickness=1):
    """Bresenham-ish thick line by sampling many points along the segment."""
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 4 + 1)
    for t in np.linspace(0, 1, n):
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx * dx + dy * dy <= thickness * thickness:
                    xi, yi = int(round(x + dx)), int(round(y + dy))
                    if 0 <= xi < W and 0 <= yi < H:
                        img[yi, xi] = val


def _gen_cross(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(11, 16)
    cy = rng.uniform(11, 16)
    L = rng.uniform(7, 10)
    theta = rng.uniform(-0.25, 0.25)
    th = rng.choice([0, 1])
    # vertical bar
    x0, y0 = _rotate_point(cx, cy - L, cx, cy, theta)
    x1, y1 = _rotate_point(cx, cy + L, cx, cy, theta)
    _draw_line(img, x0, y0, x1, y1, 1.0, th)
    # horizontal bar
    x0, y0 = _rotate_point(cx - L, cy, cx, cy, theta)
    x1, y1 = _rotate_point(cx + L, cy, cx, cy, theta)
    _draw_line(img, x0, y0, x1, y1, 1.0, th)
    return img


def _gen_L(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(11, 16)
    cy = rng.uniform(11, 16)
    L = rng.uniform(7, 10)
    theta = rng.uniform(-0.2, 0.2)
    th = rng.choice([0, 1])
    # vertical leg from top to bottom-left
    x0, y0 = _rotate_point(cx - L / 2, cy - L, cx, cy, theta)
    x1, y1 = _rotate_point(cx - L / 2, cy + L, cx, cy, theta)
    _draw_line(img, x0, y0, x1, y1, 1.0, th)
    # horizontal leg at bottom
    x2, y2 = _rotate_point(cx - L / 2, cy + L, cx, cy, theta)
    x3, y3 = _rotate_point(cx + L, cy + L, cx, cy, theta)
    _draw_line(img, x2, y2, x3, y3, 1.0, th)
    return img


def _gen_vertical_line(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(8, 19)
    cy = rng.uniform(11, 16)
    L = rng.uniform(8, 11)
    theta = rng.uniform(-0.18, 0.18)  # near-vertical
    th = rng.choice([0, 1])
    x0, y0 = _rotate_point(cx, cy - L, cx, cy, theta)
    x1, y1 = _rotate_point(cx, cy + L, cx, cy, theta)
    _draw_line(img, x0, y0, x1, y1, 1.0, th)
    return img


def _gen_horizontal_line(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(11, 16)
    cy = rng.uniform(8, 19)
    L = rng.uniform(8, 11)
    theta = rng.uniform(-0.18, 0.18)  # near-horizontal
    th = rng.choice([0, 1])
    x0, y0 = _rotate_point(cx - L, cy, cx, cy, theta)
    x1, y1 = _rotate_point(cx + L, cy, cx, cy, theta)
    _draw_line(img, x0, y0, x1, y1, 1.0, th)
    return img


def _gen_circle(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(11, 16)
    cy = rng.uniform(11, 16)
    r = rng.uniform(6, 9)
    th = rng.choice([0.7, 1.0, 1.3])
    n = 200
    for i in range(n):
        a = 2 * math.pi * i / n
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if abs(dx) + abs(dy) <= int(round(th)):
                    xi, yi = int(round(x + dx)), int(round(y + dy))
                    if 0 <= xi < W and 0 <= yi < H:
                        img[yi, xi] = 1.0
    return img


def _gen_triangle(rng):
    img = np.zeros((H, W), dtype=np.float32)
    cx = rng.uniform(11, 16)
    cy = rng.uniform(12, 17)
    r = rng.uniform(7, 10)
    theta = rng.uniform(0, 2 * math.pi)
    th = rng.choice([0, 1])
    pts = []
    for i in range(3):
        a = theta + i * 2 * math.pi / 3
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for i in range(3):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % 3]
        _draw_line(img, x0, y0, x1, y1, 1.0, th)
    return img


_GEN = {
    'cross': _gen_cross,
    'L': _gen_L,
    'vertical_line': _gen_vertical_line,
    'horizontal_line': _gen_horizontal_line,
    'circle': _gen_circle,
    'triangle': _gen_triangle,
}


def generate_dataset(n_per_class, seed):
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for cls_idx, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = _GEN[cls](rng)
            # Mild noise so inputs are not perfectly clean.
            img = img + rng.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            Xs.append(img)
            ys.append(cls_idx)
    Xs = np.stack(Xs, axis=0)
    ys = np.array(ys, dtype=np.int64)
    # Shuffle deterministically.
    perm = rng.permutation(len(Xs))
    return Xs[perm], ys[perm]


if __name__ == '__main__':
    X, y = generate_dataset(50, 42)
    print('shape', X.shape, 'labels', y[:10])
    print('range', X.min(), X.max())
