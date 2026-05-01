"""Procedural 64x64 cartoon scenes with per-pixel labels.

5 classes: 0 sky, 1 grass, 2 sun, 3 tree, 4 person.
Each sample: image RGB in [0,1] shape (3,64,64), label int shape (64,64).

Compositor (per sample):
  - Gradient sky in upper half.
  - Green grass strip in lower third (last ~22 rows).
  - Optional sun (yellow circle in sky), random position+radius.
  - Optional tree (brown trunk + green canopy on grass).
  - Optional person (head + body silhouette on grass).

Each optional element appears with probability ~0.7 to ensure variety, with
at least one foreground object guaranteed per sample.
"""

import math
import numpy as np

CLASSES = ['sky', 'grass', 'sun', 'tree', 'person']
H = W = 64


def _gradient_sky(rng):
    # Light blue at top, paler near horizon. Slight color jitter.
    top = np.array([0.45, 0.65, 0.95]) + rng.normal(0, 0.03, 3)
    bot = np.array([0.75, 0.88, 0.98]) + rng.normal(0, 0.03, 3)
    img = np.zeros((3, H, W), dtype=np.float32)
    for r in range(H):
        t = r / max(H - 1, 1)
        c = (1 - t) * top + t * bot
        img[:, r, :] = np.clip(c, 0, 1)[:, None]
    return img


def _grass_color(rng):
    base = np.array([0.25, 0.6, 0.25]) + rng.normal(0, 0.04, 3)
    return np.clip(base, 0, 1)


def _draw_disc(canvas, label, cx, cy, r, color, cls):
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    for c in range(3):
        canvas[c][mask] = color[c]
    label[mask] = cls


def _draw_rect(canvas, label, x0, y0, x1, y1, color, cls):
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    for c in range(3):
        canvas[c][y0:y1, x0:x1] = color[c]
    label[y0:y1, x0:x1] = cls


def _generate(rng):
    img = _gradient_sky(rng)
    label = np.zeros((H, W), dtype=np.int64)  # 0 = sky everywhere

    # Grass strip
    grass_top = rng.randint(40, 46)  # 18-24 rows of grass
    grass_color = _grass_color(rng)
    img[0, grass_top:, :] = grass_color[0]
    img[1, grass_top:, :] = grass_color[1]
    img[2, grass_top:, :] = grass_color[2]
    # add subtle texture to grass
    noise = rng.normal(0, 0.04, (3, H - grass_top, W)).astype(np.float32)
    img[:, grass_top:, :] = np.clip(img[:, grass_top:, :] + noise, 0, 1)
    label[grass_top:, :] = 1

    # Decide which objects to place. Ensure at least one is on.
    p_sun = rng.random() < 0.7
    p_tree = rng.random() < 0.7
    p_person = rng.random() < 0.7
    if not (p_sun or p_tree or p_person):
        # force one
        which = rng.choice(['sun', 'tree', 'person'])
        if which == 'sun': p_sun = True
        elif which == 'tree': p_tree = True
        else: p_person = True

    # Sun: yellow disc in upper area (above grass).
    if p_sun:
        sr = rng.randint(4, 8)
        scx = rng.randint(sr + 1, W - sr - 1)
        scy = rng.randint(sr + 1, max(grass_top - sr - 2, sr + 2))
        sun_color = np.array([0.98, 0.92, 0.25]) + rng.normal(0, 0.02, 3)
        sun_color = np.clip(sun_color, 0, 1)
        _draw_disc(img, label, scx, scy, sr, sun_color, 2)

    # Tree: brown trunk + green canopy on grass. Place horizontally.
    if p_tree:
        tx = rng.randint(8, W - 8)
        trunk_w = rng.randint(2, 4)
        trunk_h = rng.randint(8, 14)
        trunk_top = grass_top - trunk_h // 2  # straddles grass line
        trunk_bot = trunk_top + trunk_h
        trunk_color = np.array([0.42, 0.27, 0.13]) + rng.normal(0, 0.03, 3)
        trunk_color = np.clip(trunk_color, 0, 1)
        _draw_rect(img, label, tx - trunk_w, trunk_top, tx + trunk_w, trunk_bot,
                   trunk_color, 3)
        # Canopy (disc) on top
        cr = rng.randint(5, 9)
        ccy = trunk_top - cr // 2
        canopy_color = np.array([0.15, 0.5, 0.18]) + rng.normal(0, 0.04, 3)
        canopy_color = np.clip(canopy_color, 0, 1)
        _draw_disc(img, label, tx, ccy, cr, canopy_color, 3)

    # Person: head (small disc) + body (rect) on grass.
    if p_person:
        px = rng.randint(8, W - 8)
        head_r = rng.randint(2, 4)
        head_cy = grass_top - 6
        skin_color = np.array([0.92, 0.78, 0.62]) + rng.normal(0, 0.03, 3)
        skin_color = np.clip(skin_color, 0, 1)
        body_color = np.array([0.85, 0.2, 0.25]) + rng.normal(0, 0.05, 3)
        body_color = np.clip(body_color, 0, 1)
        # Body
        bw = rng.randint(3, 5)
        bh = rng.randint(8, 12)
        body_top = head_cy + head_r
        _draw_rect(img, label, px - bw, body_top, px + bw + 1,
                   body_top + bh, body_color, 4)
        # Head AFTER body so it overlaps cleanly.
        _draw_disc(img, label, px, head_cy, head_r, skin_color, 4)

    return img, label


def generate_dataset(n, seed):
    rng = np.random.RandomState(seed)
    Xs = np.zeros((n, 3, H, W), dtype=np.float32)
    Ys = np.zeros((n, H, W), dtype=np.int64)
    for i in range(n):
        Xs[i], Ys[i] = _generate(rng)
    return Xs, Ys


if __name__ == '__main__':
    X, Y = generate_dataset(20, 43)
    print('X', X.shape, X.min(), X.max())
    print('Y', Y.shape, np.unique(Y))
