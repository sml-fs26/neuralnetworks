"""Build precomputed loss-surface grid for scene 1.

The same noisy 1D dataset Demo 2 uses (seed 42), plus a 256x256 MSE grid
in (a, b) space for fast contour rendering. Output: ../data/datasets.js
with `window.GDV_DATA = {...}`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Keep the same data Demo 2 uses so the lecture has a single canonical
# noisy curve. Re-derive here -- do not import from sibling project.

SEED = 42
N_POINTS = 60
X_MIN, X_MAX = -3.0, 3.0
NOISE_SD = 0.18

# Loss-surface grid in (a, b) space.
A_MIN, A_MAX = -0.6, 1.0
B_MIN, B_MAX = -0.8, 0.8
N_A = 200
N_B = 200


def truth(x):
    return 0.55 * np.sin(1.6 * x) + 0.35 * np.sin(0.7 * x + 0.5) + 0.05 * x


def make_dataset(seed):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(X_MIN, X_MAX, N_POINTS))
    y = truth(x) + rng.normal(0.0, NOISE_SD, N_POINTS)
    return x, y


def mse_grid(x, y):
    a = np.linspace(A_MIN, A_MAX, N_A)
    b = np.linspace(B_MIN, B_MAX, N_B)
    A, B = np.meshgrid(a, b, indexing="ij")
    # MSE = mean((A*x + B - y)**2) over the dataset, vectorized.
    grid = np.zeros_like(A)
    for xi, yi in zip(x, y):
        grid += (A * xi + B - yi) ** 2
    grid /= len(x)
    return grid


def main():
    x, y = make_dataset(SEED)
    grid = mse_grid(x, y)

    # Optimal (a*, b*) via closed-form least squares so the contour center
    # lands inside the rendered range.
    A_design = np.vstack([x, np.ones_like(x)]).T
    a_star, b_star = np.linalg.lstsq(A_design, y, rcond=None)[0]
    loss_star = np.mean((a_star * x + b_star - y) ** 2)

    print(f"  optimum: a* = {a_star:.4f}  b* = {b_star:.4f}  loss* = {loss_star:.4f}")
    print(f"  grid range: a in [{A_MIN}, {A_MAX}], b in [{B_MIN}, {B_MAX}]")
    print(f"  grid loss range: [{grid.min():.4f}, {grid.max():.4f}]")

    # Truth curve on a fine x-grid for the right-pane render.
    x_grid = np.linspace(X_MIN, X_MAX, 240)
    y_truth = truth(x_grid)

    # Sanity invariants. The optimum loss is irreducible noise + the fact
    # that a line cannot fit a wave -- expect ~0.18 for this dataset.
    assert 0.10 < loss_star < 0.30, f"Unexpected optimum loss {loss_star:.4f}"
    assert np.isfinite(grid).all(), "NaN or inf in loss grid"
    assert A_MIN < a_star < A_MAX, f"a* = {a_star} outside grid"
    assert B_MIN < b_star < B_MAX, f"b* = {b_star} outside grid"
    # Loss grid should span at least 5x from min to max for visible contours.
    assert grid.max() / grid.min() > 5, f"Loss range too narrow ({grid.max() / grid.min():.2f}x)"

    payload = {
        "points": [[round(float(xi), 4), round(float(yi), 4)] for xi, yi in zip(x, y)],
        "truthCurve": [
            [round(float(xi), 4), round(float(yi), 4)]
            for xi, yi in zip(x_grid, y_truth)
        ],
        "xMin": X_MIN,
        "xMax": X_MAX,
        "noiseSd": NOISE_SD,
        "nPoints": N_POINTS,
        "lossGrid": {
            "aMin": A_MIN,
            "aMax": A_MAX,
            "bMin": B_MIN,
            "bMax": B_MAX,
            "nA": N_A,
            "nB": N_B,
            # Row-major: grid[i*nB + j] is loss at (a[i], b[j]).
            "values": [round(float(v), 5) for v in grid.flatten()],
        },
        "optimum": {
            "a": round(float(a_star), 5),
            "b": round(float(b_star), 5),
            "loss": round(float(loss_star), 5),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "datasets.js"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("window.GDV_DATA = ")
        json.dump(payload, fh, separators=(",", ":"))
        fh.write(";\n")

    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
