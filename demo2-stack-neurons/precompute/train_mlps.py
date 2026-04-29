"""Precompute fits for Demo 2 — Stack neurons, build curves.

Trains a 1-hidden-layer ReLU MLP at every width in {0, 1, ..., 30} on a
fixed noisy 1D regression dataset, then bakes the resulting curves and
kink locations into ``../data/datasets.js`` for the static viz to load.

Width 0 is plain linear regression (a straight line, no kinks). For each
width >= 1 we run a few seeded restarts and keep the lowest-loss fit so
the slider trajectory looks monotone-ish in fit quality. Reproducible:
the only randomness comes from a Mulberry-style seeded numpy RNG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset (fixed seed -> byte-identical regenerations)
# ---------------------------------------------------------------------------

SEED = 42
N_POINTS = 60
X_MIN, X_MAX = -3.0, 3.0
NOISE_SD = 0.18
N_GRID = 240
MAX_WIDTH = 30
N_RESTARTS = 5
N_STEPS = 4000
LR = 0.05


def truth(x):
    """Smooth wavy ground truth with a couple of bumps."""
    return 0.55 * np.sin(1.6 * x) + 0.35 * np.sin(0.7 * x + 0.5) + 0.05 * x


def make_dataset(seed: int):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(X_MIN, X_MAX, N_POINTS))
    y = truth(x) + rng.normal(0.0, NOISE_SD, N_POINTS)
    return x, y


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def fit_linear(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def fit_mlp(x, y, width: int, seed: int):
    """Single-hidden-layer ReLU MLP with scalar input/output, MSE loss, Adam.

    Parametrisation: hidden pre-activation = w1 * x + bias.
    Kink for unit i lies at x = -bias_i / w1_i.
    """
    rng = np.random.default_rng(seed)

    # Spread initial kinks across the input domain so each ReLU starts
    # somewhere useful — much faster convergence than purely random init.
    kink_init = rng.uniform(X_MIN, X_MAX, width)
    w1 = rng.normal(0.0, 1.0, width)
    # snap small magnitudes away from zero to avoid degenerate kinks
    w1 = np.where(np.abs(w1) < 0.1, np.sign(w1 + 1e-9) * 0.1, w1)
    bias = -w1 * kink_init
    w2 = rng.normal(0.0, 0.5, width)
    b2 = 0.0

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = {"w1": np.zeros_like(w1), "bias": np.zeros_like(bias), "w2": np.zeros_like(w2), "b2": 0.0}
    v = {"w1": np.zeros_like(w1), "bias": np.zeros_like(bias), "w2": np.zeros_like(w2), "b2": 0.0}

    n = len(x)
    final_loss = np.inf

    for t in range(1, N_STEPS + 1):
        pre = np.outer(x, w1) + bias                 # (n, width)
        h = np.maximum(pre, 0.0)
        y_pred = h @ w2 + b2
        err = y_pred - y
        loss = float(np.mean(err * err))
        final_loss = loss

        dy = (2.0 / n) * err
        dw2 = h.T @ dy
        db2 = float(dy.sum())
        dh = np.outer(dy, w2)
        dpre = dh * (pre > 0)
        dw1 = (dpre * x[:, None]).sum(axis=0)
        dbias = dpre.sum(axis=0)

        for name, grad, param in (
            ("w1", dw1, w1),
            ("bias", dbias, bias),
            ("w2", dw2, w2),
        ):
            m[name] = beta1 * m[name] + (1 - beta1) * grad
            v[name] = beta2 * v[name] + (1 - beta2) * grad * grad
            mh = m[name] / (1 - beta1 ** t)
            vh = v[name] / (1 - beta2 ** t)
            param -= LR * mh / (np.sqrt(vh) + eps)

        m["b2"] = beta1 * m["b2"] + (1 - beta1) * db2
        v["b2"] = beta2 * v["b2"] + (1 - beta2) * db2 * db2
        mhb = m["b2"] / (1 - beta1 ** t)
        vhb = v["b2"] / (1 - beta2 ** t)
        b2 -= LR * mhb / (np.sqrt(vhb) + eps)

    return w1, bias, w2, b2, final_loss


def predict(w1, bias, w2, b2, xg):
    pre = np.outer(xg, w1) + bias
    h = np.maximum(pre, 0.0)
    return h @ w2 + b2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    x, y = make_dataset(SEED)
    x_grid = np.linspace(X_MIN, X_MAX, N_GRID)
    y_truth = truth(x_grid)

    fits = {}

    # Width 0 -- linear regression. Stored as parameters; the frontend
    # reconstructs the line in JS from {a, b}.
    a, b = fit_linear(x, y)
    fits["0"] = {
        "kind": "linear",
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "loss": float(np.mean((a * x + b - y) ** 2)),
    }

    for w in range(1, MAX_WIDTH + 1):
        best = None
        for r in range(N_RESTARTS):
            seed = 1000 * w + r
            try:
                params = fit_mlp(x, y, w, seed)
            except Exception as exc:  # noqa: BLE001
                print(f"[width {w} restart {r}] {exc}", file=sys.stderr)
                continue
            if best is None or params[-1] < best[-1]:
                best = params

        if best is None:
            raise RuntimeError(f"No successful fit for width {w}")

        w1, bias, w2, b2, loss = best
        # Verify reconstruction: sum of per-neuron contributions + b2 must
        # equal the predict() call to 1e-6 at every grid point.
        yg_direct = predict(w1, bias, w2, b2, x_grid)
        contribs = np.maximum(np.outer(x_grid, w1) + bias, 0.0) * w2  # (n_grid, w)
        yg_sum = contribs.sum(axis=1) + b2
        max_recon_err = float(np.max(np.abs(yg_direct - yg_sum)))
        assert max_recon_err < 1e-6, (
            f"Reconstruction mismatch at width {w}: max_err={max_recon_err}"
        )

        n_kinks_in_view = sum(
            1
            for w1_i, b_i in zip(w1, bias)
            if abs(w1_i) > 1e-8 and X_MIN <= -b_i / w1_i <= X_MAX
        )

        fits[str(w)] = {
            "kind": "mlp",
            "w1": [round(float(v), 6) for v in w1],
            "bias": [round(float(v), 6) for v in bias],
            "v": [round(float(v), 6) for v in w2],
            "b2": round(float(b2), 6),
            "loss": loss,
        }
        print(f"  width {w:>2d}  loss={loss:.4f}  kinks_in_view={n_kinks_in_view}  recon_err={max_recon_err:.2e}")

    payload = {
        "points": [[round(float(xi), 4), round(float(yi), 4)] for xi, yi in zip(x, y)],
        "truthCurve": [[round(float(xi), 4), round(float(yi), 4)] for xi, yi in zip(x_grid, y_truth)],
        "fits": fits,
        "xMin": X_MIN,
        "xMax": X_MAX,
        "maxWidth": MAX_WIDTH,
        "nGrid": N_GRID,
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "datasets.js"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("window.DATA = ")
        json.dump(payload, fh, separators=(",", ":"))
        fh.write(";\n")

    # Sanity invariants -- assert in code, not in a final report.
    losses = [fits[str(w)]["loss"] for w in range(MAX_WIDTH + 1)]
    assert losses[0] > losses[MAX_WIDTH], (
        f"Loss did not decrease from width 0 ({losses[0]:.4f}) to width {MAX_WIDTH} ({losses[MAX_WIDTH]:.4f})"
    )
    rolling_min = np.minimum.accumulate(losses)
    assert rolling_min[15] < 0.5 * losses[1], (
        f"Loss did not improve enough by width 15 (got {rolling_min[15]:.4f} vs width-1 {losses[1]:.4f})"
    )
    for w_str, fit in fits.items():
        if fit["kind"] == "mlp":
            for v in fit["w1"] + fit["bias"] + fit["v"]:
                assert np.isfinite(v), f"Non-finite param in width {w_str}"
        else:
            assert np.isfinite(fit["a"]) and np.isfinite(fit["b"]), f"Non-finite linear params in width {w_str}"
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Loss(0)={losses[0]:.4f}  Loss(15)={losses[15]:.4f}  Loss(30)={losses[30]:.4f}")


if __name__ == "__main__":
    main()
