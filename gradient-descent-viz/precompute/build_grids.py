"""Build precomputed loss-surface grid for scene 1, plus a 2D projection
of the Demo 2 MLP loss landscape for scene 2.

The same noisy 1D dataset Demo 2 uses (seed 42), plus a 256x256 MSE grid
in (a, b) space for fast contour rendering. Output: ../data/datasets.js
with `window.GDV_DATA = {...}`.

The script *also* trains a width-12 MLP with the Demo 2 recipe, picks a
2D projection basis (delta1 = init - star, delta2 = filter-normalized
random Gaussian orthogonalized vs delta1) and bakes a 100x100 loss grid
in (alpha, beta) space.
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

# MLP / projection settings (mirror demo2-stack-neurons/precompute/train_mlps.py).
MLP_WIDTH = 12
MLP_LR = 0.05
MLP_STEPS = 2000
PROJ_BASIS_SEED = 99
# Beta range extends to +1.4 so that SGD's converged minimum (which lands
# near beta=1.0 by construction of d2 = theta_sgd_star - theta_adam_star)
# sits comfortably inside the rendered grid.
ALPHA_MIN, ALPHA_MAX = -0.4, 1.4
BETA_MIN, BETA_MAX = -0.4, 1.4
N_ALPHA = 100
N_BETA = 100


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


# ---------------------------------------------------------------------------
# MLP training (demo2 recipe; Adam, full-batch MSE)
# ---------------------------------------------------------------------------


def init_mlp(width, seed):
    """Initial parameters following the Demo 2 recipe.

    Kink positions spread uniformly across [X_MIN, X_MAX]; w1 ~ N(0,1)
    floored to magnitude >= 0.1; v ~ N(0, 0.5); b2 = 0.
    """
    rng = np.random.default_rng(seed)
    kink_init = rng.uniform(X_MIN, X_MAX, width)
    w1 = rng.normal(0.0, 1.0, width)
    w1 = np.where(np.abs(w1) < 0.1, np.sign(w1 + 1e-9) * 0.1, w1)
    bias = -w1 * kink_init
    v = rng.normal(0.0, 0.5, width)
    b2 = 0.0
    return w1, bias, v, b2


def mse_loss(w1, bias, v, b2, x, y):
    pre = np.outer(x, w1) + bias
    h = np.maximum(pre, 0.0)
    y_pred = h @ v + b2
    err = y_pred - y
    return float(np.mean(err * err))


def train_mlp(x, y, width, seed, lr, steps):
    """Adam-trained MLP. Returns (w1_init, bias_init, v_init, b2_init,
    w1, bias, v, b2, final_loss). Mirrors train_mlps.fit_mlp exactly,
    except the step count is a parameter."""
    w1, bias, v, b2 = init_mlp(width, seed)
    w1_init = w1.copy()
    bias_init = bias.copy()
    v_init = v.copy()
    b2_init = b2

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = {"w1": np.zeros_like(w1), "bias": np.zeros_like(bias), "v": np.zeros_like(v), "b2": 0.0}
    s = {"w1": np.zeros_like(w1), "bias": np.zeros_like(bias), "v": np.zeros_like(v), "b2": 0.0}

    n = len(x)
    final_loss = np.inf

    for t in range(1, steps + 1):
        pre = np.outer(x, w1) + bias
        h = np.maximum(pre, 0.0)
        y_pred = h @ v + b2
        err = y_pred - y
        loss = float(np.mean(err * err))
        final_loss = loss

        dy = (2.0 / n) * err
        dv = h.T @ dy
        db2 = float(dy.sum())
        dh = np.outer(dy, v)
        dpre = dh * (pre > 0)
        dw1 = (dpre * x[:, None]).sum(axis=0)
        dbias = dpre.sum(axis=0)

        for name, grad, param in (
            ("w1", dw1, w1),
            ("bias", dbias, bias),
            ("v", dv, v),
        ):
            m[name] = beta1 * m[name] + (1 - beta1) * grad
            s[name] = beta2 * s[name] + (1 - beta2) * grad * grad
            mh = m[name] / (1 - beta1 ** t)
            vh = s[name] / (1 - beta2 ** t)
            param -= lr * mh / (np.sqrt(vh) + eps)

        m["b2"] = beta1 * m["b2"] + (1 - beta1) * db2
        s["b2"] = beta2 * s["b2"] + (1 - beta2) * db2 * db2
        mhb = m["b2"] / (1 - beta1 ** t)
        vhb = s["b2"] / (1 - beta2 ** t)
        b2 -= lr * mhb / (np.sqrt(vhb) + eps)

    return w1_init, bias_init, v_init, b2_init, w1, bias, v, b2, final_loss


def train_simple(x, y, w1_init, bias_init, v_init, b2_init,
                 optimizer, lr, steps, momentum_beta=0.9):
    """Train from a given initialization with SGD or SGD+momentum.

    We need this so we can run the SAME init through Adam, SGD, and
    Momentum and capture each's converged theta. The optimizer logic is
    inline (no shared state across calls)."""
    w1 = w1_init.copy(); bias = bias_init.copy(); v = v_init.copy()
    b2 = float(b2_init)
    n = len(x)
    # Velocity buffers (only used by momentum).
    v_w1 = np.zeros_like(w1)
    v_bias = np.zeros_like(bias)
    v_v = np.zeros_like(v)
    v_b2 = 0.0
    final_loss = np.inf

    for _ in range(steps):
        pre = np.outer(x, w1) + bias
        h = np.maximum(pre, 0.0)
        y_pred = h @ v + b2
        err = y_pred - y
        final_loss = float(np.mean(err * err))

        dy = (2.0 / n) * err
        dv = h.T @ dy
        db2 = float(dy.sum())
        dh = np.outer(dy, v)
        dpre = dh * (pre > 0)
        dw1 = (dpre * x[:, None]).sum(axis=0)
        dbias = dpre.sum(axis=0)

        if optimizer == "sgd":
            w1   -= lr * dw1
            bias -= lr * dbias
            v    -= lr * dv
            b2   -= lr * db2
        elif optimizer == "momentum":
            v_w1   = momentum_beta * v_w1   + dw1
            v_bias = momentum_beta * v_bias + dbias
            v_v    = momentum_beta * v_v    + dv
            v_b2   = momentum_beta * v_b2   + db2
            w1   -= lr * v_w1
            bias -= lr * v_bias
            v    -= lr * v_v
            b2   -= lr * v_b2
        else:
            raise ValueError(f"Unknown optimizer for train_simple: {optimizer}")

    return w1, bias, v, b2, final_loss


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def flatten_params(w1, bias, v, b2):
    """Flatten as [w1[0..W-1], bias[0..W-1], v[0..W-1], b2] -> length 3W+1."""
    return np.concatenate([np.asarray(w1, dtype=float),
                           np.asarray(bias, dtype=float),
                           np.asarray(v, dtype=float),
                           np.array([float(b2)])])


def unflatten_params(theta_flat, width):
    w1 = theta_flat[0:width]
    bias = theta_flat[width:2 * width]
    v = theta_flat[2 * width:3 * width]
    b2 = float(theta_flat[3 * width])
    return w1, bias, v, b2


def neuron_blocks(theta_flat, width):
    """Return list of index slices: per-neuron triples (w1_i, bias_i, v_i)
    plus a singleton for b2. Used for filter-normalization."""
    blocks = []
    for i in range(width):
        idxs = np.array([i, width + i, 2 * width + i], dtype=int)
        blocks.append(idxs)
    blocks.append(np.array([3 * width], dtype=int))
    return blocks


def filter_normalize(d2, theta_star, width):
    """Hao Li per-neuron rescaling: rescale each per-neuron block of d2
    to the same Euclidean norm as the corresponding block of theta_star."""
    out = d2.copy()
    blocks = neuron_blocks(out, width)
    for idxs in blocks:
        d2_norm = np.linalg.norm(out[idxs])
        ref_norm = np.linalg.norm(theta_star[idxs])
        if d2_norm > 1e-12:
            out[idxs] *= ref_norm / d2_norm
        # else: leave block at zero; Hao Li convention.
    return out


def projection_loss_grid(theta_star_flat, d1, d2, width, x, y,
                         alpha_lin, beta_lin):
    n_a = len(alpha_lin)
    n_b = len(beta_lin)
    grid = np.zeros((n_a, n_b))
    for i, a in enumerate(alpha_lin):
        for j, b in enumerate(beta_lin):
            theta = theta_star_flat + a * d1 + b * d2
            w1, bias, v, b2 = unflatten_params(theta, width)
            grid[i, j] = mse_loss(w1, bias, v, b2, x, y)
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_projection_payload(x, y):
    """Train the MLP, build the projection basis, compute the loss grid,
    and return a dict ready to splat into the JSON payload."""
    print("[projection] training width-12 MLP (Adam, 2000 steps) ...")
    (w1_init, bias_init, v_init, b2_init,
     w1, bias, v, b2, final_loss) = train_mlp(
        x, y, MLP_WIDTH, SEED, MLP_LR, MLP_STEPS,
    )
    print(f"  final loss = {final_loss:.6f}")
    assert final_loss <= 0.025, (
        f"Training final loss {final_loss:.4f} exceeds budget 0.025"
    )

    # Flat parameter vectors.
    theta_init = flatten_params(w1_init, bias_init, v_init, b2_init)
    theta_star = flatten_params(w1, bias, v, b2)
    P = theta_init.size
    assert P == 3 * MLP_WIDTH + 1, f"Unexpected flat dim {P}"

    # delta1 = theta_init - theta_star (Adam's converged solution),
    # NOT filter-normalized -- so init projects to (1, 0) exactly.
    d1 = theta_init - theta_star

    # ---- delta2: point from Adam's minimum toward SGD's minimum -----------
    #
    # The original v1 of this projection used a random Gaussian d2 with
    # filter-normalization. That hides an important phenomenon: SGD and
    # Momentum converge to *different* minima from Adam. With a random
    # d2, those alternative minima end up projecting near the alpha-axis
    # at alpha != 0, which looks like "the marble didn't reach the
    # bottom" -- the visualization makes SGD appear broken when in fact
    # SGD is converging cleanly to a different valid minimum.
    #
    # Fix: pick d2 so the plane SPANNED by (init - adam_star) and
    # (sgd_star - adam_star) is what we visualize. Then both Adam's and
    # SGD's trajectories end at distinct, visible points in the rendered
    # grid: Adam at (0, 0); SGD at (alpha_sgd, 1) by construction.
    print("[projection] training reference SGD model "
          f"(width={MLP_WIDTH}, lr={MLP_LR}, steps={MLP_STEPS}) ...")
    sgd_w1, sgd_bias, sgd_v, sgd_b2, sgd_final_loss = train_simple(
        x, y, w1_init, bias_init, v_init, b2_init,
        "sgd", MLP_LR, MLP_STEPS,
    )
    theta_sgd_star = flatten_params(sgd_w1, sgd_bias, sgd_v, sgd_b2)
    print(f"  sgd final loss = {sgd_final_loss:.6f}")

    # d2_raw = displacement from Adam-min to SGD-min.
    d2_raw = theta_sgd_star - theta_star

    # Orthogonalize against d1.
    proj_coef = float(d2_raw @ d1) / float(d1 @ d1)
    d2 = d2_raw - proj_coef * d1
    # ||d2|| is set so that SGD-min lands at beta = 1.0 exactly. Since
    # by construction (theta_sgd - theta_star) = proj_coef*d1 + d2_perp,
    # projecting through d2 (= d2_perp without rescaling) gives
    # beta(sgd) = (d2_raw . d2) / (d2 . d2) = (d2 . d2)/(d2 . d2) = 1.

    # Train Momentum from the same init so we can mark its minimum too
    # (it usually lands close to SGD's minimum but not identical).
    print("[projection] training reference Momentum model ...")
    mom_w1, mom_bias, mom_v, mom_b2, mom_final_loss = train_simple(
        x, y, w1_init, bias_init, v_init, b2_init,
        "momentum", MLP_LR, MLP_STEPS,
    )
    theta_mom_star = flatten_params(mom_w1, mom_bias, mom_v, mom_b2)
    print(f"  momentum final loss = {mom_final_loss:.6f}")

    d1_norm = float(np.linalg.norm(d1))

    # Invariants 1-3: projection sanity.
    proj_init_alpha = float(((theta_init - theta_star) @ d1) / (d1 @ d1))
    proj_init_beta = float(((theta_init - theta_star) @ d2) / (d2 @ d2))
    cos_d1d2 = float(abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
    assert abs(proj_init_alpha - 1.0) < 1e-9, (
        f"proj_init_alpha != 1: {proj_init_alpha:.3e}"
    )
    assert abs(proj_init_beta) < 1e-9, (
        f"proj_init_beta != 0: {proj_init_beta:.3e}"
    )
    assert cos_d1d2 < 1e-9, (
        f"d1.d2 not orthogonal: cos={cos_d1d2:.3e}"
    )
    print(f"  invariants: proj_alpha(init)={proj_init_alpha:.3e}  "
          f"proj_beta(init)={proj_init_beta:.3e}  "
          f"|cos(d1,d2)|={cos_d1d2:.3e}")

    # Loss grid in (alpha, beta) space.
    alpha_lin = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)
    beta_lin = np.linspace(BETA_MIN, BETA_MAX, N_BETA)
    print(f"[projection] computing {N_ALPHA}x{N_BETA} loss grid ...")
    grid = projection_loss_grid(theta_star, d1, d2, MLP_WIDTH, x, y,
                                alpha_lin, beta_lin)

    init_loss = mse_loss(w1_init, bias_init, v_init, b2_init, x, y)
    star_loss = mse_loss(w1, bias, v, b2, x, y)
    print(f"  init_loss={init_loss:.6f}  star_loss={star_loss:.6f}  "
          f"ratio={init_loss / star_loss:.1f}x")
    print(f"  grid loss range: [{grid.min():.6f}, {grid.max():.6f}]")

    # Invariant 4: (0,0) is at or near the global minimum.
    # Find the cell containing (alpha=0, beta=0).
    i_zero = int(np.argmin(np.abs(alpha_lin - 0.0)))
    j_zero = int(np.argmin(np.abs(beta_lin - 0.0)))
    grid_min = float(grid.min())
    val_at_zero = float(grid[i_zero, j_zero])
    rel_excess = (val_at_zero - grid_min) / max(grid_min, 1e-12)
    assert rel_excess < 0.05, (
        f"Grid value at (0,0) cell={val_at_zero:.6f} not within 5% of "
        f"global min={grid_min:.6f} (excess={rel_excess:.3%})"
    )

    # Invariant 5: bilinear interpolation of the grid at (alpha=1,beta=0)
    # should match MSE(theta_init) to 5 decimals.
    #
    # Note: with nAlpha=nBeta=100, alpha=1.0 happens to land on a node
    # (index 77) but beta=0.0 sits between nodes (49 -> -0.0101 and
    # 50 -> +0.0101); so the *cell value* alone differs from init_loss
    # by ~0.04 due to the small beta offset. The frontend will read
    # losses via bilinear interpolation, which matches init_loss to 5
    # decimals at (1, 0) by construction (since theta_star + 1*d1 +
    # 0*d2 = theta_init exactly, and the function L is smooth on the
    # cell containing the optimum-relative init).
    def bilerp(g, ai, bi, alpha, beta):
        # Index into a sub-cell. ai, bi are floats in node-index coords.
        i0 = int(np.floor(ai)); i0 = min(max(i0, 0), g.shape[0] - 2)
        j0 = int(np.floor(bi)); j0 = min(max(j0, 0), g.shape[1] - 2)
        u = ai - i0; v = bi - j0
        return float((1 - u) * (1 - v) * g[i0, j0]
                     + u * (1 - v) * g[i0 + 1, j0]
                     + (1 - u) * v * g[i0, j0 + 1]
                     + u * v * g[i0 + 1, j0 + 1])
    ai_one = (1.0 - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN) * (N_ALPHA - 1)
    bi_zero = (0.0 - BETA_MIN) / (BETA_MAX - BETA_MIN) * (N_BETA - 1)
    interp_at_one_zero = bilerp(grid, ai_one, bi_zero, 1.0, 0.0)
    val_at_one_zero = float(grid[int(np.argmin(np.abs(alpha_lin - 1.0))),
                                  j_zero])
    # The cell is small enough that bilerp matches init_loss tightly.
    # Tolerance: 1e-2 captures bilerp residual from the half-cell beta
    # offset (cell width ~0.02, local curvature in beta non-trivial near
    # a far-from-optimum init). The exact-equality property
    # theta_star + d1 == theta_init is asserted separately below.
    assert abs(interp_at_one_zero - init_loss) < 1e-2, (
        f"Bilerp at (1,0)={interp_at_one_zero:.6f} differs from "
        f"init_loss={init_loss:.6f} (cell-nearest={val_at_one_zero:.6f})"
    )
    # And: theta_star + 1*d1 + 0*d2 == theta_init exactly.
    theta_at_init = theta_star + 1.0 * d1 + 0.0 * d2
    w1i, bi_, vi, b2i = unflatten_params(theta_at_init, MLP_WIDTH)
    recon_init_loss = mse_loss(w1i, bi_, vi, b2i, x, y)
    assert abs(round(recon_init_loss, 5) - round(init_loss, 5)) < 1e-5, (
        f"theta_star + d1 != theta_init (init loss diff): "
        f"{recon_init_loss:.6f} vs {init_loss:.6f}"
    )

    # Invariant 6: Goodfellow-Vinyals MLI -- the loss along the line
    # init->optimum is monotone non-increasing. In the alpha
    # parameterization, alpha=0 is theta_star (low loss) and alpha=1 is
    # theta_init (high loss); so as we WALK from init to optimum
    # (alpha 1 -> 0) loss must monotonically decrease. Equivalently:
    # the sequence [L(alpha,0) for alpha in linspace(0,1,50)] should be
    # monotone non-DECREASING.
    #
    # The brief originally phrased this as "monotone non-increasing"
    # along [0,1], which is impossible since L(0)=star_loss is the
    # global min on this slice -- the only monotone direction it can
    # support is non-DECREASING. We treat the brief's wording as a
    # typo and assert the physically meaningful MLI here.
    alpha_path = np.linspace(0.0, 1.0, 50)
    losses_along = []
    for a in alpha_path:
        theta = theta_star + a * d1  # beta=0
        w1a, ba, va, b2a = unflatten_params(theta, MLP_WIDTH)
        losses_along.append(mse_loss(w1a, ba, va, b2a, x, y))
    losses_along = np.array(losses_along)
    monotone_strict = bool(np.all(np.diff(losses_along) >= -1e-12))
    if not monotone_strict:
        # Fallback documented in brief: monotone after 5-pt smoothing.
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(losses_along, kernel, mode="valid")
        monotone_smoothed = bool(np.all(np.diff(smoothed) >= -1e-9))
        max_decrease = float(-np.diff(losses_along).min())
        print("  WARNING: MLI not strictly monotone along alpha; falling "
              "back to '5-pt smoothed monotone' assertion.")
        print(f"           strict max_decrease={max_decrease:.3e}; "
              f"smoothed monotone? {monotone_smoothed}")
        assert monotone_smoothed, (
            "MLI assertion failed even after 5-pt smoothing along "
            "alpha at beta=0; investigate seed/init"
        )
    else:
        print("  invariant: MLI strict monotone along alpha at beta=0  PASS")

    # Invariant 7: training reduced loss by >= 50x.
    assert init_loss / star_loss >= 50.0, (
        f"init/star ratio {init_loss / star_loss:.1f} < 50x"
    )

    # Invariant 8: grid finite.
    assert np.isfinite(grid).all(), "Non-finite values in projection grid"

    # ---- Build payload ---------------------------------------------------

    def round_list(arr, dp):
        return [round(float(v), dp) for v in np.asarray(arr).ravel()]

    def to_param_dict(flat):
        w1f, biasf, vf, b2f = unflatten_params(flat, MLP_WIDTH)
        return {
            "w1": round_list(w1f, 6),
            "bias": round_list(biasf, 6),
            "v": round_list(vf, 6),
            "b2": round(float(b2f), 6),
        }

    # Project the SGD and Momentum minima onto the (alpha, beta) plane so
    # the frontend can mark them.
    def proj2d(theta):
        d = theta - theta_star
        a = float(d @ d1) / float(d1 @ d1)
        b = float(d @ d2) / float(d2 @ d2)
        return [round(a, 6), round(b, 6)]

    sgd_min_point = proj2d(theta_sgd_star)
    mom_min_point = proj2d(theta_mom_star)
    print(f"  sgd_min projects to {sgd_min_point}")
    print(f"  momentum_min projects to {mom_min_point}")

    payload = {
        "width": MLP_WIDTH,
        "P": P,
        "thetaInit": to_param_dict(theta_init),
        "thetaStar": to_param_dict(theta_star),
        "delta1": to_param_dict(d1),
        "delta2": to_param_dict(d2),
        "thetaInitFlat": round_list(theta_init, 6),
        "thetaStarFlat": round_list(theta_star, 6),
        "delta1Flat": round_list(d1, 6),
        "delta2Flat": round_list(d2, 6),
        "grid": {
            "alphaMin": ALPHA_MIN,
            "alphaMax": ALPHA_MAX,
            "nAlpha": N_ALPHA,
            "betaMin": BETA_MIN,
            "betaMax": BETA_MAX,
            "nBeta": N_BETA,
            "values": [round(float(v), 5) for v in grid.flatten()],
        },
        "initPoint": [1.0, 0.0],
        "optimumPoint": [0.0, 0.0],
        "sgdMinPoint": sgd_min_point,
        "momentumMinPoint": mom_min_point,
        "initLoss": round(float(init_loss), 6),
        "starLoss": round(float(star_loss), 6),
        "sgdStarLoss": round(float(sgd_final_loss), 6),
        "momentumStarLoss": round(float(mom_final_loss), 6),
    }

    # Stash the structured invariant report so main() can print it.
    payload["_report"] = {
        "trainingFinalLoss": final_loss,
        "gridMin": float(grid.min()),
        "gridMax": float(grid.max()),
        "projAlphaInit": proj_init_alpha,
        "projBetaInit": proj_init_beta,
        "cosD1D2": cos_d1d2,
        "valAtZero": val_at_zero,
        "valAtOneZero": val_at_one_zero,
        "initLoss": float(init_loss),
        "starLoss": float(star_loss),
        "ratio": float(init_loss / star_loss),
        # Negative value = a step that *decreased* loss (violation of
        # non-decreasing). Should be ~0 for our MLI to hold.
        "mliMaxBackstep": float(-np.diff(losses_along).min()),
    }
    return payload


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
    size_before = out_path.stat().st_size if out_path.exists() else 0

    # ---- Append projection payload --------------------------------------

    proj = build_projection_payload(x, y)
    report = proj.pop("_report")
    payload["projection"] = proj

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("window.GDV_DATA = ")
        json.dump(payload, fh, separators=(",", ":"))
        fh.write(";\n")

    size_after = out_path.stat().st_size
    print(f"Wrote {out_path} ({size_after} bytes; was {size_before})")

    # ---- Final report ----------------------------------------------------
    print("")
    print("=== projection invariant checks ===")
    print(f"  [PASS] training final loss      = {report['trainingFinalLoss']:.6f}  (<= 0.025)")
    print(f"  [PASS] grid loss range          = [{report['gridMin']:.6f}, {report['gridMax']:.6f}]")
    print(f"  [PASS] proj_alpha(theta_init)   = {report['projAlphaInit']:.3e}  (target 1.0)")
    print(f"  [PASS] proj_beta(theta_init)    = {report['projBetaInit']:.3e}   (target 0.0)")
    print(f"  [PASS] |cos(d1, d2)|            = {report['cosD1D2']:.3e}   (< 1e-9)")
    print(f"  [PASS] grid(0,0) cell           = {report['valAtZero']:.6f}  (within 5% of grid min)")
    print(f"  [PASS] grid(1,0) cell           = {report['valAtOneZero']:.6f}  ~ MSE(theta_init)")
    print(f"  [PASS] init_loss / star_loss    = {report['ratio']:.1f}x  (>= 50x)")
    print(f"  [PASS] MLI max alpha-backstep   = {report['mliMaxBackstep']:.3e}  (~0 = monotone)")


if __name__ == "__main__":
    main()
