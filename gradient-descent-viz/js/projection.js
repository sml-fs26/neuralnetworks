/* Projection utilities for Scene 3.

   Demo 2's MLP has P = 3W + 1 parameters (where W is the hidden width).
   For width 12 that's 37 dimensions. To visualise gradient descent on
   that landscape we project onto a 2D plane through a center point θ*
   along two basis vectors δ₁, δ₂ (typically chosen so δ₁ points from
   init toward the trained model and δ₂ is an orthogonal direction).

   For any θ in P-space:
     α = (θ − θ*) · δ₁ / (δ₁ · δ₁)
     β = (θ − θ*) · δ₂ / (δ₂ · δ₂)
   For any (α, β) on the plane:
     θ(α, β) = θ* + α·δ₁ + β·δ₂

   All routines accept either flat arrays (length P) or the parameter
   structure { w1, bias, v, b2 } used by TrainingEngine. flatten()
   converts the latter to the former in canonical order. */
(function () {
  function flatten(params) {
    const W = params.w1.length;
    const out = new Array(3 * W + 1);
    for (let i = 0; i < W; i++) out[i] = params.w1[i];
    for (let i = 0; i < W; i++) out[W + i] = params.bias[i];
    for (let i = 0; i < W; i++) out[2 * W + i] = params.v[i];
    out[3 * W] = params.b2;
    return out;
  }

  function unflatten(flat, W) {
    return {
      w1:   flat.slice(0, W),
      bias: flat.slice(W, 2 * W),
      v:    flat.slice(2 * W, 3 * W),
      b2:   flat[3 * W],
    };
  }

  function dot(a, b) {
    let s = 0;
    const n = a.length;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  }

  function diff(a, b) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }

  function add(a, b) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    return out;
  }

  function scale(a, s) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] * s;
    return out;
  }

  // Project a flat θ onto the 2D plane spanned by (δ₁, δ₂) through θ*.
  // Returns [α, β]. Uses oblique projection: each axis is rescaled by
  // its own basis vector's squared norm, so that θ = θ* + δ₁ → (1, 0)
  // exactly (which is what we want when δ₁ = θ_init − θ*).
  function project(theta, theta_star, delta1, delta2) {
    const d = diff(theta, theta_star);
    const a = dot(d, delta1) / dot(delta1, delta1);
    const b = dot(d, delta2) / dot(delta2, delta2);
    return [a, b];
  }

  // Reconstruct a θ on the 2D plane at coordinates (α, β).
  function synth(theta_star, delta1, delta2, a, b) {
    return add(theta_star, add(scale(delta1, a), scale(delta2, b)));
  }

  window.Projection = {
    flatten, unflatten,
    dot, diff, add, scale,
    project, synth,
  };
})();
