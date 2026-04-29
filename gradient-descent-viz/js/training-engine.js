/* Scene 2 — Training engine (pure math, no DOM).

   1-hidden-layer ReLU MLP:
       y = b2 + sum_i v_i * ReLU(w1_i * x + bias_i)
   Trained on window.GDV_DATA.points with MSE loss. Mirrors the recipe in
   demo2-stack-neurons/precompute/train_mlps.py so the live scene converges
   to the same family of fits the static demo ships.

   Public API:
     const eng = new TrainingEngine({ width, seed, lr, optimizer });
     eng.step(batchSize?)              -> one update; default = full batch
     eng.params()                       -> { w1, bias, v, b2, loss, iter }
     eng.evalAt(xs)                     -> Float64Array of network outputs
     eng.reset(seed?)                   -> deterministic re-init
     eng.setLR(lr) / eng.setOptimizer(name)

   Caller owns no internal arrays; .params() returns shallow clones safe to
   read but not safe to mutate. */
(function () {
  'use strict';

  // -------- deterministic RNG (mirrors LossUtils.makeRng but standalone) ---

  function makeRng(seed) {
    let s = (seed | 0) || 1;
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Box-Muller standard normal sample. Returns one number per call;
  // caches the second draw for the next invocation.
  function normalSampler(rng) {
    let cached = null;
    return function () {
      if (cached !== null) {
        const v = cached;
        cached = null;
        return v;
      }
      let u1 = rng(); if (u1 < 1e-12) u1 = 1e-12;
      const u2 = rng();
      const r = Math.sqrt(-2 * Math.log(u1));
      const a = 2 * Math.PI * u2;
      cached = r * Math.sin(a);
      return r * Math.cos(a);
    };
  }

  // -------- engine constructor ---------------------------------------------

  function TrainingEngine(opts) {
    if (!(this instanceof TrainingEngine)) return new TrainingEngine(opts);
    opts = opts || {};

    this._width = Math.max(1, opts.width | 0 || 12);
    this._seed = Number.isFinite(opts.seed) ? (opts.seed | 0) : 42;
    this._lr = Number.isFinite(opts.lr) ? +opts.lr : 0.05;
    this._optimizer = opts.optimizer || 'adam';

    // Adam hyper-params (match demo2 precompute).
    this._beta1 = 0.9;
    this._beta2 = 0.999;
    this._eps = 1e-8;
    // Momentum for the SGD+momentum optimizer.
    this._momentum = 0.9;

    // Pull dataset from the shared global. We freeze our own copy so a
    // later mutation to GDV_DATA cannot quietly desync training.
    if (!(window.GDV_DATA && Array.isArray(window.GDV_DATA.points))) {
      throw new Error('TrainingEngine: window.GDV_DATA.points missing');
    }
    const pts = window.GDV_DATA.points;
    const n = pts.length;
    this._xs = new Float64Array(n);
    this._ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this._xs[i] = +pts[i][0];
      this._ys[i] = +pts[i][1];
    }
    this._xMin = Number.isFinite(window.GDV_DATA.xMin) ? +window.GDV_DATA.xMin : -3.0;
    this._xMax = Number.isFinite(window.GDV_DATA.xMax) ? +window.GDV_DATA.xMax : +3.0;

    // Pre-allocated scratch buffers for one forward/backward pass; reused
    // every step to avoid GC churn at 30 Hz.
    this._allocScratch();

    this.reset(this._seed);
  }

  TrainingEngine.prototype._allocScratch = function () {
    const n = this._xs.length;
    const W = this._width;
    this._pre = new Float64Array(n * W);   // pre-activations (n, W)
    this._h = new Float64Array(n * W);     // ReLU activations
    this._yPred = new Float64Array(n);
    this._err = new Float64Array(n);
    this._dy = new Float64Array(n);
    this._dpre = new Float64Array(n * W);  // dL/d(pre), reused
  };

  // ---- initialization (matches demo2 precompute kink-spread recipe) -------

  TrainingEngine.prototype.reset = function (seed) {
    if (seed != null && Number.isFinite(seed)) this._seed = seed | 0;
    const W = this._width;
    const rng = makeRng(this._seed);
    const randn = normalSampler(rng);

    // Spread initial kinks evenly (jittered) across [xMin, xMax] so each
    // ReLU starts somewhere useful.
    const kinkInit = new Float64Array(W);
    for (let i = 0; i < W; i++) {
      kinkInit[i] = this._xMin + (this._xMax - this._xMin) * rng();
    }
    const w1 = new Float64Array(W);
    const bias = new Float64Array(W);
    const v = new Float64Array(W);
    for (let i = 0; i < W; i++) {
      let w = randn();
      // Floor magnitude away from zero to avoid degenerate kinks.
      if (Math.abs(w) < 0.1) w = (w >= 0 ? 1 : -1) * 0.1;
      w1[i] = w;
      bias[i] = -w * kinkInit[i];
      v[i] = 0.5 * randn();
    }
    this._w1 = w1;
    this._bias = bias;
    this._v = v;
    this._b2 = 0.0;

    // Optimizer state (one per parameter, m + v moments + step count).
    this._optState = {
      m_w1: new Float64Array(W),
      v_w1: new Float64Array(W),
      m_bias: new Float64Array(W),
      v_bias: new Float64Array(W),
      m_v: new Float64Array(W),
      v_v: new Float64Array(W),
      m_b2: 0, v_b2: 0,
      // For SGD+momentum: vel buffers double as the velocity.
      vel_w1: new Float64Array(W),
      vel_bias: new Float64Array(W),
      vel_v: new Float64Array(W),
      vel_b2: 0,
      t: 0,
    };

    this._iter = 0;
    this._lastLoss = this._computeFullLoss();
    return this;
  };

  // ---- forward pass on the training dataset -------------------------------

  TrainingEngine.prototype._forwardAll = function () {
    const xs = this._xs, ys = this._ys, n = xs.length;
    const W = this._width;
    const w1 = this._w1, bias = this._bias, v = this._v;
    const b2 = this._b2;
    const pre = this._pre, h = this._h, yPred = this._yPred, err = this._err;

    let sse = 0;
    for (let i = 0; i < n; i++) {
      const xi = xs[i];
      let s = b2;
      for (let j = 0; j < W; j++) {
        const p = w1[j] * xi + bias[j];
        pre[i * W + j] = p;
        const a = p > 0 ? p : 0;
        h[i * W + j] = a;
        s += v[j] * a;
      }
      yPred[i] = s;
      const e = s - ys[i];
      err[i] = e;
      sse += e * e;
    }
    return sse / n;
  };

  TrainingEngine.prototype._computeFullLoss = function () {
    return this._forwardAll();
  };

  // ---- backward pass + parameter update on a (possibly sub-)batch ---------
  //
  // Strategy: always run a full forward pass (cheap at n=60, gives accurate
  // bookkept loss). Compute gradient over the chosen index set. Caller
  // handles which optimizer to invoke.

  TrainingEngine.prototype._backwardAndUpdate = function (idxs) {
    const W = this._width;
    const xs = this._xs, err = this._err;
    const w1 = this._w1, bias = this._bias, v = this._v;
    const pre = this._pre, h = this._h;
    const m = idxs.length;

    // Gradients over the minibatch.
    const dw1 = new Float64Array(W);
    const dbias = new Float64Array(W);
    const dv = new Float64Array(W);
    let db2 = 0;

    const scale = 2 / m;
    for (let k = 0; k < m; k++) {
      const i = idxs[k];
      const xi = xs[i];
      const dyi = scale * err[i];
      db2 += dyi;
      for (let j = 0; j < W; j++) {
        const a = h[i * W + j];
        dv[j] += dyi * a;
        if (pre[i * W + j] > 0) {
          const dp = dyi * v[j];
          dw1[j] += dp * xi;
          dbias[j] += dp;
        }
      }
    }

    // Apply the chosen optimizer's update rule.
    this._applyUpdate(dw1, dbias, dv, db2);
  };

  TrainingEngine.prototype._applyUpdate = function (dw1, dbias, dv, db2) {
    const W = this._width;
    const lr = this._lr;
    const opt = this._optimizer;
    const st = this._optState;
    st.t += 1;

    if (opt === 'sgd') {
      for (let j = 0; j < W; j++) {
        this._w1[j]   -= lr * dw1[j];
        this._bias[j] -= lr * dbias[j];
        this._v[j]    -= lr * dv[j];
      }
      this._b2 -= lr * db2;
      return;
    }

    if (opt === 'momentum') {
      const beta = this._momentum;
      for (let j = 0; j < W; j++) {
        st.vel_w1[j]   = beta * st.vel_w1[j]   + dw1[j];
        st.vel_bias[j] = beta * st.vel_bias[j] + dbias[j];
        st.vel_v[j]    = beta * st.vel_v[j]    + dv[j];
        this._w1[j]   -= lr * st.vel_w1[j];
        this._bias[j] -= lr * st.vel_bias[j];
        this._v[j]    -= lr * st.vel_v[j];
      }
      st.vel_b2 = beta * st.vel_b2 + db2;
      this._b2 -= lr * st.vel_b2;
      return;
    }

    // Default: Adam, matching the precompute exactly.
    const b1 = this._beta1, b2c = this._beta2, eps = this._eps;
    const t = st.t;
    const c1 = 1 - Math.pow(b1, t);
    const c2 = 1 - Math.pow(b2c, t);
    for (let j = 0; j < W; j++) {
      st.m_w1[j] = b1 * st.m_w1[j] + (1 - b1) * dw1[j];
      st.v_w1[j] = b2c * st.v_w1[j] + (1 - b2c) * dw1[j] * dw1[j];
      const mh1 = st.m_w1[j] / c1, vh1 = st.v_w1[j] / c2;
      this._w1[j] -= lr * mh1 / (Math.sqrt(vh1) + eps);

      st.m_bias[j] = b1 * st.m_bias[j] + (1 - b1) * dbias[j];
      st.v_bias[j] = b2c * st.v_bias[j] + (1 - b2c) * dbias[j] * dbias[j];
      const mh2 = st.m_bias[j] / c1, vh2 = st.v_bias[j] / c2;
      this._bias[j] -= lr * mh2 / (Math.sqrt(vh2) + eps);

      st.m_v[j] = b1 * st.m_v[j] + (1 - b1) * dv[j];
      st.v_v[j] = b2c * st.v_v[j] + (1 - b2c) * dv[j] * dv[j];
      const mh3 = st.m_v[j] / c1, vh3 = st.v_v[j] / c2;
      this._v[j] -= lr * mh3 / (Math.sqrt(vh3) + eps);
    }
    st.m_b2 = b1 * st.m_b2 + (1 - b1) * db2;
    st.v_b2 = b2c * st.v_b2 + (1 - b2c) * db2 * db2;
    const mhb = st.m_b2 / c1, vhb = st.v_b2 / c2;
    this._b2 -= lr * mhb / (Math.sqrt(vhb) + eps);
  };

  // ---- public step --------------------------------------------------------

  TrainingEngine.prototype.step = function (batchSize) {
    const n = this._xs.length;
    // Full-batch is the default and what we trained the precompute with.
    let idxs;
    if (batchSize == null || batchSize >= n) {
      // Recycle a persistent ascending index list.
      if (!this._fullIdx || this._fullIdx.length !== n) {
        this._fullIdx = new Int32Array(n);
        for (let i = 0; i < n; i++) this._fullIdx[i] = i;
      }
      idxs = this._fullIdx;
    } else {
      // Reservoir-style minibatch: deterministic-ish random draw using
      // the engine's seeded rng so that &epoch=N stays reproducible.
      if (!this._batchRng) this._batchRng = makeRng((this._seed * 31 + 7) | 0);
      const m = Math.max(1, batchSize | 0);
      idxs = new Int32Array(m);
      for (let i = 0; i < m; i++) idxs[i] = Math.floor(this._batchRng() * n);
    }

    // Forward over the full dataset for accurate loss bookkeeping; then
    // backward + update on the chosen indices.
    const fullLoss = this._forwardAll();
    this._lastLoss = fullLoss;
    this._backwardAndUpdate(idxs);
    this._iter += 1;
    return fullLoss;
  };

  // ---- public state accessors --------------------------------------------

  TrainingEngine.prototype.params = function () {
    return {
      w1: Array.from(this._w1),
      bias: Array.from(this._bias),
      v: Array.from(this._v),
      b2: this._b2,
      loss: this._lastLoss,
      iter: this._iter,
      width: this._width,
    };
  };

  TrainingEngine.prototype.evalAt = function (xs) {
    const n = xs.length;
    const W = this._width;
    const w1 = this._w1, bias = this._bias, v = this._v, b2 = this._b2;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b2;
      const xi = +xs[i];
      for (let j = 0; j < W; j++) {
        const p = w1[j] * xi + bias[j];
        if (p > 0) s += v[j] * p;
      }
      out[i] = s;
    }
    return out;
  };

  // Per-neuron contribution v_j * ReLU(w1_j * x + bias_j) over xs.
  TrainingEngine.prototype.evalNeuron = function (idx, xs) {
    const W = this._width;
    if (idx < 0 || idx >= W) return new Float64Array(xs.length);
    const w1 = this._w1[idx], bj = this._bias[idx], vj = this._v[idx];
    const out = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const p = w1 * (+xs[i]) + bj;
      out[i] = p > 0 ? vj * p : 0;
    }
    return out;
  };

  // Setters that take effect on the next step without resetting state.
  TrainingEngine.prototype.setLR = function (lr) {
    if (Number.isFinite(lr) && lr > 0) this._lr = +lr;
  };
  TrainingEngine.prototype.setOptimizer = function (name) {
    if (name === 'sgd' || name === 'momentum' || name === 'adam') {
      this._optimizer = name;
    }
  };
  TrainingEngine.prototype.getLR = function () { return this._lr; };
  TrainingEngine.prototype.getOptimizer = function () { return this._optimizer; };
  TrainingEngine.prototype.getWidth = function () { return this._width; };
  TrainingEngine.prototype.getSeed = function () { return this._seed; };

  // ---- export -------------------------------------------------------------

  window.TrainingEngine = TrainingEngine;
})();
