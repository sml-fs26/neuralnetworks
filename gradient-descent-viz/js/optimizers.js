/* Optimizer step functions. Pure: each optimizer takes its current state
   and a gradient, returns the parameter update. Caller owns the parameters
   and the optimizer state, so reset is trivial.

   All gradients are vectors of arbitrary length; works for both the
   2-parameter linear-regression case (scene 1) and the 3N+1 parameter MLP
   case (scene 2). */
(function () {
  function vecAdd(a, b)   { const o = new Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] + b[i]; return o; }
  function vecScale(v, s) { const o = new Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] * s; return o; }

  // ---------------- Plain SGD --------------------------------------------

  function sgdInit() { return {}; }

  function sgdStep(state, grad, hp) {
    const lr = hp.lr;
    const update = vecScale(grad, -lr);
    return { state, update };
  }

  // ---------------- SGD + momentum (heavy-ball) --------------------------

  function momentumInit(dim) {
    return { v: new Array(dim).fill(0) };
  }

  function momentumStep(state, grad, hp) {
    const lr = hp.lr;
    const beta = hp.momentum != null ? hp.momentum : 0.9;
    const v = new Array(grad.length);
    for (let i = 0; i < grad.length; i++) {
      v[i] = beta * state.v[i] + grad[i];
    }
    const update = vecScale(v, -lr);
    return { state: { v }, update };
  }

  // ---------------- Adam --------------------------------------------------

  function adamInit(dim) {
    return { m: new Array(dim).fill(0), v: new Array(dim).fill(0), t: 0 };
  }

  function adamStep(state, grad, hp) {
    const lr = hp.lr;
    const b1 = hp.beta1 != null ? hp.beta1 : 0.9;
    const b2 = hp.beta2 != null ? hp.beta2 : 0.999;
    const eps = hp.eps != null ? hp.eps : 1e-8;
    const t = state.t + 1;
    const m = new Array(grad.length);
    const v = new Array(grad.length);
    const update = new Array(grad.length);
    for (let i = 0; i < grad.length; i++) {
      m[i] = b1 * state.m[i] + (1 - b1) * grad[i];
      v[i] = b2 * state.v[i] + (1 - b2) * grad[i] * grad[i];
      const mh = m[i] / (1 - Math.pow(b1, t));
      const vh = v[i] / (1 - Math.pow(b2, t));
      update[i] = -lr * mh / (Math.sqrt(vh) + eps);
    }
    return { state: { m, v, t }, update };
  }

  // ---------------- Registry ---------------------------------------------

  const OPTIMIZERS = {
    gd:       { init: sgdInit,      step: sgdStep,      label: 'Gradient Descent (full batch)' },
    sgd:      { init: sgdInit,      step: sgdStep,      label: 'SGD' },
    momentum: { init: momentumInit, step: momentumStep, label: 'SGD + Momentum' },
    adam:     { init: adamInit,     step: adamStep,     label: 'Adam' },
  };

  window.Optimizers = { OPTIMIZERS, vecAdd, vecScale };
})();
