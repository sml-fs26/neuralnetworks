/* 1-D loss landscapes for the optimizer-comparison scene.
   Three preset scenarios chosen so that GD, SGD, Momentum, and Adam land in
   a strict dominance order from a fixed start theta_0 within ~200 steps:
     wiggly      -- GD beats SGD?  No: GD < SGD in *quality*, i.e. final loss
                    GD-loss > SGD-loss because GD is trapped in a local min
                    while SGD's noise hops it to the global minimum.
     long_valley -- SGD < Momentum (Momentum's accumulated velocity carries
                    it across the long flat side faster than SGD can crawl).
     asym_well   -- Momentum < Adam (Momentum builds large velocity on the
                    gentle side and oscillates against the steep wall;
                    Adam's per-parameter scaling by sqrt(v_hat) tames the
                    gradient-magnitude jump and converges).

   Verification (200 steps, seed=1, optimizers as in optimizers.js):
     beta=0.9 for momentum;  beta1=0.9, beta2=0.999, eps=1e-8 for adam.
     gd is noiseless; sgd/momentum/adam use the scenario's defaultSigma.

     +-------------+--------+----------+----------+----------+----------+
     | scenario    | sigma  |  gd      |  sgd     | momentum |  adam    |
     +-------------+--------+----------+----------+----------+----------+
     | wiggly      |  2.00  |  0.5566  | -0.3825  | -0.1574  |  0.5575  |
     | long_valley |  0.05  |  7.3e-3  |  1.2e-2  |  8.4e-7  |  2.1e-3  |
     | asym_well   |  1.00  |  3.3e-1  |  5.1e-1  |  2.8e-1  |  5.2e-4  |
     +-------------+--------+----------+----------+----------+----------+

   Intended dominance orderings (final loss, lower is better):
     wiggly      :  SGD ( -0.38 )  <<  GD ( 0.56 )           [SGD wins]
     long_valley :  Mom ( 8e-7 )   <<  SGD ( 1.2e-2 )        [Momentum wins]
     asym_well   :  Adam ( 5e-4 )  <<  Mom ( 0.28 )          [Adam wins]
   All three margins are >> 3x as required.

   Style: pure module, IIFE-wrapped, attaches to window.Loss1D, no DOM. */
(function () {
  // ---------------- Scenario definitions --------------------------------

  // 1) wiggly: smooth quadratic with a cosine ripple. Multiple local minima.
  //    Starting at theta=4.5, GD slides into the trap near theta=4.48
  //    (L ~ 0.557). The next local maximum is near theta=3.31 (L ~ 1.02);
  //    the global minima are at theta = +/- 1.495 (L ~ -0.382).
  //    SGD with sigma=2.0 produces enough kicks to cross the barrier and
  //    settle into the global minimum.
  const wiggly = {
    label: 'Wiggly basin (GD gets stuck, SGD escapes)',
    description:
      'A bumpy bowl with several local minima. Plain GD slides into the' +
      ' nearest valley and freezes; SGD\'s noise hops over the small bumps' +
      ' toward the global minimum.',
    thetaMin: -6,
    thetaMax: 6,
    defaultStart: 4.5,
    defaultLr: { gd: 0.1, sgd: 0.1, momentum: 0.05, adam: 0.1 },
    defaultSigma: 2.0,
    evaluate: function (theta) {
      return 0.05 * theta * theta + 0.5 * Math.cos(2 * theta);
    },
    grad: function (theta) {
      return 0.1 * theta - Math.sin(2 * theta);
    },
  };

  // 2) long_valley: piecewise quadratic, very gentle for theta < 0 and
  //    moderately steep for theta >= 0. From a far-left start the gradient
  //    is tiny so SGD/GD crawl, but Momentum accumulates velocity and
  //    arrives at the optimum much sooner.
  const long_valley = {
    label: 'Long valley (Momentum picks up speed)',
    description:
      'A long, flat slope on the left and a steeper bowl on the right.' +
      ' SGD and GD inch along the flat; Momentum accumulates velocity and' +
      ' arrives at the optimum much sooner.',
    thetaMin: -10,
    thetaMax: 4,
    defaultStart: -9,
    defaultLr: { gd: 1.0, sgd: 1.0, momentum: 0.5, adam: 0.5 },
    defaultSigma: 0.05,
    evaluate: function (theta) {
      return theta < 0 ? 0.005 * theta * theta : 0.1 * theta * theta;
    },
    grad: function (theta) {
      return theta < 0 ? 0.01 * theta : 0.2 * theta;
    },
  };

  // 3) asym_well: piecewise quadratic with a 40x curvature jump at theta=0.
  //    Gentle side (theta < 0): coefficient 0.05; steep side (theta >= 0):
  //    coefficient 2.0. Momentum builds velocity on the gentle side and
  //    crashes into the steep wall, oscillating wildly. Adam normalizes by
  //    sqrt(v_hat) so the per-step move size is tamed across the regime
  //    change and convergence is smooth.
  const asym_well = {
    label: 'Asymmetric well (Adam tames the gradient jump)',
    description:
      'A gentle slope meets a steep wall at theta=0. Momentum overshoots' +
      ' and oscillates against the wall; Adam rescales by sqrt(v_hat) and' +
      ' settles smoothly.',
    thetaMin: -8,
    thetaMax: 4,
    defaultStart: -7,
    defaultLr: { gd: 0.05, sgd: 0.05, momentum: 0.2, adam: 0.3 },
    defaultSigma: 1.0,
    evaluate: function (theta) {
      return theta < 0 ? 0.05 * theta * theta : 2.0 * theta * theta;
    },
    grad: function (theta) {
      return theta < 0 ? 0.1 * theta : 4.0 * theta;
    },
  };

  const SCENARIOS = { wiggly: wiggly, long_valley: long_valley, asym_well: asym_well };

  // ---------------- Stochastic gradient ---------------------------------

  // Adds Gaussian noise of std sigma to the analytic gradient. The caller
  // supplies a pre-made rng() (uniform on [0,1)) -- typically built via
  // window.LossUtils.makeRng(seed). We use a fresh Box-Muller sample on
  // every call (no cached half) so calls are stateless w.r.t. the rng.
  function stochasticGrad(theta, scenario, sigma, rng) {
    const g = scenario.grad(theta);
    if (sigma <= 0) return g;
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    const n = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return g + sigma * n;
  }

  // ---------------- Curve sampling --------------------------------------

  // Returns an array of [theta, L(theta)] pairs of length n, evenly spaced
  // over [scenario.thetaMin, scenario.thetaMax]. Useful for plotting.
  function sampleCurve(scenario, n) {
    const out = new Array(n);
    if (n <= 1) {
      const t = scenario.thetaMin;
      out[0] = [t, scenario.evaluate(t)];
      return out;
    }
    const span = scenario.thetaMax - scenario.thetaMin;
    for (let i = 0; i < n; i++) {
      const theta = scenario.thetaMin + (span * i) / (n - 1);
      out[i] = [theta, scenario.evaluate(theta)];
    }
    return out;
  }

  // ---------------- Export ----------------------------------------------

  window.Loss1D = {
    SCENARIOS: SCENARIOS,
    stochasticGrad: stochasticGrad,
    sampleCurve: sampleCurve,
  };
})();
