/* Loss-surface and contour utilities for the gradient-descent viz.

   Used by scene 1's bowl visualization and shared with main.js. All routines
   are pure functions of their inputs; no DOM, no state. */
(function () {
  // ---------------- Linear regression: y = a*x + b -----------------------

  // MSE on a fixed dataset for parameters (a, b).
  function mseLinear(a, b, points) {
    let s = 0;
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const r = (a * points[i][0] + b) - points[i][1];
      s += r * r;
    }
    return s / n;
  }

  // Analytic gradient of MSE with respect to (a, b).
  function gradLinear(a, b, points) {
    let da = 0, db = 0;
    const n = points.length;
    for (let i = 0; i < n; i++) {
      const x = points[i][0], y = points[i][1];
      const r = a * x + b - y;
      da += 2 * r * x;
      db += 2 * r;
    }
    return [da / n, db / n];
  }

  // Stochastic gradient over a minibatch.
  function gradLinearBatch(a, b, points, idxs) {
    let da = 0, db = 0;
    const n = idxs.length;
    for (let i = 0; i < n; i++) {
      const p = points[idxs[i]];
      const r = a * p[0] + b - p[1];
      da += 2 * r * p[0];
      db += 2 * r;
    }
    return [da / n, db / n];
  }

  // ---------------- MSE grid for contour rendering ----------------------

  // Compute MSE on a regular grid in (a, b) space.
  // Returns a Float32Array of length nA*nB, row-major (b varies fastest).
  function mseGrid(aMin, aMax, nA, bMin, bMax, nB, points) {
    const grid = new Float32Array(nA * nB);
    for (let i = 0; i < nA; i++) {
      const a = aMin + (aMax - aMin) * i / (nA - 1);
      for (let j = 0; j < nB; j++) {
        const b = bMin + (bMax - bMin) * j / (nB - 1);
        grid[i * nB + j] = mseLinear(a, b, points);
      }
    }
    return grid;
  }

  // ---------------- Marching squares for contour paths ------------------

  // Given a 2D scalar grid and an iso-level, produce SVG path-d strings as
  // collections of line segments. Coordinate system: grid[i*nB+j] is the
  // value at column i, row j; we output x = i, y = j (caller scales).
  function contourSegments(grid, nA, nB, level) {
    const segs = [];

    function interp(v1, v2) {
      const t = (level - v1) / (v2 - v1);
      return Math.max(0, Math.min(1, t));
    }

    for (let i = 0; i < nA - 1; i++) {
      for (let j = 0; j < nB - 1; j++) {
        const v00 = grid[i * nB + j];
        const v10 = grid[(i + 1) * nB + j];
        const v01 = grid[i * nB + j + 1];
        const v11 = grid[(i + 1) * nB + j + 1];

        let code = 0;
        if (v00 > level) code |= 1;
        if (v10 > level) code |= 2;
        if (v11 > level) code |= 4;
        if (v01 > level) code |= 8;
        if (code === 0 || code === 15) continue;

        // Edge-midpoints: bottom (a-b), right (b-c), top (c-d), left (d-a)
        const eb = [i + interp(v00, v10), j];
        const er = [i + 1, j + interp(v10, v11)];
        const et = [i + interp(v01, v11), j + 1];
        const el = [i, j + interp(v00, v01)];

        switch (code) {
          case 1: case 14: segs.push([el, eb]); break;
          case 2: case 13: segs.push([eb, er]); break;
          case 3: case 12: segs.push([el, er]); break;
          case 4: case 11: segs.push([er, et]); break;
          case 5:          segs.push([el, et], [eb, er]); break;
          case 6: case 9:  segs.push([eb, et]); break;
          case 7: case 8:  segs.push([el, et]); break;
          case 10:         segs.push([el, eb], [er, et]); break;
        }
      }
    }
    return segs;
  }

  // Build SVG path-d strings for an array of iso-levels. The transform fn
  // converts grid (i, j) coordinates to SVG (x, y) viewBox coords.
  function contourPaths(grid, nA, nB, levels, transform) {
    return levels.map((level) => {
      const segs = contourSegments(grid, nA, nB, level);
      const parts = new Array(segs.length);
      for (let k = 0; k < segs.length; k++) {
        const [p1, p2] = segs[k];
        const [x1, y1] = transform(p1[0], p1[1]);
        const [x2, y2] = transform(p2[0], p2[1]);
        parts[k] = `M${x1.toFixed(1)},${y1.toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}`;
      }
      return { level, d: parts.join(' ') };
    });
  }

  // Pick a set of iso-levels for nice contour spacing on a positive
  // bounded scalar field. Uses log spacing tuned for MSE-style fields
  // that grow steeply away from a small minimum.
  function chooseLevels(grid, nLevels) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < grid.length; i++) {
      const v = grid[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo <= 0) lo = 1e-6;
    const out = new Array(nLevels);
    for (let k = 0; k < nLevels; k++) {
      const t = (k + 1) / (nLevels + 1);
      out[k] = Math.exp(Math.log(lo) + t * (Math.log(hi) - Math.log(lo)));
    }
    return out;
  }

  // ---------------- Misc helpers ----------------------------------------

  // Mulberry32 PRNG -- deterministic seeding for reproducible inits.
  function makeRng(seed) {
    let s = (seed | 0) || 1;
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Fisher-Yates shuffle (in place).
  function shuffleInPlace(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }

  window.LossUtils = {
    mseLinear, gradLinear, gradLinearBatch,
    mseGrid, contourSegments, contourPaths, chooseLevels,
    makeRng, shuffleInPlace,
  };
})();
