/* U-Net JS utilities. Numerical helpers for the deepdive scenes.

   Mirrors `cnn-deepdive/js/cnn.js`'s naming conventions but is scoped to
   `window.UNET`. Agent B / B7 is expected to extend this file with the
   forward-pass machinery the scenes need (transposed conv, concat skip,
   etc.). For Agent A we ship just the array-shape helpers that the shared
   drawing helpers depend on. */
(function () {
  'use strict';

  function zeros2D(h, w) {
    const a = new Array(h);
    for (let i = 0; i < h; i++) {
      const row = new Array(w);
      for (let j = 0; j < w; j++) row[j] = 0;
      a[i] = row;
    }
    return a;
  }

  function zeros3D(c, h, w) {
    const a = new Array(c);
    for (let i = 0; i < c; i++) a[i] = zeros2D(h, w);
    return a;
  }

  /* Stats on a 2D tensor. Identical to cnn.js#range2D so painters can use
     either namespace interchangeably. */
  function range2D(x) {
    let lo = Infinity, hi = -Infinity;
    for (const row of x) for (const v of row) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo)) lo = 0;
    if (!isFinite(hi)) hi = 0;
    return { lo, hi };
  }

  window.UNET = window.UNET || {};
  window.UNET.zeros2D = zeros2D;
  window.UNET.zeros3D = zeros3D;
  window.UNET.range2D = range2D;
})();
