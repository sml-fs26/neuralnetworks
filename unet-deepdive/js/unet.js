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

  /* ----------------------------------------------------------------------
     U-Net mini-map.

     Reusable little SVG diagram of the full U-Net with an optional set of
     "you are here" highlights. Used as a sidebar in scenes that focus on
     one specific part of the network (filters, bottleneck, pool/upsample,
     etc.) so the viewer always knows where in the architecture we are.

     mountUNetMiniMap(host, opts) — builds the SVG into `host`. Returns
     a `setHighlight(keys, label)` function for live updates.

     Highlight keys (any subset):
       'enc1', 'enc2', 'enc3' (= bottleneck), 'dec2', 'dec1', 'out'
       'pool1', 'pool2'   (the two max-pool operations)
       'up2', 'up1'       (the two transposed-conv upsamples)
       'skip1', 'skip2'   (the two skip-connection arcs)

     opts:
       width:   overall SVG width in px (default 240)
       label:   optional short text under the diagram
       title:   optional title shown above the diagram
  */
  function mountUNetMiniMap(host, opts) {
    opts = opts || {};
    const W = opts.width || 240;
    const H = opts.height || 130;
    host.innerHTML = '';
    host.classList.add('unet-mini');

    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', String(W));
    svg.setAttribute('height', String(H));
    svg.setAttribute('class', 'unet-mini-svg');
    host.appendChild(svg);

    /* Layout: an upside-down U.
       Encoder column (left): enc1 (top), enc2 (mid), enc3 (bottom = bottleneck)
       Decoder column (right): dec1 (top), dec2 (mid)
       'out' is a small chip to the right of dec1.
       Skip arcs go horizontally over the top of the diagram from enc1->dec1
       and enc2->dec2.
       Pool arrows point downwards on the left; upsample arrows point
       upwards on the right. */
    const padX = 16, padY = 26;
    const colW = 38;
    const rowH = 22;
    const gap = 6;
    const encX = padX;
    const decX = W - padX - colW;
    const yTop = padY;
    const yMid = padY + rowH + gap;
    const yBot = padY + 2 * (rowH + gap);

    function box(x, y, w, h, label, key) {
      const g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'unet-mini-box');
      g.setAttribute('data-key', key);
      const r = document.createElementNS(NS, 'rect');
      r.setAttribute('x', String(x));
      r.setAttribute('y', String(y));
      r.setAttribute('width', String(w));
      r.setAttribute('height', String(h));
      r.setAttribute('rx', '2');
      g.appendChild(r);
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('x', String(x + w / 2));
      t.setAttribute('y', String(y + h / 2 + 3.5));
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('font-size', '9');
      t.setAttribute('font-family', 'SF Mono, Menlo, monospace');
      t.textContent = label;
      g.appendChild(t);
      svg.appendChild(g);
      return { g, x, y, w, h };
    }

    function arrow(x1, y1, x2, y2, key, dashed) {
      const g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'unet-mini-arrow');
      g.setAttribute('data-key', key);
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('d', `M ${x1} ${y1} L ${x2} ${y2}`);
      if (dashed) path.setAttribute('stroke-dasharray', '3 2');
      g.appendChild(path);
      svg.appendChild(g);
      return g;
    }

    function arc(x1, y, x2, key) {
      const g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'unet-mini-skip');
      g.setAttribute('data-key', key);
      const ctrlY = y - 14;
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('d',
        `M ${x1} ${y} C ${x1 + 14} ${ctrlY}, ${x2 - 14} ${ctrlY}, ${x2} ${y}`);
      path.setAttribute('fill', 'none');
      g.appendChild(path);
      svg.appendChild(g);
      return g;
    }

    // Skip arcs first (under the boxes visually if z-order matters, but
    // browsers paint in document order so anything later overlays earlier;
    // we want boxes on top).
    const skip1 = arc(encX + colW / 2, yTop, decX + colW / 2, 'skip1');
    const skip2 = arc(encX + colW / 2, yMid, decX + colW / 2, 'skip2');

    // Pool arrows (left column, going down)
    const pool1 = arrow(encX + colW / 2, yTop + rowH, encX + colW / 2, yMid, 'pool1');
    const pool2 = arrow(encX + colW / 2, yMid + rowH, encX + colW / 2, yBot, 'pool2');

    // Upsample arrows (right column, going up)
    const up2 = arrow(decX + colW / 2, yMid + rowH, decX + colW / 2, yMid + 2, 'up2', true);
    const up1 = arrow(decX + colW / 2, yMid, decX + colW / 2, yTop + rowH + 2, 'up1', true);

    // Bridge from enc3 (bottleneck) into dec2 — the up2 starts here
    const bridge = arrow(encX + colW, yBot + rowH / 2, decX, yMid + rowH, 'bridge');
    bridge.setAttribute('class', 'unet-mini-arrow unet-mini-bridge');

    // Encoder column
    const ENC1 = box(encX, yTop, colW, rowH, 'enc1', 'enc1');
    const ENC2 = box(encX, yMid, colW, rowH, 'enc2', 'enc2');
    // Bottleneck box: keep label short ("enc3") so it fits the box. The
    // explanatory caption underneath the SVG carries the "/ bottleneck" framing.
    const ENC3 = box(encX, yBot, colW, rowH, 'enc3', 'enc3');
    // Decoder column
    const DEC2 = box(decX, yMid, colW, rowH, 'dec2', 'dec2');
    const DEC1 = box(decX, yTop, colW, rowH, 'dec1', 'dec1');
    // Output head chip (1×1 conv, small, right of dec1)
    const OUT  = box(decX + colW + 4, yTop, 18, rowH, '1×1', 'out');

    // Optional caption beneath
    let captionEl = null;
    if (opts.title) {
      const ttl = document.createElement('div');
      ttl.className = 'unet-mini-title';
      ttl.textContent = opts.title;
      host.insertBefore(ttl, svg);
    }
    if (opts.label !== undefined) {
      captionEl = document.createElement('div');
      captionEl.className = 'unet-mini-label';
      captionEl.textContent = opts.label || '';
      host.appendChild(captionEl);
    }

    function setHighlight(keys, label) {
      const set = new Set(Array.isArray(keys) ? keys : (keys ? [keys] : []));
      svg.querySelectorAll('[data-key]').forEach(function (g) {
        if (set.has(g.getAttribute('data-key'))) g.classList.add('unet-mini-active');
        else g.classList.remove('unet-mini-active');
      });
      if (captionEl && label !== undefined) captionEl.textContent = label || '';
    }

    return { svg: svg, setHighlight: setHighlight };
  }

  window.UNET = window.UNET || {};
  window.UNET.zeros2D = zeros2D;
  window.UNET.zeros3D = zeros3D;
  window.UNET.range2D = range2D;
  window.UNET.mountUNetMiniMap = mountUNetMiniMap;
})();
