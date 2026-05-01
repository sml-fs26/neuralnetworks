/* Scene 4 -- "Stacking filters becomes a unit."

   Three layer-1 filters fire on different parts of a triangle:
     diag_down  -> right-leaning side
     diag_up    -> left-leaning side
     horizontal -> base
   A layer-2 unit takes a positive-weighted sum of all three feature maps,
   adds a bias, then ReLU. The unit fires only where the three responses
   overlap -- i.e. AT the triangle.

   Layout:
     - KaTeX display formula at top.
     - Three feature-map panels below, side by side.
     - A wire diagram converges the three streams to a single circle (the
       unit), then ReLU, then an output panel.
     - A running scalar readout shows the weighted sum at one chosen pixel
       as the user steps through.
     - Steps:
         0 input only
         1 reveal h^1 (diag_down)
         2 reveal h^2 (diag_up)
         3 reveal h^3 (horizontal)
         4 wire animation (particles flow toward the unit)
         5 weighted-sum readout: 0.4*h^1 + 0.4*h^2 + 0.4*h^3
         6 add bias, ReLU -> output map
         7 highlight peak location, overlay back on input

   `&run` auto-advances to step 7. */
(function () {
  'use strict';

  const W = 0.4;       // common positive weight
  const BIAS = -0.5;
  const NUM_STEPS = 8;

  // Canvas sizes (logical)
  const FMAP_PX = 140;
  const KERN_PX = 50;
  const OUT_PX = 168;
  const INPUT_PX = 168;

  function el(tag, attrs, parent) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function readHashFlag(name) {
    const re = new RegExp('[#&?]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
  }

  /* Compute h^c = ReLU(conv2d(input, filter, pad=2)) for each of the three
     filters. Then weighted sum, then bias + ReLU. Returns everything for
     stepwise reveal. */
  function computeAll(input) {
    const filters = window.DATA.handFilters;
    const h1 = window.CNN.relu2D(window.CNN.conv2d(input, filters.diag_down, 2));
    const h2 = window.CNN.relu2D(window.CNN.conv2d(input, filters.diag_up, 2));
    const h3 = window.CNN.relu2D(window.CNN.conv2d(input, filters.horizontal, 2));
    const H = h1.length, Wd = h1[0].length;
    const z = window.CNN.zeros2D(H, Wd);
    for (let i = 0; i < H; i++) for (let j = 0; j < Wd; j++) {
      z[i][j] = W * h1[i][j] + W * h2[i][j] + W * h3[i][j] + BIAS;
    }
    const y = window.CNN.relu2D(z);

    // Find the peak location of y for step 7 highlight.
    let bi = 0, bj = 0, bv = -Infinity;
    for (let i = 0; i < H; i++) for (let j = 0; j < Wd; j++) {
      if (y[i][j] > bv) { bv = y[i][j]; bi = i; bj = j; }
    }
    return { h1, h2, h3, z, y, peak: { i: bi, j: bj, v: bv } };
  }

  /* All-zero map of same shape, used to leave a feature panel blank
     until its step is reached. */
  function blankLike(map) {
    return window.CNN.zeros2D(map.length, map[0].length);
  }

  function symmetricRange(maps) {
    let m = 0;
    for (const map of maps) {
      const r = window.CNN.range2D(map);
      const v = Math.max(Math.abs(r.lo), Math.abs(r.hi));
      if (v > m) m = v;
    }
    return m || 1;
  }

  function paintInput(host, input, highlight) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, INPUT_PX, INPUT_PX);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, input, 0, 0, w, h, {
      diverging: false, valueRange: [0, 1],
    });
    if (highlight) {
      const cw = w / input[0].length;
      const ch = h / input.length;
      // Peak is in feature-map coordinates; same H/W as input here (pad=2).
      ctx.strokeStyle = t.pos;
      ctx.lineWidth = 2.5;
      const r = 5;
      ctx.strokeRect(
        (highlight.j - r) * cw,
        (highlight.i - r) * ch,
        (2 * r + 1) * cw,
        (2 * r + 1) * ch
      );
    }
  }

  function paintFmap(host, fmap, vmax, opts) {
    host.innerHTML = '';
    const px = (opts && opts.size) || FMAP_PX;
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, fmap, 0, 0, w, h, {
      diverging: true, valueRange: [-vmax, vmax],
    });
    if (opts && opts.peak) {
      const cw = w / fmap[0].length;
      const ch = h / fmap.length;
      ctx.strokeStyle = t.pos;
      ctx.lineWidth = 2;
      ctx.strokeRect(opts.peak.j * cw - 1, opts.peak.i * ch - 1, cw + 2, ch + 2);
    }
  }

  function paintKernel(host, filter) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, KERN_PX, KERN_PX);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, filter, 0, 0, w, h, {
      diverging: true, cellBorder: true,
    });
  }

  function paintBlank(host, size) {
    host.innerHTML = '';
    const px = size || FMAP_PX;
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    // Subtle dashed centre cross to communicate "not yet computed".
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters) {
      root.innerHTML = '<p style="opacity:0.5">handFilters missing.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s4-root');
    const wrap = el('div', { class: 's4-wrap' }, root);

    // ---- Hero ----------------------------------------------------------
    const hero = el('header', { class: 'hero s4-hero' }, wrap);
    el('h1', { text: 'Stacking filters becomes a unit.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'A layer-2 unit reads three layer-1 feature maps. Each filter sees a piece of the shape; the unit fires only where the pieces line up.',
    }, hero);

    // ---- Formula -------------------------------------------------------
    const formulaHost = el('div', { class: 's4-formula' }, wrap);
    window.Katex.render(
      'y \\;=\\; \\mathrm{ReLU}\\!\\Big(' +
      '\\sum_{c=1}^{3} \\sum_{u,v} w^{(c)}_{u,v}\\, h^{(c)}_{u,v} ' +
      '\\;+\\; b\\Big)',
      formulaHost, true
    );

    // ---- Diagram strip -------------------------------------------------
    const strip = el('div', { class: 's4-strip' }, wrap);

    // Left: the input column
    const inputCol = el('div', { class: 's4-col s4-col-input' }, strip);
    el('div', { class: 's4-col-label', text: 'input' }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s4-input-host' }, inputCol);

    // Middle: the three feature-map columns
    const fmapsCol = el('div', { class: 's4-col s4-col-fmaps' }, strip);
    el('div', { class: 's4-col-label', text: 'layer-1 feature maps' }, fmapsCol);
    const fmapsRow = el('div', { class: 's4-fmaps-row' }, fmapsCol);

    function buildFmapPanel(name, key) {
      const panel = el('div', { class: 's4-fpanel' }, fmapsRow);
      const top = el('div', { class: 's4-fpanel-top' }, panel);
      const kHost = el('div', { class: 'canvas-host s4-kernel-host' }, top);
      const cap = el('div', { class: 's4-fpanel-cap' }, top);
      el('div', { class: 's4-fpanel-name', html: name }, cap);
      el('div', { class: 's4-fpanel-key', text: key }, cap);
      const fHost = el('div', { class: 'canvas-host s4-fmap-host' }, panel);
      return { panel, kHost, fHost };
    }
    const p1 = buildFmapPanel('h<sup>1</sup>', 'diag ↘');
    const p2 = buildFmapPanel('h<sup>2</sup>', 'diag ↙');
    const p3 = buildFmapPanel('h<sup>3</sup>', 'horiz');

    // SVG wires (overlay across the strip): drawn in foreground so they
    // appear on top of card borders.
    const wireSvg = el('div', { class: 's4-wires' }, strip);
    wireSvg.innerHTML =
      '<svg viewBox="0 0 600 220" preserveAspectRatio="none" ' +
      'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<g class="s4-wire-group">' +
        '<path class="s4-wire" id="s4-wire-1" d="M 0 30 Q 250 30, 360 110" />' +
        '<path class="s4-wire" id="s4-wire-2" d="M 0 110 L 360 110" />' +
        '<path class="s4-wire" id="s4-wire-3" d="M 0 190 Q 250 190, 360 110" />' +
        '<circle class="s4-unit" cx="360" cy="110" r="14" />' +
        '<text class="s4-unit-label" x="360" y="114" text-anchor="middle">Σ</text>' +
        '<path class="s4-wire-out" d="M 374 110 L 470 110" />' +
        '<text class="s4-relu-label" x="430" y="98" text-anchor="middle">ReLU</text>' +
        '<g class="s4-particle-host"></g>' +
      '</g>' +
      '</svg>';

    // Right: the unit + ReLU + output column
    const outCol = el('div', { class: 's4-col s4-col-out' }, strip);
    el('div', { class: 's4-col-label', text: 'unit output' }, outCol);
    const outHost = el('div', { class: 'canvas-host s4-out-host' }, outCol);

    // ---- Numeric readout (step 5) -------------------------------------
    const readoutBlock = el('div', { class: 's4-readout' }, wrap);
    const readoutLabel = el('div', { class: 's4-readout-label',
      text: 'At the peak pixel:' }, readoutBlock);
    void readoutLabel;
    const readoutLatex = el('div', { class: 's4-readout-tex' }, readoutBlock);

    // ---- Step controls -------------------------------------------------
    const controls = el('div', { class: 'controls s4-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's4-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's4-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / 7' }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    // Caption beneath
    const caption = el('p', { class: 'caption s4-caption' }, wrap);

    // ---- State ---------------------------------------------------------
    const input = window.Drawing.makeSample('triangle', 28);
    const all = computeAll(input);
    const vmax = symmetricRange([all.h1, all.h2, all.h3, all.y]);
    const blank = blankLike(all.h1);

    const state = {
      step: 0,
    };

    function captionFor(step) {
      switch (step) {
        case 0: return 'A triangle. Three sides: down-right, down-left, and a horizontal base.';
        case 1: return 'Filter h¹ (diag ↘) lights up the right-leaning side.';
        case 2: return 'Filter h² (diag ↙) lights up the left-leaning side.';
        case 3: return 'Filter h³ (horizontal) lights up the base.';
        case 4: return 'Three streams converge into a single unit.';
        case 5: return 'Weighted sum: equal positive weights of 0.4 on each feature map.';
        case 6: return 'Add a bias of −0.5 and apply ReLU. Most pixels go dark; only the corner where all three sides meet survives.';
        case 7: return 'The peak. The unit fired exactly where the triangle is.';
        default: return '';
      }
    }

    function setReadoutForStep(step) {
      readoutLatex.innerHTML = '';
      readoutBlock.classList.remove('visible');
      if (step < 5) return;
      const pi = all.peak.i, pj = all.peak.j;
      const a = all.h1[pi][pj];
      const b = all.h2[pi][pj];
      const c = all.h3[pi][pj];
      const sum = W * a + W * b + W * c;
      const z = sum + BIAS;
      const y = Math.max(0, z);

      let tex;
      if (step === 5) {
        tex =
          'z \\;=\\; ' + W.toFixed(1) + '\\cdot ' + a.toFixed(2) +
          ' \\;+\\; ' + W.toFixed(1) + '\\cdot ' + b.toFixed(2) +
          ' \\;+\\; ' + W.toFixed(1) + '\\cdot ' + c.toFixed(2) +
          ' \\;=\\; ' + sum.toFixed(2);
      } else {
        // Step 6 and 7: include bias + ReLU.
        const biasStr = BIAS < 0 ? '\\;-\\;' + (-BIAS).toFixed(1)
                                : '\\;+\\;' + BIAS.toFixed(1);
        tex =
          'y \\;=\\; \\mathrm{ReLU}(' + sum.toFixed(2) + biasStr +
          ') \\;=\\; \\mathrm{ReLU}(' + z.toFixed(2) + ') \\;=\\; ' +
          y.toFixed(2);
      }
      window.Katex.render(tex, readoutLatex, true);
      readoutBlock.classList.add('visible');
    }

    function render() {
      const step = state.step;
      // Input panel: highlight peak location only at step 7.
      paintInput(inputHost, input, step >= 7 ? all.peak : null);

      // Filter kernels (always visible from step 0 -- they explain the labels).
      paintKernel(p1.kHost, window.DATA.handFilters.diag_down);
      paintKernel(p2.kHost, window.DATA.handFilters.diag_up);
      paintKernel(p3.kHost, window.DATA.handFilters.horizontal);

      // Feature maps reveal one step at a time.
      if (step >= 1) paintFmap(p1.fHost, all.h1, vmax); else paintBlank(p1.fHost);
      if (step >= 2) paintFmap(p2.fHost, all.h2, vmax); else paintBlank(p2.fHost);
      if (step >= 3) paintFmap(p3.fHost, all.h3, vmax); else paintBlank(p3.fHost);

      // Wires lit only from step 4 onward; particles only at step 4.
      strip.classList.toggle('s4-wires-lit', step >= 4);
      strip.classList.toggle('s4-wires-particles', step === 4);

      // Output map only from step 6.
      if (step >= 6) {
        paintFmap(outHost, all.y, vmax, { peak: step >= 7 ? all.peak : null });
      } else {
        paintBlank(outHost, OUT_PX);
      }

      // Readout & caption
      setReadoutForStep(step);
      caption.textContent = captionFor(step);

      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    prevBtn.addEventListener('click', () => applyStep(state.step - 1));
    nextBtn.addEventListener('click', () => applyStep(state.step + 1));
    resetBtn.addEventListener('click', () => applyStep(0));
    stepInput.addEventListener('input', () => {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    // ---- &run auto-advance --------------------------------------------
    let runTimer = null;
    function autoAdvance() {
      if (state.step >= NUM_STEPS - 1) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, 900);
    }

    // Initial paint
    render();

    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 200);
    }

    return {
      onEnter() { render(); },
      onLeave() {
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
      },
      onNextKey() {
        if (state.step < NUM_STEPS - 1) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey() {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene4 = function (root) { return buildScene(root); };
})();
