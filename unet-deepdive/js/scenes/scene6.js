/* Scene 6 — "How do you make a map bigger?"

   The viewer has just left the bottleneck (16×16). The decoder has to
   undo three pools and put the resolution back. *Before* we introduce
   the learned answer (transposed convolution, scene 7), this scene maps
   the design space of upsampling: nearest-neighbor, bilinear, and a
   black-box preview of transposed conv.

   Layout:
     - Hero block (h1, italic subtitle, lede).
     - Input feature map row: a small 4×4 source map with a tiny
       selector to flip between the three precomputed inputs.
     - Three side-by-side output panels (8×8 each):
         · nearest-neighbor (animated 2×2 block expansion)
         · bilinear (the same animation, but the four-cell weighted
           shading is visible during the reveal)
         · transposed conv (a black box on first reveal; on the next
           step it opens to the "+" filter output, captioned "we'll open
           this in scene 7")
     - Comparison table (HTML table, theme-pure).
     - Step engine (5 steps): 0 = input map only; 1 = nearest reveal;
       2 = bilinear reveal; 3 = transposed-conv black box (then output);
       4 = comparison table.
     - Caption + step controls.

   `&run` auto-advances 0 → 4 over ~4 s.  */
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 800;

  // Sizes (logical CSS px).
  const INPUT_PX = 200;     // 4×4 input panel
  const OUTPUT_PX = 240;    // 8×8 output panels
  const SWATCH_PX = 56;     // input-selector thumbnails
  const ANIM_MS_PER_CELL = 40; // animation pacing for the reveal sweep

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

  /* Deep-clone a 2D array. */
  function clone2D(a) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i].slice();
    return out;
  }

  /* Symmetric value range across all panels for a single input. We use a
     shared range so the three outputs are colored on the same scale --
     comparing intensities is then meaningful. */
  function sharedRange(arrays) {
    let m = 0;
    for (const a of arrays) {
      for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < a[i].length; j++) {
          const v = Math.abs(a[i][j]);
          if (v > m) m = v;
        }
      }
    }
    if (!m) m = 1;
    return [-m, m];
  }

  /* Whether any output cell is negative -- if not we can use a sequential
     ramp (visually less noisy for the all-positive nearest/bilinear case).
     For consistency with transposed-conv ("edge" case can be negative) we
     just always go diverging when *any* panel has negatives. */
  function anyNegative(arrays) {
    for (const a of arrays) {
      for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < a[i].length; j++) {
          if (a[i][j] < 0) return true;
        }
      }
    }
    return false;
  }

  /* Paint a 2D array onto a host canvas with cell borders. The mask, if
     supplied, dims unrevealed cells (used during the per-step animation
     of nearest/bilinear/tconv reveal sweeps). */
  function paintMap(host, data, px, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();

    // Background fill so rounded corners / borders read cleanly.
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);

    const range = opts.valueRange || [-1, 1];
    const diverging = opts.diverging !== false;

    window.Drawing.drawGrid(ctx, data, 0, 0, px, px, {
      diverging: diverging,
      valueRange: range,
    });

    const rows = data.length;
    const cols = data[0].length;
    const cw = px / cols;
    const ch = px / rows;

    // Optional "unrevealed" mask: re-fill those cells with the bg color
    // and a faint dashed outline so the viewer reads them as "to come".
    if (opts.mask) {
      ctx.fillStyle = t.bg;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (!opts.mask[i][j]) {
            ctx.fillRect(j * cw, i * ch, Math.ceil(cw), Math.ceil(ch));
          }
        }
      }
      ctx.strokeStyle = t.rule;
      ctx.lineWidth = 0.6;
      ctx.setLineDash([2, 3]);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (!opts.mask[i][j]) {
            ctx.strokeRect(j * cw + 0.5, i * ch + 0.5, cw - 1, ch - 1);
          }
        }
      }
      ctx.setLineDash([]);
    }

    // Cell borders so each cell reads as a discrete sample.
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 0; i <= rows; i++) {
      const y = Math.round(i * ch) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(px, y);
      ctx.stroke();
    }
    for (let j = 0; j <= cols; j++) {
      const x = Math.round(j * cw) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, px);
      ctx.stroke();
    }

    // Highlight ring on the freshly-revealed cells. Draws a subtle
    // accent border on the cells where mask transitioned ON this frame.
    if (opts.highlight) {
      const accent = t.pos;
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.6;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (opts.highlight[i][j]) {
            ctx.strokeRect(j * cw + 1, i * ch + 1, cw - 2, ch - 2);
          }
        }
      }
    }
  }

  /* Paint the small "black box" placeholder for the transposed-conv panel
     before its reveal step. */
  function paintBlackBox(host, px, label) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();

    // Solid ink panel as the "black box".
    ctx.fillStyle = t.ink;
    ctx.fillRect(0, 0, px, px);

    // A subtle question mark, in the bg color so it reads against the ink.
    ctx.fillStyle = t.bg;
    ctx.font = '500 ' + Math.floor(px * 0.32) + 'px "Iowan Old Style", Palatino, Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('?', px / 2, px / 2 - px * 0.04);

    // Caption.
    ctx.font = 'italic 13px "Iowan Old Style", Palatino, Georgia, serif';
    ctx.fillStyle = t.bg;
    ctx.globalAlpha = 0.78;
    ctx.fillText(label || 'opens in scene 7', px / 2, px / 2 + px * 0.30);
    ctx.globalAlpha = 1;
  }

  /* ---------------------------------------------------------------------
     Step-driven reveal masks.

     We sweep the 8×8 output left-to-right, top-to-bottom, revealing one
     cell per ANIM_MS_PER_CELL ms. The mask returned by
     `revealMask(progress, H, W)` is an H×W array of 0/1 with `progress`
     cells revealed, where `progress` ∈ [0, H*W].
     --------------------------------------------------------------------- */

  function fullMask(H, W, value) {
    const m = new Array(H);
    for (let i = 0; i < H; i++) {
      m[i] = new Array(W).fill(value);
    }
    return m;
  }

  function revealMask(progress, H, W) {
    const m = fullMask(H, W, 0);
    let k = 0;
    for (let i = 0; i < H && k < progress; i++) {
      for (let j = 0; j < W && k < progress; j++) {
        m[i][j] = 1;
        k++;
      }
    }
    return m;
  }

  function freshMask(prevProgress, progress, H, W) {
    const m = fullMask(H, W, 0);
    let k = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (k >= prevProgress && k < progress) m[i][j] = 1;
        k++;
      }
    }
    return m;
  }

  /* ---------------------------------------------------------------------
     Builder
     --------------------------------------------------------------------- */

  function buildScene(root) {
    if (!window.DATA || !window.DATA.upsample) {
      root.innerHTML = '<p style="opacity:0.5">upsample data missing.</p>';
      return {};
    }
    const D = window.DATA.upsample;

    // Walk the data once and map keys to UI panels. The producer (B2)
    // emits inputs/outputs/filters as documented at the top of
    // upsample_demos.py; we adapt defensively in case the layout shifts.
    const inputNames = Object.keys(D.inputs || {});
    if (inputNames.length === 0) {
      root.innerHTML = '<p style="opacity:0.5">no upsample inputs found.</p>';
      return {};
    }
    // Pick a default input that is *not* the all-positive single-cell case
    // (the cross is the most visually interesting baseline). Fall back to
    // the first input if "small_cross" is absent.
    let defaultInput = inputNames.indexOf('small_cross');
    if (defaultInput < 0) defaultInput = 0;

    // Pick the transposed-conv filter to display. The "+" filter is the
    // requested one; fall back to whatever filter the data ships with.
    const filterNames = Object.keys(D.filters || {});
    let tconvFilter = 'plus';
    if (filterNames.indexOf(tconvFilter) < 0) tconvFilter = filterNames[0] || 'plus';

    root.innerHTML = '';
    root.classList.add('s6-root');
    const wrap = el('div', { class: 's6-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s6-hero' }, wrap);
    el('h1', { text: 'How do you make a map bigger?' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'The decoder has to undo three pools. There are three classical answers — and a fourth, learnable one waiting in the next scene.',
    }, hero);
    el('p', {
      class: 'lede',
      html: 'Take a tiny <em>4&times;4</em> feature map. Grow it to <em>8&times;8</em>. ' +
        'Three rules give three different answers. None of them are learned: ' +
        'they are <em>fixed</em> recipes. The next scene replaces the recipe ' +
        'with a parameter the network can train.',
    }, hero);

    /* ---- Input row -------------------------------------------------- */
    const inputRow = el('div', { class: 's6-input-row' }, wrap);

    const inputCol = el('div', { class: 's6-input-col' }, inputRow);
    el('div', { class: 's6-panel-label', text: 'input · 4×4' }, inputCol);
    const inputHost = el('div', {
      class: 'canvas-host s6-input-host',
      style: 'width:' + INPUT_PX + 'px;height:' + INPUT_PX + 'px;',
    }, inputCol);
    el('div', {
      class: 's6-panel-sub',
      text: 'pick an input pattern',
    }, inputCol);

    const swatchRow = el('div', { class: 's6-swatch-row' }, inputCol);
    const swatchHosts = [];
    for (let i = 0; i < inputNames.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's6-swatch-btn',
        'data-input-index': String(i),
        'aria-label': 'Select input ' + inputNames[i],
        title: inputNames[i].replace('_', ' '),
      }, swatchRow);
      const sHost = el('div', { class: 'canvas-host s6-swatch-host' }, btn);
      swatchHosts.push(sHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-input-index'), 10);
        if (Number.isFinite(idx)) {
          state.inputIdx = idx;
          state.animProgress = 0;
          state.prevAnimProgress = 0;
          stopAnimation();
          render();
          startAnimationForStep();
        }
      });
    }

    /* ---- Three output panels --------------------------------------- */
    const outRow = el('div', { class: 's6-output-row' }, wrap);

    function makeOutputPanel(parent, key, name, sub) {
      const col = el('div', { class: 's6-out-col s6-out-' + key }, parent);
      el('div', { class: 's6-panel-label', text: name }, col);
      const host = el('div', {
        class: 'canvas-host s6-out-host',
        style: 'width:' + OUTPUT_PX + 'px;height:' + OUTPUT_PX + 'px;',
      }, col);
      const subEl = el('div', { class: 's6-panel-sub', text: sub }, col);
      return { col: col, host: host, subEl: subEl };
    }

    const nearestPanel = makeOutputPanel(outRow, 'nearest',
      'nearest-neighbor',
      'each input cell becomes a 2×2 block of itself');
    const bilinearPanel = makeOutputPanel(outRow, 'bilinear',
      'bilinear',
      'weighted average of the four nearest known cells');
    const tconvPanel = makeOutputPanel(outRow, 'tconv',
      'transposed conv',
      'the operator the U-Net actually uses');

    /* ---- Comparison table ------------------------------------------ */
    const tableWrap = el('div', { class: 's6-table-wrap' }, wrap);
    const table = el('table', { class: 's6-table' }, tableWrap);
    const thead = el('thead', null, table);
    const headRow = el('tr', null, thead);
    el('th', { text: 'method' }, headRow);
    el('th', { text: 'learnable?' }, headRow);
    el('th', { text: 'smooth?' }, headRow);
    el('th', { text: 'sharp boundaries?' }, headRow);
    const tbody = el('tbody', null, table);

    function tableRow(method, learnable, smooth, sharp, learnableEm) {
      const tr = el('tr', null, tbody);
      el('td', { text: method, class: 's6-td-method' }, tr);
      const c1 = el('td', null, tr);
      const span = el('span', { text: learnable }, c1);
      if (learnableEm) span.className = 's6-em';
      el('td', { text: smooth }, tr);
      el('td', { text: sharp }, tr);
      return tr;
    }

    tableRow('nearest-neighbor', 'no', 'no', 'yes (blocky)');
    tableRow('bilinear', 'no', 'yes', 'no (blurry)');
    tableRow('transposed conv', 'yes', 'depends on filter', 'depends on filter', true);

    /* ---- Stamp animation: tying scene 7 back to upsampling ---------- */
    /* User feedback: "I cannot infer how the actual transposed conv from scene 7
       maps to this 4×4 → 8×8 upsample case." So we replay scene 7's stamp
       mechanic, but with the *upsample* parameters: 4×4 input, stride 2, the
       same 3×3 filter scene 6 already uses. Stamps land at (i*2, j*2) into a
       9×9 raw output; the U-Net trims to 8×8 by dropping the last row/col.
       The mechanism (each input cell stamps a scaled copy of the kernel) is
       identical to scene 7. */
    const stampSec = el('section', { class: 's6-stamp-section' }, wrap);
    el('div', { class: 's6-stamp-eyebrow', text: 'how the transposed-conv panel was actually computed' }, stampSec);
    el('p', {
      class: 's6-stamp-intro',
      html:
        'Same stamp mechanic as scene 7 (the centerpiece you just saw), but applied ' +
        'to <em>upsampling</em>: input 4×4, stride <strong>2</strong>, kernel 3×3. ' +
        'Each of the 16 input cells stamps the kernel × its value into the output, ' +
        'placed at <code>(i·2, j·2)</code>. With stride 2 the stamps land further ' +
        'apart than in scene 7; some cells overlap, others don\'t.',
    }, stampSec);

    const stampPanel = el('div', { class: 's6-stamp-panel' }, stampSec);
    const stampInputCol = el('div', { class: 's6-stamp-col' }, stampPanel);
    el('div', { class: 's6-stamp-cap', text: 'input · 4 × 4' }, stampInputCol);
    const stampInputHost = el('div', { class: 'canvas-host s6-stamp-host' }, stampInputCol);
    el('div', { class: 's6-stamp-op', text: '×' }, stampPanel);
    const stampKernelCol = el('div', { class: 's6-stamp-col' }, stampPanel);
    const stampKernelCap = el('div', { class: 's6-stamp-cap', text: 'kernel · 3 × 3' }, stampKernelCol);
    const stampKernelHost = el('div', { class: 'canvas-host s6-stamp-host' }, stampKernelCol);
    el('div', { class: 's6-stamp-op', text: '=' }, stampPanel);
    const stampScaledCol = el('div', { class: 's6-stamp-col' }, stampPanel);
    const stampScaledCap = el('div', { class: 's6-stamp-cap', text: 'stamp = X × K' }, stampScaledCol);
    const stampScaledHost = el('div', { class: 'canvas-host s6-stamp-host' }, stampScaledCol);
    el('div', { class: 's6-stamp-op s6-stamp-op-arrow', text: '→' }, stampPanel);

    const stampOutRow = el('div', { class: 's6-stamp-out-row' }, stampSec);
    const stampOutCol = el('div', { class: 's6-stamp-col s6-stamp-col-out' }, stampOutRow);
    el('div', { class: 's6-stamp-cap', text: 'output · 9 × 9 (raw, before crop)' }, stampOutCol);
    const stampOutHost = el('div', { class: 'canvas-host s6-stamp-out-host' }, stampOutCol);
    el('div', {
      class: 's6-stamp-sub',
      text: 'each stamp lands at (i·2, j·2); overlaps sum',
    }, stampOutCol);

    const stampNarr = el('div', { class: 's6-stamp-narr' }, stampSec);
    const stampNarrTitle = el('div', { class: 's6-stamp-narr-title', text: 'press → stamp to begin' }, stampNarr);
    const stampNarrBody  = el('div', { class: 's6-stamp-narr-body' }, stampNarr);

    const stampCtrls = el('div', { class: 's6-stamp-controls' }, stampSec);
    el('label', { class: 's6-stamp-ctrl-label', text: 'stamps placed' }, stampCtrls);
    const stampSlider = el('input', {
      type: 'range', min: '0', max: '16', step: '1', value: '0',
      class: 's6-stamp-slider',
    }, stampCtrls);
    const stampOut = el('output', { class: 's6-stamp-ctrl-value', text: '0 / 16' }, stampCtrls);
    const stampPrevBtn = el('button', { type: 'button', class: 's6-btn', text: 'prev' }, stampCtrls);
    const stampNextBtn = el('button', { type: 'button', class: 's6-btn s6-btn-primary', text: 'first stamp →' }, stampCtrls);
    const stampPlayBtn = el('button', { type: 'button', class: 's6-btn', text: 'play full ▶' }, stampCtrls);
    const stampResetBtn = el('button', { type: 'button', class: 's6-btn', text: 'reset' }, stampCtrls);

    el('p', {
      class: 's6-stamp-cropnote',
      html:
        '<strong>About the size:</strong> the raw stamp output is 9×9. ' +
        'The U-Net trims a 1-pixel border to get exactly 8×8 (matching the ' +
        'encoder\'s spatial shape at this level). The trimming is a shape-' +
        'alignment detail; the stamp mechanic is what matters.',
    }, stampSec);

    /* ---- Caption ---------------------------------------------------- */
    const caption = el('p', { class: 'caption s6-caption' }, wrap);

    /* ---- Step controls --------------------------------------------- */
    const controls = el('div', { class: 'controls s6-controls' }, wrap);

    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's6-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's6-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', {
      class: 'control-value',
      text: '0 / ' + (NUM_STEPS - 1),
    }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ------------------------------------------------------ */
    const state = {
      step: 0,
      inputIdx: defaultInput,
      animProgress: 0,         // 0..64 across the 8×8 sweep
      prevAnimProgress: 0,
    };

    let animTimer = null;

    function stopAnimation() {
      if (animTimer) {
        clearInterval(animTimer);
        animTimer = null;
      }
    }

    /* Step-aware animation: nearest (step 1) and bilinear (step 2) sweep
       cell-by-cell. Transposed-conv reveal at step 3 is instantaneous (the
       black box opens in one beat, since the *operator* is what the viewer
       should look at, not the per-cell sweep). */
    function startAnimationForStep() {
      stopAnimation();
      const total = 64; // 8×8
      if (state.step === 1 || state.step === 2) {
        state.animProgress = 0;
        state.prevAnimProgress = 0;
        animTimer = setInterval(function () {
          state.prevAnimProgress = state.animProgress;
          state.animProgress = Math.min(total, state.animProgress + 4);
          if (state.animProgress >= total) {
            stopAnimation();
            // Settle: clear "fresh" highlight after a beat.
            setTimeout(function () {
              state.prevAnimProgress = state.animProgress;
              renderOutputs();
            }, 200);
          }
          renderOutputs();
        }, ANIM_MS_PER_CELL);
      } else {
        // For steps 3 / 4 the outputs render fully without animation.
        state.animProgress = total;
        state.prevAnimProgress = total;
      }
    }

    /* Captions per step. */
    function captionFor(step) {
      switch (step) {
        case 0:
          return 'A 4×4 feature map. Each cell carries a single value. ' +
                 'We want an 8×8 version of it — same content, four times the cells.';
        case 1:
          return 'Nearest-neighbor. Each input cell is copied into the four cells of its 2×2 block. ' +
                 'No new information; the result is blocky.';
        case 2:
          return 'Bilinear. Each output cell is a weighted average of the four nearest input cells. ' +
                 'Smooth, but boundaries are blurred.';
        case 3:
          return 'Transposed convolution. A black box for now: it produces an 8×8 output, ' +
                 'but the rule is *learned*. Step 4 of this scene shows what *one* hand-picked ' +
                 'filter does; scene 7 opens the operator itself.';
        case 4:
          return 'Three operators. Nearest and bilinear have zero parameters. Transposed conv has ' +
                 'a 3×3 filter that the optimizer trains. That is the U-Net upsample.';
        default:
          return '';
      }
    }

    /* ---- Renderers -------------------------------------------------- */

    function currentInputName() {
      return inputNames[state.inputIdx];
    }

    function currentOutputs() {
      const name = currentInputName();
      const out = D.outputs[name] || {};
      const tconvSet = out.tconv || {};
      const tconv = tconvSet[tconvFilter] ||
        tconvSet[Object.keys(tconvSet)[0]] ||
        fullMask(8, 8, 0);
      return {
        nearest: out.nearest || fullMask(8, 8, 0),
        bilinear: out.bilinear || fullMask(8, 8, 0),
        tconv: tconv,
      };
    }

    function paintInput() {
      const name = currentInputName();
      const data = D.inputs[name];
      // Inputs are non-negative; render with the diverging palette so the
      // baseline (=0) stays the neutral cream and bright cells read as
      // strongly positive. Symmetric range with the bilinear/nearest
      // outputs so colors are comparable across panels.
      const outs = currentOutputs();
      const range = sharedRange([data, outs.nearest, outs.bilinear, outs.tconv]);
      paintMap(inputHost, data, INPUT_PX, {
        diverging: anyNegative([data, outs.nearest, outs.bilinear, outs.tconv]),
        valueRange: range,
      });
    }

    function paintSwatches() {
      // One small thumbnail per input; the active one gets a class.
      for (let i = 0; i < inputNames.length; i++) {
        const data = D.inputs[inputNames[i]];
        const range = sharedRange([data]);
        paintMap(swatchHosts[i], data, SWATCH_PX, {
          diverging: anyNegative([data]),
          valueRange: range,
        });
      }
      const btns = swatchRow.querySelectorAll('.s6-swatch-btn');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.inputIdx);
      });
    }

    function renderOutputs() {
      const outs = currentOutputs();
      const range = sharedRange([
        D.inputs[currentInputName()],
        outs.nearest, outs.bilinear, outs.tconv,
      ]);
      const diverging = anyNegative([outs.nearest, outs.bilinear, outs.tconv]);

      // Step 1: nearest reveal animation (or fully revealed if step >= 2).
      if (state.step === 0) {
        // Pre-reveal: blank placeholders so the row reads as "to be filled".
        paintMap(nearestPanel.host, fullMask(8, 8, 0), OUTPUT_PX, {
          diverging: false, valueRange: [0, 1], mask: fullMask(8, 8, 0),
        });
        paintMap(bilinearPanel.host, fullMask(8, 8, 0), OUTPUT_PX, {
          diverging: false, valueRange: [0, 1], mask: fullMask(8, 8, 0),
        });
        paintBlackBox(tconvPanel.host, OUTPUT_PX, 'opens in scene 7');
        return;
      }

      // Step 1: nearest reveal sweep (animated).
      if (state.step === 1) {
        const m = revealMask(state.animProgress, 8, 8);
        const fresh = freshMask(state.prevAnimProgress, state.animProgress, 8, 8);
        paintMap(nearestPanel.host, outs.nearest, OUTPUT_PX, {
          diverging: diverging, valueRange: range,
          mask: m, highlight: fresh,
        });
        // Bilinear and tconv stay placeholder.
        paintMap(bilinearPanel.host, fullMask(8, 8, 0), OUTPUT_PX, {
          diverging: false, valueRange: [0, 1], mask: fullMask(8, 8, 0),
        });
        paintBlackBox(tconvPanel.host, OUTPUT_PX, 'opens in scene 7');
        return;
      }

      // Step 2: nearest fully revealed; bilinear sweep.
      if (state.step === 2) {
        paintMap(nearestPanel.host, outs.nearest, OUTPUT_PX, {
          diverging: diverging, valueRange: range,
        });
        const m = revealMask(state.animProgress, 8, 8);
        const fresh = freshMask(state.prevAnimProgress, state.animProgress, 8, 8);
        paintMap(bilinearPanel.host, outs.bilinear, OUTPUT_PX, {
          diverging: diverging, valueRange: range,
          mask: m, highlight: fresh,
        });
        paintBlackBox(tconvPanel.host, OUTPUT_PX, 'opens in scene 7');
        return;
      }

      // Step 3: nearest + bilinear fully revealed; tconv is still the
      // black box (the *step* introducing it).
      if (state.step === 3) {
        paintMap(nearestPanel.host, outs.nearest, OUTPUT_PX, {
          diverging: diverging, valueRange: range,
        });
        paintMap(bilinearPanel.host, outs.bilinear, OUTPUT_PX, {
          diverging: diverging, valueRange: range,
        });
        paintBlackBox(tconvPanel.host, OUTPUT_PX, 'we’ll open this in scene 7');
        return;
      }

      // Step 4: transposed conv output revealed too.
      paintMap(nearestPanel.host, outs.nearest, OUTPUT_PX, {
        diverging: diverging, valueRange: range,
      });
      paintMap(bilinearPanel.host, outs.bilinear, OUTPUT_PX, {
        diverging: diverging, valueRange: range,
      });
      paintMap(tconvPanel.host, outs.tconv, OUTPUT_PX, {
        diverging: diverging, valueRange: range,
      });
    }

    function render() {
      paintInput();
      paintSwatches();
      renderOutputs();
      if (typeof renderStampSection === 'function') renderStampSection();

      // Caption + control widgets.
      caption.textContent = captionFor(state.step);
      stepInput.value = String(state.step);
      stepOut.textContent = state.step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = state.step <= 0;
      nextBtn.disabled = state.step >= NUM_STEPS - 1;

      // Reveal table only on step 4.
      tableWrap.classList.toggle('s6-visible', state.step >= 4);

      // Per-panel "lit" classes so CSS can dim non-active panels.
      nearestPanel.col.classList.toggle('s6-revealed', state.step >= 1);
      bilinearPanel.col.classList.toggle('s6-revealed', state.step >= 2);
      tconvPanel.col.classList.toggle('s6-revealed', state.step >= 3);
      tconvPanel.col.classList.toggle('s6-tconv-open', state.step >= 4);

      // Active step highlights for the panel the user is currently focused on.
      nearestPanel.col.classList.toggle('s6-active', state.step === 1);
      bilinearPanel.col.classList.toggle('s6-active', state.step === 2);
      tconvPanel.col.classList.toggle('s6-active',
        state.step === 3 || state.step === 4);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      // Re-trigger animation if the new step is one of the animated ones.
      stopAnimation();
      // Reset animation progress; settled steps render full immediately.
      if (state.step === 1 || state.step === 2) {
        state.animProgress = 0;
        state.prevAnimProgress = 0;
      } else {
        state.animProgress = 64;
        state.prevAnimProgress = 64;
      }
      render();
      startAnimationForStep();
    }

    /* ---- Wire controls --------------------------------------------- */
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    /* ---- Stamp animation: state, helpers, render -------------------- */
    /* Independent state machine for the "see the stamps in action" panel
       at the bottom. 16 input cells (4×4) → up to 16 stamps into a 9×9
       output (raw transposed conv, stride 2, 3×3 kernel). */
    const STAMP_RUN_MS = 700;
    const STAMP_INPUT_PX = 168;
    const STAMP_KERNEL_PX = 168;
    const STAMP_OUT_CELL = 36;       // 9 cells × 36 px = 324 px
    const stampState = { step: 0 };
    let stampTimer = null;
    function stopStampTimer() {
      if (stampTimer) { clearTimeout(stampTimer); stampTimer = null; }
    }

    function currentInput4x4() {
      const name = currentInputName();
      const inp = D.inputs[name];
      return inp || fullMask(4, 4, 0);
    }
    function currentTconvKernel() {
      const k = D.filters && D.filters[tconvFilter];
      return k || [[0,0.2,0],[0.2,0.2,0.2],[0,0.2,0]]; // safe fallback
    }

    function buildStampTrace2(input, kernel, stride) {
      const N = input.length;
      const K = kernel.length;
      const O = (N - 1) * stride + K;            // raw output side
      const canvas = [];
      for (let i = 0; i < O; i++) {
        const row = []; for (let j = 0; j < O; j++) row.push(0);
        canvas.push(row);
      }
      const frames = [canvas.map(r => r.slice())];
      const meta = [];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const v = input[i][j];
          const stamp = [];
          for (let u = 0; u < K; u++) {
            const sr = []; for (let w = 0; w < K; w++) sr.push(v * kernel[u][w]);
            stamp.push(sr);
          }
          for (let u = 0; u < K; u++) {
            for (let w = 0; w < K; w++) {
              canvas[i * stride + u][j * stride + w] += v * kernel[u][w];
            }
          }
          frames.push(canvas.map(r => r.slice()));
          meta.push({
            k: meta.length + 1,
            inputCell: [i, j],
            inputValue: v,
            stamp: stamp,
            targetR: [i * stride, i * stride + K - 1],
            targetC: [j * stride, j * stride + K - 1],
          });
        }
      }
      return { frames, meta, O };
    }

    function paintStampGrid(host, data, opts) {
      host.innerHTML = '';
      opts = opts || {};
      const rows = data.length, cols = data[0].length;
      const cell = opts.cell || Math.floor((opts.px || 168) / Math.max(rows, cols));
      const W = cell * cols, H = cell * rows;
      const setup = window.Drawing.setupCanvas(host, W, H);
      const ctx = setup.ctx;
      const t = window.Drawing.tokens();
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, W, H);
      let m = 0;
      for (const row of data) for (const v of row) m = Math.max(m, Math.abs(v));
      if (!m) m = 1;
      if (opts.vmax) m = opts.vmax;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          ctx.fillStyle = window.Drawing.divergingColor(data[i][j] / m, t);
          ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
      ctx.strokeStyle = t.rule; ctx.lineWidth = 1;
      for (let i = 0; i <= rows; i++) {
        ctx.beginPath(); ctx.moveTo(0, i*cell); ctx.lineTo(W, i*cell); ctx.stroke();
      }
      for (let j = 0; j <= cols; j++) {
        ctx.beginPath(); ctx.moveTo(j*cell, 0); ctx.lineTo(j*cell, H); ctx.stroke();
      }
      if (opts.labels !== false) {
        ctx.font = `${Math.max(10, Math.floor(cell * 0.34))}px "SF Mono", Menlo, monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const v = data[i][j];
            const intensity = Math.min(1, Math.abs(v) / m);
            ctx.fillStyle = intensity > 0.55 ? t.bg : t.ink;
            const s = Math.abs(v) < 1e-9 ? '0'
                      : (Number.isInteger(v) ? String(v) : v.toFixed(2));
            ctx.fillText(s, (j + 0.5) * cell, (i + 0.5) * cell);
          }
        }
      }
      if (opts.highlight) {
        const h = opts.highlight;
        ctx.lineWidth = 3; ctx.strokeStyle = '#d97a1f';
        ctx.strokeRect(h.col*cell + 1.5, h.row*cell + 1.5,
          (h.cols||1)*cell - 3, (h.rows||1)*cell - 3);
      }
      if (opts.targetBox) {
        const b = opts.targetBox;
        ctx.lineWidth = 2.5; ctx.strokeStyle = '#d97a1f';
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(b.col*cell + 1.25, b.row*cell + 1.25,
          (b.cols||1)*cell - 2.5, (b.rows||1)*cell - 2.5);
        ctx.setLineDash([]);
      }
    }

    let lastStampInputIdx = state.inputIdx;
    function renderStampSection() {
      // If the input swatch has just changed, reset the stamp animation
      // so the viewer can replay it on the new input.
      if (state.inputIdx !== lastStampInputIdx) {
        stopStampTimer();
        stampState.step = 0;
        lastStampInputIdx = state.inputIdx;
      }
      const inp = currentInput4x4();
      const ker = currentTconvKernel();
      const trace = buildStampTrace2(inp, ker, 2);
      const O = trace.O;
      const total = inp.length * inp[0].length;        // 16
      stampState.step = Math.max(0, Math.min(total, stampState.step));

      // Input panel: highlight active cell at step >= 1
      let inHL = null;
      if (stampState.step >= 1 && stampState.step <= total) {
        const m = trace.meta[stampState.step - 1];
        inHL = { row: m.inputCell[0], col: m.inputCell[1] };
      }
      paintStampGrid(stampInputHost, inp, { px: STAMP_INPUT_PX, vmax: 1, highlight: inHL });
      paintStampGrid(stampKernelHost, ker, { px: STAMP_KERNEL_PX, vmax: 0.25 });

      // Scaled stamp panel
      if (stampState.step >= 1 && stampState.step <= total) {
        const m = trace.meta[stampState.step - 1];
        if (m.inputValue === 0) {
          stampScaledCap.textContent = 'stamp = 0 × K = 0';
          paintStampGrid(stampScaledHost, [[0,0,0],[0,0,0],[0,0,0]], { px: STAMP_KERNEL_PX, vmax: 1 });
        } else {
          stampScaledCap.textContent = 'stamp = ' + m.inputValue + ' × K';
          paintStampGrid(stampScaledHost, m.stamp, { px: STAMP_KERNEL_PX, vmax: Math.max(0.25, Math.abs(m.inputValue) * 0.25) });
        }
      } else {
        stampScaledCap.textContent = 'stamp = X × K';
        paintStampGrid(stampScaledHost, [[0,0,0],[0,0,0],[0,0,0]], { px: STAMP_KERNEL_PX, vmax: 1 });
      }

      // Output canvas
      const data = trace.frames[stampState.step];
      const opts = { cell: STAMP_OUT_CELL };
      if (stampState.step >= 1 && stampState.step <= total) {
        const m = trace.meta[stampState.step - 1];
        opts.targetBox = {
          row: m.targetR[0], col: m.targetC[0], rows: 3, cols: 3,
        };
      }
      // Symmetric range across the entire final frame so colors are stable
      let mFinal = 0;
      const finalGrid = trace.frames[trace.frames.length - 1];
      for (const r of finalGrid) for (const v of r) mFinal = Math.max(mFinal, Math.abs(v));
      opts.vmax = Math.max(0.25, mFinal);
      paintStampGrid(stampOutHost, data, opts);
      stampOutHost.style.width = (STAMP_OUT_CELL * O) + 'px';
      stampOutHost.style.height = (STAMP_OUT_CELL * O) + 'px';

      // Narration
      if (stampState.step === 0) {
        stampNarrTitle.textContent = 'press → stamp to begin';
        stampNarrBody.innerHTML = '';
      } else if (stampState.step > total) {
        stampNarrTitle.textContent = 'all 16 stamps placed';
        stampNarrBody.innerHTML = '';
      } else {
        const m = trace.meta[stampState.step - 1];
        const [i, j] = m.inputCell;
        const tr = m.targetR, tc = m.targetC;
        stampNarrTitle.textContent = 'Stamp ' + m.k + ' of 16 — input cell (' + i + ', ' + j + ')';
        stampNarrBody.innerHTML =
          '<div class="s6-stamp-n-row"><span class="s6-stamp-n-k">value</span>' +
          '<span class="s6-stamp-n-v">X[' + i + '][' + j + '] = ' + m.inputValue + '</span></div>' +
          '<div class="s6-stamp-n-row"><span class="s6-stamp-n-k">stamp</span>' +
          '<span class="s6-stamp-n-v">' + (m.inputValue === 0 ? '0 × K = all zeros' : (m.inputValue + ' × K')) + '</span></div>' +
          '<div class="s6-stamp-n-row"><span class="s6-stamp-n-k">target</span>' +
          '<span class="s6-stamp-n-v">rows ' + tr[0] + '–' + tr[1] + ', cols ' + tc[0] + '–' + tc[1] +
          ' &nbsp;(stride 2, so the next input row\'s stamps start 2 rows lower)</span></div>';
      }

      // Controls
      stampSlider.max = String(total);
      stampSlider.value = String(stampState.step);
      stampOut.textContent = stampState.step + ' / ' + total;
      stampPrevBtn.disabled = stampState.step <= 0;
      stampNextBtn.disabled = stampState.step >= total;
      stampResetBtn.disabled = stampState.step === 0;
      stampPlayBtn.disabled = stampState.step >= total;
      stampNextBtn.textContent = stampState.step === 0 ? 'first stamp →' : 'next stamp →';
    }

    function applyStampStep(n) {
      stampState.step = Math.max(0, Math.min(16, n));
      renderStampSection();
    }
    function autoStamp() {
      if (stampState.step >= 16) { stopStampTimer(); return; }
      applyStampStep(stampState.step + 1);
      stampTimer = setTimeout(autoStamp, STAMP_RUN_MS);
    }
    stampPrevBtn.addEventListener('click', function () { stopStampTimer(); applyStampStep(stampState.step - 1); });
    stampNextBtn.addEventListener('click', function () { stopStampTimer(); applyStampStep(stampState.step + 1); });
    stampResetBtn.addEventListener('click', function () { stopStampTimer(); applyStampStep(0); });
    stampPlayBtn.addEventListener('click', function () {
      stopStampTimer();
      applyStampStep(0);
      stampTimer = setTimeout(autoStamp, 400);
    });
    stampSlider.addEventListener('input', function () {
      stopStampTimer();
      const v = parseInt(stampSlider.value, 10);
      if (Number.isFinite(v)) applyStampStep(v);
    });

    /* If the user changes the input pattern in the main panel, restart
       the stamp animation so it reflects the new input. */
    const origStartAnimationForStep = startAnimationForStep;
    // Wrap startAnimationForStep to also reset the stamp state when the
    // input changes via the swatch buttons.
    // (We can't easily detect input changes; instead we re-render the stamp
    //  section every time the main render() runs, picking up the live input.)

    /* ---- Initial paint --------------------------------------------- */
    render();
    renderStampSection();

    /* &run -> auto-advance to the final step over ~4s. */
    let runTimer = null;
    function autoAdvance(target) {
      if (state.step >= target) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(function () { autoAdvance(target); }, RUN_INTERVAL_MS);
    }

    if (readHashFlag('run')) {
      runTimer = setTimeout(function () { autoAdvance(NUM_STEPS - 1); }, 200);
    }

    return {
      onEnter: function () { render(); },
      onLeave: function () {
        stopAnimation();
        stopStampTimer();
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
      },
      onNextKey: function () {
        if (state.step < NUM_STEPS - 1) {
          applyStep(state.step + 1);
          return true;
        }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) {
          applyStep(state.step - 1);
          return true;
        }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene6 = function (root) { return buildScene(root); };
})();
