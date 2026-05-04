/* Scene 7 — Transposed convolution: filter slides over a sparse, zero-inserted input.

   Canonical Dumoulin/Visin visualization. The input is placed sparsely on a
   strided grid (zeros between input cells when s > 1). The k×k filter slides
   at stride 1 over that grid; whenever its centre cell sits over a real input
   value v, the filter is multiplied by v and stamped into the output at the
   filter's current position.

   Step engine — Option A: each step deposits ONE stamp. We visit the N*N input
   cells in row-major order (input cell (i, j) corresponds to filter position
   (i*s, j*s) on the strided input). This is the right default because the
   animation length stays at N*N = 9 regardless of stride — at stride 4 a full
   stride-1 sweep over the strided input would be 49 positions, too long.

   Knobs:
     stride s ∈ {1, 2, 3, 4}     — output side length = (N-1)*s + K
     filter K ∈ {plus, edge, blur} — fixed 3×3 presets
*/
(function () {
  'use strict';

  // ---- DOM helper -------------------------------------------------------
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

  // ---- Constants --------------------------------------------------------
  const N_INPUT = 3;
  const K_FILT = 3;
  function outDim(s) { return (N_INPUT - 1) * s + K_FILT; }
  function stridedDim(s) { return (N_INPUT - 1) * s + 1; }
  // Padding around the strided input so the filter slides at stride 1 cleanly.
  // For an odd k, pad by (k-1)/2 on each side; the padded grid then has the
  // same extent as the output and the filter centre aligns to real input cells.
  const PAD = (K_FILT - 1) / 2;       // = 1 for k = 3
  function paddedStridedDim(s) { return stridedDim(s) + 2 * PAD; }   // = outDim(s)

  // The default input — several non-zeros plus one zero so the viewer sees
  // the "no contribution" beat (input value = 0 → all-zero stamp).
  const DEFAULT_INPUT = [
    [1, 0, 2],
    [1, 3, 0],
    [0, 2, 1],
  ];

  const FILTER_PRESETS = {
    'plus':     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    'edge':     [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    'gaussian': [[1, 2, 1], [2, 2, 2], [1, 2, 1]],
  };
  const FILTER_LABELS = { plus: 'plus  ＋', edge: 'edge ↓', gaussian: 'blur' };

  // Pixel sizes. Output 11×11 at 50 px = 550 px wide (stride 4) — fits 1500px.
  const STRIDED_CELL = 50;
  const OUT_CELL     = 50;
  const RUN_STEP_MS  = 950;

  // ---- Numerical utilities ---------------------------------------------
  function zeros2D(h, w) {
    const out = new Array(h);
    for (let i = 0; i < h; i++) {
      const row = new Array(w);
      for (let j = 0; j < w; j++) row[j] = 0;
      out[i] = row;
    }
    return out;
  }
  function cloneGrid(g) {
    const out = new Array(g.length);
    for (let i = 0; i < g.length; i++) out[i] = g[i].slice();
    return out;
  }
  function maxAbs(g) {
    let m = 0;
    for (const row of g) for (const v of row) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
    return m || 1;
  }

  // Build the strided + padded input grid.
  // The dense input is placed at positions (PAD + i*s, PAD + j*s); everything
  // else is zero. The result has the same dimensions as the output, so the
  // filter can slide at stride 1 with its top-left at output coords (i*s, j*s)
  // and its centre at the real input value (PAD + i*s, PAD + j*s).
  function makeStrided(input, s) {
    const N = input.length;
    const D = paddedStridedDim(s);
    const out = zeros2D(D, D);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        out[PAD + i * s][PAD + j * s] = input[i][j];
      }
    }
    return out;
  }

  /* Build the per-stamp trace.
     Each stamp position k corresponds to dense-input cell (i, j) in row-major
     order. The filter's centre cell sits at (i*s, j*s) on the strided input
     (i.e. exactly on the real input value), and the filter covers the k×k
     region [i*s..i*s+K-1, j*s..j*s+K-1] on the output. */
  function buildStampTrace(input, filter, stride) {
    const N = input.length;
    const K = filter.length;
    const O = (N - 1) * stride + K;
    const canvas = zeros2D(O, O);
    const frames = [cloneGrid(canvas)];
    const meta = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = input[i][j];
        for (let u = 0; u < K; u++) {
          for (let w = 0; w < K; w++) {
            canvas[i * stride + u][j * stride + w] += v * filter[u][w];
          }
        }
        frames.push(cloneGrid(canvas));
        meta.push({
          k: meta.length + 1,                           // 1-based step index
          inputCell: [i, j],
          inputValue: v,
          stamp: filter.map(function (r) { return r.map(function (x) { return x * v; }); }),
          // Filter top-left on the strided input (in strided-input cell coords)
          filterTL: [i * stride, j * stride],
          // Filter centre on the strided input (centre cell of the k×k filter)
          filterCenter: [i * stride + 1, j * stride + 1],
          // Region covered on the output canvas
          targetR: [i * stride, i * stride + K - 1],
          targetC: [j * stride, j * stride + K - 1],
        });
      }
    }
    return { frames, meta };
  }

  // ---- Painter for an arbitrary value grid ------------------------------
  /* Paints data into host as a canvas. Supports overlays:
       opts.cell        : px per cell
       opts.vmax        : color-norm reference (defaults to maxAbs(data))
       opts.dimNonZero  : if true, dim the cells that are NOT in zeroMask
       opts.zeroMask    : 2D boolean array; cells where mask[i][j] is true are
                          drawn near-neutral with no value label (used to show
                          "this is a fake zero between input cells")
       opts.filterOverlay : { tl: [r,c], filter: [[]], color, valueAlpha,
                              centreCell: [r,c] }
                            draws translucent k×k filter values on top, with a
                            ring on the centre cell.
       opts.targetBox   : { row, col, rows, cols, color, dashed } box on output
  */
  function paintGrid(host, data, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const rows = data.length, cols = data[0].length;
    const cell = opts.cell || 40;
    const W = cell * cols;
    const H = cell * rows;
    const setup = window.Drawing.setupCanvas(host, W, H);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();

    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, W, H);

    const m = (opts.vmax != null) ? opts.vmax : Math.max(1, maxAbs(data));

    // Cell fills
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = data[i][j];
        const isZeroMaskCell = opts.zeroMask && opts.zeroMask[i][j];
        if (isZeroMaskCell) {
          // Subtle hatched / striped background to read as "inserted zero"
          ctx.fillStyle = t.bg;
          ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
          ctx.save();
          ctx.globalAlpha = 0.12;
          ctx.strokeStyle = t.inkSecondary || t.ink;
          ctx.lineWidth = 1;
          const step = 6;
          ctx.beginPath();
          for (let dx = -cell; dx < cell; dx += step) {
            ctx.moveTo(j * cell + dx, i * cell);
            ctx.lineTo(j * cell + dx + cell, i * cell + cell);
          }
          ctx.stroke();
          ctx.restore();
        } else {
          ctx.fillStyle = window.Drawing.divergingColor(v / m, t);
          ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    }

    // Grid lines
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(W, i * cell);
      ctx.stroke();
    }
    for (let j = 0; j <= cols; j++) {
      ctx.beginPath();
      ctx.moveTo(j * cell, 0);
      ctx.lineTo(j * cell, H);
      ctx.stroke();
    }

    // Value labels (skip on zero-mask cells)
    if (opts.labels !== false) {
      ctx.font = `${Math.max(11, Math.floor(cell * 0.32))}px "SF Mono", Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (opts.zeroMask && opts.zeroMask[i][j]) continue;
          const v = data[i][j];
          const intensity = Math.min(1, Math.abs(v) / m);
          ctx.fillStyle = intensity > 0.55 ? t.bg : t.ink;
          const s = Math.abs(v) < 1e-9 ? '0'
                    : (Number.isInteger(v) ? String(v) : v.toFixed(1));
          ctx.fillText(s, (j + 0.5) * cell, (i + 0.5) * cell);
        }
      }
    }

    // Filter overlay (orange translucent k×k box) — draws filter values faintly
    if (opts.filterOverlay) {
      const fo = opts.filterOverlay;
      const accent = fo.color || '#d97a1f';
      const tl = fo.tl;
      const f = fo.filter;
      const kh = f.length, kw = f[0].length;
      // Translucent fill across the whole filter region (so user sees it as
      // a coherent box even where a stamp adds zero).
      ctx.save();
      ctx.globalAlpha = 0.10;
      ctx.fillStyle = accent;
      ctx.fillRect(tl[1] * cell, tl[0] * cell, kw * cell, kh * cell);
      ctx.restore();

      // Per-cell faint filter values (small text, top-left of each cell)
      ctx.font = `${Math.max(10, Math.floor(cell * 0.22))}px "SF Mono", Menlo, monospace`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.save();
      ctx.globalAlpha = 0.85;
      for (let u = 0; u < kh; u++) {
        for (let w = 0; w < kw; w++) {
          const r = tl[0] + u, c = tl[1] + w;
          if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
          ctx.fillStyle = accent;
          const v = f[u][w];
          const s = Math.abs(v) < 1e-9 ? '0'
                    : (Number.isInteger(v) ? String(v) : v.toFixed(1));
          ctx.fillText(s, c * cell + 4, r * cell + 4);
        }
      }
      ctx.restore();

      // Outer rectangle outlining the filter
      ctx.strokeStyle = accent;
      ctx.lineWidth = 3;
      ctx.strokeRect(
        tl[1] * cell + 1.5,
        tl[0] * cell + 1.5,
        kw * cell - 3,
        kh * cell - 3
      );

      // Centre-cell ring
      if (fo.centreCell) {
        const cr = fo.centreCell[0], cc = fo.centreCell[1];
        ctx.strokeStyle = accent;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([]);
        const inset = 5;
        ctx.strokeRect(
          cc * cell + inset,
          cr * cell + inset,
          cell - 2 * inset,
          cell - 2 * inset
        );
        // small dot at very centre
        ctx.beginPath();
        ctx.fillStyle = accent;
        ctx.arc((cc + 0.5) * cell, (cr + 0.5) * cell, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Dashed target box on output
    if (opts.targetBox) {
      const b = opts.targetBox;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = b.color || '#d97a1f';
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(
        b.col * cell + 1.25,
        b.row * cell + 1.25,
        (b.cols || 1) * cell - 2.5,
        (b.rows || 1) * cell - 2.5
      );
      ctx.setLineDash([]);
    }

    return { W, H, cell, rows, cols };
  }

  // ---- Scene builder ---------------------------------------------------
  function buildScene(root) {
    if (!window.Drawing || !window.UNET) {
      root.innerHTML = '<p style="opacity:0.5">Scene 7: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s7-root');
    const wrap = el('div', { class: 's7-wrap' }, root);

    // ---- Hero --------------------------------------------------------
    const hero = el('header', { class: 'hero s7-hero' }, wrap);
    el('h1', { text: 'Transposed convolution: filter slides, input stamps.' }, hero);
    el('p', {
      class: 'subtitle',
      text:
        'Place the input on a sparse grid (one cell every s positions; zeros in between). ' +
        'Slide a k×k filter at stride 1 over that grid. Whenever the filter\'s centre ' +
        'cell sits on a real input value v, multiply the filter by v and stamp the ' +
        'result into the output at the filter\'s position. Where stamps overlap they sum.',
    }, hero);

    // ---- "You are here" mini-map ------------------------------------
    const miniHost = el('div', { class: 's7-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'transposed conv · used by up1 and up2 in the U-Net decoder',
      });
      mm.setHighlight(['up1', 'up2']);
    }

    // ---- Knobs (above cheat-sheet so user sees the controls first) ---
    const knobs = el('section', { class: 's7-knobs' }, wrap);

    // Stride
    const strideGroup = el('div', { class: 's7-knob' }, knobs);
    el('span', { class: 's7-knob-label', text: 'stride s' }, strideGroup);
    const strideBtns = {};
    [1, 2, 3, 4].forEach(function (s) {
      const b = el('button', {
        type: 'button',
        class: 's7-knob-btn' + (s === 2 ? ' s7-active' : ''),
        'data-stride': String(s),
        text: String(s),
      }, strideGroup);
      strideBtns[s] = b;
    });

    // Filter
    const filtGroup = el('div', { class: 's7-knob' }, knobs);
    el('span', { class: 's7-knob-label', text: 'filter K' }, filtGroup);
    const filtBtns = {};
    Object.keys(FILTER_PRESETS).forEach(function (k) {
      const b = el('button', {
        type: 'button',
        class: 's7-knob-btn' + (k === 'plus' ? ' s7-active' : ''),
        'data-filt': k,
        text: FILTER_LABELS[k],
      }, filtGroup);
      filtBtns[k] = b;
    });

    // ---- Cheat-sheet -------------------------------------------------
    const cheat = el('section', { class: 's7-cheat' }, wrap);
    function cheatItem(label, valueSpan) {
      const it = el('div', { class: 's7-cheat-item' }, cheat);
      el('span', { class: 's7-cheat-k', text: label }, it);
      it.appendChild(valueSpan);
      return it;
    }
    const cInputV   = el('span', { class: 's7-cheat-v', text: '3 × 3' });
    const cStridedV = el('span', { class: 's7-cheat-v', text: '5 × 5' });
    const cFiltV    = el('span', { class: 's7-cheat-v', text: '3 × 3 (plus)' });
    const cStrideV  = el('span', { class: 's7-cheat-v', text: '2' });
    const cOutV     = el('span', { class: 's7-cheat-v s7-cheat-v-em', text: '7 × 7' });
    cheatItem('input X', cInputV);
    cheatItem('strided X', cStridedV);
    cheatItem('filter K', cFiltV);
    cheatItem('stride s', cStrideV);
    cheatItem('output Y', cOutV);
    el('span', {
      class: 's7-cheat-formula',
      text: 'Y = (X − 1)·s + K',
    }, cheat);

    // ---- Main panel: strided input on top, output below --------------
    const panel = el('section', { class: 's7-panel' }, wrap);
    const panelTitle = el('div', { class: 's7-panel-title' }, panel);
    el('span', { class: 's7-panel-eyebrow', text: 'how the output is built' }, panelTitle);
    const panelProgress = el('span', { class: 's7-panel-progress', text: 'press → to begin' }, panelTitle);

    // STRIDED INPUT row
    const stridedRow = el('div', { class: 's7-row' }, panel);
    const stridedCol = el('div', { class: 's7-col' }, stridedRow);
    el('div', { class: 's7-col-cap', text: 'strided input  (X placed at every s-th cell; zeros in between; padded so the filter has room to slide)' }, stridedCol);
    const stridedHost = el('div', { class: 'canvas-host s7-canvas-host' }, stridedCol);
    const stridedSub = el('div', {
      class: 's7-col-sub',
      text: 'orange box = current filter position · ring marks the filter\'s centre cell',
    }, stridedCol);

    // Arrow between the two
    el('div', { class: 's7-down-arrow', text: '↓ stamp at filter position' }, panel);

    // OUTPUT row
    const outRow = el('div', { class: 's7-row' }, panel);
    const outCol = el('div', { class: 's7-col' }, outRow);
    el('div', { class: 's7-col-cap', text: 'output Y  (running sum of stamps)' }, outCol);
    const outHost = el('div', { class: 'canvas-host s7-canvas-host s7-out-host' }, outCol);
    el('div', {
      class: 's7-col-sub',
      text: 'each stamp = (input value at centre) × filter · added into the dashed region',
    }, outCol);

    // ---- Narration card ---------------------------------------------
    const narration = el('section', { class: 's7-narration' }, wrap);
    const narrTitle = el('div', { class: 's7-narration-title', text: 'press → to start' }, narration);
    const narrBody  = el('div', { class: 's7-narration-body' }, narration);

    // ---- Step controls ----------------------------------------------
    const controls = el('section', { class: 's7-controls' }, wrap);
    const stepGroup = el('div', { class: 's7-control-group' }, controls);
    el('label', { class: 's7-control-label', text: 'step', for: 's7-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's7-step',
      type: 'range',
      min: '0',
      max: '9',
      value: '0',
      step: '1',
    }, stepGroup);
    const stepOut = el('output', { class: 's7-control-value', text: '0 / 9' }, stepGroup);

    const navGroup = el('div', { class: 's7-control-group' }, controls);
    const prevBtn  = el('button', { type: 'button', class: 's7-btn', text: '← prev' }, navGroup);
    const nextBtn  = el('button', { type: 'button', class: 's7-btn s7-btn-primary', text: 'next stamp →' }, navGroup);
    const playBtn  = el('button', { type: 'button', class: 's7-btn', text: 'play ▶' }, navGroup);
    const resetBtn = el('button', { type: 'button', class: 's7-btn', text: 'reset' }, navGroup);

    // ---- Caption -----------------------------------------------------
    const caption = el('p', { class: 'caption s7-caption' }, wrap);

    // ---- State -------------------------------------------------------
    const state = {
      step: 0,
      stride: 2,                 // default to 2 so the sparse grid is visible
      filterKey: 'plus',
      input: DEFAULT_INPUT.map(function (r) { return r.slice(); }),
    };

    function currentFilter() { return FILTER_PRESETS[state.filterKey]; }
    function maxStep() { return N_INPUT * N_INPUT; }     // = 9

    let trace = null;
    function rebuildTrace() {
      trace = buildStampTrace(state.input, currentFilter(), state.stride);
    }

    function captionFor(step) {
      const od = outDim(state.stride);
      if (step === 0) {
        return 'The output is empty. We will fill it with ' + maxStep()
          + ' stamps — one for each input cell.';
      }
      if (step >= maxStep()) {
        if (state.stride < K_FILT) {
          return 'All ' + maxStep() + ' stamps placed. With s = ' + state.stride
            + ' < k = 3, stamps overlap and sum.';
        } else if (state.stride === K_FILT) {
          return 'All ' + maxStep() + ' stamps placed. With s = k = 3 stamps tile cleanly — no overlap, no gaps.';
        } else {
          return 'All ' + maxStep() + ' stamps placed. With s = ' + state.stride
            + ' > k = 3 there are visible gaps (zero cells) between stamps in the ' + od + '×' + od + ' output.';
        }
      }
      return 'Stamp ' + step + ' of ' + maxStep() + '. Filter centre lands on the next input value; multiply the filter by it and add into the output.';
    }

    function renderCheat() {
      cInputV.textContent = N_INPUT + ' × ' + N_INPUT;
      const sd = stridedDim(state.stride);
      cStridedV.textContent = sd + ' × ' + sd;
      cFiltV.textContent = K_FILT + ' × ' + K_FILT
        + ' (' + FILTER_LABELS[state.filterKey].split(' ')[0] + ')';
      cStrideV.textContent = String(state.stride);
      const od = outDim(state.stride);
      cOutV.textContent = od + ' × ' + od;
    }

    function renderStrided() {
      const D = paddedStridedDim(state.stride);
      const grid = makeStrided(state.input, state.stride);
      // zero-mask: cells that are NOT real input positions.
      // Real input cell positions in the padded grid: (PAD + i*s, PAD + j*s)
      const zMask = zeros2D(D, D);
      const realSet = new Set();
      for (let i = 0; i < N_INPUT; i++) {
        for (let j = 0; j < N_INPUT; j++) {
          realSet.add((PAD + i * state.stride) + ',' + (PAD + j * state.stride));
        }
      }
      for (let r = 0; r < D; r++) {
        for (let c = 0; c < D; c++) {
          if (!realSet.has(r + ',' + c)) zMask[r][c] = true;
        }
      }

      // Filter overlay if we're mid-walk OR positioned on a stamp boundary.
      let overlay = null;
      // Treat the "current" stamp as state.step (1-based). At step 0 we show
      // the filter at the very first position so the user can preview the
      // mechanic; once they advance it tracks the most-recent stamp.
      const activeIdx = state.step === 0 ? 0
        : Math.min(state.step - 1, maxStep() - 1);
      const m = trace.meta[activeIdx];
      overlay = {
        tl: m.filterTL,
        filter: currentFilter(),
        centreCell: m.filterCenter,
        color: '#d97a1f',
      };
      // At step 0 we want a visually muted overlay (just to show the mechanic)
      // — we still draw it but the narration says "preview"

      paintGrid(stridedHost, grid, {
        cell: STRIDED_CELL,
        vmax: Math.max(2, maxAbs(state.input)),
        zeroMask: zMask,
        filterOverlay: overlay,
      });
    }

    function renderOutput() {
      const od = outDim(state.stride);
      const data = trace.frames[Math.min(state.step, trace.frames.length - 1)];
      // Use the final frame for color normalization so colors don't shift
      // wildly mid-animation.
      const finalMax = Math.max(2, maxAbs(trace.frames[trace.frames.length - 1]));
      const opts = { cell: OUT_CELL, vmax: finalMax };
      // Show dashed target box for the active stamp position
      const activeIdx = state.step === 0 ? 0
        : Math.min(state.step - 1, maxStep() - 1);
      const m = trace.meta[activeIdx];
      opts.targetBox = {
        row: m.targetR[0], col: m.targetC[0],
        rows: K_FILT, cols: K_FILT,
      };
      paintGrid(outHost, data, opts);
      outHost.style.width = (OUT_CELL * od) + 'px';
      outHost.style.height = (OUT_CELL * od) + 'px';
    }

    function renderNarration() {
      narrBody.innerHTML = '';
      if (state.step === 0) {
        narrTitle.textContent = 'preview — filter centre on input cell (0, 0)';
        const row = el('div', { class: 's7-n-row' }, narrBody);
        row.innerHTML = '<span class="s7-n-k">how</span>'
          + '<span class="s7-n-v">Press <kbd>→</kbd> or <kbd>next stamp</kbd> to deposit the first stamp into the output.</span>';
        return;
      }
      if (state.step >= maxStep()) {
        narrTitle.textContent = 'all ' + maxStep() + ' stamps placed';
        const od = outDim(state.stride);
        const overlap = state.stride < K_FILT;
        const tile = state.stride === K_FILT;
        let msg;
        if (overlap) {
          msg = 'Stamps overlap by ' + (K_FILT - state.stride)
            + ' cell(s) in each direction; output cells in the overlap regions sum multiple contributions.';
        } else if (tile) {
          msg = 'Stamps tile perfectly (no overlap, no gaps). Each output cell receives exactly one contribution.';
        } else {
          msg = 'Stamps leave ' + (state.stride - K_FILT)
            + ' empty column(s) and row(s) between them — visible gaps in the output.';
        }
        const row = el('div', { class: 's7-n-row' }, narrBody);
        row.innerHTML = '<span class="s7-n-k">layout</span>'
          + '<span class="s7-n-v">' + msg + '</span>';
        const row2 = el('div', { class: 's7-n-row' }, narrBody);
        row2.innerHTML = '<span class="s7-n-k">size</span>'
          + '<span class="s7-n-v">Output is ' + od + '×' + od
          + ' = (3 − 1)·' + state.stride + ' + 3.</span>';
        return;
      }
      const m = trace.meta[state.step - 1];
      const [i, j] = m.inputCell;
      const v = m.inputValue;
      const cR = m.filterCenter[0], cC = m.filterCenter[1];
      const tr = m.targetR, tc = m.targetC;
      narrTitle.textContent = 'Stamp ' + m.k + ' of ' + maxStep()
        + ' — filter centre on strided input (' + cR + ', ' + cC
        + '), which is X[' + i + '][' + j + ']';
      const lines = [];
      lines.push('<span class="s7-n-k">value</span>'
        + '<span class="s7-n-v">X[' + i + '][' + j + '] = <strong>' + v + '</strong></span>');
      if (v === 0) {
        lines.push('<span class="s7-n-k">stamp</span>'
          + '<span class="s7-n-v">0 × K = all zeros — nothing is added</span>');
      } else {
        lines.push('<span class="s7-n-k">stamp</span>'
          + '<span class="s7-n-v">' + v + ' × K placed at output rows '
          + tr[0] + '–' + tr[1] + ', cols ' + tc[0] + '–' + tc[1]
          + '  (dashed box below)</span>');
      }
      lines.forEach(function (h) {
        const row = el('div', { class: 's7-n-row' }, narrBody);
        row.innerHTML = h;
      });
    }

    function render() {
      if (state.step > maxStep()) state.step = maxStep();
      renderCheat();
      renderStrided();
      renderOutput();
      renderNarration();

      // Progress label
      const labelTxt = state.step === 0
        ? 'press → to begin (0 of ' + maxStep() + ')'
        : (state.step >= maxStep()
            ? 'all ' + maxStep() + ' stamps placed'
            : 'stamp ' + state.step + ' of ' + maxStep());
      panelProgress.textContent = labelTxt;

      // Step input
      stepInput.max = String(maxStep());
      stepInput.value = String(state.step);
      stepOut.textContent = state.step + ' / ' + maxStep();

      prevBtn.disabled = state.step <= 0;
      nextBtn.disabled = state.step >= maxStep();
      resetBtn.disabled = state.step === 0;
      playBtn.disabled = state.step >= maxStep();
      nextBtn.textContent = state.step === 0 ? 'first stamp →'
        : (state.step + 1 === maxStep() ? 'last stamp →' : 'next stamp →');

      caption.textContent = captionFor(state.step);
    }

    function applyStep(n) {
      state.step = Math.max(0, Math.min(maxStep(), n));
      render();
    }

    // ---- Wire controls ----------------------------------------------
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    // Play full
    let runTimer = null;
    function clearRun() { if (runTimer) { clearTimeout(runTimer); runTimer = null; } }
    function autoStep() {
      if (state.step >= maxStep()) { clearRun(); return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoStep, RUN_STEP_MS);
    }
    playBtn.addEventListener('click', function () {
      clearRun();
      applyStep(0);
      runTimer = setTimeout(autoStep, 500);
    });

    // Stride knob
    function setStride(s) {
      state.stride = s;
      Object.keys(strideBtns).forEach(function (kk) {
        strideBtns[kk].classList.toggle('s7-active', Number(kk) === s);
      });
      rebuildTrace();
      applyStep(0);
    }
    [1, 2, 3, 4].forEach(function (s) {
      strideBtns[s].addEventListener('click', function () { setStride(s); });
    });

    // Filter knob
    function setFilter(k) {
      state.filterKey = k;
      Object.keys(filtBtns).forEach(function (kk) {
        filtBtns[kk].classList.toggle('s7-active', kk === k);
      });
      rebuildTrace();
      applyStep(0);
    }
    Object.keys(filtBtns).forEach(function (k) {
      filtBtns[k].addEventListener('click', function () { setFilter(k); });
    });

    // ---- Initial paint ----------------------------------------------
    rebuildTrace();
    render();

    if (readHashFlag('run')) {
      runTimer = setTimeout(function () {
        applyStep(0);
        runTimer = setTimeout(autoStep, 500);
      }, 200);
    }

    return {
      onEnter: function () { render(); },
      onLeave: function () { clearRun(); },
      onNextKey: function () {
        if (state.step < maxStep()) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene7 = function (root) { return buildScene(root); };
})();
