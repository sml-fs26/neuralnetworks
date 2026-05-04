/* Scene 0 — "A U-Net, end to end."  (Ronneberger-style architecture overview)

   Replaces the previous overture with a faithful per-tensor diagram:
   all 19 intermediate tensors are drawn as bars whose HEIGHT encodes
   spatial side and WIDTH encodes channel count. Two bars per encoder /
   decoder block (conv1 + conv2) make the truth visible. Operation arrows
   are color-coded with a legend.

   Step engine (4 steps):
     0  bare schematic (bars + labels + arrows + legend)
     1  real-image bookends fade in (input on left, predicted seg on right)
     2  hover-popover sidebar enabled (shape + op + 4-channel preview where available)
     3  "play forward" sweep lights up tensors in execution order

   Reads:
     window.DATA.scene64.samples[0..5]  (input/label/pred + 4-channel previews)
     window.DATA.scene64.classes
     window.Drawing.{paintRGB, paintLabelMap, paintFeatureCard,
                      setupCanvas, tokens} */
(function () {
  'use strict';

  const NUM_STEPS = 4;
  const RUN_INTERVAL_MS = 1500;   // step-to-step duration in &run mode
  const SWEEP_STEP_MS = 500;      // tensor-to-tensor in the play sweep
  const BOOKEND_PX = 84;

  /* Bar geometry --------------------------------------------------- */
  const BAR_HEIGHT_BASE = 180;          // px at spatial=64
  const BAR_WIDTH_BASE = 6;             // px constant
  const BAR_WIDTH_PER_CHANNEL = 0.42;   // px per channel
  // Widths: 3ch≈7.3, 16ch≈12.7, 32ch≈19.4, 64ch≈32.9, 5ch≈8.1, 1ch≈6.4
  function barWidth(channels)  { return BAR_WIDTH_BASE + BAR_WIDTH_PER_CHANNEL * channels; }
  function barHeight(spatial)  { return BAR_HEIGHT_BASE * (spatial / 64); }

  /* Op color palette (theme-agnostic; chosen to be legible in light & dark) */
  const OP_COLORS = {
    conv:    '#5aa64a', // green: conv 3x3 + ReLU
    pool:    '#c95a5a', // red: max-pool 2x2
    upconv:  '#9067c2', // purple: transposed conv 2x2
    skip:    null,      // gray: pulled from --ink-secondary at render time
    onexone: '#3a8fb7', // blue: 1x1 conv
  };

  /* All 19 tensors. Order = execution order. Each gets an X position
     (pre-computed below) and a Y based on its row.

     Tensors that have a 4-channel preview in DATA.scene64.samples: enc1, enc2,
     enc3, dec2, dec1 -- which correspond to the OUTPUT of conv2 of each block,
     i.e. tensor indices 2, 5, 8, 12, 16. */
  const TENSORS = [
    { id: 't0',  name: 'input',          row: 1, spatial: 64, channels: 3,  op: '(input)',                 short: 'input',   preview: null    },
    { id: 't1',  name: 'enc1.conv1 out', row: 1, spatial: 64, channels: 16, op: 'conv 3×3 + ReLU',         short: 'e1c1',    preview: null    },
    { id: 't2',  name: 'enc1.conv2 out', row: 1, spatial: 64, channels: 16, op: 'conv 3×3 + ReLU',         short: 'e1c2',    preview: 'enc1'  },
    { id: 't3',  name: 'pool1 out',      row: 2, spatial: 32, channels: 16, op: 'max-pool 2×2',            short: 'pool1',   preview: null    },
    { id: 't4',  name: 'enc2.conv1 out', row: 2, spatial: 32, channels: 32, op: 'conv 3×3 + ReLU',         short: 'e2c1',    preview: null    },
    { id: 't5',  name: 'enc2.conv2 out', row: 2, spatial: 32, channels: 32, op: 'conv 3×3 + ReLU',         short: 'e2c2',    preview: 'enc2'  },
    { id: 't6',  name: 'pool2 out',      row: 3, spatial: 16, channels: 32, op: 'max-pool 2×2',            short: 'pool2',   preview: null    },
    { id: 't7',  name: 'enc3.conv1 out', row: 3, spatial: 16, channels: 64, op: 'conv 3×3 + ReLU',         short: 'e3c1',    preview: null    },
    { id: 't8',  name: 'enc3.conv2 out', row: 3, spatial: 16, channels: 64, op: 'conv 3×3 + ReLU',         short: 'e3c2',    preview: 'enc3', annotate: 'bottleneck' },
    { id: 't9',  name: 'up2 out',        row: 2, spatial: 32, channels: 32, op: 'transposed conv 2×2',     short: 'up2',     preview: null    },
    { id: 't10', name: 'concat(up2, e2c2)', row: 2, spatial: 32, channels: 64, op: 'concat (channel)',     short: 'cat2',    preview: null, annotate: 'concat',   split: true },
    { id: 't11', name: 'dec2.conv1 out', row: 2, spatial: 32, channels: 32, op: 'conv 3×3 + ReLU',         short: 'd2c1',    preview: null    },
    { id: 't12', name: 'dec2.conv2 out', row: 2, spatial: 32, channels: 32, op: 'conv 3×3 + ReLU',         short: 'd2c2',    preview: 'dec2'  },
    { id: 't13', name: 'up1 out',        row: 1, spatial: 64, channels: 16, op: 'transposed conv 2×2',     short: 'up1',     preview: null    },
    { id: 't14', name: 'concat(up1, e1c2)', row: 1, spatial: 64, channels: 32, op: 'concat (channel)',     short: 'cat1',    preview: null, annotate: 'concat',   split: true },
    { id: 't15', name: 'dec1.conv1 out', row: 1, spatial: 64, channels: 16, op: 'conv 3×3 + ReLU',         short: 'd1c1',    preview: null    },
    { id: 't16', name: 'dec1.conv2 out', row: 1, spatial: 64, channels: 16, op: 'conv 3×3 + ReLU',         short: 'd1c2',    preview: 'dec1'  },
    { id: 't17', name: 'logits',         row: 1, spatial: 64, channels: 5,  op: 'conv 1×1',                short: 'logits',  preview: null    },
    { id: 't18', name: 'output (argmax)',row: 1, spatial: 64, channels: 1,  op: 'argmax + softmax',        short: 'output',  preview: null, annotate: 'output (argmax)' },
  ];

  /* Operation edges between consecutive tensors. Some ops cross rows
     (pool / upconv); concat is a special skip arrow drawn separately. */
  const OPS = [
    { from: 0,  to: 1,  type: 'conv'    },
    { from: 1,  to: 2,  type: 'conv'    },
    { from: 2,  to: 3,  type: 'pool'    },
    { from: 3,  to: 4,  type: 'conv'    },
    { from: 4,  to: 5,  type: 'conv'    },
    { from: 5,  to: 6,  type: 'pool'    },
    { from: 6,  to: 7,  type: 'conv'    },
    { from: 7,  to: 8,  type: 'conv'    },
    { from: 8,  to: 9,  type: 'upconv'  },
    { from: 9,  to: 10, type: 'concat-merge' },  // up2 → cat2 (decoder side of concat)
    { from: 10, to: 11, type: 'conv'    },
    { from: 11, to: 12, type: 'conv'    },
    { from: 12, to: 13, type: 'upconv'  },
    { from: 13, to: 14, type: 'concat-merge' },  // up1 → cat1
    { from: 14, to: 15, type: 'conv'    },
    { from: 15, to: 16, type: 'conv'    },
    { from: 16, to: 17, type: 'onexone' },
    { from: 17, to: 18, type: 'argmax'  },
  ];

  /* Skip connections: encoder.conv2 → concat tensor on the decoder side. */
  const SKIPS = [
    { from: 2,  to: 14 },  // e1c2 -> cat1
    { from: 5,  to: 10 },  // e2c2 -> cat2
  ];

  /* ---------------------------------------------------------------
     Compute layout (X, Y per tensor) once. Pure function of constants.
     --------------------------------------------------------------- */
  function computeLayout() {
    const INTRA = 10;     // within a block
    const BLOCK = 22;     // between blocks within a row
    const ROW_Y = { 1: 150, 2: 330, 3: 480 };

    // Walk left-to-right per row, assigning X positions. We need the X for
    // a NEW block to be `prev block's right edge + BLOCK`. Within a block
    // it's `prev's right + INTRA`. We define block boundaries explicitly.
    //
    // Strategy: use a simulator with manual "pen" positions per row and
    // explicitly inserted gaps when a new block begins.

    const pos = {};   // tensor.id → { x, y, w, h, cx, cy }

    // Row 1 LEFT side: input → e1c1 → e1c2
    let x = 20;
    function place(id, channels, spatial, row, gap) {
      const w = barWidth(channels);
      const h = barHeight(spatial);
      const cy = ROW_Y[row];
      const y = cy - h / 2;
      pos[id] = { x: x, y: y, w: w, h: h, cx: x + w / 2, cy: cy };
      x += w + gap;
    }
    place('t0',  3,  64, 1, INTRA);
    place('t1',  16, 64, 1, INTRA);
    place('t2',  16, 64, 1, BLOCK);
    const r1LeftEnd = x - BLOCK;  // right edge of e1c2 + nothing else

    // Row 2 LEFT side: pool1 starts BLOCK to the right of e1c2's right edge
    x = pos.t2.x + pos.t2.w + BLOCK;
    place('t3',  16, 32, 2, INTRA);
    place('t4',  32, 32, 2, INTRA);
    place('t5',  32, 32, 2, BLOCK);

    // Row 3: pool2 starts BLOCK to the right of e2c2
    x = pos.t5.x + pos.t5.w + BLOCK;
    place('t6',  32, 16, 3, INTRA);
    place('t7',  64, 16, 3, INTRA);
    place('t8',  64, 16, 3, BLOCK);

    // Row 2 RIGHT side: up2 starts BLOCK to the right of e3c2
    x = pos.t8.x + pos.t8.w + BLOCK;
    place('t9',  32, 32, 2, INTRA);
    place('t10', 64, 32, 2, INTRA);
    place('t11', 32, 32, 2, INTRA);
    place('t12', 32, 32, 2, BLOCK);

    // Row 1 RIGHT side: up1 starts BLOCK to the right of d2c2
    x = pos.t12.x + pos.t12.w + BLOCK;
    place('t13', 16, 64, 1, INTRA);
    place('t14', 32, 64, 1, INTRA);
    place('t15', 16, 64, 1, INTRA);
    place('t16', 16, 64, 1, BLOCK);
    place('t17', 5,  64, 1, INTRA);
    place('t18', 1,  64, 1, INTRA);

    return { pos: pos, rowY: ROW_Y };
  }

  /* ---------------------------------------------------------------
     Tiny DOM helper.
     --------------------------------------------------------------- */
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
  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === 'text') node.textContent = attrs[k];
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

  /* ---------------------------------------------------------------
     Build scene.
     --------------------------------------------------------------- */
  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.Drawing) {
      root.innerHTML = '<p style="opacity:0.5">Scene 0: missing globals.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    if (!D.samples || !D.samples.length) {
      root.innerHTML = '<p style="opacity:0.5">Scene 0: no samples.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s0-root');
    const wrap = el('div', { class: 's0-wrap' }, root);

    /* ---- Hero ---------------------------------------------------- */
    const hero = el('header', { class: 'hero s0-hero' }, wrap);
    el('h1', { text: 'A U-Net, end to end.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Every tensor that flows through the network — drawn to scale.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Bar <em>height</em> encodes spatial size (64² → 32² → 16² → 32² → 64²). ' +
        'Bar <em>width</em> encodes channel count (the bottleneck pair is the fattest). ' +
        'Each encoder/decoder block is two convolutions, so it gets two bars. ' +
        'Hover any bar (step 2) for its shape and a peek at its feature maps; ' +
        'press <em>play forward</em> (step 3) to watch the activation flow.',
    }, hero);

    /* ---- Sample picker + step controls --------------------------- */
    const ctrl = el('div', { class: 's0-controls-top' }, wrap);
    const sampleRow = el('div', { class: 's0-sample-row' }, ctrl);
    el('span', { class: 's0-control-label', text: 'sample' }, sampleRow);
    const sampleBtns = [];
    for (let i = 0; i < D.samples.length; i++) {
      const b = el('button', {
        type: 'button', class: 's0-sample-btn', text: String(i + 1),
        'data-idx': String(i),
      }, sampleRow);
      sampleBtns.push(b);
    }
    const playBtn = el('button', {
      type: 'button', class: 's0-play-btn', text: '▶ play forward',
    }, sampleRow);

    /* ---- The diagram (SVG inside scrollable host) ---------------- */
    const layout = computeLayout();

    // Canvas extents
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    Object.keys(layout.pos).forEach(function (k) {
      const p = layout.pos[k];
      if (p.x < minX) minX = p.x;
      if (p.x + p.w > maxX) maxX = p.x + p.w;
      if (p.y < minY) minY = p.y;
      if (p.y + p.h > maxY) maxY = p.y + p.h;
    });
    // Margins to fit labels, bookends, and the legend card.
    const PAD_LEFT = 18 + BOOKEND_PX + 20;   // bookend + gap on the left of input
    const PAD_RIGHT = 18 + BOOKEND_PX + 20;  // bookend on the right of output
    const PAD_TOP = 22;     // channel-count labels (legend sits over the U interior)
    const PAD_BOTTOM = 60;  // spatial labels + annotations

    const SVG_W = maxX + PAD_LEFT + PAD_RIGHT;
    const SVG_H = maxY + PAD_TOP + PAD_BOTTOM;

    // Shift everything by PAD_LEFT, PAD_TOP so the SVG origin can be 0.
    Object.keys(layout.pos).forEach(function (k) {
      const p = layout.pos[k];
      p.x += PAD_LEFT;
      p.y += PAD_TOP;
      p.cx += PAD_LEFT;
      p.cy += PAD_TOP;
    });

    const arch = el('div', { class: 's0-arch' }, wrap);
    const svgHost = el('div', { class: 's0-svg-host' }, arch);

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + Math.round(SVG_W) + ' ' + Math.round(SVG_H));
    svg.setAttribute('width',  String(Math.round(SVG_W)));
    svg.setAttribute('height', String(Math.round(SVG_H)));
    svg.setAttribute('class', 's0-svg');
    svgHost.appendChild(svg);

    // <defs>: arrow markers, one per op color.
    const defs = svgEl('defs', null, svg);
    function makeMarker(id, color) {
      const m = svgEl('marker', {
        id: id, viewBox: '0 0 10 10', refX: '8', refY: '5',
        markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse',
      }, defs);
      svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }, m);
    }
    makeMarker('s0-mk-conv',    OP_COLORS.conv);
    makeMarker('s0-mk-pool',    OP_COLORS.pool);
    makeMarker('s0-mk-upconv',  OP_COLORS.upconv);
    makeMarker('s0-mk-onexone', OP_COLORS.onexone);
    makeMarker('s0-mk-argmax',  '#888');
    makeMarker('s0-mk-skip-light', '#9e9789');
    makeMarker('s0-mk-skip-dark',  '#5f5b54');

    // Group order matters: arrows (under), then bars (over), then labels (over).
    const gArrows = svgEl('g', { class: 's0-g-arrows' }, svg);
    const gSkips  = svgEl('g', { class: 's0-g-skips'  }, svg);
    const gBars   = svgEl('g', { class: 's0-g-bars'   }, svg);
    const gLabels = svgEl('g', { class: 's0-g-labels' }, svg);
    const gAnnot  = svgEl('g', { class: 's0-g-annot'  }, svg);

    // Tensor index → DOM rect / DOM group / hit rect, for hover and sweep.
    const tensorBars = {};   // id → { rect, group, hit }

    /* ---- Draw bars + labels ------------------------------------- */
    TENSORS.forEach(function (t, idx) {
      const p = layout.pos[t.id];
      const grp = svgEl('g', { class: 's0-bar-group', 'data-idx': String(idx), 'data-id': t.id }, gBars);

      if (t.split) {
        // Concat: two halves stacked horizontally. Front half (upsample side)
        // = lighter blue; back half (skip side) = orange-ish to read as "the
        // skip got pasted in". This makes the concat operation visible.
        const halfW = p.w / 2;
        // Upsample side (left half)
        svgEl('rect', {
          class: 's0-bar s0-bar-up',
          x: String(p.x), y: String(p.y),
          width: String(halfW), height: String(p.h),
          'data-half': 'up',
        }, grp);
        // Skip side (right half)
        svgEl('rect', {
          class: 's0-bar s0-bar-skip',
          x: String(p.x + halfW), y: String(p.y),
          width: String(halfW), height: String(p.h),
          'data-half': 'skip',
        }, grp);
        // Outline both halves together so the bar reads as one tensor.
        svgEl('rect', {
          class: 's0-bar-outline',
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
      } else {
        const cls =
          (t.id === 't0')  ? 's0-bar s0-bar-input' :
          (t.id === 't18') ? 's0-bar s0-bar-output' :
                             's0-bar s0-bar-act';
        svgEl('rect', {
          class: cls,
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
        svgEl('rect', {
          class: 's0-bar-outline',
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
      }

      // Wide invisible hit rect for hover (covers above/below labels too).
      const hit = svgEl('rect', {
        class: 's0-bar-hit',
        x: String(p.x - 4),
        y: String(p.y - 22),
        width: String(p.w + 8),
        height: String(p.h + 60),
        fill: 'transparent',
        'data-idx': String(idx),
      }, grp);

      tensorBars[t.id] = { group: grp, hit: hit };

      // Label: channel count above, spatial below.
      svgEl('text', {
        class: 's0-label-channels',
        x: String(p.cx),
        y: String(p.y - 8),
        'text-anchor': 'middle',
        text: String(t.channels),
      }, gLabels);
      svgEl('text', {
        class: 's0-label-spatial',
        x: String(p.cx),
        y: String(p.y + p.h + 14),
        'text-anchor': 'middle',
        text: t.spatial + '²',
      }, gLabels);
      if (t.annotate) {
        svgEl('text', {
          class: 's0-label-annot',
          x: String(p.cx),
          y: String(p.y + p.h + 28),
          'text-anchor': 'middle',
          text: t.annotate,
        }, gAnnot);
      }
    });

    /* ---- Draw operation arrows (skip excluded) ------------------ */
    OPS.forEach(function (op) {
      const from = TENSORS[op.from], to = TENSORS[op.to];
      const pf = layout.pos[from.id], pt = layout.pos[to.id];
      const x1 = pf.x + pf.w;       // right edge of source
      const y1 = pf.cy;
      const x2 = pt.x;              // left edge of dest
      const y2 = pt.cy;
      let pathD;
      let cls;
      let mk;
      if (op.type === 'conv') {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-conv';
        mk = 's0-mk-conv';
      } else if (op.type === 'pool') {
        // Down-diagonal between rows. Slight curve for readability.
        const midY = (y1 + y2) / 2;
        pathD = 'M ' + x1 + ' ' + y1 +
                ' C ' + (x1 + 18) + ' ' + y1 + ', ' +
                        (x2 - 18) + ' ' + y2 + ', ' +
                        x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-pool';
        mk = 's0-mk-pool';
      } else if (op.type === 'upconv') {
        // Up-diagonal between rows.
        pathD = 'M ' + x1 + ' ' + y1 +
                ' C ' + (x1 + 18) + ' ' + y1 + ', ' +
                        (x2 - 18) + ' ' + y2 + ', ' +
                        x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-upconv';
        mk = 's0-mk-upconv';
      } else if (op.type === 'concat-merge') {
        // up2 → cat2 (and up1 → cat1): a short straight in-row arrow that
        // says "the upsample becomes the front half of the concat".
        // We render it like a conv arrow but in the upsample purple to
        // emphasize the lineage. It is the SHORT arrow; the SKIP arc is
        // the long counterpart.
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-upconv';
        mk = 's0-mk-upconv';
      } else if (op.type === 'onexone') {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-onexone';
        mk = 's0-mk-onexone';
      } else { // argmax — a thin gray arrow
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's0-arrow s0-arrow-argmax';
        mk = 's0-mk-argmax';
      }
      svgEl('path', {
        class: cls,
        d: pathD,
        fill: 'none',
        'marker-end': 'url(#' + mk + ')',
      }, gArrows);
    });

    /* ---- Draw skip arcs ----------------------------------------- */
    SKIPS.forEach(function (sk, kIdx) {
      const from = TENSORS[sk.from], to = TENSORS[sk.to];
      const pf = layout.pos[from.id], pt = layout.pos[to.id];
      // Source: top-right of source bar (just above).
      // Dest: arrives at the LEFT edge of cat tensor's SKIP HALF, which is
      // the right half of the split bar.
      const halfW = pt.w / 2;
      const x1 = pf.x + pf.w - 4;
      const y1 = pf.y - 4;
      const x2 = pt.x + halfW + 2;   // tip lands inside the skip half
      const y2 = pt.y - 4;
      // Arc bows up.
      const dx = x2 - x1;
      const peak = Math.max(20, 30 - kIdx * 6);  // skip1 (top row) peaks higher
      const yc = Math.min(y1, y2) - peak;
      const cx1 = x1 + Math.max(80, dx * 0.25);
      const cx2 = x2 - Math.max(80, dx * 0.25);
      const pathD =
        'M ' + x1 + ' ' + y1 + ' ' +
        'C ' + cx1 + ' ' + yc + ', ' +
              cx2 + ' ' + yc + ', ' +
              x2  + ' ' + y2;
      svgEl('path', {
        class: 's0-arrow s0-arrow-skip',
        d: pathD,
        fill: 'none',
        'marker-end': 'url(#s0-mk-skip-light)',
      }, gSkips);
      // A small label between the apex of skip arc 2 (mid-row) and the
      // bars below it, where there's whitespace. The legend already
      // explains the dashed style, so we keep this brief.
      if (kIdx === 1) {
        svgEl('text', {
          class: 's0-arrow-label',
          x: String((x1 + x2) / 2),
          y: String(yc - 6),
          'text-anchor': 'middle',
          text: 'skip · concat along channels',
        }, gSkips);
      }
    });

    /* ---- Bookend hosts (HTML overlays on top of the SVG) -------- */
    // We position them in HTML coordinates that match the SVG viewBox by
    // using the parent's `.s0-svg-host` as the positioned ancestor.
    // SVG is `width = SVG_W`; the HTML host has the same width so 1 SVG
    // unit == 1 CSS px. (We don't apply CSS scaling to the SVG.)

    function makeBookend(host, klass, x, y, px, label) {
      const node = el('div', { class: 's0-bookend ' + klass }, host);
      node.style.left   = (x - px / 2) + 'px';
      node.style.top    = (y - px / 2) + 'px';
      node.style.width  = px + 'px';
      node.style.height = px + 'px';
      const inner = el('div', { class: 's0-bookend-canvas' }, node);
      inner.style.width = px + 'px';
      inner.style.height = px + 'px';
      el('div', { class: 's0-bookend-label', text: label }, node);
      return inner;
    }

    const inputBook = makeBookend(
      svgHost, 's0-bookend-input',
      layout.pos.t0.x - 16 - BOOKEND_PX / 2,
      layout.pos.t0.cy,
      BOOKEND_PX,
      'real input'
    );
    const outputBook = makeBookend(
      svgHost, 's0-bookend-output',
      layout.pos.t18.x + layout.pos.t18.w + 16 + BOOKEND_PX / 2,
      layout.pos.t18.cy,
      BOOKEND_PX,
      'predicted seg'
    );

    /* ---- Legend (HTML overlay anchored inside the SVG host) ----- */
    // Anchor it inside the U-shape's open top (between e1c2 right edge
    // and cat1 left edge) so it never overlaps a bar.
    const legend = el('div', { class: 's0-legend' }, svgHost);
    const legendLeft = layout.pos.t2.x + layout.pos.t2.w + 24;
    legend.style.left = legendLeft + 'px';
    legend.style.top  = '8px';
    function legendItem(swatchClass, label) {
      const row = el('div', { class: 's0-legend-row' }, legend);
      el('span', { class: 's0-legend-swatch ' + swatchClass }, row);
      el('span', { class: 's0-legend-label', text: label }, row);
    }
    el('div', { class: 's0-legend-title', text: 'operations' }, legend);
    legendItem('lg-conv',    'conv 3×3 + ReLU');
    legendItem('lg-pool',    'max-pool 2×2');
    legendItem('lg-upconv',  'transposed conv 2×2');
    legendItem('lg-skip',    'skip · concat (channels)');
    legendItem('lg-onexone', 'conv 1×1');

    /* ---- Hover sidebar ------------------------------------------ */
    const sidebar = el('aside', { class: 's0-sidebar' }, arch);
    const sbHead = el('div', { class: 's0-sb-head' }, sidebar);
    const sbName = el('div', { class: 's0-sb-name', text: 'hover a tensor' }, sbHead);
    const sbOp   = el('div', { class: 's0-sb-op',   text: 'step 2 enables this panel' }, sbHead);
    const sbShape = el('div', { class: 's0-sb-shape' }, sidebar);
    const sbPreviewWrap = el('div', { class: 's0-sb-preview-wrap' }, sidebar);
    const sbPreviewSubtitle = el('div', { class: 's0-sb-preview-subtitle' }, sidebar);
    const sbHint = el('div', { class: 's0-sb-hint' }, sidebar);

    /* ---- Caption ------------------------------------------------ */
    const caption = el('p', { class: 'caption s0-caption' }, wrap);

    /* ---- Step controls ------------------------------------------ */
    const controls = el('div', { class: 'controls s0-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's0-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's0-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State -------------------------------------------------- */
    function pickRichest(samples) {
      let best = 0, bestCount = -1;
      for (let k = 0; k < samples.length; k++) {
        const seen = new Set();
        const lbl = samples[k].label;
        for (let i = 0; i < lbl.length; i++) {
          for (let j = 0; j < lbl[0].length; j++) seen.add(lbl[i][j]);
        }
        if (seen.size > bestCount) { bestCount = seen.size; best = k; }
      }
      return best;
    }
    const state = {
      step: 0,
      sampleIdx: pickRichest(D.samples),
      hoveredIdx: null,
      sweepIdx: -1,
      sweepTimer: null,
      runTimer: null,
    };
    const sample = function () { return D.samples[state.sampleIdx]; };

    /* ---- Caption text ------------------------------------------- */
    function captionFor(step) {
      switch (step) {
        case 0: return 'Bare schematic. Three rows = three spatial scales. Five colors = five operations.';
        case 1: return 'Real bookends: a 64×64 RGB image enters left, the predicted per-pixel labels exit right.';
        case 2: return 'Hover any bar for its shape and operation. Convolution-block outputs come with a 2×2 preview of four representative channels.';
        case 3: return 'Play forward: each tensor lights up in execution order. Watch resolution shrink, channels grow, then both unwind.';
        default: return '';
      }
    }

    /* ---- Sample picker UI -------------------------------------- */
    function updateSampleBtns() {
      sampleBtns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }
    sampleBtns.forEach(function (b, i) {
      b.addEventListener('click', function () {
        state.sampleIdx = i;
        updateSampleBtns();
        renderBookends();
        // Refresh the popover's preview if a tensor is currently shown.
        if (state.hoveredIdx != null) renderPopover(state.hoveredIdx);
      });
    });

    /* ---- Bookends rendering ------------------------------------ */
    function renderBookends() {
      const s = sample();
      window.Drawing.paintRGB(inputBook, s.input, BOOKEND_PX);
      window.Drawing.paintLabelMap(outputBook, s.pred, BOOKEND_PX);
    }

    /* ---- Popover (hover sidebar) ------------------------------- */
    function renderPopover(idx) {
      if (idx == null) {
        sbName.textContent = 'hover a tensor';
        sbOp.textContent   = 'step 2 enables this panel';
        sbShape.textContent = '';
        sbPreviewWrap.innerHTML = '';
        sbPreviewSubtitle.textContent = '';
        sbHint.textContent = '';
        return;
      }
      const t = TENSORS[idx];
      sbName.textContent = t.name;
      sbOp.textContent   = t.op;
      sbShape.innerHTML  = '<span class="s0-sb-shape-label">shape</span> ' +
        t.spatial + ' × ' + t.spatial + ' × ' + t.channels;
      sbPreviewWrap.innerHTML = '';
      if (t.preview) {
        const stack = sample()[t.preview];
        if (stack && stack.length === 4) {
          const previewHost = el('div', { class: 's0-sb-preview' }, sbPreviewWrap);
          const PREV_PX = 132;
          previewHost.style.width = PREV_PX + 'px';
          previewHost.style.height = PREV_PX + 'px';
          window.Drawing.paintFeatureCard(previewHost, stack, PREV_PX);
          sbPreviewSubtitle.textContent = '4 of ' + t.channels + ' channels (top variance)';
          sbHint.textContent = '';
        } else {
          sbPreviewSubtitle.textContent = '';
          sbHint.textContent = 'preview unavailable for this sample.';
        }
      } else {
        sbPreviewSubtitle.textContent = '';
        sbHint.textContent = 'intermediate tensor — not exported as a preview.';
      }
    }

    /* ---- Bar hover wiring -------------------------------------- */
    function setHover(idx) {
      // Clear all bar 'hovered' classes
      Object.keys(tensorBars).forEach(function (k) {
        tensorBars[k].group.classList.remove('s0-bar-hover');
      });
      state.hoveredIdx = idx;
      if (idx != null) {
        tensorBars[TENSORS[idx].id].group.classList.add('s0-bar-hover');
        renderPopover(idx);
      } else {
        renderPopover(null);
      }
    }

    TENSORS.forEach(function (t, idx) {
      const handle = tensorBars[t.id].hit;
      handle.addEventListener('mouseenter', function () {
        if (state.step < 2) return;
        if (state.sweepTimer) return;  // don't fight the auto sweep
        setHover(idx);
      });
      handle.addEventListener('mouseleave', function () {
        if (state.step < 2) return;
        if (state.sweepTimer) return;
        // Don't clear if currently sweep-pinned to this same idx.
        setHover(null);
      });
    });

    /* ---- Sweep (step 3) --------------------------------------- */
    function clearSweep() {
      if (state.sweepTimer) { clearInterval(state.sweepTimer); state.sweepTimer = null; }
      state.sweepIdx = -1;
      Object.keys(tensorBars).forEach(function (k) {
        tensorBars[k].group.classList.remove('s0-bar-sweep');
        tensorBars[k].group.classList.remove('s0-bar-sweep-active');
      });
    }
    function startSweep() {
      clearSweep();
      state.sweepIdx = 0;
      state.sweepTimer = setInterval(function () {
        if (state.sweepIdx >= TENSORS.length) {
          clearInterval(state.sweepTimer);
          state.sweepTimer = null;
          return;
        }
        // Mark active + previous as visited.
        Object.keys(tensorBars).forEach(function (k) {
          tensorBars[k].group.classList.remove('s0-bar-sweep-active');
        });
        for (let i = 0; i < state.sweepIdx; i++) {
          tensorBars[TENSORS[i].id].group.classList.add('s0-bar-sweep');
        }
        const cur = TENSORS[state.sweepIdx];
        tensorBars[cur.id].group.classList.add('s0-bar-sweep');
        tensorBars[cur.id].group.classList.add('s0-bar-sweep-active');
        // Auto-pin the popover.
        renderPopover(state.sweepIdx);
        state.sweepIdx++;
      }, SWEEP_STEP_MS);
    }

    playBtn.addEventListener('click', function () {
      // Always run from the current step; if step < 3, jump to 3.
      if (state.step < 3) applyStep(3);
      clearSweep();
      startSweep();
    });

    /* ---- Render ------------------------------------------------ */
    function render() {
      const step = state.step;
      const wasSweepActive = !!state.sweepTimer;
      arch.classList.toggle('s0-show-bookends', step >= 1);
      arch.classList.toggle('s0-show-hover',    step >= 2);
      arch.classList.toggle('s0-show-sweep',    step >= 3);

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      if (step < 3) clearSweep();
      if (step < 2 && state.hoveredIdx != null) setHover(null);

      // Always paint bookends so they're cached; CSS controls visibility.
      renderBookends();

      // Auto-start the play sweep when first entering step 3 (so the user
      // immediately sees motion). They can re-trigger via the play button.
      if (step === 3 && !wasSweepActive) {
        // small delay so the dimming transition settles first
        setTimeout(function () {
          if (state.step === 3 && !state.sweepTimer) startSweep();
        }, 350);
      }
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    /* ---- Wire controls ----------------------------------------- */
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () {
      applyStep(0);
      state.sampleIdx = pickRichest(D.samples);
      updateSampleBtns();
      renderBookends();
    });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    /* ---- Initial paint ----------------------------------------- */
    updateSampleBtns();
    render();

    /* ---- &run: auto-advance through 0..3, then play sweep ----- */
    function autoAdvance() {
      if (state.step < NUM_STEPS - 1) {
        applyStep(state.step + 1);
        state.runTimer = setTimeout(autoAdvance, RUN_INTERVAL_MS);
      } else {
        // Reached step 3: trigger the play sweep.
        state.runTimer = null;
        playBtn.click();
      }
    }
    if (readHashFlag('run')) {
      state.runTimer = setTimeout(autoAdvance, 600);
    }

    return {
      onEnter: function () {
        render();
      },
      onLeave: function () {
        if (state.runTimer) { clearTimeout(state.runTimer); state.runTimer = null; }
        clearSweep();
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
  window.scenes.scene0 = function (root) { return buildScene(root); };
})();
