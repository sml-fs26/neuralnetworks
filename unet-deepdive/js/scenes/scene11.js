/* Scene 11 — "The U-Net, fully wired."

   The reference / answer-key card. Same Ronneberger-style per-tensor
   diagram as scene 0, but presented in its fully-wired final state from
   the very first paint: bookends visible, hover sidebar enabled, sample
   picker active, and four landmark arrow labels (pool, transposed conv,
   1×1 conv, skip concat) annotated next to the corresponding edges.

   No progressive reveal. The play-forward sweep is still available as a
   one-shot replay of the data-flow animation.

   Step engine: a single step (NUM_STEPS = 1). The diagram starts fully
   wired. The play button replays the sweep without changing any reveal
   state.

   Reads:
     window.DATA.scene64.samples[0..5]  (input/label/pred + 4-channel previews)
     window.DATA.scene64.classes
     window.Drawing.{paintRGB, paintLabelMap, paintFeatureCard,
                      setupCanvas, tokens} */
(function () {
  'use strict';

  const NUM_STEPS = 1;
  const SWEEP_STEP_MS = 500;      // tensor-to-tensor in the play sweep
  const BOOKEND_PX = 84;

  /* Bar geometry --------------------------------------------------- */
  const BAR_HEIGHT_BASE = 180;          // px at spatial=64
  const BAR_WIDTH_BASE = 6;             // px constant
  const BAR_WIDTH_PER_CHANNEL = 0.42;   // px per channel
  function barWidth(channels)  { return BAR_WIDTH_BASE + BAR_WIDTH_PER_CHANNEL * channels; }
  function barHeight(spatial)  { return BAR_HEIGHT_BASE * (spatial / 64); }

  /* Op color palette (theme-agnostic; chosen to be legible in light & dark) */
  const OP_COLORS = {
    conv:    '#5aa64a', // green: conv 3x3 + ReLU
    pool:    '#c95a5a', // red: max-pool 2x2
    upconv:  '#9067c2', // purple: transposed conv 2x2
    skip:    null,      // gray
    onexone: '#3a8fb7', // blue: 1x1 conv
  };

  /* All 19 tensors. Same list as scene 0; order = execution order. */
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

  /* Operation edges between consecutive tensors. */
  const OPS = [
    { from: 0,  to: 1,  type: 'conv'    },
    { from: 1,  to: 2,  type: 'conv'    },
    { from: 2,  to: 3,  type: 'pool',   landmark: 'max-pool, /2' },
    { from: 3,  to: 4,  type: 'conv'    },
    { from: 4,  to: 5,  type: 'conv'    },
    { from: 5,  to: 6,  type: 'pool',   landmark: 'max-pool, /2' },
    { from: 6,  to: 7,  type: 'conv'    },
    { from: 7,  to: 8,  type: 'conv'    },
    { from: 8,  to: 9,  type: 'upconv', landmark: 'transposed conv, ×2' },
    { from: 9,  to: 10, type: 'concat-merge' },
    { from: 10, to: 11, type: 'conv'    },
    { from: 11, to: 12, type: 'conv'    },
    { from: 12, to: 13, type: 'upconv', landmark: 'transposed conv, ×2' },
    { from: 13, to: 14, type: 'concat-merge' },
    { from: 14, to: 15, type: 'conv'    },
    { from: 15, to: 16, type: 'conv'    },
    { from: 16, to: 17, type: 'onexone', landmark: '1×1 conv → 5 classes' },
    { from: 17, to: 18, type: 'argmax'  },
  ];

  /* Skip connections. Scene 11 labels both with the concat formula. */
  const SKIPS = [
    { from: 2,  to: 14, label: 'skip · concat along channels (16+16=32)' },
    { from: 5,  to: 10, label: 'skip · concat along channels (32+32=64)' },
  ];

  /* ---------------------------------------------------------------
     Compute layout (X, Y per tensor) once. Pure function of constants.
     --------------------------------------------------------------- */
  function computeLayout() {
    const INTRA = 10;     // within a block
    const BLOCK = 22;     // between blocks within a row
    const ROW_Y = { 1: 150, 2: 330, 3: 480 };

    const pos = {};
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

    x = pos.t2.x + pos.t2.w + BLOCK;
    place('t3',  16, 32, 2, INTRA);
    place('t4',  32, 32, 2, INTRA);
    place('t5',  32, 32, 2, BLOCK);

    x = pos.t5.x + pos.t5.w + BLOCK;
    place('t6',  32, 16, 3, INTRA);
    place('t7',  64, 16, 3, INTRA);
    place('t8',  64, 16, 3, BLOCK);

    x = pos.t8.x + pos.t8.w + BLOCK;
    place('t9',  32, 32, 2, INTRA);
    place('t10', 64, 32, 2, INTRA);
    place('t11', 32, 32, 2, INTRA);
    place('t12', 32, 32, 2, BLOCK);

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
      root.innerHTML = '<p style="opacity:0.5">Scene 11: missing globals.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    if (!D.samples || !D.samples.length) {
      root.innerHTML = '<p style="opacity:0.5">Scene 11: no samples.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s11-root');
    const wrap = el('div', { class: 's11-wrap' }, root);

    /* ---- Hero ---------------------------------------------------- */
    const hero = el('header', { class: 'hero s11-hero' }, wrap);
    el('h1', { text: 'The U-Net, fully wired.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Every line of this diagram has been earned over the previous scenes. Here it is, fully labeled.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'The reference card. All bookends, hover, and arrow annotations are on from the start. ' +
        'Bar <em>height</em> encodes spatial size; bar <em>width</em> encodes channel count. ' +
        'Hover any bar for its shape and a peek at its feature maps; press ' +
        '<em>play forward</em> to replay the data flow.',
    }, hero);

    /* ---- Sample picker + play button ---------------------------- */
    const ctrl = el('div', { class: 's11-controls-top' }, wrap);
    const sampleRow = el('div', { class: 's11-sample-row' }, ctrl);
    el('span', { class: 's11-control-label', text: 'sample' }, sampleRow);
    const sampleBtns = [];
    for (let i = 0; i < D.samples.length; i++) {
      const b = el('button', {
        type: 'button', class: 's11-sample-btn', text: String(i + 1),
        'data-idx': String(i),
      }, sampleRow);
      sampleBtns.push(b);
    }
    const playBtn = el('button', {
      type: 'button', class: 's11-play-btn', text: '▶ play forward',
    }, sampleRow);

    /* ---- The diagram (SVG inside scrollable host) ---------------- */
    const layout = computeLayout();

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    Object.keys(layout.pos).forEach(function (k) {
      const p = layout.pos[k];
      if (p.x < minX) minX = p.x;
      if (p.x + p.w > maxX) maxX = p.x + p.w;
      if (p.y < minY) minY = p.y;
      if (p.y + p.h > maxY) maxY = p.y + p.h;
    });
    const PAD_LEFT = 18 + BOOKEND_PX + 20;
    const PAD_RIGHT = 18 + BOOKEND_PX + 20;
    const PAD_TOP = 22;
    const PAD_BOTTOM = 70;   // a few extra px for the landmark labels

    const SVG_W = maxX + PAD_LEFT + PAD_RIGHT;
    const SVG_H = maxY + PAD_TOP + PAD_BOTTOM;

    Object.keys(layout.pos).forEach(function (k) {
      const p = layout.pos[k];
      p.x += PAD_LEFT;
      p.y += PAD_TOP;
      p.cx += PAD_LEFT;
      p.cy += PAD_TOP;
    });

    const arch = el('div', { class: 's11-arch' }, wrap);
    // The reference scene starts in its fully-wired state. We keep the
    // class names that scene 0 uses for bookends/hover/sweep so the
    // stylesheet can reuse the same selectors (with the s11- prefix).
    arch.classList.add('s11-show-bookends');
    arch.classList.add('s11-show-hover');

    const svgHost = el('div', { class: 's11-svg-host' }, arch);

    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', '0 0 ' + Math.round(SVG_W) + ' ' + Math.round(SVG_H));
    svg.setAttribute('width',  String(Math.round(SVG_W)));
    svg.setAttribute('height', String(Math.round(SVG_H)));
    svg.setAttribute('class', 's11-svg');
    svgHost.appendChild(svg);

    const defs = svgEl('defs', null, svg);
    function makeMarker(id, color) {
      const m = svgEl('marker', {
        id: id, viewBox: '0 0 10 10', refX: '8', refY: '5',
        markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse',
      }, defs);
      svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: color }, m);
    }
    makeMarker('s11-mk-conv',    OP_COLORS.conv);
    makeMarker('s11-mk-pool',    OP_COLORS.pool);
    makeMarker('s11-mk-upconv',  OP_COLORS.upconv);
    makeMarker('s11-mk-onexone', OP_COLORS.onexone);
    makeMarker('s11-mk-argmax',  '#888');
    makeMarker('s11-mk-concat',  '#7d776c');
    makeMarker('s11-mk-skip-light', '#9e9789');
    makeMarker('s11-mk-skip-dark',  '#5f5b54');

    const gArrows = svgEl('g', { class: 's11-g-arrows' }, svg);
    const gSkips  = svgEl('g', { class: 's11-g-skips'  }, svg);
    const gBars   = svgEl('g', { class: 's11-g-bars'   }, svg);
    const gLabels = svgEl('g', { class: 's11-g-labels' }, svg);
    const gAnnot  = svgEl('g', { class: 's11-g-annot'  }, svg);
    const gLandmarks = svgEl('g', { class: 's11-g-landmarks' }, svg);

    const tensorBars = {};   // id → { group, hit }

    /* ---- Draw bars + labels ------------------------------------- */
    TENSORS.forEach(function (t, idx) {
      const p = layout.pos[t.id];
      const grp = svgEl('g', { class: 's11-bar-group', 'data-idx': String(idx), 'data-id': t.id }, gBars);

      if (t.split) {
        const halfW = p.w / 2;
        svgEl('rect', {
          class: 's11-bar s11-bar-up',
          x: String(p.x), y: String(p.y),
          width: String(halfW), height: String(p.h),
          'data-half': 'up',
        }, grp);
        svgEl('rect', {
          class: 's11-bar s11-bar-skip',
          x: String(p.x + halfW), y: String(p.y),
          width: String(halfW), height: String(p.h),
          'data-half': 'skip',
        }, grp);
        svgEl('rect', {
          class: 's11-bar-outline',
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
      } else {
        const cls =
          (t.id === 't0')  ? 's11-bar s11-bar-input' :
          (t.id === 't18') ? 's11-bar s11-bar-output' :
                             's11-bar s11-bar-act';
        svgEl('rect', {
          class: cls,
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
        svgEl('rect', {
          class: 's11-bar-outline',
          x: String(p.x), y: String(p.y),
          width: String(p.w), height: String(p.h),
        }, grp);
      }

      const hit = svgEl('rect', {
        class: 's11-bar-hit',
        x: String(p.x - 4),
        y: String(p.y - 22),
        width: String(p.w + 8),
        height: String(p.h + 60),
        fill: 'transparent',
        'data-idx': String(idx),
      }, grp);

      tensorBars[t.id] = { group: grp, hit: hit };

      svgEl('text', {
        class: 's11-label-channels',
        x: String(p.cx),
        y: String(p.y - 8),
        'text-anchor': 'middle',
        text: String(t.channels),
      }, gLabels);
      svgEl('text', {
        class: 's11-label-spatial',
        x: String(p.cx),
        y: String(p.y + p.h + 14),
        'text-anchor': 'middle',
        text: t.spatial + '²',
      }, gLabels);
      if (t.annotate) {
        svgEl('text', {
          class: 's11-label-annot',
          x: String(p.cx),
          y: String(p.y + p.h + 28),
          'text-anchor': 'middle',
          text: t.annotate,
        }, gAnnot);
      }
    });

    /* ---- Draw operation arrows --------------------------------- */
    /* Track how many landmark labels we've already placed for each
       op type so we can de-duplicate the text on the second one
       (e.g., the second pool gets no label — the first one already
       said "max-pool, /2"). Same for transposed conv. */
    const landmarkSeen = {};
    OPS.forEach(function (op) {
      const from = TENSORS[op.from], to = TENSORS[op.to];
      const pf = layout.pos[from.id], pt = layout.pos[to.id];
      const x1 = pf.x + pf.w;
      const y1 = pf.cy;
      const x2 = pt.x;
      const y2 = pt.cy;
      let pathD;
      let cls;
      let mk;
      let labelMidpoint = null;  // { x, y, dy } for landmark text
      if (op.type === 'conv') {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-conv';
        mk = 's11-mk-conv';
      } else if (op.type === 'pool') {
        pathD = 'M ' + x1 + ' ' + y1 +
                ' C ' + (x1 + 18) + ' ' + y1 + ', ' +
                        (x2 - 18) + ' ' + y2 + ', ' +
                        x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-pool';
        mk = 's11-mk-pool';
        labelMidpoint = { x: (x1 + x2) / 2, y: (y1 + y2) / 2, dx: 12, dy: -4, anchor: 'start' };
      } else if (op.type === 'upconv') {
        pathD = 'M ' + x1 + ' ' + y1 +
                ' C ' + (x1 + 18) + ' ' + y1 + ', ' +
                        (x2 - 18) + ' ' + y2 + ', ' +
                        x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-upconv';
        mk = 's11-mk-upconv';
        labelMidpoint = { x: (x1 + x2) / 2, y: (y1 + y2) / 2, dx: -12, dy: -4, anchor: 'end' };
      } else if (op.type === 'concat-merge') {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-concat';
        mk = 's11-mk-concat';
      } else if (op.type === 'onexone') {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-onexone';
        mk = 's11-mk-onexone';
        labelMidpoint = { x: (x1 + x2) / 2, y: y1 - 10, dx: 0, dy: 0, anchor: 'middle' };
      } else { // argmax
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
        cls = 's11-arrow s11-arrow-argmax';
        mk = 's11-mk-argmax';
      }
      svgEl('path', {
        class: cls,
        d: pathD,
        fill: 'none',
        'marker-end': 'url(#' + mk + ')',
      }, gArrows);

      // Landmark annotation: only place the FIRST occurrence of each
      // landmark op type, to avoid noise (we already have two pools
      // and two transposed convs, and they're identical operations).
      if (op.landmark && labelMidpoint && !landmarkSeen[op.type]) {
        landmarkSeen[op.type] = true;
        svgEl('text', {
          class: 's11-landmark s11-landmark-' + op.type,
          x: String(labelMidpoint.x + (labelMidpoint.dx || 0)),
          y: String(labelMidpoint.y + (labelMidpoint.dy || 0)),
          'text-anchor': labelMidpoint.anchor || 'middle',
          text: op.landmark,
        }, gLandmarks);
      }
    });

    /* ---- Draw skip arcs ----------------------------------------- */
    SKIPS.forEach(function (sk, kIdx) {
      const from = TENSORS[sk.from], to = TENSORS[sk.to];
      const pf = layout.pos[from.id], pt = layout.pos[to.id];
      const halfW = pt.w / 2;
      const x1 = pf.x + pf.w - 4;
      const y1 = pf.y - 4;
      const x2 = pt.x + halfW + 2;
      const y2 = pt.y - 4;
      const dx = x2 - x1;
      const peak = Math.max(20, 30 - kIdx * 6);
      const yc = Math.min(y1, y2) - peak;
      const cx1 = x1 + Math.max(80, dx * 0.25);
      const cx2 = x2 - Math.max(80, dx * 0.25);
      const pathD =
        'M ' + x1 + ' ' + y1 + ' ' +
        'C ' + cx1 + ' ' + yc + ', ' +
              cx2 + ' ' + yc + ', ' +
              x2  + ' ' + y2;
      svgEl('path', {
        class: 's11-arrow s11-arrow-skip',
        d: pathD,
        fill: 'none',
        'marker-end': 'url(#s11-mk-skip-light)',
      }, gSkips);
      // Label the FIRST skip arc (encoder1 → cat1) — because this scene
      // is a reference card we want to clearly call out the concat
      // semantics. The legend handles the rest.
      if (kIdx === 0) {
        svgEl('text', {
          class: 's11-arrow-label s11-landmark s11-landmark-skip',
          x: String((x1 + x2) / 2),
          y: String(yc - 6),
          'text-anchor': 'middle',
          text: 'concat along channels',
        }, gSkips);
      }
    });

    /* ---- Bookend hosts (HTML overlays on top of the SVG) -------- */
    function makeBookend(host, klass, x, y, px, label) {
      const node = el('div', { class: 's11-bookend ' + klass }, host);
      node.style.left   = (x - px / 2) + 'px';
      node.style.top    = (y - px / 2) + 'px';
      node.style.width  = px + 'px';
      node.style.height = px + 'px';
      const inner = el('div', { class: 's11-bookend-canvas' }, node);
      inner.style.width = px + 'px';
      inner.style.height = px + 'px';
      el('div', { class: 's11-bookend-label', text: label }, node);
      return inner;
    }

    const inputBook = makeBookend(
      svgHost, 's11-bookend-input',
      layout.pos.t0.x - 16 - BOOKEND_PX / 2,
      layout.pos.t0.cy,
      BOOKEND_PX,
      'real input'
    );
    const outputBook = makeBookend(
      svgHost, 's11-bookend-output',
      layout.pos.t18.x + layout.pos.t18.w + 16 + BOOKEND_PX / 2,
      layout.pos.t18.cy,
      BOOKEND_PX,
      'predicted seg'
    );

    /* ---- Legend (HTML overlay anchored inside the SVG host) ----- */
    const legend = el('div', { class: 's11-legend' }, svgHost);
    const legendLeft = layout.pos.t2.x + layout.pos.t2.w + 24;
    legend.style.left = legendLeft + 'px';
    legend.style.top  = '8px';
    function legendItem(swatchClass, label) {
      const row = el('div', { class: 's11-legend-row' }, legend);
      el('span', { class: 's11-legend-swatch ' + swatchClass }, row);
      el('span', { class: 's11-legend-label', text: label }, row);
    }
    el('div', { class: 's11-legend-title', text: 'operations' }, legend);
    legendItem('lg-conv',    'conv 3×3 + ReLU');
    legendItem('lg-pool',    'max-pool 2×2');
    legendItem('lg-upconv',  'transposed conv 2×2');
    legendItem('lg-skip',    'skip (long, dashed) + concat input (short)');
    legendItem('lg-onexone', 'conv 1×1');
    legendItem('lg-argmax',  'argmax (logits → label)');

    /* ---- Hover sidebar (always live) ---------------------------- */
    const sidebar = el('aside', { class: 's11-sidebar' }, arch);
    const sbHead = el('div', { class: 's11-sb-head' }, sidebar);
    const sbName = el('div', { class: 's11-sb-name', text: 'hover a tensor' }, sbHead);
    const sbOp   = el('div', { class: 's11-sb-op',   text: 'shape, op, and a peek at the activations' }, sbHead);
    const sbShape = el('div', { class: 's11-sb-shape' }, sidebar);
    const sbPreviewWrap = el('div', { class: 's11-sb-preview-wrap' }, sidebar);
    const sbPreviewSubtitle = el('div', { class: 's11-sb-preview-subtitle' }, sidebar);
    const sbHint = el('div', { class: 's11-sb-hint' }, sidebar);

    /* ---- Caption ------------------------------------------------ */
    const caption = el('p', { class: 'caption s11-caption' }, wrap);
    caption.textContent =
      'Reference card: bookends, hover, and the four landmark labels are on by default. ' +
      'Press play forward to replay the data flow in execution order.';

    /* ---- Step controls (single step → minimal nav) -------------- */
    // We keep a tiny controls row for parity with other scenes (reset
    // button + the play forward replay), but no step slider since
    // there's only one step.
    const controls = el('div', { class: 'controls s11-controls' }, wrap);
    const navGroup = el('div', { class: 'control-group' }, controls);
    const replayBtn = el('button', { type: 'button', class: 'primary', text: 'replay sweep' }, navGroup);
    const resetBtn  = el('button', { type: 'button', text: 'reset' }, navGroup);

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
        sbOp.textContent   = 'shape, op, and a peek at the activations';
        sbShape.textContent = '';
        sbPreviewWrap.innerHTML = '';
        sbPreviewSubtitle.textContent = '';
        sbHint.textContent = '';
        return;
      }
      const t = TENSORS[idx];
      sbName.textContent = t.name;
      sbOp.textContent   = t.op;
      sbShape.innerHTML  = '<span class="s11-sb-shape-label">shape</span> ' +
        t.spatial + ' × ' + t.spatial + ' × ' + t.channels;
      sbPreviewWrap.innerHTML = '';
      if (t.preview) {
        const stack = sample()[t.preview];
        if (stack && stack.length === 4) {
          const previewHost = el('div', { class: 's11-sb-preview' }, sbPreviewWrap);
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
      Object.keys(tensorBars).forEach(function (k) {
        tensorBars[k].group.classList.remove('s11-bar-hover');
      });
      state.hoveredIdx = idx;
      if (idx != null) {
        tensorBars[TENSORS[idx].id].group.classList.add('s11-bar-hover');
        renderPopover(idx);
      } else {
        renderPopover(null);
      }
    }

    TENSORS.forEach(function (t, idx) {
      const handle = tensorBars[t.id].hit;
      handle.addEventListener('mouseenter', function () {
        if (state.sweepTimer) return;
        setHover(idx);
      });
      handle.addEventListener('mouseleave', function () {
        if (state.sweepTimer) return;
        setHover(null);
      });
    });

    /* ---- Sweep -------------------------------------------------- */
    function clearSweep() {
      if (state.sweepTimer) { clearInterval(state.sweepTimer); state.sweepTimer = null; }
      state.sweepIdx = -1;
      arch.classList.remove('s11-show-sweep');
      Object.keys(tensorBars).forEach(function (k) {
        tensorBars[k].group.classList.remove('s11-bar-sweep');
        tensorBars[k].group.classList.remove('s11-bar-sweep-active');
      });
    }
    function startSweep() {
      clearSweep();
      arch.classList.add('s11-show-sweep');
      state.sweepIdx = 0;
      state.sweepTimer = setInterval(function () {
        if (state.sweepIdx >= TENSORS.length) {
          clearInterval(state.sweepTimer);
          state.sweepTimer = null;
          // After the sweep completes, drop the sweep dimming so the
          // diagram returns to its full reference-card state.
          setTimeout(function () {
            if (!state.sweepTimer) {
              arch.classList.remove('s11-show-sweep');
              Object.keys(tensorBars).forEach(function (k) {
                tensorBars[k].group.classList.remove('s11-bar-sweep');
                tensorBars[k].group.classList.remove('s11-bar-sweep-active');
              });
            }
          }, 700);
          return;
        }
        Object.keys(tensorBars).forEach(function (k) {
          tensorBars[k].group.classList.remove('s11-bar-sweep-active');
        });
        for (let i = 0; i < state.sweepIdx; i++) {
          tensorBars[TENSORS[i].id].group.classList.add('s11-bar-sweep');
        }
        const cur = TENSORS[state.sweepIdx];
        tensorBars[cur.id].group.classList.add('s11-bar-sweep');
        tensorBars[cur.id].group.classList.add('s11-bar-sweep-active');
        renderPopover(state.sweepIdx);
        state.sweepIdx++;
      }, SWEEP_STEP_MS);
    }

    playBtn.addEventListener('click', function () {
      clearSweep();
      startSweep();
    });
    replayBtn.addEventListener('click', function () {
      clearSweep();
      startSweep();
    });

    /* ---- Render ------------------------------------------------ */
    function render() {
      // Always paint bookends. Always show hover. Always show landmark
      // labels. The reference card is fully wired from the start.
      renderBookends();
    }

    resetBtn.addEventListener('click', function () {
      clearSweep();
      setHover(null);
      state.sampleIdx = pickRichest(D.samples);
      updateSampleBtns();
      renderBookends();
    });

    /* ---- Initial paint ----------------------------------------- */
    updateSampleBtns();
    render();

    /* ---- &run: trigger the play sweep automatically ------------ */
    if (readHashFlag('run')) {
      state.runTimer = setTimeout(function () {
        playBtn.click();
        state.runTimer = null;
      }, 600);
    }

    return {
      onEnter: function () {
        render();
      },
      onLeave: function () {
        if (state.runTimer) { clearTimeout(state.runTimer); state.runTimer = null; }
        clearSweep();
        setHover(null);
      },
      onNextKey: function () {
        // Single step — never consume the keystroke; let the deck
        // driver advance to the next scene.
        return false;
      },
      onPrevKey: function () {
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene11 = function (root) { return buildScene(root); };
})();
