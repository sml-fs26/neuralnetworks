/* U-Net JS utilities. Numerical helpers + the reusable mini-map.

   Mirrors `cnn-deepdive/js/cnn.js`'s naming conventions but is scoped to
   `window.UNET`. */
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

     A small Ronneberger-style diagram of the full U-Net (the same bar
     layout used in scene 0) with optional "you are here" highlights.
     Used as a sidebar in scenes that focus on one specific part of the
     network so the viewer always knows where in the architecture they are.

     mountUNetMiniMap(host, opts) -> { svg, setHighlight(keys, label?) }

     Highlight keys (any subset, passed as an array):
       'enc1', 'enc2', 'enc3' (= bottleneck), 'dec1', 'dec2', 'out'
       'pool1', 'pool2'   (the two max-pool operations)
       'up1', 'up2'       (the two transposed-conv upsamples)
       'skip1', 'skip2'   (the two skip-connection arcs)
       'bridge'           (alias of 'up2' — the bottleneck → up2 arrow)

     opts:
       width:   overall SVG width in px (default 280)
       height:  overall SVG height in px (default 130)
       title:   optional title shown above the diagram
       label:   optional caption shown below the diagram
  */

  /* The 19 tensors in execution order. Same set as scene 0; the mini-map
     renders all of them as bars so the silhouette matches scene 0 exactly. */
  const TENSORS = [
    { id: 't0',  block: 'input',  row: 1, spatial: 64, channels: 3,  op: 'input'   },
    { id: 't1',  block: 'enc1',   row: 1, spatial: 64, channels: 16, op: 'conv'    },
    { id: 't2',  block: 'enc1',   row: 1, spatial: 64, channels: 16, op: 'conv'    },
    { id: 't3',  block: 'pool',   row: 2, spatial: 32, channels: 16, op: 'pool'    },
    { id: 't4',  block: 'enc2',   row: 2, spatial: 32, channels: 32, op: 'conv'    },
    { id: 't5',  block: 'enc2',   row: 2, spatial: 32, channels: 32, op: 'conv'    },
    { id: 't6',  block: 'pool',   row: 3, spatial: 16, channels: 32, op: 'pool'    },
    { id: 't7',  block: 'enc3',   row: 3, spatial: 16, channels: 64, op: 'conv'    },
    { id: 't8',  block: 'enc3',   row: 3, spatial: 16, channels: 64, op: 'conv'    },
    { id: 't9',  block: 'up2',    row: 2, spatial: 32, channels: 32, op: 'upconv'  },
    { id: 't10', block: 'cat2',   row: 2, spatial: 32, channels: 64, op: 'concat'  },
    { id: 't11', block: 'dec2',   row: 2, spatial: 32, channels: 32, op: 'conv'    },
    { id: 't12', block: 'dec2',   row: 2, spatial: 32, channels: 32, op: 'conv'    },
    { id: 't13', block: 'up1',    row: 1, spatial: 64, channels: 16, op: 'upconv'  },
    { id: 't14', block: 'cat1',   row: 1, spatial: 64, channels: 32, op: 'concat'  },
    { id: 't15', block: 'dec1',   row: 1, spatial: 64, channels: 16, op: 'conv'    },
    { id: 't16', block: 'dec1',   row: 1, spatial: 64, channels: 16, op: 'conv'    },
    { id: 't17', block: 'out',    row: 1, spatial: 64, channels: 5,  op: 'onexone' },
    { id: 't18', block: 'output', row: 1, spatial: 64, channels: 1,  op: 'argmax'  },
  ];

  const OPS = [
    { id: 'op-conv-0',  from: 't0',  to: 't1',  type: 'conv'    },
    { id: 'op-conv-1',  from: 't1',  to: 't2',  type: 'conv'    },
    { id: 'op-pool1',   from: 't2',  to: 't3',  type: 'pool'    },
    { id: 'op-conv-3',  from: 't3',  to: 't4',  type: 'conv'    },
    { id: 'op-conv-4',  from: 't4',  to: 't5',  type: 'conv'    },
    { id: 'op-pool2',   from: 't5',  to: 't6',  type: 'pool'    },
    { id: 'op-conv-6',  from: 't6',  to: 't7',  type: 'conv'    },
    { id: 'op-conv-7',  from: 't7',  to: 't8',  type: 'conv'    },
    { id: 'op-up2',     from: 't8',  to: 't9',  type: 'upconv'  },
    { id: 'op-cat2',    from: 't9',  to: 't10', type: 'concat'  },
    { id: 'op-conv-10', from: 't10', to: 't11', type: 'conv'    },
    { id: 'op-conv-11', from: 't11', to: 't12', type: 'conv'    },
    { id: 'op-up1',     from: 't12', to: 't13', type: 'upconv'  },
    { id: 'op-cat1',    from: 't13', to: 't14', type: 'concat'  },
    { id: 'op-conv-14', from: 't14', to: 't15', type: 'conv'    },
    { id: 'op-conv-15', from: 't15', to: 't16', type: 'conv'    },
    { id: 'op-onexone', from: 't16', to: 't17', type: 'onexone' },
    { id: 'op-argmax',  from: 't17', to: 't18', type: 'argmax'  },
  ];
  const SKIPS = [
    { id: 'skip1', from: 't2', to: 't14' },
    { id: 'skip2', from: 't5', to: 't10' },
  ];

  /* Map a public key (e.g. 'enc1', 'pool2', 'skip1') to the set of tensor
     ids and arrow ids that should light up. */
  function expandKey(key) {
    const tensors = new Set();
    const arrows = new Set();
    const skips = new Set();
    switch (key) {
      case 'enc1':   tensors.add('t1'); tensors.add('t2'); break;
      case 'enc2':   tensors.add('t4'); tensors.add('t5'); break;
      case 'enc3':   tensors.add('t7'); tensors.add('t8'); break;
      case 'dec2':   tensors.add('t11'); tensors.add('t12'); break;
      case 'dec1':   tensors.add('t15'); tensors.add('t16'); break;
      case 'out':    tensors.add('t17'); tensors.add('t18'); arrows.add('op-onexone'); arrows.add('op-argmax'); break;
      case 'pool1':  tensors.add('t3'); arrows.add('op-pool1'); break;
      case 'pool2':  tensors.add('t6'); arrows.add('op-pool2'); break;
      case 'up2':
      case 'bridge': tensors.add('t9'); arrows.add('op-up2'); break;
      case 'up1':    tensors.add('t13'); arrows.add('op-up1'); break;
      case 'skip1':  skips.add('skip1'); tensors.add('t14'); break;
      case 'skip2':  skips.add('skip2'); tensors.add('t10'); break;
    }
    return { tensors, arrows, skips };
  }

  function svgEl(tag, attrs, parent) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  /* U-shape layout for the 19 tensors. Bottleneck row is centred; the
     encoder + decoder rows stair-step inward as you go down so the
     silhouette reads as a U. */
  function computeLayout(W, H) {
    const padX = 6, padTop = 18, padBottom = 8;
    const innerH = H - padTop - padBottom;
    const rowGap = 8;
    function barH(spatial) {
      const maxH = (innerH - 2 * rowGap) * 0.55;
      return Math.max(7, maxH * (spatial / 64));
    }
    function barW(ch) { return 3 + ch * 0.18; }
    const r1H = barH(64), r2H = barH(32), r3H = barH(16);
    const r1Y = padTop;
    const r2Y = r1Y + r1H + rowGap;
    const r3Y = r2Y + r2H + rowGap;
    const rowYTop = { 1: r1Y, 2: r2Y, 3: r3Y };
    const rowH = { 1: r1H, 2: r2H, 3: r3H };

    const leftIdsByRow  = { 1: ['t0','t1','t2'], 2: ['t3','t4','t5'] };
    const rightIdsByRow = { 1: ['t13','t14','t15','t16','t17','t18'],
                            2: ['t9','t10','t11','t12'] };
    const centerIdsRow3 = ['t6','t7','t8'];

    const indent = 22;
    const widthOf = id => barW(TENSORS.find(t => t.id === id).channels);

    const pos = {};
    for (const r of [1, 2]) {
      const sideIndent = indent * (r - 1);
      let x = padX + sideIndent;
      for (const id of leftIdsByRow[r]) {
        const t = TENSORS.find(tt => tt.id === id);
        const w = widthOf(id), h = barH(t.spatial);
        const y = rowYTop[r] + (rowH[r] - h) / 2;
        pos[id] = { x, y, w, h };
        x += w + 3;
      }
      x = W - padX - sideIndent;
      const rightIds = rightIdsByRow[r];
      for (let i = rightIds.length - 1; i >= 0; i--) {
        const id = rightIds[i];
        const t = TENSORS.find(tt => tt.id === id);
        const w = widthOf(id), h = barH(t.spatial);
        x -= w;
        const y = rowYTop[r] + (rowH[r] - h) / 2;
        pos[id] = { x, y, w, h };
        x -= 3;
      }
    }
    let totalW3 = 0;
    for (const id of centerIdsRow3) totalW3 += widthOf(id) + 3;
    totalW3 -= 3;
    let x3 = (W - totalW3) / 2;
    for (const id of centerIdsRow3) {
      const t = TENSORS.find(tt => tt.id === id);
      const w = widthOf(id), h = barH(t.spatial);
      const y = rowYTop[3] + (rowH[3] - h) / 2;
      pos[id] = { x: x3, y, w, h };
      x3 += w + 3;
    }
    return pos;
  }

  function mountUNetMiniMap(host, opts) {
    opts = opts || {};
    const W = opts.width || 280;
    const H = opts.height || 130;

    host.innerHTML = '';
    host.classList.add('unet-mini');

    let titleEl = null;
    if (opts.title) {
      titleEl = document.createElement('div');
      titleEl.className = 'unet-mini-title';
      titleEl.textContent = opts.title;
      host.appendChild(titleEl);
    }

    const svg = svgEl('svg', {
      xmlns: 'http://www.w3.org/2000/svg',
      viewBox: '0 0 ' + W + ' ' + H,
      width: String(W),
      height: String(H),
      class: 'unet-mini-svg',
    });
    host.appendChild(svg);

    let captionEl = null;
    if (opts.label !== undefined) {
      captionEl = document.createElement('div');
      captionEl.className = 'unet-mini-label';
      captionEl.textContent = opts.label || '';
      host.appendChild(captionEl);
    }

    const pos = computeLayout(W, H);

    // Three SVG groups so we can layer skip arcs under arrows under bars.
    const gSkips = svgEl('g', { class: 'unet-mini-g-skips' }, svg);
    const gArrows = svgEl('g', { class: 'unet-mini-g-arrows' }, svg);
    const gBars = svgEl('g', { class: 'unet-mini-g-bars' }, svg);
    const gLabels = svgEl('g', { class: 'unet-mini-g-labels' }, svg);

    // Skip arcs (under everything).
    const skipNodes = {};
    for (const sk of SKIPS) {
      const a = pos[sk.from], b = pos[sk.to];
      const ax = a.x + a.w, ay = a.y + 1;
      const bx = b.x, by = b.y + 1;
      const peak = Math.min(ay, by) - 8;
      const path = svgEl('path', {
        d: 'M ' + ax + ' ' + ay +
           ' C ' + (ax + 12) + ' ' + peak + ', ' + (bx - 12) + ' ' + peak + ', ' + bx + ' ' + by,
        class: 'unet-mini-arrow unet-mini-skip',
        'data-arrow-id': sk.id,
      }, gSkips);
      skipNodes[sk.id] = path;
    }

    // Operation arrows.
    const arrowNodes = {};
    for (const op of OPS) {
      const a = pos[op.from], b = pos[op.to];
      const x1 = a.x + a.w, y1 = a.y + a.h / 2;
      const x2 = b.x,       y2 = b.y + b.h / 2;
      let pathD;
      if (op.type === 'pool' || op.type === 'upconv') {
        pathD = 'M ' + x1 + ' ' + y1 +
                ' C ' + (x1 + 6) + ' ' + y1 + ', ' +
                        (x2 - 6) + ' ' + y2 + ', ' +
                        x2 + ' ' + y2;
      } else {
        pathD = 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
      }
      const path = svgEl('path', {
        d: pathD,
        class: 'unet-mini-arrow unet-mini-arrow-' + op.type,
        'data-arrow-id': op.id,
      }, gArrows);
      arrowNodes[op.id] = path;
    }

    // Tensor bars (on top).
    const tensorNodes = {};
    for (const t of TENSORS) {
      const p = pos[t.id];
      const g = svgEl('g', { 'data-tensor-id': t.id }, gBars);
      // Concat tensors are drawn as two stacked halves (upsample blue
      // half on top, skip amber half on bottom) — the same convention as
      // scene 0.
      if (t.op === 'concat') {
        const halfH = p.h / 2;
        svgEl('rect', {
          x: p.x, y: p.y, width: p.w, height: halfH,
          class: 'unet-mini-bar unet-mini-bar-tensor',
        }, g);
        svgEl('rect', {
          x: p.x, y: p.y + halfH, width: p.w, height: halfH,
          class: 'unet-mini-bar unet-mini-bar-skip-half',
        }, g);
      } else {
        let cls = 'unet-mini-bar unet-mini-bar-tensor';
        if (t.op === 'argmax') cls = 'unet-mini-bar unet-mini-bar-argmax';
        else if (t.op === 'onexone') cls = 'unet-mini-bar unet-mini-bar-out';
        svgEl('rect', {
          x: p.x, y: p.y, width: p.w, height: p.h, class: cls,
        }, g);
      }
      tensorNodes[t.id] = g;
    }

    // Highlight overlay layer (drawn over bars when active).
    const gHighlight = svgEl('g', { class: 'unet-mini-g-highlight' }, svg);

    // Landmark labels: enc1, enc2, bottleneck, dec2, dec1.
    const LABELS = [
      { id: 't2',  text: 'enc1' },
      { id: 't5',  text: 'enc2' },
      { id: 't8',  text: 'bottleneck' },
      { id: 't12', text: 'dec2' },
      { id: 't16', text: 'dec1' },
    ];
    for (const lbl of LABELS) {
      const p = pos[lbl.id];
      const tx = p.x + p.w / 2;
      const ty = p.y - 4;
      const text = svgEl('text', {
        x: String(tx), y: String(ty),
        'text-anchor': 'middle',
        class: 'unet-mini-label-text',
      }, gLabels);
      text.textContent = lbl.text;
    }

    /* setHighlight: pass the public keys (e.g. 'enc3', 'pool1', 'skip2'),
       optionally update the caption beneath. Internal mapping turns each
       key into the set of tensors / arrows / skips that should light up. */
    function setHighlight(keys, label) {
      const ks = Array.isArray(keys) ? keys : (keys ? [keys] : []);
      const litTensors = new Set();
      const litArrows = new Set();
      const litSkips = new Set();
      for (const k of ks) {
        const exp = expandKey(k);
        for (const t of exp.tensors) litTensors.add(t);
        for (const a of exp.arrows) litArrows.add(a);
        for (const s of exp.skips) litSkips.add(s);
      }
      const anyLit = litTensors.size + litArrows.size + litSkips.size > 0;

      // Bars: dim the unlit ones; clear any prior highlight overlays.
      while (gHighlight.firstChild) gHighlight.removeChild(gHighlight.firstChild);
      for (const t of TENSORS) {
        const g = tensorNodes[t.id];
        g.classList.toggle('unet-mini-dim', anyLit && !litTensors.has(t.id));
      }
      // Arrows: same treatment.
      for (const op of OPS) {
        const path = arrowNodes[op.id];
        path.classList.toggle('unet-mini-lit', litArrows.has(op.id));
        path.classList.toggle('unet-mini-dim', anyLit && !litArrows.has(op.id));
      }
      // Skip arcs.
      for (const sk of SKIPS) {
        const path = skipNodes[sk.id];
        path.classList.toggle('unet-mini-lit', litSkips.has(sk.id));
        path.classList.toggle('unet-mini-dim', anyLit && !litSkips.has(sk.id));
      }
      // Glow + ring overlays for lit tensors.
      for (const id of litTensors) {
        const p = pos[id];
        svgEl('rect', {
          x: p.x - 1.5, y: p.y - 1.5,
          width: p.w + 3, height: p.h + 3, rx: 2,
          class: 'unet-mini-bar-glow',
        }, gHighlight);
        svgEl('rect', {
          x: p.x - 1.2, y: p.y - 1.2,
          width: p.w + 2.4, height: p.h + 2.4, rx: 2,
          class: 'unet-mini-bar-ring',
        }, gHighlight);
      }
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
