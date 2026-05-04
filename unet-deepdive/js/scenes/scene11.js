/* Scene 11 — "The U-Net, fully wired."

   Synthesis scene. The same U-shape we have grown across the deepdive,
   now annotated on every edge with shape (H × W × C) and on every node
   with the operation it performs. Hover any node to see its formula
   (KaTeX). Hover any skip arc to see the concat arithmetic. A "play
   forward" button sweeps a real input through the architecture.

   Reuses the U-shape skeleton + layoutArrows + makeLevelCard pattern
   from cnn-deepdive scene9 (which is also where the painters were
   promoted from into js/drawing.js). The new bits: full edge labels,
   per-node hover with KaTeX formulas, per-skip hover with concat math,
   "play forward" sweep across all 6 stages.

   Step engine:
     0 = static diagram, blank cards, no labels
     1 = shape labels appear on edges
     2 = operation labels on nodes
     3 = hover mode enabled
     4 = "play forward" sweep (auto)
*/
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 700;
  const SWEEP_STEP_MS = 600;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

  const LEVEL1_PX = 132;
  const LEVEL2_PX = 100;
  const LEVEL3_PX = 80;
  const OUTPUT_PX = 132;
  const INPUT_PX  = 132;

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

  /* Per-node operation descriptions.  Each entry has:
       title : short name for the tooltip header
       shape : "H × W × C" string
       op    : caption-like English line
       tex   : KaTeX formula (display mode)
  */
  const NODE_INFO = {
    input: {
      title: 'input',
      shape: '64 × 64 × 3',
      op: 'RGB image, 3 channels in [0, 1].',
      tex: 'x \\in \\mathbb{R}^{64 \\times 64 \\times 3}',
    },
    enc1: {
      title: 'enc1 — conv block',
      shape: '64 × 64 × 16',
      op: '2× (Conv 3×3, ReLU). Resolution unchanged, channels 3 → 16.',
      tex: '\\mathrm{enc}_1 = \\mathrm{ReLU}(W_{1b} * \\mathrm{ReLU}(W_{1a} * x + b_{1a}) + b_{1b})',
    },
    pool1: {
      title: 'pool1 — max-pool 2×2',
      shape: '32 × 32 × 16',
      op: 'MaxPool with stride 2. Halves H and W; channels untouched.',
      tex: '\\mathrm{pool}_1[i, j, c] = \\max_{u, v \\in \\{0,1\\}} \\mathrm{enc}_1[2i + u, 2j + v, c]',
    },
    enc2: {
      title: 'enc2 — conv block',
      shape: '32 × 32 × 32',
      op: '2× (Conv 3×3, ReLU). Channels 16 → 32.',
      tex: '\\mathrm{enc}_2 = \\mathrm{ConvBlock}_{16 \\to 32}(\\mathrm{pool}_1)',
    },
    pool2: {
      title: 'pool2 — max-pool 2×2',
      shape: '16 × 16 × 32',
      op: 'Halve again. We are at the bottleneck resolution.',
      tex: '\\mathrm{pool}_2[i, j, c] = \\max_{u, v \\in \\{0,1\\}} \\mathrm{enc}_2[2i + u, 2j + v, c]',
    },
    enc3: {
      title: 'enc3 — bottleneck',
      shape: '16 × 16 × 64',
      op: 'Deepest features. Each cell looks at a ~30×30 input patch.',
      tex: '\\mathrm{enc}_3 = \\mathrm{ConvBlock}_{32 \\to 64}(\\mathrm{pool}_2)',
    },
    up2: {
      title: 'up2 — transposed conv (stride 2)',
      shape: '32 × 32 × 32',
      op: 'Learned upsample. Doubles H, W; channels 64 → 32.',
      tex: '\\mathrm{up}_2 = W_{u2}^{\\top} \\circledast \\mathrm{enc}_3',
    },
    cat2: {
      title: 'concat with enc2',
      shape: '32 × 32 × 64',
      op: 'Stack channels: skip path (32) + upsample path (32).',
      tex: '\\mathrm{cat}_2 = \\bigl[\\mathrm{up}_2 \\,\\Vert\\, \\mathrm{enc}_2\\bigr] \\;\\;(32+32 = 64\\ \\text{channels})',
    },
    dec2: {
      title: 'dec2 — conv block',
      shape: '32 × 32 × 32',
      op: '2× (Conv 3×3, ReLU). Mixes the upsample and the skip.',
      tex: '\\mathrm{dec}_2 = \\mathrm{ConvBlock}_{64 \\to 32}(\\mathrm{cat}_2)',
    },
    up1: {
      title: 'up1 — transposed conv (stride 2)',
      shape: '64 × 64 × 16',
      op: 'Back to full resolution. Channels 32 → 16.',
      tex: '\\mathrm{up}_1 = W_{u1}^{\\top} \\circledast \\mathrm{dec}_2',
    },
    cat1: {
      title: 'concat with enc1',
      shape: '64 × 64 × 32',
      op: 'Stack channels: skip path (16) + upsample path (16).',
      tex: '\\mathrm{cat}_1 = \\bigl[\\mathrm{up}_1 \\,\\Vert\\, \\mathrm{enc}_1\\bigr] \\;\\;(16+16 = 32\\ \\text{channels})',
    },
    dec1: {
      title: 'dec1 — conv block',
      shape: '64 × 64 × 16',
      op: '2× (Conv 3×3, ReLU). Final feature map at full resolution.',
      tex: '\\mathrm{dec}_1 = \\mathrm{ConvBlock}_{32 \\to 16}(\\mathrm{cat}_1)',
    },
    head: {
      title: '1×1 conv → 5 logits → softmax',
      shape: '64 × 64 × 5',
      op: 'Per-pixel 5-way classification.',
      tex: '\\hat{y}[i,j,k] = \\mathrm{softmax}_k\\bigl(W_{\\mathrm{out}} \\cdot \\mathrm{dec}_1[i,j]\\bigr)',
    },
    output: {
      title: 'argmax',
      shape: '64 × 64',
      op: 'Pick the highest-probability class at every pixel.',
      tex: 'y[i,j] = \\arg\\max_k \\hat{y}[i,j,k]',
    },
  };

  /* Skip arcs: source/dest + concat formula. */
  const SKIP_INFO = {
    skip1: {
      title: 'skip 1 — enc1 → dec1',
      from: 'enc1 (64 × 64 × 16)',
      to:   'cat1 (64 × 64 × 32)',
      op:   'Concatenate along the channel dimension.',
      tex:  '\\mathrm{cat}_1 = \\bigl[\\mathrm{up}_1 \\,\\Vert\\, \\mathrm{enc}_1\\bigr] \\quad 16 + 16 = 32',
    },
    skip2: {
      title: 'skip 2 — enc2 → dec2',
      from: 'enc2 (32 × 32 × 32)',
      to:   'cat2 (32 × 32 × 64)',
      op:   'Concatenate along the channel dimension.',
      tex:  '\\mathrm{cat}_2 = \\bigl[\\mathrm{up}_2 \\,\\Vert\\, \\mathrm{enc}_2\\bigr] \\quad 32 + 32 = 64',
    },
  };

  /* The temporal order in which the "play forward" sweep lights up cards. */
  const SWEEP_NODES = [
    'enc1', 'enc2', 'enc3', 'dec2', 'dec1', 'output',
  ];

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;

    root.innerHTML = '';
    root.classList.add('s11-root');
    const wrap = el('div', { class: 's11-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s11-hero' }, wrap);
    el('h1', { text: 'The U-Net, fully wired.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Every line in this diagram has been earned. Now they all fit on one page.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Hover any <em>node</em> to see the operation formula. Hover any <em>skip arc</em> ' +
        'to see the concat arithmetic. The "play forward" button runs a real input through it.',
    }, hero);

    /* ---- The architecture diagram ----------------------------------- */
    const arch = el('div', { class: 's11-arch' }, wrap);

    /* Sample selector (small inline strip). */
    const selRow = el('div', { class: 's11-selector' }, wrap);
    el('span', { class: 's11-selector-label', text: 'sample' }, selRow);
    const selBtns = [];
    for (let i = 0; i < D.samples.length; i++) {
      const b = el('button', {
        type: 'button', class: 's11-sample-btn', text: String(i + 1),
        'data-idx': String(i),
      }, selRow);
      selBtns.push(b);
      b.addEventListener('click', function () {
        state.sampleIdx = i;
        updateSampleBtns();
        renderCardContents();
      });
    }
    /* "Play forward" button lives on the same row. */
    el('span', { class: 's11-selector-spacer' }, selRow);
    const playBtn = el('button', {
      type: 'button', class: 's11-play-btn primary', text: '▶ play forward',
    }, selRow);

    /* SVG overlay for arrows. We draw 4 flow arrows (pool1, pool2,
       up2, up1), 2 skip arcs, the head arrow (dec1 → output). */
    const arrowsSvg = el('div', { class: 's11-arrows' }, arch);
    arrowsSvg.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<defs>' +
        '<marker id="s11-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
          '<path d="M 0 0 L 10 5 L 0 10 z" />' +
        '</marker>' +
      '</defs>' +
      // Skip arcs
      '<g class="s11-skip s11-skip1" data-key="skip1">' +
        '<path class="s11-skip-path s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<path class="s11-skip-hit s11-edge-hit" fill="none" stroke-width="22" stroke="transparent" />' +
        '<text class="s11-edge-label s11-skip-label" text-anchor="middle">' +
          'skip · 16 + 16 = 32</text>' +
      '</g>' +
      '<g class="s11-skip s11-skip2" data-key="skip2">' +
        '<path class="s11-skip-path s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<path class="s11-skip-hit s11-edge-hit" fill="none" stroke-width="22" stroke="transparent" />' +
        '<text class="s11-edge-label s11-skip-label" text-anchor="middle">' +
          'skip · 32 + 32 = 64</text>' +
      '</g>' +
      // Down arrows
      '<g class="s11-flow s11-flow-d1">' +
        '<path class="s11-down s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<text class="s11-edge-label s11-down-label" text-anchor="start">pool · 64 → 32</text>' +
      '</g>' +
      '<g class="s11-flow s11-flow-d2">' +
        '<path class="s11-down s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<text class="s11-edge-label s11-down-label" text-anchor="start">pool · 32 → 16</text>' +
      '</g>' +
      // Up arrows
      '<g class="s11-flow s11-flow-u1">' +
        '<path class="s11-up s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<text class="s11-edge-label s11-up-label" text-anchor="start">up2 · 16 → 32</text>' +
      '</g>' +
      '<g class="s11-flow s11-flow-u2">' +
        '<path class="s11-up s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<text class="s11-edge-label s11-up-label" text-anchor="start">up1 · 32 → 64</text>' +
      '</g>' +
      // Head arrow (dec1 → output)
      '<g class="s11-flow s11-flow-head">' +
        '<path class="s11-head-path s11-edge-path" fill="none" marker-end="url(#s11-arrow)" />' +
        '<text class="s11-edge-label s11-head-label" text-anchor="start">' +
          '1×1 conv + softmax · 16 → 5</text>' +
      '</g>' +
      '</svg>';

    /* Three columns: encoder (left), bottleneck (centre-bottom), decoder (right).
       Plus an input card on the far left and an output card on the far right. */
    const inCol = el('div', { class: 's11-col s11-col-in' }, arch);
    const inputCard = makeIOCard(inCol, 'input', '64×64×3', INPUT_PX, 'input');

    const encCol = el('div', { class: 's11-col s11-col-enc' }, arch);
    const enc1Card = makeLevelCard(encCol, 'enc1', '64×64×16', LEVEL1_PX, 'enc1');
    const enc2Card = makeLevelCard(encCol, 'enc2', '32×32×32', LEVEL2_PX, 'enc2');

    const botCol = el('div', { class: 's11-col s11-col-bot' }, arch);
    const enc3Card = makeLevelCard(botCol, 'enc3', '16×16×64 · bottleneck', LEVEL3_PX, 'enc3');

    const decCol = el('div', { class: 's11-col s11-col-dec' }, arch);
    const dec1Card = makeLevelCard(decCol, 'dec1', '64×64×16', LEVEL1_PX, 'dec1');
    const dec2Card = makeLevelCard(decCol, 'dec2', '32×32×32', LEVEL2_PX, 'dec2');

    const outCol = el('div', { class: 's11-col s11-col-out' }, arch);
    const outputCard = makeIOCard(outCol, 'output', '64×64 · argmax', OUTPUT_PX, 'output');

    /* Tooltip that follows the cursor for nodes/edges. */
    const tooltip = el('div', { class: 's11-tooltip', role: 'tooltip' }, root);
    tooltip.style.display = 'none';
    const ttHead = el('div', { class: 's11-tt-head' }, tooltip);
    el('div', { class: 's11-tt-shape' }, tooltip);
    el('div', { class: 's11-tt-op' }, tooltip);
    const ttFormula = el('div', { class: 's11-tt-formula' }, tooltip);

    /* ---- Class legend ----------------------------------------------- */
    const legend = el('div', { class: 's11-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's11-legend-item' }, legend);
      el('span', { class: 's11-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's11-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Caption + step controls ----------------------------------- */
    const caption = el('p', { class: 'caption s11-caption' }, wrap);

    const controls = el('div', { class: 'controls s11-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's11-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's11-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ------------------------------------------------------ */
    const state = {
      step: 0,
      sampleIdx: 0,
      sweep: -1,        // index into SWEEP_NODES; -1 means no sweep active
      sweepTimer: null,
      runTimer: null,
    };

    function updateSampleBtns() {
      for (let i = 0; i < selBtns.length; i++) {
        selBtns[i].classList.toggle('active', i === state.sampleIdx);
      }
    }

    /* All level/IO cards keyed by their NODE_INFO key. */
    const cardMap = {
      input:  inputCard,
      enc1:   enc1Card,
      enc2:   enc2Card,
      enc3:   enc3Card,
      dec1:   dec1Card,
      dec2:   dec2Card,
      output: outputCard,
    };

    /* Paint every card's contents according to current sample. We always
       show contents (this is the synthesis scene); step 0 just dims the
       labels, not the data. */
    function renderCardContents() {
      const sample = D.samples[state.sampleIdx];
      window.Drawing.paintRGB(inputCard.body, sample.input, INPUT_PX);
      window.Drawing.paintFeatureCard(enc1Card.body, sample.enc1, LEVEL1_PX);
      window.Drawing.paintFeatureCard(enc2Card.body, sample.enc2, LEVEL2_PX);
      window.Drawing.paintFeatureCard(enc3Card.body, sample.enc3, LEVEL3_PX);
      window.Drawing.paintFeatureCard(dec2Card.body, sample.dec2, LEVEL2_PX);
      window.Drawing.paintFeatureCard(dec1Card.body, sample.dec1, LEVEL1_PX);
      window.Drawing.paintLabelMap(outputCard.body, sample.pred, OUTPUT_PX);
    }

    /* Compute the SVG arrow paths from real DOM positions. Same approach
       as cnn-deepdive scene9's layoutArrows(). */
    function layoutArrows() {
      const archRect = arch.getBoundingClientRect();
      if (!archRect.width || !archRect.height) return;

      const svg = arrowsSvg.querySelector('svg');
      svg.setAttribute('viewBox',
        '0 0 ' + Math.round(archRect.width) + ' ' + Math.round(archRect.height));
      svg.setAttribute('width', String(Math.round(archRect.width)));
      svg.setAttribute('height', String(Math.round(archRect.height)));

      function rectOf(node) {
        const r = node.getBoundingClientRect();
        return {
          x: r.left - archRect.left,
          y: r.top - archRect.top,
          w: r.width,
          h: r.height,
          right: r.right - archRect.left,
          bottom: r.bottom - archRect.top,
          cx: (r.left + r.right) / 2 - archRect.left,
          cy: (r.top + r.bottom) / 2 - archRect.top,
        };
      }

      const rIn = rectOf(inputCard.card);
      const r1e = rectOf(enc1Card.card);
      const r2e = rectOf(enc2Card.card);
      const r3  = rectOf(enc3Card.card);
      const r2d = rectOf(dec2Card.card);
      const r1d = rectOf(dec1Card.card);
      const rOut = rectOf(outputCard.card);

      function setPathOn(group, d) {
        const ps = group.querySelectorAll('path');
        for (let i = 0; i < ps.length; i++) ps[i].setAttribute('d', d);
      }
      function setText(group, x, y, s) {
        const t = group.querySelector('text');
        t.setAttribute('x', String(x));
        t.setAttribute('y', String(y));
        if (s != null) t.textContent = s;
      }

      const skip1 = arrowsSvg.querySelector('.s11-skip1');
      const skip2 = arrowsSvg.querySelector('.s11-skip2');
      const flowD1 = arrowsSvg.querySelector('.s11-flow-d1');
      const flowD2 = arrowsSvg.querySelector('.s11-flow-d2');
      const flowU1 = arrowsSvg.querySelector('.s11-flow-u1');
      const flowU2 = arrowsSvg.querySelector('.s11-flow-u2');
      const flowHead = arrowsSvg.querySelector('.s11-flow-head');

      // skip1: enc1 right -> dec1 left, arc bowing high.
      const s1ax = r1e.right - 6;
      const s1ay = r1e.y - 8;
      const s1bx = r1d.x + 6;
      const s1by = r1d.y - 8;
      const peak1 = 24;
      const Yc1 = (8 * peak1 - s1ay - s1by) / 6;
      const c1 = Math.max(160, (s1bx - s1ax) * 0.26);
      const d1 = 'M ' + s1ax + ' ' + s1ay + ' ' +
        'C ' + (s1ax + c1) + ' ' + Yc1 + ', ' +
              (s1bx - c1) + ' ' + Yc1 + ', ' +
              s1bx + ' ' + s1by;
      setPathOn(skip1, d1);
      setText(skip1, (s1ax + s1bx) / 2, peak1 - 8, 'skip · 16 + 16 = 32');

      // skip2: enc2 right -> dec2 left, arc bowing into the centre dead zone.
      const s2ax = r2e.right;
      const s2ay = r2e.cy;
      const s2bx = r2d.x;
      const s2by = r2d.cy;
      const peak2 = (r1e.bottom + r2e.y) / 2 - 8;
      const Yc2 = (8 * peak2 - s2ay - s2by) / 6;
      const c2 = Math.max(140, (s2bx - s2ax) * 0.28);
      const d2 = 'M ' + s2ax + ' ' + s2ay + ' ' +
        'C ' + (s2ax + c2) + ' ' + Yc2 + ', ' +
              (s2bx - c2) + ' ' + Yc2 + ', ' +
              s2bx + ' ' + s2by;
      setPathOn(skip2, d2);
      setText(skip2, (s2ax + s2bx) / 2, peak2 - 4, 'skip · 32 + 32 = 64');

      // pool1: enc1 bottom centre -> enc2 top centre
      setPathOn(flowD1,
        'M ' + r1e.cx + ' ' + r1e.bottom + ' ' +
        'L ' + r2e.cx + ' ' + r2e.y);
      setText(flowD1, r1e.cx + 8, (r1e.bottom + r2e.y) / 2 + 4, 'pool · 64 → 32');

      // pool2: enc2 bottom -> enc3 top, slightly curved (enc3 is centred lower)
      const d3ax = r2e.cx;
      const d3ay = r2e.bottom;
      const d3bx = r3.x + 6;
      const d3by = r3.y;
      setPathOn(flowD2,
        'M ' + d3ax + ' ' + d3ay + ' ' +
        'C ' + d3ax + ' ' + (d3ay + 30) + ', ' +
              d3bx + ' ' + (d3by - 20) + ', ' +
              d3bx + ' ' + d3by);
      setText(flowD2, (d3ax + d3bx) / 2 - 28, (d3ay + d3by) / 2 + 16, 'pool · 32 → 16');

      // up2: enc3 top-right -> dec2 bottom-centre
      const u1ax = r3.right - 6;
      const u1ay = r3.y;
      const u1bx = r2d.cx;
      const u1by = r2d.bottom;
      setPathOn(flowU1,
        'M ' + u1ax + ' ' + u1ay + ' ' +
        'C ' + u1ax + ' ' + (u1ay - 20) + ', ' +
              u1bx + ' ' + (u1by + 30) + ', ' +
              u1bx + ' ' + u1by);
      setText(flowU1, (u1ax + u1bx) / 2 + 6, (u1ay + u1by) / 2 + 16, 'up2 · 16 → 32');

      // up1: dec2 top centre -> dec1 bottom centre.
      // Shorter label so it fits in the small gap between the two cards.
      setPathOn(flowU2,
        'M ' + r2d.cx + ' ' + r2d.y + ' ' +
        'L ' + r1d.cx + ' ' + r1d.bottom);
      setText(flowU2, r1d.cx + 8, (r1d.bottom + r2d.y) / 2 + 4, 'up1 · 32 → 64');

      // head: dec1 right -> output left (both at top row).
      // Place the label ABOVE both cards so it does not cover either.
      setPathOn(flowHead,
        'M ' + r1d.right + ' ' + r1d.cy + ' ' +
        'L ' + rOut.x + ' ' + rOut.cy);
      const headMidX = (r1d.right + rOut.x) / 2;
      const headTextY = Math.max(12, Math.min(r1d.y, rOut.y) - 6);
      const flowHeadText = flowHead.querySelector('text');
      flowHeadText.setAttribute('text-anchor', 'middle');
      setText(flowHead, headMidX, headTextY, '1×1 conv + softmax · 16 → 5');

      // input -> enc1 (no SVG arrow needed; just a CSS pseudo). For
      // simplicity we omit it; the layout makes the connection obvious.
    }

    /* ---- Tooltip behaviour ----------------------------------------- */

    function showTooltip(info, pageX, pageY) {
      ttHead.textContent = info.title;
      tooltip.querySelector('.s11-tt-shape').textContent = info.shape || '';
      tooltip.querySelector('.s11-tt-op').textContent = info.op || '';
      ttFormula.innerHTML = '';
      if (info.tex) {
        window.Katex.render(info.tex, ttFormula, true);
      }
      tooltip.style.display = '';
      positionTooltip(pageX, pageY);
    }
    function hideTooltip() {
      tooltip.style.display = 'none';
    }
    function positionTooltip(pageX, pageY) {
      // Place tooltip near the cursor but constrained to the viewport.
      const pad = 14;
      tooltip.style.left = '0px';
      tooltip.style.top = '0px';
      // Force layout so we get the right size.
      const w = tooltip.offsetWidth;
      const h = tooltip.offsetHeight;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      let x = pageX + pad;
      let y = pageY + pad;
      if (x + w > vw - 10) x = pageX - w - pad;
      if (y + h > vh - 10) y = pageY - h - pad;
      if (x < 10) x = 10;
      if (y < 10) y = 10;
      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    }

    function attachHover(target, infoLookup) {
      target.addEventListener('mouseenter', function (e) {
        if (state.step < 3) return;
        showTooltip(infoLookup(), e.clientX, e.clientY);
      });
      target.addEventListener('mousemove', function (e) {
        if (state.step < 3) return;
        positionTooltip(e.clientX, e.clientY);
      });
      target.addEventListener('mouseleave', function () {
        hideTooltip();
      });
    }

    // Wire hover for cards.
    Object.keys(cardMap).forEach(function (k) {
      attachHover(cardMap[k].card, function () { return NODE_INFO[k]; });
    });
    // Wire hover for skip arcs (the .s11-edge-hit acts as the wide hit area).
    arrowsSvg.querySelectorAll('.s11-skip').forEach(function (g) {
      const key = g.getAttribute('data-key');
      attachHover(g, function () { return SKIP_INFO[key]; });
    });

    /* ---- Step / sweep logic ---------------------------------------- */

    function captionFor(step) {
      switch (step) {
        case 0: return 'The U-Net at rest. Six cards, two skip arcs, one head.';
        case 1: return 'Edge labels: every arrow knows what shape it carries.';
        case 2: return 'Node labels: every card knows what operation it performs.';
        case 3: return 'Hover any card or arc for the formula and the shape arithmetic.';
        case 4: return 'Play forward: the input flows left to right, enc → bottleneck → dec → out.';
        default: return '';
      }
    }

    function clearSweep() {
      if (state.sweepTimer) { clearInterval(state.sweepTimer); state.sweepTimer = null; }
      state.sweep = -1;
      arch.querySelectorAll('.s11-card.s11-sweep').forEach(function (c) {
        c.classList.remove('s11-sweep');
      });
    }

    function startSweep() {
      clearSweep();
      state.sweep = 0;
      state.sweepTimer = setInterval(function () {
        // Mark current node as swept; previous stays "visited".
        if (state.sweep < SWEEP_NODES.length) {
          const k = SWEEP_NODES[state.sweep];
          cardMap[k].card.classList.add('s11-sweep');
          state.sweep++;
        } else {
          clearInterval(state.sweepTimer);
          state.sweepTimer = null;
        }
      }, SWEEP_STEP_MS);
    }

    function render() {
      const step = state.step;
      arch.classList.toggle('s11-show-edge-labels', step >= 1);
      arch.classList.toggle('s11-show-node-labels', step >= 2);
      arch.classList.toggle('s11-hover-on', step >= 3);
      // From step 1 onward, all the flow + skip arrows are drawn.
      arch.classList.toggle('s11-edges-lit', step >= 0);

      if (step !== 4) clearSweep();

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      if (step === 4 && state.sweep < 0) {
        startSweep();
      }

      requestAnimationFrame(layoutArrows);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });
    playBtn.addEventListener('click', function () {
      // Jump straight to step 4 (the sweep step).
      applyStep(4);
      // If we're already at 4, force a re-sweep.
      clearSweep();
      startSweep();
    });

    /* ---- Initial paint --------------------------------------------- */
    updateSampleBtns();
    renderCardContents();
    render();

    requestAnimationFrame(function () {
      requestAnimationFrame(layoutArrows);
    });
    setTimeout(layoutArrows, 0);
    setTimeout(layoutArrows, 50);
    setTimeout(layoutArrows, 200);

    const onResize = function () { layoutArrows(); };
    window.addEventListener('resize', onResize);
    const onScroll = function () { hideTooltip(); };
    window.addEventListener('scroll', onScroll, { passive: true });

    /* &run -> auto-advance through 0..3, then trigger play forward. */
    function autoAdvance(target) {
      if (state.step >= target) {
        // Then fire the sweep.
        playBtn.click();
        state.runTimer = null;
        return;
      }
      applyStep(state.step + 1);
      state.runTimer = setTimeout(function () { autoAdvance(target); }, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      state.runTimer = setTimeout(function () { autoAdvance(3); }, 300);
    }

    return {
      onEnter: function () {
        renderCardContents();
        render();
        requestAnimationFrame(function () {
          requestAnimationFrame(layoutArrows);
        });
      },
      onLeave: function () {
        if (state.runTimer) { clearTimeout(state.runTimer); state.runTimer = null; }
        clearSweep();
        hideTooltip();
        window.removeEventListener('resize', onResize);
        window.removeEventListener('scroll', onScroll);
      },
      onNextKey: function () {
        if (state.step < NUM_STEPS - 1) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  /* ---- Card factories -------------------------------------------- */

  function makeLevelCard(parent, name, sub, px, key) {
    const card = el('div', { class: 's11-card', 'data-node': key }, parent);
    const head = el('div', { class: 's11-card-head' }, card);
    el('span', { class: 's11-card-name', text: name }, head);
    el('span', { class: 's11-card-sub', text: sub }, head);
    const body = el('div', {
      class: 'canvas-host s11-card-body',
      style: 'width:' + px + 'px;height:' + px + 'px;',
    }, card);
    return { card: card, body: body };
  }

  function makeIOCard(parent, name, sub, px, key) {
    // Same DOM shape as a level card; the painter chooses what to draw.
    return makeLevelCard(parent, name, sub, px, key);
  }

  window.scenes = window.scenes || {};
  window.scenes.scene11 = function (root) { return buildScene(root); };
})();
