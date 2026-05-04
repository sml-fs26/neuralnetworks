/* Scene 6 — "How do you make a map bigger?"

   The decoder needs to undo three pools and put resolution back. The
   U-Net's answer is *transposed convolution* — a learned upsampling
   operator. This scene shows the result of that operator on a tiny
   4×4 → 8×8 toy. Scene 7 walks you through the mechanic.

   No nearest-neighbor / bilinear comparison panels — those were dropped
   per lecture feedback ("just talk about transposed convolution"). The
   per-cell stamp animation is gone too; scene 7 covers it with a
   richer sliding-filter visualization.

   Layout:
     hero → mini-map → input row (4×4 + swatch picker) → output panel
     (transposed conv output, 8×8) → forwarding caption.

   Single visual state; no step engine. */
(function () {
  'use strict';

  const INPUT_PX = 200;
  const OUTPUT_PX = 240;
  const SWATCH_PX = 56;

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

  function paintMap(host, data, px, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);
    const range = opts.valueRange || [-1, 1];
    window.Drawing.drawGrid(ctx, data, 0, 0, px, px, {
      diverging: true, valueRange: range,
    });
    const rows = data.length, cols = data[0].length;
    const cw = px / cols, ch = px / rows;
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 0; i <= rows; i++) {
      const y = Math.round(i * ch) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(px, y);
      ctx.stroke();
    }
    for (let j = 0; j <= cols; j++) {
      const x = Math.round(j * cw) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0); ctx.lineTo(x, px);
      ctx.stroke();
    }
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.upsample) {
      root.innerHTML = '<p style="opacity:0.5">upsample data missing.</p>';
      return {};
    }
    const D = window.DATA.upsample;
    const inputNames = Object.keys(D.inputs || {});
    if (inputNames.length === 0) {
      root.innerHTML = '<p style="opacity:0.5">no upsample inputs found.</p>';
      return {};
    }

    let defaultInput = inputNames.indexOf('small_cross');
    if (defaultInput < 0) defaultInput = 0;
    const filterNames = Object.keys(D.filters || {});
    let tconvFilter = 'plus';
    if (filterNames.indexOf(tconvFilter) < 0) tconvFilter = filterNames[0] || 'plus';

    root.innerHTML = '';
    root.classList.add('s6-root');
    const wrap = el('div', { class: 's6-wrap' }, root);

    /* ---- Hero ----------------------------------------------------- */
    const hero = el('header', { class: 'hero s6-hero' }, wrap);
    el('h1', { text: 'How do you make a map bigger?' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'The decoder has to undo three pools. The U-Net\'s answer is one specific operator: transposed convolution.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Take a tiny <em>4&times;4</em> feature map. Grow it to <em>8&times;8</em> ' +
        'using a transposed convolution with a learned 3&times;3 kernel. Here is ' +
        'what the result looks like for three different input patterns and the ' +
        '<code>+</code> kernel. The <em>next scene</em> opens the operator and ' +
        'walks through every step.',
    }, hero);

    /* ---- Mini-map ------------------------------------------------- */
    const miniHost = el('div', { class: 's6-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'upsamplers · up1 (bottleneck → 32²) and up2 (32² → 64²)',
      });
      mm.setHighlight(['up1', 'up2']);
    }

    /* ---- Main panel: input → transposed conv → output ------------- */
    const main = el('section', { class: 's6-main' }, wrap);

    // Input column
    const inputCol = el('div', { class: 's6-input-col' }, main);
    el('div', { class: 's6-panel-label', text: 'input · 4 × 4' }, inputCol);
    const inputHost = el('div', {
      class: 'canvas-host s6-input-host',
      style: 'width:' + INPUT_PX + 'px;height:' + INPUT_PX + 'px;',
    }, inputCol);
    el('div', {
      class: 's6-panel-sub',
      text: 'pick an input pattern',
    }, inputCol);
    const swatchRow = el('div', { class: 's6-swatch-row' }, inputCol);

    // Operator label between input and output
    const opLabel = el('div', { class: 's6-op-label' }, main);
    opLabel.innerHTML =
      '<div class="s6-op-arrow">→</div>' +
      '<div class="s6-op-name">transposed conv</div>' +
      '<div class="s6-op-sub">3×3 kernel · stride 2</div>';

    // Output column (single)
    const outCol = el('div', { class: 's6-out-col' }, main);
    el('div', { class: 's6-panel-label', text: 'output · 8 × 8' }, outCol);
    const outHost = el('div', {
      class: 'canvas-host s6-out-host',
      style: 'width:' + OUTPUT_PX + 'px;height:' + OUTPUT_PX + 'px;',
    }, outCol);
    el('div', {
      class: 's6-panel-sub',
      html: 'a learned <code>3&times;3</code> kernel produced this from the 4&times;4 ' +
            'input on the left',
    }, outCol);

    // Build the swatch buttons.
    const swatchHosts = [];
    for (let i = 0; i < inputNames.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's6-swatch-btn' + (i === defaultInput ? ' active' : ''),
        'data-input-index': String(i),
        'aria-label': 'Select input ' + inputNames[i],
        title: inputNames[i].replace('_', ' '),
      }, swatchRow);
      const sHost = el('div', { class: 'canvas-host s6-swatch-host' }, btn);
      swatchHosts.push(sHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-input-index'), 10);
        if (Number.isFinite(idx)) switchInput(idx);
      });
    }

    /* ---- Forwarding caption -------------------------------------- */
    el('p', {
      class: 'caption s6-caption',
      html:
        'That kernel is <em>learned</em> by gradient descent — not hand-coded. ' +
        '<strong>Scene 7</strong> opens the operator: every input cell stamps a ' +
        'scaled copy of the kernel into the output, and you can adjust the ' +
        'stride to see how the output size and stamp overlap change.',
    }, wrap);

    /* ---- State + render ------------------------------------------ */
    const state = { inputIdx: defaultInput };

    function currentInputName() { return inputNames[state.inputIdx]; }
    function currentOutputs() {
      const out = D.outputs[currentInputName()] || {};
      const tcSet = out.tconv || {};
      return tcSet[tconvFilter] || tcSet[Object.keys(tcSet)[0]] || [[0]];
    }

    function paintInput() {
      const inp = D.inputs[currentInputName()];
      if (!inp) return;
      const range = sharedRange([inp]);
      paintMap(inputHost, inp, INPUT_PX, { valueRange: range });
    }
    function paintOutput() {
      const tc = currentOutputs();
      const range = sharedRange([tc]);
      paintMap(outHost, tc, OUTPUT_PX, { valueRange: range });
    }
    function paintSwatches() {
      for (let i = 0; i < inputNames.length; i++) {
        const inp = D.inputs[inputNames[i]];
        if (!inp) continue;
        const range = sharedRange([inp]);
        paintMap(swatchHosts[i], inp, SWATCH_PX, { valueRange: range });
      }
      const btns = swatchRow.querySelectorAll('.s6-swatch-btn');
      btns.forEach((b, i) => b.classList.toggle('active', i === state.inputIdx));
    }
    function render() {
      paintInput();
      paintOutput();
      paintSwatches();
    }

    function switchInput(idx) {
      if (idx < 0 || idx >= inputNames.length) return;
      state.inputIdx = idx;
      render();
    }

    render();

    return {
      onEnter: function () { render(); },
      onLeave: function () {},
      onNextKey: function () { return false; },
      onPrevKey: function () { return false; },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene6 = function (root) { return buildScene(root); };
})();
