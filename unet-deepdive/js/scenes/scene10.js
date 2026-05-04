/* Scene 10 -- "What the filters look like."

   Slot between scene4 (encoder) and scene5 (bottleneck). The encoder scene
   shows feature maps but never the *weights* that produced them. This scene
   makes the abstract idea "the network has learned filters" concrete by
   actually showing them.

   Four reveals:
     1. The 16 filters of enc1_conv1 -- the only ones that look at raw RGB,
        and so the only ones we can render as little RGB pictures.
     2. A deeper-layer pick: each filter is 3x3 per input *channel*. Cells
        are diverging-colormap weight stencils, no longer interpretable as
        pictures.
     3. The transposed-conv kernels (up1, up2): the small 2x2 patterns the
        decoder will stamp.
     4. The 1x1 classifier head: 5 classes x 16 input channels of weights,
        rendered as 5 small bar charts.

   Step engine:
     0 = hero only
     1 = enc1_conv1 RGB filters revealed
     2 = deeper layer's filters revealed (picker)
     3 = transposed-conv kernels revealed
     4 = 1x1 head bar charts revealed

   `&run` advances 0 -> 4 over a few seconds. */
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 800;

  const RGB_TILE_PX = 72;        // each enc1_conv1 filter rendered as 3x3 RGB
  const DEEP_TILE_PX = 56;       // grayscale weight stencil tile
  const DEEP_GRID_OUT = 4;       // show first 4 input channels
  const DEEP_GRID_IN = 4;        // and 4 output filters at a time
  const TCONV_TILE_PX = 56;      // 2x2 stencils a touch bigger
  const HEAD_BAR_W = 188;
  const HEAD_BAR_H = 96;

  // The deeper-layer picker advertises these layers. Display name + key
  // into window.DATA.filters, plus the per-filter shape note.
  const DEEP_LAYERS = [
    { key: 'enc1_conv2', label: 'enc1 · conv2',  shape: '16 filters · 3×3 × 16 in' },
    { key: 'enc2_conv1', label: 'enc2 · conv1',  shape: '32 filters · 3×3 × 16 in' },
    { key: 'enc2_conv2', label: 'enc2 · conv2',  shape: '32 filters · 3×3 × 32 in' },
    { key: 'enc3_conv1', label: 'enc3 · conv1',  shape: '64 filters · 3×3 × 32 in' },
    { key: 'enc3_conv2', label: 'enc3 · conv2',  shape: '64 filters · 3×3 × 64 in' },
  ];

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

  function readVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim() || '#888';
  }

  /* Symmetric absolute-max across an arbitrarily nested numeric array. */
  function symMax(arr) {
    let m = 0;
    function walk(x) {
      if (typeof x === 'number') {
        const a = Math.abs(x);
        if (a > m) m = a;
      } else if (Array.isArray(x)) {
        for (let i = 0; i < x.length; i++) walk(x[i]);
      }
    }
    walk(arr);
    return m || 1;
  }

  /* Paint one 3x3x3 enc1_conv1 filter into a canvas as a small RGB image.
     Each filter has 3 input channels (R, G, B). At cell (i, j) the painted
     RGB color is the per-channel weight, symmetric-normalized so that
     0 -> 128, +max -> 255, -max -> 0. That makes both polarities visible:
     a "red" cell is one whose R weight is strongly positive, a "cyan" cell
     is one whose R weight is strongly negative, etc. The viewer reads the
     filter as an image. */
  function paintRGBFilter(host, filter3x3x3, m, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    // The filter is shape [3 (channels)][3][3]. Build a 3x3 RGB image.
    const cells = 3;
    const cw = px / cells;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const r = filter3x3x3[0][i][j];
        const g = filter3x3x3[1][i][j];
        const b = filter3x3x3[2][i][j];
        const R = Math.max(0, Math.min(255, Math.round(128 + (r / m) * 127)));
        const G = Math.max(0, Math.min(255, Math.round(128 + (g / m) * 127)));
        const B = Math.max(0, Math.min(255, Math.round(128 + (b / m) * 127)));
        ctx.fillStyle = 'rgb(' + R + ',' + G + ',' + B + ')';
        ctx.fillRect(j * cw, i * cw, Math.ceil(cw), Math.ceil(cw));
      }
    }
    // Light cell separators so the 3x3 reads as a grid.
    const t = window.Drawing.tokens();
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 1; i < cells; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cw); ctx.lineTo(px, i * cw);
      ctx.moveTo(i * cw, 0); ctx.lineTo(i * cw, px);
      ctx.stroke();
    }
    ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
  }

  /* Paint a 3x3 weight stencil with the diverging colormap. */
  function paintStencil(host, kernel, m, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    window.Drawing.drawGrid(ctx, kernel, 0, 0, px, px, {
      diverging: true,
      cellBorder: true,
      valueRange: [-m, m],
    });
    const t = window.Drawing.tokens();
    ctx.strokeStyle = t.rule;
    ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
  }

  /* Bar chart for the 1x1 head: input-channel index on x, weight on y.
     Positive bars upward (in `--cnn-pos`), negative downward (`--cnn-neg`). */
  function paintHeadBars(host, weights16, color, w, h) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, w, h);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);

    let m = 0;
    for (let i = 0; i < weights16.length; i++) {
      const a = Math.abs(weights16[i]);
      if (a > m) m = a;
    }
    if (!m) m = 1;

    const padX = 6, padY = 8;
    const innerW = w - 2 * padX;
    const innerH = h - 2 * padY;
    const midY = padY + innerH / 2;
    const barW = innerW / weights16.length;

    // baseline
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, midY);
    ctx.lineTo(padX + innerW, midY);
    ctx.stroke();

    for (let i = 0; i < weights16.length; i++) {
      const v = weights16[i] / m;
      const barH = (innerH / 2) * Math.abs(v);
      const x = padX + i * barW;
      ctx.fillStyle = v >= 0 ? t.pos : t.neg;
      if (v >= 0) {
        ctx.fillRect(x + 0.5, midY - barH, Math.max(1, barW - 1), barH);
      } else {
        ctx.fillRect(x + 0.5, midY, Math.max(1, barW - 1), barH);
      }
    }

    // Accent border in the slot's class color.
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0.75, 0.75, w - 1.5, h - 1.5);
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.filters) {
      root.innerHTML = '<p style="opacity:0.5">scene10: window.DATA.filters missing.</p>';
      return {};
    }
    const F = window.DATA.filters;
    const Drawing = window.Drawing;

    root.innerHTML = '';
    root.classList.add('s10-root');
    const wrap = el('div', { class: 's10-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s10-hero' }, wrap);
    el('h1', { text: 'What the filters look like.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'The weights are not abstract: layer one is small color-edge pictures. Deeper layers stop being pictures.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'We just watched the encoder squeeze a 64×64×3 image down into ' +
        '16×16×64. Underneath every feature map is a stack of <em>filters</em> ' +
        '&mdash; the actual learned weights. Most of the network never touches ' +
        'RGB again, so most filters are not images. Let&#39;s look at the ones ' +
        'we <em>can</em> see, and at the unfamiliar shapes the rest of the ' +
        'network actually learns.',
    }, hero);

    /* ---- Anatomy callout: each "block" is two convs ------------------ */
    /* The architecture diagrams elsewhere only show enc1, enc2, enc3 (and
       dec2, dec1). But each of these is actually TWO 3×3 conv layers
       stacked together. The picker below will let you pick conv1 or conv2
       within each block — they exist; we just don't usually draw them. */
    const anatomy = el('section', { class: 's10-anatomy' }, wrap);
    el('div', { class: 's10-anatomy-eyebrow', text: 'why "conv1" and "conv2"?' }, anatomy);
    el('p', {
      class: 's10-anatomy-text',
      html:
        'In the U-Net diagrams elsewhere in the deepdive you see boxes labelled ' +
        '<code>enc1</code>, <code>enc2</code>, <code>enc3</code>, <code>dec2</code>, ' +
        '<code>dec1</code>. Each of those boxes is actually <strong>two</strong> 3×3 ' +
        'convolution layers stacked back to back, named <code>conv1</code> (the first) ' +
        'and <code>conv2</code> (the second). Same for the decoder blocks. We collapse ' +
        'them into one box in the architecture diagrams to keep them readable, but the ' +
        'picker below lets you inspect each one individually.',
    }, anatomy);

    /* ---- Section 1: enc1_conv1 RGB filters ------------------------- */
    const sec1 = el('section', { class: 's10-section s10-rgb-section' }, wrap);
    const sec1Head = el('div', { class: 's10-sec-head' }, sec1);
    el('span', { class: 's10-sec-name', text: 'enc1 · conv1' }, sec1Head);
    el('span', {
      class: 's10-sec-sub',
      text: '16 filters · 3×3 × 3 RGB inputs',
    }, sec1Head);
    const rgbGrid = el('div', { class: 's10-rgb-grid' }, sec1);
    const rgbHosts = [];
    const enc1 = F.enc1_conv1;     // [16][3][3][3]
    const enc1Max = symMax(enc1);
    for (let f = 0; f < enc1.length; f++) {
      const tile = el('div', { class: 's10-rgb-tile' }, rgbGrid);
      const tHost = el('div', {
        class: 'canvas-host s10-rgb-host',
        style: 'width:' + RGB_TILE_PX + 'px;height:' + RGB_TILE_PX + 'px;',
        title: 'filter ' + f,
      }, tile);
      el('div', { class: 's10-rgb-label', text: '#' + f }, tile);
      rgbHosts.push(tHost);
    }
    el('p', {
      class: 's10-sec-caption',
      html:
        'Each tile is one of the network&#39;s 16 first-layer filters. ' +
        'These act on raw RGB, so we can paint them as 3×3 pictures: each ' +
        'cell\'s color is its own (R, G, B) weight (zero is neutral gray; ' +
        'positive in a channel pushes that color up). Squint and you can ' +
        'see directional edges and color-opponent patterns &mdash; <em>this is the ' +
        'only layer in the network where filters look like little images.</em>',
    }, sec1);

    /* ---- Section 2: deeper layer filters --------------------------- */
    const sec2 = el('section', { class: 's10-section s10-deep-section' }, wrap);
    const sec2Head = el('div', { class: 's10-sec-head' }, sec2);
    el('span', { class: 's10-sec-name', text: 'deeper layers' }, sec2Head);
    el('span', {
      class: 's10-sec-sub s10-deep-sub',
      text: 'each filter sees feature maps, not colors',
    }, sec2Head);

    // Picker row layout: buttons on the left, U-Net mini-map on the right
    // (so the viewer can see WHERE in the network they're looking).
    const pickerWrap = el('div', { class: 's10-picker-wrap' }, sec2);
    const picker = el('div', { class: 's10-picker' }, pickerWrap);
    el('span', { class: 's10-picker-label', text: 'layer' }, picker);
    const pickerRow = el('div', { class: 's10-picker-row' }, picker);
    const pickerBtns = [];
    for (let i = 0; i < DEEP_LAYERS.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's10-picker-btn',
        'data-layer-key': DEEP_LAYERS[i].key,
        text: DEEP_LAYERS[i].label,
      }, pickerRow);
      btn.addEventListener('click', function () {
        switchLayer(DEEP_LAYERS[i].key);
      });
      pickerBtns.push(btn);
    }
    // Mini-map ("you are here").
    const miniHost = el('div', { class: 's10-picker-mini' }, pickerWrap);
    const miniMap = window.UNET.mountUNetMiniMap(miniHost, {
      width: 240, label: '', title: 'you are here',
    });

    // Filter-index slider (which output filter to display).
    const filterSlider = el('div', { class: 's10-filter-slider' }, sec2);
    el('label', { for: 's10-filter-input', text: 'output filter #' }, filterSlider);
    const filterInput = el('input', {
      id: 's10-filter-input',
      type: 'range', min: '0', max: '0', step: '1', value: '0',
    }, filterSlider);
    const filterOut = el('output', {
      class: 'control-value s10-filter-out', text: '0',
    }, filterSlider);
    const filterHelp = el('span', {
      class: 's10-filter-help',
      text: '',
    }, filterSlider);

    const deepGrid = el('div', { class: 's10-deep-grid' }, sec2);
    const deepTiles = [];
    for (let i = 0; i < DEEP_GRID_IN; i++) {
      const row = el('div', { class: 's10-deep-row' }, deepGrid);
      const inLabel = el('div', { class: 's10-deep-in-label' }, row);
      el('span', { text: 'in #' + i }, inLabel);
      const tile = el('div', {
        class: 'canvas-host s10-deep-tile',
        style: 'width:' + DEEP_TILE_PX + 'px;height:' + DEEP_TILE_PX + 'px;',
        title: 'input channel ' + i,
      }, row);
      deepTiles.push(tile);
    }
    el('p', {
      class: 's10-sec-caption',
      html:
        'Now each filter is shaped <em>3×3 × C</em>, where C is the number of ' +
        'feature maps coming out of the previous layer. Each row above is one ' +
        'of those C input channels. A blue cell weighs that patch of that ' +
        'feature positively; a red cell weighs it negatively. These are no ' +
        'longer pictures &mdash; they are <em>recipes for combining features</em>. ' +
        'Slide the filter index to see the next output channel.',
    }, sec2);

    /* ---- Section 3: transposed conv kernels ------------------------ */
    const sec3 = el('section', { class: 's10-section s10-tconv-section' }, wrap);
    const sec3Head = el('div', { class: 's10-sec-head' }, sec3);
    el('span', { class: 's10-sec-name', text: 'transposed conv · up1, up2' }, sec3Head);
    el('span', {
      class: 's10-sec-sub',
      text: '2×2 stencils that the decoder stamps to grow the map',
    }, sec3Head);

    const tconvWrap = el('div', { class: 's10-tconv-wrap' }, sec3);
    const tconvLeft = el('div', { class: 's10-tconv-block' }, tconvWrap);
    el('div', { class: 's10-tconv-block-name', text: 'up2 · 64 → 32, kernel 2×2' }, tconvLeft);
    const tconvLeftRow = el('div', { class: 's10-tconv-row' }, tconvLeft);
    const tconvLeftHosts = buildTconvBlockTiles(tconvLeftRow, 6);

    const tconvRight = el('div', { class: 's10-tconv-block' }, tconvWrap);
    el('div', { class: 's10-tconv-block-name', text: 'up1 · 32 → 16, kernel 2×2' }, tconvRight);
    const tconvRightRow = el('div', { class: 's10-tconv-row' }, tconvRight);
    const tconvRightHosts = buildTconvBlockTiles(tconvRightRow, 6);

    el('p', {
      class: 's10-sec-caption',
      html:
        'PyTorch stores transposed-conv weight as ' +
        '<code>(in, out, kH, kW)</code>. We pick a few input channels and ' +
        'show the 2×2 patches each one will <em>stamp</em> when its single ' +
        'pixel is upsampled. We will see these in action in scene 7 &mdash; ' +
        'these tiny stencils are how the decoder learns to grow the feature ' +
        'map back to full resolution.',
    }, sec3);

    /* ---- Section 4: 1x1 classifier head ---------------------------- */
    const sec4 = el('section', { class: 's10-section s10-head-section' }, wrap);
    const sec4Head = el('div', { class: 's10-sec-head' }, sec4);
    el('span', { class: 's10-sec-name', text: 'output head · 1×1 conv' }, sec4Head);
    el('span', {
      class: 's10-sec-sub',
      text: '5 classes · each is a weighted sum of 16 feature channels',
    }, sec4Head);

    const headRow = el('div', { class: 's10-head-row' }, sec4);
    const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];
    const headHosts = [];
    for (let c = 0; c < 5; c++) {
      const card = el('div', { class: 's10-head-card' }, headRow);
      const cardHead = el('div', { class: 's10-head-card-head' }, card);
      const swatch = el('span', { class: 's10-head-swatch' }, cardHead);
      swatch.style.background = 'var(--class-' + CLASS_NAMES[c] + ')';
      el('span', { class: 's10-head-name', text: CLASS_NAMES[c] }, cardHead);
      const host = el('div', {
        class: 'canvas-host s10-head-host',
        style: 'width:' + HEAD_BAR_W + 'px;height:' + HEAD_BAR_H + 'px;',
      }, card);
      el('div', { class: 's10-head-axis', text: 'channel 0 ······ 15' }, card);
      headHosts.push(host);
    }
    el('p', {
      class: 's10-sec-caption',
      html:
        '<strong>What feeds this head?</strong> The last decoder block, <code>dec1</code>, ' +
        'outputs <strong>16 feature channels</strong> at every one of the 64×64 pixel ' +
        'positions. So the input to the head is a tensor of shape ' +
        '<code>64 × 64 × 16</code> &mdash; one 16-vector per pixel. ' +
        '(The 16 here has nothing to do with the 16×16 bottleneck. ' +
        'It is the channel count chosen for the last decoder block.)' +
        '<br><br>' +
        '<strong>What does "1×1 conv" mean?</strong> A regular convolution looks at a ' +
        '3×3 neighbourhood. A <em>1×1</em> convolution looks at one pixel only &mdash; ' +
        'no neighbours. So the head treats each pixel independently: it takes that ' +
        'pixel\'s 16-vector and produces 5 numbers (one per class). The mapping is a ' +
        'linear combination: ' +
        '<code>out[class] = w[class][0]·feat[0] + … + w[class][15]·feat[15]</code>. ' +
        'That is 5 × 16 = 80 weights total &mdash; the entire classifier.' +
        '<br><br>' +
        'Each bar chart above shows one class\'s 16 weights. Blue cells are channels ' +
        'that <em>raise</em> the class\'s score; red cells <em>suppress</em> it. The 1×1 ' +
        'is small precisely because the encoder + decoder already invented useful ' +
        'features &mdash; the head only has to weight them.',
    }, sec4);

    /* ---- Caption + step controls ----------------------------------- */
    const caption = el('p', { class: 'caption s10-caption' }, wrap);
    const controls = el('div', { class: 'controls s10-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's10-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's10-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', {
      class: 'control-value', text: '0 / ' + (NUM_STEPS - 1),
    }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State & rendering ----------------------------------------- */
    const state = {
      step: 0,
      deepLayerKey: DEEP_LAYERS[0].key,
      deepFilterIdx: 0,
    };

    function buildTconvBlockTiles(row, n) {
      const hosts = [];
      for (let i = 0; i < n; i++) {
        const cell = el('div', { class: 's10-tconv-cell' }, row);
        const tHost = el('div', {
          class: 'canvas-host s10-tconv-host',
          style: 'width:' + TCONV_TILE_PX + 'px;height:' + TCONV_TILE_PX + 'px;',
        }, cell);
        el('div', { class: 's10-tconv-label', text: 'in #' + i }, cell);
        hosts.push(tHost);
      }
      return hosts;
    }

    function paintRGBSection() {
      for (let f = 0; f < enc1.length; f++) {
        paintRGBFilter(rgbHosts[f], enc1[f], enc1Max, RGB_TILE_PX);
      }
    }

    function currentDeepLayer() {
      return F[state.deepLayerKey];
    }

    function paintDeepSection() {
      const layer = currentDeepLayer();    // [out][in][3][3]
      if (!layer) return;
      const numOut = layer.length;
      const numIn = layer[0].length;
      // Clamp filter index to layer size.
      if (state.deepFilterIdx >= numOut) state.deepFilterIdx = 0;
      filterInput.max = String(numOut - 1);
      filterInput.value = String(state.deepFilterIdx);
      filterOut.textContent = String(state.deepFilterIdx);

      // Symmetric range: per-(filter, all-input-channels) so cells in this
      // row of tiles are comparable to each other.
      const filt = layer[state.deepFilterIdx];
      let m = 0;
      for (let c = 0; c < Math.min(DEEP_GRID_IN, numIn); c++) {
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) {
            const a = Math.abs(filt[c][i][j]);
            if (a > m) m = a;
          }
        }
      }
      if (!m) m = 1;

      for (let i = 0; i < DEEP_GRID_IN; i++) {
        if (i < numIn) {
          paintStencil(deepTiles[i], filt[i], m, DEEP_TILE_PX);
        } else {
          deepTiles[i].innerHTML = '';
        }
      }
      // Update picker active state.
      for (let p = 0; p < pickerBtns.length; p++) {
        pickerBtns[p].classList.toggle(
          'active',
          pickerBtns[p].getAttribute('data-layer-key') === state.deepLayerKey
        );
      }
      // Update mini-map "you are here" highlight + filter-help wording.
      const blockKey = state.deepLayerKey.split('_')[0];   // 'enc1' / 'enc2' / 'enc3'
      if (miniMap && typeof miniMap.setHighlight === 'function') {
        miniMap.setHighlight([blockKey],
          state.deepLayerKey.replace('_', ' · ') + '  (' + numOut + ' filters · ' +
          numIn + ' input channels each)');
      }
      // Filter-help text: explain exactly what the slider/grid show.
      const shown = Math.min(DEEP_GRID_IN, numIn);
      filterHelp.innerHTML =
        'this layer has <strong>' + numOut + '</strong> output filters; ' +
        'each one is shaped 3×3 × <strong>' + numIn + '</strong> ' +
        '(one 3×3 stencil per input channel). ' +
        'Slide above to pick which output filter to inspect; the grid below ' +
        'shows the first <strong>' + shown + '</strong> of its <strong>' + numIn +
        '</strong> input-channel stencils.';
    }

    function paintTconvSection() {
      const up2 = F.up2;   // [64 (in)][32 (out)][2][2]
      const up1 = F.up1;   // [32 (in)][16 (out)][2][2]
      // For each block, show the kernel for output channel 0 across the
      // first 6 input channels. That makes the "stamp pattern" each input
      // would deposit immediately legible.
      function paintBlock(layer, hosts) {
        if (!layer) return;
        // Symmetric range across all input channels for the chosen output 0.
        let m = 0;
        for (let i = 0; i < Math.min(hosts.length, layer.length); i++) {
          const k = layer[i][0];
          for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
            const a = Math.abs(k[r][c]);
            if (a > m) m = a;
          }
        }
        if (!m) m = 1;
        for (let i = 0; i < hosts.length; i++) {
          if (i < layer.length) {
            paintStencil(hosts[i], layer[i][0], m, TCONV_TILE_PX);
          } else {
            hosts[i].innerHTML = '';
          }
        }
      }
      paintBlock(up2, tconvLeftHosts);
      paintBlock(up1, tconvRightHosts);
    }

    function paintHeadSection() {
      const head = F.out;  // [5][16][1][1]
      if (!head) return;
      for (let c = 0; c < 5; c++) {
        // Flatten [16][1][1] -> [16].
        const w = new Array(head[c].length);
        for (let k = 0; k < head[c].length; k++) {
          w[k] = head[c][k][0][0];
        }
        const color = readVar('--class-' + CLASS_NAMES[c]);
        paintHeadBars(headHosts[c], w, color, HEAD_BAR_W, HEAD_BAR_H);
      }
    }

    function captionFor(step) {
      switch (step) {
        case 0:
          return 'A network is its weights. We have shown you feature maps; let us show you what made them.';
        case 1:
          return 'Layer 1: 16 little RGB pictures. The only filters in the entire U-Net you can read as images.';
        case 2:
          return 'Deeper layers act on features, not colors. The same 3×3 stencil, repeated for every input channel.';
        case 3:
          return 'Transposed-conv kernels are tiny — 2×2 stencils that get stamped onto a bigger canvas. We use them in the decoder.';
        case 4:
          return 'The 1×1 head: 5 classes, each a weighted vote over the final 16 feature channels.';
        default:
          return '';
      }
    }

    function render() {
      const step = state.step;

      sec1.classList.toggle('s10-revealed', step >= 1);
      sec2.classList.toggle('s10-revealed', step >= 2);
      sec3.classList.toggle('s10-revealed', step >= 3);
      sec4.classList.toggle('s10-revealed', step >= 4);

      if (step >= 1) paintRGBSection();
      if (step >= 2) paintDeepSection();
      if (step >= 3) paintTconvSection();
      if (step >= 4) paintHeadSection();

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
    function switchLayer(key) {
      if (!F[key]) return;
      state.deepLayerKey = key;
      state.deepFilterIdx = 0;
      paintDeepSection();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });
    filterInput.addEventListener('input', function () {
      const v = parseInt(filterInput.value, 10);
      if (Number.isFinite(v)) {
        state.deepFilterIdx = v;
        paintDeepSection();
      }
    });

    render();

    /* &run -> auto-advance to last step over a few seconds. */
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
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
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

  window.scenes = window.scenes || {};
  window.scenes.scene10 = function (root) { return buildScene(root); };
})();
