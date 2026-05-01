/* Scene 10 — "What we built."

   Coda. A 5×2 grid of takeaway cards summarizing scenes 0..9. Each card
   has a tiny scene number, a one-line title, an italic one-sentence
   takeaway, and a small symbolic thumbnail — drawn from DATA so we don't
   fabricate.

   Click any card → window.CDD.goTo(N).

   Reads:
     window.DATA.handFilters             (scene 0 thumb)
     window.DATA.shapelets.{conv1FiltersNormalized, conv2FiltersNormalized}
     window.DATA.AM.neurons              (scene 8 thumb)
     window.Drawing.{drawGrid, setupCanvas, tokens, makeSample}
*/
(function () {
  'use strict';

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

  // 56×56 logical px for each thumb. We sub-divide for triptychs.
  const THUMB_PX = 56;

  // Hand-pick the AM neuron with the most striking image (highest variance).
  function pickStrikingAMIndex() {
    const neurons = window.DATA.AM.neurons;
    let best = 0, bestVar = -1;
    for (let i = 0; i < neurons.length; i++) {
      const im = neurons[i].image;
      let s = 0, s2 = 0, n = 0;
      for (const row of im) for (const v of row) { s += v; s2 += v * v; n++; }
      const mean = s / n;
      const variance = s2 / n - mean * mean;
      if (variance > bestVar) { bestVar = variance; best = i; }
    }
    return best;
  }

  // Each thumb is a function(ctx, w, h, t) where t is Drawing.tokens().
  function makeThumbDrawers() {
    const D = window.Drawing;
    const HF = window.DATA.handFilters;
    const SH = window.DATA.shapelets;
    const AM = window.DATA.AM.neurons;
    const strikingIdx = pickStrikingAMIndex();

    return {
      // Scene 0: hand-filter result — input on the left, response on the right.
      0(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const halfW = Math.floor(w / 2);
        const sample = D.makeSample('cross', 28);
        D.drawGrid(ctx, sample, 0, 0, halfW - 2, h, {
          diverging: false, valueRange: [0, 1],
        });
        const fmap = window.CNN.conv2d(sample, HF.vertical, 2);
        D.drawGrid(ctx, fmap, halfW + 2, 0, w - halfW - 2, h, {
          diverging: true,
        });
      },
      // Scene 1: a single 5×5 hand filter.
      1(ctx, w, h, t) {
        D.drawGrid(ctx, HF.vertical, 0, 0, w, h, {
          diverging: true, cellBorder: true, valueRange: [-2, 2],
        });
      },
      // Scene 2: a Σ symbol over the dot product.
      2(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
        ctx.fillStyle = t.ink;
        ctx.font = `${Math.floor(h * 0.62)}px "Iowan Old Style", Palatino, Georgia, serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('×→Σ', w / 2, h / 2 + 1);
      },
      // Scene 3: one filter on top, its (synthetic) feature map below.
      3(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const half = Math.floor(h / 2);
        D.drawGrid(ctx, HF.diag_down, 0, 0, w, half, {
          diverging: true, cellBorder: false, valueRange: [-2, 2],
        });
        // Bottom: a 14×14 feature-map-ish thing, computed for real:
        // conv2d(triangle, diag_down) reduced down for size.
        const inp = D.makeSample('triangle', 28);
        const fmap = window.CNN.conv2d(inp, HF.diag_down, 2);
        D.drawGrid(ctx, fmap, 0, half, w, h - half, {
          diverging: true, cellBorder: false,
        });
        // Subtle divider.
        ctx.strokeStyle = t.bg;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, half + 0.5);
        ctx.lineTo(w, half + 0.5);
        ctx.stroke();
      },
      // Scene 4: stack of three small filters.
      4(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const inset = 4;
        const cellW = (w - inset * 2);
        const stripH = Math.floor((h - inset * 2 - 4) / 3);
        const filters = [HF.vertical, HF.horizontal, HF.diag_down];
        for (let i = 0; i < 3; i++) {
          D.drawGrid(ctx, filters[i],
            inset, inset + i * (stripH + 2),
            cellW, stripH,
            { diverging: true, cellBorder: false, valueRange: [-2, 2] });
        }
      },
      // Scene 5: 6 colored dots — one per shapelets class.
      5(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const cs = getComputedStyle(document.documentElement);
        const colors = [
          cs.getPropertyValue('--cnn-pos').trim(),
          cs.getPropertyValue('--cnn-accent').trim(),
          cs.getPropertyValue('--cnn-purple').trim(),
          cs.getPropertyValue('--cnn-green').trim(),
          cs.getPropertyValue('--cnn-neg').trim(),
          cs.getPropertyValue('--ink-secondary').trim(),
        ];
        const r = 4.5;
        const cols = 3, rows = 2;
        const padX = 8, padY = 10;
        const dx = (w - 2 * padX) / (cols - 1);
        const dy = (h - 2 * padY) / (rows - 1);
        for (let i = 0; i < 6; i++) {
          const c = i % cols, rr = Math.floor(i / cols);
          ctx.fillStyle = colors[i];
          ctx.beginPath();
          ctx.arc(padX + c * dx, padY + rr * dy, r, 0, Math.PI * 2);
          ctx.fill();
        }
      },
      // Scene 6: a small cone-of-vision.
      6(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        // A trapezoid emanating from the apex.
        ctx.fillStyle = t.pos;
        ctx.globalAlpha = 0.18;
        ctx.beginPath();
        ctx.moveTo(w / 2, 6);
        ctx.lineTo(w - 6, h - 6);
        ctx.lineTo(6, h - 6);
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = 1;
        // Outline.
        ctx.strokeStyle = t.pos;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(w / 2, 6);
        ctx.lineTo(w - 6, h - 6);
        ctx.moveTo(w / 2, 6);
        ctx.lineTo(6, h - 6);
        ctx.stroke();
        // Apex dot.
        ctx.fillStyle = t.ink;
        ctx.beginPath();
        ctx.arc(w / 2, 6, 2, 0, Math.PI * 2);
        ctx.fill();
      },
      // Scene 7: hand filter beside a learned filter.
      7(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const half = Math.floor(w / 2);
        D.drawGrid(ctx, HF.vertical, 0, 0, half - 2, h, {
          diverging: true, cellBorder: false, valueRange: [-2, 2],
        });
        // Pick a learned conv1 filter that resembles a vertical line if any
        // exists; otherwise just take the first.
        const learned = (SH.conv1FiltersNormalized && SH.conv1FiltersNormalized[0]) || HF.horizontal;
        D.drawGrid(ctx, learned, half + 2, 0, w - half - 2, h, {
          diverging: true, cellBorder: false, valueRange: [-1, 1],
        });
      },
      // Scene 8: the most striking AM image.
      8(ctx, w, h, t) {
        const img = AM[strikingIdx].image;
        D.drawGrid(ctx, img, 0, 0, w, h, {
          diverging: false, valueRange: [0, 1],
        });
      },
      // Scene 9: a 5-color gradient strip (segmentation classes).
      9(ctx, w, h, t) {
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);
        const cs = getComputedStyle(document.documentElement);
        const colors = [
          cs.getPropertyValue('--class-sky').trim(),
          cs.getPropertyValue('--class-grass').trim(),
          cs.getPropertyValue('--class-sun').trim(),
          cs.getPropertyValue('--class-tree').trim(),
          cs.getPropertyValue('--class-person').trim(),
        ];
        const inset = 5;
        const stripH = Math.floor((h - inset * 2 - 8) / 5);
        for (let i = 0; i < 5; i++) {
          ctx.fillStyle = colors[i];
          ctx.fillRect(inset, inset + i * (stripH + 2), w - inset * 2, stripH);
        }
      },
    };
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.DATA.shapelets || !window.DATA.AM
        || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 10: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s10-root');
    const wrap = el('div', { class: 's10-wrap' }, root);

    // Hero
    const hero = el('div', { class: 'hero s10-hero' }, wrap);
    el('h1', { class: 's10-h1', text: 'What we built.' }, hero);
    el('p', {
      class: 'subtitle s10-subtitle',
      text: 'Ten scenes, one mental model.',
    }, hero);
    el('p', {
      class: 'lede s10-lede',
      text: 'From a 5×5 picture to a per-pixel labeled image. Every step used the same machinery.',
    }, hero);

    // Card data — titles match SCENE_TITLES in main.js, takeaways from the brief.
    const titles = (window.CDD && window.CDD.sceneTitles) || [
      'A convolutional network, end to end',
      'A filter is a little picture',
      'The dot product as a match score',
      'One filter, one feature map',
      'Stacking filters becomes a unit',
      'Race the detectors',
      'Receptive fields — the cone of vision',
      'Handcrafted vs. learned',
      'What does this neuron want to see?',
      'Segmentation — same machinery, per pixel',
    ];
    const takeaways = [
      'A convolutional network looks for small patterns and reports where it found them.',
      'A filter is a 5×5 picture. Slide it across the input.',
      'The match score is the dot product. Multiply, sum, drop.',
      'Same input, eight different views.',
      'Stack three filters and a unit detects a higher-order pattern.',
      'Detectors fire in parallel. Each finds its pattern wherever it lives.',
      'Each layer sees a wider patch of the input than the last.',
      'You built one. The network built dozens.',
      'Each neuron has an essence — the input that screams the loudest.',
      'The same machinery, run per pixel, paints semantic labels onto images.',
    ];

    const drawers = makeThumbDrawers();

    // Grid.
    const grid = el('div', { class: 's10-grid' }, wrap);
    const cardRefs = [];
    for (let i = 0; i < 10; i++) {
      const card = el('button', { class: 's10-card', type: 'button',
        'aria-label': `Go to scene ${i}: ${titles[i]}` }, grid);
      el('div', { class: 's10-card-num', text: `scene ${String(i).padStart(2, '0')}` }, card);
      el('h3', { class: 's10-card-title', text: titles[i] }, card);
      el('p', { class: 's10-card-take', text: takeaways[i] }, card);
      const thumbHost = el('div', { class: 's10-card-thumb canvas-host' }, card);
      const cv = window.Drawing.setupCanvas(thumbHost, THUMB_PX, THUMB_PX);
      card.addEventListener('click', () => {
        if (window.CDD && typeof window.CDD.goTo === 'function') {
          window.CDD.goTo(i);
        }
      });
      cardRefs.push({ card, ctx: cv.ctx });
    }

    // Closing line.
    el('p', {
      class: 's10-closing',
      text: 'That is a convolutional network.',
    }, wrap);

    function paintAll() {
      const t = window.Drawing.tokens();
      for (let i = 0; i < cardRefs.length; i++) {
        const { ctx } = cardRefs[i];
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, THUMB_PX, THUMB_PX);
        const draw = drawers[i];
        if (draw) {
          try { draw(ctx, THUMB_PX, THUMB_PX, t); }
          catch (e) { console.error('Scene 10 thumb', i, 'failed:', e); }
        }
      }
    }
    paintAll();

    // Repaint on theme toggle.
    const themeObserver = new MutationObserver(() => paintAll());
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    return {
      onEnter() { paintAll(); },
      // No internal step engine. ArrowRight does nothing because we are at the end:
      // returning false lets the driver advance, but goTo guards against out-of-range.
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene10 = function (root) { return buildScene(root); };
})();
