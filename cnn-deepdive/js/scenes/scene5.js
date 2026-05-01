/* Scene 5 -- "Race the detectors."

   Six detectors scan a complex 64x64 test image at the same time. As each
   slides over a position whose response exceeds its per-kernel threshold,
   it emits a colored particle. Counters tick up.

   Architecture:
     - On enter, build the test image once (composite of placed shapes).
       Run conv2d for each detector once, cache the response maps.
     - The "scan" is purely visual: a 5x5 outline rectangle sweeps over
       the test image. The particle emission schedule is precomputed --
       each above-threshold position turns into a particle whose emit
       time matches when the scanner crosses it.
     - rAF integrates particle physics: small upward drift + fade.
     - The post-animation state ("?test=raced") is the final frame:
       all particles dispersed, counters at totals.

   `&run` auto-clicks Start 200ms after build for headless capture. */
(function () {
  'use strict';

  const IMG_PX = 64;          // logical px = grid cells (1 px per cell here)
  const IMG_DRAW_PX = 384;    // upscaled display
  const KERN_PX = 56;
  const MINI_PX = 80;

  const DETECTORS = [
    { key: 'vertical',   label: 'vertical',   colorVar: '--cnn-pos' },     // blue
    { key: 'horizontal', label: 'horizontal', colorVar: '--cnn-accent' },  // amber
    { key: 'diag_down',  label: 'diag ↘',     colorVar: '--cnn-purple' }, // purple
    { key: 'dot',        label: 'dot',        colorVar: '--cnn-green' },   // green
    { key: 'ring',       label: 'ring',       colorVar: '--cnn-neg' },     // red
    { key: 'top_half',   label: 'top half',   colorVar: '--ink' },         // ink
  ];

  const SCAN_DURATION_MS = 6000;
  const PARTICLE_LIFE_MS = 1500;
  const MAX_PARTICLES = 1500;

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

  function getCssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* Composite the test image: blank 64x64, paint several samples at
     various offsets via a downscaled stamp. */
  function buildTestImage() {
    const out = window.CNN.zeros2D(IMG_PX, IMG_PX);
    function stamp(sample, top, left) {
      const sh = sample.length, sw = sample[0].length;
      for (let i = 0; i < sh; i++) {
        for (let j = 0; j < sw; j++) {
          const ti = top + i, tj = left + j;
          if (ti < 0 || ti >= IMG_PX || tj < 0 || tj >= IMG_PX) continue;
          if (sample[i][j] > out[ti][tj]) out[ti][tj] = sample[i][j];
        }
      }
    }
    // Six placed shapes scattered to give every detector something.
    stamp(window.Drawing.makeSample('cross', 14),      4,  4);
    stamp(window.Drawing.makeSample('circle', 14),     4, 44);
    stamp(window.Drawing.makeSample('triangle', 16),  42, 38);
    stamp(window.Drawing.makeSample('vertical', 14), 24, 22);
    stamp(window.Drawing.makeSample('horizontal', 14), 46, 4);
    stamp(window.Drawing.makeSample('L', 14),         24, 46);
    return out;
  }

  function paintTestImage(host) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, IMG_DRAW_PX, IMG_DRAW_PX);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    return { ctx, w, h };
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

  function paintMiniHeatmap(host, response, vmax) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, MINI_PX, MINI_PX);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, response, 0, 0, w, h, {
      diverging: true, valueRange: [-vmax, vmax],
    });
  }

  /* Walk a response map and threshold; return [{i, j, v}] for cells where
     the response exceeds the threshold. */
  function aboveThresholdCells(response, threshold) {
    const out = [];
    const H = response.length, W = response[0].length;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (response[i][j] >= threshold) {
          out.push({ i, j, v: response[i][j] });
        }
      }
    }
    return out;
  }

  /* Per-kernel threshold: 60% of the kernel's positive dynamic range. */
  function chooseThreshold(response) {
    const r = window.CNN.range2D(response);
    const m = Math.max(r.hi, 0);
    return 0.6 * m;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters) {
      root.innerHTML = '<p style="opacity:0.5">handFilters missing.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s5-root');
    const wrap = el('div', { class: 's5-wrap' }, root);

    // ---- Hero ----------------------------------------------------------
    const hero = el('header', { class: 'hero s5-hero' }, wrap);
    el('h1', { text: 'Race the detectors.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Six hand-designed detectors scan the same picture at the same time. Each fires where its pattern lives.',
    }, hero);

    // ---- Layout: detectors left | image centre | detectors right ------
    const grid = el('div', { class: 's5-grid' }, wrap);
    const colLeft = el('div', { class: 's5-side s5-side-left' }, grid);
    const colMid = el('div', { class: 's5-mid' }, grid);
    const colRight = el('div', { class: 's5-side s5-side-right' }, grid);

    // Image card
    const imageCard = el('div', { class: 'card s5-image-card' }, colMid);
    el('div', { class: 's5-mid-label', text: 'test image' }, imageCard);
    const imageHost = el('div', { class: 'canvas-host s5-image-host' }, imageCard);

    // Build detector cards. First three on the left, last three on the right.
    function buildDetectorCard(parent, dec) {
      const card = el('div', { class: 'card s5-dcard' }, parent);
      card.style.setProperty('--s5-color', `var(${dec.colorVar})`);
      const head = el('div', { class: 's5-dcard-head' }, card);
      el('div', { class: 's5-dcard-swatch' }, head);
      el('div', { class: 's5-dcard-name', text: dec.label }, head);
      const counterEl = el('div', { class: 's5-dcard-counter', text: '0' }, head);

      const body = el('div', { class: 's5-dcard-body' }, card);
      const kHost = el('div', { class: 'canvas-host s5-kernel-host' }, body);
      const miniHost = el('div', { class: 'canvas-host s5-mini-host' }, body);

      return { card, kHost, miniHost, counterEl };
    }
    const dcards = {};
    DETECTORS.forEach((dec, idx) => {
      const where = idx < 3 ? colLeft : colRight;
      dcards[dec.key] = buildDetectorCard(where, dec);
    });

    // ---- Controls ------------------------------------------------------
    const controls = el('div', { class: 'controls s5-controls' }, wrap);
    const startBtn = el('button', { type: 'button', class: 'primary',
      text: 'Start' }, controls);
    const resetBtn = el('button', { type: 'button', text: 'Reset' }, controls);
    const totalGroup = el('div', { class: 'control-group s5-total' }, controls);
    el('label', { text: 'total fires', for: 's5-total-out' }, totalGroup);
    const totalOut = el('output', { id: 's5-total-out', class: 'control-value', text: '0' }, totalGroup);

    el('p', {
      class: 'caption s5-caption',
      text: 'Each detector races across the picture. The number of fires is the sum of where each one matched.',
    }, wrap);

    el('p', {
      class: 'footnote',
      text: 'Threshold per detector = 60% of its positive peak response.',
    }, wrap);

    // ---- State ---------------------------------------------------------
    const state = {
      input: null,
      responses: {},
      thresholds: {},
      events: {},          // key -> [{x, y, t}]   t in [0,1]
      counts: {},
      particles: [],       // active particles
      animating: false,
      raced: false,        // post-animation rendered state
      rafId: null,
      startTime: 0,
    };

    function precompute() {
      state.input = buildTestImage();
      let globalVmax = 0;
      DETECTORS.forEach(dec => {
        const f = window.DATA.handFilters[dec.key];
        const resp = window.CNN.conv2d(state.input, f, 2);
        state.responses[dec.key] = resp;
        const r = window.CNN.range2D(resp);
        const v = Math.max(Math.abs(r.lo), Math.abs(r.hi));
        if (v > globalVmax) globalVmax = v;
      });
      // Per-kernel thresholds + emission schedules.
      DETECTORS.forEach(dec => {
        const resp = state.responses[dec.key];
        const thr = chooseThreshold(resp);
        state.thresholds[dec.key] = thr;
        const cells = aboveThresholdCells(resp, thr);
        // Schedule each cell's emission time t in [0, 1] by its column
        // (column-major sweep, so the scanner sweeps left-to-right).
        const events = cells.map(c => ({
          x: c.j, y: c.i, t: c.j / (IMG_PX - 1),
        }));
        // Stable sort by t; randomise tiny tie-breaker so detectors don't
        // emit in lock-step on the same column.
        events.forEach(e => { e.t += (e.y / (IMG_PX - 1)) * 0.04; });
        events.sort((a, b) => a.t - b.t);
        state.events[dec.key] = events;
        state.counts[dec.key] = 0;
      });

      // Cache the global vmax for mini-heatmaps so the detectors share scale.
      state.vmax = globalVmax || 1;
    }

    function paintAll() {
      // Test image
      const setup = paintTestImage(imageHost);
      const { ctx, w, h } = setup;
      // Draw input as upscaled grayscale.
      const cellW = w / IMG_PX, cellH = h / IMG_PX;
      const t = window.Drawing.tokens();
      window.Drawing.drawGrid(ctx, state.input, 0, 0, w, h,
        { diverging: false, valueRange: [0, 1] });
      void t; void cellW; void cellH;

      // Detector cards
      DETECTORS.forEach(dec => {
        paintKernel(dcards[dec.key].kHost, window.DATA.handFilters[dec.key]);
        paintMiniHeatmap(dcards[dec.key].miniHost, state.responses[dec.key], state.vmax);
        dcards[dec.key].counterEl.textContent = String(state.counts[dec.key]);
      });

      updateTotal();
    }

    function updateTotal() {
      let total = 0;
      DETECTORS.forEach(dec => total += state.counts[dec.key]);
      totalOut.textContent = String(total);
    }

    /* The animation overlay sits on top of the test image. It hosts:
        - per-detector scanner rectangles
        - particles
       Drawn as a separate canvas of the same logical size so we can clear
       the overlay each frame without touching the underlying picture. */
    let overlayCanvas, overlayCtx, overlayW, overlayH;
    function ensureOverlay() {
      if (overlayCanvas) return;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      overlayCanvas = document.createElement('canvas');
      overlayCanvas.className = 's5-overlay';
      overlayCanvas.width = Math.round(IMG_DRAW_PX * dpr);
      overlayCanvas.height = Math.round(IMG_DRAW_PX * dpr);
      overlayCanvas.style.width = IMG_DRAW_PX + 'px';
      overlayCanvas.style.height = IMG_DRAW_PX + 'px';
      imageHost.appendChild(overlayCanvas);
      overlayCtx = overlayCanvas.getContext('2d');
      overlayCtx.scale(dpr, dpr);
      overlayW = IMG_DRAW_PX;
      overlayH = IMG_DRAW_PX;
    }

    function clearOverlay() {
      if (!overlayCtx) return;
      overlayCtx.clearRect(0, 0, overlayW, overlayH);
    }

    /* Spawn a particle at image-grid (x, y). */
    function spawnParticle(dec, x, y, now) {
      if (state.particles.length >= MAX_PARTICLES) return;
      const cellW = IMG_DRAW_PX / IMG_PX;
      const cellH = IMG_DRAW_PX / IMG_PX;
      // Slight random jitter so emissions look organic.
      const px = (x + 0.5) * cellW + (Math.random() - 0.5) * cellW * 0.6;
      const py = (y + 0.5) * cellH + (Math.random() - 0.5) * cellH * 0.6;
      const vx = (Math.random() - 0.5) * 0.025; // px per ms (small)
      const vy = -0.03 - Math.random() * 0.02;  // upward drift
      state.particles.push({
        decKey: dec.key,
        color: getCssVar(dec.colorVar) || '#888',
        x: px, y: py, vx, vy,
        born: now,
      });
    }

    function drawScanners(progress) {
      if (!overlayCtx) return;
      const cellW = IMG_DRAW_PX / IMG_PX;
      // Sweep position in image coords (column index). Range covers the
      // whole image plus the kernel width.
      const colF = progress * (IMG_PX - 1);
      DETECTORS.forEach(dec => {
        const col = colF; // every detector races at the same speed
        const x = col * cellW - cellW * 2;
        const r = 5 * cellW;
        overlayCtx.save();
        overlayCtx.strokeStyle = getCssVar(dec.colorVar) || '#888';
        overlayCtx.globalAlpha = 0.45;
        overlayCtx.lineWidth = 1.2;
        overlayCtx.strokeRect(x, 0, r, IMG_DRAW_PX);
        overlayCtx.restore();
      });
      void cellW;
    }

    function drawParticles(now) {
      if (!overlayCtx) return;
      const survivors = [];
      for (const p of state.particles) {
        const age = now - p.born;
        if (age > PARTICLE_LIFE_MS) continue;
        const dt = 16; // approximate; we step using elapsed since spawn instead.
        void dt;
        const t = age;
        const x = p.x + p.vx * t;
        const y = p.y + p.vy * t;
        const a = 1 - age / PARTICLE_LIFE_MS;
        overlayCtx.fillStyle = p.color;
        overlayCtx.globalAlpha = a;
        overlayCtx.beginPath();
        overlayCtx.arc(x, y, 3, 0, Math.PI * 2);
        overlayCtx.fill();
        survivors.push(p);
      }
      overlayCtx.globalAlpha = 1;
      state.particles = survivors;
    }

    /* Per-detector emission cursor: index into events array. */
    let cursors;
    function resetCursors() {
      cursors = {};
      DETECTORS.forEach(dec => { cursors[dec.key] = 0; });
    }

    function step(now) {
      const elapsed = now - state.startTime;
      const t = Math.min(1, elapsed / SCAN_DURATION_MS);

      // Emit all events whose t has passed since last frame.
      DETECTORS.forEach(dec => {
        const evs = state.events[dec.key];
        let k = cursors[dec.key];
        while (k < evs.length && evs[k].t <= t) {
          spawnParticle(dec, evs[k].x, evs[k].y, now);
          state.counts[dec.key] += 1;
          k++;
        }
        cursors[dec.key] = k;
        dcards[dec.key].counterEl.textContent = String(state.counts[dec.key]);
      });
      updateTotal();

      // Render
      clearOverlay();
      if (t < 1) drawScanners(t);
      drawParticles(now);

      if (t < 1 || state.particles.length > 0) {
        state.rafId = requestAnimationFrame(step);
      } else {
        state.animating = false;
        state.raced = true;
        startBtn.disabled = false;
        clearOverlay();
      }
    }

    function startAnimation() {
      if (state.animating) return;
      state.animating = true;
      state.raced = false;
      DETECTORS.forEach(dec => { state.counts[dec.key] = 0; });
      state.particles = [];
      resetCursors();
      ensureOverlay();
      clearOverlay();
      paintAll();
      startBtn.disabled = true;
      state.startTime = performance.now();
      state.rafId = requestAnimationFrame(step);
    }

    function resetAll() {
      if (state.rafId != null) {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
      state.animating = false;
      state.raced = false;
      state.particles = [];
      DETECTORS.forEach(dec => { state.counts[dec.key] = 0; });
      clearOverlay();
      paintAll();
      startBtn.disabled = false;
    }

    /* Jump straight to the post-animation state: counters at totals,
       no particles, no scanners. */
    function jumpToRaced() {
      if (state.rafId != null) {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
      state.animating = false;
      state.raced = true;
      state.particles = [];
      DETECTORS.forEach(dec => {
        state.counts[dec.key] = state.events[dec.key].length;
      });
      paintAll();
      ensureOverlay();
      clearOverlay();
      startBtn.disabled = false;
    }

    startBtn.addEventListener('click', startAnimation);
    resetBtn.addEventListener('click', resetAll);

    // ---- Boot ---------------------------------------------------------
    precompute();
    paintAll();

    if (readHashFlag('test') === 'raced') {
      jumpToRaced();
    } else if (readHashFlag('run')) {
      // Auto-click Start ~200ms after build for headless capture.
      setTimeout(startAnimation, 200);
    }

    return {
      onEnter() { paintAll(); },
      onLeave() {
        if (state.rafId != null) {
          cancelAnimationFrame(state.rafId);
          state.rafId = null;
        }
        state.animating = false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene5 = function (root) { return buildScene(root); };
})();
