/* Scene 2 — Live neuron strip + compact architecture diagram.

   Combined module: builds two stacked SVGs that share a 720-wide viewBox so
   strip dots, architecture hidden-row nodes, and plot kink ticks all line
   up vertically. Both layers re-render on every bus.onUpdate. Strip dot
   radius scales with |v|; color encodes sign of w1. Architecture top edges
   thicken with |w1|, bottom edges with |v|; colors match.

   The strip caption above the SVG explains the encoding so the lecturer
   doesn't have to.

   Mounted via `buildStripArch(root, bus)`. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const RENDER_MIN_DT_MS = 32;

  const STRIP_VB_W = 720;
  const STRIP_VB_H = 80;
  const STRIP_CY = 40;
  const STRIP_R_MIN = 3.5;
  const STRIP_R_MAX = 10;
  const EDGE_PAD = 6;

  const ARCH_VB_W = 720;
  const ARCH_VB_H = 110;
  const ARCH_IN_X = 360, ARCH_IN_Y = 14;
  const ARCH_OUT_X = 360, ARCH_OUT_Y = 96;
  const ARCH_HIDDEN_Y = 55;
  const ARCH_IO_R = 12;
  const ARCH_HIDDEN_R = 4.5;
  const ARCH_STROKE_MIN = 0.5;
  const ARCH_STROKE_MAX = 3.5;

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function radiusFor(absV, maxAbsV) {
    if (!(maxAbsV > 0)) return (STRIP_R_MIN + STRIP_R_MAX) / 2;
    const t = Math.sqrt(Math.min(1, absV / maxAbsV));
    return STRIP_R_MIN + t * (STRIP_R_MAX - STRIP_R_MIN);
  }

  function strokeFor(absW, maxAbs) {
    if (!(maxAbs > 0)) return (ARCH_STROKE_MIN + ARCH_STROKE_MAX) / 2;
    const t = Math.sqrt(Math.min(1, absW / maxAbs));
    return ARCH_STROKE_MIN + t * (ARCH_STROKE_MAX - ARCH_STROKE_MIN);
  }

  // Build kink data from raw params, mirroring demo2's kinkData(fit).
  function kinkData(p, xMin, xMax) {
    const out = [];
    for (let i = 0; i < p.w1.length; i++) {
      const w1 = p.w1[i], b = p.bias[i], v = p.v[i];
      if (Math.abs(w1) < 1e-8) {
        out.push({ idx: i, x: 0, w1, bias: b, v, inView: false, firesRight: w1 > 0 });
        continue;
      }
      const x = -b / w1;
      const inView = x >= xMin && x <= xMax;
      out.push({
        idx: i, x, w1, bias: b, v, inView, firesRight: w1 > 0,
      });
    }
    return out;
  }

  // -------- entry point ----------------------------------------------------

  function buildStripArch(root, bus) {
    if (!root) throw new Error('buildStripArch: root missing');
    if (!bus || !bus.engine) throw new Error('buildStripArch: bus.engine missing');
    const D = window.GDV_DATA;
    if (!D) throw new Error('buildStripArch: GDV_DATA missing');

    while (root.firstChild) root.removeChild(root.firstChild);

    // Encoding caption (italic, sits above the strip and matches demo2).
    const cap = document.createElement('div');
    cap.className = 's2-strip-caption';
    cap.innerHTML =
      'each dot is one hidden unit · size encodes |v| · ' +
      'color encodes sign of w<sub>1</sub> ' +
      '(blue = fires right, amber = fires left)';
    root.appendChild(cap);

    // Strip container.
    const stripWrap = document.createElement('div');
    stripWrap.className = 's2-neuron-strip';
    root.appendChild(stripWrap);

    const stripSvg = el('svg', {
      class: 's2-strip',
      viewBox: `0 0 ${STRIP_VB_W} ${STRIP_VB_H}`,
      preserveAspectRatio: 'xMidYMid meet',
    });
    stripWrap.appendChild(stripSvg);
    const gDots = el('g', { class: 'strip-dots' });
    stripSvg.appendChild(gDots);
    const emptyCap = el('text', {
      class: 'strip-empty-caption',
      x: STRIP_VB_W / 2, y: STRIP_CY + 5,
      'text-anchor': 'middle',
    });
    emptyCap.textContent = 'no hidden units yet';
    stripSvg.appendChild(emptyCap);

    // Architecture container.
    const archWrap = document.createElement('div');
    archWrap.className = 's2-architecture';
    root.appendChild(archWrap);

    const archSvg = el('svg', {
      class: 's2-arch',
      viewBox: `0 0 ${ARCH_VB_W} ${ARCH_VB_H}`,
      preserveAspectRatio: 'xMidYMid meet',
    });
    archWrap.appendChild(archSvg);

    const gEdgesIn = el('g', { class: 'edges-in' });
    const gEdgesOut = el('g', { class: 'edges-out' });
    const gHidden = el('g', { class: 'hidden-nodes' });
    const gIO = el('g', { class: 'io-nodes' });
    archSvg.appendChild(gEdgesIn);
    archSvg.appendChild(gEdgesOut);
    archSvg.appendChild(gHidden);
    archSvg.appendChild(gIO);

    // Static input/output disks.
    function renderIO() {
      while (gIO.firstChild) gIO.removeChild(gIO.firstChild);
      const inC = el('circle', { class: 'io-node', cx: ARCH_IN_X, cy: ARCH_IN_Y, r: ARCH_IO_R });
      const inL = el('text', {
        class: 'io-label', x: ARCH_IN_X, y: ARCH_IN_Y,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
      });
      inL.textContent = 'x';
      gIO.appendChild(inC); gIO.appendChild(inL);

      const outC = el('circle', { class: 'io-node', cx: ARCH_OUT_X, cy: ARCH_OUT_Y, r: ARCH_IO_R });
      const outL = el('text', {
        class: 'io-label', x: ARCH_OUT_X, y: ARCH_OUT_Y,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
      });
      outL.textContent = 'y';
      gIO.appendChild(outC); gIO.appendChild(outL);
    }
    renderIO();

    // Map plot-x to viewBox-x, matching the plot's sx so columns align.
    // Plot uses left margin 48 and right margin 24 within VB_W=720, so we
    // replicate that here.
    const PLOT_M_LEFT = 48, PLOT_M_RIGHT = 24;
    const PLOT_W = STRIP_VB_W - PLOT_M_LEFT - PLOT_M_RIGHT;
    function sx(x) {
      return PLOT_M_LEFT + (x - D.xMin) / (D.xMax - D.xMin) * PLOT_W;
    }

    // -------- render ------------------------------------------------------

    function renderStrip(p) {
      while (gDots.firstChild) gDots.removeChild(gDots.firstChild);
      const ks = kinkData(p, D.xMin, D.xMax);
      if (ks.length === 0) {
        emptyCap.style.display = '';
        return;
      }
      emptyCap.style.display = 'none';

      let maxAbsV = 0, anyInView = false;
      for (const k of ks) if (k.inView) anyInView = true;
      for (const k of ks) {
        if (anyInView && !k.inView) continue;
        const a = Math.abs(k.v);
        if (a > maxAbsV) maxAbsV = a;
      }

      for (const k of ks) {
        let cx;
        if (k.inView) cx = sx(k.x);
        else if (k.x > D.xMax) cx = STRIP_VB_W - EDGE_PAD;
        else cx = EDGE_PAD;

        const r = radiusFor(Math.abs(k.v), maxAbsV);
        const cls = ['neuron-dot', k.firesRight ? 'fires-right' : 'fires-left'];
        if (!k.inView) cls.push('out-of-view');

        gDots.appendChild(el('circle', {
          class: cls.join(' '),
          cx: cx,
          cy: STRIP_CY,
          r: (!k.inView ? Math.max(STRIP_R_MIN - 0.5, 3) : r).toFixed(2),
        }));
      }
    }

    function renderArch(p) {
      while (gEdgesIn.firstChild) gEdgesIn.removeChild(gEdgesIn.firstChild);
      while (gEdgesOut.firstChild) gEdgesOut.removeChild(gEdgesOut.firstChild);
      while (gHidden.firstChild) gHidden.removeChild(gHidden.firstChild);

      const ks = kinkData(p, D.xMin, D.xMax);
      if (ks.length === 0) return;

      let maxAbsW1 = 0, maxAbsV = 0;
      for (const k of ks) {
        const aw = Math.abs(k.w1), av = Math.abs(k.v);
        if (aw > maxAbsW1) maxAbsW1 = aw;
        if (av > maxAbsV) maxAbsV = av;
      }

      for (const k of ks) {
        let cx;
        if (k.inView) cx = sx(k.x);
        else if (k.x > D.xMax) cx = ARCH_VB_W - EDGE_PAD;
        else cx = EDGE_PAD;

        const outOfView = !k.inView;

        // top edge: input -> hidden, color by sign(w1), width by |w1|
        const topClasses = ['edge-top'];
        topClasses.push(k.w1 > 0 ? 'fires-right' : 'fires-left');
        if (outOfView) topClasses.push('out-of-view');
        const topEdge = el('line', {
          class: topClasses.join(' '),
          x1: ARCH_IN_X, y1: ARCH_IN_Y + ARCH_IO_R,
          x2: cx, y2: ARCH_HIDDEN_Y - ARCH_HIDDEN_R,
          'stroke-width': strokeFor(Math.abs(k.w1), maxAbsW1).toFixed(2),
        });
        gEdgesIn.appendChild(topEdge);

        // bottom edge: hidden -> output, color by sign(v), width by |v|
        const botClasses = ['edge-bottom'];
        botClasses.push(k.v > 0 ? 'fires-right' : 'fires-left');
        if (outOfView) botClasses.push('out-of-view');
        const botEdge = el('line', {
          class: botClasses.join(' '),
          x1: cx, y1: ARCH_HIDDEN_Y + ARCH_HIDDEN_R,
          x2: ARCH_OUT_X, y2: ARCH_OUT_Y - ARCH_IO_R,
          'stroke-width': strokeFor(Math.abs(k.v), maxAbsV).toFixed(2),
        });
        gEdgesOut.appendChild(botEdge);

        // hidden node
        const nodeClasses = ['hidden-node'];
        nodeClasses.push(k.firesRight ? 'fires-right' : 'fires-left');
        if (outOfView) nodeClasses.push('out-of-view');
        gHidden.appendChild(el('circle', {
          class: nodeClasses.join(' '),
          cx: cx, cy: ARCH_HIDDEN_Y, r: ARCH_HIDDEN_R,
        }));
      }
    }

    let lastRender = 0;
    function render() {
      const now = Date.now();
      if (now - lastRender < RENDER_MIN_DT_MS) return;
      lastRender = now;
      const p = bus.engine.params();
      renderStrip(p);
      renderArch(p);
    }

    function renderNow() {
      lastRender = Date.now();
      const p = bus.engine.params();
      renderStrip(p);
      renderArch(p);
    }
    renderNow();

    bus.onUpdate(render);

    return { renderNow };
  }

  window.GDVScene2 = window.GDVScene2 || {};
  window.GDVScene2.buildStripArch = buildStripArch;
})();
