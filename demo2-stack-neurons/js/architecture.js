/* Demo 2 — Network architecture diagram.

   Renders the classical 1-hidden-layer MLP topology beneath the neuron
   strip. Hidden nodes sit at horizontal positions equal to each neuron's
   kink x (via bus.sx), so a strip dot, a kink tick on the plot, and a
   hidden node here all share the same vertical line. Hovering or
   focusing a hidden node broadcasts through the bus, lighting all
   three.

   Edges encode signed weights:
     - top edges (input -> hidden) coloured by sign(w1), width by |w1|
     - bottom edges (hidden -> output) coloured by sign(v),  width by |v|

   Width 0 (linear regression) renders just [x] -> [y] with a thin edge
   labelled "linear regression".

   This module mounts into <svg id="arch-svg" viewBox="0 0 720 110">. It
   only writes into that SVG; the foundation (js/main.js) creates the
   element and exposes the state bus on window.demo2. */
(function () {
  function init() {
    if (!window.demo2) {
      console.error('window.demo2 missing -- did js/main.js load?');
      return;
    }
    const bus = window.demo2;
    const svg = document.getElementById('arch-svg');
    if (!svg) return;

    const NS = 'http://www.w3.org/2000/svg';

    // ---- viewBox geometry -------------------------------------------------
    // viewBox is 0 0 720 110, identical horizontal extent to the plot's
    // viewBox, so bus.sx (a plot-coordinate x mapper into [M.left .. 720-M.right])
    // can be reused directly to align hidden nodes under their kink ticks.
    const VB_W = 720;
    const IN_X = 360, IN_Y = 14;       // input node centre
    const OUT_X = 360, OUT_Y = 96;     // output node centre
    const HIDDEN_Y = 55;                // hidden row centre
    const IO_R = 12;
    const HIDDEN_R = 5;
    const HIDDEN_R_FOCUSED = 7;
    const EDGE_PAD = 6;                 // pinned-edge cx for out-of-view kinks
    const STROKE_MIN = 0.5;
    const STROKE_MAX = 3.5;

    function el(tag, attrs) {
      const node = document.createElementNS(NS, tag);
      if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
      return node;
    }

    // ---- DOM scaffolding (created once) -----------------------------------

    // Paint order: edges first (behind), then hidden nodes, then I/O nodes
    // on top so labels remain crisp where edges meet the disks.
    const gEdgesIn  = el('g', { class: 'edges-in' });
    const gEdgesOut = el('g', { class: 'edges-out' });
    const gHidden   = el('g', { class: 'hidden-nodes' });
    const gIO       = el('g', { class: 'io-nodes' });
    svg.appendChild(gEdgesIn);
    svg.appendChild(gEdgesOut);
    svg.appendChild(gHidden);
    svg.appendChild(gIO);

    // Input/output nodes never change. Render once and keep references in
    // case we ever need to toggle classes from state (we don't today).
    function renderIONodes() {
      while (gIO.firstChild) gIO.removeChild(gIO.firstChild);

      const inCircle = el('circle', {
        class: 'io-node',
        cx: IN_X, cy: IN_Y, r: IO_R,
      });
      const inLabel = el('text', {
        class: 'io-label',
        x: IN_X, y: IN_Y,
        'text-anchor': 'middle',
        'dominant-baseline': 'central',
      });
      inLabel.textContent = 'x';
      gIO.appendChild(inCircle);
      gIO.appendChild(inLabel);

      const outCircle = el('circle', {
        class: 'io-node',
        cx: OUT_X, cy: OUT_Y, r: IO_R,
      });
      const outLabel = el('text', {
        class: 'io-label',
        x: OUT_X, y: OUT_Y,
        'text-anchor': 'middle',
        'dominant-baseline': 'central',
      });
      outLabel.textContent = 'y';
      gIO.appendChild(outCircle);
      gIO.appendChild(outLabel);
    }
    renderIONodes();

    // ---- helpers ----------------------------------------------------------

    function clearChildren(g) {
      while (g.firstChild) g.removeChild(g.firstChild);
    }

    // Linear map of |weight| onto [STROKE_MIN, STROKE_MAX] using the
    // population maximum so a single huge magnitude does not crush
    // everyone else flat. sqrt softens the contrast without hiding sign.
    function strokeWidthFor(absW, maxAbs) {
      if (!(maxAbs > 0)) return (STROKE_MIN + STROKE_MAX) / 2;
      const t = Math.sqrt(Math.min(1, absW / maxAbs));
      return STROKE_MIN + t * (STROKE_MAX - STROKE_MIN);
    }

    // Compute the architecture x-coordinate for a kink. In-view kinks use
    // the same projection as the plot/strip; out-of-view ones pin to the
    // appropriate edge of the architecture viewBox so they remain
    // hoverable / clickable but visually muted (handled by .out-of-view).
    function archCxFor(k, state) {
      if (k.inView) return state.sx(k.x);
      return k.x > state.data.xMax ? VB_W - EDGE_PAD : EDGE_PAD;
    }

    // ---- render loop ------------------------------------------------------

    function render(state) {
      clearChildren(gEdgesIn);
      clearChildren(gEdgesOut);
      clearChildren(gHidden);

      const fit = state.fit;

      // Linear regression mode: just a thin annotated edge from x to y.
      if (!fit || fit.kind === 'linear') {
        const edge = el('line', {
          class: 'edge-linear',
          x1: IN_X, y1: IN_Y + IO_R,
          x2: OUT_X, y2: OUT_Y - IO_R,
        });
        gEdgesIn.appendChild(edge);

        const note = el('text', {
          class: 'edge-linear-note',
          x: IN_X + 10, y: (IN_Y + OUT_Y) / 2,
          'dominant-baseline': 'central',
        });
        note.textContent = 'linear regression';
        gEdgesIn.appendChild(note);
        return;
      }

      const kinks = state.kinks || [];
      if (kinks.length === 0) return;

      // Population max magnitudes drive stroke-width scaling.
      let maxAbsW1 = 0, maxAbsV = 0;
      for (const k of kinks) {
        const aw = Math.abs(k.w1), av = Math.abs(k.v);
        if (aw > maxAbsW1) maxAbsW1 = aw;
        if (av > maxAbsV) maxAbsV = av;
      }

      const focused = typeof state.effectiveFocus === 'function'
        ? state.effectiveFocus()
        : state.focusedNeuron;
      const xMode = state.hover && state.hover.kind === 'x';
      const firingSet = xMode
        ? new Set(state.firingNeuronsAt(state.hover.x))
        : null;

      // Decide cross-component highlight class for one neuron index.
      // Order:
      //   1. focused -> 'focused'
      //   2. firing-at-hovered-x -> 'firing'
      //   3. some other neuron is focused -> 'dimmed'
      //   4. otherwise -> '' (no class)
      function highlightClass(idx) {
        if (focused === idx) return 'focused';
        if (firingSet && firingSet.has(idx)) return 'firing';
        if (focused != null && focused !== idx) return 'dimmed';
        return '';
      }

      for (const k of kinks) {
        const cx = archCxFor(k, state);
        const outOfView = !k.inView;
        const hl = highlightClass(k.idx);

        // ----- top edge: input -> hidden -------------------------------
        const topClasses = ['edge-top'];
        topClasses.push(k.w1 > 0 ? 'fires-right' : 'fires-left');
        if (outOfView) topClasses.push('out-of-view');
        if (hl) topClasses.push(hl);

        const topEdge = el('line', {
          class: topClasses.join(' '),
          x1: IN_X, y1: IN_Y + IO_R,
          x2: cx,    y2: HIDDEN_Y - HIDDEN_R,
        });
        topEdge.setAttribute(
          'stroke-width',
          strokeWidthFor(Math.abs(k.w1), maxAbsW1).toFixed(2)
        );
        gEdgesIn.appendChild(topEdge);

        // ----- bottom edge: hidden -> output ---------------------------
        const botClasses = ['edge-bottom'];
        botClasses.push(k.v > 0 ? 'fires-right' : 'fires-left');
        if (outOfView) botClasses.push('out-of-view');
        if (hl) botClasses.push(hl);

        const botEdge = el('line', {
          class: botClasses.join(' '),
          x1: cx,    y1: HIDDEN_Y + HIDDEN_R,
          x2: OUT_X, y2: OUT_Y - IO_R,
        });
        botEdge.setAttribute(
          'stroke-width',
          strokeWidthFor(Math.abs(k.v), maxAbsV).toFixed(2)
        );
        gEdgesOut.appendChild(botEdge);

        // ----- hidden node ---------------------------------------------
        const nodeClasses = ['hidden-node'];
        nodeClasses.push(k.firesRight ? 'fires-right' : 'fires-left');
        if (outOfView) nodeClasses.push('out-of-view');
        if (hl) nodeClasses.push(hl);

        const r = hl === 'focused' ? HIDDEN_R_FOCUSED : HIDDEN_R;
        const node = el('circle', {
          class: nodeClasses.join(' '),
          cx: cx, cy: HIDDEN_Y, r: r,
        });
        node.dataset.idx = String(k.idx);

        // Hover broadcast. mouseleave only clears the bus when *we*
        // currently own the hover, otherwise a strip-driven hover would
        // get nuked when the cursor sails over a stale architecture node.
        node.addEventListener('mouseenter', function () {
          if (bus.setHover) bus.setHover({ kind: 'neuron', idx: k.idx });
        });
        node.addEventListener('mouseleave', function () {
          if (bus.setHover && bus.hover && bus.hover.kind === 'neuron' && bus.hover.idx === k.idx) {
            bus.setHover(null);
          }
        });
        // Click toggles persistent focus, mirroring strip dot behaviour.
        node.addEventListener('click', function (ev) {
          ev.stopPropagation();
          bus.setFocus(bus.focusedNeuron === k.idx ? null : k.idx);
        });

        gHidden.appendChild(node);
      }
    }

    // Background click on the architecture SVG clears focus, matching
    // the plot and strip backgrounds.
    svg.addEventListener('click', function (e) {
      if (e.target === svg && bus.focusedNeuron != null) {
        bus.setFocus(null);
      }
    });

    bus.onChange(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
