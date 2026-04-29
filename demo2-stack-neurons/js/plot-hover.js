/* Demo 2 -- Plot hover layer.

   Listens for mouse events on #plot and converts them into x-mode hover
   on the state bus, then renders a vertical guide line, a marker dot at
   the bold fit, and a small readout (x = ..., y = ..., n firing).

   Cleared when bus.hover is null or kind === 'neuron' (so a neuron-strip
   hover from another component takes precedence and this layer steps
   aside without clobbering it). */
(function () {
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  // Render minus as a real "−" so widths stay uniform under tabular nums.
  function fmt(v) {
    const s = (Math.abs(v)).toFixed(2);
    return (v < 0 ? '−' : '') + s;
  }

  function init() {
    if (!window.demo2) {
      console.error('window.demo2 missing -- plot-hover cannot mount');
      return;
    }
    const bus = window.demo2;
    const svg = document.getElementById('plot');
    const layer = document.querySelector('#plot .hover-guide');
    if (!svg || !layer) return;

    const M = bus.M;
    const plotW = bus.plotW;
    const plotH = bus.plotH;
    const data = bus.data;
    const xMin = data.xMin;
    const xMax = data.xMax;

    // Plot data-region bounds in viewBox coords.
    const regionLeft = M.left;
    const regionRight = M.left + plotW;
    const regionTop = M.top;
    const regionBottom = M.top + plotH;

    // ---------------- Mouse -> data x ------------------------------------

    let svgPoint = null;
    function clientToSvg(evt) {
      // Use createSVGPoint + getScreenCTM().inverse() so we work no matter
      // how the SVG is sized or transformed by surrounding CSS.
      if (!svgPoint) svgPoint = svg.createSVGPoint();
      svgPoint.x = evt.clientX;
      svgPoint.y = evt.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return null;
      return svgPoint.matrixTransform(ctm.inverse());
    }

    function invSx(svgX) {
      return xMin + (svgX - M.left) / plotW * (xMax - xMin);
    }

    function inRegion(p) {
      return p.x >= regionLeft && p.x <= regionRight
          && p.y >= regionTop && p.y <= regionBottom;
    }

    function onMouseMove(evt) {
      const p = clientToSvg(evt);
      if (!p) return;
      if (!inRegion(p)) {
        // Outside the data region: clear x-hover (but leave neuron-hovers alone).
        if (bus.hover && bus.hover.kind === 'x') bus.setHover(null);
        return;
      }
      let x = invSx(p.x);
      if (x < xMin) x = xMin;
      else if (x > xMax) x = xMax;
      bus.setHover({ kind: 'x', x: x });
    }

    function onMouseLeave() {
      if (bus.hover && bus.hover.kind === 'x') bus.setHover(null);
    }

    svg.addEventListener('mousemove', onMouseMove);
    svg.addEventListener('mouseleave', onMouseLeave);
    // mouseenter fires before the first mousemove on entry; we don't need
    // its data, but listening to it documents intent and silences any
    // edge case where mousemove is throttled before entering.
    svg.addEventListener('mouseenter', onMouseMove);

    // ---------------- Render --------------------------------------------

    function clear() {
      while (layer.firstChild) layer.removeChild(layer.firstChild);
    }

    function render(state) {
      clear();
      const hov = state.hover;
      if (!hov || hov.kind !== 'x') return;

      const x = hov.x;
      const y = state.evalFitAt(x);
      const sx = state.sx;
      const sy = state.sy;
      const cx = sx(x);
      const cy = sy(y);

      // Vertical guide line (full plot height).
      layer.appendChild(el('line', {
        class: 'hover-guide-line',
        x1: cx.toFixed(1), x2: cx.toFixed(1),
        y1: regionTop, y2: regionBottom,
      }));

      // Marker dot at the fit.
      layer.appendChild(el('circle', {
        class: 'hover-guide-dot',
        cx: cx.toFixed(1), cy: cy.toFixed(1), r: 3.5,
      }));

      // Readout group.
      const firing = state.firingNeuronsAt(x);
      const nFiring = firing.length;
      const line1 = 'x = ' + fmt(x) + '   y = ' + fmt(y);
      const line2 = nFiring + ' firing';

      // Approximate measurement: 12px monospaced, ~7.2px per char.
      const charW = 7.2;
      const padX = 6, padY = 4;
      const lineH = 14;
      const textW = Math.max(line1.length, line2.length) * charW;
      const boxW = Math.ceil(textW + padX * 2);
      const boxH = padY * 2 + lineH * 2;

      // Default position: above-and-right of the marker dot.
      // Spec asks for (sx(x) + 8, sy(y) - 28) -- treat that as the box origin.
      let bx = cx + 8;
      let by = cy - 28;

      // Flip horizontally if it would overflow the plot's right edge.
      if (bx + boxW > regionRight) {
        bx = cx - 8 - boxW;
      }
      // If still off the left edge, clamp to left.
      if (bx < regionLeft) bx = regionLeft + 2;

      // Flip vertically if it would clip the top.
      if (by < regionTop) {
        by = cy + 12;
        // If that overflows the bottom, clamp.
        if (by + boxH > regionBottom) by = regionBottom - boxH - 2;
      }

      const g = el('g', { class: 'hover-readout' });
      g.appendChild(el('rect', {
        class: 'hover-readout-bg',
        x: bx.toFixed(1), y: by.toFixed(1),
        width: boxW, height: boxH,
        rx: 4, ry: 4,
      }));

      const t1 = el('text', {
        class: 'hover-readout-text',
        x: (bx + padX).toFixed(1),
        y: (by + padY + lineH - 3).toFixed(1),
      });
      t1.textContent = line1;
      g.appendChild(t1);

      const t2 = el('text', {
        class: 'hover-readout-text hover-readout-sub',
        x: (bx + padX).toFixed(1),
        y: (by + padY + lineH * 2 - 3).toFixed(1),
      });
      t2.textContent = line2;
      g.appendChild(t2);

      layer.appendChild(g);
    }

    bus.onChange(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
