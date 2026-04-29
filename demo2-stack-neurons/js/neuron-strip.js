/* Demo 2 — Neuron strip layer.

   Renders one circle per active hidden unit beneath the plot card. Circles
   align horizontally with the plot's kink ticks (shared 720-wide viewBox,
   shared bus.sx). Click toggles focus on that neuron via bus.setFocus;
   hover shows a tooltip with that neuron's parameters; arrow keys cycle
   focus when the strip itself has document focus; Escape clears it. */
(function () {
  function init() {
    if (!window.demo2) {
      console.error('window.demo2 missing -- did js/main.js load?');
      return;
    }
    const bus = window.demo2;
    const stripSvg = document.getElementById('strip');
    const tooltip = document.getElementById('strip-tooltip');
    if (!stripSvg || !tooltip) return;

    const NS = 'http://www.w3.org/2000/svg';

    // Strip viewBox is 0 0 720 64. Plot viewBox is 0 0 720 380. Both render
    // at the same on-page width (parents share matching horizontal padding),
    // so cx = bus.sx(kink.x) lines a strip dot up under its kink tick.
    const VB_W = 720;
    const VB_H = 48;
    const CY = 24;             // strip vertical center
    const EDGE_PAD = 6;         // out-of-window pinned dots inset from edge
    const R_MIN = 3.5;
    const R_MAX = 10;

    // Make the strip a focusable region for keyboard navigation.
    if (!stripSvg.hasAttribute('tabindex')) {
      stripSvg.setAttribute('tabindex', '0');
    }

    function el(tag, attrs) {
      const node = document.createElementNS(NS, tag);
      if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
      return node;
    }

    // -------- formatting ---------------------------------------------------

    // Use real Unicode minus so plus and minus glyphs are the same width.
    function signed(n, digits) {
      if (!Number.isFinite(n)) return String(n);
      const s = n.toFixed(digits);
      return n >= 0 ? '+' + s : '−' + s.slice(1);
    }
    function plain(n, digits) {
      if (!Number.isFinite(n)) return String(n);
      if (n >= 0) return n.toFixed(digits);
      return '−' + (-n).toFixed(digits);
    }

    function tooltipHtml(k) {
      const lines = [
        'Neuron ' + k.idx,
        'w₁ = ' + signed(k.w1, 3) + '   b = ' + signed(k.bias, 3),
        'v = ' + signed(k.v, 3) + '   kink at x = ' + plain(k.x, 3),
      ];
      if (!k.inView) lines.push('(kink outside visible x range)');
      return lines.map(function (s) { return escapeHtml(s); }).join('<br>');
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    // -------- radius scaling ----------------------------------------------

    // Sqrt of |v| compressed against the population max so a single huge
    // outlier doesn't shrink everyone else to the floor.
    function radiusFor(absV, maxAbsV) {
      if (!(maxAbsV > 0)) return (R_MIN + R_MAX) / 2;
      const t = Math.sqrt(Math.min(1, absV / maxAbsV));
      return R_MIN + t * (R_MAX - R_MIN);
    }

    // -------- DOM groups (created once) -----------------------------------

    const gDots = el('g', { class: 'strip-dots' });
    stripSvg.appendChild(gDots);

    const emptyCaption = el('text', {
      class: 'strip-empty-caption',
      x: VB_W / 2, y: CY + 5,
      'text-anchor': 'middle',
    });
    emptyCaption.textContent = 'no hidden units yet';
    stripSvg.appendChild(emptyCaption);

    // -------- tooltip helpers ---------------------------------------------

    function hideTooltip() {
      tooltip.style.display = 'none';
      tooltip.innerHTML = '';
    }

    function showTooltip(dotEl, kink) {
      tooltip.innerHTML = tooltipHtml(kink);
      tooltip.style.display = 'block';
      // Position uses CSS pixels relative to the .neuron-strip parent.
      const stripRect = stripSvg.getBoundingClientRect();
      const dotRect = dotEl.getBoundingClientRect();
      // The .neuron-strip section is position: relative; its left edge
      // matches the SVG's left edge plus its own padding. Use the SVG box
      // as the reference and translate relative to the section.
      const section = stripSvg.parentElement;
      const sectionRect = section.getBoundingClientRect();
      const dotCenterX = (dotRect.left + dotRect.right) / 2 - sectionRect.left;
      const dotTopY = dotRect.top - sectionRect.top;

      // Measure tooltip after content is set.
      const tw = tooltip.offsetWidth;
      const th = tooltip.offsetHeight;
      let left = dotCenterX - tw / 2;
      const minLeft = stripRect.left - sectionRect.left + 4;
      const maxLeft = stripRect.right - sectionRect.left - tw - 4;
      if (left < minLeft) left = minLeft;
      if (left > maxLeft) left = Math.max(minLeft, maxLeft);

      let top = dotTopY - th - 8;
      // If the tooltip would be cut off at the top, drop it below the dot.
      if (top < 0) top = dotTopY + dotRect.height + 8;

      tooltip.style.left = left + 'px';
      tooltip.style.top = top + 'px';
    }

    // -------- render -------------------------------------------------------

    let kinksSorted = [];   // sorted by x, used for keyboard cycling

    function render(state) {
      // Clear dots.
      while (gDots.firstChild) gDots.removeChild(gDots.firstChild);

      const kinks = state.kinks || [];
      if (kinks.length === 0) {
        emptyCaption.style.display = '';
        kinksSorted = [];
        hideTooltip();
        return;
      }
      emptyCaption.style.display = 'none';

      // Population max |v| for radius scaling, only over in-view neurons
      // when any are in view, otherwise over all so out-of-view dots still
      // get reasonable sizes.
      let maxAbsV = 0;
      let anyInView = false;
      for (const k of kinks) {
        if (k.inView) anyInView = true;
      }
      for (const k of kinks) {
        if (anyInView && !k.inView) continue;
        const a = Math.abs(k.v);
        if (a > maxAbsV) maxAbsV = a;
      }

      // Sorted view for keyboard nav (deterministic left-to-right).
      kinksSorted = kinks.slice().sort(function (a, b) {
        // Within view first by x; pinned-left before all in-view; pinned-right after.
        const ax = a.inView ? a.x : (a.firesRight ? Infinity : -Infinity);
        const bx = b.inView ? b.x : (b.firesRight ? Infinity : -Infinity);
        if (ax === bx) return a.idx - b.idx;
        return ax - bx;
      });

      const focused = state.effectiveFocus
        ? state.effectiveFocus()
        : state.focusedNeuron;
      const xMode = state.hover && state.hover.kind === 'x';
      const firingSet = xMode
        ? new Set(state.firingNeuronsAt(state.hover.x))
        : null;

      for (const k of kinks) {
        let cx;
        let outOfView = !k.inView;
        if (k.inView) {
          cx = state.sx(k.x);
        } else {
          // Pin to left or right edge based on which side the kink lies on.
          // For w1 > 0 (fires right), kink x = -b/w1; if x > xMax it's far
          // right, if x < xMin it's far left.
          if (k.x > state.data.xMax) cx = VB_W - EDGE_PAD;
          else cx = EDGE_PAD;
        }

        const r = radiusFor(Math.abs(k.v), maxAbsV);
        const sideClass = k.firesRight ? 'fires-right' : 'fires-left';
        const cls = ['neuron-dot', sideClass];
        if (outOfView) cls.push('out-of-view');
        if (focused === k.idx) cls.push('focused');
        else if (firingSet && firingSet.has(k.idx)) cls.push('firing');

        const dot = el('circle', {
          class: cls.join(' '),
          cx: cx,
          cy: CY,
          r: outOfView ? Math.max(R_MIN - 0.5, 3) : r,
        });
        dot.dataset.idx = String(k.idx);

        // Hover: show tooltip AND broadcast a transient single-neuron hover
        // through the bus so other components (architecture, ghost) light up.
        dot.addEventListener('mouseenter', function () {
          showTooltip(dot, k);
          if (bus.setHover) bus.setHover({ kind: 'neuron', idx: k.idx });
        });
        dot.addEventListener('mouseleave', function () {
          hideTooltip();
          if (bus.setHover && bus.hover && bus.hover.kind === 'neuron' && bus.hover.idx === k.idx) {
            bus.setHover(null);
          }
        });

        // Click toggles persistent focus.
        dot.addEventListener('click', function (ev) {
          ev.stopPropagation();
          bus.setFocus(bus.focusedNeuron === k.idx ? null : k.idx);
        });

        gDots.appendChild(dot);
      }

      // If the currently focused neuron disappeared (e.g. width shrank),
      // any tooltip showing for it is stale.
      if (focused == null) hideTooltip();
    }

    // -------- background click clears focus --------------------------------

    stripSvg.addEventListener('click', function (e) {
      if (e.target === stripSvg) {
        if (bus.focusedNeuron != null) bus.setFocus(null);
        hideTooltip();
      }
    });

    // -------- keyboard navigation ------------------------------------------

    function cycleFocus(delta) {
      if (kinksSorted.length === 0) return;
      const order = kinksSorted.map(function (k) { return k.idx; });
      const cur = bus.focusedNeuron;
      let nextPos;
      if (cur == null) {
        nextPos = delta > 0 ? 0 : order.length - 1;
      } else {
        const curPos = order.indexOf(cur);
        if (curPos < 0) nextPos = delta > 0 ? 0 : order.length - 1;
        else nextPos = (curPos + delta + order.length) % order.length;
      }
      bus.setFocus(order[nextPos]);
    }

    stripSvg.addEventListener('keydown', function (e) {
      // Only act when the strip itself is the active element. This guards
      // against arrow keys leaking from the slider or document body.
      if (document.activeElement !== stripSvg) return;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        cycleFocus(+1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        cycleFocus(-1);
      } else if (e.key === 'Escape') {
        if (bus.focusedNeuron != null) {
          e.preventDefault();
          bus.setFocus(null);
        }
        hideTooltip();
      }
    });

    // Hide tooltip when strip loses focus entirely.
    stripSvg.addEventListener('blur', hideTooltip);

    // -------- subscribe ----------------------------------------------------

    bus.onChange(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
