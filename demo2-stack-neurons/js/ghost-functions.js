/* Demo 2 -- Ghost shape functions layer.

   For each active hidden unit at the current width, render a faint polyline
   showing that neuron's individual contribution v_i * ReLU(w1_i * x + b_i)
   over the same x-grid the bold fit uses.

   Three highlight modes (priority, top wins):
     1. hover.kind === 'neuron' (transient, single)        -> isolate
     2. focusedNeuron != null    (sticky single, click)    -> isolate
     3. hover.kind === 'x'       (firing-pattern at this x) -> highlight set
     4. otherwise                                          -> default. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';

  function defaultOpacity(width) {
    // ~1/sqrt(N) clamped to [0.06, 0.18].
    if (width <= 0) return 0.18;
    const o = 1 / Math.sqrt(width);
    if (o < 0.06) return 0.06;
    if (o > 0.18) return 0.18;
    return o;
  }

  function init() {
    if (!window.demo2) {
      console.error('window.demo2 missing -- ghost-functions cannot mount');
      return;
    }
    const bus = window.demo2;
    const layer = document.querySelector('#plot .ghost-layer');
    const fitPath = document.querySelector('#plot .fit-curve');
    if (!layer || !fitPath) return;

    function clear() {
      while (layer.firstChild) layer.removeChild(layer.firstChild);
    }

    function buildPoints(ys) {
      const xs = bus.xGrid;
      const sx = bus.sx, sy = bus.sy;
      const out = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) {
        out[i] = sx(xs[i]).toFixed(1) + ',' + sy(ys[i]).toFixed(1);
      }
      return out.join(' ');
    }

    function render(state) {
      clear();

      const fit = state.fit;
      if (!fit || fit.kind !== 'mlp') {
        fitPath.classList.remove('dimmed');
        return;
      }

      const N = fit.w1.length;
      const baseOp = defaultOpacity(N);

      // Determine which mode wins.
      const isolated = state.effectiveFocus();          // single-neuron isolate
      const xMode = !isolated && state.hover && state.hover.kind === 'x';
      const firingSet = xMode ? new Set(state.firingNeuronsAt(state.hover.x)) : null;

      // Opacity policy.
      // - isolate: focused = 1.0, others = 0.04
      // - x-mode: firing  = 0.45, non-firing = max(baseOp * 0.4, 0.03)
      // - default: all at baseOp
      const ISO_OFF = 0.04;
      const FIRING_ON = 0.45;
      const NON_FIRING = Math.max(baseOp * 0.4, 0.03);

      for (let i = 0; i < N; i++) {
        const ys = bus.evalNeuron(i);
        if (!ys) continue;
        const poly = document.createElementNS(NS, 'polyline');
        poly.setAttribute('class', 'ghost-line');
        poly.setAttribute('points', buildPoints(ys));

        let op;
        if (isolated != null) {
          if (i === isolated) {
            poly.classList.add('focused');
            op = 1;
          } else {
            op = ISO_OFF;
          }
        } else if (xMode) {
          if (firingSet.has(i)) {
            poly.classList.add('firing');
            op = FIRING_ON;
          } else {
            op = NON_FIRING;
          }
        } else {
          op = baseOp;
        }
        poly.setAttribute('opacity', op.toFixed(3));
        layer.appendChild(poly);
      }

      if (isolated != null) fitPath.classList.add('dimmed');
      else fitPath.classList.remove('dimmed');
    }

    bus.onChange(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
