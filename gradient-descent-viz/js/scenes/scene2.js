/* Scene 2 — Watch your network train.

   Synthesis demo of the lecture: builds a TrainingEngine for the same MLP
   used in Demo 2, mounts the four sub-panels (plot, loss curve, strip +
   architecture, controls), and drives a 30 Hz animation loop on the Train
   button. The static fit from Demo 2 is what this scene produces *live*.

   Sub-panels are mounted by independent modules:
     - GDV.buildPlot(plotRoot, bus)
     - GDV.buildLossCurve(lossRoot, bus)
     - GDV.buildStripArch(saRoot, bus)

   The state bus is the only contract between this orchestrator and the
   sub-panels. Sub-panels read engine state via bus.engine.params() and
   subscribe via bus.onUpdate(fn). They never mutate the engine. */
(function () {
  'use strict';

  const TICK_MS = 33;             // ~30 Hz training tick
  const DEFAULT_WIDTH = 12;
  const DEFAULT_LR = 0.05;
  const DEFAULT_SEED = 42;
  const DEFAULT_OPT = 'adam';

  // Read &epoch=N from the hash for headless verification (deterministic
  // pre-step before first paint).
  function readHashEpoch() {
    const m = (window.location.hash || '').match(/[#&?]epoch=(\d+)/);
    if (!m) return 0;
    const n = parseInt(m[1], 10);
    return Number.isFinite(n) && n > 0 ? n : 0;
  }

  function readHashSeed() {
    const m = (window.location.hash || '').match(/[#&?]seed=(\d+)/);
    if (!m) return DEFAULT_SEED;
    const n = parseInt(m[1], 10);
    return Number.isFinite(n) ? n : DEFAULT_SEED;
  }

  function captionFor(state) {
    const p = state.engine.params();
    if (state.isTraining) {
      return `Training… iter ${p.iter} · loss ${p.loss.toFixed(4)}`;
    }
    if (p.iter === 0) {
      return 'Press Train. Random ReLU kinks slide into place; the bold curve learns the wave.';
    }
    return `Paused at iter ${p.iter} · loss ${p.loss.toFixed(4)} — press Train to continue.`;
  }

  function buildScene(root) {
    if (!window.GDV_DATA) {
      root.innerHTML = '<p style="opacity:0.5">GDV_DATA missing.</p>';
      return {};
    }
    if (!window.TrainingEngine) {
      root.innerHTML = '<p style="opacity:0.5">TrainingEngine missing.</p>';
      return {};
    }
    if (!window.GDVScene2 || !window.GDVScene2.buildPlot || !window.GDVScene2.buildStripArch || !window.GDVScene2.buildLossCurve) {
      root.innerHTML = '<p style="opacity:0.5">Sub-panel builders missing.</p>';
      return {};
    }

    // ---- DOM scaffolding ----------------------------------------------
    root.innerHTML = '';
    root.classList.add('s2-root');

    const hero = document.createElement('header');
    hero.className = 'hero';
    hero.innerHTML =
      '<h1>Watch your network train</h1>' +
      '<p class="subtitle">Adam + gradient descent slides the random ReLU kinks into place. ' +
      'The bold curve at the start? That’s the same network from Demo 2 — before training.</p>';
    root.appendChild(hero);

    // Two-column row: plot (wide) + loss curve (narrow).
    const topRow = document.createElement('div');
    topRow.className = 's2-top-row';
    root.appendChild(topRow);

    const plotCard = document.createElement('div');
    plotCard.className = 'card s2-plot-card';
    topRow.appendChild(plotCard);

    const lossCard = document.createElement('div');
    lossCard.className = 'card s2-loss-card';
    topRow.appendChild(lossCard);

    // Strip + architecture row.
    const saCard = document.createElement('div');
    saCard.className = 'card s2-sa-card';
    root.appendChild(saCard);

    // Controls.
    const controls = document.createElement('div');
    controls.className = 'controls s2-controls';
    root.appendChild(controls);

    const trainBtn = makeButton('Train', 'primary');
    const pauseBtn = makeButton('Pause');
    const resetBtn = makeButton('Reset');
    controls.appendChild(group([trainBtn, pauseBtn, resetBtn]));

    const widthGroup = document.createElement('div');
    widthGroup.className = 'control-group';
    const widthLabel = document.createElement('label');
    widthLabel.textContent = 'Width';
    const widthInput = document.createElement('input');
    widthInput.type = 'range';
    widthInput.min = '1'; widthInput.max = '24'; widthInput.step = '1';
    widthInput.value = String(DEFAULT_WIDTH);
    const widthOut = document.createElement('output');
    widthOut.textContent = String(DEFAULT_WIDTH);
    widthGroup.appendChild(widthLabel);
    widthGroup.appendChild(widthInput);
    widthGroup.appendChild(widthOut);
    controls.appendChild(widthGroup);

    const lrGroup = document.createElement('div');
    lrGroup.className = 'control-group';
    const lrLabel = document.createElement('label');
    lrLabel.textContent = 'LR';
    const lrInput = document.createElement('input');
    lrInput.type = 'range';
    lrInput.min = '0.005'; lrInput.max = '0.2'; lrInput.step = '0.005';
    lrInput.value = String(DEFAULT_LR);
    const lrOut = document.createElement('output');
    lrOut.textContent = (+DEFAULT_LR).toFixed(3);
    lrGroup.appendChild(lrLabel);
    lrGroup.appendChild(lrInput);
    lrGroup.appendChild(lrOut);
    controls.appendChild(lrGroup);

    const seedGroup = document.createElement('div');
    seedGroup.className = 'control-group';
    const seedLabel = document.createElement('label');
    seedLabel.textContent = 'Seed';
    const seedInput = document.createElement('input');
    seedInput.type = 'number';
    seedInput.min = '0'; seedInput.step = '1';
    seedInput.value = String(readHashSeed());
    seedInput.className = 's2-seed-input';
    seedGroup.appendChild(seedLabel);
    seedGroup.appendChild(seedInput);
    controls.appendChild(seedGroup);

    const optGroup = document.createElement('div');
    optGroup.className = 'control-group';
    const optLabel = document.createElement('label');
    optLabel.textContent = 'Optimizer';
    const optSelect = document.createElement('select');
    [['adam', 'Adam'], ['momentum', 'SGD + Momentum'], ['sgd', 'SGD']].forEach(([v, l]) => {
      const o = document.createElement('option');
      o.value = v; o.textContent = l;
      if (v === DEFAULT_OPT) o.selected = true;
      optSelect.appendChild(o);
    });
    optGroup.appendChild(optLabel);
    optGroup.appendChild(optSelect);
    controls.appendChild(optGroup);

    // Optimizer explanation -- updates when the dropdown changes.
    const optInfo = document.createElement('p');
    optInfo.className = 'optimizer-info s2-opt-info';
    root.appendChild(optInfo);
    if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, DEFAULT_OPT);

    const caption = document.createElement('p');
    caption.className = 'caption s2-caption';
    root.appendChild(caption);

    // ---- engine + bus --------------------------------------------------
    const engine = new window.TrainingEngine({
      width: DEFAULT_WIDTH,
      seed: readHashSeed(),
      lr: DEFAULT_LR,
      optimizer: DEFAULT_OPT,
    });

    const bus = {
      engine,
      isTraining: false,
      _callbacks: [],
      onUpdate(fn) { if (typeof fn === 'function') this._callbacks.push(fn); },
      emit() {
        for (const fn of this._callbacks) {
          try { fn(); } catch (e) { console.error(e); }
        }
      },
    };

    // ---- mount sub-panels ----------------------------------------------
    const plotHandle = window.GDVScene2.buildPlot(plotCard, bus);
    const lossHandle = window.GDVScene2.buildLossCurve(lossCard, bus);
    const saHandle = window.GDVScene2.buildStripArch(saCard, bus);

    // Headless pre-stepping for &epoch=N before the first user-visible paint.
    const epochN = readHashEpoch();
    if (epochN > 0) {
      for (let i = 0; i < epochN; i++) bus.engine.step();
      // Reset the loss-curve so the pre-stepped iters land in the buffer
      // as a single starting sample.
      if (lossHandle && typeof lossHandle.reset === 'function') lossHandle.reset();
      if (plotHandle && typeof plotHandle.renderNow === 'function') plotHandle.renderNow();
      if (saHandle && typeof saHandle.renderNow === 'function') saHandle.renderNow();
    }

    caption.textContent = captionFor(bus);

    // ---- training interval --------------------------------------------
    let intervalId = null;

    function startTraining() {
      if (intervalId != null) return;
      bus.isTraining = true;
      caption.textContent = captionFor(bus);
      intervalId = setInterval(() => {
        try {
          bus.engine.step();
          bus.emit();
          // Refresh caption every tick (cheap).
          caption.textContent = captionFor(bus);
        } catch (e) {
          console.error('training step failed:', e);
          stopTraining();
        }
      }, TICK_MS);
    }

    function stopTraining() {
      if (intervalId != null) {
        clearInterval(intervalId);
        intervalId = null;
      }
      bus.isTraining = false;
      caption.textContent = captionFor(bus);
    }

    function fullReset() {
      stopTraining();
      const seed = parseInt(seedInput.value, 10);
      bus.engine.reset(Number.isFinite(seed) ? seed : DEFAULT_SEED);
      // Rebuild the loss-curve buffer with the fresh init.
      if (lossHandle && typeof lossHandle.reset === 'function') lossHandle.reset();
      bus.emit();
      caption.textContent = captionFor(bus);
    }

    // Width changes require a fresh engine (parameter shape changes).
    function changeWidth(newW) {
      const w = Math.max(1, Math.min(24, newW | 0));
      if (w === bus.engine.getWidth()) return;
      stopTraining();
      const seed = parseInt(seedInput.value, 10);
      const lr = parseFloat(lrInput.value);
      const opt = optSelect.value;
      bus.engine = new window.TrainingEngine({
        width: w,
        seed: Number.isFinite(seed) ? seed : DEFAULT_SEED,
        lr: Number.isFinite(lr) ? lr : DEFAULT_LR,
        optimizer: opt || DEFAULT_OPT,
      });
      if (lossHandle && typeof lossHandle.reset === 'function') lossHandle.reset();
      bus.emit();
      caption.textContent = captionFor(bus);
    }

    // ---- control wiring ------------------------------------------------
    trainBtn.addEventListener('click', startTraining);
    pauseBtn.addEventListener('click', stopTraining);
    resetBtn.addEventListener('click', fullReset);

    widthInput.addEventListener('input', () => {
      const v = parseInt(widthInput.value, 10);
      widthOut.textContent = String(v);
      changeWidth(v);
    });

    lrInput.addEventListener('input', () => {
      const v = parseFloat(lrInput.value);
      lrOut.textContent = v.toFixed(3);
      // Live update: do not reset.
      if (Number.isFinite(v)) bus.engine.setLR(v);
    });

    seedInput.addEventListener('change', () => {
      const v = parseInt(seedInput.value, 10);
      if (!Number.isFinite(v)) return;
      // Reset uses the new seed.
      fullReset();
    });

    optSelect.addEventListener('change', () => {
      bus.engine.setOptimizer(optSelect.value);
      if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, optSelect.value);
      // Switching optimizer mid-training is fine; momentum/Adam state is
      // kept in their separate buffers, so a switch is clean.
    });

    // ---- lifecycle -----------------------------------------------------
    return {
      onEnter() {
        // Re-render is enough; we don't auto-start training.
        bus.emit();
        caption.textContent = captionFor(bus);
      },
      onLeave() {
        // Hard rule: stop the training interval when the user navigates
        // away. Otherwise the engine keeps stepping in the background.
        stopTraining();
      },
    };
  }

  function makeButton(label, cls) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    if (cls) b.className = cls;
    return b;
  }

  function group(buttons) {
    const g = document.createElement('div');
    g.className = 'control-group';
    for (const b of buttons) g.appendChild(b);
    return g;
  }

  window.scenes = window.scenes || {};
  window.scenes.scene2 = function (root) { return buildScene(root); };
})();
