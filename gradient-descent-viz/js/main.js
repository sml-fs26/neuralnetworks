/* Click-step scene engine for the gradient-descent viz.

   Two scenes:
     0: "The bowl" -- 2D loss surface for linear regression, split-pane.
     1: "Watch your network train" -- live in-browser training of the
        Demo 2 MLP with synchronized plot/strip/architecture/loss curve.

   Pattern lifted from the course-viz skill (kmeans-deepdive). Each scene
   registers itself on window.scenes; the engine handles transitions,
   keyboard navigation, hash routing, and the dot pager. */
(function () {
  const SCENE_TITLES = [
    'Scene 1 · The bowl',
    'Scene 2 · Watch your network train',
  ];

  const sceneNodes = [];
  const sceneState = [];
  let current = -1;

  function readHashScene() {
    const m = (window.location.hash || '').match(/[#&?]scene=(\d+)/);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    return Number.isFinite(n) && n >= 0 && n < SCENE_TITLES.length ? n : null;
  }

  function syncHash(idx) {
    const cur = window.location.hash || '';
    const m = cur.match(/[#&?]scene=(\d+)/);
    let next;
    if (m) {
      next = cur.replace(/([#&?])scene=\d+/, `$1scene=${idx}`);
    } else if (cur.length > 1) {
      next = cur + `&scene=${idx}`;
    } else {
      next = `#scene=${idx}`;
    }
    if (next !== cur) history.replaceState(null, '', next);
  }

  function updateDots(idx) {
    const dots = document.querySelectorAll('.dot-pager .dot');
    dots.forEach((dot, i) => dot.classList.toggle('active', i === idx));
  }

  function updateButtons(idx) {
    const prev = document.getElementById('prev-btn');
    const next = document.getElementById('next-btn');
    if (prev) prev.disabled = idx <= 0;
    if (next) next.disabled = idx >= SCENE_TITLES.length - 1;
  }

  function updateTitle(idx) {
    const el = document.getElementById('scene-title');
    if (el) el.textContent = SCENE_TITLES[idx];
  }

  function goTo(idx) {
    if (idx < 0 || idx >= SCENE_TITLES.length) return;
    if (idx === current) {
      // Re-render request -- still useful for hash updates.
      syncHash(idx);
      return;
    }

    if (current >= 0) {
      const old = sceneNodes[current];
      if (old) old.classList.remove('active');
      const oldState = sceneState[current];
      if (oldState && typeof oldState.onLeave === 'function') {
        try { oldState.onLeave(); } catch (e) { console.error(e); }
      }
    }

    const stage = document.getElementById('stage');
    if (!sceneNodes[idx]) {
      const node = document.createElement('div');
      node.className = 'scene';
      node.dataset.scene = String(idx);
      stage.appendChild(node);
      sceneNodes[idx] = node;
      // Builders register as window.scenes.scene1, scene2, ... (1-indexed
       // to match user-facing labels); internal array indexing is 0-indexed.
      const builder = window.scenes && window.scenes['scene' + (idx + 1)];
      if (builder) {
        try {
          sceneState[idx] = builder(node) || {};
        } catch (e) {
          console.error('Scene builder failed:', e);
          sceneState[idx] = {};
        }
      } else {
        node.innerHTML = `<p style="opacity:0.5">Scene ${idx + 1} not yet implemented.</p>`;
        sceneState[idx] = {};
      }
    } else {
      const st = sceneState[idx];
      if (st && typeof st.onEnter === 'function') {
        try { st.onEnter(); } catch (e) { console.error(e); }
      }
    }

    current = idx;
    setTimeout(() => sceneNodes[idx].classList.add('active'), 20);
    updateDots(idx);
    updateButtons(idx);
    updateTitle(idx);
    syncHash(idx);
  }

  function init() {
    if (!window.GDV_DATA) {
      console.error('GDV_DATA missing -- did data/datasets.js load?');
      return;
    }

    // Build the dot pager.
    const pager = document.getElementById('dot-pager');
    if (pager) {
      for (let i = 0; i < SCENE_TITLES.length; i++) {
        const dot = document.createElement('button');
        dot.className = 'dot';
        dot.type = 'button';
        dot.setAttribute('aria-label', `Go to ${SCENE_TITLES[i]}`);
        dot.addEventListener('click', () => goTo(i));
        pager.appendChild(dot);
      }
    }

    // Prev/next buttons.
    const prev = document.getElementById('prev-btn');
    const next = document.getElementById('next-btn');
    if (prev) prev.addEventListener('click', () => goTo(current - 1));
    if (next) next.addEventListener('click', () => goTo(current + 1));

    // Keyboard nav.
    window.addEventListener('keydown', (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName || '')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const st = sceneState[current];
      if (e.key === 'ArrowRight') {
        const handled = st && typeof st.onNextKey === 'function' && st.onNextKey();
        if (!handled) goTo(current + 1);
      } else if (e.key === 'ArrowLeft') {
        const handled = st && typeof st.onPrevKey === 'function' && st.onPrevKey();
        if (!handled) goTo(current - 1);
      }
    });

    window.addEventListener('hashchange', () => {
      const n = readHashScene();
      if (n != null) goTo(n);
    });

    // Initial scene: hash override or scene 0.
    const initialScene = readHashScene();
    goTo(initialScene != null ? initialScene : 0);
  }

  if (!window.scenes) window.scenes = {};

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.GDV = { goTo, getCurrentScene: () => current };
})();
