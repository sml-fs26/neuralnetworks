/* Theme toggle. Persists to localStorage; 't' shortcut toggles mid-lecture. */
(function () {
  const STORAGE_KEY = 'gradient-descent-viz.theme';
  const root = document.documentElement;

  function apply(theme) {
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem(STORAGE_KEY, theme); } catch (_e) {}
  }

  function current() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function readHashTheme() {
    const m = (window.location.hash || '').match(/[#&?]theme=(light|dark)/);
    return m ? m[1] : null;
  }

  function init() {
    let theme = 'light';
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved === 'light' || saved === 'dark') theme = saved;
    } catch (_e) {}
    const hashed = readHashTheme();
    if (hashed) theme = hashed;
    apply(theme);

    const btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', () => apply(current() === 'light' ? 'dark' : 'light'));

    window.addEventListener('keydown', (e) => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName || '')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 't' || e.key === 'T') apply(current() === 'light' ? 'dark' : 'light');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.Theme = { apply, current };
})();
