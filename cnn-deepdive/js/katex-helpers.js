/* Tiny KaTeX shim. All scenes render formulas via `Katex.render(tex, host, displayMode)`.
   Errors do not throw; they fall back to literal text rendering. */
(function () {
  function render(tex, host, displayMode) {
    if (!host) return;
    try {
      window.katex.render(tex, host, { throwOnError: false, displayMode: !!displayMode });
    } catch (e) {
      host.textContent = tex;
    }
  }

  /* Convenience: render an inline formula and return the host element. */
  function inline(tex) {
    const span = document.createElement('span');
    render(tex, span, false);
    return span;
  }

  /* Convenience: render a display formula in a centered block. */
  function display(tex) {
    const div = document.createElement('div');
    div.className = 'formula-block';
    render(tex, div, true);
    return div;
  }

  window.Katex = { render, inline, display };
})();
