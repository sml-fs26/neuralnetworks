/* Optimizer explanations -- self-contained, every term defined.

   Each algorithm's explanation has four parts:
     1. Name (short label + full meaning)
     2. Update rule(s) as display-mode KaTeX
     3. Variable glossary -- every symbol in the equations defined
     4. Prose paragraph: what it actually does, when to use it

   Renders into a target element via render(target, key). */
(function () {
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // Each variable: { tex, defn }. tex is rendered inline, defn is plain HTML.
  const INFO = {
    sgd: {
      name: 'SGD',
      fullName: 'Stochastic Gradient Descent',
      equations: [
        '\\theta_{t+1} \\;=\\; \\theta_t \\;-\\; \\eta \\cdot g_t \\;,\\qquad g_t \\;=\\; \\nabla L(\\theta_t)',
      ],
      vars: [
        { tex: '\\theta_t',    defn: 'parameter vector at step <em>t</em> (the model’s weights)' },
        { tex: '\\eta',         defn: 'learning rate (a positive scalar; the step size)' },
        { tex: 'g_t',           defn: 'gradient of the loss at the current parameters' },
        { tex: '\\nabla L',     defn: 'gradient operator applied to the loss function <em>L</em>' },
      ],
      desc:
        'At every step, compute the gradient of the loss with respect to the parameters — the direction of steepest <em>ascent</em>. ' +
        'Subtract a fraction of it from θ to descend. The “stochastic” part: in practice ' +
        'g<sub>t</sub> is computed on a random minibatch of training data, not the full dataset. ' +
        'Minibatch noise adds randomness that helps escape saddle points and lets us train on data ' +
        'that doesn’t fit in memory. SGD has no memory — each step looks only at the current slope.',
    },

    momentum: {
      name: 'SGD + Momentum',
      fullName: 'Heavy-ball method',
      equations: [
        'v_{t+1} \\;=\\; \\beta \\, v_t \\;+\\; g_t',
        '\\theta_{t+1} \\;=\\; \\theta_t \\;-\\; \\eta \\, v_{t+1}',
      ],
      vars: [
        { tex: '\\theta_t', defn: 'parameter vector at step <em>t</em>' },
        { tex: 'v_t',       defn: 'velocity vector at step <em>t</em> (initialised to <strong>0</strong>)' },
        { tex: '\\beta',    defn: 'momentum coefficient, <em>0 ≤ β &lt; 1</em> (typical: <strong>0.9</strong>)' },
        { tex: '\\eta',      defn: 'learning rate' },
        { tex: 'g_t',       defn: 'gradient of the loss at θ<sub>t</sub>' },
      ],
      desc:
        'Maintain a running velocity vector that retains a fraction β of the previous step’s ' +
        'velocity, then add the current gradient. The parameter update steps opposite this accumulated ' +
        'velocity, not the raw gradient. Setting β = 0 reduces to plain SGD; β = 0.9 means ' +
        '90&thinsp;% of the previous velocity persists — a heavy ball rolling downhill. ' +
        'On long, narrow valleys, the gradient direction along the valley axis is consistent so ' +
        'velocity accumulates; the perpendicular components alternate sign and partly cancel. ' +
        'Net effect: faster progress along the valley, less zig-zagging across it.',
    },

    adam: {
      name: 'Adam',
      fullName: 'Adaptive Moment Estimation',
      equations: [
        'm_{t+1} \\;=\\; \\beta_1 \\, m_t \\;+\\; (1-\\beta_1) \\, g_t',
        'v_{t+1} \\;=\\; \\beta_2 \\, v_t \\;+\\; (1-\\beta_2) \\, g_t \\odot g_t',
        '\\hat m_{t+1} \\;=\\; \\frac{m_{t+1}}{1-\\beta_1^{\\,t+1}} \\,,\\qquad \\hat v_{t+1} \\;=\\; \\frac{v_{t+1}}{1-\\beta_2^{\\,t+1}}',
        '\\theta_{t+1} \\;=\\; \\theta_t \\;-\\; \\eta \\,\\cdot\\, \\frac{\\hat m_{t+1}}{\\sqrt{\\hat v_{t+1}} \\;+\\; \\varepsilon}',
      ],
      vars: [
        { tex: '\\theta_t',  defn: 'parameter vector at step <em>t</em>' },
        { tex: 'g_t',         defn: 'gradient of the loss at θ<sub>t</sub>' },
        { tex: 'm_t',         defn: 'running mean of the gradient (“first moment”), initialised to <strong>0</strong>' },
        { tex: 'v_t',         defn: 'running mean of the squared gradient (“second moment”), initialised to <strong>0</strong>; tracks variance per coordinate' },
        { tex: 'g_t \\odot g_t', defn: 'element-wise square of the gradient' },
        { tex: '\\beta_1, \\beta_2', defn: 'decay rates for the moment estimates (typical: <strong>0.9</strong> and <strong>0.999</strong>)' },
        { tex: '\\hat m, \\hat v', defn: 'bias-corrected moments — since <em>m</em> and <em>v</em> start at 0 they are biased toward 0 in early steps; dividing by <em>1−β<sup>t+1</sup></em> undoes this' },
        { tex: '\\eta',        defn: 'learning rate (typical: <strong>10<sup>-3</sup></strong>)' },
        { tex: '\\varepsilon', defn: 'small constant (typical: <strong>10<sup>-8</sup></strong>) to prevent division by zero' },
      ],
      desc:
        'Adam combines momentum (the smoothed gradient <em>m̂</em>) with a per-parameter learning ' +
        'rate (division by <em>√v̂</em>). Every coordinate of θ ends up with its own ' +
        'effective step size: a parameter whose gradient has been wildly fluctuating (large <em>v</em>) ' +
        'gets a smaller step; a parameter whose gradient has been consistently small or steady gets a ' +
        'step closer to the full η. All operations are element-wise, so this scaling happens ' +
        'independently per coordinate. Adam is the default optimiser for almost every modern neural ' +
        'network: it works well across very different problem scales without per-parameter tuning.',
    },
  };

  function renderEqRow(target, tex) {
    const div = document.createElement('div');
    div.className = 'oi-eq';
    target.appendChild(div);
    if (window.katex) {
      try {
        window.katex.render(tex, div, { throwOnError: false, displayMode: true });
      } catch (e) {
        div.textContent = tex;
      }
    } else {
      div.textContent = tex;
    }
  }

  function renderInline(target, tex) {
    const span = document.createElement('span');
    span.className = 'oi-tex';
    if (window.katex) {
      try {
        window.katex.render(tex, span, { throwOnError: false, displayMode: false });
      } catch (e) {
        span.textContent = tex;
      }
    } else {
      span.textContent = tex;
    }
    target.appendChild(span);
  }

  function render(target, key) {
    const info = INFO[key] || INFO.sgd;
    target.innerHTML = '';
    target.classList.add('oi-block');

    // Header: name + full name
    const header = document.createElement('div');
    header.className = 'oi-header';
    header.innerHTML =
      '<strong class="oi-name">' + escapeHtml(info.name) + '</strong>' +
      ' <span class="oi-fullname">' + escapeHtml(info.fullName) + '</span>';
    target.appendChild(header);

    // Equations
    const eqs = document.createElement('div');
    eqs.className = 'oi-equations';
    info.equations.forEach(function (tex) { renderEqRow(eqs, tex); });
    target.appendChild(eqs);

    // Variable glossary
    if (info.vars && info.vars.length) {
      const where = document.createElement('div');
      where.className = 'oi-where';
      const lab = document.createElement('span');
      lab.className = 'oi-where-lab';
      lab.textContent = 'where';
      where.appendChild(lab);
      info.vars.forEach(function (v, i) {
        const item = document.createElement('span');
        item.className = 'oi-var';
        renderInline(item, v.tex);
        const dl = document.createElement('span');
        dl.className = 'oi-defn';
        // Allow safe HTML entities/tags in defn -- it's authored content here.
        dl.innerHTML = ' — ' + v.defn;
        item.appendChild(dl);
        where.appendChild(item);
        if (i < info.vars.length - 1) {
          const sep = document.createElement('span');
          sep.className = 'oi-sep';
          sep.textContent = '·';
          where.appendChild(sep);
        }
      });
      target.appendChild(where);
    }

    // Prose
    const desc = document.createElement('p');
    desc.className = 'oi-desc';
    desc.innerHTML = info.desc;
    target.appendChild(desc);
  }

  window.OptimizerInfo = { render, INFO };
})();
