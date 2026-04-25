/* Phase 1 — Observatory.
 *
 * Consumes /runs (list of recorded trained-agent traces) and /watch/{run_id}
 * (single trace) to render an interactive replay with reward decomposition,
 * per-component sparklines, service map, action timeline, and aggregate stats.
 *
 * Talks only to existing read-only endpoints. No backend changes needed.
 */

(function () {
  'use strict';

  const COMPONENT_COLORS = {
    diagnostic: '#06b6d4',  // cyan
    correct_op: '#22c55e',  // green
    resolution: '#f59e0b',  // orange
    format:     '#a855f7',  // purple
    efficiency: '#3b82f6',  // blue
    penalty:    '#ef4444',  // red
  };
  const COMPONENT_ORDER = ['diagnostic', 'correct_op', 'resolution', 'format', 'efficiency', 'penalty'];

  const $ = (id) => document.getElementById(id);

  const state = {
    runs: [],
    filteredRuns: [],
    activeFilter: 'all',
    events: [],
    cursor: 0,
    timer: null,
    currentRunId: null,
    summary: null,
  };

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }

  // ---- API ------------------------------------------------------------------

  async function fetchJSON(path) {
    try {
      const r = await fetch(path);
      if (!r.ok) return null;
      return await r.json();
    } catch (e) { return null; }
  }

  // ---- Renderers ------------------------------------------------------------

  function renderRunsPicker() {
    const sel = $('obs-run-picker');
    sel.innerHTML = '<option value="">— select a recorded run —</option>';
    state.filteredRuns.forEach((r) => {
      const o = document.createElement('option');
      o.value = r.run_id;
      const tag = r.resolved ? '✓' : '✗';
      const score = r.score != null ? r.score.toFixed(2) : '—';
      o.textContent = `${tag} ${r.run_id.slice(0, 22)}… [${r.task_id || '?'}] score=${score} model=${r.model || '?'}`;
      sel.appendChild(o);
    });
  }

  function renderTopStats() {
    const total = state.runs.length;
    const resolved = state.runs.filter(r => r.resolved).length;
    const scores = state.runs.map(r => r.score).filter(s => s != null);
    const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2) : '—';
    const conditions = new Set(state.runs.map(r => r.model).filter(Boolean));
    $('obs-stat-runs').textContent = total;
    $('obs-stat-resolved').textContent = total ? `${resolved}/${total}` : '—';
    $('obs-stat-avg-score').textContent = avg;
    $('obs-stat-conditions').textContent = conditions.size || '—';
  }

  function applyFilter() {
    const f = state.activeFilter;
    state.filteredRuns = state.runs.filter(r => {
      if (f === 'all') return true;
      if (f === 'resolved') return r.resolved === true;
      return r.task_id === f;
    });
    renderRunsPicker();
  }

  function renderSummary(summary) {
    $('obs-ep-tag').textContent = summary.task_id || '—';
    $('obs-run-id').textContent = summary.run_id || '—';
    $('obs-task').textContent = summary.task_id || '—';
    $('obs-seed').textContent = summary.seed != null ? String(summary.seed) : '—';
    $('obs-model').textContent = summary.model || '—';
    $('obs-alert').textContent = summary.alert || '—';
    let verdict = '<span class="obs-pill bad">UNRESOLVED</span>';
    if (summary.resolved === true) verdict = '<span class="obs-pill good">RESOLVED</span>';
    else if (summary.resolved == null) verdict = '<span class="obs-pill warn">PENDING</span>';
    $('obs-verdict').innerHTML = verdict;
    $('obs-score').textContent = summary.score != null ? summary.score.toFixed(3) : '—';
    $('obs-steps').textContent = summary.steps_used != null ? String(summary.steps_used) : '—';
  }

  function renderRewardDecomp(totals) {
    const stack = $('obs-reward-stack');
    const tbl = $('obs-reward-table');
    stack.innerHTML = ''; tbl.innerHTML = '';
    if (!totals) {
      tbl.innerHTML = '<div></div><div class="reward-name">—</div><div class="reward-val">no breakdown</div>';
      return;
    }
    let totalAbs = 0;
    COMPONENT_ORDER.forEach(k => { totalAbs += Math.abs(totals[k] || 0); });
    if (totalAbs <= 0) totalAbs = 1;
    COMPONENT_ORDER.forEach(k => {
      const v = totals[k] || 0;
      const seg = document.createElement('span');
      seg.className = 'seg';
      seg.style.width = (Math.abs(v) / totalAbs * 100).toFixed(1) + '%';
      seg.style.background = COMPONENT_COLORS[k];
      seg.title = `${k}: ${v.toFixed(3)}`;
      stack.appendChild(seg);

      const sw = document.createElement('span');
      sw.className = 'reward-sw';
      sw.style.background = COMPONENT_COLORS[k];
      const nm = document.createElement('span');
      nm.className = 'reward-name';
      nm.textContent = k.replace('_', ' ');
      const va = document.createElement('span');
      va.className = 'reward-val';
      va.textContent = (v >= 0 ? '+' : '') + v.toFixed(3);
      va.style.color = v < 0 ? 'var(--accent-red)' : 'var(--text-primary)';
      tbl.appendChild(sw); tbl.appendChild(nm); tbl.appendChild(va);
    });
  }

  function renderSparklines(events) {
    const grid = $('obs-sparklines');
    grid.innerHTML = '';
    const stepEvents = events.filter(e => e.type === 'step');
    if (stepEvents.length === 0) {
      grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No step events to plot.</div>';
      return;
    }
    COMPONENT_ORDER.forEach(k => {
      const series = stepEvents.map(e => Number((e.reward_breakdown || {})[k] || 0));
      const total = series.reduce((a, b) => a + b, 0);
      const row = document.createElement('div');
      row.className = 'obs-spark-row';
      row.innerHTML = `
        <span class="label">${k.replace('_', ' ')}</span>
        <div class="spark">${sparkSvg(series, COMPONENT_COLORS[k])}</div>
        <span class="total">${(total >= 0 ? '+' : '') + total.toFixed(2)}</span>
      `;
      grid.appendChild(row);
    });
  }

  function sparkSvg(series, color) {
    if (!series.length) return '';
    const w = 200, h = 32, pad = 2;
    const maxAbs = Math.max(0.001, ...series.map(v => Math.abs(v)));
    const xStep = (w - pad * 2) / Math.max(1, series.length - 1);
    const baseY = h / 2;
    let path = '';
    series.forEach((v, i) => {
      const x = pad + i * xStep;
      const y = baseY - (v / maxAbs) * (h / 2 - pad);
      path += (i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`);
    });
    return `
      <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
        <line x1="0" y1="${baseY}" x2="${w}" y2="${baseY}" stroke="rgba(99,115,148,.25)" stroke-width="0.5"/>
        <path d="${path}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    `;
  }

  function renderMap(services, label) {
    const root = $('obs-map');
    $('obs-map-tag').textContent = label || 'final state';
    root.innerHTML = '';
    if (!services || Object.keys(services).length === 0) {
      root.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="ic">🗺️</div>No services in trace</div>';
      return;
    }
    Object.values(services).forEach(s => {
      const card = document.createElement('div');
      const h = (s.health || 'unknown').toLowerCase();
      card.className = 'obs-svc ' + h;
      const cpu = s.cpu_percent != null ? Math.round(s.cpu_percent) + '%' : '—';
      const mem = s.memory_mb != null ? Math.round(s.memory_mb) + 'MB' : '—';
      card.innerHTML = `
        <div class="nm">${escapeHtml(s.name)}</div>
        <div class="row"><span>${escapeHtml(s.health || '—')}</span><span>v${escapeHtml(s.version || '?')}</span></div>
        <div class="row"><span>cpu ${cpu}</span><span>${mem}</span></div>
      `;
      root.appendChild(card);
    });
  }

  function appendStep(ev) {
    const tl = $('obs-timeline');
    if (tl.firstChild && tl.firstChild.classList && tl.firstChild.classList.contains('empty-state')) {
      tl.innerHTML = '';
    }
    const action = ev.action || {};
    const obs = ev.observation || {};
    const bd = ev.reward_breakdown || {};
    let total = 0;
    Object.values(bd).forEach(v => { total += Number(v) || 0; });
    let cls = 'step';
    if (total < 0) cls += ' bad';
    else if (total > 0.1) cls += ' good';
    const node = document.createElement('div');
    node.className = cls;
    const target = action.target_service ? ' → ' + action.target_service : '';
    const params = action.parameters && Object.keys(action.parameters).length > 0
      ? ' ' + JSON.stringify(action.parameters) : '';
    const message = (obs.message || ev.message || '').slice(0, 800);
    node.innerHTML = `
      <div class="top">
        <span><strong>step ${ev.step}</strong> · ${escapeHtml(action.action_type || '?')}${escapeHtml(target)}${escapeHtml(params)}</span>
        <span style="color:${total >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${total >= 0 ? '+' : ''}${total.toFixed(3)}</span>
      </div>
      <div class="body">${escapeHtml(message)}</div>
      <div class="reward-chips">${rewardChips(bd)}</div>
    `;
    node.addEventListener('click', () => node.classList.toggle('expanded'));
    tl.appendChild(node);
    tl.scrollTop = tl.scrollHeight;

    // Update map progressively if observation has services_summary
    if (obs.services_summary && Array.isArray(obs.services_summary)) {
      const services = {};
      obs.services_summary.forEach(s => { services[s.name] = s; });
      renderMap(services, `step ${ev.step}`);
    }
  }

  function rewardChips(bd) {
    if (!bd) return '';
    return COMPONENT_ORDER.map(k => {
      const v = bd[k];
      if (v == null || v === 0) return '';
      const sign = v >= 0 ? '+' : '';
      return `<span class="chip" style="color:${COMPONENT_COLORS[k]}">${k.replace('_', ' ')}:${sign}${Number(v).toFixed(2)}</span>`;
    }).filter(Boolean).join(' ');
  }

  // ---- Aggregate ------------------------------------------------------------

  function renderAggregate() {
    const root = $('obs-agg-grid');
    if (state.runs.length === 0) {
      root.innerHTML = '<div class="empty-state" style="grid-column:1/-1">No runs recorded yet. Run <code>training/eval_runner.py</code> with <code>runs_root=\'runs\'</code>.</div>';
      return;
    }
    const families = ['oom_crash', 'db_pool_exhaustion', 'bad_deployment_cascade'];
    const conditions = [...new Set(state.runs.map(r => r.model).filter(Boolean))].sort();
    if (conditions.length === 0) conditions.push('unknown');

    root.innerHTML = '';
    families.forEach(family => {
      const familyRuns = state.runs.filter(r => r.task_id === family);
      if (familyRuns.length === 0) return;
      const card = document.createElement('div');
      card.style.cssText = 'background:var(--bg-elevated);border:1px solid var(--border);border-radius:10px;padding:14px';
      card.innerHTML = `<div style="font-size:13px;font-weight:600;margin-bottom:10px">${escapeHtml(family)}</div>`;

      conditions.forEach(c => {
        const condRuns = familyRuns.filter(r => (r.model || 'unknown') === c);
        if (condRuns.length === 0) return;
        const successes = condRuns.filter(r => r.resolved).length;
        const successRate = successes / condRuns.length;
        const row = document.createElement('div');
        row.className = 'agg-bar-row';
        row.innerHTML = `
          <div class="nm">${escapeHtml(c)}</div>
          <div class="agg-bar"><div class="fill" style="width:${(successRate * 100).toFixed(0)}%"></div></div>
          <div class="pct">${(successRate * 100).toFixed(0)}%</div>
        `;
        card.appendChild(row);
      });
      root.appendChild(card);
    });
  }

  // ---- Replay control -------------------------------------------------------

  function clearTimeline() {
    $('obs-timeline').innerHTML = '<div class="empty-state"><div class="ic">▶</div>Replaying…</div>';
  }

  function loadRun(events, summary) {
    state.events = events || [];
    state.summary = summary || {};
    state.cursor = 0;
    pause();
    clearTimeline();
    $('obs-timeline').innerHTML = '';
    renderSummary(Object.assign({ run_id: state.currentRunId }, state.summary));
    const endEv = state.events.find(e => e.type === 'end');
    renderRewardDecomp(endEv ? endEv.breakdown_totals : null);
    renderSparklines(state.events);
    // Initial map: from earliest step that has services_summary, fall back to empty
    const firstWithMap = state.events.find(e => e.type === 'step' && e.observation && e.observation.services_summary);
    if (firstWithMap) {
      const services = {};
      (firstWithMap.observation.services_summary || []).forEach(s => { services[s.name] = s; });
      renderMap(services, 'pre-action');
    } else {
      renderMap({});
    }
  }

  function step() {
    if (!state.events || state.cursor >= state.events.length) {
      pause();
      return;
    }
    const ev = state.events[state.cursor++];
    if (ev.type === 'step') appendStep(ev);
    if (ev.type === 'end') {
      const node = document.createElement('div');
      node.className = 'step ' + (ev.resolved ? 'good' : 'bad');
      node.innerHTML = `
        <div class="top">
          <span><strong>END</strong></span>
          <span>${ev.resolved ? 'RESOLVED' : 'UNRESOLVED'}</span>
        </div>
        <div class="body">score=${(ev.score || 0).toFixed(3)} · steps=${ev.steps_used || '?'}</div>
      `;
      $('obs-timeline').appendChild(node);
      pause();
    }
  }

  function play() {
    pause();
    state.timer = setInterval(step, 650);
  }

  function pause() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  // ---- Init -----------------------------------------------------------------

  async function init() {
    if (window.Observatory && window.Observatory._inited) return;
    const data = await fetchJSON('/runs');
    state.runs = (data && data.runs) || [];
    applyFilter();
    renderTopStats();
    renderAggregate();

    // Wire toolbar
    $('obs-run-picker').addEventListener('change', e => selectRun(e.target.value));
    $('obs-replay').addEventListener('click', () => {
      if (state.currentRunId) selectRun(state.currentRunId).then(play);
    });
    $('obs-step').addEventListener('click', () => { pause(); step(); });
    $('obs-pause').addEventListener('click', pause);

    // Filter chips
    document.querySelectorAll('#obs-filter-chips .obs-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#obs-filter-chips .obs-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        state.activeFilter = chip.dataset.filter;
        applyFilter();
      });
    });

    // Honor ?run=<id> query param for direct linking from videos
    const params = new URLSearchParams(location.search);
    const requested = params.get('run');
    if (requested && state.runs.some(r => r.run_id === requested)) {
      $('obs-run-picker').value = requested;
      await selectRun(requested);
    }

    if (window.Observatory) window.Observatory._inited = true;
  }

  async function selectRun(runId) {
    if (!runId) return;
    state.currentRunId = runId;
    const data = await fetchJSON('/watch/' + encodeURIComponent(runId));
    if (!data) return;
    loadRun(data.events || [], data.summary || {});
    play();
  }

  window.Observatory = { init, _inited: false };
  // Auto-init if landing on observatory tab (default)
  document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('tab-observatory')?.classList.contains('active')) {
      init();
    }
  });
})();
