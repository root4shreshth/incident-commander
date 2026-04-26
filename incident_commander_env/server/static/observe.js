/* IncidentCommander - Observe-mode replayer.
 *
 * Talks to two server endpoints:
 *   GET /runs            -> list of recorded trained-agent runs
 *   GET /watch/{run_id}  -> full event stream for one run (start/step/end)
 *
 * Replays events at a steady cadence so the human watching can follow what
 * the trained agent did. URL ?watch=<run_id> auto-loads a specific run.
 */

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const COMPONENT_COLORS = {
    diagnostic: '#69e0d6',
    correct_op: '#6cd86b',
    resolution: '#ffb86b',
    format: '#9aa0ff',
    efficiency: '#5cc7ff',
    penalty: '#ff6b6b',
  };

  const state = {
    runs: [],
    events: [],
    cursor: 0,
    timer: null,
    currentRunId: null,
    summary: null,
  };

  // ---- API ------------------------------------------------------------------

  async function fetchRuns() {
    try {
      const r = await fetch('/runs');
      if (!r.ok) return [];
      const data = await r.json();
      return data.runs || [];
    } catch (e) {
      return [];
    }
  }

  async function fetchRun(runId) {
    try {
      const r = await fetch('/watch/' + encodeURIComponent(runId));
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }

  // ---- Renderers ------------------------------------------------------------

  function renderRunsPicker() {
    const sel = $('run-picker');
    sel.innerHTML = '<option value="">- select a run -</option>';
    state.runs.forEach((r) => {
      const o = document.createElement('option');
      o.value = r.run_id;
      const tag = r.resolved ? '✓' : '✗';
      const score = r.score != null ? r.score.toFixed(2) : '-';
      o.textContent = `${tag} ${r.run_id}  [${r.task_id || '?'}]  score=${score}  model=${r.model || '?'}`;
      sel.appendChild(o);
    });
  }

  function renderSummary(summary) {
    $('m-run-id').textContent = summary.run_id || '-';
    $('m-task').textContent = summary.task_id || '-';
    $('m-seed').textContent = summary.seed != null ? String(summary.seed) : '-';
    $('m-model').textContent = summary.model || '-';
    $('m-alert').textContent = summary.alert || '-';
    const verdict = summary.resolved == null
      ? '-'
      : (summary.resolved
          ? '<span class="pill good">RESOLVED</span>'
          : '<span class="pill bad">UNRESOLVED</span>');
    $('m-verdict').innerHTML = verdict;
    $('m-score').textContent = summary.score != null ? summary.score.toFixed(3) : '-';
    $('m-steps').textContent = summary.steps_used != null ? String(summary.steps_used) : '-';
  }

  function renderRewardDecomp(totals) {
    const bar = $('reward-bar');
    const tbl = $('reward-table');
    bar.innerHTML = '';
    tbl.innerHTML = '';
    if (!totals) {
      tbl.innerHTML = '<div class="key">-</div><div>no breakdown yet</div>';
      return;
    }
    const components = ['diagnostic', 'correct_op', 'resolution', 'format', 'efficiency', 'penalty'];
    let total = 0;
    components.forEach((k) => { total += Math.abs(totals[k] || 0); });
    if (total <= 0) total = 1;
    components.forEach((k) => {
      const v = totals[k] || 0;
      // bar segment width is proportional to abs(v)
      const seg = document.createElement('span');
      seg.className = 'seg';
      seg.style.width = (Math.abs(v) / total * 100).toFixed(1) + '%';
      seg.style.background = COMPONENT_COLORS[k] || '#888';
      seg.title = `${k}: ${v.toFixed(3)}`;
      bar.appendChild(seg);

      const k1 = document.createElement('div'); k1.className = 'key';
      const sw = document.createElement('span');
      sw.style.cssText = 'display:inline-block;width:8px;height:8px;border-radius:2px;background:' + (COMPONENT_COLORS[k] || '#888') + ';margin-right:6px;';
      k1.appendChild(sw);
      k1.appendChild(document.createTextNode(k));
      const v1 = document.createElement('div');
      v1.textContent = (v >= 0 ? '+' : '') + v.toFixed(3);
      v1.style.color = v < 0 ? 'var(--bad)' : 'var(--text)';
      tbl.appendChild(k1); tbl.appendChild(v1);
    });
  }

  function renderMap(services) {
    const root = $('map');
    root.innerHTML = '';
    if (!services || Object.keys(services).length === 0) {
      root.innerHTML = '<div class="empty">No services in trace.</div>';
      return;
    }
    Object.values(services).forEach((s) => {
      const card = document.createElement('div');
      card.className = 'svc ' + (s.health || 'unknown');
      card.innerHTML = `
        <div class="name">${s.name}</div>
        <div class="row"><span>${s.health || '-'}</span><span>v${s.version || '?'}</span></div>
        <div class="row"><span>cpu ${s.cpu_percent != null ? s.cpu_percent.toFixed(0) : '-'}%</span><span>${s.memory_mb != null ? s.memory_mb.toFixed(0) : '-'}MB</span></div>
      `;
      root.appendChild(card);
    });
  }

  function appendStep(ev) {
    const tl = $('timeline');
    if (tl.firstChild && tl.firstChild.classList && tl.firstChild.classList.contains('empty')) {
      tl.innerHTML = '';
    }
    const node = document.createElement('div');
    const action = ev.action || {};
    const observation = ev.observation || {};
    const bd = ev.reward_breakdown || {};
    let total = 0;
    Object.values(bd).forEach((v) => { total += Number(v) || 0; });
    let cls = 'step';
    if (total < 0) cls += ' bad';
    else if (total > 0.1) cls += ' good';
    node.className = cls;
    const target = action.target_service ? ' → ' + action.target_service : '';
    const params = action.parameters && Object.keys(action.parameters).length > 0
      ? ' ' + JSON.stringify(action.parameters)
      : '';
    node.innerHTML = `
      <div class="top">
        <span><strong>step ${ev.step}</strong> · ${escape(action.action_type || '?')}${escape(target)}${escape(params)}</span>
        <span>${total >= 0 ? '+' : ''}${total.toFixed(3)}</span>
      </div>
      <div class="body">${escape((observation.message || ev.message || '').slice(0, 600))}</div>
      <div class="reward">${renderRewardChips(bd)}</div>
    `;
    tl.appendChild(node);
    tl.scrollTop = tl.scrollHeight;
  }

  function renderRewardChips(bd) {
    if (!bd) return '';
    const order = ['diagnostic', 'correct_op', 'resolution', 'format', 'efficiency', 'penalty'];
    return order.map((k) => {
      const v = bd[k];
      if (v == null || v === 0) return '';
      const color = COMPONENT_COLORS[k] || '#888';
      const sign = v >= 0 ? '+' : '';
      return `<span style="color:${color}">${k}:${sign}${Number(v).toFixed(2)}</span>`;
    }).filter(Boolean).join(' ');
  }

  function escape(s) {
    return String(s == null ? '' : s).replace(/[&<>]/g, (c) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;'}[c]));
  }

  // ---- Replay control -------------------------------------------------------

  function clearTimeline() {
    $('timeline').innerHTML = '<div class="empty">Replaying…</div>';
  }

  function loadRun(events, summary) {
    state.events = events || [];
    state.summary = summary || {};
    state.cursor = 0;
    pause();
    clearTimeline();
    $('timeline').innerHTML = '';
    renderSummary(Object.assign({ run_id: state.currentRunId }, state.summary));
    // Reward decomp: pull from final 'end' event if present
    const endEv = state.events.find((e) => e.type === 'end');
    renderRewardDecomp(endEv ? endEv.breakdown_totals : null);
    // Service map: derive from final-step observation if present
    const lastStep = [...state.events].reverse().find((e) => e.type === 'step' && e.observation && e.observation.services_summary);
    if (lastStep) {
      const services = {};
      (lastStep.observation.services_summary || []).forEach((s) => {
        services[s.name] = s;
      });
      renderMap(services);
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
      $('timeline').appendChild(node);
      pause();
    }
  }

  function play() {
    pause();
    state.timer = setInterval(step, 700);
  }

  function pause() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  // ---- Boot -----------------------------------------------------------------

  async function init() {
    state.runs = await fetchRuns();
    renderRunsPicker();

    const params = new URLSearchParams(location.search);
    const requested = params.get('watch');
    const sel = $('run-picker');
    if (requested && state.runs.some((r) => r.run_id === requested)) {
      sel.value = requested;
      await selectRun(requested);
    }

    sel.addEventListener('change', (e) => selectRun(e.target.value));
    $('btn-replay').addEventListener('click', () => {
      if (state.currentRunId) {
        selectRun(state.currentRunId).then(() => play());
      }
    });
    $('btn-step').addEventListener('click', step);
    $('btn-pause').addEventListener('click', pause);
  }

  async function selectRun(runId) {
    if (!runId) return;
    state.currentRunId = runId;
    const data = await fetchRun(runId);
    if (!data) return;
    loadRun(data.events || [], data.summary || {});
    play();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
