/* Phase 3 — Real-Time / Sim-to-Real on a deployed site.
 *
 * UI flow:
 *   1) User pastes deployed URL (and optionally a GitHub repo for tier 2).
 *      → POST /realtime/connect  (validates /ops/health on the site)
 *   2) Three chaos buttons get enabled.
 *      → POST /realtime/inject  (calls site's /ops/break)
 *   3) "Run trained agent" enabled.
 *      → POST /realtime/run-agent
 *      → poll GET /realtime/status/<run_id> every 700ms
 *      → render unified tier-1 + tier-2 timeline as events stream in
 */

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    connected: false,
    siteUrl: null,
    repoLinked: false,
    services: [],
    runId: null,
    pollTimer: null,
    activeScenario: null,
    seenEventIdx: 0,
  };

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  }

  // ---- API ------------------------------------------------------------------

  async function api(method, path, body) {
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const r = await fetch(path, opts);
      return await r.json();
    } catch (e) {
      return { error: 'network: ' + e.message };
    }
  }

  // ---- Step indicator -------------------------------------------------------

  function setStep(n, status) {
    // status ∈ {'pending','active','done'}
    [1, 2, 3].forEach(i => {
      const el = $(`rt-num-${i}`);
      if (!el) return;
      el.classList.remove('active', 'done');
      if (i < n || (i === n && status === 'done')) el.classList.add('done');
      else if (i === n) el.classList.add('active');
    });
  }

  // ---- Status panel ---------------------------------------------------------

  function setStatus(text, kind /* 'ok' | 'bad' | null */) {
    const panel = $('rt-status');
    panel.classList.remove('ok', 'bad');
    if (kind) panel.classList.add(kind);
    $('rt-status-text').textContent = text;
  }

  // ---- Connect --------------------------------------------------------------

  async function onConnect() {
    const url = ($('rt-site-url').value || '').trim();
    if (!url) {
      setStatus('Please paste a site URL first', 'bad');
      return;
    }
    setStatus('Connecting…', null);
    $('rt-connect').disabled = true;
    const repoUrl = ($('rt-repo-url').value || '').trim();
    const repoToken = ($('rt-repo-token').value || '').trim();
    const data = await api('POST', '/realtime/connect', {
      site_url: url,
      repo_url: repoUrl || null,
      repo_token: repoToken || null,
    });
    $('rt-connect').disabled = false;
    if (!data.connected) {
      setStatus(`Connection failed — ${data.error || 'unknown error'}`, 'bad');
      state.connected = false;
      enableChaosButtons(false);
      return;
    }
    state.connected = true;
    state.siteUrl = data.site_url;
    state.repoLinked = !!data.repo_linked;
    state.services = data.services_discovered || [];
    const svcText = state.services.length
      ? state.services.join(', ')
      : '(no services advertised)';
    setStatus(
      `Connected · /ops/health = ${data.status || '?'} · services: ${svcText}` +
      (state.repoLinked ? ' · repo linked' : ''),
      'ok'
    );
    setStep(1, 'done');
    setStep(2, 'active');
    enableChaosButtons(true);
    const viewBtn = $('rt-view-site');
    if (viewBtn) {
      viewBtn.href = data.site_url;
      viewBtn.style.display = 'inline-flex';
    }
    $('rt-heal').disabled = false;
  }

  function enableChaosButtons(on) {
    document.querySelectorAll('.rt-chaos-btn').forEach(b => {
      b.disabled = !on;
    });
  }

  // ---- Inject chaos ---------------------------------------------------------

  async function onInject(scenario, btnEl) {
    if (!state.connected) return;
    document.querySelectorAll('.rt-chaos-btn').forEach(b => b.classList.remove('fired'));
    btnEl.classList.add('fired');
    btnEl.style.opacity = 0.6;
    btnEl.disabled = true;
    const res = await api('POST', '/realtime/inject', { scenario });
    if (!res.injected) {
      setStatus(`Injection failed — ${res.error || 'unknown'}`, 'bad');
      btnEl.classList.remove('fired');
      btnEl.style.opacity = 1;
      btnEl.disabled = false;
      return;
    }
    state.activeScenario = scenario;
    setStep(2, 'done');
    setStep(3, 'active');
    $('rt-run').disabled = false;
    pushTimelineLog({
      type: 'inject',
      message: `Injected scenario "${scenario}" — site now in fault state.`,
    });
    document.querySelectorAll('.rt-chaos-btn').forEach((b) => {
      if (b !== btnEl) b.disabled = true;
    });
    setTimeout(() => { btnEl.style.opacity = 1; }, 800);
  }

  // ---- Heal -----------------------------------------------------------------

  async function onHeal() {
    if (!state.connected) return;
    setStatus('Healing site…', null);
    $('rt-heal').disabled = true;
    const r = await api('POST', '/realtime/heal');
    $('rt-heal').disabled = false;
    if (r.healed) {
      setStatus('Site healed — ready for another run', 'ok');
      pushTimelineLog({ type: 'heal', message: 'Site reset to clean state.' });
      // Re-enable chaos buttons
      enableChaosButtons(true);
      document.querySelectorAll('.rt-chaos-btn').forEach(b => b.classList.remove('fired'));
      $('rt-run').disabled = true;
      setStep(2, 'active');
      hideFinalReport();
    } else {
      setStatus(`Heal failed — ${r.error}`, 'bad');
    }
  }

  // ---- Run agent ------------------------------------------------------------

  async function onRunAgent() {
    if (!state.connected || !state.activeScenario) return;
    $('rt-run').disabled = true;
    $('rt-heal').disabled = true;
    state.seenEventIdx = 0;
    clearTimeline();
    hideFinalReport();
    showTierBanner('ops', 'Tier 1 — runtime ops · trained policy is investigating');
    const data = await api('POST', '/realtime/run-agent', {
      scenario: state.activeScenario,
      enable_tier2: true,
    });
    if (!data.run_id) {
      setStatus(`Could not start agent run — ${data.error || 'unknown'}`, 'bad');
      $('rt-run').disabled = false;
      $('rt-heal').disabled = false;
      return;
    }
    state.runId = data.run_id;
    pollStatus();
  }

  function pollStatus() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      if (!state.runId) return;
      const data = await api('GET', '/realtime/status/' + encodeURIComponent(state.runId));
      if (data.error) return;
      const events = data.events || [];
      // Stream new events into the timeline
      while (state.seenEventIdx < events.length) {
        renderEvent(events[state.seenEventIdx]);
        state.seenEventIdx++;
      }
      // Tier banner switches when status changes
      if (data.tier === 'tier2' || (data.tier2_report && !data.tier1_resolved)) {
        // banner already pushed by escalate event
      }
      if (data.status && data.status !== 'running') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        finalizeRun(data);
      }
    }, 700);
  }

  function clearTimeline() {
    $('rt-timeline').innerHTML = '';
  }

  function pushTimelineLog(payload) {
    const tl = $('rt-timeline');
    if (tl.firstChild && tl.firstChild.classList && tl.firstChild.classList.contains('empty-state')) {
      tl.innerHTML = '';
    }
    const node = document.createElement('div');
    node.className = 'step';
    node.innerHTML = `
      <div class="top"><span><strong>${escapeHtml(payload.type || 'event')}</strong></span><span></span></div>
      <div class="body">${escapeHtml(payload.message || '')}</div>
    `;
    tl.appendChild(node);
    tl.scrollTop = tl.scrollHeight;
  }

  function renderEvent(ev) {
    if (ev.type === 'start') {
      pushTimelineLog({
        type: 'start',
        message: `Agent started against ${ev.site_url} · scenario=${ev.scenario}`,
      });
      return;
    }
    if (ev.type === 'step') {
      const tl = $('rt-timeline');
      if (tl.firstChild && tl.firstChild.classList && tl.firstChild.classList.contains('empty-state')) {
        tl.innerHTML = '';
      }
      const a = ev.action || {};
      const target = a.target_service ? ' → ' + a.target_service : '';
      const params = a.parameters && Object.keys(a.parameters).length
        ? ' ' + JSON.stringify(a.parameters) : '';
      const node = document.createElement('div');
      node.className = 'step ' + (ev.error ? 'bad' : 'good');
      node.innerHTML = `
        <div class="top">
          <span><strong>step ${ev.step}</strong> · ${escapeHtml(a.action_type || '?')}${escapeHtml(target)}${escapeHtml(params)}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-muted)">tier1</span>
        </div>
        <div class="body">${escapeHtml((ev.message || '').slice(0, 600))}</div>
      `;
      tl.appendChild(node);
      tl.scrollTop = tl.scrollHeight;
      return;
    }
    if (ev.type === 'tier1_done') {
      if (ev.resolved) {
        showTierBanner('resolved', '✓ Tier 1 brought the site back to healthy.');
      } else {
        showTierBanner('escalate', '⚠ Tier 1 left the site degraded — preparing tier 2…');
      }
      return;
    }
    if (ev.type === 'escalate') {
      showTierBanner('escalate', '⚙ Tier 2 — code investigation: ' + (ev.message || ''));
      return;
    }
    if (ev.type === 'tier2_done') {
      pushTimelineLog({
        type: 'tier2_done',
        message: `Tier 2 complete · ${ev.n_findings || 0} candidate code locations`,
      });
      return;
    }
    if (ev.type === 'tier2_error') {
      pushTimelineLog({
        type: 'tier2_error',
        message: 'Tier 2 escalation failed: ' + (ev.error || ''),
      });
      return;
    }
    if (ev.type === 'inject' || ev.type === 'heal') {
      pushTimelineLog(ev);
      return;
    }
    if (ev.type === 'error') {
      pushTimelineLog({ type: 'error', message: ev.message || 'error' });
      return;
    }
  }

  function showTierBanner(kind, text) {
    const slot = $('rt-tier-banner-slot');
    slot.innerHTML = '';
    const banner = document.createElement('div');
    banner.className = 'rt-tier-banner ' + kind;
    banner.textContent = text;
    slot.appendChild(banner);
  }

  function hideFinalReport() {
    $('rt-final-report').style.display = 'none';
    $('rt-final-body').innerHTML = '';
  }

  function finalizeRun(data) {
    setStep(3, 'done');
    $('rt-heal').disabled = false;
    const card = $('rt-final-report');
    const body = $('rt-final-body');
    card.style.display = 'block';

    let html = '';
    if (data.tier1_resolved) {
      html += `<div class="obs-pill good" style="margin-bottom:10px">RESOLVED in tier 1</div>`;
      html += `<p style="font-size:13px;color:var(--text-secondary);line-height:1.5">The trained agent's runtime ops actions brought the site back to healthy without needing code investigation. Site is now responding 200 on /ops/health.</p>`;
    } else if (data.tier2_report) {
      html += `<div class="obs-pill warn" style="margin-bottom:10px">ESCALATED — tier 1 ops did not fully heal</div>`;
      html += renderTier2Report(data.tier2_report);
    } else {
      html += `<div class="obs-pill bad" style="margin-bottom:10px">UNRESOLVED</div>`;
      html += `<p style="font-size:13px;color:var(--text-secondary)">Tier 1 ops actions did not fully heal the site, and tier 2 was not enabled (no GitHub repo linked).</p>`;
    }
    body.innerHTML = html;

    // Re-enable controls so the user can heal + try again
    $('rt-run').disabled = false;
  }

  function renderTier2Report(report) {
    let html = '';
    html += `<h4 style="font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--text-muted);margin-bottom:8px">Code escalation report</h4>`;
    if (report.error) {
      html += `<p style="color:var(--accent-red);font-size:13px">${escapeHtml(report.error)}</p>`;
      return html;
    }
    html += `<p style="font-size:13px;line-height:1.55;color:var(--text-secondary);margin-bottom:12px">${escapeHtml(report.summary || '')}</p>`;
    if (report.suggested_fix) {
      html += `<div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:12px">
        <div style="font-size:11px;color:var(--accent-orange);text-transform:uppercase;letter-spacing:.6px;font-weight:700;margin-bottom:4px">Suggested fix</div>
        <div style="font-size:13px;color:var(--text-primary);line-height:1.5">${escapeHtml(report.suggested_fix)}</div>
      </div>`;
    }
    if (report.findings && report.findings.length) {
      html += `<div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600;margin-bottom:6px">Top candidate locations</div>`;
      html += `<ul style="list-style:none;padding:0;margin:0;font-family:'JetBrains Mono',monospace;font-size:11.5px">`;
      report.findings.slice(0, 8).forEach(f => {
        html += `<li style="padding:6px 0;border-bottom:1px dashed var(--border)">
          <div style="color:var(--accent-cyan)">${escapeHtml(f.file_path)}:${f.line_no} <span style="color:var(--text-muted)">[${f.score}]</span></div>
          <div style="color:var(--text-secondary);margin-top:2px">${escapeHtml(f.snippet)}</div>
        </li>`;
      });
      html += `</ul>`;
    }
    return html;
  }

  // ---- Init -----------------------------------------------------------------

  function init() {
    if (window.Realtime && window.Realtime._inited) return;
    $('rt-connect').addEventListener('click', onConnect);
    $('rt-site-url').addEventListener('keydown', e => { if (e.key === 'Enter') onConnect(); });
    $('rt-heal').addEventListener('click', onHeal);
    $('rt-run').addEventListener('click', onRunAgent);
    document.querySelectorAll('.rt-chaos-btn').forEach(btn => {
      btn.addEventListener('click', () => onInject(btn.dataset.scenario, btn));
    });
    setStep(1, 'active');
    // Pre-populate from /realtime/config if a previous session connected
    api('GET', '/realtime/config').then(cfg => {
      if (cfg && cfg.site_url) {
        $('rt-site-url').value = cfg.site_url;
        setStatus(`Previously connected: ${cfg.site_url}`, null);
      }
    });
    if (window.Realtime) window.Realtime._inited = true;
  }

  window.Realtime = { init, _inited: false };
})();
