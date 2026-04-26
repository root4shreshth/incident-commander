/* Phase 3 — Real-Time / Praetor on a deployed site.
 *
 * Flow:
 *   1) User pastes deployed URL → POST /realtime/connect
 *      ⤷ server probes /ops/health, /ops/metrics, /ops/logs and
 *        AUTO-CLASSIFIES the fault. UI renders the verdict card.
 *   2) (Optional) link a codebase: GitHub URL, Azure DevOps URL, or ZIP upload
 *      ⤷ POST /realtime/codebase/{link,upload-multipart}
 *   3) User clicks "Run Praetor" → POST /realtime/run-agent (no scenario param —
 *      the server uses the auto-classified one).
 *      ⤷ poll GET /realtime/status/<run_id> every 700ms
 *      ⤷ stream events into the unified tier-1 + tier-2 timeline.
 *
 * The "inject test fault" buttons are SECONDARY (collapsed in a <details>) —
 * useful when the connected site is healthy and we want to demo the agent's
 * response without waiting for a real outage.
 */

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    connected: false,
    siteUrl: null,
    services: [],
    classification: null,
    codebaseLinked: false,
    codebaseSource: null,
    runId: null,
    pollTimer: null,
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
    [1, 2, 3, 4].forEach(i => {
      const el = $(`rt-num-${i}`);
      if (!el) return;
      el.classList.remove('active', 'done');
      if (i < n || (i === n && status === 'done')) el.classList.add('done');
      else if (i === n) el.classList.add('active');
    });
  }

  // ---- Status panel ---------------------------------------------------------

  function setStatus(text, kind) {
    const panel = $('rt-status');
    panel.classList.remove('ok', 'bad');
    if (kind) panel.classList.add(kind);
    $('rt-status-text').textContent = text;
  }

  function setCbStatus(text, kind) {
    const panel = $('cb-status');
    panel.style.display = 'flex';
    panel.classList.remove('ok', 'bad');
    if (kind) panel.classList.add(kind);
    $('cb-status-text').textContent = text;
  }

  // ---- Connect → auto-classify ----------------------------------------------

  async function onConnect() {
    const url = ($('rt-site-url').value || '').trim();
    if (!url) {
      setStatus('Please paste a site URL first', 'bad');
      return;
    }
    setStatus('Connecting…', null);
    $('rt-connect').disabled = true;
    const data = await api('POST', '/realtime/connect', { site_url: url });
    $('rt-connect').disabled = false;
    if (!data.connected) {
      const err = data.error || 'unknown error';
      let detail = `Connection failed — ${err}`;
      // If the site doesn't implement /ops/health, suggest the built-in demo target
      if (/404|not found|failed/i.test(err)) {
        detail += '. The site needs to expose /ops/health (operator contract). ' +
                  'Try the "Use built-in demo target" link below — it points at this server\'s own /ops/* endpoints.';
      }
      setStatus(detail, 'bad');
      state.connected = false;
      return;
    }
    state.connected = true;
    state.siteUrl = data.site_url;
    state.services = data.services_discovered || [];
    state.classification = data.classification || null;
    const svcText = state.services.length ? state.services.join(', ') : '(no services advertised)';
    setStatus(`Connected · /ops/health = ${data.status || '?'} · services: ${svcText}`, 'ok');
    setStep(1, 'done');
    setStep(2, 'active');
    renderClassification(state.classification);
    enableChaosButtons(true);
    const viewBtn = $('rt-view-site');
    if (viewBtn) {
      viewBtn.href = data.site_url;
      viewBtn.style.display = 'inline-flex';
    }
    $('rt-heal').disabled = false;
    $('rt-run').disabled = false;  // even on healthy site: agent will probe + report
  }

  function renderClassification(cls) {
    const card = $('rt-classify-card');
    const body = $('rt-classify-body');
    const ev = $('rt-evidence');
    if (!cls) {
      card.style.display = 'none';
      return;
    }
    card.style.display = 'block';
    let verdict, narrative;
    if (cls.fault_detected) {
      verdict = `<div class="rt-classify-verdict fault">⚡ ${escapeHtml((cls.scenario || 'unknown').replace(/_/g, ' ').toUpperCase())} · confidence ${(cls.confidence * 100).toFixed(0)}%</div>`;
    } else {
      verdict = `<div class="rt-classify-verdict healthy">✓ Site healthy</div>`;
    }
    narrative = (cls.narrative || '').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    body.innerHTML = verdict + '<div>' + narrative + '</div>';
    if (cls.evidence && cls.evidence.length) {
      let html = '<div class="rt-evidence-list">';
      cls.evidence.forEach(e => { html += `<div class="rt-evidence-item">${escapeHtml(e)}</div>`; });
      html += '</div>';
      ev.innerHTML = html;
    } else {
      ev.innerHTML = '';
    }
  }

  function enableChaosButtons(on) {
    document.querySelectorAll('.rt-chaos-btn').forEach(b => { b.disabled = !on; });
  }

  // ---- Codebase source tabs -------------------------------------------------

  function wireCodebaseTabs() {
    document.querySelectorAll('.cb-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const src = tab.dataset.cbsrc;
        document.querySelectorAll('.cb-tab').forEach(t => t.classList.toggle('active', t === tab));
        document.querySelectorAll('.cb-panel').forEach(p => {
          p.style.display = p.dataset.cbsrc === src ? '' : 'none';
        });
      });
    });
    $('cb-github-link').addEventListener('click', () => linkRemote('github'));
    $('cb-azure-link').addEventListener('click', () => linkRemote('azure'));

    // ZIP drag-and-drop + click-to-pick
    const drop = $('cb-zip-drop');
    const input = $('cb-zip-input');
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', e => {
      if (e.target.files && e.target.files[0]) uploadZip(e.target.files[0]);
    });
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) uploadZip(e.dataTransfer.files[0]);
    });
  }

  async function linkRemote(source) {
    const url = ($(`cb-${source}-url`).value || '').trim();
    const token = ($(`cb-${source}-token`).value || '').trim() || null;
    if (!url) {
      setCbStatus(`Please paste the ${source === 'github' ? 'GitHub' : 'Azure DevOps'} repo URL`, 'bad');
      return;
    }
    setCbStatus('Linking…', null);
    const data = await api('POST', '/realtime/codebase/link', {
      source, repo_url: url, repo_token: token,
    });
    if (!data.linked) {
      setCbStatus(`Link failed — ${data.error}`, 'bad');
      state.codebaseLinked = false;
      return;
    }
    state.codebaseLinked = true;
    state.codebaseSource = source;
    setCbStatus(`Linked (${source}): ${data.repo_url}`, 'ok');
  }

  async function uploadZip(file) {
    if (!file.name.toLowerCase().endsWith('.zip')) {
      setCbStatus('Expected a .zip file', 'bad');
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      setCbStatus(`File too large (${(file.size/1024/1024).toFixed(1)} MB > 25 MB)`, 'bad');
      return;
    }
    setCbStatus(`Uploading ${file.name} (${(file.size/1024).toFixed(0)} KB)…`, null);
    $('cb-drop-text').innerHTML = `<strong>${escapeHtml(file.name)}</strong> — uploading…`;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/realtime/codebase/upload-multipart', {
        method: 'POST', body: fd,
      });
      const data = await r.json();
      if (!data.linked) {
        setCbStatus(`Upload failed — ${data.error}`, 'bad');
        return;
      }
      state.codebaseLinked = true;
      state.codebaseSource = 'zip';
      setCbStatus(`Uploaded ${data.filename} (${(data.size_bytes/1024).toFixed(0)} KB)`, 'ok');
      $('cb-drop-text').innerHTML = `✓ <strong>${escapeHtml(data.filename)}</strong> ready for tier-2 escalation`;
    } catch (e) {
      setCbStatus('Upload failed: ' + e.message, 'bad');
    }
  }

  // ---- Inject chaos (secondary path) ----------------------------------------

  async function onInject(scenario, btnEl) {
    if (!state.connected) return;
    document.querySelectorAll('.rt-chaos-btn').forEach(b => b.classList.remove('fired'));
    btnEl.classList.add('fired');
    btnEl.disabled = true;
    const res = await api('POST', '/realtime/inject', { scenario });
    if (!res.injected) {
      setStatus(`Injection failed — ${res.error || 'unknown'}`, 'bad');
      btnEl.classList.remove('fired');
      btnEl.disabled = false;
      return;
    }
    pushTimelineLog({
      type: 'inject',
      message: `Injected scenario "${scenario}" — site is now in a fault state. Re-classifying…`,
    });
    // Re-fetch classification by reconnecting with same URL
    setTimeout(async () => {
      const cfg = await api('GET', '/realtime/config');
      if (cfg && cfg.site_url) {
        const data = await api('POST', '/realtime/connect', { site_url: cfg.site_url });
        if (data.classification) {
          state.classification = data.classification;
          renderClassification(data.classification);
        }
      }
      btnEl.disabled = false;
    }, 800);
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
      enableChaosButtons(true);
      document.querySelectorAll('.rt-chaos-btn').forEach(b => b.classList.remove('fired'));
      hideFinalReport();
      // Re-classify after heal
      const cfg = await api('GET', '/realtime/config');
      if (cfg && cfg.site_url) {
        const data = await api('POST', '/realtime/connect', { site_url: cfg.site_url });
        if (data.classification) {
          state.classification = data.classification;
          renderClassification(data.classification);
        }
      }
    } else {
      setStatus(`Heal failed — ${r.error}`, 'bad');
    }
  }

  // ---- Run agent ------------------------------------------------------------

  async function onRunAgent() {
    if (!state.connected) return;
    $('rt-run').disabled = true;
    $('rt-heal').disabled = true;
    state.seenEventIdx = 0;
    clearTimeline();
    hideFinalReport();
    showTierBanner('ops', 'Tier 1 — runtime ops · Praetor is investigating');
    // Don't pass scenario — let the server auto-classify
    const data = await api('POST', '/realtime/run-agent', { enable_tier2: true });
    if (!data.run_id) {
      setStatus(`Could not start agent run — ${data.error || 'unknown'}`, 'bad');
      $('rt-run').disabled = false;
      $('rt-heal').disabled = false;
      return;
    }
    state.runId = data.run_id;
    if (data.auto_classified) {
      pushTimelineLog({
        type: 'classify',
        message: `Praetor classified the fault as "${(data.scenario || '').replace(/_/g, ' ')}" and is now acting on it.`,
      });
    }
    pollStatus();
  }

  function pollStatus() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      if (!state.runId) return;
      const data = await api('GET', '/realtime/status/' + encodeURIComponent(state.runId));
      if (data.error) return;
      const events = data.events || [];
      while (state.seenEventIdx < events.length) {
        renderEvent(events[state.seenEventIdx]);
        state.seenEventIdx++;
      }
      if (data.status && data.status !== 'running') {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        finalizeRun(data);
      }
    }, 700);
  }

  function clearTimeline() { $('rt-timeline').innerHTML = ''; }

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
      pushTimelineLog({ type: 'start', message: `Praetor started · ${ev.site_url} · scenario=${ev.scenario}` });
      return;
    }
    if (ev.type === 'step') {
      const tl = $('rt-timeline');
      if (tl.firstChild && tl.firstChild.classList && tl.firstChild.classList.contains('empty-state')) {
        tl.innerHTML = '';
      }
      // Drop the "latest" marker from the previous step (so only the newest animates).
      tl.querySelectorAll('.step.is-latest').forEach(el => el.classList.remove('is-latest'));

      const a = ev.action || {};
      const target = a.target_service ? ' → ' + a.target_service : '';
      const params = a.parameters && Object.keys(a.parameters).length
        ? ' ' + JSON.stringify(a.parameters) : '';
      const why = ev.why || '';
      const tier = ev.tier || 'tier1';
      const node = document.createElement('div');
      node.className = 'step is-latest is-fresh ' + (ev.error ? 'bad' : 'good');
      node.innerHTML = `
        <div class="top">
          <span class="step-head-left">
            <span class="step-num-pill">step ${ev.step}</span>
            <span class="step-action-name">${escapeHtml(a.action_type || '?')}</span>
            ${target ? `<span class="step-target">${escapeHtml(target)}</span>` : ''}
            ${params ? `<span class="step-params">${escapeHtml(params)}</span>` : ''}
          </span>
          <span class="step-tier-pill">${escapeHtml(tier)}</span>
        </div>
        <div class="body">${escapeHtml((ev.message || '').slice(0, 600))}</div>
        ${why ? `
        <div class="step-why-row">
          <button class="step-why-toggle" data-expanded="false" aria-expanded="false">
            <span class="why-icon">▸</span> Why this step?
          </button>
          <div class="step-why-body" hidden>${escapeHtml(why)}</div>
        </div>` : ''}
      `;
      tl.appendChild(node);

      // Wire the Why button (if present)
      const toggle = node.querySelector('.step-why-toggle');
      if (toggle) {
        toggle.addEventListener('click', () => {
          const body = node.querySelector('.step-why-body');
          const expanded = toggle.dataset.expanded === 'true';
          toggle.dataset.expanded = String(!expanded);
          toggle.setAttribute('aria-expanded', String(!expanded));
          toggle.querySelector('.why-icon').textContent = expanded ? '▸' : '▾';
          body.hidden = expanded;
        });
      }

      // Remove the "fresh" class after the entrance animation completes
      // so re-rendering doesn't keep re-triggering it.
      setTimeout(() => node.classList.remove('is-fresh'), 600);

      tl.scrollTop = tl.scrollHeight;
      return;
    }
    if (ev.type === 'tier1_done') {
      if (ev.resolved) {
        showTierBanner('resolved', '✓ Tier 1 brought the site back to healthy.');
      } else {
        showTierBanner('escalate', '⚠ Tier 1 left the site degraded — preparing tier 2 (code investigation)…');
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
        message: `Tier 2 complete · ${ev.n_findings || 0} candidate code locations identified`,
      });
      return;
    }
    if (ev.type === 'tier2_error') {
      pushTimelineLog({ type: 'tier2_error', message: 'Tier 2 escalation failed: ' + (ev.error || '') });
      return;
    }
    if (ev.type === 'inject' || ev.type === 'heal' || ev.type === 'classify') {
      pushTimelineLog(ev); return;
    }
    if (ev.type === 'error') {
      pushTimelineLog({ type: 'error', message: ev.message || 'error' }); return;
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

  // Map a scenario name + observed actions into a small set of human-readable
  // tags surfacing the root-cause category, the fix mechanism, and the
  // operating envelope. These appear as colored chips in the final report.
  function deriveReportTags(scenario, events) {
    const tagsByScenario = {
      oom_crash:               { cause: ['oom', 'memory-pressure'],    fix: ['memory-bump', 'restart-curable'] },
      db_pool_exhaustion:      { cause: ['connection-leak', 'pool-saturated'], fix: ['config-tune', 'pool-size-bump'] },
      bad_deployment_cascade:  { cause: ['bad-deploy', 'cascading-failure'],   fix: ['rollback', 'version-revert'] },
      disk_full:               { cause: ['disk-full', 'storage-exhausted'],    fix: ['restart-cycle-volume', 'log-rotation'] },
      slow_query:              { cause: ['lock-contention', 'query-regression'], fix: ['rollback', 'feature-flag'] },
      cert_expiry:             { cause: ['cert-expired', 'tls-handshake-fail'],  fix: ['restart-renew', 'cert-rotation'] },
      dns_failure:             { cause: ['dns-resolution', 'resolver-stale'],    fix: ['cache-flush', 'restart-resolver'] },
      rate_limit_exhaustion:   { cause: ['rate-limited', '429-storm'],           fix: ['scale-out', 'key-rotate'] },
    };
    const t = tagsByScenario[scenario] || { cause: [], fix: [] };
    // Affected services derived from step targets
    const services = new Set();
    (events || []).forEach(ev => {
      if (ev.type === 'step' && ev.action && ev.action.target_service) {
        services.add(ev.action.target_service);
      }
    });
    return {
      cause: t.cause,
      fix: t.fix,
      services: Array.from(services),
    };
  }

  function summarizeActions(events) {
    const counts = {};
    (events || []).forEach(ev => {
      if (ev.type === 'step' && ev.action && ev.action.action_type) {
        counts[ev.action.action_type] = (counts[ev.action.action_type] || 0) + 1;
      }
    });
    return counts; // { read_logs: 2, restart_service: 1, ... }
  }

  function buildResolutionPath(events) {
    // Pick the meaningful actions — drop diagnostics, keep the moves
    const meaningful = new Set([
      'restart_service', 'rollback_deployment', 'scale_service',
      'update_config', 'resolve_incident',
    ]);
    return (events || [])
      .filter(ev => ev.type === 'step' && ev.action && meaningful.has(ev.action.action_type))
      .map(ev => {
        const a = ev.action;
        const tgt = a.target_service ? ' on ' + a.target_service : '';
        const params = a.parameters && Object.keys(a.parameters).length
          ? ' ' + JSON.stringify(a.parameters) : '';
        return `${a.action_type}${tgt}${params}`;
      });
  }

  function buildPraetorSummary(scenario, events, resolved, durationS, tier2) {
    const stepsTaken = (events || []).filter(ev => ev.type === 'step').length;
    const acts = summarizeActions(events);
    const moves = Object.keys(acts).filter(k => /restart|rollback|update_config|scale/.test(k));
    const friendly = (scenario || 'unknown').replace(/_/g, ' ');
    if (resolved) {
      const moveStr = moves.length ? ` Praetor's decisive move was <strong>${moves.join(', ')}</strong>.` : '';
      return `Praetor diagnosed a <strong>${friendly}</strong> incident in <strong>${stepsTaken} step${stepsTaken === 1 ? '' : 's'}</strong>${durationS ? ' over ' + durationS.toFixed(1) + ' seconds' : ''}, walked the dependency graph from symptom to root cause, and resolved the incident using only tier-1 runtime operations.${moveStr} The site is now responding 200 on /ops/health and the fix is durable.`;
    }
    if (tier2) {
      return `Praetor investigated a <strong>${friendly}</strong> incident, took ${stepsTaken} runtime ops action${stepsTaken === 1 ? '' : 's'}, and determined the fault was not fully restorable from runtime ops alone. It escalated to tier-2 code investigation against the linked repository and identified candidate code locations to review (see below).`;
    }
    return `Praetor took <strong>${stepsTaken}</strong> runtime ops action${stepsTaken === 1 ? '' : 's'} against the <strong>${friendly}</strong> fault but did not fully heal the site. Tier-2 code investigation was not enabled — link a repository in step 3 to let Praetor inspect the code path next time.`;
  }

  function finalizeRun(data) {
    setStep(4, 'done');
    $('rt-heal').disabled = false;
    $('rt-run').disabled = false;
    const card = $('rt-final-report');
    const body = $('rt-final-body');
    card.style.display = 'block';
    card.scrollIntoView({behavior: 'smooth', block: 'start'});

    const events = data.events || [];
    const stepEvents = events.filter(ev => ev.type === 'step');
    const startEv = events.find(ev => ev.type === 'start') || {};
    const tier2Done = events.find(ev => ev.type === 'tier2_done');
    const scenario = data.scenario || startEv.scenario || 'unknown';
    const siteUrl = data.site_url || startEv.site_url || '(in-process simulator)';
    const alert = data.alert_title || data.alert_summary || '';
    const durationS = (data.started_at && data.finished_at)
      ? (data.finished_at - data.started_at) : null;

    const resolved = !!data.tier1_resolved;
    const escalated = !resolved && !!data.tier2_report;
    let statusClass, statusLabel;
    if (resolved)       { statusClass = 'good';  statusLabel = 'RESOLVED in tier 1'; }
    else if (escalated) { statusClass = 'warn';  statusLabel = 'ESCALATED — tier 1 did not fully heal'; }
    else                { statusClass = 'bad';   statusLabel = 'UNRESOLVED'; }

    const tags = deriveReportTags(scenario, events);
    const acts = summarizeActions(events);
    const path = buildResolutionPath(events);
    const summary = buildPraetorSummary(scenario, events, resolved, durationS, tier2Done);

    let html = '';
    // Status row
    html += `<div class="fr-status-row">
      <div class="obs-pill ${statusClass}">${escapeHtml(statusLabel)}</div>
      <div style="font-size:11.5px;color:var(--text-muted)">scenario: <code>${escapeHtml(scenario)}</code></div>
    </div>`;

    // Praetor's summary
    html += `<div class="fr-summary">${summary}</div>`;

    // Stats grid
    html += `<div class="fr-stats">
      <div class="fr-stat"><div class="lbl">Steps taken</div><div class="val">${stepEvents.length}</div><div class="sub">tier-1 ops actions</div></div>
      <div class="fr-stat"><div class="lbl">Wall-clock</div><div class="val">${durationS ? durationS.toFixed(1) + 's' : '—'}</div><div class="sub">end-to-end</div></div>
      <div class="fr-stat"><div class="lbl">Outcome</div><div class="val" style="color:var(--accent-${resolved ? 'green' : (escalated ? 'orange' : 'red')})">${resolved ? '✓ FIXED' : (escalated ? '↗ ESCALATED' : '✗ UNRESOLVED')}</div><div class="sub">${resolved ? '/ops/health = ok' : (escalated ? 'tier-2 surfaced fixes' : 'no tier-2 link')}</div></div>
      <div class="fr-stat"><div class="lbl">Services touched</div><div class="val">${tags.services.length || 0}</div><div class="sub">${tags.services.length ? tags.services.slice(0,3).join(', ') : 'none'}</div></div>
    </div>`;

    // Original alert
    if (alert) {
      html += `<div class="fr-section">
        <div class="fr-section-h">⚠ The problem we saw</div>
        <div class="fr-alert-box">${escapeHtml(alert)}</div>
      </div>`;
    }

    // Root cause tags
    if (tags.cause.length || tags.fix.length || tags.services.length) {
      html += `<div class="fr-section"><div class="fr-section-h">🏷 Root cause &amp; fix tags</div><div class="fr-tags">`;
      tags.cause.forEach(t => { html += `<span class="fr-tag tag-cause" title="root cause">${escapeHtml(t)}</span>`; });
      tags.fix.forEach(t   => { html += `<span class="fr-tag tag-fix"   title="fix mechanism">${escapeHtml(t)}</span>`; });
      tags.services.forEach(s => { html += `<span class="fr-tag tag-svc" title="affected service">${escapeHtml(s)}</span>`; });
      html += `</div></div>`;
    }

    // Resolution path
    if (path.length) {
      html += `<div class="fr-section"><div class="fr-section-h">⚙ Resolution path</div><ul class="fr-resolution-list">`;
      path.forEach(p => { html += `<li><code>${escapeHtml(p)}</code></li>`; });
      html += `</ul></div>`;
    }

    // Action breakdown
    const actEntries = Object.entries(acts).sort((a, b) => b[1] - a[1]);
    if (actEntries.length) {
      html += `<div class="fr-section"><div class="fr-section-h">📊 Action breakdown</div><div class="fr-actions-list">`;
      actEntries.forEach(([k, v]) => {
        html += `<div class="fr-action-row"><span class="fr-action-name">${escapeHtml(k)}</span><span class="fr-action-count">×${v}</span></div>`;
      });
      html += `</div></div>`;
    }

    // Tier-2 report (if escalated)
    if (escalated && data.tier2_report) {
      html += `<div class="fr-section">${renderTier2Report(data.tier2_report)}</div>`;
    }
    // Export-as-PDF button (opens a print-ready report in a new tab).
    // We render real <button>s rather than <a target=_blank> so we can
    // (a) catch popup blockers and fall back to a same-tab navigation,
    // (b) probe the endpoint first and surface a clear error if the run
    //     record was already evicted server-side.
    if (state.runId) {
      html += `
        <div class="rt-export-row">
          <button class="btn btn-primary btn-sm" data-rt-export="pdf"
            title="Opens the report in a new tab and triggers the browser print dialog. Pick 'Save as PDF' to download.">
            📄 Export as PDF
          </button>
          <button class="btn btn-outline btn-sm" data-rt-export="preview"
            title="Open the report without the auto-print dialog.">
            ↗ Preview report
          </button>
          <span class="rt-export-hint">PDF includes the alert, every step with reasoning, and the final result.</span>
          <span class="rt-export-msg" data-rt-export-msg></span>
        </div>
      `;
    }
    body.innerHTML = html;

    // Wire export buttons after the DOM has been swapped in.
    if (state.runId) {
      const runId = state.runId;
      const pdfUrl = '/realtime/run/' + encodeURIComponent(runId) + '/report.pdf';
      const htmlUrl = '/realtime/run/' + encodeURIComponent(runId) + '/report?noprint=1';
      const msgEl = body.querySelector('[data-rt-export-msg]');
      const setMsg = (txt, kind) => {
        if (!msgEl) return;
        msgEl.textContent = txt || '';
        msgEl.dataset.kind = kind || '';
      };

      // Real PDF download: fetch the .pdf endpoint as a blob, build an
      // object-URL, and click an invisible <a download> to trigger a real
      // file download — no browser print dialog, no popup blocker, no
      // window.open. Falls back informatively on errors.
      const downloadPdf = async () => {
        setMsg('Generating PDF…', 'pending');
        try {
          const resp = await fetch(pdfUrl, { method: 'GET', cache: 'no-store' });
          if (!resp.ok) {
            const txt = await resp.text().catch(() => '');
            setMsg(
              `PDF not available (HTTP ${resp.status})${txt ? ': ' + txt.slice(0, 200) : '.'} ` +
              `Start a new run and try again.`,
              'error',
            );
            return;
          }
          const blob = await resp.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `praetor-incident-${runId}.pdf`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 2000);
          const sizeKb = Math.round(blob.size / 1024);
          setMsg(`PDF downloaded (${sizeKb} KB).`, 'ok');
        } catch (err) {
          setMsg('Download failed: ' + err, 'error');
        }
      };

      // Preview opens the HTML report in a new tab so the user can read it
      // on screen without printing. Probes the endpoint first to give a
      // clear error if the run record was already evicted.
      const openPreview = async () => {
        setMsg('Opening preview…', 'pending');
        try {
          const head = await fetch(htmlUrl, { method: 'GET', cache: 'no-store' });
          if (!head.ok) {
            setMsg(`Preview not available (HTTP ${head.status}). Start a new run and try again.`, 'error');
            return;
          }
        } catch (err) {
          setMsg('Network error: ' + err, 'error');
          return;
        }
        const win = window.open(htmlUrl, '_blank', 'noopener');
        if (!win) {
          setMsg('Popup blocked — opening in this tab.', 'warn');
          window.location.href = htmlUrl;
          return;
        }
        setMsg('Preview opened in a new tab.', 'ok');
      };

      const pdfBtn = body.querySelector('[data-rt-export="pdf"]');
      const prevBtn = body.querySelector('[data-rt-export="preview"]');
      if (pdfBtn) pdfBtn.addEventListener('click', downloadPdf);
      if (prevBtn) prevBtn.addEventListener('click', openPreview);
    }
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
        <div style="font-size:11px;color:var(--phase-accent);text-transform:uppercase;letter-spacing:.6px;font-weight:700;margin-bottom:4px">Suggested fix</div>
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
    // "Use built-in demo target" — points at this server's own /ops/* endpoints
    // so the user can demo Real-Time without deploying a separate site.
    const builtinLink = $('rt-use-builtin');
    if (builtinLink) {
      builtinLink.addEventListener('click', (ev) => {
        ev.preventDefault();
        $('rt-site-url').value = window.location.origin;
        onConnect();
      });
    }
    wireCodebaseTabs();
    setStep(1, 'active');
    api('GET', '/realtime/config').then(cfg => {
      if (cfg && cfg.site_url) {
        $('rt-site-url').value = cfg.site_url;
        setStatus(`Previously connected: ${cfg.site_url}`, null);
      }
      if (cfg && cfg.repo_linked) {
        state.codebaseLinked = true;
        state.codebaseSource = cfg.repo_source;
        setCbStatus(`Codebase linked (${cfg.repo_source})`, 'ok');
      }
    });
    if (window.Realtime) window.Realtime._inited = true;
  }

  window.Realtime = { init, _inited: false };
})();
