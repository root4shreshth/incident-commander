// demo.js — the human-facing incident response trainer.
// Everything: state, API, service-map, action buttons, modals, coach, notebook, post-mortem.

(function () {
  'use strict';

  // ---------- State ----------
  const state = {
    tasks: {},              // task_id → metadata
    completed: new Set(Auth.loadCompleted()),
    currentTask: null,
    episodeActive: false,
    stepCount: 0,
    maxSteps: 15,
    score: 0.0,
    services: [],           // live services_summary
    rewards: [],            // per-step rewards for timeline
    actionCards: [],        // notebook cards
    lastObservation: null,
    lastAction: null,
    mode: localStorage.getItem('ic_mode') || 'junior',   // junior | pro
    selectedSvc: null,
  };

  function reloadCompletedForCurrentUser() {
    state.completed = new Set(Auth.loadCompleted());
  }

  // ---------- Action intent metadata ----------
  const ACTIONS = {
    list_services:       { group: 'investigate', label: 'See all services',   icon: () => Icons.layoutGrid(18),    desc: 'Cluster overview.' },
    describe_service:    { group: 'investigate', label: 'Service details',    icon: () => Icons.fileText(18),      desc: 'Config, deployment history, dependencies.', needsTarget: true },
    read_logs:           { group: 'investigate', label: 'Read logs',          icon: () => Icons.scrollText(18),    desc: 'Recent log lines.', needsTarget: true, params: ['lines','severity'] },
    check_metrics:       { group: 'investigate', label: 'Check metrics',      icon: () => Icons.barChart3(18),     desc: 'CPU, memory, latency.', needsTarget: true },
    run_diagnostic:      { group: 'investigate', label: 'Run diagnostic',     icon: () => Icons.stethoscope(18),   desc: 'Health, connectivity, resources.', needsTarget: true, params: ['command'] },
    restart_service:     { group: 'remediate',   label: 'Restart service',    icon: () => Icons.rotateCw(18),      desc: 'Bounce the service. -0.10 if it was healthy.', needsTarget: true, params: ['memory_limit'], danger: true },
    rollback_deployment: { group: 'remediate',   label: 'Rollback version',   icon: () => Icons.rotateCcw(18),     desc: 'Roll back to a prior deploy.', needsTarget: true, params: ['to_version'] },
    scale_service:       { group: 'remediate',   label: 'Scale replicas',     icon: () => Icons.trendingUp(18),    desc: 'Change replica count.', needsTarget: true, params: ['replicas'] },
    update_config:       { group: 'remediate',   label: 'Update config',      icon: () => Icons.settings2(18),     desc: 'Change runtime config (e.g. pool size).', needsTarget: true, params: ['key','value'] },
    resolve_incident:    { group: 'declare',     label: 'Resolve incident',   icon: () => Icons.checkCircle2(18),  desc: 'Declare the incident resolved.', params: ['root_cause','resolution'] },
  };

  // ---------- Helpers ----------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const el = (tag, attrs = {}, children = []) => {
    const n = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'class') n.className = v;
      else if (k === 'onclick') n.addEventListener('click', v);
      else if (k === 'html') n.innerHTML = v;
      else n.setAttribute(k, v);
    });
    (Array.isArray(children) ? children : [children]).forEach(c => c && n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return n;
  };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // ---------- API ----------
  const api = {
    async tasks() { return (await fetch('/tasks')).json(); },
    async reset(taskId) {
      const r = await fetch('/reset', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ task_id: taskId }) });
      return r.json();
    },
    async step(action, target, params) {
      const r = await fetch('/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ action_type: action, target_service: target, parameters: params || {} }) });
      return r.json();
    },
    async listServices() {
      return this.step('list_services', null, {});
    },
  };

  // ---------- Init ----------
  async function init() {
    document.body.setAttribute('data-mode', state.mode);
    bindGlobal();
    populateStaticIcons();
    renderAuthScreen();
    renderUserSlot();
    await loadTasks();
    applyAuthGate();
    if (Auth.isSignedIn() && !localStorage.getItem('ic_tutorial_done')) {
      showTutorial();
    }
  }

  // ---------- Static icon slot population ----------
  function populateStaticIcons() {
    const slot = (id, svg) => { const e = document.getElementById(id); if (e) e.innerHTML = svg; };
    slot('svc-map-head-ic', Icons.network(16));
    slot('notebook-head-ic', Icons.fileText(16));
    slot('coach-head-ic', Icons.sparkles(16));
    slot('coach-refresh-ic', Icons.rotateCcw(13));
    slot('reset-progress-ic', Icons.rotateCcw(12));
    slot('back-to-picker-ic', Icons.chevronLeft(14));
    const help = document.getElementById('tutorial-restart');
    if (help) help.innerHTML = Icons.helpCircle(14);
  }

  // ---------- Auth screen icons ----------
  function renderAuthScreen() {
    const brandHtml = `<div class="auth-brand-icon">${Icons.siren(20)}</div><span>IncidentCommander</span>`;
    const desktop = document.getElementById('auth-brand-desktop');
    const mobile = document.getElementById('auth-brand-mobile');
    if (desktop) desktop.innerHTML = brandHtml;
    if (mobile) mobile.innerHTML = brandHtml;

    const back = document.getElementById('auth-back-btn');
    if (back) {
      back.innerHTML = `${Icons.chevronLeft(14)} Home`;
      back.addEventListener('click', () => switchTab('home'));
    }

    const social = document.getElementById('auth-social-btns');
    if (social) {
      social.innerHTML = `
        <button type="button" class="auth-btn" data-provider="google">${Icons.google(18)} Continue with Google</button>
        <button type="button" class="auth-btn" data-provider="apple">${Icons.apple(18)} Continue with Apple</button>
        <button type="button" class="auth-btn" data-provider="github">${Icons.github(18)} Continue with GitHub</button>`;
    }

    const emailWrap = document.getElementById('auth-email-wrap');
    if (emailWrap && !emailWrap.querySelector('svg')) {
      const iconSpan = document.createElement('span');
      iconSpan.style.cssText = 'position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--text-muted);pointer-events:none;display:flex';
      iconSpan.innerHTML = Icons.atSign(16);
      emailWrap.insertBefore(iconSpan, emailWrap.firstChild);
    }
  }

  // (floating paths animation removed — replaced with a static photo background for reliability)

  // ---------- Auth gating ----------
  function applyAuthGate() {
    const authScreen = $('#auth-screen');
    const picker = $('#scenario-picker');
    const incident = $('#incident-screen');
    if (!authScreen || !picker) return;
    if (Auth.isSignedIn()) {
      authScreen.classList.add('hidden');
      picker.classList.remove('hidden');
      reloadCompletedForCurrentUser();
      renderScenarioPicker();
    } else {
      authScreen.classList.remove('hidden');
      picker.classList.add('hidden');
      if (incident) incident.classList.add('hidden');
    }
  }

  function renderUserSlot() {
    const slot = $('#user-slot');
    if (!slot) return;
    const user = Auth.currentUser();
    if (!user) {
      slot.innerHTML = `<button class="nav-signin" id="nav-signin">Sign in</button>`;
      $('#nav-signin').addEventListener('click', () => { switchTab('demo'); applyAuthGate(); });
      return;
    }
    const initials = (user.name || user.email).split(/\s+/).map(w => w[0]).slice(0,2).join('').toUpperCase();
    slot.innerHTML = `
      <div class="user-badge" id="user-badge">
        <div class="user-avatar">${escapeHtml(initials)}</div>
        <span class="user-name">${escapeHtml(user.name)}</span>
        <div class="user-menu">
          <div class="user-menu-info">
            <div class="user-menu-email">${escapeHtml(user.email)}</div>
            <div class="user-menu-provider">via ${escapeHtml(user.provider)}</div>
          </div>
          <button class="user-menu-btn" id="menu-reset-progress">${Icons.rotateCcw(14)} Reset my progress</button>
          <button class="user-menu-btn danger" id="menu-signout">${Icons.logOut(14)} Sign out</button>
        </div>
      </div>`;
    const badge = $('#user-badge');
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      badge.classList.toggle('open');
    });
    document.addEventListener('click', () => badge.classList.remove('open'), { once: true });
    $('#menu-signout').addEventListener('click', (e) => {
      e.stopPropagation();
      Auth.signOut();
      renderUserSlot();
      applyAuthGate();
      showToast('Signed out. See you next shift.');
    });
    $('#menu-reset-progress').addEventListener('click', (e) => {
      e.stopPropagation();
      if (!confirm('Reset your training progress? All scenarios will be re-locked.')) return;
      Auth.saveCompleted([]);
      reloadCompletedForCurrentUser();
      renderScenarioPicker();
      badge.classList.remove('open');
      showToast('Progress cleared. Back to square one.');
    });
  }

  function showToast(text, ms = 2400) {
    const t = document.createElement('div');
    t.className = 'toast';
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  async function loadTasks() {
    const d = await api.tasks();
    state.tasks = d.tasks || {};
  }

  // ---------- Scenario picker ----------
  function renderScenarioPicker() {
    const grid = $('#scenario-grid');
    if (!grid) return;
    grid.innerHTML = '';
    const order = ['oom_crash', 'db_pool_exhaustion', 'bad_deployment_cascade'];
    order.forEach(tid => {
      const t = state.tasks[tid];
      if (!t) return;
      const prereq = t.prerequisite;
      const locked = prereq && !state.completed.has(prereq);
      const done = state.completed.has(tid);
      const diffClass = { easy: 'diff-easy', medium: 'diff-medium', hard: 'diff-hard' }[t.difficulty];
      const card = el('div', { class: `scene-card ${locked ? 'is-locked' : ''} ${done ? 'is-done' : ''}` });
      card.innerHTML = `
        <div class="scene-head">
          <span class="diff-badge ${diffClass}">${escapeHtml(t.skill_tag)}</span>
          ${done ? `<span class="scene-done"><span class="inline-ic">${Icons.checkCircle2(12)}</span> Completed</span>` : ''}
          ${locked ? `<span class="scene-locked"><span class="inline-ic">${Icons.lock(12)}</span> Locked</span>` : ''}
        </div>
        <h3>${escapeHtml(prettyName(tid))}</h3>
        <p class="scene-backstory">${escapeHtml(t.backstory)}</p>
        <div class="scene-goals">
          <div class="goals-title">What you'll learn</div>
          <ul>${t.learning_goals.map(g => `<li>${escapeHtml(g)}</li>`).join('')}</ul>
        </div>
        <div class="scene-meta">
          <span class="inline-meta">${Icons.clock(12)} ~${t.est_minutes} min</span>
          <span class="inline-meta">${Icons.listChecks(12)} ${t.max_steps} step budget</span>
          ${prereq ? `<span class="inline-meta">${Icons.arrowRight(12)} Prereq: ${escapeHtml(prettyName(prereq))}</span>` : ''}
        </div>
        <button class="btn ${locked ? 'btn-outline' : 'btn-green'} scene-start" data-task="${tid}" ${locked ? 'disabled' : ''}>
          ${locked ? `<span class="inline-meta">${Icons.lock(14)} Complete prerequisite first</span>` : (done ? 'Replay incident' : 'Start incident')}
        </button>`;
      card.querySelector('.scene-start').addEventListener('click', () => {
        if (!locked) startIncident(tid);
      });
      grid.appendChild(card);
    });
  }

  function prettyName(tid) {
    return ({
      oom_crash: 'OOM Crash',
      db_pool_exhaustion: 'DB Pool Exhaustion',
      bad_deployment_cascade: 'Bad Deployment Cascade',
    }[tid]) || tid;
  }

  // ---------- Start / end an incident ----------
  async function startIncident(taskId) {
    state.currentTask = taskId;
    state.rewards = [];
    state.actionCards = [];
    state.stepCount = 0;
    state.score = 0;
    state.lastAction = null;
    state.lastObservation = null;
    state.selectedSvc = null;
    state.episodeActive = true;

    const r = await api.reset(taskId);
    const obs = r.observation || {};
    state.maxSteps = r.info?.max_steps || 15;

    // Pull an initial service list so the map reflects the incident
    const lsResp = await api.listServices();
    state.services = lsResp.observation?.services_summary || [];
    state.stepCount = 0; // the list_services was our warm-up, but we reset the display to zero

    showIncidentScreen();
    renderAlert(obs.message);
    renderServiceMap();
    renderNotebook(); // empty state
    renderActionToolbox();
    renderProgress();
    await refreshCoach();
  }

  async function endIncident() {
    state.episodeActive = false;
    const pm = await Coach.api.postmortem();
    if (!pm || pm.error) return;
    if (pm.resolved || pm.score >= 0.5) {
      state.completed.add(state.currentTask);
      Auth.saveCompleted(Array.from(state.completed));
    }
    renderPostMortem(pm);
  }

  // ---------- Action flow ----------
  async function executeAction(actionType, target, params) {
    if (!state.episodeActive) return;

    // Refresh services before the action so the map reflects reality
    const resp = await api.step(actionType, target, params || {});
    const obs = resp.observation || {};
    state.stepCount = resp.info?.step_count ?? state.stepCount + 1;
    state.score = resp.info?.final_score ?? state.score;
    state.rewards.push(obs.reward || 0);
    state.lastAction = { action_type: actionType, target_service: target, parameters: params || {} };
    state.lastObservation = obs;

    addNotebookCard(actionType, target, params, obs);
    renderProgress();

    // Refresh live services for the map (unless it's a pure declare/resolve)
    if (actionType !== 'resolve_incident') {
      const ls = await api.listServices();
      // Don't count list_services toward step budget beyond what server already did —
      // server will count it, but that's acceptable transparency.
      state.services = ls.observation?.services_summary || state.services;
      state.stepCount = (await fetch('/state').then(r => r.json())).step_count;
    }
    renderServiceMap();
    await refreshCoach();

    if (resp.done) {
      await sleep(600);
      endIncident();
    } else if (state.stepCount >= state.maxSteps) {
      await sleep(400);
      endIncident();
    }
  }

  // ---------- UI: incident screen ----------
  function showIncidentScreen() {
    $('#scenario-picker').classList.add('hidden');
    $('#incident-screen').classList.remove('hidden');
    $('#postmortem-overlay').classList.add('hidden');
    document.body.classList.add('in-incident');
  }

  function backToPicker() {
    $('#scenario-picker').classList.remove('hidden');
    $('#incident-screen').classList.add('hidden');
    $('#postmortem-overlay').classList.add('hidden');
    document.body.classList.remove('in-incident');
    state.episodeActive = false;
    renderScenarioPicker();
  }

  function renderAlert(message) {
    const alertBox = $('#alert-banner');
    if (!alertBox || !message) return;
    // Extract the first meaningful CRITICAL/WARNING line (skip the "INCIDENT ALERT" banner)
    const lines = message.split('\n');
    const headline = (lines.find(l => /CRITICAL|WARNING/.test(l) && !/^INCIDENT\s+ALERT/.test(l.trim())) || lines[2] || lines[0] || '').trim();
    const task = state.tasks[state.currentTask] || {};
    alertBox.innerHTML = `
      <div class="alert-flash"></div>
      <div class="alert-content">
        <div class="alert-tag">${Icons.bell(12)} PagerDuty alert · ${escapeHtml(task.skill_tag || '')}</div>
        <div class="alert-headline">${escapeHtml(headline)}</div>
        <div class="alert-sub">${escapeHtml(task.backstory || '')}</div>
      </div>`;
  }

  function renderProgress() {
    const ring = $('#score-ring-fill');
    const pct = Math.round((state.score || 0) * 100);
    if (ring) {
      const C = 2 * Math.PI * 26;
      ring.setAttribute('stroke-dasharray', `${C}`);
      ring.setAttribute('stroke-dashoffset', `${C - (C * pct / 100)}`);
    }
    const scoreNum = $('#score-num'); if (scoreNum) scoreNum.textContent = pct;
    const stepNum = $('#step-num'); if (stepNum) stepNum.textContent = `${state.stepCount} / ${state.maxSteps}`;

    const bar = $('#battery-fill');
    if (bar) {
      const remain = Math.max(0, state.maxSteps - state.stepCount);
      const pctLeft = Math.round(100 * remain / state.maxSteps);
      bar.style.width = pctLeft + '%';
      bar.className = 'battery-fill ' + (pctLeft < 25 ? 'bat-low' : pctLeft < 50 ? 'bat-mid' : 'bat-ok');
    }
  }

  // ---------- Service map ----------
  function renderServiceMap() {
    const container = $('#svc-map');
    if (!container) return;
    ServiceMap.render(container, state.services, {
      selected: state.selectedSvc,
      onNodeClick: (svcName) => {
        state.selectedSvc = svcName;
        renderServiceMap();
        renderServicePanel(svcName);
      },
    });
  }

  function renderServicePanel(svcName) {
    const panel = $('#svc-detail-panel');
    if (!panel) return;
    const svc = state.services.find(s => s.name === svcName);
    if (!svc) return;
    const healthClass = { healthy: 'h-ok', degraded: 'h-deg', unhealthy: 'h-bad', crashed: 'h-crash', restarting: 'h-restart' }[svc.health] || 'h-unknown';
    panel.innerHTML = `
      <div class="svc-detail-head">
        <span class="health-dot ${healthClass}"></span>
        <div>
          <div class="svc-detail-name">${escapeHtml(svcName)}</div>
          <div class="svc-detail-health">${svc.health.toUpperCase()} · v${escapeHtml(svc.version)} · ${svc.replicas} replicas</div>
        </div>
      </div>
      <div class="svc-detail-metrics">
        <div class="metric-tile"><span class="m-label">CPU</span><span class="m-val">${svc.cpu_percent}%</span></div>
        <div class="metric-tile"><span class="m-label">Memory</span><span class="m-val">${svc.memory_mb}MB</span></div>
        <div class="metric-tile"><span class="m-label">Errors</span><span class="m-val">${svc.error_rate_percent}%</span></div>
      </div>
      <div class="svc-quick-actions">
        <button class="btn btn-xs btn-outline svc-qa" data-sa="read_logs">${Icons.scrollText(14)} Read logs</button>
        <button class="btn btn-xs btn-outline svc-qa" data-sa="check_metrics">${Icons.barChart3(14)} Check metrics</button>
        <button class="btn btn-xs btn-outline svc-qa" data-sa="describe_service">${Icons.fileText(14)} Details</button>
      </div>`;
    panel.querySelectorAll('[data-sa]').forEach(b => {
      b.addEventListener('click', () => {
        const act = b.getAttribute('data-sa');
        executeAction(act, svcName, act === 'read_logs' ? { lines: 50 } : {});
      });
    });
  }

  // ---------- Action toolbox ----------
  function renderActionToolbox() {
    const box = $('#action-toolbox');
    if (!box) return;
    const groups = { investigate: [], remediate: [], declare: [] };
    Object.entries(ACTIONS).forEach(([key, meta]) => {
      groups[meta.group].push([key, meta]);
    });
    box.innerHTML = `
      <div class="tbox-group">
        <div class="tbox-title">Investigate</div>
        <div class="tbox-btns">${groups.investigate.map(([k, m]) => btnHtml(k, m)).join('')}</div>
      </div>
      <div class="tbox-group">
        <div class="tbox-title">Remediate</div>
        <div class="tbox-btns">${groups.remediate.map(([k, m]) => btnHtml(k, m)).join('')}</div>
      </div>
      <div class="tbox-group">
        <div class="tbox-title">Declare</div>
        <div class="tbox-btns">${groups.declare.map(([k, m]) => btnHtml(k, m)).join('')}</div>
      </div>`;
    box.querySelectorAll('[data-action]').forEach(b => {
      b.addEventListener('click', () => handleActionClick(b.getAttribute('data-action')));
    });
  }

  function btnHtml(key, meta) {
    const danger = meta.danger ? 'is-danger' : '';
    const iconHtml = typeof meta.icon === 'function' ? meta.icon() : (meta.icon || '');
    return `<button class="action-btn ${danger}" data-action="${key}" title="${escapeHtml(meta.desc)}">
      <span class="action-icon">${iconHtml}</span>
      <span class="action-label">${escapeHtml(meta.label)}</span>
    </button>`;
  }

  function handleActionClick(actionType, prefill = {}) {
    const meta = ACTIONS[actionType];
    if (!meta) return;
    const needsTarget = !!meta.needsTarget;
    const needsParams = !!(meta.params && meta.params.length);
    if (!needsTarget && !needsParams) {
      executeAction(actionType, null, prefill.params || {});
      return;
    }
    openActionModal(actionType, meta, prefill);
  }

  function openActionModal(actionType, meta, prefill = {}) {
    const modal = $('#action-modal');
    const body = $('#action-modal-body');
    const pretarget = prefill.target || (state.selectedSvc && meta.needsTarget ? state.selectedSvc : '');
    const pparams = prefill.params || {};
    let targetField = '';
    if (meta.needsTarget) {
      const opts = state.services.map(s => `<option value="${s.name}" ${s.name === pretarget ? 'selected' : ''}>${s.name} (${s.health})</option>`).join('');
      targetField = `<label>Target service</label><select id="modal-target">${opts}</select>`;
    }
    const isDbPool = pretarget === 'postgres-db';
    const defaultKey = pparams.key != null ? pparams.key : (isDbPool ? 'db.pool.max_size' : '');
    const defaultValue = pparams.value != null ? pparams.value : (isDbPool ? '100' : '');
    const defaultRootCause = pparams.root_cause != null ? pparams.root_cause : '';
    const defaultResolution = pparams.resolution != null ? pparams.resolution : '';
    const defaultMem = pparams.memory_limit || '512Mi';
    const defaultReplicas = pparams.replicas != null ? pparams.replicas : 3;
    const defaultVersion = pparams.to_version || 'v2.3.1';
    const defaultCmd = pparams.command || 'check_health';
    const defaultLines = pparams.lines != null ? pparams.lines : 50;
    const defaultSeverity = pparams.severity != null ? pparams.severity : '';

    const sel = (v, opt) => v === opt ? 'selected' : '';
    let paramFields = '';
    (meta.params || []).forEach(p => {
      if (p === 'memory_limit') paramFields += `<label>New memory limit</label>
        <select id="p-memory_limit">
          <option value="" ${sel(defaultMem, '')}>(keep current)</option>
          <option value="256Mi" ${sel(defaultMem, '256Mi')}>256 Mi</option>
          <option value="512Mi" ${sel(defaultMem, '512Mi')}>512 Mi</option>
          <option value="1024Mi" ${sel(defaultMem, '1024Mi')}>1024 Mi</option>
          <option value="2048Mi" ${sel(defaultMem, '2048Mi')}>2048 Mi</option>
        </select>`;
      else if (p === 'replicas') paramFields += `<label>Replicas</label><input type="number" id="p-replicas" min="0" max="20" value="${defaultReplicas}">`;
      else if (p === 'to_version') paramFields += `<label>Roll back to version</label>
        <select id="p-to_version">
          <option value="v2.3.1" ${sel(defaultVersion, 'v2.3.1')}>v2.3.1 (last stable)</option>
          <option value="v2.3.0" ${sel(defaultVersion, 'v2.3.0')}>v2.3.0</option>
          <option value="v2.2.5" ${sel(defaultVersion, 'v2.2.5')}>v2.2.5</option>
        </select>`;
      else if (p === 'command') paramFields += `<label>Diagnostic command</label>
        <select id="p-command">
          <option value="check_health" ${sel(defaultCmd, 'check_health')}>check_health</option>
          <option value="check_connectivity" ${sel(defaultCmd, 'check_connectivity')}>check_connectivity</option>
          <option value="check_resources" ${sel(defaultCmd, 'check_resources')}>check_resources</option>
          <option value="check_dns" ${sel(defaultCmd, 'check_dns')}>check_dns</option>
        </select>`;
      else if (p === 'lines') paramFields += `<label>Lines to fetch</label><input type="number" id="p-lines" min="10" max="200" value="${defaultLines}">`;
      else if (p === 'severity') paramFields += `<label>Severity filter</label>
        <select id="p-severity">
          <option value="" ${sel(defaultSeverity, '')}>(all)</option>
          <option value="ERROR" ${sel(defaultSeverity, 'ERROR')}>ERROR</option>
          <option value="WARN" ${sel(defaultSeverity, 'WARN')}>WARN</option>
          <option value="INFO" ${sel(defaultSeverity, 'INFO')}>INFO</option>
        </select>`;
      else if (p === 'key') paramFields += `<label>Config key</label><input type="text" id="p-key" value="${escapeHtml(defaultKey)}" placeholder="db.pool.max_size">`;
      else if (p === 'value') paramFields += `<label>Config value</label><input type="text" id="p-value" value="${escapeHtml(defaultValue)}" placeholder="100">`;
      else if (p === 'root_cause') paramFields += `<label>Root cause (in your words)</label><textarea id="p-root_cause" rows="2" placeholder="What actually caused this incident?">${escapeHtml(defaultRootCause)}</textarea>`;
      else if (p === 'resolution') paramFields += `<label>How you fixed it</label><textarea id="p-resolution" rows="2" placeholder="What actions resolved it?">${escapeHtml(defaultResolution)}</textarea>`;
    });

    const danger = meta.danger ? `<div class="modal-warning">⚠️ ${escapeHtml(meta.desc)}</div>` : '';

    body.innerHTML = `
      <div class="modal-head">
        <div class="modal-icon">${meta.icon}</div>
        <div>
          <h3>${escapeHtml(meta.label)}</h3>
          <div class="modal-sub">${escapeHtml(meta.desc)}</div>
        </div>
      </div>
      ${danger}
      <div class="modal-form">
        ${targetField}
        ${paramFields}
      </div>
      <div class="modal-actions">
        <button class="btn btn-outline btn-sm" id="modal-cancel">Cancel</button>
        <button class="btn btn-primary btn-sm" id="modal-go">Execute</button>
      </div>`;
    modal.classList.remove('hidden');

    $('#modal-cancel').addEventListener('click', () => modal.classList.add('hidden'));
    $('#modal-go').addEventListener('click', () => {
      const target = meta.needsTarget ? $('#modal-target').value : null;
      const params = {};
      (meta.params || []).forEach(p => {
        const e = $(`#p-${p}`);
        if (!e) return;
        let v = e.value;
        if (p === 'replicas' || p === 'lines') v = v ? parseInt(v, 10) : undefined;
        if (p === 'value') { const n = Number(v); if (!Number.isNaN(n) && v !== '') v = n; }
        if (v !== '' && v !== undefined) params[p] = v;
      });
      modal.classList.add('hidden');
      executeAction(actionType, target, params);
    });
  }

  // ---------- Notebook cards ----------
  function renderNotebook() {
    const nb = $('#notebook');
    if (!nb) return;
    if (state.actionCards.length === 0) {
      nb.innerHTML = `
        <div class="nb-empty">
          <div class="nb-empty-title">Your action log will appear here.</div>
          <div class="nb-empty-sub">Each action becomes a card. The key fact is highlighted; raw output is collapsible.</div>
        </div>`;
      return;
    }
    nb.innerHTML = '';
    state.actionCards.forEach(c => nb.appendChild(c));
    nb.scrollTop = nb.scrollHeight;
  }

  function addNotebookCard(actionType, target, params, obs) {
    const meta = ACTIONS[actionType];
    const rewardVal = obs.reward || 0;
    const rewardCls = rewardVal > 0.05 ? 'rw-good' : rewardVal < -0.02 ? 'rw-bad' : 'rw-neutral';
    const isError = !!obs.error;
    const outlineCls = isError ? 'nb-err' : rewardVal > 0.05 ? 'nb-good' : rewardVal < -0.02 ? 'nb-warn' : 'nb-neutral';

    const keyFact = extractKeyFact(obs, actionType, target);
    const hasRaw = obs.message && obs.message.length > 0;

    const nbIconHtml = meta && typeof meta.icon === 'function' ? meta.icon(16) : (meta?.icon || '•');
    const card = el('div', { class: `nb-card ${outlineCls}` });
    card.innerHTML = `
      <div class="nb-head">
        <div class="nb-head-left">
          <span class="nb-icon">${nbIconHtml}</span>
          <span class="nb-action">${escapeHtml(meta?.label || actionType)}</span>
          ${target ? `<span class="nb-target">${escapeHtml(target)}</span>` : ''}
        </div>
        <div class="nb-head-right">
          <span class="nb-reward ${rewardCls}">${rewardVal > 0 ? '+' : ''}${rewardVal.toFixed(3)}</span>
          <span class="nb-step">step ${state.stepCount}</span>
        </div>
      </div>
      <div class="nb-body">
        <div class="nb-keyfact">${keyFact}</div>
        ${hasRaw ? `<details class="nb-raw"><summary>Show raw output</summary><pre>${escapeHtml(obs.message)}</pre></details>` : ''}
        <button class="btn btn-xs btn-outline nb-why"><span style="display:inline-flex;align-items:center;gap:4px">${Icons.sparkles(13)} Why?</span></button>
      </div>`;
    card.querySelector('.nb-why').addEventListener('click', async () => {
      const bubble = $('#coach-bubble');
      const expl = await Coach.api.explain({ action_type: actionType, target_service: target }, obs.message || '');
      Coach.renderExplanation(bubble, expl);
      bubble.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
    state.actionCards.push(card);
    renderNotebook();
  }

  function extractKeyFact(obs, actionType, target) {
    const inline = (icon, cls = '', gap = 4) => `<span style="display:inline-flex;align-items:center;gap:${gap}px" class="${cls}">${icon}`;
    if (obs.error) return `${inline(Icons.xCircle(14), 'kf-bad')} ${escapeHtml(obs.error)}</span>`;
    const msg = obs.message || '';
    if (actionType === 'list_services' && obs.services_summary) {
      const bad = obs.services_summary.filter(s => s.health !== 'healthy');
      if (bad.length === 0) return `${inline(Icons.checkCircle2(14), 'kf-ok')} All 9 services healthy.</span>`;
      return `${inline(Icons.alertTriangle(14), 'kf-bad')} ${bad.length} unhealthy:</span> ${bad.map(s => `<strong>${escapeHtml(s.name)}</strong> (${s.health})`).join(', ')}`;
    }
    if (actionType === 'check_metrics' && obs.metrics) {
      const m = obs.metrics;
      const memPct = m.memory_utilization_percent;
      const memFlag = memPct > 90 ? `${inline(Icons.flame(13), 'kf-bad')} Memory ${memPct}% — OOM imminent</span>` : memPct > 70 ? `<span class="kf-warn">Memory ${memPct}%</span>` : `Memory ${memPct}%`;
      const errFlag = m.error_rate_percent > 20 ? `<span class="kf-bad">Errors ${m.error_rate_percent}%</span>` : `Errors ${m.error_rate_percent}%`;
      return `<strong>${escapeHtml(target || '')}</strong> · CPU ${m.cpu_percent}% · ${memFlag} · ${errFlag} · p99 ${m.request_latency_p99_ms}ms`;
    }
    if (actionType === 'read_logs' && obs.logs) {
      const errLines = obs.logs.filter(l => /ERROR|CRITICAL|FATAL/i.test(l));
      if (errLines.length === 0) return `${inline(Icons.checkCircle2(14), 'kf-ok')} ${obs.logs.length} lines, no errors.</span>`;
      const first = errLines[0].length > 180 ? errLines[0].slice(0, 180) + '…' : errLines[0];
      return `${inline(Icons.alertOctagon(14), 'kf-bad')} ${errLines.length} error line${errLines.length > 1 ? 's' : ''}.</span> <code>${escapeHtml(first)}</code>`;
    }
    if (actionType === 'describe_service' && obs.service_detail) {
      const d = obs.service_detail;
      const recent = (d.deployment_history || []).slice(-2).map(h => `${h.version} (${h.status})`).join(', ');
      return `<strong>${escapeHtml(d.name)}</strong> v${escapeHtml(d.version)} · ${d.replicas} replicas · Recent deploys: ${escapeHtml(recent)}`;
    }
    if (actionType === 'restart_service') return `${inline(Icons.rotateCw(14))} ${escapeHtml(msg.split('\n')[0])}</span>`;
    if (actionType === 'rollback_deployment') return `${inline(Icons.rotateCcw(14))} ${escapeHtml(msg.split('\n')[0])}</span>`;
    if (actionType === 'update_config') return `${inline(Icons.settings2(14))} ${escapeHtml(msg.split('\n')[0])}</span>`;
    if (actionType === 'resolve_incident') return `${inline(Icons.checkCircle2(14), 'kf-ok')} Incident declared resolved.</span>`;
    return escapeHtml(msg.split('\n').slice(0, 2).join(' '));
  }

  // ---------- Coach ----------
  async function refreshCoach() {
    const bubble = $('#coach-bubble');
    if (!bubble) return;
    const data = await Coach.api.hint();
    Coach.renderHint(bubble, data);
    bubble.querySelectorAll('.coach-cta').forEach(b => {
      b.addEventListener('click', () => {
        const action = b.getAttribute('data-action');
        const target = b.getAttribute('data-target') || null;
        let params = {};
        try { params = JSON.parse(b.getAttribute('data-params') || '{}'); } catch(e) {}
        handleActionClick(action, { target, params });
      });
    });
  }

  // ---------- Post-mortem ----------
  function renderPostMortem(pm) {
    const overlay = $('#postmortem-overlay');
    overlay.classList.remove('hidden');
    const grade = pm.grade_letter || '—';
    const gradeColor = { A: '#22c55e', B: '#3b82f6', C: '#f59e0b', D: '#ef4444', F: '#ef4444' }[grade] || '#94a3b8';
    const critIcon = (ok) => ok ? Icons.checkCircle2(16) : Icons.xCircle(16);
    const actIcon = (name) => {
      const m = ACTIONS[name];
      return m && typeof m.icon === 'function' ? m.icon(14) : (m?.icon || '');
    };
    const criteriaHtml = (pm.criteria || []).map(c => `
      <div class="pm-crit ${c.passed ? 'passed' : 'missed'}">
        <span class="pm-crit-icon">${critIcon(c.passed)}</span>
        <span class="pm-crit-text">${escapeHtml(c.criterion)}</span>
        <span class="pm-crit-w">+${c.weight.toFixed(2)}</span>
      </div>`).join('');

    const userTrajHtml = (pm.user_trajectory || []).map(a => `
      <div class="pm-traj-row">
        <span class="pm-traj-step">${a.step}</span>
        <span class="pm-traj-act"><span class="inline-ic">${actIcon(a.action)}</span> ${escapeHtml(a.action)}</span>
        <span class="pm-traj-tgt">${a.target ? escapeHtml(a.target) : ''}</span>
      </div>`).join('') || '<div class="pm-traj-empty">No actions recorded.</div>';

    const idealHtml = (pm.ideal_trajectory || []).map((s, i) => `
      <div class="pm-traj-row ideal">
        <span class="pm-traj-step">${i + 1}</span>
        <span class="pm-traj-act"><span class="inline-ic">${actIcon(s.action)}</span> ${escapeHtml(s.action)}</span>
        <span class="pm-traj-tgt">${s.target ? escapeHtml(s.target) : ''}</span>
        <div class="pm-traj-why"><span class="inline-ic" style="opacity:.65">${Icons.messageCircle(12)}</span> ${escapeHtml(s.why || '')}</div>
      </div>`).join('');

    const studyLink = pm.study_link
      ? `<div class="pm-study">
          <div class="pm-study-title"><span class="inline-ic">${Icons.bookOpen(14)}</span> One thing to study next</div>
          <div class="pm-study-body">${escapeHtml(pm.study_link.topic)} — you missed ${pm.study_link.weight_missed.toFixed(2)} points on this criterion. Open the relevant log again and see if you can spot what the rubric wanted.</div>
        </div>`
      : '';

    overlay.innerHTML = `
      <div class="pm-card">
        <div class="pm-close" id="pm-close">${Icons.x(14)}</div>
        <div class="pm-top">
          <div class="pm-grade" style="color:${gradeColor}">${grade}</div>
          <div>
            <div class="pm-title">Incident ${pm.resolved ? 'Resolved' : 'Timed Out'}</div>
            <div class="pm-score">Score: <strong>${pm.score.toFixed(2)}</strong> / 1.00 · Steps: ${pm.steps_used}/${pm.max_steps}</div>
          </div>
        </div>
        <div class="pm-section">
          <div class="pm-section-title">Rubric breakdown</div>
          <div class="pm-crits">${criteriaHtml}</div>
          ${pm.penalties < 0 ? `<div class="pm-penalty">Penalties applied: <strong>${pm.penalties.toFixed(2)}</strong></div>` : ''}
        </div>
        <div class="pm-two-col">
          <div class="pm-col">
            <div class="pm-section-title">Your trajectory</div>
            <div class="pm-traj">${userTrajHtml}</div>
          </div>
          <div class="pm-col">
            <div class="pm-section-title">What a senior SRE would have done</div>
            <div class="pm-traj">${idealHtml}</div>
          </div>
        </div>
        ${studyLink}
        <div class="pm-footer">
          <button class="btn btn-outline btn-sm" id="pm-replay"><span class="inline-ic">${Icons.rotateCcw(14)}</span> Retry this incident</button>
          <button class="btn btn-primary btn-sm" id="pm-back"><span class="inline-ic">${Icons.chevronLeft(14)}</span> Back to scenarios</button>
        </div>
      </div>`;

    $('#pm-close').addEventListener('click', () => overlay.classList.add('hidden'));
    $('#pm-replay').addEventListener('click', () => { overlay.classList.add('hidden'); startIncident(state.currentTask); });
    $('#pm-back').addEventListener('click', () => backToPicker());
  }

  // ---------- Tutorial overlay ----------
  function showTutorial() {
    const overlay = $('#tutorial-overlay');
    if (!overlay) return;
    overlay.classList.remove('hidden');
    overlay.innerHTML = `
      <div class="tut-card">
        <div class="tut-header"><span class="tut-brand-icon">${Icons.siren(22)}</span><h2>Welcome, on-call SRE</h2></div>
        <p class="tut-lead">This is a <strong>training ground for incident response</strong>. You'll be paged, investigate a real-looking outage, and fix it — with an AI coach helping along the way.</p>
        <div class="tut-steps">
          <div class="tut-step"><span class="tut-num">1</span><div><strong>Pick a scenario</strong> — three difficulty levels. Start with "Your first page."</div></div>
          <div class="tut-step"><span class="tut-num">2</span><div><strong>Watch the service map</strong> — red = broken. Click any service to see details.</div></div>
          <div class="tut-step"><span class="tut-num">3</span><div><strong>Use the toolbox</strong> — buttons are grouped by intent: Investigate → Remediate → Declare.</div></div>
          <div class="tut-step"><span class="tut-num">4</span><div><strong>Your AI coach</strong> is in the right panel. Stuck? It'll nudge you. Click "Why?" on any result for a plain-English explanation.</div></div>
          <div class="tut-step"><span class="tut-num">5</span><div><strong>Switch to Pro mode</strong> (top right) to turn off hints when you're ready.</div></div>
        </div>
        <button class="btn btn-primary" id="tut-go">Got it — let's go</button>
      </div>`;
    $('#tut-go').addEventListener('click', () => {
      overlay.classList.add('hidden');
      localStorage.setItem('ic_tutorial_done', '1');
    });
  }

  // ---------- Global bindings ----------
  function bindGlobal() {
    document.addEventListener('click', (e) => {
      if (e.target.matches('#mode-toggle') || e.target.closest('#mode-toggle')) {
        state.mode = state.mode === 'junior' ? 'pro' : 'junior';
        localStorage.setItem('ic_mode', state.mode);
        document.body.setAttribute('data-mode', state.mode);
        const lbl = $('#mode-label'); if (lbl) lbl.textContent = state.mode === 'junior' ? 'Junior' : 'Pro';
      }
    });
    const backBtn = $('#back-to-picker');
    if (backBtn) backBtn.addEventListener('click', backToPicker);
    const restartBtn = $('#tutorial-restart');
    if (restartBtn) restartBtn.addEventListener('click', () => { localStorage.removeItem('ic_tutorial_done'); showTutorial(); });
    const resetBtn = $('#reset-progress');
    if (resetBtn) resetBtn.addEventListener('click', () => {
      if (!confirm('Reset your training progress? All scenarios will be re-locked.')) return;
      Auth.saveCompleted([]);
      state.completed = new Set();
      renderScenarioPicker();
    });

    // Auth form handlers
    const emailForm = $('#auth-email-form');
    if (emailForm) {
      emailForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const input = $('#auth-email-input');
        const errBox = $('#auth-error');
        errBox.classList.add('hidden');
        try {
          Auth.signInWithEmail(input.value);
          renderUserSlot();
          applyAuthGate();
          showToast(`Welcome, ${Auth.currentUser().name}. Let's go.`);
          if (!localStorage.getItem('ic_tutorial_done')) showTutorial();
        } catch (err) {
          errBox.textContent = err.message;
          errBox.classList.remove('hidden');
        }
      });
    }
    // Delegated handler — auth social buttons are injected after init
    document.addEventListener('click', (ev) => {
      const btn = ev.target.closest('.auth-btn[data-provider]');
      if (!btn) return;
      const provider = btn.getAttribute('data-provider');
      if (!provider) return;
      Auth.signInWithProvider(provider);
      renderUserSlot();
      applyAuthGate();
      const u = Auth.currentUser();
      showToast(`Signed in as ${u.name} (${u.provider} demo).`);
      if (!localStorage.getItem('ic_tutorial_done')) showTutorial();
    });
  }

  // Expose for debug
  window.__IC__ = { state, api, executeAction };

  document.addEventListener('DOMContentLoaded', init);
})();
