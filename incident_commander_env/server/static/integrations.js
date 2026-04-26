/* Integrations panel — connect a real platform to Praetor.
 *
 * Three tabs:
 *   GitHub  — REAL OAuth via device flow. Demo fallback if GITHUB_CLIENT_ID
 *             is not set on the server.
 *   Cloud   — DEMO MODE. Forms render, credentials live in localStorage,
 *             never transmitted. Honest about its limits.
 *   Adapter — REAL code generation. Produces a praetor_adapter.py the user
 *             drops into their own deployment.
 *
 * Talks to the /integrations/* router defined in incident_commander_env/server/integrations.py.
 */

(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    githubConnected: false,
    githubUsername: null,
    githubDemo: false,
    githubSelectedRepo: null,
    githubPollTimer: null,
    cloudCreds: {}, // localStorage-backed; keyed by provider id
    adapterPreviewLoaded: false,
  };

  // ---- Generic helpers ------------------------------------------------------

  async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    try {
      const resp = await fetch(path, opts);
      const text = await resp.text();
      try { return JSON.parse(text); } catch { return { _raw: text, _status: resp.status }; }
    } catch (err) {
      return { error: String(err) };
    }
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  // ---- Tab switcher --------------------------------------------------------

  function wireTabs() {
    const tabs = document.querySelectorAll('.int-tab');
    tabs.forEach((t) => {
      t.addEventListener('click', () => {
        tabs.forEach((x) => x.classList.remove('active'));
        t.classList.add('active');
        const which = t.getAttribute('data-tab');
        document.querySelectorAll('.int-pane').forEach((p) => (p.style.display = 'none'));
        const pane = $(`int-pane-${which}`);
        if (pane) pane.style.display = 'block';
        if (which === 'cloud') renderCloudProviders();
      });
    });
  }

  // ---- GitHub flow ---------------------------------------------------------

  function setGhMode(mode) {
    // mode: disconnected | pending | connected
    ['disconnected', 'pending', 'connected'].forEach((m) => {
      const el = $(`int-gh-${m}`);
      if (el) el.style.display = m === mode ? 'block' : 'none';
    });
  }

  async function refreshGhStatus() {
    const s = await api('GET', '/integrations/github/status');
    if (s && s.connected) {
      state.githubConnected = true;
      state.githubUsername = s.username;
      state.githubDemo = !!s.demo_mode;
      state.githubSelectedRepo = s.selected_repo;
      $('int-gh-username').textContent = s.username || '(unknown)';
      $('int-gh-mode-tag').textContent = s.demo_mode ? 'DEMO mode (no Client ID configured)' : 'Real GitHub OAuth';
      if (s.avatar_url) $('int-gh-avatar').src = s.avatar_url;
      setGhMode('connected');
      await loadGhRepos();
      if (s.selected_repo) renderSelectedRepo(s.selected_repo);
    } else {
      const hint = $('int-gh-mode-hint');
      if (hint) {
        hint.innerHTML = s && s.client_id_configured
          ? 'Real OAuth ready (GITHUB_CLIENT_ID is set).'
          : 'No GITHUB_CLIENT_ID set — will run in DEMO mode. To enable real OAuth: register an OAuth App at <a href="https://github.com/settings/developers" target="_blank" rel="noopener" style="color:var(--accent-blue)">github.com/settings/developers</a>, set GITHUB_CLIENT_ID in .env, restart.';
      }
      setGhMode('disconnected');
    }
  }

  async function startGhConnect() {
    const btn = $('int-gh-connect');
    if (btn) { btn.disabled = true; btn.textContent = 'Starting…'; }
    const data = await api('POST', '/integrations/github/start');
    if (btn) { btn.disabled = false; btn.innerHTML = '<span id="int-gh-connect-icon">⚡</span> Connect GitHub'; }
    if (data.error) {
      alert('GitHub start failed: ' + data.error);
      return;
    }

    const link = $('int-gh-verif-link');
    const codeEl = $('int-gh-user-code');
    if (link) {
      link.href = data.verification_uri;
      link.textContent = data.verification_uri;
    }
    if (codeEl) {
      codeEl.textContent = data.user_code;
      // Click-to-copy convenience
      codeEl.onclick = () => {
        try {
          navigator.clipboard.writeText(data.user_code);
          const original = codeEl.textContent;
          codeEl.textContent = 'copied!';
          setTimeout(() => { codeEl.textContent = original; }, 900);
        } catch (e) { /* fall back to user-select */ }
      };
    }
    setGhMode('pending');

    // Poll for token. Server enforces GitHub's interval; the client just
    // polls every 2s and the server replies "throttled" when appropriate.
    if (state.githubPollTimer) clearInterval(state.githubPollTimer);
    const throttleNote = $('int-gh-throttle-note');
    const poll = async () => {
      const s = await api('GET', '/integrations/github/poll');
      if (s.status === 'authorized') {
        clearInterval(state.githubPollTimer);
        state.githubPollTimer = null;
        if (throttleNote) throttleNote.textContent = '';
        await refreshGhStatus();
      } else if (s.status === 'expired' || s.status === 'error') {
        clearInterval(state.githubPollTimer);
        state.githubPollTimer = null;
        if (throttleNote) throttleNote.textContent = '';
        alert('GitHub auth: ' + (s.error || s.status));
        setGhMode('disconnected');
      } else if (s.throttled) {
        if (throttleNote) {
          throttleNote.textContent = `(retrying in ${s.retry_in_seconds || 5}s)`;
        }
      } else if (s.slow_down && throttleNote) {
        throttleNote.textContent = `(GitHub asked us to slow down; new interval ${s.new_interval || 10}s)`;
      } else if (throttleNote) {
        throttleNote.textContent = '';
      }
    };
    // First poll after 1.5s, then every 2s — server gates real GitHub calls
    setTimeout(poll, 1500);
    state.githubPollTimer = setInterval(poll, 2000);
  }

  function cancelGhConnect() {
    if (state.githubPollTimer) {
      clearInterval(state.githubPollTimer);
      state.githubPollTimer = null;
    }
    api('POST', '/integrations/github/disconnect');
    setGhMode('disconnected');
  }

  async function disconnectGh() {
    await api('POST', '/integrations/github/disconnect');
    state.githubConnected = false;
    state.githubUsername = null;
    state.githubSelectedRepo = null;
    setGhMode('disconnected');
    await refreshGhStatus();
  }

  async function loadGhRepos() {
    const slot = $('int-gh-repos');
    if (!slot) return;
    slot.innerHTML = '<div style="font-size:12px;color:var(--text-muted);padding:8px">Loading repos…</div>';
    const data = await api('GET', '/integrations/github/repos');
    if (data.error) {
      slot.innerHTML = '<div style="font-size:12px;color:var(--accent-red);padding:8px">' + escapeHtml(data.error) + '</div>';
      return;
    }
    const repos = data.repos || [];
    if (!repos.length) {
      slot.innerHTML = '<div style="font-size:12px;color:var(--text-muted);padding:8px">No repos found.</div>';
      return;
    }
    slot.innerHTML = '';
    repos.slice(0, 30).forEach((r) => {
      const row = document.createElement('div');
      row.className = 'int-repo-row';
      if (state.githubSelectedRepo === r.full_name) row.classList.add('selected');
      row.innerHTML = `
        <div style="flex:1;min-width:0">
          <div class="name" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${escapeHtml(r.full_name)}</div>
          <div class="meta" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${escapeHtml(r.language || '—')} · ${r.private ? '🔒 private' : 'public'} · ⭐${r.stargazers_count || 0}</div>
        </div>
        <button class="btn btn-outline btn-xs">Pick</button>
      `;
      row.querySelector('button').addEventListener('click', async (ev) => {
        ev.stopPropagation();
        await api('POST', '/integrations/github/select-repo', { full_name: r.full_name });
        state.githubSelectedRepo = r.full_name;
        // Re-highlight rows
        slot.querySelectorAll('.int-repo-row').forEach((x) => x.classList.remove('selected'));
        row.classList.add('selected');
        renderSelectedRepo(r.full_name);
      });
      slot.appendChild(row);
    });
  }

  function renderSelectedRepo(fullName) {
    const el = $('int-gh-selected');
    if (!el) return;
    el.style.display = 'block';
    el.innerHTML =
      '✓ <strong>' + escapeHtml(fullName) + '</strong> picked for tier-2 code escalation. ' +
      'Praetor will clone this repo when runtime ops actions don\'t fully heal a fault.';
  }

  // ---- Cloud providers (DEMO ONLY) ----------------------------------------

  async function renderCloudProviders() {
    const slot = $('int-cloud-providers');
    if (!slot || slot.dataset.rendered === '1') return;
    const data = await api('GET', '/integrations/cloud/providers');
    if (data.error) {
      slot.innerHTML = '<div style="color:var(--accent-red)">' + escapeHtml(data.error) + '</div>';
      return;
    }
    slot.innerHTML = '';
    (data.providers || []).forEach((p) => {
      const card = document.createElement('div');
      card.className = 'cloud-provider';
      const fields = (p.fields || []).map((f) => {
        const isSecret = /key|secret|token|password/i.test(f);
        return `<input type="${isSecret ? 'password' : 'text'}" data-field="${escapeHtml(f)}" placeholder="${escapeHtml(f)}" autocomplete="off">`;
      }).join('');
      card.innerHTML = `
        <div class="cp-h">
          <h5>${escapeHtml(p.label)}</h5>
          <div class="cp-state" data-state>not connected</div>
        </div>
        ${fields}
        <div class="cp-actions">
          <button class="btn btn-primary btn-xs" data-save>Save (local only)</button>
          <button class="btn btn-outline btn-xs" data-clear>Clear</button>
        </div>
      `;
      slot.appendChild(card);

      // Restore from localStorage
      const stored = readCloudCreds(p.id);
      if (stored) {
        Object.entries(stored).forEach(([k, v]) => {
          const inp = card.querySelector(`input[data-field="${k}"]`);
          if (inp) inp.value = v;
        });
        markConnected(card, true);
      }

      card.querySelector('[data-save]').addEventListener('click', () => {
        const creds = {};
        card.querySelectorAll('input[data-field]').forEach((inp) => {
          if (inp.value) creds[inp.dataset.field] = inp.value;
        });
        if (Object.keys(creds).length === 0) {
          alert('Enter at least one field first.');
          return;
        }
        writeCloudCreds(p.id, creds);
        markConnected(card, true);
      });
      card.querySelector('[data-clear]').addEventListener('click', () => {
        clearCloudCreds(p.id);
        card.querySelectorAll('input[data-field]').forEach((inp) => (inp.value = ''));
        markConnected(card, false);
      });
    });
    slot.dataset.rendered = '1';
  }

  function markConnected(card, on) {
    const tag = card.querySelector('[data-state]');
    if (!tag) return;
    if (on) {
      tag.classList.add('connected');
      tag.textContent = 'saved (local)';
    } else {
      tag.classList.remove('connected');
      tag.textContent = 'not connected';
    }
  }

  function readCloudCreds(provider) {
    try {
      const raw = localStorage.getItem('praetor_cloud_' + provider);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }
  function writeCloudCreds(provider, creds) {
    localStorage.setItem('praetor_cloud_' + provider, JSON.stringify(creds));
    state.cloudCreds[provider] = creds;
  }
  function clearCloudCreds(provider) {
    localStorage.removeItem('praetor_cloud_' + provider);
    delete state.cloudCreds[provider];
  }

  // ---- Adapter generator --------------------------------------------------

  function adapterRequest() {
    return {
      project_name: ($('int-adapter-name').value || 'praetor-target').trim(),
      services: ($('int-adapter-services').value || 'frontend, api, postgres')
        .split(',').map((s) => s.trim()).filter(Boolean),
      platform: $('int-adapter-platform').value || 'fastapi',
    };
  }

  async function previewAdapter() {
    const out = $('int-adapter-out');
    if (out) {
      out.style.display = 'block';
      out.textContent = 'Generating…';
    }
    const data = await api('POST', '/integrations/adapter/preview', adapterRequest());
    if (data.error) {
      if (out) out.textContent = 'Error: ' + data.error;
      return;
    }
    const combined =
      '# === DEPLOYMENT NOTES ===\n' +
      (data.notes || '') + '\n\n' +
      '# === praetor_adapter.py ===\n\n' +
      (data.adapter_py || '');
    if (out) {
      out.textContent = combined;
      state.adapterPreviewLoaded = true;
    }
  }

  function downloadAdapter() {
    const req = adapterRequest();
    fetch('/integrations/adapter/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }).then((resp) => resp.blob()).then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'praetor_adapter.py';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }).catch((err) => {
      alert('Download failed: ' + err);
    });
  }

  // ---- Init ---------------------------------------------------------------

  function init() {
    if (window.Integrations && window.Integrations._inited) return;
    wireTabs();

    const ghBtn = $('int-gh-connect');
    if (ghBtn) ghBtn.addEventListener('click', startGhConnect);
    const ghCancel = $('int-gh-cancel');
    if (ghCancel) ghCancel.addEventListener('click', cancelGhConnect);
    const ghDisc = $('int-gh-disconnect');
    if (ghDisc) ghDisc.addEventListener('click', disconnectGh);

    const previewBtn = $('int-adapter-preview');
    if (previewBtn) previewBtn.addEventListener('click', previewAdapter);
    const downloadBtn = $('int-adapter-download');
    if (downloadBtn) downloadBtn.addEventListener('click', downloadAdapter);

    refreshGhStatus().catch(() => {});
    if (window.Integrations) window.Integrations._inited = true;
  }

  window.Integrations = { init, _inited: false };

  document.addEventListener('DOMContentLoaded', () => {
    // Defer init until the user opens the Real-Time tab. We hook into the
    // existing Realtime.init pattern by also exposing a manual call.
    if (document.getElementById('integrations-card')) init();
  });
})();
