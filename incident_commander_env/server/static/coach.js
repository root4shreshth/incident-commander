// coach.js — client-side wrapper around /coach/hint, /coach/explain and ideal trajectory.
// Produces friendly human messages and renders them into the AI Coach panel.

const Coach = (() => {
  const api = {
    async hint() {
      const r = await fetch('/coach/hint');
      return r.json();
    },
    async explain(lastAction, lastMessage) {
      const r = await fetch('/coach/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ last_action: lastAction, last_message: lastMessage }),
      });
      return r.json();
    },
    async ideal(taskId) {
      const r = await fetch(`/ideal-trajectory/${taskId}`);
      return r.json();
    },
    async postmortem() {
      const r = await fetch('/postmortem');
      return r.json();
    },
  };

  const toneToIcon = (tone) => {
    if (!window.Icons) return '';
    switch (tone) {
      case 'celebrate': return Icons.award(16);
      case 'encourage': return Icons.lightbulb(16);
      case 'warn': return Icons.alertTriangle(16);
      case 'neutral':
      default: return Icons.sparkles(16);
    }
  };

  function renderHint(bubble, data) {
    if (!data) return;
    const iconSvg = toneToIcon(data.tone);
    const paramsJson = data.suggested_action && data.suggested_action.params
      ? escapeHtml(JSON.stringify(data.suggested_action.params)) : '';
    const html = `
      <div class="coach-bubble coach-${data.tone || 'neutral'}">
        <div class="coach-head"><span class="coach-avatar">${iconSvg}</span><span class="coach-name">AI Coach</span></div>
        <div class="coach-body">${formatMarkdown(data.hint || '')}</div>
        ${data.suggested_action ? `<button class="coach-cta btn btn-xs btn-outline" data-action="${data.suggested_action.action}" data-target="${data.suggested_action.target || ''}" data-params="${paramsJson}">Try this action</button>` : ''}
      </div>`;
    bubble.innerHTML = html;
  }

  function renderExplanation(bubble, data) {
    if (!data) return;
    const iconSvg = window.Icons ? Icons.bookOpen(16) : '';
    const html = `
      <div class="coach-bubble coach-explain">
        <div class="coach-head"><span class="coach-avatar">${iconSvg}</span><span class="coach-name">Plain-English Explainer</span></div>
        <div class="coach-body">${formatMarkdown(data.explanation || '')}</div>
        ${(data.matched_terms || []).length ? `<div class="coach-terms">Covered: ${data.matched_terms.map(t => `<span class="term-chip">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
      </div>`;
    bubble.innerHTML = html;
  }

  function formatMarkdown(text) {
    // Very small markdown subset: **bold** and newlines.
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  return { api, renderHint, renderExplanation, formatMarkdown, escapeHtml };
})();

window.Coach = Coach;
