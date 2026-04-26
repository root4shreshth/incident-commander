// map.js - SVG service dependency map with health-coded nodes and animated traffic lines.

const ServiceMap = (() => {
  // Tier layout (top → bottom follows request flow).
  // Tier 0: frontend-bff (entry)
  // Tier 1: api-gateway
  // Tier 2: order, user, notification, inventory (parallel), frontend-facing services
  // Tier 3: payment, auth
  // Tier 4: postgres-db (leaf)
  const LAYOUT = {
    'frontend-bff':         { x: 440, y: 50,  tier: 0 },
    'api-gateway':          { x: 440, y: 140, tier: 1 },
    'order-service':        { x: 130, y: 250, tier: 2 },
    'user-service':         { x: 300, y: 250, tier: 2 },
    'notification-service': { x: 460, y: 250, tier: 2 },
    'inventory-service':    { x: 620, y: 250, tier: 2 },
    'payment-service':      { x: 200, y: 370, tier: 3 },
    'auth-service':         { x: 400, y: 370, tier: 3 },
    'postgres-db':          { x: 440, y: 480, tier: 4 },
  };

  const EDGES = [
    ['frontend-bff', 'api-gateway'],
    ['api-gateway', 'order-service'],
    ['api-gateway', 'user-service'],
    ['api-gateway', 'notification-service'],
    ['api-gateway', 'inventory-service'],
    ['order-service', 'payment-service'],
    ['user-service', 'auth-service'],
    ['order-service', 'postgres-db'],
    ['payment-service', 'postgres-db'],
    ['inventory-service', 'postgres-db'],
    ['user-service', 'postgres-db'],
  ];

  const HEALTH_STYLE = {
    healthy:    { fill: '#0f2620', stroke: '#22c55e', glow: 'rgba(34,197,94,.5)', label: '#86efac' },
    degraded:   { fill: '#2b1f0a', stroke: '#f59e0b', glow: 'rgba(245,158,11,.6)', label: '#fcd34d' },
    unhealthy:  { fill: '#2b120a', stroke: '#ef4444', glow: 'rgba(239,68,68,.7)', label: '#fca5a5' },
    crashed:    { fill: '#3b0a0a', stroke: '#ef4444', glow: 'rgba(239,68,68,.9)', label: '#fca5a5' },
    restarting: { fill: '#0a1e3b', stroke: '#3b82f6', glow: 'rgba(59,130,246,.6)', label: '#93c5fd' },
    unknown:    { fill: '#14182a', stroke: '#475569', glow: 'rgba(71,85,105,.3)', label: '#94a3b8' },
  };

  function render(container, services, opts = {}) {
    const onNodeClick = opts.onNodeClick || (() => {});
    const selected = opts.selected || null;
    const highlighted = opts.highlighted || null;

    const W = 760, H = 540;

    const svg = `<svg viewBox="0 0 ${W} ${H}" class="svc-map-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="rgba(148,163,184,0.5)"/>
        </marker>
        <marker id="arrow-bad" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#ef4444"/>
        </marker>
      </defs>
      ${renderEdges(services)}
      ${renderNodes(services, selected, highlighted)}
    </svg>`;

    container.innerHTML = svg;

    container.querySelectorAll('[data-svc]').forEach(el => {
      el.addEventListener('click', () => onNodeClick(el.getAttribute('data-svc')));
    });
  }

  function renderEdges(services) {
    const health = {};
    (services || []).forEach(s => { health[s.name] = s.health; });

    return EDGES.map(([from, to]) => {
      const a = LAYOUT[from], b = LAYOUT[to];
      if (!a || !b) return '';
      const fromBad = ['unhealthy', 'crashed', 'degraded'].includes(health[from]);
      const toBad = ['unhealthy', 'crashed', 'degraded'].includes(health[to]);
      const bad = fromBad || toBad;
      const cls = bad ? 'edge edge-bad' : 'edge edge-ok';
      const marker = bad ? 'arrow-bad' : 'arrow';
      return `<line x1="${a.x}" y1="${a.y + 30}" x2="${b.x}" y2="${b.y - 30}" class="${cls}" marker-end="url(#${marker})"/>`;
    }).join('');
  }

  function renderNodes(services, selected, highlighted) {
    const healthMap = {};
    (services || []).forEach(s => { healthMap[s.name] = s; });

    return Object.keys(LAYOUT).map(name => {
      const { x, y } = LAYOUT[name];
      const svc = healthMap[name] || { name, health: 'unknown' };
      const style = HEALTH_STYLE[svc.health] || HEALTH_STYLE.unknown;
      const isSel = selected === name;
      const isHl = highlighted === name;
      const ring = isSel ? 4 : isHl ? 3 : 2;
      const radius = 30;
      const shortName = name.replace(/-service$/, '').replace(/-bff$/, '-BFF').replace(/postgres-db/, 'postgres');
      const glowFilter = svc.health === 'crashed' || svc.health === 'unhealthy' ? 'url(#glow-red)' : '';
      const pulseClass = svc.health === 'crashed' ? 'node-pulse-crash' : svc.health === 'unhealthy' ? 'node-pulse-bad' : '';
      // Outer <g> carries the translate; inner <g> carries the CSS pulse scale animation
      // so the two transforms don't collide.
      return `<g class="svc-node" data-svc="${name}" transform="translate(${x},${y})" style="cursor:pointer">
        <g class="svc-node-inner ${pulseClass}">
          <circle r="${radius + 6}" fill="${style.glow}" opacity="${svc.health === 'healthy' ? 0.15 : 0.35}"/>
          <circle r="${radius}" fill="${style.fill}" stroke="${style.stroke}" stroke-width="${ring}" filter="${glowFilter}"/>
          <text text-anchor="middle" y="3" fill="${style.label}" font-size="9.5" font-weight="700" font-family="Inter,sans-serif">${escapeXml(shortName)}</text>
          ${svc.cpu_percent != null ? `<text text-anchor="middle" y="16" fill="${style.label}" font-size="8" opacity="0.7" font-family="JetBrains Mono,monospace">${Math.round(svc.cpu_percent)}%</text>` : ''}
        </g>
      </g>`;
    }).join('');
  }

  function escapeXml(s) {
    return String(s).replace(/[<>&'"]/g, c => ({ '<':'&lt;','>':'&gt;','&':'&amp;',"'":'&apos;','"':'&quot;' }[c]));
  }

  return { render, LAYOUT, EDGES };
})();

window.ServiceMap = ServiceMap;
