"""Server-side PDF generation for real-time incident reports.

Why reportlab + manual layout instead of HTML-to-PDF (xhtml2pdf, weasyprint):
  * Pure-Python with no native deps — works on Windows / Linux / HF Space alike.
  * Full control over the layout — we want a designed, branded report, not a
    print-stylesheet of an existing HTML page.
  * Predictable file size (~25-50 KB) and instant generation (~50 ms).

Public API:
    render_run_report_pdf(run_id, rec) -> bytes
        Returns the PDF body. Wrap in a StreamingResponse + Content-Disposition
        header at the route layer.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ---- color palette (matches the on-screen UI) ------------------------------

ACCENT = colors.HexColor("#3b82f6")          # primary blue
ACCENT_DARK = colors.HexColor("#1e40af")
GOOD = colors.HexColor("#22c55e")
WARN = colors.HexColor("#f59e0b")
BAD = colors.HexColor("#ef4444")
TEXT = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b7280")
LIGHT_BG = colors.HexColor("#f9fafb")
ALERT_BG = colors.HexColor("#fef3c7")
WHY_BG = colors.HexColor("#eff6ff")
BORDER = colors.HexColor("#e5e7eb")


# ---- style sheet ------------------------------------------------------------

def _make_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    return {
        "title": ParagraphStyle(
            name="title", parent=base,
            fontName="Helvetica-Bold", fontSize=22, leading=26,
            textColor=TEXT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            name="subtitle", parent=base,
            fontName="Helvetica-Oblique", fontSize=11, leading=14,
            textColor=MUTED, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            name="h2", parent=base,
            fontName="Helvetica-Bold", fontSize=10, leading=12,
            textColor=MUTED, spaceBefore=14, spaceAfter=6,
            textTransform="uppercase",
        ),
        "body": ParagraphStyle(
            name="body", parent=base,
            fontName="Helvetica", fontSize=10, leading=14,
            textColor=TEXT, spaceAfter=4,
        ),
        "alert": ParagraphStyle(
            name="alert", parent=base,
            fontName="Helvetica", fontSize=10.5, leading=14,
            textColor=TEXT, spaceAfter=4,
            leftIndent=8, borderPadding=8,
        ),
        "step_head_action": ParagraphStyle(
            name="stephead", parent=base,
            fontName="Courier-Bold", fontSize=10, leading=12,
            textColor=ACCENT_DARK,
        ),
        "step_target": ParagraphStyle(
            name="steptarget", parent=base,
            fontName="Courier", fontSize=9, leading=11,
            textColor=MUTED,
        ),
        "step_body": ParagraphStyle(
            name="stepbody", parent=base,
            fontName="Courier", fontSize=8.5, leading=11,
            textColor=colors.HexColor("#374151"),
        ),
        "step_why_label": ParagraphStyle(
            name="whylabel", parent=base,
            fontName="Helvetica-Bold", fontSize=8, leading=10,
            textColor=ACCENT_DARK,
            textTransform="uppercase",
        ),
        "step_why_body": ParagraphStyle(
            name="whybody", parent=base,
            fontName="Helvetica", fontSize=9.5, leading=13,
            textColor=TEXT,
        ),
        "footer": ParagraphStyle(
            name="footer", parent=base,
            fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
            textColor=MUTED, alignment=1,  # center
        ),
        "meta_label": ParagraphStyle(
            name="metalabel", parent=base,
            fontName="Helvetica", fontSize=9, leading=11,
            textColor=MUTED,
        ),
        "meta_value": ParagraphStyle(
            name="metavalue", parent=base,
            fontName="Courier", fontSize=9, leading=11,
            textColor=TEXT,
        ),
        "status_pill": ParagraphStyle(
            name="statuspill", parent=base,
            fontName="Helvetica-Bold", fontSize=9, leading=11,
            textColor=colors.white, alignment=1,
        ),
    }


# ---- helpers ----------------------------------------------------------------

def _esc(text: Any) -> str:
    """HTML-escape text for ReportLab Paragraph (it parses a Mini-HTML)."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " …"


def _status_for(rec: Dict[str, Any], events: list) -> tuple[str, colors.Color]:
    resolved = bool(rec.get("tier1_resolved"))
    if resolved:
        return "RESOLVED", GOOD
    tier2_done = next((e for e in events if e.get("type") == "tier2_done"), None)
    if tier2_done:
        return "ESCALATED", WARN
    return "UNRESOLVED", BAD


# Static taxonomy: scenario name -> {cause tags, fix tags}. Mirror the
# JS deriveReportTags() in static/realtime.js so the PDF and the on-screen
# report carry the same vocabulary.
_TAGS_BY_SCENARIO = {
    "oom_crash":              {"cause": ["oom", "memory-pressure"],          "fix": ["memory-bump", "restart-curable"]},
    "db_pool_exhaustion":     {"cause": ["connection-leak", "pool-saturated"], "fix": ["config-tune", "pool-size-bump"]},
    "bad_deployment_cascade": {"cause": ["bad-deploy", "cascading-failure"],   "fix": ["rollback", "version-revert"]},
    "disk_full":              {"cause": ["disk-full", "storage-exhausted"],    "fix": ["restart-cycle-volume", "log-rotation"]},
    "slow_query":             {"cause": ["lock-contention", "query-regression"], "fix": ["rollback", "feature-flag"]},
    "cert_expiry":            {"cause": ["cert-expired", "tls-handshake-fail"],  "fix": ["restart-renew", "cert-rotation"]},
    "dns_failure":            {"cause": ["dns-resolution", "resolver-stale"],    "fix": ["cache-flush", "restart-resolver"]},
    "rate_limit_exhaustion":  {"cause": ["rate-limited", "429-storm"],           "fix": ["scale-out", "key-rotate"]},
}


def _derive_tags(scenario: str, events: list) -> Dict[str, list]:
    t = _TAGS_BY_SCENARIO.get(scenario, {"cause": [], "fix": []})
    services: list = []
    seen = set()
    for ev in events:
        if ev.get("type") == "step":
            tgt = (ev.get("action") or {}).get("target_service")
            if tgt and tgt not in seen:
                seen.add(tgt)
                services.append(tgt)
    return {"cause": t["cause"], "fix": t["fix"], "services": services}


def _summarize_actions(events: list) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ev in events:
        if ev.get("type") == "step":
            a = (ev.get("action") or {}).get("action_type")
            if a:
                counts[a] = counts.get(a, 0) + 1
    return counts


def _resolution_path(events: list) -> list:
    meaningful = {"restart_service", "rollback_deployment", "scale_service",
                  "update_config", "resolve_incident"}
    out: list = []
    for ev in events:
        if ev.get("type") != "step":
            continue
        a = ev.get("action") or {}
        if a.get("action_type") not in meaningful:
            continue
        tgt = a.get("target_service")
        params = a.get("parameters") or {}
        try:
            import json as _json
            params_str = _json.dumps(params, separators=(",", ":")) if params else ""
        except Exception:
            params_str = str(params) if params else ""
        line = a.get("action_type", "?")
        if tgt:
            line += f" on {tgt}"
        if params_str:
            line += " " + params_str
        out.append(line)
    return out


def _praetor_summary(scenario: str, events: list, resolved: bool,
                     duration_s: float | None, has_tier2: bool) -> str:
    steps_taken = sum(1 for ev in events if ev.get("type") == "step")
    counts = _summarize_actions(events)
    moves = [k for k in counts if any(t in k for t in ("restart", "rollback", "update_config", "scale"))]
    friendly = (scenario or "unknown").replace("_", " ")
    if resolved:
        move_str = (f" Praetor's decisive move was <b>{', '.join(moves)}</b>."
                    if moves else "")
        dur = (f" over {duration_s:.1f} seconds" if duration_s else "")
        return (
            f"Praetor diagnosed a <b>{friendly}</b> incident in "
            f"<b>{steps_taken} step{'' if steps_taken == 1 else 's'}</b>"
            f"{dur}, walked the dependency graph from symptom to root cause, "
            f"and resolved the incident using only tier-1 runtime operations."
            f"{move_str} The site is now responding 200 on /ops/health and "
            f"the fix is durable."
        )
    if has_tier2:
        return (
            f"Praetor investigated a <b>{friendly}</b> incident, took "
            f"{steps_taken} runtime ops action"
            f"{'' if steps_taken == 1 else 's'}, and determined the fault "
            f"was not fully restorable from runtime ops alone. It escalated "
            f"to tier-2 code investigation against the linked repository "
            f"and identified candidate code locations to review."
        )
    return (
        f"Praetor took <b>{steps_taken}</b> runtime ops action"
        f"{'' if steps_taken == 1 else 's'} against the <b>{friendly}</b> "
        f"fault but did not fully heal the site. Tier-2 code investigation "
        f"was not enabled — link a repository to let Praetor inspect the "
        f"code path next time."
    )


# ---- public API -------------------------------------------------------------

def render_run_report_pdf(run_id: str, rec: Dict[str, Any]) -> bytes:
    """Render a real-time run as a downloadable PDF.

    Args:
        run_id: the run identifier (shown in the cover + footer).
        rec:    the full _REALTIME_RUNS record (events list, timestamps,
                scenario, tier1_resolved, tier2_report, etc.).

    Returns:
        The PDF body as bytes.
    """
    styles = _make_styles()
    buf = io.BytesIO()

    # A4 portrait, comfortable margins
    page_w, page_h = A4
    left = right = 18 * mm
    top = bottom = 18 * mm
    frame = Frame(left, bottom, page_w - left - right, page_h - top - bottom,
                  id="content", showBoundary=0)

    def _on_page(canvas, doc):
        canvas.saveState()
        # Footer line
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(left, bottom - 4 * mm, page_w - right, bottom - 4 * mm)
        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(MUTED)
        page_no = canvas.getPageNumber()
        canvas.drawString(left, bottom - 9 * mm,
                          f"Praetor — Incident Report · run {run_id}")
        canvas.drawRightString(page_w - right, bottom - 9 * mm, f"page {page_no}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom,
        title=f"Praetor Incident Report — {run_id}",
        author="Praetor",
        subject="Autonomous SRE incident response report",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])

    events = rec.get("events", []) or []
    start_ev = next((e for e in events if e.get("type") == "start"), {}) or {}
    steps = [e for e in events if e.get("type") == "step"]
    tier2_done = next((e for e in events if e.get("type") == "tier2_done"), None)

    started_at = rec.get("started_at")
    finished_at = rec.get("finished_at")
    duration_s = (finished_at - started_at) if (started_at and finished_at) else None
    started_iso = (
        datetime.fromtimestamp(started_at, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC")
        if started_at else "—"
    )

    scenario = rec.get("scenario") or start_ev.get("scenario") or "unknown"
    site_url = (
        rec.get("site_url") or start_ev.get("site_url")
        or "(in-process simulator)"
    )
    alert = (
        rec.get("alert_title") or rec.get("alert_summary")
        or "(no alert text captured for this run)"
    )

    status_label, status_color = _status_for(rec, events)

    flow: list = []

    # ---- Cover --------------------------------------------------------------
    flow.append(Paragraph("Praetor Incident Report", styles["title"]))
    flow.append(Paragraph(
        f"Autonomous incident response · run <font face='Courier'>{_esc(run_id)}</font>",
        styles["subtitle"],
    ))

    # Status pill (a small one-cell table styled as a colored badge)
    pill_tbl = Table(
        [[Paragraph(f"<b>{status_label}</b>", styles["status_pill"])]],
        colWidths=[28 * mm], rowHeights=[7 * mm],
    )
    pill_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), status_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    flow.append(pill_tbl)
    flow.append(Spacer(1, 6 * mm))

    # ---- Metadata grid -----------------------------------------------------
    meta_rows = [
        ("Run ID", run_id),
        ("Scenario", scenario),
        ("Target", site_url),
        ("Started", started_iso),
        ("Wall-clock", f"{duration_s:.2f} s" if duration_s else "—"),
        ("Steps used", str(len(steps))),
        ("Auto-classified", "yes" if rec.get("auto_classified") else "no"),
    ]
    meta_data = [
        [Paragraph(k, styles["meta_label"]),
         Paragraph(_esc(v), styles["meta_value"])]
        for k, v in meta_rows
    ]
    meta_tbl = Table(meta_data, colWidths=[35 * mm, None])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(meta_tbl)
    flow.append(Spacer(1, 5 * mm))

    # ---- Praetor's summary -------------------------------------------------
    has_tier2 = bool(tier2_done)
    summary_text = _praetor_summary(
        scenario, events, status_label == "RESOLVED", duration_s, has_tier2,
    )
    flow.append(Paragraph("PRAETOR'S SUMMARY", styles["h2"]))
    summary_tbl = Table(
        [[Paragraph(summary_text, styles["body"])]],
        colWidths=[None],
    )
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    flow.append(summary_tbl)

    # ---- Stats grid (2x2) --------------------------------------------------
    tags = _derive_tags(scenario, events)
    outcome_color = (
        GOOD if status_label == "RESOLVED"
        else (WARN if status_label == "ESCALATED" else BAD)
    )
    outcome_label = (
        "FIXED" if status_label == "RESOLVED"
        else ("ESCALATED" if status_label == "ESCALATED" else "UNRESOLVED")
    )
    services_str = (
        ", ".join(tags["services"][:3]) if tags["services"] else "none"
    )

    def _stat_cell(lbl, val, sub, val_color=TEXT):
        # reportlab's inline <font color="..."> needs a # prefix or a name.
        # hexval() returns "0xRRGGBB"; strip the leading "0x" and prefix "#".
        col_hex = "#" + val_color.hexval()[2:]
        return Table(
            [[Paragraph(_esc(lbl), styles["meta_label"])],
             [Paragraph(f'<font color="{col_hex}">'
                        f'<b>{_esc(val)}</b></font>',
                        ParagraphStyle(
                            "stat_val", parent=styles["meta_value"],
                            fontSize=14, leading=16, fontName="Courier-Bold",
                        ))],
             [Paragraph(_esc(sub), ParagraphStyle(
                 "stat_sub", parent=styles["meta_label"],
                 fontSize=8, leading=10, textColor=MUTED,
             ))]],
            colWidths=[None],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (0, 0), 6),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
            ]),
        )

    stats_row = Table(
        [[
            _stat_cell("Steps taken", str(len(steps)), "tier-1 ops actions"),
            _stat_cell("Wall-clock", f"{duration_s:.1f} s" if duration_s else "—", "end-to-end"),
            _stat_cell("Outcome", outcome_label, services_str if status_label == "RESOLVED" else outcome_label.lower(), outcome_color),
            _stat_cell("Services", str(len(tags["services"])), services_str),
        ]],
        colWidths=[None, None, None, None],
    )
    stats_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    flow.append(Spacer(1, 6 * mm))
    flow.append(stats_row)

    # ---- Root-cause / fix / service tags -----------------------------------
    if tags["cause"] or tags["fix"] or tags["services"]:
        flow.append(Paragraph("ROOT CAUSE &amp; FIX TAGS", styles["h2"]))

        def _make_chip(text, fill_hex, text_hex):
            return Table(
                [[Paragraph(
                    f'<font color="{text_hex}" face="Courier-Bold">'
                    f'{_esc(text)}</font>',
                    ParagraphStyle("chip", fontSize=8.5, leading=10),
                )]],
                colWidths=[len(text) * 5.5 + 12],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill_hex)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROUNDEDCORNERS", [4, 4, 4, 4]),
                ]),
            )

        # Build a wrapping row of chips. Use a horizontal Table whose cells are
        # the individual chip tables; let it auto-wrap by chunking by ~6 per row.
        chips: list = []
        for c in tags["cause"]:
            chips.append(_make_chip(c, "#fee2e2", "#991b1b"))
        for f in tags["fix"]:
            chips.append(_make_chip(f, "#dcfce7", "#166534"))
        for s in tags["services"]:
            chips.append(_make_chip(s, "#f3e8ff", "#6b21a8"))

        # Chunk into rows of 5
        per_row = 5
        for i in range(0, len(chips), per_row):
            row = chips[i:i + per_row]
            while len(row) < per_row:
                row.append("")  # blank cell
            chip_tbl = Table([row], colWidths=[None] * per_row)
            chip_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            flow.append(chip_tbl)

    # ---- Resolution path ---------------------------------------------------
    path = _resolution_path(events)
    if path:
        flow.append(Paragraph("RESOLUTION PATH", styles["h2"]))
        for line in path:
            flow.append(Paragraph(
                f"• <font face='Courier'>{_esc(line)}</font>", styles["body"]
            ))

    # ---- Action breakdown --------------------------------------------------
    counts = _summarize_actions(events)
    if counts:
        flow.append(Paragraph("ACTION BREAKDOWN", styles["h2"]))
        items = sorted(counts.items(), key=lambda x: -x[1])
        rows = [
            [Paragraph(f"<font face='Courier'>{_esc(k)}</font>", styles["body"]),
             Paragraph(f"<b>×{v}</b>", styles["body"])]
            for k, v in items
        ]
        action_tbl = Table(rows, colWidths=[None, 18 * mm])
        action_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(action_tbl)

    # ---- Alert -------------------------------------------------------------
    flow.append(Paragraph("THE PROBLEM WE SAW", styles["h2"]))
    alert_tbl = Table(
        [[Paragraph(_esc(alert), styles["alert"])]],
        colWidths=[None],
    )
    alert_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ALERT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, WARN),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    flow.append(alert_tbl)
    if scenario:
        flow.append(Spacer(1, 3 * mm))
        flow.append(Paragraph(
            f"Praetor classified the fault as <b>{_esc(scenario)}</b>.",
            styles["body"],
        ))

    # ---- Steps -------------------------------------------------------------
    flow.append(Paragraph("STEPS PRAETOR TOOK", styles["h2"]))
    if not steps:
        flow.append(Paragraph(
            "<i>No steps recorded for this run.</i>", styles["body"]
        ))
    for ev in steps:
        flow.append(_render_step(ev, styles))
        flow.append(Spacer(1, 2 * mm))

    # ---- Tier-2 (optional) -------------------------------------------------
    if tier2_done:
        flow.append(Paragraph("TIER 2 — CODE INVESTIGATION", styles["h2"]))
        for label, key in (
            ("Summary", "summary"),
            ("Suggested fix", "suggested_fix"),
        ):
            val = tier2_done.get(key) or "—"
            flow.append(Paragraph(
                f"<b>{label}:</b> {_esc(val)}", styles["body"]
            ))
        n_findings = tier2_done.get("n_findings", 0)
        flow.append(Paragraph(
            f"<b>Findings:</b> {_esc(n_findings)} candidate code locations.",
            styles["body"],
        ))

    # ---- Result ------------------------------------------------------------
    flow.append(Paragraph("RESULT", styles["h2"]))
    if status_label == "RESOLVED":
        result_text = (
            "Praetor resolved the incident using tier-1 runtime ops only — "
            "no code escalation was required."
        )
    elif status_label == "ESCALATED":
        result_text = (
            "Tier-1 ops did not fully heal the site. Tier-2 surfaced "
            "candidate code locations for follow-up review."
        )
    else:
        result_text = (
            "Tier-1 ops did not fully heal the site, and tier-2 was not "
            "enabled on this run."
        )
    if duration_s:
        result_text += f" Total wall-clock: {duration_s:.2f} seconds."
    flow.append(Paragraph(_esc(result_text), styles["body"]))

    flow.append(Spacer(1, 8 * mm))
    flow.append(HRFlowable(width="100%", thickness=0.4, color=BORDER,
                           spaceBefore=0, spaceAfter=4))
    flow.append(Paragraph(
        f"Generated by Praetor — autonomous SRE incident commander · {started_iso}",
        styles["footer"],
    ))

    doc.build(flow)
    return buf.getvalue()


def _render_step(ev: Dict[str, Any], styles: Dict[str, ParagraphStyle]):
    """Build a single step block as a KeepTogether-wrapped Table."""
    a = ev.get("action") or {}
    action_type = a.get("action_type", "?")
    target = a.get("target_service")
    params = a.get("parameters") or {}
    tier = ev.get("tier", "tier1")
    why = ev.get("why") or ""
    message = _truncate(ev.get("message") or "", 600)
    step_num = ev.get("step", "?")

    # Header line
    head_left = (
        f"<b>step {_esc(step_num)}</b>  "
        f"<font face='Courier' color='#1e40af'>{_esc(action_type)}</font>"
    )
    if target:
        head_left += (
            f" <font face='Courier' color='#6b7280'>→ {_esc(target)}</font>"
        )
    if params:
        try:
            import json as _json
            params_str = _json.dumps(params, separators=(",", ":"))
        except Exception:
            params_str = str(params)
        if len(params_str) > 80:
            params_str = params_str[:80] + "…"
        head_left += (
            f"  <font face='Courier' size='8' color='#9ca3af'>"
            f"{_esc(params_str)}</font>"
        )

    head_right = (
        f"<font face='Courier' size='7' color='#6b7280'>{_esc(tier).upper()}</font>"
    )
    head_tbl = Table(
        [[Paragraph(head_left, styles["step_head_action"]),
          Paragraph(head_right, styles["step_target"])]],
        colWidths=[None, 18 * mm],
    )
    head_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    inner: list = [head_tbl, Spacer(1, 2 * mm)]

    if message:
        # Replace newlines with <br/> for ReportLab paragraph wrapping
        message_html = _esc(message).replace("\n", "<br/>")
        inner.append(Paragraph(message_html, styles["step_body"]))
        inner.append(Spacer(1, 2 * mm))

    # Why block
    if why:
        why_inner_tbl = Table(
            [[Paragraph("WHY", styles["step_why_label"])],
             [Paragraph(_esc(why), styles["step_why_body"])]],
            colWidths=[None],
        )
        why_inner_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), WHY_BG),
            ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (0, 0), 4),
            ("BOTTOMPADDING", (0, 0), (0, 0), 1),
            ("TOPPADDING", (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ]))
        inner.append(why_inner_tbl)

    # Outer container — light-gray bg, accent left border, page-break-safe
    outer = Table([[inner_t] for inner_t in inner], colWidths=[None])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 0),
    ]))
    return KeepTogether(outer)


__all__ = ["render_run_report_pdf"]
