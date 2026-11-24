"""Standalone, zero-dependency HTML report generator for Funnel-GPT."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from src.score import BreakpointCluster, SessionTrace


def _pill_style_class(surprisal: float, threshold: float, is_breakpoint: bool) -> str:
    if is_breakpoint:
        return "pill-breakpoint"
    if threshold > 0 and surprisal >= 0.70 * threshold:
        return "pill-elevated"
    return "pill-normal"


def render_html_report(
    metrics: dict[str, Any],
    top_transitions: list[BreakpointCluster],
    sample_traces: list[SessionTrace],
    source_csv: str | Path,
) -> str:
    """Render a self-contained HTML/CSS Funnel-GPT diagnostic report."""
    p95 = float(metrics["p95_threshold"])
    goal = html.escape(str(metrics["goal_event"]))
    source_name = html.escape(str(source_csv))

    # Build Top 5 Broken Funnel Transitions rows
    if top_transitions:
        table_rows_parts: list[str] = []
        for rank, cluster in enumerate(top_transitions, start=1):
            prev_ev = html.escape(cluster.prev_event)
            dt_b = html.escape(cluster.dt_bucket)
            bp_ev = html.escape(cluster.breakpoint_event)
            bar_w = min(100.0, max(2.0, cluster.pct_dropoffs))
            table_rows_parts.append(
                f"""
                <tr>
                    <td class="rank-cell">#{rank}</td>
                    <td class="transition-cell">
                        <span class="tok tok-ev">{prev_ev}</span>
                        <span class="arrow">&rarr;</span>
                        <span class="tok tok-dt">{dt_b}</span>
                        <span class="arrow">&rarr;</span>
                        <span class="tok tok-bp">{bp_ev}</span>
                    </td>
                    <td class="num-cell">{cluster.count:,}</td>
                    <td class="pct-cell">
                        <div class="pct-wrap">
                            <span>{cluster.pct_dropoffs:.1f}%</span>
                            <div class="bar-track"><div class="bar-fill" style="width: {bar_w:.1f}%;"></div></div>
                        </div>
                    </td>
                    <td class="num-cell"><span class="loss-badge">{cluster.mean_surprisal:.3f}</span></td>
                </tr>
                """
            )
        table_rows_html = "\n".join(table_rows_parts)
    else:
        table_rows_html = (
            '<tr><td colspan="5" class="empty-state">No abandoned sessions found to score.</td></tr>'
        )

    # Build Color-Coded Session Traces
    traces_parts: list[str] = []
    for trace in sample_traces:
        sid = html.escape(trace.session_id)
        bp_label = html.escape(trace.transition_label)
        pills_html: list[str] = []
        for idx, ev in enumerate(trace.events):
            cls = _pill_style_class(ev.surprisal, p95, ev.is_breakpoint)
            ev_name = html.escape(ev.event_name)
            dt_b = html.escape(ev.dt_bucket)
            tooltip = (
                f"Event #{idx + 1}: {ev_name} | Time delta: {dt_b} | "
                f"Surprisal loss: {ev.surprisal:.4f} (P95 baseline: {p95:.4f})"
                + (" [BREAKPOINT]" if ev.is_breakpoint else "")
            )
            connector = '<span class="trace-arrow">&rarr;</span>' if idx > 0 else ""
            pills_html.append(
                f"""{connector}<span class="pill {cls}" title="{html.escape(tooltip)}" data-tooltip="Loss: {ev.surprisal:.3f} ({dt_b})">
                    <span class="pill-dt">{dt_b}</span>
                    <span class="pill-ev">{ev_name}</span>
                </span>"""
            )
        traces_parts.append(
            f"""
            <div class="trace-card">
                <div class="trace-header">
                    <span class="trace-sid">{sid}</span>
                    <span class="trace-meta">Breakpoint: <code>{bp_label}</code> &middot; Loss <strong>{trace.breakpoint_surprisal:.3f}</strong></span>
                </div>
                <div class="trace-pills">
                    {''.join(pills_html)}
                </div>
            </div>
            """
        )
    traces_html = (
        "\n".join(traces_parts)
        if traces_parts
        else '<p class="empty-state">No abandoned session traces available.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Funnel-GPT Report &mdash; {goal}</title>
<style>
  :root {{
    --bg: #0f172a;
    --surface: #1e293b;
    --surface-2: #0b1220;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --green-bg: #064e3b;
    --green-border: #10b981;
    --green-text: #d1fae5;
    --amber-bg: #78350f;
    --amber-border: #f59e0b;
    --amber-text: #fef3c7;
    --red-bg: #dc2626;
    --red-border: #fca5a5;
    --red-text: #ffffff;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 2rem 1.5rem 4rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .container {{
    max-width: 1120px;
    margin: 0 auto;
  }}
  header {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
    margin-bottom: 1.75rem;
  }}
  h1 {{
    margin: 0;
    font-size: 1.75rem;
    letter-spacing: -0.02em;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.9rem;
  }}
  .subtitle code {{
    color: var(--accent);
    background: var(--surface);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
  }}
  h2 {{
    font-size: 1.15rem;
    margin: 2rem 0 0.9rem;
    color: #e2e8f0;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.9rem;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
  }}
  .metric-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-bottom: 0.35rem;
  }}
  .metric-value {{
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
  }}
  .metric-sub {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.2rem;
  }}
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }}
  th {{
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
  }}
  td {{
    padding: 0.85rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .rank-cell {{
    font-weight: 700;
    color: var(--muted);
    width: 48px;
  }}
  .num-cell {{
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }}
  .transition-cell {{
    white-space: nowrap;
  }}
  .tok {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.82rem;
  }}
  .tok-ev {{
    background: #1e3a5f;
    color: #bae6fd;
  }}
  .tok-dt {{
    background: #312e81;
    color: #c7d2fe;
  }}
  .tok-bp {{
    background: var(--red-bg);
    color: var(--red-text);
    font-weight: 700;
  }}
  .arrow {{
    color: var(--muted);
    margin: 0 0.3rem;
  }}
  .pct-wrap {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    min-width: 140px;
  }}
  .bar-track {{
    flex: 1;
    height: 7px;
    background: var(--surface-2);
    border-radius: 999px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #f97316, #ef4444);
    border-radius: 999px;
  }}
  .loss-badge {{
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600;
    font-size: 0.84rem;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.85rem;
  }}
  .legend-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 999px;
    margin-right: 0.35rem;
    vertical-align: middle;
  }}
  .trace-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
  }}
  .trace-header {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
    font-size: 0.84rem;
  }}
  .trace-sid {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 700;
    color: var(--accent);
  }}
  .trace-meta {{
    color: var(--muted);
  }}
  .trace-meta code {{
    color: #fca5a5;
    background: var(--surface-2);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
  }}
  .trace-pills {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }}
  .trace-arrow {{
    color: #64748b;
    font-size: 0.85rem;
  }}
  .pill {{
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    border: 1px solid transparent;
    cursor: default;
  }}
  .pill:hover::after {{
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #020617;
    color: #f8fafc;
    border: 1px solid var(--border);
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.74rem;
    white-space: nowrap;
    z-index: 20;
    pointer-events: none;
  }}
  .pill-dt {{
    font-size: 0.68rem;
    opacity: 0.82;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .pill-ev {{
    font-weight: 600;
  }}
  .pill-normal {{
    background: var(--green-bg);
    border-color: var(--green-border);
    color: var(--green-text);
  }}
  .pill-elevated {{
    background: var(--amber-bg);
    border-color: var(--amber-border);
    color: var(--amber-text);
  }}
  .pill-breakpoint {{
    background: var(--red-bg);
    border-color: var(--red-border);
    color: var(--red-text);
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.35);
  }}
  .empty-state {{
    text-align: center;
    color: var(--muted);
    padding: 1.5rem;
  }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Funnel-GPT Diagnostic Report</h1>
        <div class="subtitle">
          Source: <code>{source_name}</code> &middot; Goal Event: <code>{goal}</code> &middot; Device: <code>{html.escape(str(metrics["device"]))}</code>
        </div>
      </div>
      <div class="subtitle">Zero-PII From-Scratch PyTorch GPT</div>
    </header>

    <section>
      <h2>1. Executive Summary</h2>
      <div class="summary-grid">
        <div class="metric-card">
          <div class="metric-label">Total Sessions</div>
          <div class="metric-value">{metrics["total_sessions"]:,}</div>
          <div class="metric-sub">{metrics["converting_sessions"]:,} converting / {metrics["non_converting_sessions"]:,} dropped</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Conversion Rate</div>
          <div class="metric-value">{metrics["conversion_rate_pct"]:.2f}%</div>
          <div class="metric-sub">Goal: {goal}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Model Parameters</div>
          <div class="metric-value">{metrics["param_count"]:,}</div>
          <div class="metric-sub">Vocab size: {metrics["vocab_size"]}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Final Train / Val Loss</div>
          <div class="metric-value">{metrics["final_train_loss"]:.3f} / {metrics["final_val_loss"]:.3f}</div>
          <div class="metric-sub">{metrics["epochs"]} epochs (AdamW)</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">P95 Surprisal Threshold</div>
          <div class="metric-value">{p95:.3f}</div>
          <div class="metric-sub">Validation happy-path baseline</div>
        </div>
      </div>
    </section>

    <section>
      <h2>2. Top 5 Broken Funnel Transitions</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Transition (Prev &rarr; Time Delta &rarr; Breakpoint)</th>
              <th>Affected Sessions</th>
              <th>% of Drop-Offs</th>
              <th>Mean Surprisal</th>
            </tr>
          </thead>
          <tbody>
            {table_rows_html}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>3. Color-Coded Session Traces (Sample Abandoned Sessions)</h2>
      <div class="legend">
        <span><span class="legend-dot" style="background: var(--green-border);"></span>Normal (low surprisal)</span>
        <span><span class="legend-dot" style="background: var(--amber-border);"></span>Elevated (&ge;70% of P95)</span>
        <span><span class="legend-dot" style="background: var(--red-bg);"></span>Exact Breakpoint (hover any pill for exact loss)</span>
      </div>
      {traces_html}
    </section>
  </div>
</body>
</html>
"""


def write_html_report(
    out_path: str | Path,
    metrics: dict[str, Any],
    top_transitions: list[BreakpointCluster],
    sample_traces: list[SessionTrace],
    source_csv: str | Path,
) -> Path:
    """Write the standalone HTML report to disk."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html_report(
        metrics=metrics,
        top_transitions=top_transitions,
        sample_traces=sample_traces,
        source_csv=source_csv,
    )
    out.write_text(html_text, encoding="utf-8")
    return out
