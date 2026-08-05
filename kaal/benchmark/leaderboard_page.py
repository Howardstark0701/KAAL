"""KAAL Leaderboard HTML generator — kaal/benchmark/leaderboard_page.py

Produces a standalone single-file HTML leaderboard from a list of
BenchmarkEntry objects. No external CSS, no JS frameworks, no CDN
dependencies (except one optional Google Fonts link for JetBrains Mono).

Usage:
    from kaal.benchmark.leaderboard_page import generate_leaderboard_html
    from kaal.benchmark.runner import BenchmarkEntry

    path = generate_leaderboard_html(entries, "./leaderboard.html")
    print("Saved to", path)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kaal.benchmark.runner import BenchmarkEntry


# ---------------------------------------------------------------------------
# Risk colour mapping (hex values, used in CSS and badges)
# ---------------------------------------------------------------------------

def _kvs_color(score: float) -> str:
    if score <= 2.0:  return "#22C55E"   # green
    if score <= 4.0:  return "#EAB308"   # yellow
    if score <= 6.0:  return "#F97316"   # orange
    return "#EF4444"                      # red


def _kvs_label(score: float) -> str:
    if score <= 2.0:  return "Robust"
    if score <= 4.0:  return "Low Risk"
    if score <= 6.0:  return "Medium Risk"
    if score <= 8.0:  return "High Risk"
    if score <= 9.5:  return "Critical"
    return "Catastrophic"


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.0f}%" if v is not None else "—"


def _bar(score: float) -> str:
    """Render a CSS-only horizontal KVS bar."""
    pct   = score / 10 * 100
    color = _kvs_color(score)
    return (
        f'<div class="bar-wrap">'
        f'  <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'  <span class="bar-label">{score:.1f}</span>'
        f'</div>'
    )


def _badge(score: float) -> str:
    """Render a coloured risk level badge."""
    label = _kvs_label(score)
    color = _kvs_color(score)
    return (
        f'<span class="badge" style="color:{color};border-color:{color};'
        f'background:{color}18;">{label}</span>'
    )


def _fmt_ts(ts: str) -> str:
    """Format ISO timestamp to human-readable date."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ts[:10] if len(ts) >= 10 else ts


# ---------------------------------------------------------------------------
# generate_leaderboard_html()
# ---------------------------------------------------------------------------

def generate_leaderboard_html(
    entries: list[BenchmarkEntry],
    output_path: str = "./leaderboard.html",
) -> str:
    """Generate a standalone HTML leaderboard page.

    Args:
        entries:     List of BenchmarkEntry (will be sorted by kvs_score desc).
        output_path: Where to save the HTML file.

    Returns:
        Absolute path to the saved HTML file.
    """
    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort descending (most vulnerable first)
    sorted_entries = sorted(entries, key=lambda e: e.kvs_score, reverse=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Build table rows ─────────────────────────────────────────────────────
    rows_html = ""
    for rank, e in enumerate(sorted_entries, start=1):
        rows_html += f"""
        <tr>
          <td class="num">{rank}</td>
          <td class="model-name">{e.model_name}</td>
          <td>{_bar(e.kvs_score)}</td>
          <td>{_badge(e.kvs_score)}</td>
          <td class="num">{_pct(e.fgsm_success_rate)}</td>
          <td class="num">{_pct(e.pgd_success_rate)}</td>
          <td class="num">{_pct(e.patch_success_rate)}</td>
          <td class="ts">{_fmt_ts(e.audit_timestamp)}</td>
        </tr>"""

    # ── JSON data for sort (embedded as JS array) ─────────────────────────────
    import json as _json
    js_data = _json.dumps([
        {
            "rank": i + 1,
            "model_name": e.model_name,
            "kvs_score": e.kvs_score,
            "kvs_label": _kvs_label(e.kvs_score),
            "fgsm": e.fgsm_success_rate,
            "pgd":  e.pgd_success_rate,
            "patch": e.patch_success_rate,
            "ts":   _fmt_ts(e.audit_timestamp),
            "color": _kvs_color(e.kvs_score),
        }
        for i, e in enumerate(sorted_entries)
    ], indent=2)

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KAAL Model Vulnerability Leaderboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    /* ── Reset & base ───────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:       #0A0A0A;
      --surface:  #111111;
      --border:   #1F1F1F;
      --text:     #F2F2F2;
      --muted:    #888888;
      --red:      #CC0000;
      --font:     'JetBrains Mono', 'Courier New', Courier, monospace;
    }}

    html, body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
    }}

    /* ── Layout ─────────────────────────────────────────────────────── */
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 48px 24px 64px;
    }}

    /* ── Header ─────────────────────────────────────────────────────── */
    .header {{ margin-bottom: 40px; }}

    .kaal-logo {{
      font-size: 13px;
      font-weight: 700;
      color: var(--red);
      letter-spacing: 0.15em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}

    h1 {{
      font-size: 26px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 8px;
      line-height: 1.2;
    }}

    .subtitle {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 20px;
    }}

    .meta-row {{
      display: flex;
      gap: 24px;
      font-size: 12px;
      color: var(--muted);
    }}

    .meta-row span {{ display: flex; align-items: center; gap: 6px; }}

    /* ── Table wrapper ───────────────────────────────────────────────── */
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}

    thead tr {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}

    th {{
      padding: 12px 14px;
      text-align: left;
      font-weight: 600;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
      position: relative;
      transition: color 0.15s;
    }}

    th:hover {{ color: var(--text); }}

    th .sort-icon {{
      margin-left: 5px;
      font-size: 10px;
      opacity: 0.4;
    }}

    th.active-sort .sort-icon {{ opacity: 1; color: var(--red); }}

    tbody tr {{
      border-bottom: 1px solid var(--border);
      transition: background 0.1s;
    }}

    tbody tr:last-child {{ border-bottom: none; }}
    tbody tr:hover {{ background: #ffffff08; }}

    td {{
      padding: 12px 14px;
      vertical-align: middle;
      color: var(--text);
    }}

    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}

    td.model-name {{
      font-weight: 600;
      white-space: nowrap;
    }}

    td.ts {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}

    /* ── KVS bar ─────────────────────────────────────────────────────── */
    .bar-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 120px;
    }}

    .bar-fill {{
      height: 6px;
      border-radius: 3px;
      flex-shrink: 0;
      transition: width 0.4s ease;
    }}

    .bar-label {{
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      min-width: 28px;
    }}

    /* ── Risk badge ──────────────────────────────────────────────────── */
    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      border: 1px solid;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }}

    /* ── Empty state ─────────────────────────────────────────────────── */
    .empty {{
      text-align: center;
      padding: 60px 0;
      color: var(--muted);
      font-size: 13px;
    }}

    /* ── Footer ──────────────────────────────────────────────────────── */
    footer {{
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
      flex-wrap: wrap;
      gap: 8px;
    }}

    footer a {{
      color: var(--red);
      text-decoration: none;
    }}

    footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="page">

    <!-- Header -->
    <header class="header">
      <div class="kaal-logo">KAAL</div>
      <h1>Model Vulnerability Leaderboard</h1>
      <p class="subtitle">Adversarial robustness scores across popular open-source models</p>
      <div class="meta-row">
        <span>&#9678; {len(sorted_entries)} model{'s' if len(sorted_entries) != 1 else ''}</span>
        <span>&#9678; Generated {generated_at}</span>
        <span>&#9678; Click column headers to sort</span>
      </div>
    </header>

    <!-- Table -->
    <div class="table-wrap">
      <table id="lb-table">
        <thead>
          <tr>
            <th data-col="rank"       onclick="sortBy('rank')">      #  <span class="sort-icon">&#8597;</span></th>
            <th data-col="model_name" onclick="sortBy('model_name')">Model <span class="sort-icon">&#8597;</span></th>
            <th data-col="kvs_score"  onclick="sortBy('kvs_score')" class="active-sort"> KVS Score <span class="sort-icon">&#8595;</span></th>
            <th data-col="kvs_label"  onclick="sortBy('kvs_label')"> Risk Level <span class="sort-icon">&#8597;</span></th>
            <th data-col="fgsm"       onclick="sortBy('fgsm')">       FGSM <span class="sort-icon">&#8597;</span></th>
            <th data-col="pgd"        onclick="sortBy('pgd')">        PGD <span class="sort-icon">&#8597;</span></th>
            <th data-col="patch"      onclick="sortBy('patch')">      Patch <span class="sort-icon">&#8597;</span></th>
            <th data-col="ts"         onclick="sortBy('ts')">         Audited <span class="sort-icon">&#8597;</span></th>
          </tr>
        </thead>
        <tbody id="lb-body">
          {'<tr><td colspan="8" class="empty">No entries yet.</td></tr>' if not sorted_entries else rows_html}
        </tbody>
      </table>
    </div>

    <!-- Footer -->
    <footer>
      <span>Generated by <strong>KAAL</strong> &mdash; Adversarial Robustness Auditing Tool</span>
      <a href="https://github.com/Howardstark0701/Kaal" target="_blank" rel="noreferrer">
        github.com/Howardstark0701/Kaal
      </a>
    </footer>

  </div>

  <!-- Embedded data + sort logic (vanilla JS, no frameworks) -->
  <script>
    const DATA = {js_data};

    // Current sort state
    let sortCol = 'kvs_score';
    let sortAsc = false;   // descending by default (most vulnerable first)

    function pctVal(v) {{
      return v === null || v === undefined ? -1 : v;
    }}

    function cellValue(row, col) {{
      if (col === 'rank')       return row._rank;
      if (col === 'model_name') return (row.model_name || '').toLowerCase();
      if (col === 'kvs_score')  return row.kvs_score;
      if (col === 'kvs_label')  return row.kvs_label || '';
      if (col === 'fgsm')       return pctVal(row.fgsm);
      if (col === 'pgd')        return pctVal(row.pgd);
      if (col === 'patch')      return pctVal(row.patch);
      if (col === 'ts')         return row.ts || '';
      return '';
    }}

    function barHtml(score) {{
      const pct   = (score / 10 * 100).toFixed(1);
      const color = kvsColor(score);
      return `<div class="bar-wrap">
        <div class="bar-fill" style="width:${{pct}}%;background:${{color}};"></div>
        <span class="bar-label">${{score.toFixed(1)}}</span>
      </div>`;
    }}

    function badgeHtml(score, label) {{
      const color = kvsColor(score);
      return `<span class="badge" style="color:${{color}};border-color:${{color}};background:${{color}}18;">${{label}}</span>`;
    }}

    function kvsColor(score) {{
      if (score <= 2.0) return '#22C55E';
      if (score <= 4.0) return '#EAB308';
      if (score <= 6.0) return '#F97316';
      return '#EF4444';
    }}

    function fmtPct(v) {{
      return (v === null || v === undefined) ? '<span style="color:#555">—</span>'
        : (v * 100).toFixed(0) + '%';
    }}

    function renderTable() {{
      const rows = DATA.map((r, i) => ({{ ...r, _rank: i + 1 }}));

      rows.sort((a, b) => {{
        const va = cellValue(a, sortCol);
        const vb = cellValue(b, sortCol);
        if (va < vb) return sortAsc ? -1 :  1;
        if (va > vb) return sortAsc ?  1 : -1;
        return 0;
      }});

      // Re-assign ranks after sort
      rows.forEach((r, i) => r._rank = i + 1);

      const tbody = document.getElementById('lb-body');
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td class="num">${{r._rank}}</td>
          <td class="model-name">${{r.model_name}}</td>
          <td>${{barHtml(r.kvs_score)}}</td>
          <td>${{badgeHtml(r.kvs_score, r.kvs_label)}}</td>
          <td class="num">${{fmtPct(r.fgsm)}}</td>
          <td class="num">${{fmtPct(r.pgd)}}</td>
          <td class="num">${{fmtPct(r.patch)}}</td>
          <td class="ts">${{r.ts}}</td>
        </tr>`).join('');
    }}

    function sortBy(col) {{
      if (sortCol === col) {{
        sortAsc = !sortAsc;
      }} else {{
        sortCol = col;
        // Numeric cols default descending, text cols default ascending
        sortAsc = ['model_name', 'kvs_label', 'ts'].includes(col);
      }}

      // Update header styling
      document.querySelectorAll('th').forEach(th => {{
        const isActive = th.dataset.col === sortCol;
        th.classList.toggle('active-sort', isActive);
        const icon = th.querySelector('.sort-icon');
        if (icon) {{
          icon.innerHTML = isActive ? (sortAsc ? '&#8593;' : '&#8595;') : '&#8597;';
        }}
      }});

      renderTable();
    }}

    // Initial render (data already in tbody from server-side, but JS takes over)
    renderTable();
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
