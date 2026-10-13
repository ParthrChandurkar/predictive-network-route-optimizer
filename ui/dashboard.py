"""Reusable NOC-style presentation components for Streamlit."""
from __future__ import annotations

from html import escape


THEME_CSS = """
<style>
:root {
  --surface-0: #070b14;
  --surface-1: #0b1220;
  --surface-2: #111b2d;
  --surface-3: #172337;
  --border: #23334d;
  --text: #e8eef8;
  --muted: #8fa1bb;
  --cyan: #22d3ee;
  --blue: #3b82f6;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #fb7185;
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 72% -10%, rgba(34, 211, 238, .08), transparent 31rem),
    linear-gradient(180deg, #070b14 0%, #0a101c 100%);
}
[data-testid="stHeader"] { background: rgba(7, 11, 20, .72); }
.main .block-container { padding-top: 1.25rem; max-width: 1560px; }
[data-testid="stSidebar"] {
  background: #080d17;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .65rem; }
.noc-header {
  position: relative;
  overflow: hidden;
  padding: 1.25rem 1.4rem 1.1rem;
  margin: 0 0 1rem;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: linear-gradient(110deg, rgba(17, 27, 45, .96), rgba(9, 17, 30, .96));
  box-shadow: 0 18px 60px rgba(0, 0, 0, .25);
}
.noc-header:after {
  content: "";
  position: absolute;
  inset: 0 0 0 auto;
  width: 38%;
  opacity: .2;
  background-image: linear-gradient(rgba(34,211,238,.25) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.25) 1px, transparent 1px);
  background-size: 22px 22px;
  mask-image: linear-gradient(90deg, transparent, #000);
}
.noc-eyebrow, .section-kicker, .sidebar-kicker {
  color: var(--cyan);
  font: 700 .68rem/1.2 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.noc-title { color: var(--text); font-size: 1.65rem; font-weight: 720; margin: .35rem 0 .3rem; }
.noc-subtitle { color: var(--muted); font-size: .88rem; margin: 0; max-width: 760px; }
.status-strip { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .9rem; }
.status-chip {
  display: inline-flex;
  align-items: center;
  gap: .42rem;
  color: #cbd7e8;
  background: rgba(7, 12, 22, .78);
  border: 1px solid #263752;
  border-radius: 999px;
  padding: .28rem .62rem;
  font: 650 .66rem/1.2 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .05em;
  text-transform: uppercase;
}
.status-dot { width: .42rem; height: .42rem; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px currentColor; }
.status-dot.warning { background: var(--amber); }
.status-dot.critical { background: var(--red); }
.status-dot.offline { background: #64748b; box-shadow: none; }
.sidebar-brand {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: .9rem 1rem;
  background: linear-gradient(145deg, #101a2b, #0a111e);
  margin-bottom: .35rem;
}
.sidebar-brand strong { color: var(--text); display: block; font-size: 1.02rem; margin-top: .2rem; }
.sidebar-brand span { color: var(--muted); font-size: .72rem; }
.section-heading { margin: .4rem 0 .75rem; }
.section-title { color: var(--text); font-size: 1.05rem; font-weight: 700; margin-top: .15rem; }
.ops-panel {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: rgba(14, 23, 39, .82);
  padding: .85rem .95rem;
  margin: .2rem 0 .75rem;
}
.ops-row { display: flex; justify-content: space-between; gap: 1rem; padding: .4rem 0; border-bottom: 1px solid rgba(51,65,85,.55); }
.ops-row:last-child { border-bottom: 0; }
.ops-label { color: var(--muted); font-size: .74rem; }
.ops-value { color: var(--text); font: 650 .72rem/1.3 "SFMono-Regular", Consolas, monospace; text-align: right; }
[data-testid="metric-container"] {
  min-height: 92px;
  background: linear-gradient(145deg, rgba(17,27,45,.96), rgba(11,18,32,.96));
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: .78rem .95rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: .67rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: .11em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: #f2f7ff !important;
  font: 720 1.42rem/1.15 "SFMono-Regular", Consolas, monospace !important;
}
.stTabs [data-baseweb="tab-list"] {
  gap: .2rem;
  padding: .3rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: rgba(8, 13, 23, .92);
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted);
  height: 2.35rem;
  padding: 0 .8rem;
  border-radius: 7px;
  font-size: .69rem;
  font-weight: 700;
  letter-spacing: .045em;
  text-transform: uppercase;
}
.stTabs [aria-selected="true"] { color: #dffaff !important; background: #11283a !important; }
.stButton button {
  border: 1px solid #285176 !important;
  border-radius: 8px !important;
  color: #e7f8ff !important;
  background: linear-gradient(180deg, #123a5a, #0e2d47) !important;
  font-weight: 650 !important;
}
.stButton button:hover { border-color: var(--cyan) !important; color: #fff !important; }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 11px;
  overflow: hidden;
}
h1, h2, h3 { color: var(--text) !important; }
h2 { font-size: 1.28rem !important; }
h3 { font-size: .98rem !important; }
hr { border-color: var(--border); }
.stCaptionContainer, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
.novelty-card { background: var(--surface-2); border-left: 3px solid var(--cyan); border-radius: 8px; padding: 1rem; }
.gap-card { background: var(--surface-2); border-left: 3px solid var(--amber); border-radius: 8px; padding: 1rem; }
@media (max-width: 900px) {
  .main .block-container { padding-left: .8rem; padding-right: .8rem; }
  .noc-header { padding: 1rem; }
  .noc-title { font-size: 1.3rem; }
  .noc-header:after { display: none; }
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
  .stTabs [data-baseweb="tab"] { padding: 0 .6rem; white-space: nowrap; }
}
</style>
"""


def network_operating_state(health: dict, connected: bool) -> tuple[str, str]:
    """Return a concise state label and CSS severity for dashboard status chips."""
    if not connected or health.get("failed_links", 0) > 0:
        return "Degraded", "critical"
    if health.get("high_risk", 0) > 0 or health.get("health_score", 1.0) < 0.7:
        return "At risk", "warning"
    return "Nominal", "healthy"


def prepare_link_inventory(frame):
    """Return a display-safe link inventory with an explicit operating state."""
    display = frame.copy()
    if display.empty:
        return display
    risk = display["risk_label"] if "risk_label" in display else ["Unknown"] * len(display)
    failed = display["failed"] if "failed" in display else [False] * len(display)
    display.insert(
        0,
        "operating_state",
        [
            "FAILED" if is_failed else "CRITICAL" if label == "High" else "WATCH" if label == "Medium" else "HEALTHY"
            for is_failed, label in zip(failed, risk)
        ],
    )
    if "routing_risk" in display:
        display["routing_risk_percent"] = (display["routing_risk"] * 100).round(1)
    return display


def apply_dashboard_theme(st) -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_sidebar_brand(st) -> None:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-kicker">Control plane</div>'
        '<strong>NetRoute Operations</strong><span>Degradation-aware routing lab</span></div>',
        unsafe_allow_html=True,
    )


def _chip(label: str, value: str, severity: str = "healthy") -> str:
    dot_class = "status-dot" if severity == "healthy" else f"status-dot {severity}"
    return (
        f'<span class="status-chip"><span class="{dot_class}"></span>'
        f'{escape(label)}: {escape(value)}</span>'
    )


def render_app_header(
    st,
    health: dict,
    connected: bool,
    model_loaded: bool,
    observability_enabled: bool,
) -> None:
    state, severity = network_operating_state(health, connected)
    chips = "".join(
        [
            _chip("Network", state, severity),
            _chip("Risk engine", "Ready" if model_loaded else "Standby", "healthy" if model_loaded else "warning"),
            _chip("Telemetry", "OTel live" if observability_enabled else "Local only", "healthy" if observability_enabled else "offline"),
            _chip("Environment", "Simulation lab"),
        ]
    )
    st.markdown(
        '<div class="noc-header"><div class="noc-eyebrow">Network intelligence platform // validated build</div>'
        '<div class="noc-title">Predictive Route Operations</div>'
        '<p class="noc-subtitle">Inspect topology health, detect degradation risk, compare path decisions, '
        f'and validate routing behavior from one control surface.</p><div class="status-strip">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_section_header(st, kicker: str, title: str) -> None:
    st.markdown(
        f'<div class="section-heading"><div class="section-kicker">{escape(kicker)}</div>'
        f'<div class="section-title">{escape(title)}</div></div>',
        unsafe_allow_html=True,
    )


def render_status_panel(st, rows: list[tuple[str, str]]) -> None:
    content = "".join(
        f'<div class="ops-row"><span class="ops-label">{escape(label)}</span>'
        f'<span class="ops-value">{escape(value)}</span></div>'
        for label, value in rows
    )
    st.markdown(f'<div class="ops-panel">{content}</div>', unsafe_allow_html=True)
