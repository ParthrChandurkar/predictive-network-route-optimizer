"""Streamlit control surface for the Predictive Network Degradation & Route Optimizer.

The eight workflows cover operations, LTE risk, pattern discovery, topology editing,
route control, algorithm benchmarks, controlled failures, and system insights.
"""
import streamlit as st

st.set_page_config(
    page_title="NetRoute Operations",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_generator  import save_dataset
from model           import train_model, load_model, predict_link, predict_graph_edges
from network_builder import (build_default_topology, add_router, remove_router,
                              add_link, remove_link, update_link, set_link_failed,
                              get_edge_df, graph_summary)
from routing         import (route_traditional, route_ai, compute_ai_weights, compare_routes,
                              route_bellman_ford, route_bellman_ford_ai,
                              route_floyd_warshall, route_floyd_warshall_ai,
                              compare_all_algorithms)
from simulation      import (apply_stress, apply_random_failures,
                              restore_all_failures, network_health, snapshot, restore)
from analytics       import (network_figure, confusion_matrix_fig, feature_importance_fig,
                              route_comparison_fig, health_gauge, risk_pie,
                              lte_degradation_timeline, pattern_rate_figure,
                              feature_shift_figure)
from telemetry       import TelemetryValidationError, load_lte_dataset
from modeling.lte_model import load_lte_risk_model, predict_lte_risk
from integration     import (apply_cell_risk_to_graph, build_demo_edge_cell_mapping,
                             cell_risk_snapshot)
from settings        import (MININET_RESULTS_PATH, PHASE2_REPORT_PATH,
                             PHASE7_REPORT_PATH, PHASE8_REPORT_PATH)
from utils           import RISK_COLORS
from observability   import configure_observability
from ui              import (apply_dashboard_theme, render_app_header,
                             prepare_link_inventory,
                             render_section_header, render_sidebar_brand,
                             render_status_panel)
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np

OBSERVABILITY = configure_observability()
apply_dashboard_theme(st)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "G":        None,
    "model":    None,
    "scaler":   None,
    "metrics":  None,
    "trad":     None,
    "ai":       None,
    "bf":       None,
    "bf_ai":    None,
    "fw":       None,
    "fw_ai":    None,
    "df_train": None,
    "lte_df":   None,
    "lte_report": None,
    "lte_scored": None,
    "lte_routing_summary": None,
    "snap":     None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.G is None:
    st.session_state.G = build_default_topology()
if st.session_state.model is None:
    m, s = load_model()
    st.session_state.model  = m
    st.session_state.scaler = s

G = st.session_state.G

page_health = network_health(G)
page_summary = graph_summary(G)
render_app_header(
    st,
    page_health,
    page_summary["connected"],
    st.session_state.model is not None,
    OBSERVABILITY.enabled,
)


@st.cache_resource
def get_lte_risk_model():
    return load_lte_risk_model()


def refresh_predictions():
    predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
    compute_ai_weights(G)


@st.cache_data
def get_mininet_evidence():
    if not MININET_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(MININET_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data
def get_lte_pattern_report():
    if not PHASE7_REPORT_PATH.exists():
        return None
    try:
        return json.loads(PHASE7_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data
def get_phase8_evaluation():
    if not PHASE8_REPORT_PATH.exists():
        return None
    try:
        return json.loads(PHASE8_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand(st)
    render_section_header(st, "Intelligence", "Risk engine status")
    if st.session_state.model is not None:
        st.success("✅ Model loaded")
        if st.session_state.metrics:
            m = st.session_state.metrics
            st.metric("Synthetic accuracy", f"{m['accuracy']*100:.1f}%")
            st.metric("Synthetic F1", f"{m['f1']*100:.1f}%")
    else:
        st.warning("No model is loaded. Open Risk Intelligence to train the legacy baseline.")

    if st.button("Reset topology", width="stretch"):
        st.session_state.G = build_default_topology()
        G = st.session_state.G
        if st.session_state.model:
            predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
            compute_ai_weights(G)
        for k in ["trad","ai","bf","bf_ai","fw","fw_ai"]:
            st.session_state[k] = None
        st.rerun()

    info   = graph_summary(G)
    health = network_health(G)
    render_section_header(st, "Live topology", "Network summary")
    c1, c2 = st.columns(2)
    c1.metric("Routers", info["routers"])
    c2.metric("Links",   info["links"])
    st.metric("Health", f"{health['health_score']*100:.0f}%")
    if health["failed_links"]:
        st.error(f"⚠️ {health['failed_links']} link(s) FAILED")
    render_status_panel(st, [
        ("Connectivity", "Connected" if info["connected"] else "Partitioned"),
        ("Risk source", "LTE demo" if st.session_state.lte_routing_summary else "Synthetic baseline"),
        ("Trace export", "OTLP active" if OBSERVABILITY.enabled else "Disabled"),
    ])
    st.info("This Streamlit application is the project’s interactive simulation and operations interface.")
    st.caption("Simulation environment · decisions do not configure physical routers")


# ── Tabs ──────────────────────────────────────────────────────────────────────
(t_overview, t_model, t_patterns, t_builder, t_routing,
 t_compare, t_sim, t_analytics) = st.tabs([
    "Network Ops", "Risk Intelligence", "Pattern Discovery", "Topology", "Route Control",
    "Benchmarks", "Failure Lab", "Insights"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:
    render_section_header(st, "Network operations", "Live topology and route posture")
    st.caption(
        "Topology health and selected paths. Cyan indicates link risk, blue shows the "
        "distance route, and green shows the degradation-aware route."
    )
    health = network_health(G)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Active nodes", G.number_of_nodes())
    c2.metric("Monitored links", G.number_of_edges())
    c3.metric("Failed links", health["failed_links"])
    c4.metric("High-risk links", health["high_risk"])
    c5.metric("Fabric health", f"{health['health_score']*100:.0f}%")

    col_g, col_h = st.columns([3.25, 1], gap="medium")
    with col_g:
        trad_edges = st.session_state.trad["edges"] if st.session_state.trad and st.session_state.trad["found"] else []
        ai_edges   = st.session_state.ai["edges"]   if st.session_state.ai   and st.session_state.ai["found"]   else []
        st.plotly_chart(
            network_figure(G, trad_edges, ai_edges, "Network fabric · live route overlay"),
            width="stretch",
            key="ov_network",
        )
    with col_h:
        render_section_header(st, "Control plane", "Operational posture")
        active_route = "Not calculated"
        if st.session_state.ai and st.session_state.ai.get("found"):
            active_route = st.session_state.ai["path_str"]
        render_status_panel(st, [
            ("Topology", "Connected" if graph_summary(G)["connected"] else "Partitioned"),
            ("Risk provenance", "LTE telemetry" if st.session_state.lte_routing_summary else "Synthetic baseline"),
            ("Risk-aware path", active_route),
            ("Failed edges", str(health["failed_links"])),
        ])
        if health["failed_links"]:
            st.error(f"{health['failed_links']} failed link(s) require operator attention.")
        elif health["high_risk"]:
            st.warning(f"{health['high_risk']} high-risk link(s) are being monitored.")
        else:
            st.success("No active link incidents in the current topology.")
        st.plotly_chart(health_gauge(health["health_score"]), width="stretch", key="ov_gauge")
        evidence = get_mininet_evidence()
        if evidence:
            comparison = evidence["comparison"]
            render_section_header(st, "Emulation evidence", "Mininet validation")
            ec1, ec2 = st.columns(2)
            ec1.metric("Throughput", f"+{comparison['throughput_gain_percent']:.0f}%")
            ec2.metric("RTT", f"-{comparison['rtt_reduction_percent']:.0f}%")
            st.caption("Three controlled repetitions · not a production guarantee")

    render_section_header(st, "Link telemetry", "Risk and operating-state inventory")
    df = get_edge_df(G)
    if not df.empty:
        inventory = prepare_link_inventory(df)
        show_cols = [c for c in ["source","target","distance","bandwidth","latency",
                                  "packet_loss","traffic_load","failure_prob","routing_risk_percent",
                                  "risk_source","telemetry_cell","risk_label","failed"]
                     if c in inventory.columns]
        show_cols.insert(0, "operating_state")
        st.dataframe(
            inventory[show_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "operating_state": st.column_config.TextColumn("State"),
                "routing_risk_percent": st.column_config.ProgressColumn(
                    "Route risk", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with t_model:
    render_section_header(st, "Risk intelligence", "Degradation scoring and model evidence")
    st.markdown(
        "Inspect the real LTE unusual-behavior baseline or train the legacy synthetic "
        "link-risk model. Scores influence route cost; they do not prove physical failure."
    )

    with st.expander("Real LTE telemetry · validated degradation model", expanded=True):
        st.caption(
            "Upload the Kaggle training or test CSV to validate and normalize it. "
            "The held-out-cell baseline detects unusual LTE behavior and can be applied to "
            "the demonstration topology through an explicitly synthetic cell-to-link mapping."
        )
        if PHASE2_REPORT_PATH.exists():
            phase2_report = json.loads(PHASE2_REPORT_PATH.read_text(encoding="utf-8"))
            held_out = phase2_report["protocols"]["unseen_cell"]["models"]["random_forest"]["aggregate"]
            st.markdown("#### Held-out-cell baseline")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("F1", f"{held_out['f1']:.3f}")
            bc2.metric("ROC-AUC", f"{held_out['roc_auc']:.3f}")
            bc3.metric("Recall", f"{held_out['recall']:.3f}")
            st.caption("Cells in each validation fold are absent from that fold's training data.")
        uploaded_lte = st.file_uploader(
            "LTE telemetry CSV",
            type=["csv"],
            key="lte_telemetry_upload",
        )
        if uploaded_lte is not None:
            try:
                lte_dataset = load_lte_dataset(uploaded_lte, require_label=False)
            except TelemetryValidationError as exc:
                st.error(f"Dataset validation failed: {exc}")
            else:
                st.session_state.lte_df = lte_dataset.frame
                st.session_state.lte_report = lte_dataset.report.to_dict()
                report = st.session_state.lte_report
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Rows", f"{report['rows']:,}")
                lc2.metric("Cells", report["cells"])
                lc3.metric("Labeled", "Yes" if report["labeled"] else "No")
                st.dataframe(st.session_state.lte_df.head(20), width="stretch")
                if report["missing_values"]:
                    st.warning(f"Missing values: {report['missing_values']}")
                for warning in report["warnings"]:
                    st.info(warning)
                lte_model = get_lte_risk_model()
                if lte_model is None:
                    st.warning("Train the local LTE artifact with `python -m modeling.lte_model`.")
                else:
                    st.markdown("#### Experimental routing integration")
                    time_slot = st.selectbox(
                        "Time-of-day snapshot",
                        sorted(st.session_state.lte_df["time_of_day"].unique()),
                        key="lte_time_slot",
                    )
                    st.caption(
                        "The demo assigns LTE cells to abstract links deterministically. "
                        "It does not claim knowledge of the deployment's physical topology."
                    )
                    if st.button("Apply LTE degradation risk to demo topology", key="apply_lte_risk"):
                        with st.spinner("Scoring telemetry and updating routing weights…"):
                            scored = predict_lte_risk(lte_model, st.session_state.lte_df)
                            st.session_state.lte_scored = scored
                            cell_risks = cell_risk_snapshot(scored, time_slot)
                            mapping = build_demo_edge_cell_mapping(G, cell_risks)
                            summary = apply_cell_risk_to_graph(G, cell_risks, mapping)
                            compute_ai_weights(G)
                            st.session_state.lte_routing_summary = summary
                        st.success(
                            f"Applied {summary['mapped_cells']} cell scores to "
                            f"{summary['mapped_edges']} links for {time_slot}."
                        )

        if st.session_state.lte_df is not None:
            render_section_header(st, "Temporal signal", "LTE degradation timeline")
            st.plotly_chart(
                lte_degradation_timeline(
                    st.session_state.lte_df,
                    st.session_state.lte_scored,
                ),
                width="stretch",
                key="lte_degradation_timeline",
            )
            st.caption(
                "The source provides time of day but no calendar date; this is a recurring "
                "time-slot profile, not a chronological incident history."
            )

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Train Model")
        n_samples = st.slider("Dataset size", 400, 2000, 1700, 100)
        if st.button("Train synthetic baseline", width="stretch"):
            with st.spinner("Generating dataset…"):
                df = save_dataset(n_samples=n_samples)
                st.session_state.df_train = df
            with st.spinner("Training RandomForest…"):
                mdl, scl, mtr = train_model(df)
                st.session_state.model   = mdl
                st.session_state.scaler  = scl
                st.session_state.metrics = mtr
            with st.spinner("Updating edge predictions…"):
                predict_graph_edges(mdl, scl, G)
                compute_ai_weights(G)
            st.success(f"✅ Trained! Accuracy: {mtr['accuracy']*100:.1f}%  F1: {mtr['f1']*100:.1f}%")
            st.rerun()

        if st.session_state.df_train is not None:
            df = st.session_state.df_train
            st.markdown("#### Dataset Preview")
            st.dataframe(df.head(8), width="stretch")
            st.caption(f"Rows: {len(df)}  |  Failure rate: {df['failure'].mean()*100:.1f}%")

        st.markdown("---")
        st.markdown("### Algorithm Complexity")
        st.markdown("""
| Algorithm | Time | Space |
|-----------|------|-------|
| Dijkstra | O((V+E) log V) | O(V) |
| Bellman-Ford | O(V·E) | O(V) |
| Floyd-Warshall | O(V³) | O(V²) |
| RF Train | O(n·d·k·log n) | O(k·d) |
| RF Infer | O(k·log n) | O(k) |
        """)

    with right:
        if st.session_state.metrics:
            m = st.session_state.metrics
            st.markdown("### Evaluation Metrics")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
            mc2.metric("Precision", f"{m['precision']*100:.1f}%")
            mc3.metric("Recall",    f"{m['recall']*100:.1f}%")
            mc4.metric("F1 Score",  f"{m['f1']*100:.1f}%")
            st.markdown("---")
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(confusion_matrix_fig(m["cm"]), width="stretch", key="ml_cm")
            with r2:
                st.plotly_chart(feature_importance_fig(m["feature_importance"]), width="stretch", key="ml_fi")
            with st.expander("📄 Full Classification Report"):
                st.code(m["report"])
        else:
            st.info("Use **Train synthetic baseline** to unlock the legacy model panels.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LTE PATTERN DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
with t_patterns:
    render_section_header(st, "LTE evidence", "Validated unusual-behavior patterns")
    st.markdown(
        "This analysis separates **where and when labels occur** from the multivariate KPI "
        "signatures that predict them. Rules are discovered on one set of cells and checked "
        "on disjoint cells."
    )
    pattern_report = get_lte_pattern_report()
    if pattern_report is None:
        st.info("Generate the evidence report with `python -m analysis.lte_patterns`.")
    else:
        dataset = pattern_report["dataset"]
        findings = pattern_report["headline_findings"]
        rule_evidence = pattern_report["validated_interpretable_rules"]
        top_rule = findings["dominant_validated_signature"]

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Analyzed rows", f"{dataset['deduplicated_rows']:,}")
        pc2.metric("Unusual labels", f"{dataset['unusual_rate'] * 100:.1f}%")
        pc3.metric("Rule precision", f"{top_rule['validation_unusual_rate'] * 100:.1f}%")
        pc4.metric("Rule coverage", f"{top_rule['validation_unusual_coverage'] * 100:.1f}%")

        st.warning(
            "The strongest cross-cell signature is a narrow measurement band. Its very high "
            "precision may indicate KPI quantization or label construction—not a physical "
            "network mechanism."
        )
        render_section_header(st, "Strongest held-out signature", "Interpretable rule")
        st.code(top_rule["rule"], language=None)
        render_status_panel(st, [
            ("Validation cells", str(len(rule_evidence["validation_cells"]))),
            ("Validation observations", f"{top_rule['validation_observations']:,}"),
            ("Unusual-label lift", f"{top_rule['validation_lift']:.2f}×"),
            ("Shallow-tree ROC-AUC", f"{rule_evidence['validation_metrics']['roc_auc']:.3f}"),
        ])

        spatial_col, temporal_col = st.columns(2)
        with spatial_col:
            st.plotly_chart(
                pattern_rate_figure(
                    pattern_report["cell_patterns"],
                    "cell_id",
                    "Unusual-label rate by LTE cell",
                    dataset["unusual_rate"],
                ),
                width="stretch",
                key="pattern_cells",
            )
        with temporal_col:
            hourly = sorted(pattern_report["hour_patterns"], key=lambda item: item["hour"])
            st.plotly_chart(
                pattern_rate_figure(
                    hourly,
                    "hour",
                    "Unusual-label rate by hour of day",
                    dataset["unusual_rate"],
                ),
                width="stretch",
                key="pattern_hours",
            )

        st.caption(
            f"Cell-rate spread: {findings['cell_rate_spread_points']:.1f} percentage points · "
            f"hour-rate spread: {findings['hour_rate_spread_points']:.1f} points. "
            "The evidence does not show a small set of dominant cell or hourly hotspots."
        )
        shift_col, rules_col = st.columns([1.2, 1])
        with shift_col:
            st.plotly_chart(
                feature_shift_figure(pattern_report["feature_shifts"]),
                width="stretch",
                key="pattern_feature_shifts",
            )
        with rules_col:
            render_section_header(st, "Held-out cells", "High-lift rule evidence")
            rules_frame = pd.DataFrame(rule_evidence["high_lift_rules"])
            st.dataframe(
                rules_frame[[
                    "rule",
                    "validation_observations",
                    "validation_unusual_rate",
                    "validation_lift",
                    "validation_unusual_coverage",
                ]],
                width="stretch",
                hide_index=True,
                column_config={
                    "validation_unusual_rate": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "validation_lift": st.column_config.NumberColumn("Lift", format="%.2f×"),
                    "validation_unusual_coverage": st.column_config.NumberColumn("Coverage", format="%.3f"),
                },
            )
            st.caption(
                "Rules are predictive associations. They must not be presented as causal "
                "failure mechanisms without independent operational evidence."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — NETWORK BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with t_builder:
    render_section_header(st, "Topology management", "Network fabric configuration")
    bl, br = st.columns([1, 2])

    with bl:
        st.markdown("### Routers")
        r_name = st.text_input("Router name", placeholder="e.g. R9")
        ra, rb = st.columns(2)
        if ra.button("Add", width="stretch"):
            if r_name.strip():
                ok, msg = add_router(G, r_name.strip())
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
            else:
                st.warning("Enter a router name.")
        if rb.button("Remove", width="stretch"):
            if r_name.strip():
                ok, msg = remove_router(G, r_name.strip())
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
            else:
                st.warning("Enter a router name.")

        st.markdown("---")
        st.markdown("### Add Link")
        nodes = sorted(G.nodes())
        if len(nodes) >= 2:
            nc1, nc2 = st.columns(2)
            l_src  = nc1.selectbox("Source", nodes, key="bld_src")
            l_dst  = nc2.selectbox("Dest",   nodes, key="bld_dst", index=min(1, len(nodes)-1))
            l_dist = st.slider("Distance (km)",    1, 200,  20, key="bld_dist")
            l_bw   = st.slider("Bandwidth (Mbps)", 10, 1000, 300, key="bld_bw")
            l_lat  = st.slider("Latency (ms)",     1, 300,  20, key="bld_lat")
            l_pkt  = st.slider("Packet Loss (%)",  0,  20,   2, key="bld_pkt")
            l_traf = st.slider("Traffic Load (%)", 0, 100,  50, key="bld_traf")
            if st.button("Add link", width="stretch"):
                attrs = dict(distance=float(l_dist), bandwidth=float(l_bw),
                             latency=float(l_lat), packet_loss=float(l_pkt),
                             traffic_load=float(l_traf))
                ok, msg = add_link(G, l_src, l_dst, attrs)
                (st.success if ok else st.error)(msg)
                if ok:
                    if st.session_state.model:
                        refresh_predictions()
                    st.rerun()
        else:
            st.info("Add at least 2 routers first.")

        st.markdown("---")
        st.markdown("### Remove Link")
        edge_opts = [f"{u}↔{v}" for u,v in G.edges()]
        if edge_opts:
            sel_rm = st.selectbox("Select link to remove", edge_opts, key="rm_link")
            if st.button("Remove link", width="stretch"):
                p = sel_rm.split("↔")
                ok, msg = remove_link(G, p[0], p[1])
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()

        st.markdown("---")
        st.markdown("### Edit Link Attributes")
        if edge_opts:
            sel_ed = st.selectbox("Select link to edit", edge_opts, key="ed_link")
            p2 = sel_ed.split("↔"); u2, v2 = p2[0], p2[1]
            ex = G[u2][v2] if G.has_edge(u2, v2) else {}
            ec1, ec2 = st.columns(2)
            e_dist = ec1.number_input("Distance",    value=float(ex.get("distance",    20)), min_value=1.0,  step=1.0,  key="e_dist")
            e_bw   = ec2.number_input("Bandwidth",   value=float(ex.get("bandwidth",  300)), min_value=1.0,  step=10.0, key="e_bw")
            e_lat  = ec1.number_input("Latency",     value=float(ex.get("latency",     20)), min_value=1.0,  step=1.0,  key="e_lat")
            e_pkt  = ec2.number_input("Packet Loss", value=float(ex.get("packet_loss",  1)), min_value=0.0,  step=0.5,  key="e_pkt")
            e_traf = ec1.number_input("Traffic Load",value=float(ex.get("traffic_load",50)), min_value=0.0,  step=5.0,  key="e_traf")
            if st.button("Update link", width="stretch"):
                ok, msg = update_link(G, u2, v2, dict(distance=e_dist, bandwidth=e_bw,
                    latency=e_lat, packet_loss=e_pkt, traffic_load=e_traf))
                (st.success if ok else st.error)(msg)
                if ok and st.session_state.model:
                    refresh_predictions(); st.rerun()

    with br:
        st.markdown("### Current Topology")
        st.plotly_chart(network_figure(G, title="Network topology"), width="stretch", key="bld_network")
        st.markdown("### Edge Table")
        df_e = get_edge_df(G)
        if not df_e.empty:
            topology_inventory = prepare_link_inventory(df_e)
            cols = [c for c in ["operating_state","source","target","distance","bandwidth","latency",
                                 "packet_loss","traffic_load","failure_prob","routing_risk_percent",
                                 "risk_source","telemetry_cell","risk_label","failed"]
                    if c in topology_inventory.columns]
            st.dataframe(topology_inventory[cols], width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROUTING (original 2-algo view)
# ══════════════════════════════════════════════════════════════════════════════
with t_routing:
    render_section_header(st, "Route control", "Path calculation and decision comparison")
    st.markdown(
        "**Traditional Dijkstra** minimises distance only. "
        "**Risk-aware Dijkstra** adds `ai_weight = distance + routing_risk × penalty` "
        "to steer traffic around high-risk links."
    )
    nodes = sorted(G.nodes())
    if len(nodes) < 2:
        st.warning("Add at least 2 routers.")
    else:
        rc1, rc2, rc3 = st.columns([1,1,2])
        src     = rc1.selectbox("Source",      nodes, key="rt_src")
        dst     = rc2.selectbox("Destination", nodes, key="rt_dst", index=min(1,len(nodes)-1))
        penalty = rc3.slider("Risk penalty factor", 50, 500, 200, 10)

        if st.button("Compute routes", width="stretch"):
            if src == dst:
                st.error("Source and destination must differ.")
            else:
                compute_ai_weights(G, penalty)
                st.session_state.trad = route_traditional(G, src, dst)
                st.session_state.ai   = route_ai(G, src, dst, penalty)

        trad = st.session_state.trad
        ai   = st.session_state.ai

        if trad is not None and ai is not None:
            st.markdown("---")
            t_edges = trad["edges"] if trad["found"] else []
            a_edges = ai["edges"]   if ai["found"]   else []
            st.plotly_chart(network_figure(G, t_edges, a_edges, "Routing Visualisation"),
                            width="stretch", key="rt_network")
            st.markdown("---")
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("#### 🔵 Traditional Dijkstra")
                if trad["found"]:
                    st.success(trad["path_str"])
                    m1,m2 = st.columns(2)
                    m1.metric("Distance", f"{trad['dist_cost']} km")
                    m2.metric("Hops",     trad["hops"])
                    m3,m4 = st.columns(2)
                    m3.metric("Risky Links", trad["risky_links"])
                    m4.metric("Avg Risk",    f"{trad['avg_risk']*100:.1f}%")
                    st.caption(f"⏱ {trad['elapsed_ms']:.2f} ms")
                else:
                    st.error("No path found.")

            with dc2:
                st.markdown("#### Risk-aware Dijkstra")
                if ai["found"]:
                    st.success(ai["path_str"])
                    m1,m2 = st.columns(2)
                    m1.metric("Distance", f"{ai['dist_cost']} km")
                    m2.metric("Hops",     ai["hops"])
                    m3,m4 = st.columns(2)
                    m3.metric("Risky Links", ai["risky_links"])
                    m4.metric("Avg Risk",    f"{ai['avg_risk']*100:.1f}%")
                    st.caption(f"⏱ {ai['elapsed_ms']:.2f} ms")
                else:
                    st.error("No path found.")

            if trad["found"] and ai["found"]:
                recommended = "Risk-aware" if ai["avg_risk"] < trad["avg_risk"] else "Traditional"
                render_section_header(st, "Decision record", "Route recommendation")
                render_status_panel(st, [
                    ("Recommended policy", recommended),
                    ("Traditional path", trad["path_str"]),
                    ("Risk-aware path", ai["path_str"]),
                    ("Risk delta", f"{(trad['avg_risk'] - ai['avg_risk']) * 100:+.1f} points"),
                    ("Distance overhead", f"{ai['dist_cost'] - trad['dist_cost']:+.1f} km"),
                ])

            st.plotly_chart(route_comparison_fig(trad, ai), width="stretch", key="rt_compare")

            if trad["found"] and ai["found"]:
                risk_delta = trad["avg_risk"] - ai["avg_risk"]
                dist_delta = ai["dist_cost"] - trad["dist_cost"]
                if risk_delta > 0.005:
                    st.info(f"Risk-aware routing reduces average path risk by **{risk_delta*100:.1f}%** "
                            f"at an extra **{dist_delta:.0f} km** distance overhead.")
                elif risk_delta < -0.005:
                    st.warning("⚠️ Traditional route has lower risk in current topology state.")
                else:
                    st.info("ℹ️ Both algorithms chose paths with similar risk levels.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALGORITHM COMPARISON  ★ NEW TAB ★
# ══════════════════════════════════════════════════════════════════════════════
with t_compare:
    render_section_header(st, "Routing benchmarks", "Dijkstra · Bellman-Ford · Floyd-Warshall")
    st.markdown(
        "Run all **six routing algorithms** (three traditional and three risk-aware) simultaneously "
        "and compare their paths, distances, risk levels, and execution times."
    )

    final_evaluation = get_phase8_evaluation()
    if final_evaluation:
        headline = final_evaluation["headline"]
        verification = final_evaluation["verification"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Held-out LTE F1", f"{headline['held_out_cell_f1']:.3f}")
        e2.metric("Held-out ROC-AUC", f"{headline['held_out_cell_roc_auc']:.3f}")
        e3.metric("Mininet throughput", f"+{headline['mininet_throughput_gain_percent']:.1f}%")
        e4.metric("Evidence checks", f"{verification['passed']}/{verification['total']}")
        st.caption(
            "Held-out-cell model evidence and controlled Mininet measurements are separate. "
            "The LTE-to-link mapping is a deterministic integration demonstration, not a physical topology."
        )

    if not st.session_state.model:
        st.warning("Train the legacy model in Risk Intelligence to populate synthetic risk scores.")

    nodes = sorted(G.nodes())
    if len(nodes) < 2:
        st.warning("Add at least 2 routers in the Builder tab.")
    else:
        cc1, cc2, cc3 = st.columns([1,1,2])
        cmp_src     = cc1.selectbox("Source",      nodes, key="cmp_src")
        cmp_dst     = cc2.selectbox("Destination", nodes, key="cmp_dst", index=min(1,len(nodes)-1))
        cmp_penalty = cc3.slider("Risk penalty factor", 50, 500, 200, 10, key="cmp_pen")

        if st.button("Run all six algorithms", width="stretch", key="cmp_run"):
            if cmp_src == cmp_dst:
                st.error("Source and destination must differ.")
            else:
                with st.spinner("Computing all routes…"):
                    compute_ai_weights(G, cmp_penalty)
                    st.session_state.trad  = route_traditional(G, cmp_src, cmp_dst)
                    st.session_state.ai    = route_ai(G, cmp_src, cmp_dst, cmp_penalty)
                    st.session_state.bf    = route_bellman_ford(G, cmp_src, cmp_dst)
                    st.session_state.bf_ai = route_bellman_ford_ai(G, cmp_src, cmp_dst, cmp_penalty)
                    st.session_state.fw    = route_floyd_warshall(G, cmp_src, cmp_dst)
                    st.session_state.fw_ai = route_floyd_warshall_ai(G, cmp_src, cmp_dst, cmp_penalty)
                st.success("All 6 algorithms computed!")

        all_results = [
            st.session_state.trad, st.session_state.ai,
            st.session_state.bf,   st.session_state.bf_ai,
            st.session_state.fw,   st.session_state.fw_ai,
        ]

        if any(r is not None for r in all_results):
            valid = [r for r in all_results if r is not None]
            rows  = compare_all_algorithms(valid)
            df_cmp = pd.DataFrame(rows)

            st.markdown("---")
            st.markdown("### 📊 Summary Comparison Table")
            st.dataframe(df_cmp, width="stretch", hide_index=True)

            st.markdown("---")

            # ── Bar chart: Distance comparison ──────────────────────────────
            found_valid = [r for r in valid if r.get("found")]
            if found_valid:
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("#### Distance (km) by Algorithm")
                    algos = [r["algorithm"] for r in found_valid]
                    dists = [r["dist_cost"]  for r in found_valid]
                    colors = ["#3b82f6","#10b981","#f59e0b","#f472b6","#a78bfa","#34d399"]
                    fig_dist = go.Figure(go.Bar(
                        x=algos, y=dists,
                        marker_color=colors[:len(algos)],
                        text=[f"{d} km" for d in dists],
                        textposition="outside"
                    ))
                    fig_dist.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#e2e8f0"), height=350,
                        yaxis=dict(title="Distance (km)", gridcolor="#1e293b"),
                        xaxis=dict(tickangle=-20),
                        margin=dict(l=10,r=10,t=30,b=80)
                    )
                    st.plotly_chart(fig_dist, width="stretch", key="cmp_dist")

                with col_chart2:
                    st.markdown("#### Avg Risk (%) by Algorithm")
                    risks = [r["avg_risk"] * 100 for r in found_valid]
                    fig_risk = go.Figure(go.Bar(
                        x=algos, y=risks,
                        marker_color=colors[:len(algos)],
                        text=[f"{rv:.1f}%" for rv in risks],
                        textposition="outside"
                    ))
                    fig_risk.update_layout(
                        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                        font=dict(color="#e2e8f0"), height=350,
                        yaxis=dict(title="Avg Risk (%)", gridcolor="#1e293b"),
                        xaxis=dict(tickangle=-20),
                        margin=dict(l=10,r=10,t=30,b=80)
                    )
                    st.plotly_chart(fig_risk, width="stretch", key="cmp_risk")

                # ── Execution time chart ─────────────────────────────────────
                st.markdown("#### ⏱ Execution Time (ms) by Algorithm")
                times = [r["elapsed_ms"] for r in valid]
                algos_all = [r["algorithm"] for r in valid]
                fig_time = go.Figure(go.Bar(
                    x=algos_all, y=times,
                    marker_color=colors[:len(algos_all)],
                    text=[f"{t:.3f} ms" for t in times],
                    textposition="outside"
                ))
                fig_time.update_layout(
                    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"), height=300,
                    yaxis=dict(title="Time (ms)", gridcolor="#1e293b"),
                    xaxis=dict(tickangle=-20),
                    margin=dict(l=10,r=10,t=30,b=80)
                )
                st.plotly_chart(fig_time, width="stretch", key="cmp_time")

            st.markdown("---")
            # ── Per-algorithm detail cards ───────────────────────────────────
            st.markdown("### 🔍 Per-Algorithm Path Details")
            labels = ["🔵 Dijkstra (Trad.)", "🟢 AI Dijkstra",
                      "🟠 Bellman-Ford", "🟣 AI Bellman-Ford",
                      "🔷 Floyd-Warshall", "💜 AI Floyd-Warshall"]
            result_keys = ["trad","ai","bf","bf_ai","fw","fw_ai"]

            for i in range(0, 6, 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    r = getattr(st.session_state, result_keys[idx], None) if idx < 6 else None
                    if r is None: continue
                    with col:
                        st.markdown(f"**{labels[idx]}**")
                        if r["found"]:
                            st.code(r["path_str"])
                            mc1,mc2,mc3,mc4 = st.columns(4)
                            mc1.metric("km",   r["dist_cost"])
                            mc2.metric("Hops", r["hops"])
                            mc3.metric("Risk", f"{r['avg_risk']*100:.1f}%")
                            mc4.metric("ms",   r["elapsed_ms"])
                        else:
                            st.error("No path found")

            st.markdown("---")
            # ── Floyd-Warshall All-Pairs Distance Matrix ─────────────────────
            fw_result = st.session_state.fw or st.session_state.fw_ai
            if fw_result and fw_result.get("dist_matrix"):
                st.markdown("### 🗂️ Floyd-Warshall All-Pairs Distance Matrix")
                st.caption("Shows the shortest distance (km) between every pair of routers — computed in a single O(V³) pass.")
                dm = fw_result["dist_matrix"]
                all_nodes = sorted(dm.keys())
                matrix_data = []
                for u in all_nodes:
                    row_data = {"Router": u}
                    for v in all_nodes:
                        val = dm.get(u, {}).get(v, float("inf"))
                        row_data[v] = "—" if (u == v) else (f"{val:.0f}" if val < 1e8 else "∞")
                    matrix_data.append(row_data)
                df_matrix = pd.DataFrame(matrix_data).set_index("Router")
                st.dataframe(df_matrix, width="stretch")

            st.markdown("---")
            # ── Algorithm Theory ─────────────────────────────────────────────
            st.markdown("### 📚 Algorithm Theory & Trade-offs")
            theory_data = {
                "Algorithm":     ["Dijkstra", "Bellman-Ford", "Floyd-Warshall"],
                "Time Complexity": ["O((V+E) log V)", "O(V·E)", "O(V³)"],
                "Space Complexity": ["O(V)", "O(V)", "O(V²)"],
                "Negative Weights": ["❌ No", "✅ Yes", "✅ Yes"],
                "All-Pairs": ["❌ No", "❌ No", "✅ Yes"],
                "Best For":  ["Large sparse graphs", "Failure-penalised weights", "Small dense networks / routing tables"],
            }
            st.dataframe(pd.DataFrame(theory_data), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with t_sim:
    render_section_header(st, "Failure laboratory", "Controlled degradation and outage injection")
    st.markdown(
        "Apply controlled traffic, latency, packet-loss, and outage conditions. "
        "Scenarios mutate only the in-memory laboratory topology and are fully reversible."
    )

    sl, sr = st.columns([1, 2])
    with sl:
        st.markdown("### Stress Parameters")
        t_mult = st.slider("Traffic Multiplier",  0.5, 4.0, 1.0, 0.1, key="sim_t")
        l_mult = st.slider("Latency Multiplier",  0.5, 4.0, 1.0, 0.1, key="sim_l")
        p_mult = st.slider("Packet Loss Mult.",   0.5, 4.0, 1.0, 0.1, key="sim_p")
        n_fail = st.slider("Random Link Failures", 0, max(1, G.number_of_edges()), 0, key="sim_f")
        seed_v = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, step=1)

        st.markdown("---")
        sa, sb = st.columns(2)
        if sa.button("Run scenario", width="stretch"):
            st.session_state.snap = snapshot(G)
            apply_stress(G, t_mult, l_mult, p_mult)
            apply_random_failures(G, n_fail=n_fail, seed=int(seed_v))
            if st.session_state.model:
                predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
                compute_ai_weights(G)
            st.success("Simulation applied!")
            st.rerun()

        if sb.button("Restore", width="stretch"):
            if st.session_state.snap:
                restore(G, st.session_state.snap)
                restore_all_failures(G)
                if st.session_state.model:
                    predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
                    compute_ai_weights(G)
                st.success("Network restored.")
            else:
                restore_all_failures(G)
                if st.session_state.model:
                    predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
                    compute_ai_weights(G)
                st.info("Failures cleared.")
            st.rerun()

        st.markdown("---")
        st.markdown("### Manual Link Control")
        edge_opts_sim = [f"{u}↔{v}" for u,v in G.edges()]
        if edge_opts_sim:
            sel_sim = st.selectbox("Select link", edge_opts_sim, key="sim_link_sel")
            p_sim = sel_sim.split("↔")
            scol1, scol2 = st.columns(2)
            if scol1.button("Fail link", width="stretch"):
                ok,msg = set_link_failed(G, p_sim[0], p_sim[1], True)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
            if scol2.button("Restore link", width="stretch"):
                ok,msg = set_link_failed(G, p_sim[0], p_sim[1], False)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()

        st.markdown("---")
        st.markdown("### Post-Simulation Health")
        h = network_health(G)
        st.metric("Health Score",   f"{h['health_score']*100:.0f}%")
        st.metric("Avg Risk",       f"{h['avg_risk']*100:.1f}%")
        st.metric("Failed Links",   h["failed_links"])
        st.metric("High Risk Links", h["high_risk"])

    with sr:
        st.markdown("### Simulated Topology")
        st.plotly_chart(network_figure(G, title="Simulated Network State"),
                        width="stretch", key="sim_network")
        st.plotly_chart(health_gauge(network_health(G)["health_score"]),
                        width="stretch", key="sim_gauge")
        df_sim = get_edge_df(G)
        if not df_sim.empty:
            simulation_inventory = prepare_link_inventory(df_sim)
            show = [c for c in ["operating_state","source","target","traffic_load","latency","packet_loss",
                                  "failure_prob","routing_risk_percent","risk_source","telemetry_cell",
                                  "risk_label","failed"] if c in simulation_inventory.columns]
            st.markdown("### Live Edge Status")
            st.dataframe(simulation_inventory[show], width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t_analytics:
    render_section_header(st, "Network insights", "Risk, model, and route evidence")
    h = network_health(G)
    ac1,ac2,ac3,ac4,ac5 = st.columns(5)
    ac1.metric("Total Links",   h["total_links"])
    ac2.metric("Failed",        h["failed_links"])
    ac3.metric("High Risk",     h["high_risk"])
    ac4.metric("Medium Risk",   h["medium_risk"])
    ac5.metric("Avg Routing Risk", f"{h['avg_risk']*100:.1f}%")

    evidence_left, evidence_right = st.columns([1.25, 1])
    with evidence_left:
        render_section_header(st, "Observability", "Telemetry pipeline")
        render_status_panel(st, [
            ("Application export", "Enabled" if OBSERVABILITY.enabled else "Opt-in / disabled"),
            ("Protocol", "OTLP over HTTP"),
            ("Trace backend", "Jaeger · local memory"),
            ("Metrics backend", "Prometheus · local volume"),
        ])
        ol1, ol2 = st.columns(2)
        ol1.link_button("Open Jaeger", "http://localhost:16686", width="stretch")
        ol2.link_button("Open Prometheus", "http://localhost:9090", width="stretch")
        st.caption("Local endpoints require the observability Compose stack to be running.")
    with evidence_right:
        render_section_header(st, "Experiment evidence", "Controlled Mininet result")
        mininet_evidence = get_mininet_evidence()
        if mininet_evidence:
            comparison = mininet_evidence["comparison"]
            ev1, ev2 = st.columns(2)
            ev1.metric("Throughput gain", f"{comparison['throughput_gain_percent']:.1f}%")
            ev2.metric("RTT reduction", f"{comparison['rtt_reduction_percent']:.1f}%")
            render_status_panel(st, [
                ("Repetitions", str(mininet_evidence["aggregates"]["traditional"]["runs"])),
                ("Packet-loss change", f"{comparison['packet_loss_reduction_points']:.1f} points"),
                ("Evidence class", "Emulator / controlled"),
            ])
        else:
            st.info("Run the Phase 4 Mininet experiment to populate this evidence panel.")

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(risk_pie(h), width="stretch", key="an_pie")
    with r2:
        if st.session_state.metrics:
            st.plotly_chart(feature_importance_fig(st.session_state.metrics["feature_importance"]),
                            width="stretch", key="an_fi")
        else:
            st.info("Train a model (ML Model tab) to see feature importance.")

    st.markdown("---")
    if st.session_state.metrics:
        ac_l, ac_r = st.columns(2)
        with ac_l:
            st.plotly_chart(confusion_matrix_fig(st.session_state.metrics["cm"]),
                            width="stretch", key="an_cm")
        with ac_r:
            if st.session_state.trad and st.session_state.ai:
                st.plotly_chart(route_comparison_fig(st.session_state.trad, st.session_state.ai),
                                width="stretch", key="an_compare")
            else:
                st.info("Compute routes (Routing tab) to see comparison chart.")
    else:
        st.info("Train a model first to unlock all analytics panels.")

    st.markdown("---")
    st.markdown("### Full Link Table")
    df_a = get_edge_df(G)
    if not df_a.empty:
        analytics_inventory = prepare_link_inventory(df_a)
        cols_a = [c for c in ["operating_state","source","target","distance","bandwidth","latency",
                               "packet_loss","traffic_load","failure_prob","routing_risk_percent",
                               "risk_source","telemetry_cell","risk_label","ai_weight","failed"]
                  if c in analytics_inventory.columns]
        st.dataframe(analytics_inventory[cols_a], width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Algorithm Complexity Reference")
    st.markdown("""
| Component | Algorithm | Time Complexity | Space Complexity |
|-----------|-----------|-----------------|-----------------|
| Traditional Routing | Dijkstra (distance) | O((V+E) log V) | O(V+E) |
| AI-Enhanced Routing | Dijkstra (ai_weight) | O((V+E) log V) | O(V+E) |
| Traditional Routing | Bellman-Ford (distance) | O(V·E) | O(V) |
| AI-Enhanced Routing | Bellman-Ford (ai_weight) | O(V·E) | O(V) |
| All-Pairs Routing | Floyd-Warshall (distance) | O(V³) | O(V²) |
| AI All-Pairs | Floyd-Warshall (ai_weight) | O(V³) | O(V²) |
| Failure Prediction | Random Forest (inference) | O(k·log n) | O(k) |
| Model Training | Random Forest (fit) | O(n·d·k·log n) | O(k·d) |

V = routers, E = links, k = trees, n = training samples, d = max tree depth
    """)
