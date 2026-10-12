"""
app.py — AI-Based Network Routing Optimizer  (Enhanced)
Tabs: Overview | ML Model | Builder | Routing | Algorithm Comparison | Simulation | Analytics | Novelty
"""
import streamlit as st

st.set_page_config(
    page_title="Predictive NetRoute",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main .block-container{padding-top:1rem;max-width:1400px;}
[data-testid="metric-container"]{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;}
[data-testid="metric-container"] label{color:#94a3b8!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.08em;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#38bdf8!important;font-size:24px!important;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{background:#0f172a;border-bottom:1px solid #334155;}
.stTabs [data-baseweb="tab"]{color:#94a3b8;font-size:12px;font-weight:600;padding:8px 14px;letter-spacing:.04em;text-transform:uppercase;}
.stTabs [aria-selected="true"]{color:#38bdf8!important;border-bottom:2px solid #38bdf8;background:transparent!important;}
[data-testid="stSidebar"]{background:#0f172a;border-right:1px solid #1e293b;}
h1{color:#f1f5f9!important;font-weight:700;}
h2{color:#e2e8f0!important;font-weight:600;font-size:1.1rem!important;}
h3{color:#cbd5e1!important;font-size:.95rem!important;}
.stButton button{background:#1e40af!important;color:#e2e8f0!important;border:none;border-radius:6px;font-weight:600;}
.stButton button:hover{background:#2563eb!important;}
hr{border-color:#1e293b;}
.novelty-card{background:#1e293b;border-left:4px solid #38bdf8;border-radius:8px;padding:16px 20px;margin-bottom:14px;}
.gap-card{background:#1e293b;border-left:4px solid #f59e0b;border-radius:8px;padding:14px 18px;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

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
                              route_comparison_fig, health_gauge, risk_pie)
from telemetry       import TelemetryValidationError, load_lte_dataset
from modeling.lte_model import load_lte_risk_model, predict_lte_risk
from integration     import (apply_cell_risk_to_graph, build_demo_edge_cell_mapping,
                             cell_risk_snapshot)
from settings        import PHASE2_REPORT_PATH
from utils           import RISK_COLORS
from observability   import configure_observability
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np

configure_observability()

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


@st.cache_resource
def get_lte_risk_model():
    return load_lte_risk_model()


def refresh_predictions():
    predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
    compute_ai_weights(G)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 NetRoute AI")
    st.markdown("---")
    st.markdown("### Model Status")
    if st.session_state.model is not None:
        st.success("✅ Model loaded")
        if st.session_state.metrics:
            m = st.session_state.metrics
            st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
            st.metric("F1 Score", f"{m['f1']*100:.1f}%")
    else:
        st.warning("⚠️ No model trained yet\nGo to **ML Model** tab to train.")

    st.markdown("---")
    if st.button("🔄 Reset Topology", use_container_width=True):
        st.session_state.G = build_default_topology()
        G = st.session_state.G
        if st.session_state.model:
            predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
            compute_ai_weights(G)
        for k in ["trad","ai","bf","bf_ai","fw","fw_ai"]:
            st.session_state[k] = None
        st.rerun()

    st.markdown("---")
    info   = graph_summary(G)
    health = network_health(G)
    st.markdown("### Network Summary")
    c1, c2 = st.columns(2)
    c1.metric("Routers", info["routers"])
    c2.metric("Links",   info["links"])
    st.metric("Health", f"{health['health_score']*100:.0f}%")
    if health["failed_links"]:
        st.error(f"⚠️ {health['failed_links']} link(s) FAILED")
    st.caption(f"Connected: {'✅' if info['connected'] else '❌'}")
    st.markdown("---")
    st.caption("DAA + ML Final Year Project — VIIT 2024-25")


# ── Tabs ──────────────────────────────────────────────────────────────────────
(t_overview, t_model, t_builder, t_routing,
 t_compare, t_sim, t_analytics) = st.tabs([
    "📊 Overview", "🤖 ML Model", "🏗️ Builder", "🗺️ Routing",
    "⚖️ Algorithm Comparison", "⚡ Simulation", "📈 Analytics"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:
    st.markdown("## Predictive Network Route Optimizer")
    st.markdown(
        "Intelligent network dashboard combining **Dijkstra**, **Bellman-Ford**, and "
        "**Floyd-Warshall** path-finding with **ML-predicted link failure probabilities** "
        "to route traffic around risky links."
    )
    health = network_health(G)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Routers",      G.number_of_nodes())
    c2.metric("Links",        G.number_of_edges())
    c3.metric("Failed Links", health["failed_links"])
    c4.metric("High Risk",    health["high_risk"])
    c5.metric("Health Score", f"{health['health_score']*100:.0f}%")

    st.markdown("---")
    col_g, col_h = st.columns([3, 1])
    with col_g:
        trad_edges = st.session_state.trad["edges"] if st.session_state.trad and st.session_state.trad["found"] else []
        ai_edges   = st.session_state.ai["edges"]   if st.session_state.ai   and st.session_state.ai["found"]   else []
        st.plotly_chart(network_figure(G, trad_edges, ai_edges), use_container_width=True, key="ov_network")
    with col_h:
        st.plotly_chart(health_gauge(health["health_score"]), use_container_width=True, key="ov_gauge")
        st.plotly_chart(risk_pie(health), use_container_width=True, key="ov_pie")

    st.markdown("---")
    st.markdown("### Link Risk Table")
    df = get_edge_df(G)
    if not df.empty:
        show_cols = [c for c in ["source","target","distance","bandwidth","latency",
                                  "packet_loss","traffic_load","failure_prob","routing_risk",
                                  "risk_source","telemetry_cell","risk_label","failed"]
                     if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with t_model:
    st.markdown("## Machine Learning — Link Failure Prediction")
    st.markdown(
        "Trains a **RandomForestClassifier** on synthetic link telemetry. "
        "The predicted failure probability is used as an edge penalty in AI-Enhanced routing."
    )

    with st.expander("Real LTE telemetry — rebuild preview", expanded=False):
        st.caption(
            "Upload the Kaggle training or test CSV to validate and normalize it. "
            "The displayed baseline detects unusual LTE behavior; routing integration comes next."
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
                st.dataframe(st.session_state.lte_df.head(20), use_container_width=True)
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
                            cell_risks = cell_risk_snapshot(scored, time_slot)
                            mapping = build_demo_edge_cell_mapping(G, cell_risks)
                            summary = apply_cell_risk_to_graph(G, cell_risks, mapping)
                            compute_ai_weights(G)
                            st.session_state.lte_routing_summary = summary
                        st.success(
                            f"Applied {summary['mapped_cells']} cell scores to "
                            f"{summary['mapped_edges']} links for {time_slot}."
                        )

    left, right = st.columns([1, 2])

    with left:
        st.markdown("### Train Model")
        n_samples = st.slider("Dataset size", 400, 2000, 1700, 100)
        if st.button("🚀 Generate Data & Train", use_container_width=True):
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
            st.dataframe(df.head(8), use_container_width=True)
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
                st.plotly_chart(confusion_matrix_fig(m["cm"]), use_container_width=True, key="ml_cm")
            with r2:
                st.plotly_chart(feature_importance_fig(m["feature_importance"]), use_container_width=True, key="ml_fi")
            with st.expander("📄 Full Classification Report"):
                st.code(m["report"])
        else:
            st.info("ℹ️ Click **Generate Data & Train** on the left to train the model.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NETWORK BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with t_builder:
    st.markdown("## Network Topology Builder")
    bl, br = st.columns([1, 2])

    with bl:
        st.markdown("### Routers")
        r_name = st.text_input("Router name", placeholder="e.g. R9")
        ra, rb = st.columns(2)
        if ra.button("➕ Add", use_container_width=True):
            if r_name.strip():
                ok, msg = add_router(G, r_name.strip())
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
            else:
                st.warning("Enter a router name.")
        if rb.button("🗑️ Remove", use_container_width=True):
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
            if st.button("➕ Add Link", use_container_width=True):
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
            if st.button("🗑️ Remove Link", use_container_width=True):
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
            if st.button("💾 Update Link", use_container_width=True):
                ok, msg = update_link(G, u2, v2, dict(distance=e_dist, bandwidth=e_bw,
                    latency=e_lat, packet_loss=e_pkt, traffic_load=e_traf))
                (st.success if ok else st.error)(msg)
                if ok and st.session_state.model:
                    refresh_predictions(); st.rerun()

    with br:
        st.markdown("### Current Topology")
        st.plotly_chart(network_figure(G, title="Network Topology"), use_container_width=True, key="bld_network")
        st.markdown("### Edge Table")
        df_e = get_edge_df(G)
        if not df_e.empty:
            cols = [c for c in ["source","target","distance","bandwidth","latency",
                                 "packet_loss","traffic_load","failure_prob","routing_risk",
                                 "risk_source","telemetry_cell","risk_label","failed"]
                    if c in df_e.columns]
            st.dataframe(df_e[cols], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ROUTING (original 2-algo view)
# ══════════════════════════════════════════════════════════════════════════════
with t_routing:
    st.markdown("## Routing Engine")
    st.markdown(
        "**Traditional Dijkstra** minimises distance only. "
        "**AI-Enhanced Dijkstra** adds `ai_weight = distance + routing_risk × penalty` "
        "to steer traffic around high-risk links."
    )
    nodes = sorted(G.nodes())
    if len(nodes) < 2:
        st.warning("Add at least 2 routers.")
    else:
        rc1, rc2, rc3 = st.columns([1,1,2])
        src     = rc1.selectbox("Source",      nodes, key="rt_src")
        dst     = rc2.selectbox("Destination", nodes, key="rt_dst", index=min(1,len(nodes)-1))
        penalty = rc3.slider("AI Penalty Factor", 50, 500, 200, 10)

        if st.button("🚀 Compute Routes", use_container_width=True):
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
                            use_container_width=True, key="rt_network")
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
                st.markdown("#### 🟢 AI-Enhanced Dijkstra")
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

            st.markdown("---")
            st.plotly_chart(route_comparison_fig(trad, ai), use_container_width=True, key="rt_compare")

            if trad["found"] and ai["found"]:
                risk_delta = trad["avg_risk"] - ai["avg_risk"]
                dist_delta = ai["dist_cost"] - trad["dist_cost"]
                if risk_delta > 0.005:
                    st.info(f"✅ AI-Enhanced route reduces average risk by **{risk_delta*100:.1f}%** "
                            f"at an extra **{dist_delta:.0f} km** distance overhead.")
                elif risk_delta < -0.005:
                    st.warning("⚠️ Traditional route has lower risk in current topology state.")
                else:
                    st.info("ℹ️ Both algorithms chose paths with similar risk levels.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALGORITHM COMPARISON  ★ NEW TAB ★
# ══════════════════════════════════════════════════════════════════════════════
with t_compare:
    st.markdown("## ⚖️ Algorithm Comparison: Dijkstra vs Bellman-Ford vs Floyd-Warshall")
    st.markdown(
        "Run all **6 routing algorithms** (3 traditional + 3 AI-enhanced) simultaneously "
        "and compare their paths, distances, risk levels, and execution times."
    )

    if not st.session_state.model:
        st.warning("⚠️ Please train the ML model first (ML Model tab) for AI-enhanced results.")

    nodes = sorted(G.nodes())
    if len(nodes) < 2:
        st.warning("Add at least 2 routers in the Builder tab.")
    else:
        cc1, cc2, cc3 = st.columns([1,1,2])
        cmp_src     = cc1.selectbox("Source",      nodes, key="cmp_src")
        cmp_dst     = cc2.selectbox("Destination", nodes, key="cmp_dst", index=min(1,len(nodes)-1))
        cmp_penalty = cc3.slider("AI Penalty Factor", 50, 500, 200, 10, key="cmp_pen")

        if st.button("🚀 Run All 6 Algorithms", use_container_width=True, key="cmp_run"):
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
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

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
                    st.plotly_chart(fig_dist, use_container_width=True, key="cmp_dist")

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
                    st.plotly_chart(fig_risk, use_container_width=True, key="cmp_risk")

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
                st.plotly_chart(fig_time, use_container_width=True, key="cmp_time")

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
                st.dataframe(df_matrix, use_container_width=True)

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
            st.dataframe(pd.DataFrame(theory_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with t_sim:
    st.markdown("## Simulation Engine")
    st.markdown("Apply network stress conditions. After running, edge failure probabilities are recalculated.")

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
        if sa.button("▶️ Run Simulation", use_container_width=True):
            st.session_state.snap = snapshot(G)
            apply_stress(G, t_mult, l_mult, p_mult)
            apply_random_failures(G, n_fail=n_fail, seed=int(seed_v))
            if st.session_state.model:
                predict_graph_edges(st.session_state.model, st.session_state.scaler, G)
                compute_ai_weights(G)
            st.success("Simulation applied!")
            st.rerun()

        if sb.button("🔄 Restore", use_container_width=True):
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
            if scol1.button("🔴 Fail",    use_container_width=True):
                ok,msg = set_link_failed(G, p_sim[0], p_sim[1], True)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
            if scol2.button("🟢 Restore", use_container_width=True):
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
                        use_container_width=True, key="sim_network")
        st.plotly_chart(health_gauge(network_health(G)["health_score"]),
                        use_container_width=True, key="sim_gauge")
        df_sim = get_edge_df(G)
        if not df_sim.empty:
            show = [c for c in ["source","target","traffic_load","latency","packet_loss",
                                  "failure_prob","routing_risk","risk_source","telemetry_cell",
                                  "risk_label","failed"] if c in df_sim.columns]
            st.markdown("### Live Edge Status")
            st.dataframe(df_sim[show], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with t_analytics:
    st.markdown("## Analytics Dashboard")
    h = network_health(G)
    ac1,ac2,ac3,ac4,ac5 = st.columns(5)
    ac1.metric("Total Links",   h["total_links"])
    ac2.metric("Failed",        h["failed_links"])
    ac3.metric("High Risk",     h["high_risk"])
    ac4.metric("Medium Risk",   h["medium_risk"])
    ac5.metric("Avg Routing Risk", f"{h['avg_risk']*100:.1f}%")

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(risk_pie(h), use_container_width=True, key="an_pie")
    with r2:
        if st.session_state.metrics:
            st.plotly_chart(feature_importance_fig(st.session_state.metrics["feature_importance"]),
                            use_container_width=True, key="an_fi")
        else:
            st.info("Train a model (ML Model tab) to see feature importance.")

    st.markdown("---")
    if st.session_state.metrics:
        ac_l, ac_r = st.columns(2)
        with ac_l:
            st.plotly_chart(confusion_matrix_fig(st.session_state.metrics["cm"]),
                            use_container_width=True, key="an_cm")
        with ac_r:
            if st.session_state.trad and st.session_state.ai:
                st.plotly_chart(route_comparison_fig(st.session_state.trad, st.session_state.ai),
                                use_container_width=True, key="an_compare")
            else:
                st.info("Compute routes (Routing tab) to see comparison chart.")
    else:
        st.info("Train a model first to unlock all analytics panels.")

    st.markdown("---")
    st.markdown("### Full Link Table")
    df_a = get_edge_df(G)
    if not df_a.empty:
        cols_a = [c for c in ["source","target","distance","bandwidth","latency",
                               "packet_loss","traffic_load","failure_prob","routing_risk",
                               "risk_source","telemetry_cell","risk_label","ai_weight","failed"]
                  if c in df_a.columns]
        st.dataframe(df_a[cols_a], use_container_width=True)

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
