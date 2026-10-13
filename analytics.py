"""
analytics.py — Plotly-based chart builders.
Requires: pip install plotly
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from utils import RISK_COLORS, ALGO_COLORS

BG   = "#0b1220"
GRID = "#23334d"
TEXT = "#e8eef8"


def _edge_risk(attributes):
    return attributes.get("routing_risk", attributes.get("failure_prob", 0.0))

def _base(**kw):
    return dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        hoverlabel=dict(bgcolor="#111b2d", bordercolor="#2d4261", font_color=TEXT),
        margin=dict(l=16, r=16, t=54, b=16),
        **kw,
    )


def network_figure(G, trad_edges=None, ai_edges=None, title="Network Topology"):
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        raw = nx.spring_layout(G, seed=42)
        pos = {n: (float(x), float(y)) for n, (x,y) in raw.items()}

    trad_set = {(u,v) for u,v in (trad_edges or [])}
    ai_set   = {(u,v) for u,v in (ai_edges   or [])}
    traces   = []

    for u, v, d in G.edges(data=True):
        x0,y0 = pos.get(u,(0,0));  x1,y1 = pos.get(v,(0,0))
        fwd,rev = (u,v),(v,u)
        if d.get("failed"):
            col,w,dash = ALGO_COLORS["Failed"], 3, "dash"
        elif fwd in trad_set or rev in trad_set:
            col,w,dash = ALGO_COLORS["Traditional"], 5, "solid"
        elif fwd in ai_set or rev in ai_set:
            col,w,dash = ALGO_COLORS["AI-Enhanced"],  5, "solid"
        else:
            col  = RISK_COLORS.get(d.get("risk_label","Low"), RISK_COLORS["Unknown"])
            w,dash = 2.5,"solid"
        route_risk = _edge_risk(d)
        ht = (f"<b>{u}↔{v}</b><br>Dist:{d.get('distance',0)}km  BW:{d.get('bandwidth',0)}Mbps<br>"
              f"Latency:{d.get('latency',0)}ms  PktLoss:{d.get('packet_loss',0)}%<br>"
              f"Traffic:{d.get('traffic_load',0)}%  RouteRisk:{route_risk:.2%}<br>"
              f"Source:{d.get('risk_source','synthetic_default')}  Risk:{d.get('risk_label','Low')}<br>"
              f"{'❌FAILED' if d.get('failed') else '✅Active'}")
        traces.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                                 line=dict(width=w,color=col,dash=dash),
                                 hoverinfo="text",hovertext=ht,showlegend=False))

    nx_l,ny_l,nlbl,nhov = [],[],[],[]
    for nd in G.nodes():
        x,y = pos.get(nd,(0,0))
        nx_l.append(x); ny_l.append(y); nlbl.append(nd)
        nhov.append(f"<b>{nd}</b><br>Degree:{G.degree(nd)}")
    traces.append(go.Scatter(x=nx_l,y=ny_l,mode="markers+text",text=nlbl,
                             textposition="top center",
                             textfont=dict(size=13,color=TEXT,family="monospace"),
                             hoverinfo="text",hovertext=nhov,showlegend=False,
                             marker=dict(size=29,color="#0b1526",line=dict(width=2.5,color="#22d3ee"))))
    for lbl,col in [("Traditional Route",ALGO_COLORS["Traditional"]),
                    ("AI-Enhanced Route", ALGO_COLORS["AI-Enhanced"]),
                    ("Failed Link",       ALGO_COLORS["Failed"]),
                    ("Low Risk",          RISK_COLORS["Low"]),
                    ("Medium Risk",       RISK_COLORS["Medium"]),
                    ("High Risk",         RISK_COLORS["High"])]:
        traces.append(go.Scatter(x=[None],y=[None],mode="lines",
                                 line=dict(color=col,width=3),name=lbl))
    fig = go.Figure(data=traces)
    fig.update_layout(title=dict(text=title,font=dict(size=15,color=TEXT)),
                      legend=dict(
                          bgcolor="rgba(7,11,20,.78)", font=dict(color=TEXT,size=10),
                          orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                      ),
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      height=560, hovermode="closest", uirevision="network-topology", **_base())
    return fig


def confusion_matrix_fig(cm):
    labels = ["No Failure","Failure"]
    fig = px.imshow(np.array(cm),x=labels,y=labels,color_continuous_scale="Blues",
                    text_auto=True,labels=dict(x="Predicted",y="Actual"),title="Confusion Matrix")
    fig.update_traces(textfont_size=18)
    fig.update_layout(coloraxis_showscale=False,height=320,**_base())
    return fig


def feature_importance_fig(imp):
    feats = list(imp.keys())
    vals  = [imp[f] for f in feats]
    cols  = ["#38bdf8","#34d399","#f59e0b","#f472b6"]
    fig = go.Figure(go.Bar(x=vals,y=feats,orientation="h",
                           marker=dict(color=cols[:len(feats)]),
                           text=[f"{v:.3f}" for v in vals],textposition="outside"))
    fig.update_layout(title="Feature Importance",
                      xaxis=dict(showgrid=True,gridcolor=GRID),
                      yaxis=dict(showgrid=False),height=280,**_base())
    return fig


def route_comparison_fig(trad, ai):
    if not trad["found"] or not ai["found"]:
        fig = go.Figure()
        fig.update_layout(title="No valid routes to compare",height=300,**_base())
        return fig
    metrics = ["Distance (km)","Hops","Risky Links","Avg Risk ×100"]
    tv = [trad["dist_cost"],trad["hops"],trad["risky_links"],round(trad["avg_risk"]*100,2)]
    av = [ai["dist_cost"],  ai["hops"],  ai["risky_links"],  round(ai["avg_risk"]*100,  2)]
    fig = go.Figure([
        go.Bar(name="Traditional",x=metrics,y=tv,marker_color=ALGO_COLORS["Traditional"]),
        go.Bar(name="AI-Enhanced", x=metrics,y=av,marker_color=ALGO_COLORS["AI-Enhanced"]),
    ])
    fig.update_layout(barmode="group",title="Route Comparison",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True,gridcolor=GRID),
                      legend=dict(bgcolor=BG),height=320,**_base())
    return fig


def health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score*100,1),
        title={"text":"Network Health","font":{"color":TEXT,"size":13}},
        number={"suffix":"%","font":{"color":TEXT,"size":30}},
        gauge={"axis":{"range":[0,100],"tickcolor":TEXT},
               "bar":{"color":"#38bdf8"},
               "steps":[{"range":[0,40],"color":"#7f1d1d"},
                         {"range":[40,70],"color":"#78350f"},
                         {"range":[70,100],"color":"#14532d"}]},
    ))
    fig.update_layout(paper_bgcolor=BG,font=dict(color=TEXT),
                      margin=dict(l=20,r=20,t=55,b=20),height=240)
    return fig


def risk_pie(health):
    labels = ["Low","Medium","High"]
    values = [health.get("low_risk",0),health.get("medium_risk",0),health.get("high_risk",0)]
    if sum(values) == 0: values = [1,0,0]
    fig = go.Figure(go.Pie(labels=labels,values=values,
                           marker=dict(colors=[RISK_COLORS["Low"],RISK_COLORS["Medium"],RISK_COLORS["High"]]),
                           hole=0.45,textfont=dict(color="#fff")))
    fig.update_layout(title="Risk Distribution",
                      legend=dict(bgcolor=BG,font=dict(color=TEXT)),
                      height=280,**_base())
    return fig


def lte_degradation_timeline(frame, scored=None):
    """Plot labeled anomaly rate or model degradation score by time-of-day slot."""
    value_column = None
    title = "LTE degradation timeline"
    source = scored if scored is not None and "degradation_score" in scored else frame
    if "degradation_score" in source:
        value_column = "degradation_score"
        title = "Predicted degradation score by time of day"
    elif "unusual_behavior" in source:
        value_column = "unusual_behavior"
        title = "Observed unusual-behavior rate by time of day"

    fig = go.Figure()
    if value_column is None or source.empty:
        fig.update_layout(title="No timeline signal available", height=300, **_base())
        return fig

    timeline = (
        source.groupby("time_of_day", as_index=False)[value_column]
        .mean()
        .sort_values("time_of_day")
    )
    timeline[value_column] = timeline[value_column] * 100
    fig.add_trace(
        go.Scatter(
            x=timeline["time_of_day"],
            y=timeline[value_column],
            mode="lines",
            line=dict(color="#22d3ee", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.10)",
            hovertemplate="%{x}<br>Signal: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=320,
        xaxis=dict(title="Time of day", gridcolor=GRID, rangeslider=dict(visible=True, thickness=.08)),
        yaxis=dict(title="Signal (%)", gridcolor=GRID, rangemode="tozero"),
        **_base(),
    )
    return fig


def pattern_rate_figure(records, category: str, title: str, baseline_rate: float):
    """Visualize grouped unusual-label rates from the Phase 7 evidence report."""
    labels = [str(item[category]) for item in records]
    rates = [item["unusual_rate"] * 100 for item in records]
    low = [item["wilson_95_low"] * 100 for item in records]
    high = [item["wilson_95_high"] * 100 for item in records]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color="#22d3ee",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[upper - rate for upper, rate in zip(high, rates)],
                arrayminus=[rate - lower for rate, lower in zip(rates, low)],
                color="#64748b",
                thickness=1,
            ),
            hovertemplate="%{x}<br>Unusual rate: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=baseline_rate * 100,
        line_dash="dash",
        line_color="#fbbf24",
        annotation_text="dataset baseline",
    )
    fig.update_layout(
        title=title,
        height=350,
        xaxis=dict(title=category.replace("_", " ").title(), tickangle=-35),
        yaxis=dict(title="Unusual-label rate (%)", gridcolor=GRID, rangemode="tozero"),
        **_base(),
    )
    return fig


def feature_shift_figure(records):
    """Plot standardized marginal KPI shifts without implying causality."""
    ordered = sorted(records, key=lambda item: item["standardized_mean_difference"])
    fig = go.Figure(
        go.Bar(
            x=[item["standardized_mean_difference"] for item in ordered],
            y=[item["feature"] for item in ordered],
            orientation="h",
            marker_color=[
                "#fb7185" if item["standardized_mean_difference"] < 0 else "#34d399"
                for item in ordered
            ],
            hovertemplate="%{y}<br>Standardized difference: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#64748b")
    fig.update_layout(
        title="Marginal KPI shift · unusual minus normal",
        height=420,
        xaxis=dict(title="Standardized mean difference", gridcolor=GRID),
        yaxis=dict(title=""),
        **_base(),
    )
    return fig
