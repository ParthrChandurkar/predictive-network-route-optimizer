"""
analytics.py — Plotly-based chart builders.
Requires: pip install plotly
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from utils import RISK_COLORS, ALGO_COLORS

BG   = "#0f172a"
GRID = "#1e293b"
TEXT = "#e2e8f0"


def _edge_risk(attributes):
    return attributes.get("routing_risk", attributes.get("failure_prob", 0.0))

def _base(**kw):
    return dict(paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color=TEXT), margin=dict(l=10,r=10,t=50,b=10), **kw)


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
            col,w,dash = ALGO_COLORS["Failed"], 2, "dash"
        elif fwd in trad_set or rev in trad_set:
            col,w,dash = ALGO_COLORS["Traditional"], 5, "solid"
        elif fwd in ai_set or rev in ai_set:
            col,w,dash = ALGO_COLORS["AI-Enhanced"],  5, "solid"
        else:
            col  = RISK_COLORS.get(d.get("risk_label","Low"), RISK_COLORS["Unknown"])
            w,dash = 2,"solid"
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
                             marker=dict(size=24,color="#1e293b",line=dict(width=2.5,color="#38bdf8"))))
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
                      legend=dict(bgcolor=BG,font=dict(color=TEXT,size=11)),
                      xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                      height=480, **_base())
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
