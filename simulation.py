"""
simulation.py — Apply traffic/latency/packet-loss stress + random failures.
All mutations are in-place on G. Always call predict_graph_edges after running.
"""
import networkx as nx
import numpy as np


def snapshot(G: nx.Graph) -> dict:
    """Save edge attrs so we can restore later. Returns {(u,v): attr_dict}."""
    return {(u, v): dict(data) for u, v, data in G.edges(data=True)}


def restore(G: nx.Graph, snap: dict):
    """Restore edge attrs from a snapshot."""
    for (u, v), attrs in snap.items():
        if G.has_edge(u, v):
            G[u][v].clear()
            G[u][v].update(attrs)


def apply_stress(G: nx.Graph,
                 traffic_mult:  float = 1.0,
                 latency_mult:  float = 1.0,
                 pkt_loss_mult: float = 1.0):
    """Scale traffic_load, latency, packet_loss by the given multipliers."""
    for u, v, data in G.edges(data=True):
        G[u][v]["traffic_load"] = round(min(100.0,  data.get("traffic_load", 50) * traffic_mult),  2)
        G[u][v]["latency"]      = round(min(1000.0, data.get("latency",      20) * latency_mult),  2)
        G[u][v]["packet_loss"]  = round(min(20.0,   data.get("packet_loss",   1) * pkt_loss_mult), 2)


def apply_random_failures(G: nx.Graph, n_fail: int = 0, seed: int = 42):
    """Mark up to n_fail random non-failed links as failed."""
    if n_fail <= 0:
        return []
    rng   = np.random.default_rng(seed)
    edges = [(u, v) for u, v, data in G.edges(data=True) if not data.get("failed", False)]
    n     = min(n_fail, len(edges))
    if n == 0:
        return []
    idxs  = rng.choice(len(edges), size=n, replace=False)
    failed_edges = []
    for i in idxs:
        u, v = edges[int(i)]
        G[u][v]["failed"] = True
        failed_edges.append((u, v))
    return failed_edges


def restore_all_failures(G: nx.Graph):
    """Un-fail every link."""
    for u, v in G.edges():
        G[u][v]["failed"] = False


def network_health(G: nx.Graph) -> dict:
    """Aggregate health statistics across all edges."""
    edges = list(G.edges(data=True))
    total = len(edges)
    if total == 0:
        return {"total_links": 0, "failed_links": 0, "high_risk": 0,
                "medium_risk": 0, "low_risk": 0, "avg_risk": 0.0, "health_score": 1.0}
    failed = sum(1 for _, _, d in edges if d.get("failed", False))
    high   = sum(1 for _, _, d in edges if d.get("risk_label") == "High")
    medium = sum(1 for _, _, d in edges if d.get("risk_label") == "Medium")
    low    = sum(1 for _, _, d in edges if d.get("risk_label") == "Low")
    avg_r  = float(np.mean([
        d.get("routing_risk", d.get("failure_prob", 0.0)) for _, _, d in edges
    ]))
    score  = round(max(0.0, 1.0 - avg_r - (failed / max(total, 1)) * 0.4), 4)
    return {
        "total_links": total, "failed_links": failed,
        "high_risk": high, "medium_risk": medium, "low_risk": low,
        "avg_risk": round(avg_r, 4), "health_score": score,
    }
