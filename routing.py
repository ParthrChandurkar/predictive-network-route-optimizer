"""
routing.py — Traditional Dijkstra + AI-Enhanced Dijkstra + Bellman-Ford + Floyd-Warshall.
Novel contribution: AI weight penalty layer applied across ALL algorithms uniformly,
allowing a fair apples-to-apples comparison of how each algorithm responds to
ML-predicted failure probabilities.
"""
import networkx as nx
import numpy as np
import time

from settings import DEFAULT_ROUTE_PENALTY

PENALTY_DEFAULT = DEFAULT_ROUTE_PENALTY


def _edge_risk(data: dict) -> float:
    """Prefer an explicitly integrated risk score over the synthetic baseline."""
    return float(data.get("routing_risk", data.get("failure_prob", 0.0)))


def compute_ai_weights(G: nx.Graph, penalty: float = PENALTY_DEFAULT):
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    for u, v, data in G.edges(data=True):
        if data.get("failed", False):
            G[u][v]["ai_weight"] = 1e9
        else:
            G[u][v]["ai_weight"] = data.get("distance", 1) + _edge_risk(data) * penalty


def _active_graph(G: nx.Graph) -> nx.Graph:
    """Return a copy of G containing all nodes and only non-failed edges."""
    H = G.__class__()
    H.graph.update(G.graph)
    H.add_nodes_from((node, data.copy()) for node, data in G.nodes(data=True))
    H.add_edges_from(
        (u, v, data.copy())
        for u, v, data in G.edges(data=True)
        if not data.get("failed", False)
    )
    return H


def _path_metrics(G: nx.Graph, path: list, algo_name: str, elapsed_ms: float) -> dict:
    edges, risks, risky = [], [], 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        d = G[u][v] if G.has_edge(u, v) else {}
        prob = _edge_risk(d)
        risks.append(prob)
        if d.get("risk_label", "Low") in ("Medium", "High"):
            risky += 1
        edges.append((u, v))
    dist_cost = sum(G[u][v].get("distance", 0) for u, v in edges if G.has_edge(u, v))
    return {
        "edges":       edges,
        "path_str":    " → ".join(path),
        "hops":        len(path) - 1,
        "dist_cost":   round(dist_cost, 2),
        "risky_links": risky,
        "avg_risk":    round(float(np.mean(risks)) if risks else 0.0, 4),
        "found":       True,
        "algorithm":   algo_name,
        "elapsed_ms":  round(elapsed_ms, 3),
    }


def _not_found(algo_name: str, elapsed_ms: float = 0.0) -> dict:
    return {
        "edges": [], "path_str": "No path found", "hops": 0,
        "dist_cost": float("inf"), "risky_links": 0, "avg_risk": 0.0,
        "found": False, "algorithm": algo_name, "elapsed_ms": round(elapsed_ms, 3),
        "algo_cost": float("inf"),
    }


def route_traditional(G: nx.Graph, src: str, dst: str) -> dict:
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        path = nx.dijkstra_path(H, src, dst, weight="distance")
        cost = nx.dijkstra_path_length(H, src, dst, weight="distance")
        elapsed = (time.perf_counter() - t0) * 1000
        result = _path_metrics(G, path, "Traditional Dijkstra", elapsed)
        result["algo_cost"] = round(cost, 2)
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return _not_found("Traditional Dijkstra", (time.perf_counter() - t0) * 1000)


def route_ai(G: nx.Graph, src: str, dst: str, penalty: float = PENALTY_DEFAULT) -> dict:
    compute_ai_weights(G, penalty)
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        path = nx.dijkstra_path(H, src, dst, weight="ai_weight")
        cost = nx.dijkstra_path_length(H, src, dst, weight="ai_weight")
        elapsed = (time.perf_counter() - t0) * 1000
        result = _path_metrics(G, path, "AI-Enhanced Dijkstra", elapsed)
        result["algo_cost"] = round(cost, 2)
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return _not_found("AI-Enhanced Dijkstra", (time.perf_counter() - t0) * 1000)


def route_bellman_ford(G: nx.Graph, src: str, dst: str) -> dict:
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        path = nx.bellman_ford_path(H, src, dst, weight="distance")
        cost = nx.bellman_ford_path_length(H, src, dst, weight="distance")
        elapsed = (time.perf_counter() - t0) * 1000
        result = _path_metrics(G, path, "Bellman-Ford", elapsed)
        result["algo_cost"] = round(cost, 2)
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXUnbounded):
        return _not_found("Bellman-Ford", (time.perf_counter() - t0) * 1000)


def route_bellman_ford_ai(G: nx.Graph, src: str, dst: str, penalty: float = PENALTY_DEFAULT) -> dict:
    compute_ai_weights(G, penalty)
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        path = nx.bellman_ford_path(H, src, dst, weight="ai_weight")
        cost = nx.bellman_ford_path_length(H, src, dst, weight="ai_weight")
        elapsed = (time.perf_counter() - t0) * 1000
        result = _path_metrics(G, path, "AI Bellman-Ford", elapsed)
        result["algo_cost"] = round(cost, 2)
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXUnbounded):
        return _not_found("AI Bellman-Ford", (time.perf_counter() - t0) * 1000)


def route_floyd_warshall(G: nx.Graph, src: str, dst: str) -> dict:
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        pred, dist_fw = nx.floyd_warshall_predecessor_and_distance(H, weight="distance")
        elapsed = (time.perf_counter() - t0) * 1000
        if dst not in dist_fw.get(src, {}) or dist_fw[src][dst] == float("inf"):
            return _not_found("Floyd-Warshall", elapsed)
        path = nx.reconstruct_path(src, dst, pred)
        cost = dist_fw[src][dst]
        result = _path_metrics(G, path, "Floyd-Warshall", elapsed)
        result["algo_cost"] = round(cost, 2)
        result["dist_matrix"] = {u: {v: round(d, 2) for v, d in row.items()} for u, row in dist_fw.items()}
        return result
    except Exception:
        return _not_found("Floyd-Warshall", (time.perf_counter() - t0) * 1000)


def route_floyd_warshall_ai(G: nx.Graph, src: str, dst: str, penalty: float = PENALTY_DEFAULT) -> dict:
    compute_ai_weights(G, penalty)
    H = _active_graph(G)
    t0 = time.perf_counter()
    try:
        pred, dist_fw = nx.floyd_warshall_predecessor_and_distance(H, weight="ai_weight")
        elapsed = (time.perf_counter() - t0) * 1000
        if dst not in dist_fw.get(src, {}) or dist_fw[src][dst] == float("inf"):
            return _not_found("AI Floyd-Warshall", elapsed)
        path = nx.reconstruct_path(src, dst, pred)
        cost = dist_fw[src][dst]
        result = _path_metrics(G, path, "AI Floyd-Warshall", elapsed)
        result["algo_cost"] = round(cost, 2)
        result["dist_matrix"] = {u: {v: round(d, 2) for v, d in row.items()} for u, row in dist_fw.items()}
        return result
    except Exception:
        return _not_found("AI Floyd-Warshall", (time.perf_counter() - t0) * 1000)


def compare_routes(trad: dict, ai: dict) -> dict:
    return {
        "Traditional": {
            "Path": trad["path_str"], "Distance (km)": trad["dist_cost"],
            "Hops": trad["hops"], "Risky Links": trad["risky_links"],
            "Avg Risk": f"{trad['avg_risk']*100:.1f}%",
        },
        "AI-Enhanced": {
            "Path": ai["path_str"], "Distance (km)": ai["dist_cost"],
            "Hops": ai["hops"], "Risky Links": ai["risky_links"],
            "Avg Risk": f"{ai['avg_risk']*100:.1f}%",
        },
    }


def compare_all_algorithms(results: list) -> list:
    rows = []
    for r in results:
        rows.append({
            "Algorithm":     r.get("algorithm", "?"),
            "Path":          r.get("path_str", "N/A"),
            "Distance (km)": r.get("dist_cost", "∞"),
            "Hops":          r.get("hops", 0),
            "Risky Links":   r.get("risky_links", 0),
            "Avg Risk (%)":  f"{r.get('avg_risk', 0)*100:.1f}%",
            "Time (ms)":     r.get("elapsed_ms", 0),
            "Found":         "✅" if r.get("found") else "❌",
        })
    return rows
