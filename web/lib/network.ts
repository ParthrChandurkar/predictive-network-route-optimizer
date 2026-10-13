export type NodeId = "R1" | "R2" | "R3" | "R4" | "R5" | "R6" | "R7" | "R8";

export type NetworkEdge = {
  id: string;
  from: NodeId;
  to: NodeId;
  distance: number;
  risk: number;
};

export type ScenarioId = "healthy" | "degraded" | "outage";

export type RouteResult = {
  path: NodeId[];
  distance: number;
  averageRisk: number;
  objective: number;
};

export const nodes: NodeId[] = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"];

export const baseEdges: NetworkEdge[] = [
  { id: "R1-R2", from: "R1", to: "R2", distance: 12, risk: 0.08 },
  { id: "R1-R3", from: "R1", to: "R3", distance: 20, risk: 0.05 },
  { id: "R1-R4", from: "R1", to: "R4", distance: 28, risk: 0.05 },
  { id: "R2-R4", from: "R2", to: "R4", distance: 13, risk: 0.06 },
  { id: "R2-R5", from: "R2", to: "R5", distance: 25, risk: 0.12 },
  { id: "R3-R4", from: "R3", to: "R4", distance: 14, risk: 0.04 },
  { id: "R3-R6", from: "R3", to: "R6", distance: 20, risk: 0.07 },
  { id: "R4-R7", from: "R4", to: "R7", distance: 16, risk: 0.04 },
  { id: "R5-R8", from: "R5", to: "R8", distance: 22, risk: 0.09 },
  { id: "R6-R8", from: "R6", to: "R8", distance: 20, risk: 0.05 },
  { id: "R7-R8", from: "R7", to: "R8", distance: 12, risk: 0.05 },
];

export function scenarioEdges(scenario: ScenarioId): NetworkEdge[] {
  if (scenario === "healthy") return baseEdges.map((edge) => ({ ...edge }));
  if (scenario === "degraded") {
    return baseEdges.map((edge) => edge.id === "R1-R2" ? { ...edge, risk: 0.95 } : { ...edge });
  }
  return baseEdges.filter((edge) => edge.id !== "R4-R7").map((edge) => ({ ...edge }));
}

export function route(
  edges: NetworkEdge[],
  source: NodeId,
  destination: NodeId,
  riskPenalty = 0,
): RouteResult {
  const costs = new Map<NodeId, number>(nodes.map((node) => [node, Number.POSITIVE_INFINITY]));
  const previous = new Map<NodeId, NodeId>();
  const unvisited = new Set(nodes);
  costs.set(source, 0);

  while (unvisited.size) {
    const current = [...unvisited].reduce((best, node) =>
      (costs.get(node) ?? Infinity) < (costs.get(best) ?? Infinity) ? node : best
    );
    if (!Number.isFinite(costs.get(current))) break;
    unvisited.delete(current);
    if (current === destination) break;

    for (const edge of edges) {
      const neighbor = edge.from === current ? edge.to : edge.to === current ? edge.from : null;
      if (!neighbor || !unvisited.has(neighbor)) continue;
      const candidate = (costs.get(current) ?? Infinity) + edge.distance + edge.risk * riskPenalty;
      if (candidate < (costs.get(neighbor) ?? Infinity)) {
        costs.set(neighbor, candidate);
        previous.set(neighbor, current);
      }
    }
  }

  if (!Number.isFinite(costs.get(destination))) {
    return { path: [], distance: 0, averageRisk: 0, objective: Number.POSITIVE_INFINITY };
  }

  const path: NodeId[] = [destination];
  while (path[0] !== source) path.unshift(previous.get(path[0])!);
  const used = path.slice(0, -1).map((node, index) =>
    edges.find((edge) =>
      (edge.from === node && edge.to === path[index + 1]) ||
      (edge.to === node && edge.from === path[index + 1])
    )!
  );

  return {
    path,
    distance: used.reduce((sum, edge) => sum + edge.distance, 0),
    averageRisk: used.reduce((sum, edge) => sum + edge.risk, 0) / used.length,
    objective: costs.get(destination)!,
  };
}

export function edgeInPath(edge: NetworkEdge, path: NodeId[]): boolean {
  return path.slice(0, -1).some((node, index) =>
    (edge.from === node && edge.to === path[index + 1]) ||
    (edge.to === node && edge.from === path[index + 1])
  );
}
