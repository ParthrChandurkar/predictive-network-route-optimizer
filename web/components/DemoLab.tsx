"use client";

import { useMemo, useState } from "react";
import { baseEdges, edgeInPath, nodes, route, scenarioEdges, type NodeId, type ScenarioId } from "../lib/network";

const positions: Record<NodeId, [number, number]> = {
  R1: [70, 200], R2: [220, 70], R3: [220, 330], R4: [390, 200],
  R5: [520, 65], R6: [520, 335], R7: [610, 190], R8: [760, 200],
};

const scenarios: { id: ScenarioId; label: string; note: string }[] = [
  { id: "healthy", label: "Healthy", note: "Nominal link risk" },
  { id: "degraded", label: "Degraded", note: "R1—R2 risk = 0.95" },
  { id: "outage", label: "Outage", note: "R4—R7 unavailable" },
];

export function DemoLab() {
  const [scenario, setScenario] = useState<ScenarioId>("degraded");
  const [penalty, setPenalty] = useState(200);
  const [source, setSource] = useState<NodeId>("R1");
  const [destination, setDestination] = useState<NodeId>("R8");
  const edges = useMemo(() => scenarioEdges(scenario), [scenario]);
  const traditional = useMemo(() => route(edges, source, destination), [edges, source, destination]);
  const aware = useMemo(() => route(edges, source, destination, penalty), [edges, source, destination, penalty]);
  const changed = traditional.path.join() !== aware.path.join();

  return (
    <div className="demo-console">
      <div className="demo-toolbar">
        <div className="scenario-picker">
          <span className="control-title">Network condition</span>
          <div className="scenario-buttons">
            {scenarios.map((item) => <button key={item.id} className={scenario === item.id ? "active" : ""} onClick={() => setScenario(item.id)} type="button"><b>{item.label}</b><small>{item.note}</small></button>)}
          </div>
        </div>
        <label><span className="control-title">Source</span><select value={source} onChange={(event) => setSource(event.target.value as NodeId)}>{nodes.filter((node) => node !== destination).map((node) => <option key={node}>{node}</option>)}</select></label>
        <label><span className="control-title">Destination</span><select value={destination} onChange={(event) => setDestination(event.target.value as NodeId)}>{nodes.filter((node) => node !== source).map((node) => <option key={node}>{node}</option>)}</select></label>
        <label className="penalty-control"><span className="control-title">Risk penalty <b>{penalty}</b></span><input aria-label="Risk penalty" type="range" min="0" max="300" step="10" value={penalty} onChange={(event) => setPenalty(Number(event.target.value))} /></label>
      </div>

      <div className="demo-workspace">
        <div className="demo-map">
          <div className="map-meta"><span><i /> LIVE DECISION MODEL</span><span>Distance + risk × penalty</span></div>
          <svg viewBox="0 0 830 410" role="img" aria-label="Interactive demonstration network topology">
            {baseEdges.map((edge) => {
              const [x1, y1] = positions[edge.from];
              const [x2, y2] = positions[edge.to];
              const current = edges.find((item) => item.id === edge.id);
              const failed = !current;
              const degraded = current?.risk === 0.95;
              return <g key={edge.id}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} className={`demo-edge ${failed ? "failed" : ""} ${degraded ? "degraded" : ""}`} />
                {!failed && edgeInPath(current, traditional.path) && <line x1={x1} y1={y1} x2={x2} y2={y2} className="demo-route traditional" />}
                {!failed && edgeInPath(current, aware.path) && <line x1={x1} y1={y1} x2={x2} y2={y2} className="demo-route aware" />}
                <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 8} className={failed ? "edge-text failed" : "edge-text"}>{failed ? "DOWN" : degraded ? "0.95 risk" : `${edge.distance} km`}</text>
              </g>;
            })}
            {nodes.map((node) => <g className="demo-node" key={node} transform={`translate(${positions[node][0]} ${positions[node][1]})`}><circle r="25" /><circle className="node-core" r="6" /><text y="44">{node}</text></g>)}
          </svg>
          <div className="map-legend"><span><i className="legend-traditional" />Distance only</span><span><i className="legend-aware" />Risk aware</span><span><i className="legend-danger" />Impaired</span></div>
        </div>

        <aside className="decision-panel">
          <div className={`decision-state ${changed ? "changed" : "stable"}`}><span>POLICY OUTCOME</span><strong>{changed ? "ROUTE CHANGED" : "ROUTE STABLE"}</strong><small>{changed ? "Risk weighting selected an alternative path." : "Both policies currently agree."}</small></div>
          <RouteCard label="Distance only" tone="traditional" result={traditional} />
          <RouteCard label="Risk aware" tone="aware" result={aware} />
          <p className="demo-boundary">Guided NetworkX-equivalent logic for explanation only. This page does not run the trained LTE model, Mininet, or physical network traffic.</p>
        </aside>
      </div>
    </div>
  );
}

function RouteCard({ label, tone, result }: { label: string; tone: string; result: ReturnType<typeof route> }) {
  return <article className={`decision-card ${tone}`}><span>{label}</span><strong>{result.path.length ? result.path.join(" → ") : "No path"}</strong><dl><div><dt>Distance</dt><dd>{result.distance} km</dd></div><div><dt>Avg. risk</dt><dd>{result.averageRisk.toFixed(2)}</dd></div><div><dt>Objective</dt><dd>{Number.isFinite(result.objective) ? result.objective.toFixed(1) : "—"}</dd></div></dl></article>;
}
