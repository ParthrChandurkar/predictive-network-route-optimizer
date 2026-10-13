const githubUrl = "https://github.com/ParthrChandurkar/predictive-network-route-optimizer";
const dashboardUrl = process.env.NEXT_PUBLIC_STREAMLIT_URL || `${githubUrl}#run-the-streamlit-dashboard`;
const hasHostedDashboard = Boolean(process.env.NEXT_PUBLIC_STREAMLIT_URL);

const metrics = [
  { value: "0.832", label: "Held-out cell F1", context: "Group-separated validation" },
  { value: "0.954", label: "Held-out ROC-AUC", context: "Unseen LTE cells" },
  { value: "+251%", label: "TCP throughput", context: "3-run Mininet mean" },
  { value: "−66.6%", label: "Mean RTT", context: "Controlled emulation" },
];

const stages = [
  { number: "01", title: "Validate telemetry", copy: "Normalize 36,904 LTE observations across 33 cells, preserve source hashes, and keep duplicates inside the same evaluation fold." },
  { number: "02", title: "Estimate degradation", copy: "Classify unusual cellular behavior with a leakage-aware Random Forest. Scores mean degradation risk—not physical-link failure probability." },
  { number: "03", title: "Reweight the graph", copy: "Map a selected telemetry snapshot to demonstration links and combine network distance with an adjustable risk penalty." },
  { number: "04", title: "Test the decision", copy: "Compare routing policies in a controlled eight-router Mininet scenario and export traces and metrics through OpenTelemetry." },
];

const dashboardTabs = [
  ["01", "Network Ops", "Live topology, link health, control-plane posture, and Mininet evidence."],
  ["02", "Risk Intelligence", "Held-out LTE metrics, validated uploads, model scoring, and risk timelines."],
  ["03", "Pattern Discovery", "Spatial and temporal failure signatures with explicit artifact warnings."],
  ["04", "Topology", "Add, remove, and update routers and links in the working graph."],
  ["05", "Route Control", "Compare conventional and risk-aware paths with a recorded decision trail."],
  ["06", "Benchmarks", "Inspect six routing variants, complexity, and final evaluation metrics."],
  ["07", "Failure Lab", "Stress, fail, and restore links to test route behavior interactively."],
  ["08", "Insights", "Review telemetry, observability endpoints, evidence, and system-wide summaries."],
];

export default function Home() {
  return (
    <main>
      <nav className="site-nav" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="NetRoute Intelligence home"><span className="brand-mark">N</span><span>NetRoute <b>Intelligence</b></span></a>
        <div className="nav-links"><a href="#system">System</a><a href="#evidence">Evidence</a><a href="#dashboard">Dashboard</a></div>
        <a className="nav-cta" href="/demo">Open live demo <span>→</span></a>
      </nav>

      <section className="hero section-shell" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="live-pulse" /> Network degradation simulation platform</div>
          <h1>Detect weak signals.<br /><em>Route around risk.</em></h1>
          <p className="hero-lead">An evidence-led platform that turns real LTE telemetry into degradation scores, compares risk-aware paths, and measures the outcome under controlled network impairment.</p>
          <div className="hero-actions"><a className="button primary" href="/demo">Try the live demo <span>→</span></a><a className="button secondary" href={dashboardUrl} target="_blank" rel="noreferrer">{hasHostedDashboard ? "Launch Streamlit" : "Run Streamlit locally"} <span>↗</span></a></div>
          <div className="hero-meta"><span>REAL LTE DATA</span><span>6 ROUTING VARIANTS</span><span>OTEL OBSERVABILITY</span></div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="radar-grid" />
          <svg viewBox="0 0 620 520" className="hero-network">
            <defs><linearGradient id="hero-link" x1="0" x2="1"><stop stopColor="#1f4962" /><stop offset="1" stopColor="#2ea987" /></linearGradient></defs>
            {[[88,250,218,116],[88,250,222,372],[218,116,370,220],[222,372,370,220],[370,220,516,112],[370,220,532,340],[516,112,570,258],[532,340,570,258]].map((line, index) => <line key={index} x1={line[0]} y1={line[1]} x2={line[2]} y2={line[3]} />)}
            {[[88,250,"R1"],[218,116,"R2"],[222,372,"R3"],[370,220,"R4"],[516,112,"R7"],[532,340,"R6"],[570,258,"R8"]].map(([x,y,label], index) => <g key={String(label)} className={index === 1 ? "alert-node" : ""} transform={`translate(${x} ${y})`}><circle className="node-ring" r="24" /><circle className="node-dot" r="7" /><text y="43">{label}</text></g>)}
            <circle className="packet packet-one" r="5" /><circle className="packet packet-two" r="5" />
          </svg>
          <div className="signal-card signal-card--top"><span>LINK R1—R2</span><b>DEGRADATION 0.95</b><small>Risk threshold exceeded</small></div>
          <div className="signal-card signal-card--bottom"><span>POLICY RESPONSE</span><b>REROUTE AVAILABLE</b><small>R1 → R4 → R7 → R8</small></div>
        </div>
      </section>

      <section className="metric-band" id="evidence"><div className="section-shell metric-grid">{metrics.map((metric) => <article key={metric.label}><strong>{metric.value}</strong><div><b>{metric.label}</b><span>{metric.context}</span></div></article>)}</div></section>

      <section className="story-section section-shell" id="system">
        <div className="section-heading"><div><span className="kicker">THE SYSTEM</span><h2>From noisy telemetry<br />to a testable route decision.</h2></div><p>The project does not claim production automation. It creates a reproducible laboratory for discovering degradation patterns and asking whether a risk-aware route performs better under controlled failure conditions.</p></div>
        <div className="stage-grid">{stages.map((stage) => <article key={stage.number}><span>{stage.number}</span><div className="stage-icon"><i /><i /><i /></div><h3>{stage.title}</h3><p>{stage.copy}</p></article>)}</div>
      </section>

      <section className="architecture-section">
        <div className="section-shell architecture-grid">
          <div><span className="kicker">ARCHITECTURE</span><h2>One decision path.<br />Three evidence classes.</h2><p>Held-out model results, a guided browser demonstration, and emulator measurements stay visibly separate. That boundary keeps the prototype useful and honest.</p><a href={githubUrl} target="_blank" rel="noreferrer" className="text-link">Inspect the implementation <span>↗</span></a></div>
          <div className="architecture-map">
            <div className="arch-row"><div className="arch-node"><small>INPUT</small><b>LTE telemetry</b><span>33 cells · 15 min slots</span></div><i>→</i><div className="arch-node"><small>MODEL</small><b>Degradation risk</b><span>Held-out cell evaluation</span></div></div>
            <div className="arch-spine"><span>↓</span></div>
            <div className="arch-row"><div className="arch-node"><small>POLICY</small><b>Graph reweighting</b><span>Distance + risk × penalty</span></div><i>→</i><div className="arch-node accent"><small>OUTPUT</small><b>Safer route</b><span>Decision + provenance</span></div></div>
            <div className="observability-rail"><b>OpenTelemetry</b><span>traces</span><span>metrics</span><span>Jaeger</span><span>Prometheus</span></div>
          </div>
        </div>
      </section>

      <section className="dashboard-section section-shell" id="dashboard">
        <div className="section-heading"><div><span className="kicker">STREAMLIT CONTROL SURFACE</span><h2>One dashboard.<br />Eight operational workflows.</h2></div><p>The web demo explains a single route decision. The Python dashboard is the detailed implementation where the network is inspected, changed, stressed, scored, and compared.</p></div>
        <div className="dashboard-preview">
          <div className="dashboard-bar"><span><i /> STREAMLIT / NETWORK OPERATIONS</span><b>SIMULATION MODE</b></div>
          <div className="dashboard-body">
            <div className="dashboard-sidebar"><span>CONTROL PLANE</span><b>8 routers</b><b>11 links</b><b>6 algorithms</b><small>Interactive state is managed by the Streamlit application.</small></div>
            <div className="dashboard-canvas">
              <div className="dashboard-tabs"><span className="active">Network Ops</span><span>Risk Intelligence</span><span>Route Control</span><span>Failure Lab</span></div>
              <div className="dashboard-topology" aria-hidden="true">
                <svg viewBox="0 0 720 250">
                  {[[68,125,210,48],[68,125,210,202],[210,48,365,125],[210,202,365,125],[365,125,520,48],[365,125,520,202],[520,48,660,125],[520,202,660,125]].map((line, index) => <line key={index} x1={line[0]} y1={line[1]} x2={line[2]} y2={line[3]} className={index === 0 ? "degraded" : index > 4 ? "active" : ""} />)}
                  {[[68,125,"R1"],[210,48,"R2"],[210,202,"R3"],[365,125,"R4"],[520,48,"R5"],[520,202,"R6"],[660,125,"R8"]].map(([x,y,label]) => <g key={String(label)} transform={`translate(${x} ${y})`}><circle r="18" /><circle className="core" r="5" /><text y="34">{label}</text></g>)}
                </svg>
              </div>
              <div className="dashboard-status"><span><b>ROUTE</b> R1 → R3 → R4 → R6 → R8</span><span><b>RISK</b> 0.18</span><span><b>STATUS</b> reroute available</span></div>
            </div>
          </div>
        </div>
        <div className="workflow-grid">{dashboardTabs.map(([number, title, copy]) => <article key={title}><span>{number}</span><div><h3>{title}</h3><p>{copy}</p></div></article>)}</div>
        <div className="dashboard-action"><p><b>Two levels of access:</b> try the guided route decision in the browser, then clone the repository for topology edits, telemetry scoring, failure injection, benchmarks, and observability in Streamlit.</p><div className="dashboard-action-buttons"><a className="button secondary" href="/demo">Try browser demo <span>→</span></a><a className="button primary" href={dashboardUrl} target="_blank" rel="noreferrer">{hasHostedDashboard ? "Launch Streamlit dashboard" : "Clone & run locally"} <span>↗</span></a></div></div>
      </section>

      <section className="findings-section section-shell">
        <div className="section-heading"><div><span className="kicker">WHAT THE DATA SAID</span><h2>Useful results,<br />without inflated claims.</h2></div><p>Failure analysis depends on understanding what the data can actually support. These are the strongest findings—and the caveats attached to them.</p></div>
        <div className="findings-grid">
          <article className="finding-primary"><div className="finding-index">01 / HELD-OUT PATTERN</div><strong>100%</strong><h3>unusual-label precision</h3><p>A narrow <code>mean_ue_ul</code> signature covered 40.6% of unusual validation rows on disjoint cells.</p><div className="warning-line">Possible quantization or label-construction artifact—not a causal failure mechanism.</div></article>
          <article><div className="finding-index">02 / DISTRIBUTION</div><h3>No single failing cell cluster.</h3><p>Labels remained broadly distributed across all 33 cells, with only a 5.67-point cell-rate spread after deduplication.</p></article>
          <article><div className="finding-index">03 / EMULATION</div><h3>Latency improved. Loss did not.</h3><p>Risk-aware routing reduced mean RTT by 66.6%, while mean packet loss stayed unchanged at 13.3%.</p></article>
        </div>
      </section>

      <section className="stack-section"><div className="section-shell stack-inner"><span className="kicker">BUILT WITH OPEN TOOLS</span><div className="stack-list">{["Python","NetworkX","scikit-learn","Mininet","OpenTelemetry","Jaeger","Prometheus","Streamlit","Next.js"].map((tool) => <span key={tool}>{tool}</span>)}</div></div></section>

      <section className="cta-section section-shell"><div><span className="kicker">EXPLORE THE SYSTEM</span><h2>Try the decision.<br />Then inspect the evidence.</h2></div><div className="cta-actions"><a className="button primary" href="/demo">Open live demo <span>→</span></a><a className="button secondary" href={dashboardUrl} target="_blank" rel="noreferrer">{hasHostedDashboard ? "Launch dashboard" : "Clone & run Streamlit"} <span>↗</span></a></div></section>

      <footer><div className="section-shell footer-inner"><a className="brand" href="#top"><span className="brand-mark">N</span><span>NetRoute <b>Intelligence</b></span></a><p>Validated prototype · no physical device control</p><p>Built by Parth Chandurkar</p></div></footer>
    </main>
  );
}
