import type { Metadata } from "next";
import Link from "next/link";
import { DemoLab } from "../../components/DemoLab";

const githubUrl = "https://github.com/ParthrChandurkar/predictive-network-route-optimizer";

export const metadata: Metadata = {
  title: "Interactive Route Demo | NetRoute Intelligence",
  description: "Explore how degradation risk changes a route decision in a guided browser demonstration.",
  alternates: { canonical: "/demo" },
};

export default function DemoPage() {
  return (
    <main>
      <nav className="site-nav demo-nav" aria-label="Demo navigation">
        <Link className="brand" href="/"><span className="brand-mark">N</span><span>NetRoute <b>Intelligence</b></span></Link>
        <div className="nav-links"><Link href="/">Overview</Link><a href="#simulation">Simulation</a><a href="#full-lab">Full lab</a></div>
        <a className="nav-cta" href={githubUrl} target="_blank" rel="noreferrer">Clone repository <span>↗</span></a>
      </nav>

      <section className="demo-hero section-shell">
        <div>
          <span className="eyebrow"><span className="live-pulse" /> Guided browser demonstration</span>
          <h1>See risk change<br /><em>the route decision.</em></h1>
        </div>
        <div className="demo-intro">
          <p>Apply a known degradation scenario, tune the risk penalty, and compare the shortest path with the degradation-aware alternative.</p>
          <div className="scope-strip"><span>DETERMINISTIC TOPOLOGY</span><span>CLIENT-SIDE ROUTING</span><span>NO DEVICE CONTROL</span></div>
        </div>
      </section>

      <section className="demo-section section-shell" id="simulation">
        <DemoLab />
      </section>

      <section className="full-lab-section" id="full-lab">
        <div className="section-shell full-lab-grid">
          <div><span className="kicker">FULL IMPLEMENTATION</span><h2>The demo explains one decision.<br />Streamlit runs the complete simulation.</h2></div>
          <div className="full-lab-copy"><p>Clone the repository and run Streamlit locally for the complete experience: real LTE CSV validation and scoring, pattern discovery, topology editing, six routing variants, failure injection, Mininet evidence, and OpenTelemetry links.</p><div className="full-lab-actions"><a className="button primary" href={`${githubUrl}#run-the-streamlit-dashboard`} target="_blank" rel="noreferrer">Clone & run Streamlit <span>↗</span></a><Link className="button secondary" href="/">Read the project story <span>←</span></Link></div></div>
        </div>
      </section>

      <section className="demo-notes section-shell">
        <article><span>01 / MODEL BOUNDARY</span><h3>A score is not a failure probability.</h3><p>The browser scenario uses fixed illustrative risk values. The repository’s trained model estimates unusual LTE behavior from validated telemetry.</p></article>
        <article><span>02 / TOPOLOGY BOUNDARY</span><h3>The graph is a controlled lab.</h3><p>The Kaggle cells have no physical mapping to these routers. The integration deliberately uses a documented deterministic mapping.</p></article>
        <article><span>03 / EVIDENCE BOUNDARY</span><h3>Measured traffic lives in Mininet.</h3><p>The published throughput and RTT results come from repeated emulator trials, not this client-side visualization.</p></article>
      </section>

      <footer><div className="section-shell footer-inner"><Link className="brand" href="/"><span className="brand-mark">N</span><span>NetRoute <b>Intelligence</b></span></Link><p>Browser demo · illustrative routing logic</p><p>Built by Parth Chandurkar</p></div></footer>
    </main>
  );
}
