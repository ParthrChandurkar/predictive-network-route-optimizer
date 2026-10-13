# v1.0.0 — Research system release

This release completes the Predictive Network Degradation & Route Optimizer as a reproducible portfolio and research project.

## Included

- Validated Kaggle LTE ingestion and leakage-aware held-out-cell evaluation
- Random Forest degradation scoring and interpretable pattern discovery
- Six conventional and risk-aware routing variants
- Eight-workflow Streamlit network-operations dashboard
- Repeated Mininet throughput, RTT, and packet-loss measurements
- OpenTelemetry Collector, Jaeger, and Prometheus integration
- Vercel-hosted Next.js project story and guided route demo
- Python and web continuous integration

## Headline evidence

- Held-out-cell F1: `0.8323`
- Held-out-cell ROC-AUC: `0.9536`
- Mininet mean throughput: `0.333 → 1.168 Mbps` (`+251.19%`)
- Mininet mean RTT: `278.154 → 92.977 ms` (`−66.57%`)
- Mininet mean packet loss: unchanged at `13.33%`

These are held-out-data and controlled-emulator results, not production-network guarantees. See `reports/phase8_final_evaluation.json` for evidence classes and limitations.
