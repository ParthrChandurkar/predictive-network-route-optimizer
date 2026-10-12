# Predictive Network Route Optimizer

A Streamlit application for comparing conventional shortest-path routing with ML-assisted, risk-aware routing. The current version combines NetworkX graph algorithms, synthetic network telemetry, a Random Forest baseline, stress simulation, and Plotly visualizations.

> **Rebuild in progress:** The project is being redesigned around real network telemetry and time-aware degradation prediction. The existing synthetic workflow remains available as a reproducible demonstration while the new data pipeline is developed.

## Overview

Traditional routing in this project uses link distance as its edge weight. Risk-aware variants add a penalty from either the synthetic failure baseline or an explicitly integrated LTE degradation score:

```text
ai_weight = distance + (routing_risk * penalty_factor)
```

A higher penalty makes the routing algorithms prefer lower-risk links, even when the resulting path is longer.

## Features

- Interactive topology creation and editing
- Dijkstra, Bellman-Ford, and Floyd-Warshall comparisons
- Random Forest risk prediction using the current synthetic baseline
- Traffic, latency, packet-loss, and random-failure simulation
- Traditional and failure-aware path comparison
- Route risk, model metric, feature importance, and network-health charts

## Architecture

`network_builder.py` creates and updates the NetworkX topology. `model.py` trains or loads the model and assigns risk values to links. `routing.py` calculates conventional and failure-aware paths. `simulation.py` applies temporary network conditions, while `analytics.py` builds the Plotly figures used by `app.py`.

The included `network_data.csv` is synthetic and contains bandwidth, latency, packet loss, traffic load, and a binary failure label. Saved model and scaler artifacts are provided for local use.

## Rebuild Dataset

The rebuild will begin with Kaggle's [Anomaly Detection in 4G Cellular Networks](https://www.kaggle.com/competitions/anomaly-detection-in-4g-cellular-networks/data) dataset. It contains telemetry collected from a real LTE deployment across multiple base stations and cells at 15-minute intervals. Its measurements include radio-resource utilization, uplink and downlink throughput, active-user counts, and normal or unusual operating labels.

This dataset supports learning real network-degradation patterns rather than reproducing a hand-written synthetic failure rule. It represents cellular-network anomalies, not confirmed physical-link failures, so the rebuild will describe its predictions as **degradation risk** unless stronger failure evidence is available.

The Kaggle data is not bundled in this repository. Download access and usage remain subject to the dataset's competition rules.

After accepting the competition rules and configuring the Kaggle CLI, download both files into the ignored raw-data directory:

```powershell
New-Item -ItemType Directory -Path "data/raw/lte" -Force | Out-Null
kaggle competitions download -c anomaly-detection-in-4g-cellular-networks -f ML-MATT-CompetitionQT2021_train.csv -p data/raw/lte
kaggle competitions download -c anomaly-detection-in-4g-cellular-networks -f ML-MATT-CompetitionQT2021_test.csv -p data/raw/lte
Expand-Archive -LiteralPath "data/raw/lte/ML-MATT-CompetitionQT2021_train.csv.zip" -DestinationPath "data/raw/lte" -Force
python -m telemetry.prepare_lte
```

The preparation command validates both splits, normalizes their schema, writes canonical files under `data/processed/lte/`, and refreshes `reports/phase1_ingestion.json`. The Streamlit ML Model tab can also validate an uploaded labeled or unlabeled competition CSV. Raw and processed datasets under `data/` are intentionally excluded from Git.

## Rebuild Direction

The next version will focus on a validated telemetry pipeline, time-aware ML evaluation, and measurable risk-aware routing experiments. Further implementation details will be documented as each rebuild milestone is completed.

Phase 0 stabilization is complete: project settings are centralized, training data is validated, model artifacts are written safely, regression coverage has been expanded, and a reproducible synthetic baseline is stored in `reports/phase0_baseline.json`.

Phase 1 ingestion is complete: the adapter supports Kaggle's semicolon-delimited LTE files, validates labeled and unlabeled splits, produces a canonical schema, records source hashes and data-quality findings, and keeps licensed raw data outside Git. The dataset contains 36,904 training observations and 9,158 test observations across 33 cells.

Phase 2 leakage-aware baselines are complete: row-local cyclical and aggregate features are evaluated with duplicate-safe and held-out-cell cross-validation. On held-out cells, the Random Forest baseline reaches F1 0.832 and ROC-AUC 0.954; these scores describe unusual-behavior detection, not physical failure prediction. Full fold metrics and limitations are stored in `reports/phase2_model_baselines.json`.

Phase 3 model integration is complete: a reproducible local model artifact produces LTE degradation scores, and an explicit adapter can apply a selected time-of-day snapshot to routing weights. Because the Kaggle data contains no physical network topology, the included cell-to-link assignment is clearly marked as a deterministic demo mapping rather than a real deployment map. The large trained artifact stays outside Git; its hash and provenance are recorded in `reports/phase3_model_artifact.json`.

## Tech Stack

- Python
- Streamlit and Plotly
- NetworkX
- pandas and NumPy
- scikit-learn
- Python `unittest`

## Getting Started

Prerequisites: Python 3.9 or newer and pip.

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the dashboard:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit normally serves the application at `http://localhost:8501`.

## Usage

1. Train or inspect the failure model from the ML Model tab.
2. Review or edit the topology in the Overview and Builder tabs.
3. Select a source and destination in the Routing tab.
4. Compare conventional and failure-aware paths.
5. Apply degraded conditions in the Simulation tab and calculate the routes again.
6. Review model and network results in the Analytics tab.

## Testing

```bash
python -m unittest discover -s tests
python baseline.py
python -m telemetry.prepare_lte
python -m modeling.lte_evaluation
python -m modeling.lte_model
```

The tests cover dataset reproducibility, telemetry validation, leakage-safe LTE features, model persistence, topology mutation, routing consistency, failed-link exclusion, and simulation restoration.

## Current Version Limitations

- The active routing model still uses synthetic link data; real LTE telemetry is ingested but is not yet connected to routing decisions.
- LTE labels indicate unusual behavior, not confirmed physical link failures.
- The LTE data contains time of day but no calendar date, preventing a provably chronological split.
- The LTE cell-to-link mapping is experimental because the source dataset provides no physical topology.
- The application does not ingest SNMP, NetFlow, router logs, or streaming telemetry.
- Model predictions support route comparison but do not configure real network devices.
- Results depend on the selected penalty factor and simulated conditions.

## License

No license file is included. Add one before redistributing or using the project commercially.
