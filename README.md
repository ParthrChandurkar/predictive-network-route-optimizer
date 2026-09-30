# AI-Based Network Route Optimizer

A Streamlit application for comparing conventional shortest-path routing with failure-aware routing. It combines NetworkX graph algorithms, synthetic network telemetry, a Random Forest failure model, stress simulation, and Plotly visualizations.

## Overview

Traditional routing in this project uses link distance as its edge weight. The failure-aware variants add a penalty based on each link's predicted failure probability:

```text
ai_weight = distance + (failure_probability * penalty_factor)
```

A higher penalty makes the routing algorithms prefer lower-risk links, even when the resulting path is longer.

## Features

- Interactive topology creation and editing
- Dijkstra, Bellman-Ford, and Floyd-Warshall comparisons
- Random Forest link-failure prediction
- Traffic, latency, packet-loss, and random-failure simulation
- Traditional and failure-aware path comparison
- Route risk, model metric, feature importance, and network-health charts

## Architecture

`network_builder.py` creates and updates the NetworkX topology. `model.py` trains or loads the model and assigns risk values to links. `routing.py` calculates conventional and failure-aware paths. `simulation.py` applies temporary network conditions, while `analytics.py` builds the Plotly figures used by `app.py`.

The included `network_data.csv` is synthetic and contains bandwidth, latency, packet loss, traffic load, and a binary failure label. Saved model and scaler artifacts are provided for local use.

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
```

The tests cover failed-link exclusion, random failure selection, and restoration of graph state after simulations.

## Limitations

- The dataset is synthetic and does not represent a live network.
- The application does not ingest SNMP, NetFlow, router logs, or streaming telemetry.
- Model predictions support route comparison but do not configure real network devices.
- Results depend on the selected penalty factor and simulated conditions.

## License

No license file is included. Add one before redistributing or using the project commercially.

