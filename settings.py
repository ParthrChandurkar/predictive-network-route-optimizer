"""Project-wide defaults for the current synthetic baseline.

Keeping these values in one dependency-free module makes the baseline
reproducible and gives the real-telemetry rebuild a stable configuration seam.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "network_data.csv"
MODEL_PATH = PROJECT_ROOT / "failure_model.pkl"
SCALER_PATH = PROJECT_ROOT / "failure_scaler.pkl"

RANDOM_SEED = 42
MODEL_FEATURES = (
    "bandwidth",
    "latency",
    "packet_loss",
    "traffic_load",
)
TARGET_COLUMN = "failure"

LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65
DEFAULT_ROUTE_PENALTY = 200.0

