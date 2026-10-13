"""Project-wide defaults for the current synthetic baseline.

Keeping these values in one dependency-free module makes the baseline
reproducible and gives the real-telemetry rebuild a stable configuration seam.
"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "network_data.csv"
MODEL_PATH = PROJECT_ROOT / "failure_model.pkl"
SCALER_PATH = PROJECT_ROOT / "failure_scaler.pkl"

LTE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "lte"
LTE_TRAIN_RAW_PATH = LTE_RAW_DIR / "ML-MATT-CompetitionQT2021_train.csv"
LTE_TEST_RAW_PATH = LTE_RAW_DIR / "ML-MATT-CompetitionQT2021_test.csv"
LTE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "lte"
LTE_TRAIN_PROCESSED_PATH = LTE_PROCESSED_DIR / "train.csv"
LTE_TEST_PROCESSED_PATH = LTE_PROCESSED_DIR / "test.csv"
PHASE1_REPORT_PATH = PROJECT_ROOT / "reports" / "phase1_ingestion.json"
PHASE2_REPORT_PATH = PROJECT_ROOT / "reports" / "phase2_model_baselines.json"
LTE_MODEL_PATH = PROJECT_ROOT / "artifacts" / "lte_degradation_model.pkl"
PHASE3_REPORT_PATH = PROJECT_ROOT / "reports" / "phase3_model_artifact.json"
PHASE7_REPORT_PATH = PROJECT_ROOT / "reports" / "phase7_lte_patterns.json"
PHASE8_REPORT_PATH = PROJECT_ROOT / "reports" / "phase8_final_evaluation.json"
MININET_PLAN_PATH = PROJECT_ROOT / "reports" / "phase4_mininet_plan.json"
MININET_RESULTS_PATH = PROJECT_ROOT / "reports" / "phase4_mininet_results.json"

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

