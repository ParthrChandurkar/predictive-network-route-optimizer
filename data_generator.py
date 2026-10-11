"""
data_generator.py — Synthetic dataset generator for link failure prediction.
Failure probability is driven by: low bandwidth, high latency,
high packet_loss, and high traffic_load.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from settings import DATASET_PATH, MODEL_FEATURES, RANDOM_SEED

FEATURES = list(MODEL_FEATURES)

def generate_network_dataset(n_samples: int = 800, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero")
    rng = np.random.default_rng(random_state)
    bandwidth    = rng.uniform(10,  1000, n_samples)
    latency      = rng.uniform(1,   300,  n_samples)
    packet_loss  = rng.uniform(0,   20,   n_samples)
    traffic_load = rng.uniform(0,   100,  n_samples)
    stress = (
        (1 - bandwidth / 1000) * 0.30 +
        (latency / 300)        * 0.25 +
        (packet_loss / 20)     * 0.30 +
        (traffic_load / 100)   * 0.15
    )
    noise   = rng.normal(0, 0.08, n_samples)
    failure = (np.clip(stress + noise, 0, 1) > 0.55).astype(int)
    return pd.DataFrame({
        "bandwidth":    np.round(bandwidth,    2),
        "latency":      np.round(latency,      2),
        "packet_loss":  np.round(packet_loss,  2),
        "traffic_load": np.round(traffic_load, 2),
        "failure":      failure,
    })

def save_dataset(
    path: Union[str, Path] = DATASET_PATH,
    n_samples: int = 800,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate and save the reproducible synthetic baseline dataset."""
    df = generate_network_dataset(n_samples=n_samples, random_state=random_state)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = save_dataset()
    print(f"Saved {len(df)} rows. Failure rate: {df.failure.mean():.2%}")
