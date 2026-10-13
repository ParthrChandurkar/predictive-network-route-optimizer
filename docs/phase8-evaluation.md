# Phase 8 Evaluation

Phase 8 joins the project's previously separate evidence into one reproducible evaluation without treating unlike results as interchangeable.

## Run it

Prepare the LTE data and model first, then run:

```bash
python -m evaluation.phase8
```

The command writes `reports/phase8_final_evaluation.json` and exits with a non-zero status if any required evidence check fails.

## Evidence classes

- **Held-out data:** LTE model metrics and interpretable patterns evaluated with cell-separated groups.
- **Integration demo:** Real LTE degradation scores assigned deterministically to demonstration links. This validates the software path, not a physical cell-to-router mapping.
- **Controlled emulation:** Repeated Mininet measurements from an intentionally impaired link in WSL2.

## Verified workflow

```text
validated LTE rows -> degradation model -> per-cell scores
                   -> explicit demo mapping -> routing-risk weights
                   -> six routing algorithms -> route comparison

controlled scenario -> Mininet repetitions -> throughput/RTT/loss evidence

all phase reports -> consistency and artifact checks -> final JSON report
```

The consolidated report checks row-count continuity, the saved model hash, algorithm-family agreement, LTE integration route availability, and repeated Mininet evidence. It also carries forward the project's claim limitations so UI or deployment work can display the results honestly.
