# Engine Health Lab

**A leakage-aware benchmark for remaining-useful-life prediction under operating-condition shift.**

This project asks a practical maintenance question: when an engine is operating in conditions unlike those seen during training, can an engineer trust the model's uncertainty enough to make a maintenance decision?

Rather than ranking models by one headline RMSE, the experiment compares transparent baselines and reports error, the asymmetric NASA scoring function, prediction-interval coverage/width, and performance by operating regime. Engine trajectories stay intact across train/test partitions. Preprocessing is fitted only on training engines. The official NASA test units and RUL targets are used only for final evaluation.

## Status

Initial benchmark implemented and run once on official FD002. The NASA C-MAPSS data are simulated run-to-failure trajectories, not measurements from a commercial fleet; conclusions must be limited accordingly. The single-seed results are exploratory, not a publication-grade comparison.

## Research question

Do uncertainty-aware, regime-conditioned simple models make more reliable maintenance decisions than a pooled regression baseline on C-MAPSS FD002 (multiple operating conditions), without sacrificing acceptable RUL error?

Planned ablations:

1. Pooled versus regime-conditioned preprocessing/modeling.
2. Point estimates versus calibrated prediction intervals.
3. Random-cycle split (demonstration of leakage only) versus engine-grouped validation and the official engine-level test set.
4. Evaluation by operating-condition cluster and engine, not just by row.

## Metrics

- RMSE and MAE on one final prediction per held-out engine.
- NASA asymmetric prognostics score (late RUL estimates receive a larger penalty).
- Empirical prediction-interval coverage and mean interval width.
- Regime-stratified metrics and bootstrap confidence intervals over engines.

No result is called “accurate” without naming the split, subset, target convention, and metric. A score on C-MAPSS is not a claim of real-world aircraft-maintenance readiness.

## Data

Download the NASA Prognostics Center of Excellence **Turbofan Engine Degradation Simulation** dataset from the [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The archive is not committed. `scripts/download_data.py` fetches the official ZIP and verifies its contents. Cite A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Research Center.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/download_data.py
python -m engine_health.run --subset FD002
pytest
```

The benchmark uses scikit-learn models and deterministic feature engineering so the evaluation can be inspected and reproduced before any deep-learning extension. Training uses trailing windows from complete training-engine trajectories; validation engines are disjoint from fit engines; official test targets are used only for final evaluation. The calibration protocol deliberately samples one as-of point per held-out engine and evaluates at the official test endpoint, making coverage under this endpoint shift an explicit measurement.

## Initial FD002 result (seed 42)

Five deterministic engine-level split seeds (7, 21, 42, 73, 101); values are mean ± sample SD across seeds.

| Model | Test RMSE (cycles) | Test MAE (cycles) | Nominal 90% interval coverage on official test |
|---|---:|---:|---:|
| Ridge | 32.81 ± 1.30 | 24.12 ± 0.64 | 82.9% ± 2.3% |
| Random forest | 58.61 ± 1.87 | 45.37 ± 1.44 | 50.7% ± 6.6% |
| Histogram gradient boosting | 57.43 ± 0.97 | 44.18 ± 0.81 | 54.5% ± 4.5% |

This is a useful engineering result, not a victory claim: the simple ridge baseline beat both tree models across these five splits, and all split-conformal intervals under-covered at the official test endpoint. Regime-stratified errors were uneven (worst ridge regime RMSE 64.84 cycles in seed 42). The result motivates a regime-conditioned comparison and better endpoint-matched calibration. Full values and sample counts are in `results/FD002_summary.md`.

## Repository map

- `engine_health/`: parsing, features, metrics, and evaluation.
- `scripts/`: data download and experiment entry point.
- `tests/`: leakage, scoring, parsing, and metric checks.
- `results/`: generated machine-readable results (tracked only after verification).

## Limitations

FD002 is a simulated benchmark and is small compared with industrial fleets. Operating-regime clustering is descriptive, not a physical fault diagnosis. Prediction intervals estimated from a finite validation set are not guaranteed under arbitrary distribution shift. This project is decision-support research, not a certified maintenance system.

## License and attribution

Code: MIT. The NASA dataset remains subject to the source repository's terms and attribution requirements; it is not redistributed here.
