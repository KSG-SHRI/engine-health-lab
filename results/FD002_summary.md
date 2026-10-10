# Exploratory FD002 benchmark

Five deterministic engine-level split seeds: 7, 21, 42, 73, 101. Values below are mean ± sample standard deviation. Same official NASA test units are used for each seed; only the fit/calibration split changes. Results are exploratory and do not establish generalization to real aircraft fleets.

| Model | RMSE (cycles) | MAE (cycles) | Nominal 90% interval test coverage |
|---|---:|---:|---:|
| Ridge | 32.81 ± 1.30 | 24.12 ± 0.64 | 82.9% ± 2.3% |
| Random forest | 58.61 ± 1.87 | 45.37 ± 1.44 | 50.7% ± 6.6% |
| Histogram gradient boosting | 57.43 ± 0.97 | 44.18 ± 0.81 | 54.5% ± 4.5% |

## Protocol

- Data: NASA C-MAPSS FD002, 260 run-to-failure train engines, 259 official test engines.
- Per seed: 208 fit engines and 52 calibration engines, disjoint by engine ID.
- Target: conventional RUL capped at 125 cycles.
- Features: trailing 30-cycle setting means and sensor means, standard deviations, and linear slopes.
- Fit: training windows sampled every five cycles where capped RUL is at most 125.
- Calibration: one seeded random as-of window per held-out engine in the same capped-RUL range.
- Test: one final observed window per official test engine; official RUL file used only for final metrics.
- Operating-regime clusters are fitted only on fit-engine settings. In seed 42, ridge RMSE ranged from 24.23 to 64.84 cycles across six clusters.

## Interpretation

Greater model complexity did not improve this particular fixed protocol. The calibrated intervals did not maintain nominal coverage at test endpoints, with materially different shortfalls across model families. This may reflect both endpoint/operating-condition shift and model error; it is not evidence of a causal mechanism. Next checks: compare against an endpoint-matched calibration design, report bootstrap intervals over test engines, and evaluate a regime-conditioned model. The C-MAPSS data are simulated and do not establish real-world maintenance performance.
