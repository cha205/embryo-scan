# Image models vs baselines: lb

Dev set: 603 embryos, 585 patients, 187 positives. Patient-level 5-fold CV, 95% CIs from a patient-level bootstrap; differences are paired.

| model | AUC (95% CI) | vs age only | vs embryologist's grade | Brier |
|---|---|---|---|---|
| age only | 0.533 (0.482-0.584) |  | -0.017 (-0.085 to +0.050) | 0.213 |
| embryologist's grade | 0.550 (0.497-0.598) | +0.017 (-0.050 to +0.085) |  | 0.213 |
| AI grade | 0.557 (0.508-0.605) | +0.024 (-0.041 to +0.091) | +0.008 (-0.040 to +0.056) | 0.214 |
| AI embedding | 0.518 (0.470-0.566) | -0.015 (-0.078 to +0.055) | -0.032 (-0.088 to +0.025) | 0.218 |
| AI grade + clinical | 0.530 (0.481-0.577) | -0.003 (-0.058 to +0.055) | -0.019 (-0.070 to +0.032) | 0.220 |
| AI embedding + clinical | 0.502 (0.452-0.552) | -0.031 (-0.089 to +0.032) | -0.048 (-0.101 to +0.009) | 0.223 |
| AI grade + embryologist's grade + age | 0.564 (0.514-0.612) | +0.031 (-0.025 to +0.089) | +0.015 (-0.030 to +0.059) | 0.213 |