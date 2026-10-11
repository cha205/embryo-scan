# Baselines: live birth

Dev set: 603 embryos from 585 patients, 187 positives (31%). Patient-level 5-fold cross-validation; 95% CIs from a patient-level bootstrap. Logistic regression unless noted.

| model | AUC (95% CI) | AUC vs age only | Brier |
|---|---|---|---|
| age only | 0.533 (0.482-0.584) |  | 0.213 |
| embryologist's grade | 0.550 (0.497-0.598) | +0.017 (-0.050 to +0.085) | 0.213 |
| all clinical data | 0.467 (0.416-0.518) | -0.066 (-0.115 to -0.017) | 0.221 |
| all clinical data (LightGBM) | 0.478 (0.423-0.532) | -0.055 (-0.124 to +0.014) | 0.231 |
| embryologist's grade + clinical | 0.498 (0.445-0.548) | -0.035 (-0.095 to +0.020) | 0.221 |

AMH sensitivity (ambiguous dates read as the low vs high value): clinical-model AUC 0.469 vs 0.466.

AUC 0.5 = coin flip. Brier = mean squared error of the predicted probability (lower is better; always predicting the base rate scores 0.214).