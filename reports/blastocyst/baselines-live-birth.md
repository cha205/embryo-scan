# Baseline models: live birth

The development set comprises 603 embryos from 585 patients, with 187 positive outcomes (31%). Models were evaluated by patient-level five-fold cross-validation, with 95% confidence intervals from a patient-level bootstrap, and are logistic regressions unless stated otherwise.

| model | AUC (95% CI) | AUC difference vs. age only | Brier score |
|---|---|---|---|
| age only | 0.533 (0.482-0.584) |  | 0.213 |
| embryologist's grade | 0.550 (0.497-0.598) | +0.017 (-0.050 to +0.085) | 0.213 |
| all clinical data | 0.467 (0.416-0.518) | -0.066 (-0.115 to -0.017) | 0.221 |
| all clinical data (LightGBM) | 0.478 (0.423-0.532) | -0.055 (-0.124 to +0.014) | 0.231 |
| embryologist's grade + clinical | 0.498 (0.445-0.548) | -0.035 (-0.095 to +0.020) | 0.221 |

In a sensitivity analysis in which ambiguous AMH values were read as the lower or the higher candidate value, the AUC of the clinical model was 0.469 and 0.466, respectively.

An AUC of 0.5 corresponds to chance-level discrimination. The Brier score is the mean squared error of the predicted probability (lower is better); a constant prediction of the base rate scores 0.214.