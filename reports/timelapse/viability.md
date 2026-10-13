# Viability ranking from day 0-3 division timings

The outcome is blastocyst formation by day 5 (116 h). Of the 486 videos recorded for at least that long, 58% formed a blastocyst. The logistic regression was trained on 389 training and validation videos and evaluated on 97 videos from held-out patients, 61 of which formed a blastocyst. AUCs are reported with 95% patient-level bootstrap confidence intervals.

| timings from | ranking method | AUC (95% CI) |
|---|---|---|
| embryologist's annotations | Meseguer 2011 categories | 0.675 (0.551-0.779) |
| embryologist's annotations | logistic regression on timings | 0.717 (0.611-0.809) |
| phase model, end to end | Meseguer 2011 categories | 0.610 (0.480-0.727) |
| phase model, end to end | logistic regression on timings | 0.690 (0.566-0.794) |

Blastocyst formation rate by Meseguer category (annotated timings, test videos):

| Meseguer category | videos | blastocyst by day 5 |
|---|---|---|
| A+ | 16 | 0.88 |
| A- | 14 | 0.57 |
| B+ | 8 | 0.88 |
| B- | 7 | 0.43 |
| C+ | 13 | 0.85 |
| C- | 8 | 0.50 |
| D+ | 21 | 0.57 |
| D- | 10 | 0.20 |