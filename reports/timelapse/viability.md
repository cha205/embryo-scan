# Viability ranking from day 0-3 division timings

Outcome: blastocyst formed by day 5 (116 h). Videos recorded at least that long: 486 (58% formed a blastocyst). Logistic regression trained on 389 train/val videos; tested on 97 videos from held-out patients (61 blastocysts). AUC with 95% patient-bootstrap CI.

| timings from | ranking method | AUC (95% CI) |
|---|---|---|
| embryologist's timings | Meseguer 2011 categories | 0.675 (0.551-0.779) |
| embryologist's timings | logistic regression on timings | 0.717 (0.611-0.809) |
| AI-detected timings (end to end) | Meseguer 2011 categories | 0.610 (0.480-0.727) |
| AI-detected timings (end to end) | logistic regression on timings | 0.690 (0.566-0.794) |

Blastocyst rate by Meseguer category (embryologist's timings, test videos):

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