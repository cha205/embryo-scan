# Division timing: test videos

141 videos from patients the model never saw, 58,988 labelled frames.

- Frame-by-frame phase accuracy: 0.697 raw, **0.718** after the never-go-backwards decoding (0.928 within one phase).
- Median timing error across all divisions: **1.00 h** (frames are ~0.25 h apart).
- Meseguer 2011 category (A+ ... D-) from detected vs annotated timings: same category for 47% of 141 videos.

Timing error per phase (detected onset minus the embryologist's):

| phase | videos | median_abs | within_1h | within_2h |
|---|---|---|---|---|
| t2 | 141 | 0.20 | 0.92 | 0.94 |
| t3 | 141 | 0.30 | 0.78 | 0.84 |
| t4 | 141 | 0.50 | 0.76 | 0.87 |
| t5 | 141 | 1.00 | 0.57 | 0.70 |
| t6 | 140 | 0.80 | 0.59 | 0.73 |
| t7 | 137 | 1.00 | 0.53 | 0.69 |
| t8 | 135 | 1.70 | 0.35 | 0.56 |
| t9+ | 123 | 3.10 | 0.31 | 0.46 |
| tM | 111 | 1.70 | 0.40 | 0.56 |
| tSB | 104 | 1.30 | 0.48 | 0.66 |
| tB | 94 | 1.10 | 0.50 | 0.68 |
| tEB | 73 | 4.30 | 0.21 | 0.29 |