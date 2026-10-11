# Division timing: val videos

73 videos from patients the model never saw, 31,770 labelled frames.

- Frame-by-frame phase accuracy: 0.684 raw, **0.708** after the never-go-backwards decoding (0.916 within one phase).
- Median timing error across all divisions: **0.85 h** (frames are ~0.25 h apart).
- Meseguer 2011 category (A+ ... D-) from detected vs annotated timings: same category for 47% of 73 videos.

Timing error per phase (detected onset minus the embryologist's):

| phase | videos | median_abs | within_1h | within_2h |
|---|---|---|---|---|
| t2 | 73 | 0.20 | 0.96 | 0.96 |
| t3 | 73 | 0.50 | 0.64 | 0.81 |
| t4 | 73 | 0.50 | 0.73 | 0.84 |
| t5 | 72 | 0.75 | 0.64 | 0.76 |
| t6 | 72 | 0.60 | 0.67 | 0.75 |
| t7 | 71 | 1.50 | 0.45 | 0.62 |
| t8 | 71 | 1.30 | 0.44 | 0.61 |
| t9+ | 64 | 2.90 | 0.28 | 0.47 |
| tM | 59 | 2.00 | 0.24 | 0.54 |
| tSB | 55 | 1.00 | 0.53 | 0.67 |
| tB | 45 | 1.00 | 0.53 | 0.64 |
| tEB | 34 | 2.00 | 0.41 | 0.50 |