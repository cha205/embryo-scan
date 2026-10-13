# Division timing: test partition

The evaluation comprises 141 videos from patients who were not used for training, with 58,988 annotated frames.

- Frame-level phase accuracy is 0.697 before decoding and **0.718** after monotonic decoding; 0.928 of frames are assigned to the annotated phase or an adjacent one.
- The median absolute onset error across all divisions is **1.00 h**, for a frame interval of approximately 0.25 h.
- The Meseguer et al. (2011) category (A+ to D-) derived from the detected timings agrees with the category derived from the annotations in 47% of 141 videos.

Timing error per event (detected onset minus annotated onset, in hours):

| event | videos | median absolute error (h) | within 1 h | within 2 h |
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