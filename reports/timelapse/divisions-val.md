# Division timing: val partition

The evaluation comprises 73 videos from patients who were not used for training, with 31,770 annotated frames.

- Frame-level phase accuracy is 0.684 before decoding and **0.708** after monotonic decoding; 0.916 of frames are assigned to the annotated phase or an adjacent one.
- The median absolute onset error across all divisions is **0.85 h**, for a frame interval of approximately 0.25 h.
- The Meseguer et al. (2011) category (A+ to D-) derived from the detected timings agrees with the category derived from the annotations in 47% of 73 videos.

Timing error per event (detected onset minus annotated onset, in hours):

| event | videos | median absolute error (h) | within 1 h | within 2 h |
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