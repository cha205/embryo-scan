# Classical baseline: division timing without machine learning

The same 73 held-out val videos as [divisions-val.md](divisions-val.md), timed by a hand-built MATLAB pipeline (Image Processing Toolbox, nothing trained) instead of the deep model. Code: [matlab/](../../matlab).

1. **Find the embryo**: circular Hough transform (`imfindcircles`) on the median of the first 20 frames found it in 67 of 73 videos; in the rest, the most textured region (cells are grainy, the culture medium is smooth) stood in.
2. **Motion energy**: the mean change between consecutive frames inside the embryo, in units of the video's own noise. A cell splitting in two changes many pixels at once, so divisions show up as spikes.
3. **Rules from embryology**: t2 is the biggest spike between 18 and 38 h, t3 the first spike 5 to 18 h later, t4 a spike within 4 h of t3 (none means both cells split together), t5 the first spike 6 to 20 h after t4.

Timing error per division (detected minus the embryologist's time). A division a method missed counts as a miss in the within-1 h and within-2 h columns; median errors are over the divisions it found. The deep model's numbers are recomputed from its saved predictions with the same rules.

| division | videos | found by classical | median error, classical | median error, deep | within 1 h, classical | within 1 h, deep | within 2 h, classical | within 2 h, deep |
|---|---|---|---|---|---|---|---|---|
| t2 | 73 | 89% | 0.70 h | 0.20 h | 51% | 96% | 59% | 96% |
| t3 | 73 | 86% | 1.30 h | 0.50 h | 41% | 64% | 52% | 81% |
| t4 | 73 | 86% | 0.80 h | 0.50 h | 45% | 73% | 53% | 84% |
| t5 | 72 | 75% | 2.50 h | 0.75 h | 24% | 64% | 36% | 76% |

- **Meseguer 2011 category** (A+ ... D-) matches the one from the embryologist's times for 27% of 63 videos with the classical times and 47% of 73 with the deep model's.
