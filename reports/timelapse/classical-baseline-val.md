# Classical baseline: division timing without machine learning

The 73 held-out val videos of [divisions-val.md](divisions-val.md) were timed by a hand-built MATLAB pipeline (Image Processing Toolbox, without any trained component) instead of the deep model. The code is in [matlab/](../../matlab).

1. **Embryo localization.** A circular Hough transform (`imfindcircles`) applied to the median of the first 20 frames located the embryo in 67 of 73 videos; in the remaining videos, the most textured region was used instead, because the cells appear granular whereas the culture medium is homogeneous.
2. **Motion energy.** The mean change between consecutive frames inside the embryo is expressed in units of the video's own noise level. A cleaving cell changes many pixels at once, so divisions appear as spikes.
3. **Timing rules derived from embryology.** t2 is the strongest spike between 18 and 38 h, t3 the first spike 5 to 18 h later, t4 a spike within 4 h of t3 (its absence indicates that both cells divided synchronously), and t5 the first spike 6 to 20 h after t4.

The timing error per division is the detected minus the annotated time. A division that a method failed to detect counts as a failure in the within-1 h and within-2 h columns, and median errors are computed over the detected divisions. The deep model's results are recomputed from its saved predictions with the same rules.

| division | videos | detected by classical | median error, classical | median error, deep | within 1 h, classical | within 1 h, deep | within 2 h, classical | within 2 h, deep |
|---|---|---|---|---|---|---|---|---|
| t2 | 73 | 89% | 0.70 h | 0.20 h | 51% | 96% | 59% | 96% |
| t3 | 73 | 86% | 1.30 h | 0.50 h | 41% | 64% | 52% | 81% |
| t4 | 73 | 86% | 0.80 h | 0.50 h | 45% | 73% | 53% | 84% |
| t5 | 72 | 75% | 2.50 h | 0.75 h | 24% | 64% | 36% | 76% |

- **Meseguer et al. (2011) category** (A+ to D-): the category derived from the classical timings agrees with the category derived from the annotations in 27% of 63 videos, compared with 47% of 73 videos for the deep model.
