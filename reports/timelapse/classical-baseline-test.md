# Classical baseline: division timing without machine learning

The 141 held-out test videos of [divisions-test.md](divisions-test.md) were timed by a hand-built MATLAB pipeline (Image Processing Toolbox, without any trained component) instead of the deep model. The code is in [matlab/](../../matlab).

1. **Embryo localization.** A circular Hough transform (`imfindcircles`) applied to the median of the first 20 frames located the embryo in 134 of 141 videos; in the remaining videos, the most textured region was used instead, because the cells appear granular whereas the culture medium is homogeneous.
2. **Motion energy.** The mean change between consecutive frames inside the embryo is expressed in units of the video's own noise level. A cleaving cell changes many pixels at once, so divisions appear as spikes.
3. **Timing rules derived from embryology.** t2 is the strongest spike between 18 and 38 h, t3 the first spike 5 to 18 h later, t4 a spike within 4 h of t3 (its absence indicates that both cells divided synchronously), and t5 the first spike 6 to 20 h after t4.

The timing error per division is the detected minus the annotated time. A division that a method failed to detect counts as a failure in the within-1 h and within-2 h columns, and median errors are computed over the detected divisions. The deep model's results are recomputed from its saved predictions with the same rules.

| division | videos | detected by classical | median error, classical | median error, deep | within 1 h, classical | within 1 h, deep | within 2 h, classical | within 2 h, deep |
|---|---|---|---|---|---|---|---|---|
| t2 | 141 | 94% | 1.50 h | 0.20 h | 44% | 92% | 51% | 94% |
| t3 | 141 | 92% | 3.05 h | 0.30 h | 34% | 78% | 43% | 84% |
| t4 | 141 | 92% | 1.30 h | 0.50 h | 45% | 76% | 50% | 87% |
| t5 | 141 | 82% | 3.70 h | 1.00 h | 22% | 57% | 33% | 70% |

- **Meseguer et al. (2011) category** (A+ to D-): the category derived from the classical timings agrees with the category derived from the annotations in 25% of 130 videos, compared with 47% of 141 videos for the deep model.

![Motion energy for one representative video](../figures/timelapse-classical-motion-energy.png)

![Divisions timed within 1 h by the classical pipeline and the deep model](../figures/timelapse-classical-vs-deep.png)
