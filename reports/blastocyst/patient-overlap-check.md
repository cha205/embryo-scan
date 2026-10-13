# Effect of patient overlap on the grading results

The gold test set was drawn by image rather than by patient: 511 of 1,697 training-pool images (30%) come from 163 patients who also contribute a gold test image. Because sibling embryos resemble one another, a model could achieve higher accuracy on patients represented in its training data. Three five-fold ensembles with identical settings were therefore scored on the 300 gold images (accuracy with 95% patient-level bootstrap confidence intervals, 2,000 resamples):

| model | training images per fold | expansion | inner cell mass | trophectoderm |
|---|---|---|---|---|
| whole pool (as reported) | 1,358 | 0.853 (0.81-0.89) | 0.714 (0.66-0.77) | 0.660 (0.60-0.72) |
| without gold patients | 949 | 0.847 (0.80-0.88) | 0.733 (0.68-0.79) | 0.650 (0.59-0.71) |
| size-matched control | 949 | 0.853 (0.81-0.89) | 0.724 (0.67-0.78) | 0.646 (0.59-0.70) |

Paired differences were computed on identical resamples (95% CI). If the overlap inflated the results, the differences in the first row would be clearly negative:

| comparison | expansion | inner cell mass | trophectoderm |
|---|---|---|---|
| without gold patients minus size-matched control (overlap effect) | -0.007 (-0.030 to +0.016) | +0.008 (-0.016 to +0.034) | +0.004 (-0.028 to +0.035) |
| size-matched control minus whole pool (effect of reduced training data) | +0.000 (-0.026 to +0.026) | +0.011 (-0.013 to +0.036) | -0.014 (-0.041 to +0.013) |

Accuracy (expansion / ICM / TE) on gold images from patients represented in the training pool (197 images) and from patients absent from it (103 images):

| model | patient in the training pool | patient not in the pool |
|---|---|---|
| whole pool (as reported) | 0.858 / 0.697 / 0.684 | 0.845 / 0.744 / 0.616 |
| without gold patients | 0.858 / 0.739 / 0.675 | 0.825 / 0.721 / 0.605 |
| size-matched control | 0.858 / 0.720 / 0.662 | 0.845 / 0.733 / 0.616 |

The intervals reflect the sampling of embryos in the test set, not the randomness of training; the single models of the reported run differ by up to 0.04 in expansion accuracy from one fold to the next.