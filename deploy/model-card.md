---
license: cc-by-4.0
tags:
  - image-classification
  - medical-imaging
  - ivf
  - embryology
---

# Embryo Scan blastocyst grader

Five ConvNeXt-Tiny models (5-fold cross-fitting) that grade a day-5 human blastocyst photo on the Gardner scale: expansion (1-5), inner cell mass (A-C) and trophectoderm (A-C). Code, training and evaluation: [github.com/cha205/embryo-scan](https://github.com/cha205/embryo-scan).

## Results
On the 300-photo test set graded by a panel of embryologists (Kromp et al. 2023), the 5-model ensemble with 8-view test-time augmentation scores accuracy 0.853 (expansion), 0.714 (inner cell mass) and 0.660 (trophectoderm), against 0.78 / 0.74 / 0.70 for individual embryologists and 0.85 / 0.69 / 0.62 for the best model published with the dataset. A check that retrains without the test set's patients finds no sign that patient overlap inflates these numbers.

## Files
`fold-0.pt` ... `fold-4.pt`: PyTorch state dicts for `embryoscan.blastocyst.grading.Grader` (timm `convnext_tiny.fb_in22k_ft_in1k` backbone, three heads). Input: grayscale 384 x 288. Load them with `embryoscan.blastocyst.inference.load_ensemble`.

## Training data and license
Trained on the blastocyst dataset of Kromp et al. 2023, *An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization*, Scientific Data ([figshare](https://doi.org/10.6084/m9.figshare.20123153.v3), CC BY 4.0). The weights are released under CC BY 4.0.

## Limitations
- Photos come from one clinic and one microscope; other setups may grade poorly.
- Grades follow a single embryologist's training labels, which tend to call grade-B inner cell masses "A".
- Research project, not a medical device.
