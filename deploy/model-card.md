---
license: cc-by-4.0
tags:
  - image-classification
  - medical-imaging
  - ivf
  - embryology
---

# Embryo Scan blastocyst grader

This repository contains five ConvNeXt-Tiny models, trained with five-fold patient-level cross-fitting, that grade an image of a day-5 human blastocyst on the Gardner scale: expansion (1-5), inner cell mass (A-C) and trophectoderm (A-C). The code for training and evaluation is available at [github.com/cha205/embryo-scan](https://github.com/cha205/embryo-scan).

## Results

On the 300-image test set graded by a panel of embryologists (Kromp et al., 2023), the five-model ensemble with eight-view test-time augmentation reaches accuracies of 0.853 for expansion, 0.714 for the inner cell mass and 0.660 for the trophectoderm. For comparison, individual embryologists reach 0.78, 0.74 and 0.70, and the best models published with the dataset reach 0.85, 0.69 and 0.62. A control experiment that retrains the ensemble without the patients of the test set finds no evidence that patient overlap inflates these results.

## Files

`fold-0.pt` to `fold-4.pt` are PyTorch state dictionaries for `embryoscan.blastocyst.grading.Grader` (timm backbone `convnext_tiny.fb_in22k_ft_in1k` with three heads). The input is a grey-scale image of 384 × 288 pixels, and the models are loaded with `embryoscan.blastocyst.inference.load_ensemble`.

## Training data and licence

The models were trained on the blastocyst dataset of Kromp et al. (2023), *An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization*, Scientific Data ([figshare](https://doi.org/10.6084/m9.figshare.20123153.v3), CC BY 4.0). The weights are released under CC BY 4.0.

## Limitations

- All images come from a single clinic and microscope, and performance on other imaging systems has not been validated.
- The training labels were assigned by a single embryologist. The models tend to grade inner cell masses as A where the expert panel assigned B, which may reflect systematic differences between that annotator and the panel.
- The models are intended for research use only and are not a medical device.
