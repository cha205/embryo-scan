---
title: Embryo Scan grader
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Grades day-5 IVF embryo photos (research demo)
---

# Embryo Scan photo grader

The API behind the "grade your own photo" feature of [Embryo Scan](https://github.com/cha205/embryo-scan): a 5-model ConvNeXt-Tiny ensemble grades a day-5 blastocyst photo on the Gardner scale and returns Grad-CAM heat maps of what it looked at.

- `GET /api/health`: whether the models are loaded
- `POST /api/grade`: multipart upload of a PNG or JPG, field `file`

Runs on a free CPU, so it uses fewer test-time views than the evaluated setup (2 instead of 8) to answer in a few seconds. Research demo, not for medical use.
