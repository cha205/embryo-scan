# Embryo Scan image grading service

This container provides the API behind the image-upload function of [Embryo Scan](https://github.com/cha205/embryo-scan). A five-model ConvNeXt-Tiny ensemble grades a day-5 blastocyst image on the Gardner scale and returns a Grad-CAM attribution map for each grading criterion.

- `GET /api/health` reports whether the models are loaded.
- `POST /api/grade` accepts a multipart upload of a PNG or JPEG image in the field `file`.

The service runs on CPU (Google Cloud Run) and therefore uses fewer test-time views than the evaluated configuration (2 instead of 8), which allows it to respond within a few seconds. It is intended for research use only and is not suitable for clinical decision-making.
