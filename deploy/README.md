# Deploying the demonstrator

The public demonstrator consists of two components:

| Component | Host | Rationale |
|---|---|---|
| Web application (`web/`) | Vercel or Netlify | static files, which are served quickly and continuously |
| Image grading service (`server/`) | Google Cloud Run (container, scales to zero) | requires Python, PyTorch and 550 MB of model weights |

All views except the grading of uploaded images operate without the grading service. The service scales to zero when idle, so the first request after a quiet period starts a new instance and loads the models. Each page of the application therefore sends a request to the service when it loads, and the Grading view displays "Starting the grading service…" until the service responds.

In the steps below, `USER` denotes the Hugging Face user name, `PROJECT` the Google Cloud project and `APP_URL` the address of the web application, for example `https://embryo-scan.vercel.app`. The container is built from the public GitHub repository.

## 1. Publish the trained weights

Create a Hugging Face account and an access token with write permission (Settings → Access Tokens). Then run the following commands from the repository root with the project's environment activated:

```bash
hf auth login
hf repos create USER/embryo-scan-grader
hf upload USER/embryo-scan-grader runs/blastocyst/grader-convnext-tiny . --include "fold-*.pt"
hf upload USER/embryo-scan-grader deploy/model-card.md README.md
```

## 2. Deploy the grading service

Install the Google Cloud CLI, sign in with `gcloud auth login` and choose a project with billing enabled. When deploying from a fork or from a different Hugging Face account, first change `GITHUB_REPO` and `MODEL_REPO` at the top of `deploy/grader/Dockerfile`. Then run the following command from the repository root:

```bash
gcloud run deploy embryo-scan-grader --source deploy/grader --region us-west1 --project PROJECT --allow-unauthenticated --port 7860 --memory 2Gi --cpu 2 --min-instances 0 --max-instances 2 --concurrency 4 --timeout 120 --set-env-vars ALLOWED_ORIGINS=APP_URL
```

Cloud Build builds the image, which takes approximately ten minutes, and the command prints the service address (`SERVICE_URL`). The service is available once `SERVICE_URL/api/health` reports `"models": 5`. With no minimum instances the service incurs no compute charges while idle, and the instance limit caps the cost of heavy use. Several addresses in `ALLOWED_ORIGINS` are separated by commas.

## 3. Deploy the web application

Import the GitHub repository as a new Vercel or Netlify project with the following settings:

| Setting | Value |
|---|---|
| Root directory | `web` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_URL` = `SERVICE_URL` |

The application loads its files from the site root, so it requires its own address (a project URL or a subdomain) rather than a subdirectory of another site. If the final address differs from the value of `ALLOWED_ORIGINS`, update it:

```bash
gcloud run services update embryo-scan-grader --region us-west1 --project PROJECT --update-env-vars ALLOWED_ORIGINS=APP_URL
```

## 4. Verify the deployment

Open `APP_URL/analyze.html`, wait until the upload card reads "Grade an image", and upload a day-5 embryo image. Each image is graded within a few seconds once the service is running. The CPU service uses 2 test-time views instead of the 8 used in the evaluation, so a grade can occasionally differ from that of the evaluated configuration.
