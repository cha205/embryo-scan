# Deploying the demonstrator

The public demonstrator consists of two components:

| Component | Host | Rationale |
|---|---|---|
| Web application (`web/`) | Vercel or Netlify | static files, which are served quickly and continuously |
| Image grading service (`server/`) | Hugging Face Space (Docker, free CPU tier) | requires Python, PyTorch and 550 MB of model weights |

All views except the grading of uploaded images operate without the grading service. A free Space is suspended after two days without traffic. Each page of the application therefore sends a request to the service when it loads, and the Grading view displays "Starting the grading service…" until the service responds, which takes approximately one minute.

In the steps below, `USER` denotes the Hugging Face user name and `APP_URL` the address of the web application, for example `https://embryo-scan.vercel.app`. The Space is built from the public GitHub repository.

## 1. Publish the trained weights

Create a Hugging Face account and an access token with write permission (Settings → Access Tokens). Then run the following commands from the repository root with the project's environment activated:

```bash
hf auth login
hf repos create USER/embryo-scan-grader
hf upload USER/embryo-scan-grader runs/blastocyst/grader-convnext-tiny . --include "fold-*.pt"
hf upload USER/embryo-scan-grader deploy/model-card.md README.md
```

## 2. Create the Space

1. On huggingface.co, create a new Space named `embryo-scan` with the Docker SDK (blank template), the free CPU basic hardware and public visibility.
2. Upload `Dockerfile` and `README.md` from `deploy/huggingface-space/`. When deploying from a fork or from a different Hugging Face account, first change `GITHUB_REPO` and `MODEL_REPO` at the top of the Dockerfile.
3. Under Settings → Variables, add `ALLOWED_ORIGINS` with the value `APP_URL`; several addresses are separated by commas.
4. The build takes approximately five minutes. The service is available once `https://USER-embryo-scan.hf.space/api/health` reports `"models": 5`.

## 3. Deploy the web application

Import the GitHub repository as a new Vercel or Netlify project with the following settings:

| Setting | Value |
|---|---|
| Root directory | `web` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_URL` = `https://USER-embryo-scan.hf.space` |

The application loads its files from the site root, so it requires its own address (a project URL or a subdomain) rather than a subdirectory of another site. If the final address differs from the value of `ALLOWED_ORIGINS`, update that variable on the Space.

## 4. Verify the deployment

Open `APP_URL/analyze.html`, wait until the upload card reads "Grade an image", and upload a day-5 embryo image. Each image is graded within a few seconds on the free CPU. The CPU service uses 2 test-time views instead of the 8 used in the evaluation, so a grade can occasionally differ from that of the evaluated configuration.
