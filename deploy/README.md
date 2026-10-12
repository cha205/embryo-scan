# Deploying the demo

The public demo has two parts:

| Part | Host | Why |
|---|---|---|
| Web app (`web/`) | Vercel or Netlify | static files, fast and always on |
| Photo grader (`server/`) | Hugging Face Space (Docker, free CPU) | needs Python, PyTorch and 550 MB of model weights |

Everything except grading your own photo works without the grader. A free Space goes to sleep after two days without traffic. Each page of the app pings it on load, and the Analyze tab shows "Waking the grader…" until it responds, which takes about a minute.

In the steps below, `USER` is your Hugging Face username and `APP_URL` is the web app's address, for example `https://embryo-scan.vercel.app`. The Space builds from the public GitHub repo.

## 1. Publish the trained weights

Make a Hugging Face account and an access token with write permission (Settings → Access Tokens). Then, from the repository root with the project's environment active:

```bash
hf auth login
hf repos create USER/embryo-scan-grader
hf upload USER/embryo-scan-grader runs/blastocyst/grader-convnext-tiny . --include "fold-*.pt"
hf upload USER/embryo-scan-grader deploy/model-card.md README.md
```

## 2. Create the Space

1. On huggingface.co, create a new Space named `embryo-scan` with the Docker SDK (blank template), the free CPU basic hardware, and public visibility.
2. Upload `Dockerfile` and `README.md` from `deploy/huggingface-space/`. If your usernames aren't `cha205`, change `GITHUB_REPO` and `MODEL_REPO` at the top of the Dockerfile first.
3. Under Settings → Variables, add `ALLOWED_ORIGINS` set to `APP_URL`. Separate several addresses with commas.
4. The build takes about 5 minutes. When `https://USER-embryo-scan.hf.space/api/health` shows `"models": 5`, the grader is up.

## 3. Deploy the web app

Import the GitHub repo as a new Vercel or Netlify project:

| Setting | Value |
|---|---|
| Root directory | `web` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_URL` = `https://USER-embryo-scan.hf.space` |

The app loads its files from the site root, so it needs its own address (a project URL or a subdomain) rather than a subfolder of another site. If the final address isn't the one in `ALLOWED_ORIGINS`, update that variable on the Space.

## 4. Check it

Open `APP_URL/analyze.html`, wait for "Grade a photo", and drop in a day-5 embryo photo. Each photo takes a few seconds on the free CPU. The CPU server uses 2 test-time views instead of the 8 used in the evaluation, so a grade can occasionally differ from the reported setup.
