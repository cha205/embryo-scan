# Deploying the demo

The public demo is two free pieces:

| piece | host | why |
|---|---|---|
| The web app (`web/`) | Vercel or Netlify | static files on a CDN: loads instantly, always on |
| The photo grader (`server/`) | Hugging Face Space (Docker, free CPU) | needs Python, PyTorch and 550 MB of model weights |

Everything except "grade your own photo" works without the grader. A free Space sleeps after two days without visitors; every page of the app pings it on load, and the Analyze tab shows "Waking the grader…" until it answers (about a minute).

Placeholders below: `USER` is your Hugging Face username, `APP_URL` is the address Vercel or Netlify gives the app (e.g. `https://embryo-scan.vercel.app`).

## 1. Make the GitHub repo public
The Space builds from it. GitHub: Settings → General → Danger Zone → Change visibility.

## 2. Publish the trained weights (Hugging Face model repo)
Create a free account at huggingface.co, then an access token with write permission (Settings → Access Tokens). From the repository root, with the project's environment active (it includes the `hf` command):

```bash
hf auth login
hf repos create USER/embryo-scan-grader
hf upload USER/embryo-scan-grader runs/blastocyst/grader-convnext-tiny . --include "fold-*.pt"
hf upload USER/embryo-scan-grader deploy/model-card.md README.md
```

## 3. Create the Space (the grader)
1. huggingface.co → New Space: name `embryo-scan`, SDK **Docker** (blank template), hardware **CPU basic** (free), public.
2. Add the two files from `deploy/huggingface-space/` (Files → Add file → Upload): `Dockerfile` and `README.md`. If your usernames aren't `cha205`, edit `GITHUB_REPO` and `MODEL_REPO` at the top of the Dockerfile first.
3. Settings → Variables: add `ALLOWED_ORIGINS` = `APP_URL` (comma-separate several, e.g. a custom domain too).
4. Wait for the build (about 5 minutes), then check `https://USER-embryo-scan.hf.space/api/health` shows `"models": 5`.

## 4. Deploy the web app (Vercel or Netlify)
Import the GitHub repo as a new project with:

| setting | value |
|---|---|
| Root directory | `web` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_URL` = `https://USER-embryo-scan.hf.space` |

Give it its own address (a project URL or a subdomain): the app loads its files from the site root, so it can't live in a subfolder of another site. If the final address differs from what you put in `ALLOWED_ORIGINS`, update that variable on the Space.

## 5. Check it
Open `APP_URL/analyze.html`, wait for "Grade a photo", and drop in a day-5 photo. A free CPU takes a few seconds per photo; it uses 2 test-time views instead of the 8 used in the evaluation, so a grade can occasionally differ from the reported setup.
