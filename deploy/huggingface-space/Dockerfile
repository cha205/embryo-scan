# Hugging Face Space (Docker, free CPU) that serves the Embryo Scan photo grader.
# It builds from the public GitHub repo and downloads the trained weights from a
# Hugging Face model repo; see docs/deploy.md.
FROM python:3.12-slim

ARG GITHUB_REPO=https://github.com/cha205/embryo-scan.git
ARG MODEL_REPO=cha205/embryo-scan-grader

# git for the clone; OpenCV (used by pytorch-grad-cam) needs the GL and GLib libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN git clone --depth 1 "$GITHUB_REPO" app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r deploy/requirements-server.txt \
    && pip install --no-cache-dir --user --no-deps -e .
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', local_dir='runs/blastocyst/grader-convnext-tiny', allow_patterns=['fold-*.pt'])"

EXPOSE 7860
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
