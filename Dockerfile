# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Lightweight ffmpeg: imageio-ffmpeg (requirements.txt) ships a single static
# ffmpeg binary (~25 MB, includes all audio decoders + libmp3lame) — zero apt
# packages needed, image stays slim. Decode = any format, encode = always MP3.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # keep torch single-op friendly on small ACA cores; override via env
    OMP_NUM_THREADS=2 \
    TORCH_HOME=/opt/torch-cache

WORKDIR /srv
COPY requirements.txt .
# Two steps on purpose: requirements.txt holds everything with real deps
# (torch CPU + API stack), then denoiser with --no-deps — its PyPI metadata
# drags in training-only junk (hydra, pystoi, sounddevice, torchaudio) that
# inference never imports (verified: pretrained/demucs/utils import torch only).
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && pip install --no-deps denoiser==0.1.5

COPY app ./app

# Bake model weights into the image so cold starts don't download from GitHub.
ARG DENOISER_MODEL=dns48
ENV DENOISER_MODEL_NAME=${DENOISER_MODEL}
RUN python -c "from denoiser import pretrained; getattr(pretrained, '${DENOISER_MODEL}')(); print('weights cached')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
