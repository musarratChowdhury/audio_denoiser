# Audio Denoiser (Demucs on FastAPI → Azure Container Apps)

Upload an audio file, get the denoised version back. Powered by
[facebookresearch/denoiser](https://github.com/facebookresearch/denoiser)
(Demucs waveform speech enhancement, Interspeech 2020).

> ⚠️ **License**: upstream is **CC-BY-NC 4.0** — non-commercial/research use only.
> Don't deploy this commercially without replacing the model.

## Quickstart (local)

```bash
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -r requirements.txt  # CPU torch (~2 GB download)
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

First run downloads `dns48` weights (~75 MB) to the torch hub cache.

## Docker

```bash
docker build -t denoiser .
docker run -p 8000:8000 denoiser
```

Build with the faster/lighter model: `docker build --build-arg DENOISER_MODEL=dns48 .`

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | upload UI |
| `GET` | `/health` | liveness |
| `POST` | `/api/enhance` | multipart `file` + optional `dry` (0–1) → denoised WAV (+ `X-Job-Id`) |
| `GET` | `/api/download/{job_id}` | re-download within TTL |

```bash
curl -F file=@noisy.mp3 -F dry=0 http://localhost:8000/api/enhance -o clean.wav
```

Models: `dns48` (default — fastest, smallest, RTF ~0.6–0.8 on laptop CPU),
`dns64` (balanced), `master64` (best quality, slowest). Set via `DENOISER_MODEL_NAME`.
Clips over `DENOISER_STREAM_SECONDS` (default 60s) use constant-memory
streaming inference; shorter clips use a full-utterance forward.

## CI/CD → GHCR → Azure Container Apps

* `.github/workflows/ci.yml`: pytest → buildx → push `ghcr.io/<owner>/audio_denoiser:sha-…` + `:latest` → `az containerapp update` (needs `AZURE_*`/`ACA_*` secrets).
* Full ACA steps + **file-upload strategy (ephemeral vs Blob vs Files)** → [`infra/containerapp.md`](infra/containerapp.md).

## Config (env)

| Var | Default | Notes |
|---|---|---|
| `DENOISER_MODEL_NAME` | `dns48` | `dns48`/`dns64`/`master64` |
| `DENOISER_DEVICE` | `cpu` | ACA is CPU unless you attach GPU workload |
| `DENOISER_TORCH_THREADS` | `2` | match container vCPU |
| `DENOISER_MAX_UPLOAD_MB` | `50` | ingress/ephemeral friendly |
| `DENOISER_MAX_AUDIO_SECONDS` | `300` | bounds CPU time per request |
| `DENOISER_JOB_DIR` | `/tmp/denoiser_jobs` | point at Azure Files mount for shared disk |
| `DENOISER_JOB_TTL_SECONDS` | `3600` | re-download window |
| `DENOISER_CORS_ORIGINS` | `` | e.g. your frontend domain |
