"""FastAPI front door: upload -> denoise -> download.

Storage model (deliberate, ACA-friendly):
  * MVP/default: ephemeral local disk (job_dir, /tmp). File is processed
    in-container and streamed back. Survives ~job_ttl_seconds for re-download,
    wiped on restart/scale — no infra to manage.
  * Prod path for large files / persistence: Azure Blob Storage
    (see infra/containerapp.md — direct-to-blob upload via SAS + output blob).
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from .audio_io import ALLOWED_EXTS
from .config import get_settings
from .denoiser_service import (
    enhance_wav,
    get_model,
    load_audio_mono,
    model_sample_rate,
    write_mp3,
)

log = logging.getLogger("uvicorn.error")
settings = get_settings()


def _job_paths(job_id: str) -> tuple[Path, Path]:
    job = Path(settings.job_dir) / job_id
    return job / "input", job / "output.mp3"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model at startup so the first user doesn't pay the download+load cost.
    # (Weights ~90-150 MB, cached under ~/.cache/torch/hub + denoiser checkpoints.)
    try:
        get_model(settings.model_name, settings.device, settings.torch_threads)
        log.info("denoiser model warm: %s", settings.model_name)
    except Exception as exc:  # don't crash boot if offline; fail lazily per-request
        log.warning("model preload failed (%s); will retry per request", exc)
    Path(settings.job_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


def _sweep_old_jobs() -> None:
    cutoff = time.time() - settings.job_ttl_seconds
    root = Path(settings.job_dir)
    if not root.exists():
        return
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"model_name": settings.model_name,
         "max_mb": settings.max_upload_mb, "max_sec": settings.max_audio_seconds},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_name}


@app.post("/api/enhance")
async def enhance(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    dry: float = Form(0.0),
):
    """Accept audio, denoise synchronously, keep output for TTL, stream it back."""
    _sweep_old_jobs()
    ext = Path(file.filename or "audio").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported type {ext!r}; use: {sorted(ALLOWED_EXTS)}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(settings.job_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / f"input{ext}"
    out_path = job_dir / "output.mp3"

    size = 0
    with open(in_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"file too large (>{settings.max_upload_mb} MB)")
            f.write(chunk)

    try:
        model = get_model(settings.model_name, settings.device, settings.torch_threads)
        sr = model_sample_rate()
        wav = load_audio_mono(in_path, sr)
        dur = wav.shape[1] / sr
        if dur > settings.max_audio_seconds:
            raise HTTPException(413, f"audio too long ({dur:.0f}s > {settings.max_audio_seconds}s)")
        if not (0.0 <= float(dry) <= 1.0):
            raise HTTPException(400, "dry must be between 0 and 1")
        t0 = time.perf_counter()
        enhanced, mode = enhance_wav(wav, model, settings.device, float(dry),
                                     sr=sr, stream_threshold_s=settings.stream_seconds)
        write_mp3(out_path, enhanced, sr, bitrate=settings.mp3_bitrate)
        took = time.perf_counter() - t0
        log.info("job=%s dur=%.1fs took=%.1fs rtf=%.2f mode=%s", job_id, dur, took, took / max(dur, 1e-3), mode)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        log.exception("enhance failed")
        raise HTTPException(500, f"denoise failed: {exc}") from exc

    # refresh mtime so TTL counts from last download; cleanup is best-effort
    background.add_task(lambda: Path(job_dir).touch(exist_ok=True))
    return FileResponse(
        out_path,
        media_type="audio/mpeg",
        filename=f"{Path(file.filename or 'audio').stem}_denoised.mp3",
        headers={"X-Job-Id": job_id},
    )


@app.get("/api/download/{job_id}")
def download(job_id: str):
    _, out_path = _job_paths(job_id)
    if not out_path.exists():
        raise HTTPException(404, "job expired or unknown (ephemeral disk, TTL exceeded?)")
    return FileResponse(out_path, media_type="audio/mpeg",
                        filename=f"{job_id}_denoised.mp3")
