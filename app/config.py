"""Centralised runtime configuration (env-var driven for Azure Container Apps)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DENOISER_")

    app_name: str = "audio-denoiser"
    # dns48 = fastest + smallest (~75 MB weights, RTF ~0.6-0.8 on laptop CPU).
    # dns64 = balanced, master64 = best quality (slower, larger).
    model_name: str = "dns48"
    device: str = "cpu"
    # seconds of threads torch may use; 1-4 is the sweet spot on ACA consumption workload
    torch_threads: int = 2
    # clips longer than this use constant-memory streaming inference
    # (a full-utterance forward holds GBs of activations — OOM risk on 4 GiB)
    stream_seconds: int = 60
    # upload guardrails (ACA ingress + ephemeral disk friendly)
    max_upload_mb: int = 50
    max_audio_seconds: int = 300  # 5 min cap to bound CPU time per request
    job_dir: str = "/tmp/denoiser_jobs"
    job_ttl_seconds: int = 3600  # re-download window before cleanup
    # Comma-separated extra origins for CORS (ACA FQDN goes here in prod)
    cors_origins: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
