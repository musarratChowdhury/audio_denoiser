from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_index_renders():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Audio Denoiser" in r.text


def test_reject_unsupported_extension():
    with TestClient(app) as client:
        r = client.post("/api/enhance", files={"file": ("evil.exe", b"xx")})
        assert r.status_code == 400


def test_accept_all_audio_extensions():
    # With the lightweight static ffmpeg (imageio-ffmpeg), every audio
    # container is accepted at validation time — m4a/mp4/webm decode via
    # ffmpeg (audio track extracted, video dropped). Only truly unknown
    # suffixes are rejected. Garbage bytes still fail, but at decode time
    # (500), not at validation (400).
    from app.audio_io import ALLOWED_EXTS

    assert ".m4a" in ALLOWED_EXTS and ".webm" in ALLOWED_EXTS
    assert ".wma" in ALLOWED_EXTS and ".opus" in ALLOWED_EXTS
    with TestClient(app) as client:
        r = client.post("/api/enhance", files={"file": ("clip.m4a", b"xx")})
        assert r.status_code in (400, 500)
        assert "unsupported type" not in r.text
