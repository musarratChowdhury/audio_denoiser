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


def test_reject_bad_extension():
    with TestClient(app) as client:
        r = client.post("/api/enhance", files={"file": ("evil.exe", b"xx")})
        assert r.status_code == 400


def test_reject_m4a_without_ffmpeg():
    # No ffmpeg in the image: AAC-in-mp4 (m4a/mp4/webm) is unsupported by design.
    # libsndfile (bundled with soundfile) covers wav/mp3/flac/ogg/aiff.
    with TestClient(app) as client:
        r = client.post("/api/enhance", files={"file": ("clip.m4a", b"xx")})
        assert r.status_code == 400
        assert "unsupported type" in r.text
