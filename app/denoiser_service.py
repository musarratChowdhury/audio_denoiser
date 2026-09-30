"""Model singleton + pure inference helpers.

Wraps facebookresearch/denoiser pretrained models (dns48 / dns64 / master64).
Models are mono 16 kHz Demucs variants; we resample/mixdown on the way in.

Audio I/O (see app/audio_io.py):
  * Decode: libsndfile fast path for wav/flac/ogg/mp3/aiff, otherwise the
    lightweight static ffmpeg binary (imageio-ffmpeg) decodes anything else
    (m4a/aac/opus/webm/wma/amr/3gp/mkv/mov/...) to mono PCM.
  * Encode: final output is ALWAYS MP3 (libmp3lame) — the API never returns WAV.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_model_name_loaded: str | None = None
_sample_rate_loaded: int = 16_000


def _load_pretrained(name: str):
    from denoiser import pretrained

    name = name.lower().strip()
    if name == "dns48":
        return pretrained.dns48()
    if name == "master64":
        return pretrained.master64()
    # default: balanced quality/speed
    return pretrained.dns64()


def get_model(name: str, device: str = "cpu", torch_threads: int = 2):
    """Load once, reuse across requests. Thread-safe lazy singleton."""
    import torch

    global _model, _model_name_loaded, _sample_rate_loaded
    with _lock:
        if _model is None or _model_name_loaded != name.lower().strip():
            log.info("loading denoiser model %s on %s …", name, device)
            torch.set_num_threads(max(1, torch_threads))
            m = _load_pretrained(name)
            m.eval()
            m.to(device)
            for p in m.parameters():
                p.requires_grad_(False)
            _model = m
            _model_name_loaded = name.lower().strip()
            _sample_rate_loaded = int(getattr(m, "sample_rate", 16_000))
            log.info("model %s ready (sr=%d)", _model_name_loaded, _sample_rate_loaded)
        return _model


def model_sample_rate() -> int:
    return _sample_rate_loaded


def load_audio_mono(path: str | Path, target_sr: int):
    """Read any ffmpeg/soundfile-decodable audio -> mono Tensor [1, T] at target_sr.

    Fast path: libsndfile (soundfile) for wav/flac/ogg/mp3/aiff.
    Fallback: static ffmpeg for everything else (m4a/aac/opus/webm/wma/...).
    """
    import numpy as np
    import torch
    import julius

    from .audio_io import SOUNDFILE_NATIVE_EXTS, decode_to_mono_float32

    ext = Path(path).suffix.lower()
    tensor = None
    if ext in SOUNDFILE_NATIVE_EXTS:
        try:
            import soundfile as sf

            wav, sr = sf.read(str(path), always_2d=True)  # (T, C)
            wav = wav.T  # (C, T)
            if wav.shape[0] > 1:
                wav = wav.mean(axis=0, keepdims=True)
            tensor = torch.from_numpy(wav.astype(np.float32))
            if sr != target_sr:
                tensor = julius.resample_frac(tensor, sr, target_sr)
        except Exception:
            tensor = None  # fall through to ffmpeg
    if tensor is None:
        tensor = decode_to_mono_float32(path, target_sr)
    # guard against empty / extremely long inputs upstream; hard-clip here too
    if tensor.numel() == 0:
        raise ValueError("empty audio file")
    # clamp to [-1, 1] to avoid clipping blow-ups
    tensor = tensor.clamp(-1.0, 1.0)
    return tensor


def denoise_tensor(wav, model, device: str, dry: float = 0.0):
    """Run full-utterance (non-streaming) enhancement. wav: [1, T] at model sr."""
    import torch

    dry = float(min(1.0, max(0.0, dry)))
    with torch.inference_mode():
        noisy = wav.to(device)[None]  # [1, 1, T]
        est = model(noisy)  # [1, 1, T]
        if dry > 0:
            est = (1.0 - dry) * est + dry * noisy
        out = est[0].cpu()
        # normalise only if it prevents clipping (same rule as upstream enhance.py)
        peak = out.abs().max().item()
        if peak > 1.0:
            out = out / peak
        return out.clamp(-1.0, 1.0)


def denoise_streaming(wav, model, device: str, dry: float = 0.0,
                      chunk_seconds: int = 30, sr: int = 16_000):
    """Constant-memory enhancement via DemucsStreamer (same pattern as upstream
    `enhance.py --streaming`). Feeds the utterance in slices so peak RAM does
    not grow with file length. wav: [1, T] at model sr."""
    import torch
    from denoiser.demucs import DemucsStreamer

    dry = float(min(1.0, max(0.0, dry)))
    n = wav.shape[1]
    step = max(1, int(chunk_seconds * sr))
    streamer = DemucsStreamer(model, dry=dry)
    outs = []
    with torch.inference_mode():
        w = wav.to(device)
        for start in range(0, n, step):
            outs.append(streamer.feed(w[:, start:start + step]))
        outs.append(streamer.flush())
        out = torch.cat(outs, dim=1)[:, :n].cpu()
    peak = out.abs().max().item()
    if peak > 1.0:
        out = out / peak
    return out.clamp(-1.0, 1.0)


def enhance_wav(wav, model, device: str, dry: float = 0.0,
                sr: int = 16_000, stream_threshold_s: int = 60):
    """Offline forward for short clips, constant-memory streaming past the
    threshold. A full 5-min forward holds GBs of skip activations (encoder
    feature maps scale with length), which risks OOM on a 4 GiB container —
    hence the streaming path for long files. Returns (audio, mode)."""
    dur = wav.shape[1] / sr
    if dur > stream_threshold_s:
        return denoise_streaming(wav, model, device, dry, sr=sr), "streaming"
    return denoise_tensor(wav, model, device, dry), "offline"


def write_wav(path, wav, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    arr = wav.squeeze(0).numpy() if hasattr(wav, "squeeze") else np.asarray(wav)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def write_mp3(path, wav, sr: int, bitrate: str = "192k") -> None:
    """Write enhanced audio as MP3 (final delivery format, always)."""
    from .audio_io import encode_mp3

    encode_mp3(wav, sr, path, bitrate=bitrate)
