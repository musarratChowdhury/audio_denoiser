"""Audio I/O via a lightweight static ffmpeg binary.

Why imageio-ffmpeg (and not apt ffmpeg)?
  * ``pip install imageio-ffmpeg`` ships a single static ffmpeg binary
    (~25 MB, no system packages, no apt, no root). The ``python:3.11-slim``
    image stays dependency-free — important for small ACA images.
  * Static builds from BtbN/gyan include decoders for every common audio
    codec (AAC/M4A, Opus/WebM, WMA, AMR, AC3, DTS, ...) plus the
    ``libmp3lame`` MP3 encoder, so one binary covers decode-all/encode-mp3.

Design:
  * Fast path: libsndfile (soundfile) decodes wav/flac/ogg/mp3/aiff natively
    without spawning a subprocess — keep it for speed.
  * Fallback: ffmpeg decodes *anything else* to mono float32 raw PCM piped
    over stdout (no temp files), then julius resamples if needed.
  * Output: always MP3 (libmp3lame, configurable bitrate), piped raw PCM
    into ffmpeg stdin — constant memory, no intermediate WAV on disk.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

#: Extensions the API accepts. Decoding itself is delegated to ffmpeg, so any
#: container ffmpeg understands works — this set is just the UX/validation
#: allow-list (audio-first + common video containers whose audio track we
#: extract; ffmpeg drops video with ``-vn``).
ALLOWED_EXTS = frozenset({
    # lossless / PCM
    ".wav", ".flac", ".aif", ".aiff", ".aifc", ".caf", ".w64", ".voc", ".tta",
    # lossy codecs
    ".mp3", ".ogg", ".oga", ".opus", ".aac", ".adts", ".ac3", ".eac3",
    ".dts", ".mp2", ".mpc", ".spx", ".wv", ".ra",
    # containers / voice notes (audio track extracted, video dropped)
    ".m4a", ".mp4", ".wma", ".webm", ".mkv", ".mka", ".mov",
    ".3gp", ".3g2", ".amr", ".awb", ".gsm", ".ts", ".m2ts", ".rm",
})

#: Extensions soundfile/libsndfile decodes natively (fast path, no ffmpeg).
#: Everything else goes straight to ffmpeg.
SOUNDFILE_NATIVE_EXTS = frozenset({
    ".wav", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff",
})


def ffmpeg_exe() -> str:
    """Resolve the bundled static ffmpeg binary (raises with hint if missing)."""
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed; run `pip install -r requirements.txt`"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_to_mono_float32(path: str | Path, target_sr: int):
    """Decode *any* audio file to mono float32 Tensor ``[1, T]`` at ``target_sr``.

    Uses ``ffmpeg -ac 1 -ar <sr> -f f32le -`` piped over stdout, so peak RAM
    is one file-length buffer and no temp files are created. Raises
    ``ValueError`` (bad/empty media) or ``RuntimeError`` (ffmpeg crashed).
    """
    import numpy as np
    import torch

    exe = ffmpeg_exe()
    cmd = [
        exe, "-v", "error",
        "-i", str(path),
        "-vn",            # drop video streams (mp4/mkv/mov voice notes)
        "-ac", "1",       # mixdown to mono
        "-ar", str(int(target_sr)),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not launch ffmpeg ({exe}): {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()[-500:]
        raise ValueError(f"unsupported or corrupt audio file ({err or 'ffmpeg exit != 0'})")
    if not proc.stdout:
        raise ValueError("empty audio file")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if pcm.size == 0:
        raise ValueError("empty audio file")
    tensor = torch.from_numpy(pcm).unsqueeze(0)  # [1, T]
    return tensor.clamp(-1.0, 1.0)


def encode_mp3(wav, sr: int, out_path: str | Path, bitrate: str = "192k") -> None:
    """Encode mono/stereo float tensor (or numpy) at ``sr`` to MP3 via libmp3lame.

    Raw f32le PCM is piped to ffmpeg stdin — no intermediate WAV on disk.
    ``bitrate`` like ``"128k"``/``"192k"``/``"320k"``; also accepts ``"v0"``
    style VBR selectors (passed as ``-q:a`` instead of ``-b:a``).
    """
    import numpy as np

    exe = ffmpeg_exe()
    # torch Tensor [C, T] -> numpy; pass numpy straight through.
    if hasattr(wav, "numpy"):  # torch tensor (numpy arrays have no .numpy())
        arr = wav.squeeze(0).numpy() if wav.ndim == 2 else wav.numpy()
    else:
        arr = np.asarray(wav)
    arr = np.asarray(arr, dtype=np.float32)
    # Normalise to (channels, samples): denoiser outputs [1, T] (mono).
    if arr.ndim == 1:
        channels, buf = 1, np.ascontiguousarray(arr)
    elif arr.ndim == 2:
        channels = min(int(arr.shape[0]), 2)
        buf = np.ascontiguousarray(arr[:channels].T.reshape(-1))
    else:
        channels, buf = 1, np.ascontiguousarray(arr.reshape(-1))

    if buf.size == 0:
        raise ValueError("empty audio buffer, nothing to encode")

    # VBR shortcut: bitrate "v0".."v9" -> -q:a N ; else CBR -b:a
    bitrate = str(bitrate).strip().lower()
    if bitrate.startswith("v") and bitrate[1:].isdigit():
        codec_args = ["-q:a", bitrate[1:]]
    else:
        codec_args = ["-b:a", bitrate]

    cmd = [
        exe, "-v", "error", "-y",
        "-f", "f32le", "-ar", str(int(sr)), "-ac", str(channels),
        "-i", "-",
        "-codec:a", "libmp3lame", *codec_args,
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, input=buf.tobytes(),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not launch ffmpeg ({exe}): {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()[-500:]
        raise RuntimeError(f"mp3 encode failed ({err or 'ffmpeg exit != 0'})")
