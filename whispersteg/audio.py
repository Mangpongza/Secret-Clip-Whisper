from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import numpy as np

from .exceptions import DependencyError, WhisperStegError


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise DependencyError(
            "ไม่พบ FFmpeg/FFprobe กรุณาติดตั้ง FFmpeg และเพิ่มโฟลเดอร์ bin ลง PATH"
        )


def _run(command: list[str]) -> None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise WhisperStegError(f"เรียก FFmpeg ไม่สำเร็จ: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr else "unknown error"
        raise WhisperStegError(f"FFmpeg ทำงานล้มเหลว: {detail}")


def compress_to_opus(source: Path, destination: Path, bitrate_kbps: int = 16) -> None:
    """
    แปลงไฟล์เสียงเป็น Opus 16 kbps แบบโมโน (Mono) ตาม RFC 7587 โหมด VoIP
    - c:a libopus
    - b:a 16k (Constant Bitrate / CBR)
    - vbr off
    - application voip
    - ac 1 (Mono)
    - ar 16000 (16 kHz Wideband speech)
    """
    ensure_ffmpeg()
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vn",
        "-c:a", "libopus",
        "-b:a", f"{bitrate_kbps}k",
        "-vbr", "off",
        "-application", "voip",
        "-frame_duration", "20",
        "-ac", "1",
        "-ar", "16000",
        str(destination),
    ])


def opus_to_wav(source: Path, destination: Path) -> None:
    ensure_ffmpeg()
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vn", "-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000", str(destination),
    ])


def save_audio_buffer_to_wav(audio_data: np.ndarray, sample_rate: int, destination: Path) -> None:
    """บันทึกข้อมูลเสียงที่บันทึกจากไมโครโฟนลงไฟล์ WAV ชั่วคราว"""
    import scipy.io.wavfile as wavfile
    # ensure int16 or float32 range
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        scaled = np.int16(np.clip(audio_data, -1.0, 1.0) * 32767)
    else:
        scaled = audio_data.astype(np.int16)
    wavfile.write(str(destination), sample_rate, scaled)
