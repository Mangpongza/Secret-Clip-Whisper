from __future__ import annotations

import binascii
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2

from .audio import compress_to_opus, ensure_ffmpeg, opus_to_wav
from .bits import bits_to_bytes, bytes_to_bits
from .crypto import decrypt_aes256, encrypt_aes256
from .ecc import deinterleave_bytes, interleave_bytes, rs_decode, rs_encode
from .exceptions import CapacityError, WhisperStegError
from .models import CapacityReport, VideoInfo
from .payload import (
    HEADER_SIZE,
    StegHeader,
    pack_chunk_payload,
    unpack_chunk_payload,
)
from .steg import (
    _DEFAULT_STEP,
    capacity_bits,
    embed_frame_dct,
    extract_frame_dct,
)

Progress = Callable[[int, str], None]


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise WhisperStegError(f"เปิดวิดีโอไม่ได้: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frames <= 0:
            raise WhisperStegError("อ่านข้อมูลวิดีโอไม่สำเร็จ")
        return VideoInfo(width, height, fps, frames)
    finally:
        capture.release()


def capacity_report_video(path: Path) -> CapacityReport:
    info = video_info(path)
    per_frame = capacity_bits(info.width, info.height)
    return CapacityReport(per_frame, per_frame * info.frame_count)


def estimate_opus_size(audio: Path, bitrate_kbps: int = 16) -> int:
    """ประมาณขนาดเสียง Opus CBR 16 kbps ล่วงหน้า (รวม overhead เล็กน้อย)"""
    ensure_ffmpeg()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise WhisperStegError("อ่านระยะเวลาไฟล์เสียงไม่ได้")
    try:
        seconds = float(json.loads(result.stdout)["format"]["duration"])
    except Exception as exc:
        raise WhisperStegError("FFprobe ไม่ส่งค่า duration ที่ถูกต้อง") from exc
    # CBR 16 kbps = 2,000 bytes/sec + เผื่อ Ogg Header ~1-2 KB
    return int(seconds * 2000 + 1500)


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"
        raise WhisperStegError(f"ประกอบวิดีโอไม่สำเร็จ: {detail}")


def embed_audio_into_video(
    video: Path,
    audio: Path,
    output: Path,
    password: str = "",
    use_encryption: bool = False,
    chunk_index: int = 1,
    total_chunks: int = 1,
    progress: Progress | None = None,
) -> None:
    """
    กระบวนการฝังข้อความเสียงลงในวิดีโอ (ฝั่งผู้ส่ง)
    1. แปลงเสียงเป็น Opus 16 kbps Mono (CBR)
    2. เข้ารหัสลับ AES-256 (ทางเลือก)
    3. เติม ECC Reed-Solomon RS(255, 223) + Interleaving
    4. สร้าง Header ระบุ Magic, CRC, Chunk info, QIM parameters
    5. ฝังบิตลงในช่อง Y ด้วย 2D-DCT 8x8 และ QIM
    6. บันทึกเป็น H.264 MP4 พร้อม Stream Copy แทร็กเสียงเดิม
    """
    ensure_ffmpeg()
    if use_encryption and len(password) < 8:
        raise WhisperStegError("หากเลือกเข้ารหัส รหัสผ่านต้องยาวอย่างน้อย 8 ตัวอักษร")

    info = video_info(video)
    report = capacity_report_video(video)

    with tempfile.TemporaryDirectory(prefix="whispersteg_") as temp_name:
        temp = Path(temp_name)
        opus_path = temp / "payload.opus"
        frames_dir = temp / "frames"
        frames_dir.mkdir()

        if progress:
            progress(5, "กำลังแปลงเสียงเป็น Opus 16 kbps Mono (CBR RFC 7587)…")
        compress_to_opus(audio, opus_path, bitrate_kbps=16)

        raw_audio_bytes = opus_path.read_bytes()

        # 2. เข้ารหัสลับ (ทางเลือก)
        if use_encryption:
            if progress:
                progress(10, "กำลังเข้ารหัสข้อมูลเสียงด้วย AES-256 (FIPS 197)…")
            processed_data = encrypt_aes256(raw_audio_bytes, password)
        else:
            processed_data = raw_audio_bytes

        # 3. รหัสแก้ไขข้อผิดพลาด Reed-Solomon + Interleaving
        if progress:
            progress(15, "กำลังเติม Reed-Solomon RS(255, 223) และทำ Interleaving…")
        rs_data = rs_encode(processed_data)
        interleaved_data = interleave_bytes(rs_data)

        # 4. ส่วนหัวข้อมูล (Header)
        # คำนวณการเว้น N เฟรมเพื่อกระจายข้อมูลตลอดความยาวคลิป (ตามข้อ 2.1.4.1 ข้อ 4)
        bits_per_frame = capacity_bits(info.width, info.height)
        header_raw_bits_len = HEADER_SIZE * 8
        header_total_bits_needed = header_raw_bits_len * 5

        # จำนวนบิตของข้อมูลตัวเนื้อเสียง + ECC
        body_bytes = interleaved_data
        body_bits = bytes_to_bits(body_bytes)
        body_bits_len = len(body_bits)

        # คำนวณจำนวนเฟรมที่ต้องใช้สำหรับ Body
        body_frames_needed = (body_bits_len + bits_per_frame - 1) // bits_per_frame
        available_body_frames = max(1, info.frame_count - 1)

        # N = ระยะก้าวข้ามเฟรม (เว้นทุก N เฟรม) เพื่อกระจายให้ทั่วทั้งความยาวคลิป
        frame_interval = max(1, available_body_frames // max(1, body_frames_needed))
        # จำกัด N ไม่ให้เกิน 255 ตามขนาดของฟิลด์ใน Header
        frame_interval = min(frame_interval, 250)

        full_chunk_packet = pack_chunk_payload(
            payload_bytes=interleaved_data,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            frame_interval=frame_interval,
            step_size=_DEFAULT_STEP,
            is_encrypted=use_encryption,
        )

        # การทำ Repetition Code เพื่อความทนทานของ Header ในเฟรมที่ 0
        header_bytes = full_chunk_packet[:HEADER_SIZE]
        header_bits = bytes_to_bits(header_bytes) * 5

        if len(body_bits) > (report.total_bits - len(header_bits)):
            req_kb = (len(body_bits) // 8) / 1024
            avail_kb = report.total_bytes / 1024
            raise CapacityError(
                f"ความจุวิดีโอไม่เพียงพอ: ต้องการ {req_kb:.1f} KB แต่วิดีโอรองรับ {avail_kb:.1f} KB "
                "(กรุณาใช้วิดีโอที่ยาวขึ้น หรือแบ่งข้อมูลเป็น Chunk)"
            )

        if progress:
            progress(20, f"เริ่มฝังข้อมูล (Header ในเฟรมแรก, กระจาย Body เว้นทุก {frame_interval} เฟรม)…")

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise WhisperStegError("เปิดวิดีโอต้นฉบับไม่ได้")

        # ส่งสตรีมเฟรมผ่าน ffmpeg stdin ตรง (Raw Video Pipe) เพื่อความเร็วสูงสุดและไม่กินเนื้อที่ดิสก์
        output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{info.width}x{info.height}",
            "-pix_fmt", "bgr24",
            "-r", f"{info.fps:.6f}",
            "-i", "-",
            "-i", str(video),
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "1", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest", str(output),
        ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        header_iter = iter(header_bits)
        body_iter = iter(body_bits)
        body_offset = 0
        frame_idx = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # เฟรมที่ 0: ฝัง Header (ซ้ำ 5 ชุด)
                if frame_idx == 0:
                    modified_frame, _ = embed_frame_dct(frame, header_iter, step_size=_DEFAULT_STEP)
                # เฟรมถัดๆ ไป: ฝัง Body เฉพาะเฟรมที่ตรงกับก้าว N (frame_idx % frame_interval == 0)
                elif (frame_idx % frame_interval == 0) and (body_offset < body_bits_len):
                    modified_frame, used = embed_frame_dct(frame, body_iter, step_size=_DEFAULT_STEP)
                    body_offset += used
                else:
                    modified_frame = frame

                proc.stdin.write(modified_frame.tobytes())
                frame_idx += 1

                if progress and frame_idx % 10 == 0:
                    pct = 20 + int(70 * frame_idx / max(info.frame_count, 1))
                    progress(min(pct, 90), f"กำลังเข้ารหัสและฝังข้อมูลลงเฟรม {frame_idx}/{info.frame_count}…")
        finally:
            capture.release()
            if proc.stdin:
                proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise WhisperStegError("FFmpeg encoding failed during pipeline.")

        if progress:
            progress(100, f"สร้างวิดีโอที่ซ่อนเสียงสำเร็จ: {output.name}")


def _majority_vote_bits(copies: list[list[int]]) -> list[int]:
    threshold = len(copies) // 2 + 1
    return [1 if sum(col) >= threshold else 0 for col in zip(*copies)]


def extract_audio_from_video(
    video: Path,
    output_opus: Path,
    output_wav: Path,
    password: str = "",
    progress: Progress | None = None,
) -> tuple[StegHeader, bool]:
    """
    กระบวนการสกัดข้อความเสียงออกจากวิดีโอ (ฝั่งผู้รับ)
    1. อ่านส่วนหัว (Header) จากเฟรมแรก (Frame 0) และทำ Majority vote คลายการซ้ำ
    2. อ่านค่า frame_interval (N) จาก Header แล้วดึงบิต Body เฉพาะเฟรมที่เว้นทุก N เฟรม
    3. ทำ De-interleaving และถอดรหัส Reed-Solomon RS(255, 223) ซ่อมแซมไบต์เสีย
    4. ถอดรหัส AES-256 หากมีรหัสผ่าน
    5. สกัดได้ไฟล์ Opus และแปลงเป็น WAV ให้พร้อมเปิดฟัง
    """
    info = video_info(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise WhisperStegError("เปิด stego video ไม่ได้")

    header_raw_bits_len = HEADER_SIZE * 8
    header_total_bits_needed = header_raw_bits_len * 5  # ซ้ำ 5 รอบ

    body_bits: list[int] = []
    frame_idx = 0
    header: StegHeader | None = None
    body_bits_needed: int | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # เฟรม 0: อ่าน Header
            if frame_idx == 0:
                frame_bits = list(extract_frame_dct(frame, step_size=_DEFAULT_STEP))
                if len(frame_bits) < header_total_bits_needed:
                    raise WhisperStegError("ขนาดเฟรมแรกเล็กเกินไป ไม่สามารถอ่าน Header ได้")
                h_chunk = frame_bits[:header_total_bits_needed]
                copies = [
                    h_chunk[i * header_raw_bits_len : (i + 1) * header_raw_bits_len]
                    for i in range(5)
                ]
                voted_header_bits = _majority_vote_bits(copies)
                header_bytes = bits_to_bytes(voted_header_bits)
                try:
                    header = StegHeader.unpack(header_bytes)
                    body_bits_needed = header.payload_len * 8
                except Exception as exc:
                    raise WhisperStegError(f"ไม่พบ Header ของ WhisperSteg: {exc}") from exc

            # เฟรมอื่นๆ: อ่านเฉพาะเฟรมที่ตรงกับระยะก้าว N (frame_idx % header.frame_interval == 0)
            elif header is not None and body_bits_needed is not None:
                if (frame_idx % header.frame_interval == 0) and (len(body_bits) < body_bits_needed):
                    b = list(extract_frame_dct(frame, step_size=header.step_size))
                    body_bits.extend(b)
                    if len(body_bits) >= body_bits_needed:
                        body_bits = body_bits[:body_bits_needed]
                        break

            frame_idx += 1
            if progress and frame_idx % 4 == 0:
                pct = int(frame_idx * 75 / max(info.frame_count, 1))
                progress(min(pct, 75), f"กำลังสกัดข้อมูลจากเฟรม {frame_idx}/{info.frame_count}…")
    finally:
        capture.release()

    if header is None or body_bits_needed is None or len(body_bits) < body_bits_needed:
        raise WhisperStegError("สกัดข้อมูลไม่ครบถ้วน วิดีโออาจถูกตัดทอนหรือเสียหาย")

    if progress:
        progress(80, "สกัดบิตสำเร็จ กำลังคลาย Interleaving และถอดรหัส Reed-Solomon…")

    interleaved_bytes = bits_to_bytes(body_bits[:body_bits_needed])

    # De-interleaving & Reed-Solomon Decode
    deinterleaved_bytes = deinterleave_bytes(interleaved_bytes)
    try:
        recovered_data = rs_decode(deinterleaved_bytes)
    except Exception as exc:
        raise WhisperStegError(f"การกู้คืนข้อมูลล้มเหลว: {exc}") from exc

    # AES-256 Decrypt (ถ้ามีการเข้ารหัส)
    if header.is_encrypted:
        if not password:
            raise WhisperStegError("ไฟล์เสียงนี้ถูกเข้ารหัสด้วย AES-256 กรุณาระบุรหัสผ่าน")
        if progress:
            progress(90, "กำลังถอดรหัสลับ AES-256 และตรวจสอบความถูกต้อง…")
        opus_data = decrypt_aes256(recovered_data, password)
    else:
        opus_data = recovered_data

    output_opus.parent.mkdir(parents=True, exist_ok=True)
    output_opus.write_bytes(opus_data)

    if progress:
        progress(95, "กำลังแปลงเสียง Opus เป็น WAV เพื่อเปิดฟัง…")
    opus_to_wav(output_opus, output_wav)

    if progress:
        progress(100, "กู้คืนข้อความเสียงสำเร็จสมบูรณ์!")

    return header, True
