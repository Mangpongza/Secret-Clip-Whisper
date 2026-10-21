from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass
from .exceptions import PayloadFormatError

MAGIC = b"WSTG"  # WhisperSteg Magic
VERSION = 1

# Header Fields (Total 16 bytes):
# magic: 4s (b"WSTG")
# version: B (1 byte)
# flags: B (bit 0: is_encrypted with AES-256)
# chunk_index: B (1 byte, 1-indexed, e.g. 1)
# total_chunks: B (1 byte, e.g. 1, up to 5)
# frame_interval: B (1 byte, N interval, e.g. 1 = every frame, 10 = every 10 frames)
# step_size: B (1 byte, QIM delta, e.g. 36)
# payload_len: I (4 bytes, unsigned int)
# crc16: H (2 bytes, CRC of header fields prior to crc16)
_HEADER_STRUCT = struct.Struct(">4sBBBBBBIH")
HEADER_SIZE = _HEADER_STRUCT.size  # exactly 16 bytes


@dataclass(frozen=True)
class StegHeader:
    version: int
    is_encrypted: bool
    chunk_index: int
    total_chunks: int
    frame_interval: int
    step_size: float
    payload_len: int

    def pack(self) -> bytes:
        flags = 1 if self.is_encrypted else 0
        raw_pre = struct.pack(
            ">4sBBBBBBI",
            MAGIC,
            self.version,
            flags,
            self.chunk_index,
            self.total_chunks,
            max(1, min(255, self.frame_interval)),
            int(round(self.step_size)),
            self.payload_len,
        )
        crc = binascii.crc_hqx(raw_pre, 0)
        return raw_pre + struct.pack(">H", crc)

    @classmethod
    def unpack(cls, data: bytes) -> StegHeader:
        if len(data) < HEADER_SIZE:
            raise PayloadFormatError("Header สั้นเกินไป")
        raw_pre = data[: HEADER_SIZE - 2]
        expected_crc = struct.unpack(">H", data[HEADER_SIZE - 2 : HEADER_SIZE])[0]
        actual_crc = binascii.crc_hqx(raw_pre, 0)
        if actual_crc != expected_crc:
            raise PayloadFormatError("CRC Header ไม่ตรง: ข้อมูลเสียหายหรือวิดีโอไม่ใช่ Stego Video")

        magic, version, flags, chunk_idx, total_chunks, frame_interval, step_raw, payload_len = struct.unpack(
            ">4sBBBBBBI", raw_pre
        )
        if magic != MAGIC:
            raise PayloadFormatError("Magic Number ไม่ถูกต้อง (ไม่ใช่ WhisperSteg Video)")
        if version != VERSION:
            raise PayloadFormatError(f"ไม่รองรับ Header version {version}")

        return cls(
            version=version,
            is_encrypted=bool(flags & 1),
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            frame_interval=max(1, frame_interval),
            step_size=float(step_raw),
            payload_len=payload_len,
        )


def pack_chunk_payload(
    payload_bytes: bytes,
    chunk_index: int,
    total_chunks: int,
    frame_interval: int,
    step_size: float,
    is_encrypted: bool,
) -> bytes:
    """รวม Header (16 bytes) เข้ากับ Payload ข้อมูล"""
    header = StegHeader(
        version=VERSION,
        is_encrypted=is_encrypted,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        frame_interval=frame_interval,
        step_size=step_size,
        payload_len=len(payload_bytes),
    )
    return header.pack() + payload_bytes


def unpack_chunk_payload(raw_bytes: bytes) -> tuple[StegHeader, bytes]:
    """ถอดแยก Header และตรวจสอบความสมบูรณ์ของ Payload"""
    if len(raw_bytes) < HEADER_SIZE:
        raise PayloadFormatError("ข้อมูลไม่พอที่จะอ่าน Header")
    header = StegHeader.unpack(raw_bytes[:HEADER_SIZE])
    payload = raw_bytes[HEADER_SIZE : HEADER_SIZE + header.payload_len]
    if len(payload) != header.payload_len:
        raise PayloadFormatError(
            f"ขนาด Payload ไม่ครบ: ได้รับ {len(payload)} จาก {header.payload_len} ไบต์"
        )
    return header, payload
