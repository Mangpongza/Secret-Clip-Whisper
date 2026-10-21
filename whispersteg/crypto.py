from __future__ import annotations

import struct
from dataclasses import dataclass

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

from .exceptions import AuthenticationError, PayloadFormatError

MAGIC = b"WAES"
VERSION = 1
ITERATIONS = 100_000
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16

# magic (4), version (1), iterations (4), salt (16), nonce (12), inner_len (4)
_HEADER = struct.Struct(">4sBI16s12sI")


@dataclass(frozen=True)
class CryptoEnvelope:
    raw: bytes


def _key(password: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    if not password:
        raise ValueError("กรุณากรอกรหัสผ่าน")
    return PBKDF2(
        password.encode("utf-8"),
        salt,
        dkLen=32,
        count=iterations,
        hmac_hash_module=SHA256,
    )


def encrypt_aes256(plaintext: bytes, password: str) -> bytes:
    """เข้ารหัส AES-256-GCM ด้วยรหัสผ่าน ตามมาตรฐาน FIPS 197"""
    salt = get_random_bytes(SALT_SIZE)
    nonce = get_random_bytes(NONCE_SIZE)
    key = _key(password, salt, ITERATIONS)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(b"WhisperSteg/AES256")
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    header = _HEADER.pack(MAGIC, VERSION, ITERATIONS, salt, nonce, len(ciphertext))
    # format: header + tag(16) + ciphertext
    return header + tag + ciphertext


def decrypt_aes256(blob: bytes, password: str) -> bytes:
    """ถอดรหัส AES-256-GCM ตรวจสอบความถูกต้องของรหัสผ่านและข้อมูล"""
    if len(blob) < _HEADER.size + TAG_SIZE:
        raise PayloadFormatError("ข้อมูลเข้ารหัสสั้นเกินไปหรือเสียหาย")

    magic, version, iterations, salt, nonce, inner_len = _HEADER.unpack(blob[:_HEADER.size])
    if magic != MAGIC:
        raise PayloadFormatError("รูปแบบข้อมูลเข้ารหัสไม่ถูกต้อง (ไม่ใช่ WAES)")
    if version != VERSION:
        raise PayloadFormatError(f"เวอร์ชันการเข้ารหัสไม่ตรงกัน: {version}")

    tag = blob[_HEADER.size : _HEADER.size + TAG_SIZE]
    ciphertext = blob[_HEADER.size + TAG_SIZE : _HEADER.size + TAG_SIZE + inner_len]
    if len(ciphertext) != inner_len:
        raise PayloadFormatError("ขนาด Ciphertext ไม่ตรงกับที่ระบุใน Header")

    key = _key(password, salt, iterations)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(b"WhisperSteg/AES256")

    try:
        return cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as exc:
        raise AuthenticationError("รหัสผ่านไม่ถูกต้อง หรือข้อมูลลับถูกดัดแปลง") from exc
