from __future__ import annotations

import reedsolo

# Reed-Solomon RS(255, 223)
# n = 255 (block size), k = 223 (data payload size), 2t = 32 parity bytes
# ซ่อมไบต์ที่เสียหายได้สูงสุด t = 16 ไบต์ ต่อ 1 บล็อก
RS_BLOCK_SIZE = 255
RS_DATA_SIZE = 223
RS_PARITY_SIZE = 32

_rs_codec = reedsolo.RSCodec(RS_PARITY_SIZE)


def rs_encode(data: bytes) -> bytes:
    """เข้ารหัส Reed-Solomon RS(255, 223) ทีละบล็อกขนาด 223 ไบต์"""
    encoded_blocks = bytearray()
    for i in range(0, len(data), RS_DATA_SIZE):
        chunk = data[i : i + RS_DATA_SIZE]
        enc = _rs_codec.encode(chunk)
        encoded_blocks.extend(enc)
    return bytes(encoded_blocks)


def rs_decode(data: bytes) -> bytes:
    """ถอดรหัสและซ่อมแซมความเสียหายของข้อมูลด้วย Reed-Solomon"""
    decoded_blocks = bytearray()
    # ขนาดของแต่ละบล็อกหลังเข้ารหัสคือ len(chunk) + 32
    # เมื่ออ่านกลับมา บล็อกส่วนใหญ่จะมีความยาว RS_BLOCK_SIZE (255)
    offset = 0
    while offset < len(data):
        # บล็อกทั่วไปคือ 255 ไบต์ หรือถ้าเป็นบล็อกสุดท้ายอาจมีขนาดตามที่เหลือ
        rem = len(data) - offset
        curr_len = RS_BLOCK_SIZE if rem >= RS_BLOCK_SIZE else rem
        if curr_len <= RS_PARITY_SIZE:
            # ไม่พอให้ถอดรหัส parity
            break
        block = data[offset : offset + curr_len]
        try:
            dec, _, _ = _rs_codec.decode(block)
            decoded_blocks.extend(dec)
        except reedsolo.ReedSolomonError as exc:
            raise ValueError(f"Reed-Solomon ซ่อมแซมข้อมูลไม่สำเร็จ: ความเสียหายเกิน 16 ไบต์ในบล็อก ({exc})") from exc
        offset += curr_len

    return bytes(decoded_blocks)


def interleave_bytes(data: bytes, stride: int = 16) -> bytes:
    """
    Interleaving: สลับลำดับไบต์แบบ Matrix Transpose
    เพื่อแปลง Burst Error (ความเสียหายติดกันจาก Macroblock)
    ให้กลายเป็น Random Error กระจายข้ามบล็อก RS
    """
    n = len(data)
    if n <= stride:
        return data
    rows = (n + stride - 1) // stride
    padded_len = rows * stride
    padded = data + b"\x00" * (padded_len - n)
    
    interleaved = bytearray(padded_len)
    idx = 0
    for col in range(stride):
        for row in range(rows):
            interleaved[idx] = padded[row * stride + col]
            idx += 1
            
    # เก็บความยาวเดิม 4 bytes ไว้ข้างหน้า
    import struct
    return struct.pack(">I", n) + bytes(interleaved)


def deinterleave_bytes(data: bytes, stride: int = 16) -> bytes:
    """De-interleaving: คืนลำดับไบต์เดิมกลับมาเพื่อส่งต่อให้ Reed-Solomon"""
    import struct
    if len(data) < 4:
        return data
    orig_len = struct.unpack(">I", data[:4])[0]
    payload = data[4:]
    
    padded_len = len(payload)
    if padded_len == 0:
        return b""
    rows = padded_len // stride
    if rows * stride != padded_len:
        return payload[:orig_len]

    matrix = bytearray(padded_len)
    idx = 0
    for col in range(stride):
        for row in range(rows):
            matrix[row * stride + col] = payload[idx]
            idx += 1

    return bytes(matrix[:orig_len])
