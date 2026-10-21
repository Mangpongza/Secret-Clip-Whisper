from __future__ import annotations

from collections.abc import Iterable, Iterator
import cv2
import numpy as np

# ค่าเริ่มต้นสำหรับ DCT Mid-frequency coefficients (Zig-Zag order 9-16)
# ตำแหน่งความถี่กลางในบล็อก 8x8 (row, col)
_DEFAULT_DCT_POSITIONS = (
    (1, 2), (2, 1),
    (0, 3), (1, 3), (2, 2), (3, 1),
    (3, 0), (2, 3)
)
_DEFAULT_STEP = 36.0  # QIM step size (robust against H.264 rounding)


def qim_embed(value: float, bit: int, step: float) -> float:
    """
    QIM (Quantization Index Modulation):
    - bit 0: ปัดไปยังระดับเลขคู่ (Even)
    - bit 1: ปัดไปยังระดับเลขคี่ (Odd)
    """
    sign = -1.0 if value < 0 else 1.0
    val_abs = abs(value)
    level = int(round(val_abs / step))
    if level % 2 != bit:
        lower = max(0, level - 1)
        upper = level + 1
        candidates = [x for x in (lower, upper) if x % 2 == bit]
        level = min(candidates, key=lambda x: abs(val_abs - x * step))
    return sign * (level * step)


def qim_extract(value: float, step: float) -> int:
    """ถอดบิตแบบ Blind ด้วย QIM: ดูเศษส่วนของระดับสเกล"""
    return int(round(abs(value) / step)) & 1


def capacity_bits(width: int, height: int, positions_per_block: int = 2) -> int:
    """ความจุบิตต่อหนึ่งเฟรม (นับเฉพาะบล็อก 8x8 เต็ม)"""
    return (width // 8) * (height // 8) * positions_per_block


def embed_frame_dct(
    frame_bgr: np.ndarray,
    bits_iter: Iterator[int],
    step_size: float = _DEFAULT_STEP,
    positions: tuple[tuple[int, int], ...] = ((1, 2), (2, 1)),
) -> tuple[np.ndarray, int]:
    """
    ฝังบิตข้อมูลลงในช่องความสว่าง Y (Luma) ของภาพ ด้วย 2D-DCT 8x8 และ QIM
    ตรงตามข้อกำหนดบทที่ 1 และ 2 ของเล่มโครงงาน
    ปรับแก้ด้วย:
    1. Dynamic Headroom Clamping: ป้องกันการเกิด Clipping ที่ค่า 0 และ 255 ในบล็อกที่มืดสนิทหรือสว่างจัด
    2. Delta Y Injection: ปรับแก้ตรงช่อง BGR เพื่อความเที่ยงตรงของสมการสี 100%
    """
    y_orig = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    y_channel = y_orig.copy()
    h, w = y_channel.shape
    used = 0
    exhausted = False

    # ค่าความปลอดภัย headroom เพื่อป้องกันการตัดทอนค่า (Clipping)
    headroom = step_size * 0.5 + 4.0

    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            block = y_channel[r : r + 8, c : c + 8]
            b_mean = float(np.mean(block))
            if b_mean < headroom:
                block = block + (headroom - b_mean)
            elif b_mean > (255.0 - headroom):
                block = block - (b_mean - (255.0 - headroom))

            coeff = cv2.dct(block - 128.0)
            changed = False

            for pos in positions:
                try:
                    bit = next(bits_iter)
                except StopIteration:
                    exhausted = True
                    break

                coeff[pos] = qim_embed(float(coeff[pos]), bit, step_size)
                used += 1
                changed = True

            if changed:
                y_channel[r : r + 8, c : c + 8] = np.clip(cv2.idct(coeff) + 128.0, 0, 255)

            if exhausted:
                break
        if exhausted:
            break

    # นำส่วนต่างของค่าความสว่าง (Delta Y) ไปปรับลงใน BGR โดยตรง
    delta_y = np.round(y_channel).astype(int) - np.round(y_orig).astype(int)
    stego_frame = np.clip(frame_bgr.astype(int) + delta_y[:, :, None], 0, 255).astype(np.uint8)
    return stego_frame, used


def extract_frame_dct(
    frame_bgr: np.ndarray,
    step_size: float = _DEFAULT_STEP,
    positions: tuple[tuple[int, int], ...] = ((1, 2), (2, 1)),
) -> Iterator[int]:
    """สกัดบิตข้อมูลจากช่อง Y ด้วย 2D-DCT 8x8 และ QIM แบบ Blind"""
    y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    h, w = y.shape
    for r in range(0, h - 7, 8):
        for c in range(0, w - 7, 8):
            coeff = cv2.dct(y[r : r + 8, c : c + 8] - 128.0)
            for pos in positions:
                yield qim_extract(float(coeff[pos]), step_size)
