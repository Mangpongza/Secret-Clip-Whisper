from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .exceptions import WhisperStegError

Progress = Callable[[int, str], None]


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM มาตรฐานแบบ local-window Gaussian 11x11 ตามงานวิจัย Wang et al. (2004)"""
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def calculate_bit_error_rate(original_bits: list[int] | bytes, recovered_bits: list[int] | bytes) -> float:
    """คำนวณ Bit Error Rate (BER)"""
    if isinstance(original_bits, bytes):
        from .bits import bytes_to_bits
        original_bits = bytes_to_bits(original_bits)
    if isinstance(recovered_bits, bytes):
        from .bits import bytes_to_bits
        recovered_bits = bytes_to_bits(recovered_bits)

    min_len = min(len(original_bits), len(recovered_bits))
    if min_len == 0:
        return 1.0
    errors = sum(1 for i in range(min_len) if original_bits[i] != recovered_bits[i])
    errors += abs(len(original_bits) - len(recovered_bits))
    max_len = max(len(original_bits), len(recovered_bits))
    return float(errors / max_len)


def compare_videos_and_histogram(
    original_video: Path,
    stego_video: Path,
    progress: Progress | None = None,
) -> dict:
    """
    ประเมินคุณภาพวิดีโอ (PSNR, SSIM) และสกัดฮิสโทแกรมของทั้งสองไฟล์
    ตรงตามข้อ 1.3.1.5 และรูป 1.4 ของข้อเสนอโครงงาน
    """
    cap_a = cv2.VideoCapture(str(original_video))
    cap_b = cv2.VideoCapture(str(stego_video))
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise WhisperStegError("เปิดวิดีโอสำหรับประเมินผลไม่ได้")

    total = int(min(cap_a.get(cv2.CAP_PROP_FRAME_COUNT), cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    # รวมฮิสโทแกรมช่อง Y
    hist_original = np.zeros(256, dtype=np.float64)
    hist_stego = np.zeros(256, dtype=np.float64)

    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break
            if frame_a.shape != frame_b.shape:
                raise WhisperStegError("ความละเอียดของวิดีโอทั้งสองไม่ตรงกัน")

            # MSE & PSNR
            mse = float(np.mean((frame_a.astype(np.float64) - frame_b.astype(np.float64)) ** 2))
            psnr_sum += 100.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)

            # SSIM (Luminance)
            y_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            y_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            ssim_sum += ssim_gray(y_a, y_b)

            # Histograms (สะสมทุกๆ 3 เฟรมเพื่อความเร็ว)
            if count % 3 == 0:
                h_a, _ = np.histogram(y_a, bins=256, range=(0, 256))
                h_b, _ = np.histogram(y_b, bins=256, range=(0, 256))
                hist_original += h_a
                hist_stego += h_b

            count += 1
            if progress and count % 5 == 0:
                progress(int(count * 100 / max(total, 1)), f"กำลังวิเคราะห์เฟรม {count}/{total}")
    finally:
        cap_a.release()
        cap_b.release()

    if count == 0:
        raise WhisperStegError("ไม่พบเฟรมสำหรับเปรียบเทียบ")

    return {
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
        "frames_compared": count,
        "hist_original": hist_original,
        "hist_stego": hist_stego,
    }
