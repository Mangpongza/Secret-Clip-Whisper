from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class StegMode(IntEnum):
    DCT = 1
    DWT = 2
    HYBRID = 3

    @property
    def label(self) -> str:
        return {
            StegMode.DCT: "DCT (8×8 Block)",
            StegMode.DWT: "DWT (Haar Wavelet)",
            StegMode.HYBRID: "Hybrid DWT-DCT",
        }[self]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class CapacityReport:
    bits_per_frame: int
    total_bits: int

    @property
    def total_bytes(self) -> int:
        return self.total_bits // 8
