from __future__ import annotations

from collections.abc import Iterable, Iterator


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    values = list(bits)
    if len(values) % 8:
        raise ValueError("จำนวนบิตต้องหารด้วย 8 ลงตัว")
    return bytes(
        sum((values[i + j] & 1) << (7 - j) for j in range(8))
        for i in range(0, len(values), 8)
    )


def take_bits(source: Iterator[int], limit: int) -> list[int]:
    result: list[int] = []
    for _ in range(limit):
        try:
            result.append(next(source))
        except StopIteration:
            break
    return result
