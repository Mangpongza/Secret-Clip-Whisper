from __future__ import annotations

import random
import unittest

from whispersteg.bits import bits_to_bytes, bytes_to_bits
from whispersteg.pipeline import (
    MP4_HEADER_BITS,
    MP4_HEADER_TOTAL_BITS,
    MP4_ROBUST_REPS,
    _majority_vote_bits,
    _rsc_body,
    _rsc_header,
)


class RobustECCTests(unittest.TestCase):
    def test_header_ecc_with_bit_errors(self) -> None:
        raw_header = b'WSPG' + bytes([1, 1, 0, 0]) + (1000).to_bytes(4, 'big') + (b'a' * 32)
        encoded = _rsc_header.encode(raw_header)
        bits = bytes_to_bits(encoded)
        reps_bits = bits * MP4_ROBUST_REPS

        rng = random.Random(42)
        corrupted = [b ^ 1 if rng.random() < 0.01 else b for b in reps_bits]

        h_copies = [corrupted[i * MP4_HEADER_BITS : (i + 1) * MP4_HEADER_BITS] for i in range(MP4_ROBUST_REPS)]
        voted = _majority_vote_bits(h_copies)
        decoded = bytes(_rsc_header.decode(bits_to_bytes(voted))[0])
        self.assertEqual(raw_header, decoded)

    def test_body_ecc_with_burst_and_random_errors(self) -> None:
        data = b'SecretAudioOpusPayload' * 50
        encoded = _rsc_body.encode(data)
        bits = bytes_to_bits(encoded)
        reps_bits = bits * MP4_ROBUST_REPS

        rng = random.Random(123)
        corrupted = [b ^ 1 if rng.random() < 0.015 else b for b in reps_bits]

        unit_len = len(bits)
        b_copies = [corrupted[i * unit_len : (i + 1) * unit_len] for i in range(MP4_ROBUST_REPS)]
        voted = _majority_vote_bits(b_copies)
        decoded = bytes(_rsc_body.decode(bits_to_bytes(voted))[0])
        self.assertEqual(data, decoded)


if __name__ == '__main__':
    unittest.main()
