from __future__ import annotations

import unittest

import numpy as np

from whispersteg.models import StegMode
from whispersteg.steg import capacity_bits, embed_frame, extract_frame


class StegTests(unittest.TestCase):
    def test_capacity_positive(self) -> None:
        for mode in StegMode:
            self.assertGreater(capacity_bits(320, 240, mode), 0)

    def test_short_bit_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        frame = rng.integers(20, 236, size=(256, 320, 3), dtype=np.uint8)
        bits = [int(x) for x in rng.integers(0, 2, size=96)]
        for mode in StegMode:
            with self.subTest(mode=mode):
                stego, used = embed_frame(frame.copy(), bits, mode)
                recovered = list(extract_frame(stego, mode))[:used]
                self.assertEqual(bits[:used], recovered)


if __name__ == "__main__":
    unittest.main()
