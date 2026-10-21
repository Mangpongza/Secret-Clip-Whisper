from __future__ import annotations

import unittest

from whispersteg.crypto import decrypt_dual_layer, encrypt_dual_layer
from whispersteg.exceptions import AuthenticationError, PayloadFormatError
from whispersteg.models import StegMode
from whispersteg.payload import pack_payload, unpack_payload


class CryptoPayloadTests(unittest.TestCase):
    def test_dual_layer_round_trip(self) -> None:
        source = (b"secret-opus-data" * 100) + bytes(range(256))
        encrypted = encrypt_dual_layer(source, "correct horse battery staple").raw
        self.assertNotEqual(source, encrypted)
        self.assertEqual(source, decrypt_dual_layer(encrypted, "correct horse battery staple"))

    def test_wrong_password_is_rejected(self) -> None:
        encrypted = encrypt_dual_layer(b"secret", "right-password").raw
        with self.assertRaises(AuthenticationError):
            decrypt_dual_layer(encrypted, "wrong-password")

    def test_tampering_is_rejected(self) -> None:
        blob = bytearray(encrypt_dual_layer(b"secret", "password123").raw)
        blob[-1] ^= 1
        with self.assertRaises(AuthenticationError):
            decrypt_dual_layer(bytes(blob), "password123")

    def test_payload_round_trip_and_mode_guard(self) -> None:
        encrypted = b"encrypted" * 50
        packed = pack_payload(encrypted, StegMode.HYBRID)
        self.assertEqual(encrypted, unpack_payload(packed, StegMode.HYBRID))
        with self.assertRaises(PayloadFormatError):
            unpack_payload(packed, StegMode.DCT)


if __name__ == "__main__":
    unittest.main()
