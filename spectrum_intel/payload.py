"""Information layer: turn user content into bits and back.

Bit order is MSB-first within each byte (the usual network order), so the
bits shown for "H" (0x48) are 0 1 0 0 1 0 0 0.
"""
from __future__ import annotations

import numpy as np


def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def bytes_to_text(data: bytes) -> str:
    """Decode UTF-8; damaged bytes become U+FFFD so errors stay visible instead of crashing."""
    return data.decode("utf-8", errors="replace")


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack bits (MSB first). A trailing partial byte is zero-padded."""
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits).tobytes()


def text_to_bits(text: str) -> np.ndarray:
    return bytes_to_bits(text_to_bytes(text))


def bits_to_text(bits: np.ndarray) -> str:
    return bytes_to_text(bits_to_bytes(bits))


def describe_bytes(data: bytes, limit: int = 16) -> list[dict]:
    """Per-byte view for teaching/demo output: character, hex value and bits."""
    rows = []
    for b in data[:limit]:
        ch = chr(b) if 32 <= b < 127 else "·"
        rows.append({"char": ch, "hex": f"0x{b:02X}", "bits": f"{b:08b}"})
    return rows
