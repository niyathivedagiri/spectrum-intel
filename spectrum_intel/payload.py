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


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
# An image is sent as RAW 8-bit pixels, not as a compressed file (PNG/JPEG).
# Raw pixels degrade gracefully: one wrong bit changes one pixel slightly.
# In a compressed file one wrong bit can make the whole file undecodable.
#
# Byte layout:  b"IM" | height (2) | width (2) | channels (1) | 1 spare | pixels (row-major)

IMAGE_MAGIC = b"IM"
IMAGE_HEADER_BYTES = 8


def load_image(path: str | None = None, size: int = 64, gray: bool = False) -> np.ndarray:
    """Load and resize an image to size x size, uint8. Default: the Grace Hopper photo shipped with matplotlib."""
    from PIL import Image
    if path is None:
        import matplotlib.cbook as cbook
        path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    img = Image.open(path).convert("L" if gray else "RGB")
    w, h = img.size
    side = min(w, h)                                    # centre crop to a square, then resize
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def image_to_bytes(img: np.ndarray) -> bytes:
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    header = IMAGE_MAGIC + h.to_bytes(2, "big") + w.to_bytes(2, "big") + bytes([c, 0])
    return header + img.tobytes()


def bytes_to_image(data: bytes, shape: tuple | None = None) -> tuple[np.ndarray, bool]:
    """Rebuild the image. Returns (image, header_ok).

    If the 8-byte header was damaged or lost, the known `shape` is used instead
    (and header_ok is False, so the fallback is visible).
    """
    ok = data[:2] == IMAGE_MAGIC
    if ok:
        h, w, c = int.from_bytes(data[2:4], "big"), int.from_bytes(data[4:6], "big"), data[6]
        ok = h * w * max(c, 1) == len(data) - IMAGE_HEADER_BYTES
        if ok:
            shape = (h, w) if c == 1 else (h, w, c)
    if shape is None:
        raise ValueError("image header unreadable and no fallback shape given")
    n = int(np.prod(shape))
    pix = np.frombuffer(data[IMAGE_HEADER_BYTES:IMAGE_HEADER_BYTES + n], dtype=np.uint8)
    pix = np.concatenate([pix, np.zeros(n - len(pix), np.uint8)])
    return pix.reshape(shape), ok


def png_bytes(img: np.ndarray) -> bytes:
    """The same image as a compressed PNG file (used to show why raw pixels are sent)."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def decode_png(data: bytes) -> np.ndarray | None:
    """Decode PNG bytes; None if the file is damaged beyond reading."""
    import io

    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return np.asarray(im)
    except Exception:
        return None
