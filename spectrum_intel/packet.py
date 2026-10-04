"""Packets: split a message into numbered pieces and put it back together.

MAC packet layout (bytes):
    seq (2) | total (2) | length (2) | data (length) | CRC-16 (2)

The CRC covers the header and data, so a damaged sequence number is caught too.
Packets that fail the CRC are dropped; reassembly reports which ones are missing
instead of silently hiding them.
"""
from __future__ import annotations

from dataclasses import dataclass

from spectrum_intel.coding import crc_append, crc_check

HEADER_BYTES = 6
CRC_BYTES = 2
OVERHEAD_BYTES = HEADER_BYTES + CRC_BYTES


@dataclass
class Packet:
    seq: int
    total: int
    data: bytes

    def to_bytes(self) -> bytes:
        header = (self.seq.to_bytes(2, "big") + self.total.to_bytes(2, "big")
                  + len(self.data).to_bytes(2, "big"))
        return crc_append(header + self.data)

    @staticmethod
    def from_bytes(block: bytes) -> tuple["Packet | None", bool]:
        """Parse a received block. Returns (packet or None, crc_ok)."""
        body, ok = crc_check(block)
        if not ok or len(body) < HEADER_BYTES:
            return None, False
        seq = int.from_bytes(body[0:2], "big")
        total = int.from_bytes(body[2:4], "big")
        length = int.from_bytes(body[4:6], "big")
        data = body[HEADER_BYTES:HEADER_BYTES + length]
        if len(data) != length:
            return None, False
        return Packet(seq, total, data), True


def parse_unchecked(block: bytes, expected_total: int, max_payload: int) -> "Packet | None":
    """Read the header fields WITHOUT trusting the CRC (error-tolerant mode for media).

    Only accepted if the fields are self-consistent (plausible sequence number,
    matching total and length); otherwise the packet is treated as lost.
    """
    if len(block) < OVERHEAD_BYTES:
        return None
    seq = int.from_bytes(block[0:2], "big")
    total = int.from_bytes(block[2:4], "big")
    length = int.from_bytes(block[4:6], "big")
    if total != expected_total or not 0 <= seq < total or length > max_payload \
            or HEADER_BYTES + length + CRC_BYTES != len(block):
        return None
    return Packet(seq, total, block[HEADER_BYTES:HEADER_BYTES + length])


def segment(data: bytes, max_payload: int = 64) -> list[Packet]:
    """Split data into packets of at most max_payload bytes (at least one packet)."""
    chunks = [data[i:i + max_payload] for i in range(0, len(data), max_payload)] or [b""]
    return [Packet(i, len(chunks), c) for i, c in enumerate(chunks)]


def reassemble(packets: list[Packet], total: int, fill: int = 0x00, chunk: int = 64,
               expected_len: int | None = None) -> tuple[bytes, list[int]]:
    """Rebuild the message from the packets that arrived.

    Missing packets are filled with `fill` bytes (so an image keeps its shape) and
    their sequence numbers are returned, so losses are always reported.
    expected_len (if known) trims/pads the result to the original length.
    """
    by_seq = {p.seq: p for p in packets}
    out, missing = bytearray(), []
    for i in range(total):
        if i in by_seq:
            out += by_seq[i].data
        else:
            missing.append(i)
            out += bytes([fill]) * chunk
    if expected_len is not None:
        out = out[:expected_len] + bytes([fill]) * max(0, expected_len - len(out))
    return bytes(out), missing
