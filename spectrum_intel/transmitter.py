"""Transmitter: bytes -> framed, pulse-shaped complex baseband waveform.

Frame (symbol stream, before pulse shaping):

    | preamble (63 BPSK) | P d d ... d | P d d ... d | ... | P |
                           \\___ 16 data symbols per block ___/

* Preamble: a length-63 maximum-length sequence. Its sharp autocorrelation lets
  the receiver find the frame start and symbol timing, and it gives references
  for frequency offset, phase, gain and SNR estimation.
* Pilots (P = +1) before every block of 16 data symbols and one at the end:
  the receiver interpolates the channel's gain/phase between them, so slow
  phase drift and fading are tracked through the frame.
* Data = 32-bit PHY header in BPSK (modulation id, payload length, CRC-16),
  then the payload in the chosen modulation. Sending the header in the most
  robust modulation is how real systems (e.g. Wi-Fi's SIGNAL field) tell the
  receiver what follows.

Pulse shaping: root-raised cosine, 8 samples/symbol, roll-off 0.35, exactly as
in signals.py, so link waveforms match the classifier's training signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectrum_intel.coding import crc16
from spectrum_intel.modulation import LINK_MODULATIONS, modulate
from spectrum_intel.payload import bits_to_bytes, bytes_to_bits
from spectrum_intel.signals import pulse_shape

SPS = 8
BETA = 0.35
SPAN = 8
PILOT_SPACING = 16
HEADER_BITS = 32
MAX_PAYLOAD_BYTES = 4095


def _m_sequence(n_bits: int = 6, taps=(6, 5)) -> np.ndarray:
    """Maximum-length sequence from a Fibonacci LFSR (x^6 + x^5 + 1), as +/-1."""
    state = [1] * n_bits
    out = []
    for _ in range(2 ** n_bits - 1):
        out.append(state[-1])
        fb = state[taps[0] - 1] ^ state[taps[1] - 1]
        state = [fb] + state[:-1]
    return 1.0 - 2.0 * np.array(out)


PREAMBLE = _m_sequence().astype(complex)
PILOT = 1.0 + 0j


def header_bits(mod: str, n_bytes: int) -> np.ndarray:
    """32-bit PHY header: 4-bit modulation id, 12-bit payload length, CRC-16 of those 2 bytes."""
    if not 0 <= n_bytes <= MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload must be 0..{MAX_PAYLOAD_BYTES} bytes")
    word = (LINK_MODULATIONS.index(mod) << 12) | n_bytes
    two = word.to_bytes(2, "big")
    return bytes_to_bits(two + crc16(two).to_bytes(2, "big"))


def parse_header(bits: np.ndarray):
    """Returns (mod, n_bytes, ok)."""
    raw = bits_to_bytes(bits[:HEADER_BITS])
    ok = crc16(raw[:2]) == int.from_bytes(raw[2:4], "big")
    word = int.from_bytes(raw[:2], "big")
    mod_id, n_bytes = word >> 12, word & 0xFFF
    if not ok or mod_id >= len(LINK_MODULATIONS):
        return None, 0, False
    return LINK_MODULATIONS[mod_id], n_bytes, True


def insert_pilots(data: np.ndarray) -> np.ndarray:
    """[P, 16 data, P, 16 data, ..., P]; the last block is zero-padded to 16."""
    n_blocks = int(np.ceil(len(data) / PILOT_SPACING))
    padded = np.zeros(n_blocks * PILOT_SPACING, dtype=complex)
    padded[:len(data)] = data
    blocks = padded.reshape(n_blocks, PILOT_SPACING)
    body = np.hstack([np.full((n_blocks, 1), PILOT), blocks]).ravel()
    return np.concatenate([body, [PILOT]])


def stream_length(n_data: int) -> int:
    """Number of symbols after the preamble for n_data data symbols."""
    return int(np.ceil(n_data / PILOT_SPACING)) * (PILOT_SPACING + 1) + 1


@dataclass
class TxFrame:
    iq: np.ndarray                 # complex baseband samples
    mod: str
    payload: bytes
    payload_bits: np.ndarray
    symbols: np.ndarray            # full symbol stream (preamble + pilots + data)
    data_symbols: np.ndarray       # header + payload symbols (no pilots)
    n_payload_symbols: int
    info: dict = field(default_factory=dict)


def build_frame(payload: bytes, mod: str) -> TxFrame:
    if mod not in LINK_MODULATIONS:
        raise ValueError(f"mod must be one of {LINK_MODULATIONS}")
    bits = bytes_to_bits(payload)
    hdr = modulate(header_bits(mod, len(payload)), "bpsk")
    pay = modulate(bits, mod)
    data = np.concatenate([hdr, pay])
    stream = np.concatenate([PREAMBLE, insert_pilots(data)])
    iq = pulse_shape(stream, SPS, BETA, SPAN)
    return TxFrame(iq=iq, mod=mod, payload=payload, payload_bits=bits, symbols=stream,
                   data_symbols=data, n_payload_symbols=len(pay),
                   info={"n_symbols": len(stream), "n_samples": len(iq),
                         "overhead_symbols": len(stream) - len(pay)})
