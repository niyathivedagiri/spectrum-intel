"""Forward error correction (FEC): convolutional code + Viterbi decoder, puncturing, interleaving.

Code: rate 1/2, constraint length K = 7, generators 133 and 171 (octal), the code used by
Wi-Fi (802.11a/g), DVB-S and many satellite links. Every input bit produces two output bits,
each the XOR (parity) of the bit and some of the previous 6 bits. The encoder is
terminated with 6 zero "tail" bits, so the decoder knows it ends in state 0.

Puncturing: some of the coded bits are not sent, giving rates 2/3 and 3/4 with the same
decoder (the 802.11 patterns). The receiver puts back "no information" (LLR 0) where bits
were removed. Fewer parity bits = less protection but less airtime.

Decoder: Viterbi (maximum-likelihood sequence decoding over the 64-state trellis).
  soft  = uses log-likelihood ratios (LLRs): how sure the demodulator is about each bit.
          A bit received close to the decision boundary counts little; a clear one counts a lot.
  hard  = uses only the 0/1 decisions (every bit counts the same). Typically ~2 dB worse.

Interleaver: a fixed pseudo-random permutation of the coded bits in a packet. A fade or an
interference burst then hits bits that are far apart in the code, where the Viterbi decoder
can correct them; without it the errors arrive in bursts that overwhelm it.

LLR convention: LLR = log P(bit = 0) / P(bit = 1); positive means "probably 0".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

CODES = ("none", "1/2", "2/3", "3/4")
DECODINGS = ("soft", "hard", "soft_fixed")
LOCAL_WINDOW = 9               # symbols averaged for the local (decision-directed) noise estimate
K = 7
N_STATES = 2 ** (K - 1)
G = (0o133, 0o171)
TAIL = K - 1
# rows = output A / output B, columns = consecutive input bits; 1 = sent, 0 = punctured
PUNCTURE = {"1/2": np.array([[1], [1]]),
            "2/3": np.array([[1, 1], [1, 0]]),
            "3/4": np.array([[1, 1, 0], [1, 0, 1]])}
RATE = {"none": 1.0, "1/2": 1 / 2, "2/3": 2 / 3, "3/4": 3 / 4}


def check_code(code: str) -> str:
    if code not in CODES:
        raise ValueError(f"code must be one of {CODES}")
    return code


def _parity(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    p = np.zeros_like(x)
    for b in range(K):
        p ^= (x >> b) & 1
    return p


@lru_cache(maxsize=1)
def _trellis():
    """Encoder register = (new bit << 6) | state; state = the previous 6 inputs, newest at bit 5.

    For each NEXT state ns there are two predecessors (ps0, ps1), both entered with the same
    input bit u = ns >> 5. Returns predecessors and the +-1 outputs of each branch.
    """
    ns = np.arange(N_STATES)
    u = ns >> 5
    prev = np.stack([((ns & 31) << 1) | 0, ((ns & 31) << 1) | 1])        # (2, 64)
    reg = (u[None, :] << 6) | prev
    outs = np.stack([_parity(reg & g) for g in G])                     # (2 outputs, 2 branches, 64)
    sign = 1.0 - 2.0 * outs                                            # bit 0 -> +1, bit 1 -> -1
    return prev, sign, u


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Rate-1/2 terminated encoding: len(bits) + 6 tail steps -> 2 coded bits per step (A, B, A, B ...)."""
    b = np.concatenate([np.asarray(bits, dtype=np.int64), np.zeros(TAIL, np.int64)])
    s, out = 0, np.empty((len(b), 2), np.int64)
    for i, u in enumerate(b):
        reg = (int(u) << 6) | s
        out[i, 0] = bin(reg & G[0]).count("1") & 1
        out[i, 1] = bin(reg & G[1]).count("1") & 1
        s = reg >> 1
    return out.ravel().astype(np.uint8)


def _keep_mask(n_steps: int, code: str) -> np.ndarray:
    pat = PUNCTURE[code]
    cols = np.arange(n_steps) % pat.shape[1]
    return pat[:, cols].T.ravel().astype(bool)                         # (A, B) order per step


def coded_length(n_info_bits: int, code: str) -> int:
    """Bits on air for n_info_bits of data (incl. the 6 tail bits for coded modes)."""
    check_code(code)
    if code == "none":
        return n_info_bits
    return int(_keep_mask(n_info_bits + TAIL, code).sum())


@lru_cache(maxsize=64)
def _perm(n: int) -> np.ndarray:
    return np.random.default_rng([n, 802]).permutation(n)              # fixed per length, same at both ends


def interleave(x: np.ndarray) -> np.ndarray:
    return np.asarray(x)[_perm(len(x))]


def deinterleave(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(np.asarray(x))
    out[_perm(len(x))] = x
    return out


def encode(bits: np.ndarray, code: str, interleaved: bool = True) -> np.ndarray:
    """Data bits -> bits on air (coded, punctured, interleaved). code 'none' returns the input."""
    check_code(code)
    if code == "none":
        return np.asarray(bits, dtype=np.uint8)
    c = conv_encode(bits)
    c = c[_keep_mask(len(c) // 2, code)]
    return interleave(c) if interleaved else c


def viterbi(llr: np.ndarray) -> np.ndarray:
    """Soft-input Viterbi for the terminated K=7 code. llr: 2 per step (A, B). Returns the data bits.

    Path metric = sum of (expected sign x LLR) along the path (correlation), maximised.
    """
    prev, sign, u = _trellis()
    L = np.asarray(llr, float).reshape(-1, 2)
    T = len(L)
    pm = np.full(N_STATES, -1e18)
    pm[0] = 0.0
    dec = np.empty((T, N_STATES), np.int8)
    for t in range(T):
        bm = sign[0] * L[t, 0] + sign[1] * L[t, 1]                    # (2 branches, 64)
        cand = pm[prev] + bm
        d = cand[1] > cand[0]
        dec[t] = d
        pm = np.where(d, cand[1], cand[0])
    s, bits = 0, np.empty(T, np.uint8)                                 # terminated: end in state 0
    for t in range(T - 1, -1, -1):
        bits[t] = u[s]
        s = prev[dec[t, s], s]
    return bits[:T - TAIL]


def decode(llr: np.ndarray, n_info_bits: int, code: str, interleaved: bool = True) -> np.ndarray:
    """LLRs of the bits on air -> data bits (hard decisions from the LLR sign for code 'none')."""
    check_code(code)
    llr = np.asarray(llr, float)
    if code == "none":
        return (llr[:n_info_bits] < 0).astype(np.uint8)
    n_coded = coded_length(n_info_bits, code)
    x = llr[:n_coded]
    if interleaved:
        x = deinterleave(x)
    full = np.zeros(2 * (n_info_bits + TAIL))
    full[_keep_mask(n_info_bits + TAIL, code)] = x                     # punctured bits: LLR 0 (unknown)
    return viterbi(full)[:n_info_bits]


# --------------------------------------------------------------------------
# Soft demapping
# --------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _bit_table(mod: str):
    from spectrum_intel.modulation import _pattern_bits
    from spectrum_intel.signals import constellation
    return constellation(mod), _pattern_bits(mod).astype(bool)


def soft_demap(symbols: np.ndarray, mod: str, noise_var) -> np.ndarray:
    """Max-log LLR per bit (MSB first per symbol, as modulation.demodulate).

    LLR_k = (min over points with bit k = 1 of |y - s|^2  -  min over points with bit k = 0) / noise_var.
    noise_var may be one value or one per symbol: after dividing by a weak channel gain the noise
    is larger, so those bits are trusted less (channel-state information).
    """
    pts, bits = _bit_table(mod)
    y = np.asarray(symbols)
    d = np.abs(y[:, None] - pts[None, :]) ** 2                         # (n, M)
    nv = np.broadcast_to(np.maximum(np.asarray(noise_var, float), 1e-12), (len(y),))
    k = bits.shape[1]
    out = np.empty((len(y), k))
    for j in range(k):
        one, zero = bits[:, j], ~bits[:, j]
        out[:, j] = (d[:, one].min(axis=1) - d[:, zero].min(axis=1)) / nv
    return out.ravel()


def local_noise(symbols: np.ndarray, mod: str, window: int = LOCAL_WINDOW) -> np.ndarray:
    """Noise power around each symbol, from the distance to the nearest constellation point
    (decision-directed), averaged over `window` neighbouring symbols.

    In steady noise it is close to (or, at low SNR, below) the preamble estimate, so taking the
    larger of the two changes little. In an interference burst or a deep fade it jumps up, so the
    bits there get small LLRs instead of large, wrong ones.
    """
    from spectrum_intel.modulation import nearest_index
    from spectrum_intel.signals import constellation
    y = np.asarray(symbols)
    e = np.abs(y - constellation(mod)[nearest_index(y, mod)]) ** 2
    k = np.ones(min(window, max(len(e), 1)))
    return np.convolve(e, k, mode="same") / np.convolve(np.ones(len(e)), k, mode="same")


def hard_llr(bits: np.ndarray) -> np.ndarray:
    """0/1 decisions as LLRs of equal size (+1 for 0, -1 for 1): what a hard-decision decoder sees."""
    return 1.0 - 2.0 * np.asarray(bits, float)
