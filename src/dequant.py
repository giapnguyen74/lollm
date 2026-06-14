"""
dequant.py — turn GGUF quantized blocks back into fp32 weights.

Each function takes the raw bytes of a tensor (uint8) plus the element count and
returns a flat float32 array. Implementations follow the ggml reference layouts
(see docs/quantization.md for the Q4_K walk-through). Everything is vectorized
over blocks with numpy — readable, and fast enough to dequantize a whole model
once at load time.

Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_K, Q5_K, Q6_K.
(Real Q4_K_M files mix K-quants with a legacy quant — often Q5_0 — for the
token embedding, so the legacy types are needed too.)
"""

from __future__ import annotations

import numpy as np


def _blocks(raw: np.ndarray, type_size: int) -> np.ndarray:
    """Reshape the flat byte buffer into (n_blocks, type_size)."""
    return raw.reshape(-1, type_size)


def _f16(cols: np.ndarray) -> np.ndarray:
    """(nb, 2) uint8 → (nb, 1) float32 (a little-endian fp16 per row)."""
    return cols.copy().view(np.float16).astype(np.float32)


# ───────────────────────── unquantized ─────────────────────────

def dq_f32(raw, n):
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def dq_f16(raw, n):
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def dq_bf16(raw, n):
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


# ───────────────────────── legacy block quants (block = 32) ─────────────────────────

def dq_q8_0(raw, n):
    b = _blocks(raw, 34)
    d = _f16(b[:, 0:2])                       # (nb,1)
    qs = b[:, 2:34].copy().view(np.int8).astype(np.float32)   # (nb,32)
    return (d * qs).reshape(-1)


def dq_q4_0(raw, n):
    b = _blocks(raw, 18)
    d = _f16(b[:, 0:2])                       # (nb,1)
    qs = b[:, 2:18]                           # (nb,16)
    lo = (qs & 0x0F).astype(np.float32) - 8
    hi = (qs >> 4).astype(np.float32) - 8
    out = np.empty((b.shape[0], 32), np.float32)
    out[:, :16] = d * lo                      # ggml: low nibble → index i
    out[:, 16:] = d * hi                      #       high nibble → index i+16
    return out.reshape(-1)


def dq_q4_1(raw, n):
    b = _blocks(raw, 20)
    d = _f16(b[:, 0:2]); m = _f16(b[:, 2:4])
    qs = b[:, 4:20]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.empty((b.shape[0], 32), np.float32)
    out[:, :16] = d * lo + m
    out[:, 16:] = d * hi + m
    return out.reshape(-1)


def _q5_bits(qh4: np.ndarray):
    """qh4: (nb,4) uint8 → (low 5th-bit, high 5th-bit) each (nb,16) in {0,16}."""
    qh = qh4.copy().view(np.uint32)            # (nb,1) little-endian
    j = np.arange(16, dtype=np.uint32)
    lo = ((qh >> j) & 1).astype(np.float32) * 16
    hi = ((qh >> (j + 16)) & 1).astype(np.float32) * 16
    return lo, hi


def dq_q5_0(raw, n):
    b = _blocks(raw, 22)
    d = _f16(b[:, 0:2])
    bit_lo, bit_hi = _q5_bits(b[:, 2:6])
    qs = b[:, 6:22]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.empty((b.shape[0], 32), np.float32)
    out[:, :16] = d * (lo + bit_lo - 16)
    out[:, 16:] = d * (hi + bit_hi - 16)
    return out.reshape(-1)


def dq_q5_1(raw, n):
    b = _blocks(raw, 24)
    d = _f16(b[:, 0:2]); m = _f16(b[:, 2:4])
    bit_lo, bit_hi = _q5_bits(b[:, 4:8])
    qs = b[:, 8:24]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.empty((b.shape[0], 32), np.float32)
    out[:, :16] = d * (lo + bit_lo) + m
    out[:, 16:] = d * (hi + bit_hi) + m
    return out.reshape(-1)


def dq_q8_1(raw, n):
    b = _blocks(raw, 36)
    d = _f16(b[:, 0:2])                         # bytes 2:4 are the sum `s` (unused for dequant)
    qs = b[:, 4:36].copy().view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)


# ───────────────────────── K-quants (superblock = 256) ─────────────────────────

def _k_scales_mins(scales12: np.ndarray):
    """
    Unpack the 12-byte K-quant scales block into (nb,8) 6-bit scales and mins,
    mirroring ggml's get_scale_min_k4.
    """
    q = scales12.astype(np.uint16)            # avoid overflow in shifts
    nb = q.shape[0]
    sc = np.empty((nb, 8), np.float32)
    mn = np.empty((nb, 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = q[:, j] & 63
            mn[:, j] = q[:, j + 4] & 63
        else:
            sc[:, j] = (q[:, j + 4] & 0x0F) | ((q[:, j - 4] >> 6) << 4)
            mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return sc, mn


def dq_q4_k(raw, n):
    b = _blocks(raw, 144)
    d = _f16(b[:, 0:2])                        # (nb,1)
    dmin = _f16(b[:, 2:4])                     # (nb,1)
    sc, mn = _k_scales_mins(b[:, 4:16])        # (nb,8) each
    qs = b[:, 16:144]                          # (nb,128)

    nb = b.shape[0]
    out = np.empty((nb, 256), np.float32)
    for g in range(4):                         # 4 groups of 64 weights
        qb = qs[:, g * 32:(g + 1) * 32]        # (nb,32)
        lo = (qb & 0x0F).astype(np.float32)
        hi = (qb >> 4).astype(np.float32)
        i0, i1 = 2 * g, 2 * g + 1
        out[:, g*64:g*64+32]    = d * sc[:, i0:i0+1] * lo - dmin * mn[:, i0:i0+1]
        out[:, g*64+32:g*64+64] = d * sc[:, i1:i1+1] * hi - dmin * mn[:, i1:i1+1]
    return out.reshape(-1)


def dq_q5_k(raw, n):
    b = _blocks(raw, 176)
    d = _f16(b[:, 0:2])
    dmin = _f16(b[:, 2:4])
    sc, mn = _k_scales_mins(b[:, 4:16])
    qh = b[:, 16:48]                           # (nb,32) — the 5th bits
    qs = b[:, 48:176]                          # (nb,128)

    nb = b.shape[0]
    out = np.empty((nb, 256), np.float32)
    for g in range(4):
        qb = qs[:, g * 32:(g + 1) * 32]
        lo = (qb & 0x0F).astype(np.float32)
        hi = (qb >> 4).astype(np.float32)
        # 5th bit: group g uses bit 2g (low half) and 2g+1 (high half) of qh.
        bit_lo = ((qh >> (2 * g)) & 1).astype(np.float32) * 16
        bit_hi = ((qh >> (2 * g + 1)) & 1).astype(np.float32) * 16
        i0, i1 = 2 * g, 2 * g + 1
        out[:, g*64:g*64+32]    = d * sc[:, i0:i0+1] * (lo + bit_lo) - dmin * mn[:, i0:i0+1]
        out[:, g*64+32:g*64+64] = d * sc[:, i1:i1+1] * (hi + bit_hi) - dmin * mn[:, i1:i1+1]
    return out.reshape(-1)


def dq_q6_k(raw, n):
    b = _blocks(raw, 210)
    ql = b[:, 0:128]                           # lower 4 bits
    qh = b[:, 128:192]                         # upper 2 bits
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)  # (nb,16) signed
    d = _f16(b[:, 208:210])                    # (nb,1)

    nb = b.shape[0]
    out = np.empty((nb, 256), np.float32)
    is_arr = np.repeat([0, 1], 16)             # l//16 for l in 0..31
    for h in range(2):                         # two halves of 128
        qlh = ql[:, h * 64:h * 64 + 64]
        qhh = qh[:, h * 32:h * 32 + 32].astype(np.int16)   # (nb,32)
        sch = sc[:, h * 8:h * 8 + 8]           # (nb,8)
        ql_l = qlh[:, :32].astype(np.int16)
        ql_l32 = qlh[:, 32:64].astype(np.int16)

        q1 = ((ql_l   & 0x0F) | (((qhh >> 0) & 3) << 4)).astype(np.float32) - 32
        q2 = ((ql_l32 & 0x0F) | (((qhh >> 2) & 3) << 4)).astype(np.float32) - 32
        q3 = ((ql_l   >> 4)   | (((qhh >> 4) & 3) << 4)).astype(np.float32) - 32
        q4 = ((ql_l32 >> 4)   | (((qhh >> 6) & 3) << 4)).astype(np.float32) - 32

        base = h * 128
        out[:, base:base+32]      = d * sch[:, is_arr + 0] * q1
        out[:, base+32:base+64]   = d * sch[:, is_arr + 2] * q2
        out[:, base+64:base+96]   = d * sch[:, is_arr + 4] * q3
        out[:, base+96:base+128]  = d * sch[:, is_arr + 6] * q4
    return out.reshape(-1)


# ───────────────────────── dispatch ─────────────────────────

_DEQUANT = {
    0: dq_f32, 1: dq_f16, 30: dq_bf16,
    2: dq_q4_0, 3: dq_q4_1, 6: dq_q5_0, 7: dq_q5_1, 8: dq_q8_0, 9: dq_q8_1,
    12: dq_q4_k, 13: dq_q5_k, 14: dq_q6_k,
}


def dequantize(ggml_type: int, raw: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize one tensor's raw bytes to a flat float32 array."""
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"ggml type {ggml_type} not supported yet. "
            f"Supported: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K."
        )
    return fn(raw, n_elements)


def is_supported(ggml_type: int) -> bool:
    return ggml_type in _DEQUANT
