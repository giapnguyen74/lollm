"""
gguf_reader.py — a self-contained parser for the GGUF binary format.

Parses the four regions described in docs/gguf-format.md:
  1. header        — magic, version, tensor_count, metadata_kv_count
  2. metadata KV   — hyperparameters + embedded tokenizer
  3. tensor info   — name, shape, ggml type, offset (per tensor)
  4. tensor data   — aligned blob; we expose each tensor's raw bytes

No external libraries beyond numpy. The file is memory-mapped, so opening a
multi-GB model is instant — bytes page in only when a tensor is read.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass

import numpy as np


# ── GGUF metadata value types (spec §) ──
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f", BOOL: "<?",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

# ── ggml tensor types → (elements per block, bytes per block) ──
# Only what we need to compute byte sizes; dequant.py interprets the bytes.
GGML_TYPE_SIZE = {
    0:  (1, 4),    # F32
    1:  (1, 2),    # F16
    2:  (32, 18),  # Q4_0
    3:  (32, 20),  # Q4_1
    6:  (32, 22),  # Q5_0
    7:  (32, 24),  # Q5_1
    8:  (32, 34),  # Q8_0
    9:  (32, 36),  # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    30: (1, 2),    # BF16
}

GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}


@dataclass
class TensorInfo:
    name: str
    ggml_type: int
    gguf_dims: tuple          # as stored (ne0 = innermost/contiguous)
    torch_shape: tuple        # reversed dims → PyTorch (out, in) convention
    offset: int               # relative to the tensor-data region
    nbytes: int


class GGUFReader:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0

        self.metadata: dict = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._parse()

    # ---- low-level cursor helpers ----
    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self._mm, self.pos)[0]
        self.pos += size
        return val

    def _read_string(self) -> str:
        n = self._read("<Q")
        s = self._mm[self.pos:self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def _read_value(self, vtype: int):
        if vtype in _SCALAR:
            return self._read(_SCALAR[vtype])
        if vtype == STRING:
            return self._read_string()
        if vtype == ARRAY:
            elem_type = self._read("<I")
            n = self._read("<Q")
            return [self._read_value(elem_type) for _ in range(n)]
        raise ValueError(f"unknown GGUF metadata value type: {vtype}")

    # ---- structure ----
    def _parse(self):
        magic = self._mm[0:4]
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        self.pos = 4
        self.version = self._read("<I")
        tensor_count = self._read("<Q")
        kv_count = self._read("<Q")

        # region 2: metadata key-values
        for _ in range(kv_count):
            key = self._read_string()
            vtype = self._read("<I")
            self.metadata[key] = self._read_value(vtype)

        alignment = self.metadata.get("general.alignment", 32)

        # region 3: tensor info table
        infos = []
        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = self._read("<I")
            dims = tuple(self._read("<Q") for _ in range(n_dims))
            ggml_type = self._read("<I")
            offset = self._read("<Q")

            block, type_size = GGML_TYPE_SIZE[ggml_type]
            n_elem = 1
            for d in dims:
                n_elem *= d
            nbytes = (n_elem // block) * type_size
            infos.append((name, ggml_type, dims, offset, nbytes))

        # region 3→4 boundary: pad to alignment
        self._data_start = (self.pos + alignment - 1) // alignment * alignment

        for name, ggml_type, dims, offset, nbytes in infos:
            self.tensors[name] = TensorInfo(
                name=name,
                ggml_type=ggml_type,
                gguf_dims=dims,
                torch_shape=tuple(reversed(dims)),
                offset=offset,
                nbytes=nbytes,
            )

    # ---- access ----
    def raw_tensor(self, name: str) -> np.ndarray:
        """Return the raw bytes of a tensor as a uint8 numpy array (a view)."""
        t = self.tensors[name]
        start = self._data_start + t.offset
        return np.frombuffer(self._mm, dtype=np.uint8, count=t.nbytes, offset=start)

    def close(self):
        self._mm.close()
        self._f.close()
