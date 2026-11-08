"""pack: QVB8 pack reader (release build -- the pack writer / calibration is not included)."""
from __future__ import annotations
import struct
import numpy as np
from ...common.errors import Qvb8Error, UncertifiableTensor
from .decode import MAGIC, VERSION, HEADER_BYTES, _HEADER_FMT, _HEADER_FMT_V10, _HEADER_FMT_V9, MAX_NBITS, MAX_CERTIFIED_NBITS, FP16_MANTISSA_BITS, _PACK_BLOCK_ELEMS, _E4M3_POS_CODES, _e4m3_lut, _e4m3_decode, _nbits_for, _combine_escapes, fma32, reconstruct, pack_bits_rows, unpack_bits_rows, read_header, unpack_tensor, decode_tensor
_CHUNK_ELEMS = 4 << 20
