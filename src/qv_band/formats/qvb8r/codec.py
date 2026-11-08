"""codec: QVB8 pack reader (release build -- the pack writer / calibration is not included)."""
from __future__ import annotations
import numpy as np
PB = 12
MQ = 1 << PB
MAX_ID = 2048
MAX_FREQ = 1024
CENTRE = 256

def slot_tables(F: np.ndarray) -> np.ndarray:
    """(K, N) frequencies -> (K, 4096) packed slot entries."""
    F = F.astype(np.int64)
    K, N = F.shape
    C = np.concatenate([np.zeros((K, 1), np.int64), np.cumsum(F, 1)[:, :-1]], 1)
    tab = np.zeros((K, MQ), np.uint32)
    for k in range(K):
        s = np.repeat(np.arange(N), F[k])
        d = np.arange(MQ) - C[k][s]
        tab[k] = s | d << 11 | F[k][s] - 1 << 22
    return tab
ROWS_PER_TILE_DIV = 4096
