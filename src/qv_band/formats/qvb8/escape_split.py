"""escape_split: QVB8 pack reader (release build -- the pack writer / calibration is not included)."""
from __future__ import annotations
import numpy as np
FP16_MANTISSA_BITS = 10
import json as _json
import os as _os
import warnings as _warnings
TABLES_FILE = 'escape_tables.json'
_TABLES_DOC = {'bf16': 'the escape list stored in the pack file (all escapes); used only by decodes that round the weight to bf16', 'fp16': "used by every fp16 / fp32 decode: the pack's list restricted to 'fp16', or without 'bf16_only' (the shorter of the two is stored). A bf16-only escape is one where the band value already holds |fp16(band) - w| <= tau and |band - w| <= tau (checked when this file was written). fp16-only tensors: one table, counts only"}
_CACHE: dict = {}

def fp16_list_for(path: str | None):
    """The fp16 escape list of pack file `path` from its folder's escape_tables.json, or None (no file / no entry)."""
    if not path:
        return None
    d = _os.path.abspath(_os.path.dirname(path))
    if d not in _CACHE:
        f = _os.path.join(d, TABLES_FILE)
        try:
            if _os.path.exists(f):
                with open(f) as fh:
                    data = _json.load(fh)
                if data.get('version') != 2:
                    raise EscapeTableError(f'{f}: version {data.get('version')}, this reader knows 2; {_rebuild_hint(d)}')
                _CACHE[d] = data['tensors']
            else:
                _CACHE[d] = {}
        except (OSError, ValueError, KeyError) as ex:
            raise EscapeTableError(f'{f}: unreadable ({ex}); {_rebuild_hint(d)}') from None
    e = _CACHE[d].get(_os.path.basename(path))
    if e is None or e.get('restore') != 'fp16+bf16':
        return None
    if 'fp16' in e:
        return ('fp16', np.asarray(e['fp16'], np.int64), int(e['n_bf16']), int(e['n_fp16']))
    return ('bf16_only', np.asarray(e['bf16_only'], np.int64), int(e['n_bf16']), int(e['n_fp16']))

class EscapeTableError(RuntimeError):
    pass

def _rebuild_hint(where: str) -> str:
    return "download escape_tables.json from the pack's Hugging Face repository into the pack folder"

def select_fp16(full_idx: np.ndarray, fp16_list, where: str='') -> np.ndarray:
    """Keep-mask over the pack's full escape list for the fp16 table given in escape_tables.json. Validation only:
    the file must match the pack (same escape count, every listed position in the pack's list), else an error."""
    kind, lst, n_bf16, n_fp16 = fp16_list
    full_idx = np.asarray(full_idx, np.int64)
    if n_bf16 != len(full_idx):
        raise EscapeTableError(f'{where}: {TABLES_FILE} says {n_bf16} escapes, the pack has {len(full_idx)} -- the file belongs to another pack; {_rebuild_hint(where)}')
    hit = np.isin(full_idx, lst)
    if int(hit.sum()) != len(lst):
        raise EscapeTableError(f'{where}: {TABLES_FILE} lists {len(lst) - int(hit.sum())} positions that are not escapes of this pack; {_rebuild_hint(where)}')
    keep = hit if kind == 'fp16' else ~hit
    if int(keep.sum()) != n_fp16:
        raise EscapeTableError(f'{where}: {TABLES_FILE} fp16 table has {int(keep.sum())} entries, n_fp16 says {n_fp16}; {_rebuild_hint(where)}')
    return keep

def fp16_missing_error(where: str) -> EscapeTableError:
    return EscapeTableError(f'{where}: fp16 run on an fp16+bf16 pack without {TABLES_FILE} (the fp16 escape table); {_rebuild_hint(where)}')

def require_fp16_tables(pack_dir: str) -> None:
    """For an fp16 run: every fp16+bf16 tensor of the pack at `pack_dir` (and its component folders) must have its fp16
    table in escape_tables.json. Raises EscapeTableError otherwise. Reads the manifests only."""
    man_path = _os.path.join(pack_dir, 'manifest.json')
    if not _os.path.exists(man_path):
        for c in sorted(_os.listdir(pack_dir)) if _os.path.isdir(pack_dir) else []:
            if _os.path.isfile(_os.path.join(pack_dir, c, 'manifest.json')):
                require_fp16_tables(_os.path.join(pack_dir, c))
        return
    with open(man_path) as fh:
        man = _json.load(fh)
    for comp in man.get('components', []) or []:
        require_fp16_tables(_os.path.join(pack_dir, comp))
    tensors = man.get('tensors')
    if not isinstance(tensors, dict):
        return
    need = [i['pack'] for i in tensors.values() if i['restore'] == 'fp16+bf16']
    if not need:
        return
    f = _os.path.join(pack_dir, TABLES_FILE)
    have = {}
    if _os.path.exists(f):
        with open(f) as fh:
            have = _json.load(fh).get('tensors', {})
    missing = [p for p in need if p not in have]
    if missing:
        raise EscapeTableError(f'{pack_dir}: fp16 run, but {len(missing)} fp16+bf16 tensors have no fp16 escape table ({TABLES_FILE} {('missing' if not have else 'incomplete')}); {_rebuild_hint(pack_dir)}')
