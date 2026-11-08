"""Runtime K/V cache storage, one file per concern.

    (the KV codec itself now lives in ../kv/codec.py and the
    resident banded cache in ../kv/banded_cache.py; the old fixed-tau
    kv_band.py / kv_band_batch.py are archived in backup/kv_channel_20261002/removed)
    kv_cache_v7.py         26  DEPRECATED alias for kv_cache_qvb8.py; it
                               re-exports under the old name and warns on
                               import. Nothing imports it.
    int8_kv_cache.py      433  INT8 quantized cache, an independent baseline
                               to compare the banded one against

The retired rANS experiment's measured reason this package uses no entropy coding -- see ../entropy/, which is the
retired rANS experiment.
"""
