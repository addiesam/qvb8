"""One subpackage per pack-format version (qvb8 today; a future qvb9 is a sibling).

A version owns what is specific to its bytes: the pack writer (calibration), the
reader classes and CUDA kernels. Everything version-independent -- the format registry,
the certified KV cache (kv/), the MTP draft head and speculative decode (decode/),
the loaders (run/, swap/) -- lives outside and reaches a version only through
common/formats.py.
"""
