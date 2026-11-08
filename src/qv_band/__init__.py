"""QV-band: permutation-free, error-bounded weight compression (per-row width).

See publication/qv_band_complete.tex and the qv-band-perrow design.  This is the
runtime-optimal variant: one shared codebook per matrix + per-row band-id width,
pure-lookup decode (no entropy/rANS), ~0.91 B/elem = 2.2x smaller than fp16.
"""
