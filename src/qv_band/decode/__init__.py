"""Speculative decoding: the engine and the drafters it can use.

    spec_decode.py    3165  self-speculative greedy decoding on the
                         checkpoint's own MTP head. The engine.
    ring_snapshot.py  405  per-draft-token recurrent-state ring for
                         hybrid linear-attention models. Without it a
                         PARTIAL accept (59% of all rejections on this
                         project's traffic) has to replay the whole prefix.
    dspark_draft.py   280  drafter variant
    dflash_draft.py        DFlash / DFlash 2 block drafter: a separately
                         trained draft model fed by intermediate target-
                         layer outputs (spec_decode captures them from the
                         prefill and verify forwards). Optionally qvb8-packed.
    ngram_draft.py    110  n-gram / prompt-lookup drafter, needs no draft
                         model at all

ring_snapshot.py has no importer outside this package -- spec_decode.py is
its only caller. That is not dead code; it is one consumer of a 405-line
module that would be 400 wasted lines inlined.
"""
