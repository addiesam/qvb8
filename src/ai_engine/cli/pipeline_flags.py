"""Multi-card placement as server flags (qv_band/run/placement.py reads QV_PIPELINE / QV_PIPELINE_SPLIT when the
model loads; the flags set them first thing in an entry point, like kv_offload.py).

    --pipeline cuda:0,cuda:1,...,cuda:N-1   any number of cards N >= 2, in layer order; each holds a contiguous range
                                            of decoder layers; the LAST card is the main card (embedding, norm,
                                            lm_head, drafter, sampling)
    --pipeline-split s1,...,sN-1            the first layer of cards 2..N; omitted: balanced by weight bytes
                                            (deterministic, printed at load)
Served by the continuous-batching scheduler like one card.

Layers on another machine (qv_band/decode/remote_stage.py; the other machine runs ai_engine.cli.stage_worker):

    --pipeline HOST:PORT,local --pipeline-split s    layers 0..s-1 on the stage at HOST:PORT, layers s.. here
"""
import argparse
import os
import sys


def add_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pipeline", default=None, metavar="cuda:0,cuda:1,...",
                    help="place the layers over these cards (any number >= 2; the last = main card). "
                         "See ai_engine/cli/pipeline_flags.py")
    ap.add_argument("--pipeline-split", default=None, metavar="s1,...",
                    help="--pipeline: first layer of cards 2..N (default: balanced by weight bytes)")


def apply_from_argv(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_flags(ap)
    a, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if a.pipeline is not None and not a.pipeline.replace(" ", "").startswith("cuda:"):
        parts = [p.strip() for p in a.pipeline.split(",")]
        split = (a.pipeline_split or "").strip()
        if len(parts) != 2 or parts[1] != "local" or ":" not in parts[0] or not split.isdigit():
            raise SystemExit("--pipeline with a remote stage: HOST:PORT,local together with --pipeline-split s "
                             "(layers 0..s-1 on the stage, the rest here)")
        os.environ["QV_STAGES"] = f"{parts[0]}=0:{split}"
        os.environ["QV_LAYERS_HERE"] = f"{split}:"          # the loader reads only the layers that run here
        return
    if a.pipeline is not None:
        os.environ["QV_PIPELINE"] = a.pipeline
    if a.pipeline_split is not None:
        if a.pipeline is None:
            raise SystemExit("--pipeline-split needs --pipeline")
        os.environ["QV_PIPELINE_SPLIT"] = a.pipeline_split
