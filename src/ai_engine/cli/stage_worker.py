"""Stage worker: runs a range of decoder layers of a pack for a main server on another machine
(qv_band/decode/remote_stage.py; design: qv_band/design_doc/engine_architecture.md section 6).

    python -m ai_engine.cli.stage_worker PACK --stage-port 50818 --stage-layers 0:12 [the server's model flags]

then, on the main machine:

    scripts/servers/llm_server.sh -m PACK --pipeline HOST:50818,local --pipeline-split 12

Both read the same pack with the same --dtype and --kv-* flags. Each loads only its own layers; the worker loads
no embedding, no drafter and samples nothing. Several cards on the worker: add --pipeline cuda:0,cuda:1.
The link carries no authentication: bind it to a private network (--stage-host).
"""
import argparse

import os
import sys

from ai_engine.cli import kv_offload, pipeline_flags

kv_offload.apply_from_argv()          # must precede the first qv_band.kv import
pipeline_flags.apply_from_argv()      # --pipeline cuda:0,cuda:1: the stage's own cards


def _layers_from_argv() -> None:
    """--stage-layers a:b -> QV_LAYERS_HERE=a:b,only, before the loader is imported: only those layers are read."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--stage-layers", default=None)
    a, _ = ap.parse_known_args(sys.argv[1:])
    if a.stage_layers:
        os.environ["QV_LAYERS_HERE"] = f"{a.stage_layers},only"


_layers_from_argv()

from ai_engine.cli.serve import tolerate_autotune_launch_oom  # noqa: E402


def main() -> None:
    tolerate_autotune_launch_oom()
    from ai_engine.generation import add_generation_args, load_qvb8_llm, params_from_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="self-contained pack folder (the same pack as the main server)")
    ap.add_argument("--stage-port", type=int, required=True)
    ap.add_argument("--stage-host", default="0.0.0.0")
    ap.add_argument("--stage-layers", required=True, metavar="a:b", help="decoder layers a..b-1 run here")
    kv_offload.add_flags(ap)
    pipeline_flags.add_flags(ap)
    add_generation_args(ap)
    args = ap.parse_args()
    a, _, b = args.stage_layers.partition(":")
    if not (a.isdigit() and b.isdigit()):
        raise SystemExit("--stage-layers a:b")
    args.draft_k = 0
    params = params_from_args(args)
    llm = load_qvb8_llm(args.model, params, verbose=True)
    from qv_band.decode.remote_stage import StageHost
    StageHost(llm.model, llm.tok, int(a), int(b), pack=str(llm.bundle.pack)).serve(args.stage_port, args.stage_host)


if __name__ == "__main__":
    main()
