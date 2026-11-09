"""chat.py for Ripple-65B: installs the Ripple hooks (attention, cache, rollback, NoPE drafter) before the model
loads, fills in the Ripple defaults, then runs the normal chat CLI (all of chat.py's flags work; yours win).

    python -m ai_engine.cli.chat_ripple                       # interactive, default pack + DFlash2 drafter
    python -m ai_engine.cli.chat_ripple "Explain entropy" --no-interactive

Defaults (same as ripple_server.sh): pack RIPPLE_PACK, --drafter dflash (k = full block, 15) --draft-model RIPPLE_DRAFT,
--kv-cache qvb8, --prefill-chunk 4096 (the CLI default is 512), --max-memory-gb 100 (the generic CLI default of 48 GB is below the 65B's ~62 GB).
"""
import os
import sys

PACK = os.environ.get("RIPPLE_PACK", "/mnt/backup/qvb8/Ripple-65B-Preview_qvb8_1e-3")
DRAFT = os.environ.get("RIPPLE_DRAFT", "/mnt/backup/qvb8/Ripple-65B-Preview-DFlash2_qvb8_5e-3")


def _with_defaults(argv: list[str]) -> list[str]:
    def has(flag):
        return any(a == flag or a.startswith(flag + "=") for a in argv)
    args = list(argv)
    if not args or args[0].startswith("-") or not os.path.isdir(args[0]):
        args = [PACK] + args                         # model path is chat.py's first positional
    for flag, val in (("--drafter", "dflash"), ("--draft-model", DRAFT),
                      ("--kv-cache", "qvb8"), ("--prefill-chunk", "4096"), ("--max-memory-gb", "100")):
        if not has(flag):
            args += [flag, val]
    return args


sys.argv = [sys.argv[0]] + _with_defaults(sys.argv[1:])

import qv_band.models.ripple.run  # noqa: E402,F401  -- installs the Ripple hooks before the model loads

from ai_engine.cli.chat import main  # noqa: E402

if __name__ == "__main__":
    main()
