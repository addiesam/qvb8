"""LLM server entry for Ripple-65B: the SAME server and engine path as Qwen (llm_server, continuous-batching
scheduler, prefix store, qvb8 KV, DFlash). Importing qv_band.models.ripple.run first installs the Ripple-specific
pieces: attention math via the engine's attention registry (gate, value residual, NoPE globals, sliding window),
the chat format (model-opened <think>, JSON tool calls) and the NoPE drafter. Its 64 sliding layers are ordinary
qvb8 BandedKVLayers with window=1024 (BandedKVCache), so the scheduler treats them like any other layer.

    scripts/ripple_server.sh            (launcher with the Ripple defaults)
"""
import qv_band.models.ripple.run  # noqa: F401  -- installs the Ripple hooks before the model loads

from ai_engine.cli.serve import main

if __name__ == "__main__":
    main()
