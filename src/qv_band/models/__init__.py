"""Model adapters: one folder per model family, holding only facts about that model.

  <name>/adapter.py      which framework mechanisms the model uses (layer, cache, drafter, loop kinds) and its defaults
  <name>/chat_format.py  chat-template quirks, reasoning and tool-call parsing (only if they differ)

Mechanisms never live here: layer kinds are in qv_band/layers, cache kinds in qv_band/kv, drafters and decode loops
in qv_band/decode, the loader in qv_band/run. The loader finds the adapter with engine.find_adapter(config).

  qwen/      Qwen3.5 / 3.6 / 3.8
  ripple/    Ripple-65B
  nemotron/  Nemotron-H Omni (not run since the 2026-11-10 rebuild: no pack on the the dev GPU machine)
"""
