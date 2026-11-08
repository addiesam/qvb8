"""Interactive and one-shot text interfaces.

    chat.py                 prompt-loop or one-shot, straight to stdout

This is where a human types. It owns argv parsing and the stdout stream and
nothing else -- the generation itself is ai_engine.generation, shared byte-for-byte
with servers/llm_server.py so the two cannot drift.
"""
