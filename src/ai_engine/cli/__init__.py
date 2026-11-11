"""Interactive and one-shot text interfaces.

    chat_client.py          terminal chat for a running server (API client, stdlib only)

This is where a human types. It owns argv parsing and the stdout stream and
nothing else -- the generation itself is ai_engine.generation, shared byte-for-byte
with servers/llm_server.py so the two cannot drift.
"""
