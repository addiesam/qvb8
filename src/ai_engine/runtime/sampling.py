"""Qwen3.5's recommended sampling presets, shared by chat.py and llm_server.py.

Qwen publishes DIFFERENT recommended sampling for its two modes, so the
preset follows think/no-think rather than being one fixed pair: thinking
0.6/20/0.95, non-thinking 0.7/20/0.8. greedy opts out of both into plain
deterministic decoding.
"""

THINK_PRESET = (0.6, 20, 0.95)
NO_THINK_PRESET = (0.7, 20, 0.8)
GREEDY_PRESET = (0.0, 20, 1.0)


def sampling_preset(think: bool, greedy: bool = False) -> tuple[float, int, float]:
    """Return (temperature, top_k, top_p) for the given mode."""
    if greedy:
        return GREEDY_PRESET
    return THINK_PRESET if think else NO_THINK_PRESET


def resolve_sampling(think: bool, greedy: bool, temperature, top_k, top_p):
    """Fill in any of temperature/top_k/top_p that are None from the preset.

    Values already set (not None) are left untouched, so a caller's explicit
    choice is never overridden by the preset.
    """
    preset = sampling_preset(think, greedy)
    if temperature is None:
        temperature = preset[0]
    if top_k is None:
        top_k = preset[1]
    if top_p is None:
        top_p = preset[2]
    return temperature, top_k, top_p


# Whether tok.decode() drops special tokens (<think>, </think>, etc.) from
# what the caller gets back -- the same kind of shared, per-script model
# control as think/no-think and the sampling preset above, not something
# each script should decide on its own.
#
# chat.py is the one place meant for inspecting what the model actually
# generated (see its own history: it silently hid the model's <think> tag
# for a long time because it inherited True from spec_decode.py without
# anyone deciding that on purpose). Every pipeline script, by contrast,
# consumes the text programmatically -- make_script.py's rewrite,
# generate_slides.py's JSON, llm_server.py's API response -- and special tokens
# in there are noise at best, a parse break at worst.
PIPELINE_SKIP_SPECIAL_TOKENS = True
CHAT_SKIP_SPECIAL_TOKENS = False
