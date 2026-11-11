"""Shared inference core for every QV-band interface.

An interface (ai_engine.servers.llm_server, a future
tool) differs from every other
only in HOW a request arrives (argv, HTTP JSON, a tool call) and HOW its
output is delivered (a stdout stream, SSE, a return value). Everything
between -- device resolution, the
CPU thread guard, the memory ceiling, model load, the four drafter modes,
the KV-cache presets, and the single speculative_generate() call with its
KV-cache-preset -> parameter mapping -- is identical and lives here, so it
cannot drift between interfaces. (It had drifted: a max_new default bound
to ctx_size caused a 70 GB VRAM blowup in one interface, and the KV-cache
flags existed in one but not the other.)

WHAT THIS IS, AND WHY IT IS NOT IN qv_band
-------------------------------------------
This is the GLUE that turns qv_band's parts into something a server can
serve -- it defines no codec, no pack reader and no weight format. qv_band
supplies the pieces it assembles (run.run_inference for the pack reader,
decode.spec_decode for generation, decode.ngram_draft / dspark_draft for
the drafters, kv.codec for the codec tolerances) and
ai_engine.runtime supplies the device and memory decisions, which are
modality-agnostic and shared with the image, lip-sync and TTS paths.

It has lived in three places -- engine.gen_core, then qv_band.engine, then
here. The current home is the one that removes a cycle: qv_band needs
nothing from this module (qv_band's own calibration/eval/inference never
import it), so keeping it inside qv_band made qv_band depend on ai_engine
for no reason while ai_engine's servers imported it back. Here, the arrow
points one way.

Typical use::

    from ai_engine.generation.gen_core import (
        GenParams, add_generation_args, params_from_args,
        load_model, build_drafter, encode_prompt, run_generate)

    ap = argparse.ArgumentParser()
    add_generation_args(ap)                 # shared flags, worded once
    args = ap.parse_args()
    params = params_from_args(args)         # -> GenParams (validated)

    bundle = load_model(args.model, params)         # fills params.ctx_size
    drafter = build_drafter(params.drafter, bundle, max_ctx=params.draft_ctx)
    ids = encode_prompt(bundle, prompt, chat=True, think=True)
    out, stats = run_generate(bundle, drafter, ids, params, stream=True)

Nothing here prints a UI, owns a socket, or parses argv/JSON. An
interface may also build a GenParams by hand instead of via
params_from_args(); GenParams.__post_init__ validates it either way.
"""
from __future__ import annotations

import dataclasses
import os
import re
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Defaults -- the ONE definition. Interfaces import these; they never
# redefine them.
# ---------------------------------------------------------------------------


class DEFAULTS:
    # --ctx-size caps PROMPT tokens (llama.cpp semantics), checked before a
    # forward runs. When the user does not pass --ctx-size, load_model
    # derives it from the loaded model's own max_position_embeddings
    # (Qwen's declared context window); CTX_SIZE here is only the fallback
    # if that field is absent from the config. It is NOT backed by a memory
    # measurement on any machine -- see MEMORY_LIMIT_GB for the ceiling
    # that actually holds.
    CTX_SIZE = 262144

    # Default OUTPUT length when a request gives none. An OUTPUT budget,
    # unrelated to CTX_SIZE. This checkpoint's <think> block routinely runs
    # 1200-5700+ tokens with no internal awareness of any cap, so a low
    # default truncates mid-thought and the model never reaches its answer.
    # Still a real ceiling, so a model that never emits EOS stops on its own
    # instead of growing the KV cache and replay scratch until it exhausts
    # VRAM.
    MAX_OUTPUT_TOKENS = 32767

    MEMORY_LIMIT_GB = 48.0

    # KV-cache method -- ONE named preset, not a pile of independent flags.
    # See KV_CACHE_PRESETS for the full menu.
    #
    # int8 is the default: per-token per-head quantization, cheap
    # elementwise dequant (multiply, not a codebook gather/decode), ~1.0
    # B/elem -- real VRAM saving over bf16 with a much smaller per-step
    # decode cost than the banded codec. Measured on the real 9B
    # (4646-token prompt, the dev GPU): bf16 5.64 tok/s vs banded KV 1.56 tok/s --
    # the banded codec pays a real reconstruction cost every decode step
    # (see kv_cache_qvb8.py's module docstring: "Attention still receives a
    # DENSE tensor... The saving is in what stays resident BETWEEN calls,
    # not in what attention reads during one") that dominated decode time
    # on this hardware. int8 was picked as the default over bf16 for the
    # VRAM saving without paying that codec tax; pass --kv-cache qvb8
    # explicitly when VRAM is tight enough to be worth the extra decode
    # cost, or --kv-cache bf16 for the fastest possible decode when VRAM
    # is not the constraint.
    KV_CACHE = "int8"

    # Banded-KV certified tolerance (qv_band/kv/codec.py), used
    # only by the qvb8 preset: every cached K/V element is guaranteed to decode
    # within this absolute error (verified on the real decode, bf16 storage
    # included). NOT the weight pack's tau. The codec is per-channel affine with
    # a 6-bit floor and raises a head's bits until the guarantee holds -- no
    # escape channel. 0.1 measured (2026-10-02): perplexity within 0.002 (27B)
    # and 0.003 (35B) of a dense KV cache, prefill/decode speed unchanged;
    # KV ~47% of dense at 8k context. A larger tau (0.5) saves bits but costs
    # accuracy; a tau below bf16's grid at the cache's magnitudes is refused.
    KV_TAU = 0.1

    # Tokens kept dense in the KV cache before a bulk flush (band or
    # quantize) -- shared by BOTH the int8 cache and the qvb8 preset
    # (same residual-buffer mechanism, same kwarg passed to each cache
    # class; see GenParams.to_speculative_kwargs). Also the speculative-
    # rollback headroom.
    #
    # The banded cache reconstructs EVERY accumulated segment to dense on every
    # forward step -- genuinely O(n) in segment count, same complexity
    # class FlashAttention/PagedAttention pay in KV-cache length (per-segment
    # reconstruction is cached so an unchanged segment is decoded once, not every step,
    # but a NEW segment still costs its own one-time decode -- the growth
    # is real attention/decode compute, not a caching bug). Raising the
    # flush size directly cuts segment COUNT for a
    # given context length -- fewer, larger segments to attend over per
    # step -- the same lever vLLM/SGLang's block/page-size knob is.
    #
    # 2048 was tried and reverted: a conversation shorter than the
    # residual threshold NEVER FLUSHES AT ALL, so its whole KV cache sits
    # in the dense residual buffer with zero compression -- measured on a
    # real 236-output-token chat: reported ratio was 80.3% (should be
    # ~50%, matching int8) purely because 2048 tokens of context never
    # accumulated. The KV codec (qv_band kv/codec.py) has no rANS
    # recalibration cost of the kind 2048 once compensated for, so there is no longer a
    # reason to trade away short-conversation compression for fewer
    # flushes on long ones. 256 gives real compression on ordinary chat
    # lengths while still keeping segment count low on long contexts.
    KV_RESIDUAL = 256

    # -------------------------------------------------------------------------
    # KV decode-accelerator shadow: how much DECODED fp16 K/V is kept resident so a
    # decode step can skip re-expanding every segment. NOT the compression ratio -- the
    # banded store stays packed either way. A size PER full-attention layer (the unit the cap
    # is enforced in; the total is this times the layer count). It is a cap, not a
    # reservation: the views grow only as the context does, at 2 * kv_heads * head_dim * 2
    # bytes per token per layer (4 KB on the 27B, so 1 MB covers 256 tokens).
    #
    # A slot is cached only while its decoded K/V fits the layer's size; past that NOTHING of
    # it is cached and every step re-decodes the whole history (measured 9B decode, 4 kv
    # heads x 256: 1.45 ms/step at 16k context, 2.82 at 32k, 8.34 at 65k, 16.66 at 131k --
    # the fall-off starts where the shadow passes the size; on the 27B at 48k, 205 ms/step
    # cached against 353 uncached). 512MB per layer covers 131k tokens on the 27B (8 GB in
    # all over its 16 layers); 1024MB measured no faster. For a memory-limited machine give
    # the size the context needs: tokens * 4 KB. 0 keeps none (least VRAM, slower decode).
    KV_SHADOW = "512MB"

    # Chunked-prefill size (qv_band/spec_decode.py's PREFILL_CHUNK): a long
    # prompt is fed through the model this many tokens at a time so peak
    # prefill VRAM is bounded by one chunk, not the whole prompt. 512
    # matches llama.cpp server's default n_ubatch. QV_BAND_PREFILL_CHUNK
    # (spec_decode.py's own env fallback) still applies when neither this
    # default nor --prefill-chunk is what the caller wants changed.
    PREFILL_CHUNK = int(os.environ.get("QV_BAND_PREFILL_CHUNK", "512"))

    # Speculative draft tokens proposed per step. 0 disables drafting.
    # DRAFT_K is the default for the non-DFlash drafters (mtp / n-gram ...); a DFlash drafter (dflash,
    # n-gram_with_dflash) defaults to its full block (block_size - 1 from its config.json), and to DRAFT_K_DFLASH
    # only if that cannot be read. An explicit --draft / draft_k always wins.
    DRAFT_K = 8
    DRAFT_K_DFLASH = 7
    # DFlash: the model's own trained block drafter (qv_band.decode.
    # dflash_draft). Measured 1.7-3.2x over plain decoding on the 9B/27B/
    # 35B-A3B packs. A model with no DFlash drafter uses n-gram instead.
    #
    # The drafter's checkpoint is NOT guessed: --draft-model must name the
    # directory. There is no "auto" any more -- a silent fallback to n-gram
    # costs ~40% throughput and is invisible in the logs, so the path is
    # always explicit. llm_server.sh fills it in for you (it knows where the
    # checkpoints live); the Python API requires it too.
    DRAFTER = "dflash"           # see DRAFTER_CHOICES
    DRAFT_MODEL = None            # dflash draft dir; REQUIRED for dflash
    # Which WEIGHTS the DFlash drafter runs: "dense" (the plain HF
    # checkpoint), "pack" (its qvb8-quantised pack) or "auto".
    #
    # Default is "dense", and that is deliberate. A qvb8 pack of the drafter
    # shrinks its weights to ~52-62% of the dense ones, but the drafter only
    # ever proposes a BLOCK of 8-16 tokens -- i.e. it runs its GEMMs at
    # M=8..16, where the packed path's overhead dominates its weight saving.
    # Measured on this box (Qwen3.8-27B_qvb8_1e-3_bf16, DFlash2, k=7): the
    # packed drafter is SLOWER end to end than the dense one even though it
    # is 1.1 GB smaller on VRAM. qv_band's own dflash_draft docstring says
    # the same thing ("draft steps get slower on the 27B/9B (packed GEMMs at
    # M=8..16), so dense stays the default") -- qv_band's --draft-pack
    # defaults to None for that reason.
    #
    # "auto" uses the pack when one exists for this checkpoint; "pack"
    # forces it; "dense" forces the plain checkpoint.
    DRAFT_PACK = "dense"
    DRAFT_CTX = 4               # n-gram / n-gram_with_mtp lookup window

    # Greedy by default; interfaces layer their own sampling presets on top
    # (resolve_sampling) before building a GenParams.
    TEMPERATURE = 0.0
    TOP_K = 0
    TOP_P = 1.0


# KV-cache presets: the ONE public menu. Each maps to the kv_mode string
# speculative_generate actually consumes -- callers pick a name, never the
# mode. Compression figures are for the 9B at tau=0.15 (bf16 is the fp16
# baseline).
#
#   name                 store / decode                              ~B/elem
#   bf16                 dense fp16, no compression                     2.00
#   int8                 int8 per-token per-head, elementwise decode    ~1.0
#   qvb8                 virtual-band (fixed-tau), decode to dense       ~0.24   <- default
#
# The old flat-direction presets (FlatDirection, raw ids instead of rANS)
# removed -- their storage still needed a <=256-level codebook ceiling
# that broke at ordinary KV tau (measured: tau=0.15 and 0.3 both need
# 258-276 levels, over the ceiling), so they were never usable at this
# codec's actual operating point.
#
# The GPU-resident gather path (a custom C/CUDA kernel reading straight
# from banded storage, qv_band/banded_attention.py) has been removed
# entirely -- the cache always decodes every segment to dense via
# kv_band.py's closed-form fixed-tau formula
#   tau_s = 2*tau                  (global)
#   min_s = segment_min + tau      (per row, at flush time)
#   index = floor((value - segment_min) / tau_s)
#   value = min_s + index * tau_s  (one FMA, no codebook gather)
# cached per segment so an unchanged segment is decoded once, not every
# forward call. That formula IS the QVB8 virtual-band codec, which is
# what the preset is named after.
KV_CACHE_PRESETS: dict[str, str] = {
    # name                 -> kv_mode
    "bf16":                "bf16",
    "int8":                "int8",
    "qvb8":                "qvb8-resident",
}
KV_CACHE_CHOICES = tuple(KV_CACHE_PRESETS)


DRAFTER_CHOICES = ("dflash", "n-gram", "mtp", "dspark", "n-gram_with_mtp", "n-gram_with_dflash")

# Which weights the DFlash drafter runs -- see DEFAULTS.DRAFT_PACK.
#   dense = the plain HF checkpoint (fastest on the 27B/9B; the default)
#   pack  = its qvb8-quantised pack (smaller on VRAM, slower per draft)
#   auto  = pack when one exists for this checkpoint, else dense
DRAFT_PACK_CHOICES = ("dense", "pack", "auto")

# DEVICE_CHOICES/valid_device are engine.runtime's -- imported below rather
# than redefined, after this exact duplication drifted out of sync (this
# module's own copy was missing "all", silently rejecting the multi-GPU
# shard mode runtime.py has always supported, both in GenParams validation
# and in the --device argparse choices list).
from ai_engine.runtime.device import DEVICE_CHOICES, valid_device as _valid_device


def _kv_shadow_bytes_per_token(kv_heads: int, head_dim: int) -> int:
    """Resident bytes of the decoded shadow ONE token costs, one slot, one layer.

    K and V, both fp16: 2 * kv_heads * head_dim * 2. The banded store is not
    part of this -- it is separately compressed and stays packed.
    """
    return 2 * kv_heads * head_dim * 2


def kv_shadow_covers_tokens(per_layer_mb: float, kv_heads: int,
                            head_dim: int) -> int:
    """How much context a per-layer shadow budget can hold before it misses.

    Above this, _decode_banded_joined stops caching the slot entirely (see
    the --kv-shadow share), and decode re-expands the whole history every
    step. This is the number that decides whether a given budget is enough,
    so it is also what /health reports.
    """
    per_token = _kv_shadow_bytes_per_token(kv_heads, head_dim)
    if per_token <= 0 or per_layer_mb <= 0:
        return 0
    return int(per_layer_mb * 1024 * 1024 // per_token)


def parse_kv_shadow(text) -> float:
    """--kv-shadow as a size PER full-attention layer, in MB. '512MB' / '1GB' (case-insensitive,
    decimals allowed), or '0' for none. Anything else is refused, with the accepted forms in the message."""
    t = str(text).strip().lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(gb|g|mb|m)", t)
    if m:
        return float(m.group(1)) * (1024.0 if m.group(2) in ("gb", "g") else 1.0)
    if t == "0":
        return 0.0
    raise ValueError(f"invalid kv_shadow {text!r}: give a size per attention layer with a unit, such as "
                     f"'512MB' or '1GB', or '0' for none")


def _kv_shadow_arg(text: str) -> str:
    """argparse `type`: validate --kv-shadow so a bad value is a normal usage error, not a traceback."""
    import argparse
    try:
        parse_kv_shadow(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


_ENV_SET_BY_US: dict[str, str] = {}


def _set_unless_user(name: str, value: str) -> None:
    """Export `value` unless the USER set `name` themselves. A plain setdefault
    would also keep what a previous apply_kv_shadow() put there, so loading a
    second model in the same process would run on the first model's budget."""
    cur = os.environ.get(name)
    if cur is None or _ENV_SET_BY_US.get(name) == cur:
        os.environ[name] = value
        _ENV_SET_BY_US[name] = value


def apply_kv_shadow(model_dir: str, params: "GenParams") -> dict:
    """Resolve --kv-shadow against the model's real shape and export it.

    WHY AN ENV VAR AND NOT A FUNCTION ARGUMENT. qv_band/kv_cache_qvb8.py
    reads QV_BAND_JOINED_CACHE_MB at IMPORT time (module level, line ~189)
    into _JOINED_CACHE_BYTES / _JOINED_CACHE_CAP_BYTES, so the only way to
    set it is os.environ before the first `import qv_band`. Anything later
    would be read after the module is loaded and silently have no effect --
    which is why every entry point must call this BEFORE load_model().

    WHY THE MODEL'S OWN SHAPE. A slot's decoded shadow is
    2 * ctx * kv_heads * head_dim * 2 bytes, and it is cached only if it
    fits the per-layer budget, so "enough budget" means a specific context
    length. That depends on head_dim (256 across the Qwen3.5/3.6/3.8 family
    here) and on how many layers share the process-wide cap (8 on the 9B, 16
    on the 27B, 10 on the 35B-A3B). A fixed byte count is therefore wrong
    for all but one of them, and wrong SILENTLY: the old 128 MB default
    covered 64k on the 9B but only 32k on the 27B with nothing reporting it.

    Falls back to a 0 budget if the config cannot be read -- the correct
    failure direction, since a missing cache only costs speed while a wrong
    large one costs VRAM the user did not ask to spend.
    """
    import json
    kv_heads, head_dim, n_full, mpe = 0, 0, 0, 0
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        cfg = raw.get("text_config") or raw.get("llm_config") or raw   # llm_config: Nemotron-H Omni
        kv_heads = int(cfg.get("num_key_value_heads") or 0)
        head_dim = int(cfg.get("head_dim")
                       or (cfg.get("hidden_size", 0)
                           // max(1, int(cfg.get("num_attention_heads") or 1))))
        mpe = int(cfg.get("max_position_embeddings") or 0)
        types = cfg.get("layer_types")
        if cfg.get("layers_block_type"):                     # Nemotron-H: mamba / moe / attention
            n_full = sum(1 for t in cfg["layers_block_type"] if t == "attention")
        elif types:
            n_full = sum(1 for t in types if t == "full_attention")
        else:
            # No layer_types: assume every layer attends, which is the
            # conservative direction for the process-wide cap.
            n_full = int(cfg.get("num_hidden_layers") or 0)
    except Exception as exc:            # noqa: BLE001 - config is best-effort
        print(f"[kv-shadow] could not read {model_dir}/config.json ({exc}); "
              f"continuing with NO decode shadow", flush=True)

    # The context the server will accept: load_model (which fills params.ctx_size) runs AFTER
    # this, so use the value resolve_ctx_size will pick. Only used to say whether the size covers it.
    ctx_sz = params.ctx_size or mpe or DEFAULTS.CTX_SIZE
    per_layer = params.kv_shadow_mb(n_full)
    glob = params.kv_shadow_global_mb(n_full)
    # kv_cache_qvb8.py parses both of these with int(), so they must be
    # integral strings -- the per-layer share is a float, and handing it "512.0" raises "invalid literal for int() with base 10"
    # at the first decode step, long after load() looked successful.
    os.environ["QV_BAND_JOINED_CACHE_MB"] = str(int(round(per_layer)))
    os.environ["QV_BAND_GLOBAL_JOINED_CACHE_MB"] = str(int(round(glob)))
    # The engine's decode path (BandedKVLayer.ragged_kv_views) draws on a
    # SEPARATE pair of budgets, QV_BAND_DENSE_VIEW_MB / ..._GLOBAL_DENSE_VIEW_MB
    # (512 / 4096 by default), which the two lines above never reached: with
    # 16 full-attention layers only half of them fit the 4096 MB cap, and the
    # rest rebuilt their whole history every step. An explicit env var wins.
    _set_unless_user("QV_BAND_DENSE_VIEW_MB", str(int(round(per_layer))))
    _set_unless_user("QV_BAND_GLOBAL_DENSE_VIEW_MB", str(int(round(glob))))
    covers = params.kv_shadow_covers_tokens(kv_heads, head_dim, n_full)
    ctx = ctx_sz
    print(f"[kv-shadow] {per_layer:g} MB per layer x {n_full} full-attention layers = {glob:g} MB total "
          f"(kv_heads={kv_heads} head_dim={head_dim})", flush=True)
    if covers:
        # The number that matters, and the one that used to be unknowable:
        # past this context the shadow stops being cached and every decode
        # step re-expands the whole history.
        flag = "OK" if (ctx and covers >= ctx) else "SHORT of --ctx-size"
        print(f"[kv-shadow] covers {covers} tokens of context ({flag})",
              flush=True)
    return dict(kv_shadow_mode=params.kv_shadow, kv_shadow_mb=per_layer,
                kv_shadow_global_mb=glob, kv_shadow_covers_tokens=covers)


# ---------------------------------------------------------------------------
# GenParams -- the settled generation configuration, interface-agnostic.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GenParams:
    """Everything the core needs that is not the model, the tokenizer, or
    the prompt. Built by an interface from argv, a JSON body, or by hand.

    kv_cache is ONE named preset (KV_CACHE_CHOICES). The flags
    speculative_generate() actually takes -- kv_mode, int8_kv,
    qvb8_resident, kv_tau -- are all derived from it (via
    KV_CACHE_PRESETS) in to_speculative_kwargs(), nowhere else.

    ctx_size may be None at construction (user did not pass --ctx-size);
    load_model fills it from the model's own max_position_embeddings. It is
    non-None after load_model. A path that builds a GenParams and skips
    load_model must set ctx_size explicitly or call resolve_ctx_size().
    """
    # generation length / sampling
    max_output_tokens: int = DEFAULTS.MAX_OUTPUT_TOKENS
    temperature: float = DEFAULTS.TEMPERATURE
    top_k: int = DEFAULTS.TOP_K
    top_p: float = DEFAULTS.TOP_P

    # prompt ceiling (the interface enforces it; the number lives here so
    # load_model and callers can see it)
    ctx_size: int | None = None

    # KV cache: one named preset + the qvb8 tolerance + residual length.
    # kv_residual is shared by int8 AND the qvb8 preset.
    kv_cache: str = DEFAULTS.KV_CACHE
    # Decode-accelerator shadow: a size per full-attention layer
    # ('512MB', '1GB', or '0'), per attention layer. See DEFAULTS.KV_SHADOW for what it buys.
    kv_shadow: str = DEFAULTS.KV_SHADOW
    kv_tau: float = DEFAULTS.KV_TAU
    kv_residual: int = DEFAULTS.KV_RESIDUAL
    prefill_chunk: int = DEFAULTS.PREFILL_CHUNK
    # Prefill linears: "qvb8" (certified, default) or "int8" (opt-in, int8 tensor cores on the pack's own ids; see generation/int8_prefill.py)
    prefill_linear: str = "qvb8"

    # drafting
    draft_k: int | None = None          # None -> DRAFT_K_DFLASH for a DFlash drafter, else DRAFT_K
    drafter: str = DEFAULTS.DRAFTER
    draft_ctx: int = DEFAULTS.DRAFT_CTX
    draft_model: str = DEFAULTS.DRAFT_MODEL
    draft_pack: str = DEFAULTS.DRAFT_PACK

    # process
    device: str = "auto"
    # compute dtype of the model: "bf16" / "fp16" (--dtype), None = the model's own (its config). Chosen, never automatic.
    dtype: str | None = None
    max_memory_gb: float = DEFAULTS.MEMORY_LIMIT_GB
    # torch intra-op threads when the model runs on CPU. None = the engine's
    # measured default (1, see load_model). An application passes its own
    # setting here; it must not call torch.set_num_threads itself, because
    # load_model runs after it and would silently override the value.
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.kv_cache not in KV_CACHE_PRESETS:
            raise ValueError(
                f"kv_cache must be one of {KV_CACHE_CHOICES}, "
                f"got {self.kv_cache!r}")
        if self.drafter not in DRAFTER_CHOICES:
            raise ValueError(
                f"drafter must be one of {DRAFTER_CHOICES}, got {self.drafter!r}")
        parse_kv_shadow(self.kv_shadow)         # a bad size fails here, before the model loads
        if self.draft_pack not in DRAFT_PACK_CHOICES:
            raise ValueError(
                f"draft_pack must be one of {DRAFT_PACK_CHOICES}, "
                f"got {self.draft_pack!r}")
        if self.drafter in ("dflash", "n-gram_with_dflash") and self.draft_model in (None, "", "auto"):
            # Fail here, not later inside build_drafter: a dflash run without
            # a checkpoint is a configuration mistake, and the message must
            # name the flag that fixes it.
            raise ValueError(
                "drafter 'dflash' requires draft_model (the DFlash checkpoint "
                "directory) -- 'auto' is not accepted any more. Set "
                "draft_model=/path/to/<model>-DFlash2, or use drafter='n-gram'.")
        if not _valid_device(self.device):
            raise ValueError(
                f"device must be one of {DEVICE_CHOICES} or 'cuda:<int>', "
                f"got {self.device!r}")
        if self.draft_k is None:
            if self.drafter in ("dflash", "n-gram_with_dflash"):
                # Full block: a DFlash drafter drafts its whole block in one pass anyway, and verifying 8 or 16 rows
                # costs about the same here (mma_multi), so k = block_size - 1 (Qwen DFlash2: 8 -> 7; Ripple and
                # Qwen3.5-9B DFlash: 16 -> 15). DRAFT_K_DFLASH only when the block size cannot be read.
                blk = _dflash_block_size(self.draft_model)
                self.draft_k = blk - 1 if blk and blk > 1 else DEFAULTS.DRAFT_K_DFLASH
            else:
                self.draft_k = DEFAULTS.DRAFT_K
        if self.draft_k < 0:
            raise ValueError(f"draft_k must be >= 0, got {self.draft_k}")
        if self.prefill_chunk < 1:
            raise ValueError(
                f"prefill_chunk must be >= 1, got {self.prefill_chunk}")
        if self.max_output_tokens < 1:
            raise ValueError(
                f"max_output_tokens must be >= 1, got {self.max_output_tokens}")
        if self.ctx_size is not None and self.ctx_size < 1:
            raise ValueError(f"ctx_size must be >= 1, got {self.ctx_size}")
        self._clamp_output_budget()

    def _clamp_output_budget(self) -> None:
        """An OUTPUT budget cannot exceed the PROMPT ceiling -- the model
        runs out of context first. No-op while ctx_size is unknown."""
        if self.ctx_size is not None:
            self.max_output_tokens = min(self.max_output_tokens, self.ctx_size)

    def resolve_ctx_size(self, model: Any) -> int:
        """Fill ctx_size from the model's declared context window if the
        user did not pass --ctx-size, and return the resolved value.
        Idempotent: a second call with ctx_size already set is a no-op.

        max_position_embeddings sits on the text sub-config for the
        multimodal Qwen3.5 wrapper and on the top-level config otherwise;
        try both, fall back to DEFAULTS.CTX_SIZE.
        """
        if self.ctx_size is None:
            cfg = getattr(model, "config", None)
            mpe = None
            for obj in (getattr(cfg, "text_config", None), cfg):
                if obj is not None:
                    mpe = getattr(obj, "max_position_embeddings", None)
                    if mpe:
                        break
            self.ctx_size = int(mpe) if mpe else DEFAULTS.CTX_SIZE
            self._clamp_output_budget()
        return self.ctx_size

    def require_ctx_size(self) -> int:
        """ctx_size or a clear error -- for the point where an interface
        compares a prompt length against it. Anything that reaches here
        without ctx_size set skipped load_model AND skipped setting it by
        hand, which is a bug in that path, not a runtime condition."""
        if self.ctx_size is None:
            raise RuntimeError(
                "GenParams.ctx_size is unset: call load_model() (which fills "
                "it from the model config) or resolve_ctx_size(model) before "
                "enforcing a prompt ceiling")
        return self.ctx_size

    # -- kv_cache preset -> speculative_generate kwargs (the ONE mapping) --
    @property
    def kv_mode(self) -> str:
        """kv_mode string for this preset -- the internal value the engine
        consumes. Derived, never set directly."""
        return KV_CACHE_PRESETS[self.kv_cache]

    @property
    def qvb8_resident(self) -> bool:
        return self.kv_mode == "qvb8-resident"

    @property
    def int8_kv(self) -> bool:
        return self.kv_mode == "int8"

    @property
    def effective_kv_tau(self) -> float:
        """tau is meaningful only for the qvb8 preset; the codec is off
        (0.0) for bf16 and int8."""
        return self.kv_tau if self.qvb8_resident else 0.0

    def to_speculative_kwargs(self) -> dict[str, Any]:
        """The keyword arguments speculative_generate() takes for this
        configuration. `ctx_size` must be resolved first (max_new is
        clamped against it) -- require_ctx_size() enforces that."""
        self.require_ctx_size()
        return dict(
            k=self.draft_k,
            max_new=self.max_output_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            kv_tau=self.effective_kv_tau,
            int8_kv=self.int8_kv,
            int8_residual_length=self.kv_residual,   # engine kwarg name; shared by every preset
            qvb8_resident=self.qvb8_resident,
            prefill_chunk=self.prefill_chunk,
        )

    def describe_kv(self) -> str:
        """One-line human description of the KV-cache preset."""
        if self.qvb8_resident:
            return (f"{self.kv_cache}  (per-channel min_s + id*tau_s, "
                    f"certified |err|<={self.kv_tau:g}, no escapes)")
        if self.int8_kv:
            return f"{self.kv_cache}  (int8 per-token per-head)"
        return f"{self.kv_cache}  (dense fp16, no compression)"

    # -- decode-accelerator shadow budget ---------------------------------

    def kv_shadow_mb(self, n_full_attn: int = 1) -> float:
        """The --kv-shadow size per full-attention layer, in MB."""
        return parse_kv_shadow(self.kv_shadow)

    def kv_shadow_global_mb(self, n_full_attn: int) -> float:
        """The process-wide total (per-layer size x layers), which is what the decode path's counter enforces."""
        return round(parse_kv_shadow(self.kv_shadow) * max(1, n_full_attn), 1)

    def kv_shadow_covers_tokens(self, kv_heads: int, head_dim: int, n_full_attn: int) -> int:
        """Context the resolved budget holds cached, for /health.

        Reported rather than merely used, because a budget that silently
        under-covers the configured ctx is the failure that made the old
        fixed 128 MB default wrong on every model with more than 8
        full-attention layers: the 9B covered 64k, the 27B only 32k, and
        nothing said so.
        """
        return kv_shadow_covers_tokens(self.kv_shadow_mb(n_full_attn), kv_heads, head_dim)


# ---------------------------------------------------------------------------
# argparse convenience -- optional; an interface may build GenParams by hand.
# ---------------------------------------------------------------------------

# The GenParams field each argparse dest maps to, and how to read it.
# `dest` is what add_generation_args registers; `field` is the GenParams
# kwarg; `sentinel` is the argparse default that means "not supplied by the
# user" (None) -- params_from_args substitutes the GenParams default for it
# so a hand-built Namespace missing the attribute and a supplied None are
# treated identically.
_ARG_FIELDS = (
    # (dest,            field,             maps None to)
    ("max_output_tokens", "max_output_tokens", DEFAULTS.MAX_OUTPUT_TOKENS),
    ("ctx_size",          "ctx_size",          None),   # None is meaningful here
    ("kv_cache",          "kv_cache",          DEFAULTS.KV_CACHE),
    ("kv_shadow",         "kv_shadow",         DEFAULTS.KV_SHADOW),
    ("kv_tau",            "kv_tau",            DEFAULTS.KV_TAU),
    ("kv_residual",       "kv_residual",       DEFAULTS.KV_RESIDUAL),
    ("prefill_chunk",     "prefill_chunk",     DEFAULTS.PREFILL_CHUNK),
    ("prefill_linear",    "prefill_linear",    "qvb8"),
    ("draft_k",           "draft_k",           None),   # None -> per-drafter default (GenParams)
    ("drafter",           "drafter",           DEFAULTS.DRAFTER),
    ("draft_ctx",         "draft_ctx",         DEFAULTS.DRAFT_CTX),
    ("draft_model",       "draft_model",       DEFAULTS.DRAFT_MODEL),
    ("draft_pack",        "draft_pack",        DEFAULTS.DRAFT_PACK),
    ("temperature",       "temperature",       DEFAULTS.TEMPERATURE),
    ("top_k",             "top_k",             DEFAULTS.TOP_K),
    ("top_p",             "top_p",             DEFAULTS.TOP_P),
    ("device",            "device",            "auto"),
    ("dtype",             "dtype",             None),   # None = the model's own dtype
    ("max_memory_gb",     "max_memory_gb",     DEFAULTS.MEMORY_LIMIT_GB),
)


def add_generation_args(ap, *, include_device_mem: bool = True) -> None:
    """Register the generation flags shared by every CLI interface, worded
    once. Pair with params_from_args() to get a GenParams back.

    `include_device_mem=False` skips --device / --max-memory-gb for an
    interface that resolves those elsewhere; params_from_args() then fills
    them from the GenParams defaults.
    """
    from ai_engine.runtime.memlimit import add_memory_limit_arg

    ap.add_argument(
        "--max-output-tokens", dest="max_output_tokens", type=int,
        default=None, metavar="N",
        help=f"tokens the model may generate in its reply (default "
             f"{DEFAULTS.MAX_OUTPUT_TOKENS}; also stops at EOS). This is the "
             f"OUTPUT limit -- --ctx-size is the separate PROMPT limit.")
    ap.add_argument(
        "--ctx-size", type=int, default=None, metavar="N",
        help="max PROMPT tokens accepted (llama.cpp --ctx-size); a longer "
             "prompt is rejected before any forward runs. Default: the "
             "loaded model's own context window (max_position_embeddings). "
             "Not a generation limit -- see --max-output-tokens.")
    ap.add_argument(
        "--kv-cache", choices=list(KV_CACHE_CHOICES), default=DEFAULTS.KV_CACHE,
        help="KV-cache method, one named preset (default: %(default)s). "
             "bf16 = dense, no compression. int8 = per-token per-head "
             "quantized (~1.0 B/elem, cheap decode). qvb8 = virtual-band "
             "(fixed-tau closed-form codec, ~0.24 B/elem) -- each segment "
             "decodes to dense on demand, cached per segment so an "
             "unchanged segment is only decoded once.")
    ap.add_argument(
        "--kv-shadow", dest="kv_shadow", default=DEFAULTS.KV_SHADOW, type=_kv_shadow_arg,
        metavar="SIZE",
        help="decoded K/V kept resident as a decode accelerator: a size PER full-attention "
             "layer, such as 512MB or 1GB (the startup line shows the layer count and the "
             "total: 16 layers on the 27B, 10 on the 35B-A3B). A cap, not a reservation: it "
             "fills only as the context grows, 4 KB per token per layer on the 27B, so 1MB "
             "covers 256 tokens. Past it decode re-expands the history every step (slower). "
             "0 keeps none. Not the compression ratio. (default: %(default)s)")
    ap.add_argument(
        "--kv-tau-threshold", dest="kv_tau", type=float, default=DEFAULTS.KV_TAU,
        metavar="TAU",
        help="Certified per-element KV error for the qvb8 preset (default "
             "%(default)s): every cached K/V value decodes within TAU. "
             "Smaller = more accurate, more bits; the codec raises bits per "
             "head until it holds. NOT the weight pack's tau.")
    ap.add_argument(
        "--kv-residual", dest="kv_residual", type=int,
        default=DEFAULTS.KV_RESIDUAL, metavar="N",
        help="tokens kept dense in the KV cache before a bulk flush -- "
             "shared by int8 AND the qvb8 preset (also the "
             "speculative-rollback headroom). Default %(default)s.")
    ap.add_argument(
        "--prefill-chunk", dest="prefill_chunk", type=int,
        default=DEFAULTS.PREFILL_CHUNK, metavar="N",
        help="tokens per chunked-prefill forward pass (default "
             "%(default)s, matches llama.cpp server's n_ubatch). A long "
             "prompt is fed through the model N tokens at a time so peak "
             "prefill VRAM is bounded by one chunk, not the whole prompt. "
             "Larger values raise prefill throughput at the cost of a "
             "higher peak-VRAM chunk; smaller values lower peak VRAM at "
             "the cost of more forward-pass overhead. (Previously "
             "QV_BAND_PREFILL_CHUNK env var only; still respected as the "
             "default when this flag is not passed.)")
    ap.add_argument(
        "--dtype", dest="dtype", choices=("bf16", "fp16"), default=None,
        help="compute dtype of the model (default: the model's own, from its config). fp16 for a GPU without bf16 "
             "(V100); an fp16+bf16 pack then needs its escape_tables.json. Never chosen automatically.")
    ap.add_argument(
        "--prefill-linear", dest="prefill_linear", choices=("qvb8", "int8"), default="qvb8",
        help="prefill linears (calls of >= 256 tokens): qvb8 = certified (default); int8 = opt-in int8 tensor cores on the "
             "pack's own qvb8 ids (weights exact) with per-token int8 activations and 2%% escape channels; decode is unchanged. "
             "Needs a one-off calibration: python -m ai_engine.generation.int8_prefill calibrate <pack>.")
    ap.add_argument(
        "--draft", dest="draft_k", type=int, default=None,
        metavar="N",
        help=f"speculative draft tokens proposed per step (default: a DFlash drafter's full block, block_size-1 "
             f"from its config -- 7 for the Qwen DFlash2 drafters, 15 for Ripple; {DEFAULTS.DRAFT_K} for the "
             f"other drafters; 0 disables drafting).")
    _dd = DEFAULTS.DRAFTER
    if os.environ.get("QV_USE_DSPARK"):
        _dd = "dspark"
    elif os.environ.get("QV_USE_MTP"):
        _dd = "mtp"
    ap.add_argument(
        "--drafter", choices=list(DRAFTER_CHOICES), default=_dd,
        help="speculative drafter, mutually exclusive (default: %(default)s, "
             "or QV_USE_DSPARK=1 / QV_USE_MTP=1). dflash: the model's own "
             "trained DFlash/DFlash 2 block drafter (--draft-model DIR is "
             "REQUIRED; a model with none falls back to n-gram). "
             "n-gram: prompt-lookup, "
             "near-zero cost. mtp: the checkpoint's MTP head, one pass per "
             "drafted token. dspark: MTP head as a two-pass whole-block draft "
             "(arXiv 2607.05147). n-gram_with_dflash: n-gram first, DFlash fills the rest of the block. n-gram_with_mtp: both in one draft -- n-gram "
             "fills repetition, MTP fills the rest of the budget.")
    ap.add_argument(
        "--draft-model", dest="draft_model", default=DEFAULTS.DRAFT_MODEL,
        metavar="DIR",
        help="DFlash draft model dir for --drafter dflash (REQUIRED -- there "
             "is no auto-discovery any more; a wrong path fails at startup "
             "instead of silently degrading to n-gram). Example: "
             "/mnt/backup/models/incoai/Qwen3.8-27B-DFlash2")
    ap.add_argument(
        "--draft-pack", dest="draft_pack", default=DEFAULTS.DRAFT_PACK,
        choices=list(DRAFT_PACK_CHOICES), metavar="MODE",
        help="which weights the DFlash drafter runs: dense (default) = "
             "the plain HF checkpoint; pack = its qvb8-quantised pack "
             "(smaller on VRAM but SLOWER on the 27B/9B, where the drafter "
             "runs its GEMMs at M=8..16); auto = use the pack when one "
             "exists for this checkpoint.")
    ap.add_argument(
        "--draft-ctx", dest="draft_ctx", type=int, default=DEFAULTS.DRAFT_CTX,
        metavar="N",
        help="n-gram / n-gram_with_mtp lookup window (default %(default)s).")
    if include_device_mem:
        ap.add_argument(
            "--device", default="auto", choices=list(DEVICE_CHOICES),
            help="compute device; auto picks CUDA, then XPU, then CPU.")
        add_memory_limit_arg(ap, default=DEFAULTS.MEMORY_LIMIT_GB)


def _dflash_block_size(draft_dir) -> int | None:
    """dflash_config.block_size from a DFlash checkpoint or its qvb8 pack (both carry config.json); None if unreadable."""
    try:
        import json
        with open(os.path.join(str(draft_dir), "config.json")) as f:
            return int(json.load(f).get("dflash_config", {}).get("block_size") or 0) or None
    except Exception:                                  # noqa: BLE001
        return None


def params_from_args(args) -> GenParams:
    """Build a GenParams from a parsed argparse.Namespace.

    For each field: use the attribute if present and not None, else the
    field's GenParams default. A None from argparse means "user did not
    supply this flag" and is treated the same as the attribute being
    absent -- so an interface that only registers a subset of the flags
    still gets a coherent GenParams. GenParams.__post_init__ validates the
    result.
    """
    kwargs: dict[str, Any] = {}
    for dest, field, none_maps_to in _ARG_FIELDS:
        val = getattr(args, dest, None)
        kwargs[field] = none_maps_to if val is None else val
    # GenParams requires an explicit DFlash checkpoint (no silent guessing inside the engine). A command-line entry point that was
    # NOT given --draft-model resolves it HERE, once, loudly: the model's own "<name>-DFlash2" / "-DFlash" next to the checkpoints
    # (the same rule llm_server.sh uses); a model without a drafter switches to n-gram with a printed warning. Without this
    # a launch without --draft-model died with a ValueError for every model.
    if kwargs.get("drafter") in ("dflash", "n-gram_with_dflash") and kwargs.get("draft_model") in (None, "", "auto"):
        model = getattr(args, "model", None)
        found = None
        if model:
            try:
                from qv_band.decode.dflash_draft import find_dflash_draft
                found = find_dflash_draft(model)
            except Exception:
                found = None
        if found:
            kwargs["draft_model"] = found
            print(f"[gen_core] DFlash drafter: {found}", flush=True)
        else:
            kwargs["drafter"] = "n-gram"
            print(f"[gen_core] WARNING: no DFlash drafter found for {model}; using n-gram "
                  f"(~40% slower; pass --draft-model DIR to use one)", flush=True)
    return GenParams(**kwargs)


# ---------------------------------------------------------------------------
# Model bundle: what load_model returns and every core call takes.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelBundle:
    model: Any
    tok: Any
    pack: str                    # pack folder path (== model_dir)
    gpu_dev: str | None          # "cuda", "cuda:1", "xpu", or None for CPU
    use_gpu: bool
    use_cuda: bool               # gpu_dev is a CUDA device (drafter builders care)
    load_seconds: float

    def empty_cache(self) -> None:
        """Trim the accelerator's caching allocator (a no-op on CPU). Call
        between requests on a long-lived server: torch's CUDA allocator
        keeps a pool sized to the largest recent allocation and does not
        return it to the driver on its own, so on this box's unified memory
        an oversized prompt's scratch keeps showing as used long after."""
        if not self.use_gpu:
            return
        import torch
        (torch.xpu if self.gpu_dev == "xpu" else torch.cuda).empty_cache()


def resolve_device(device: str) -> tuple[str | None, bool, bool]:
    """(gpu_dev, use_gpu, use_cuda) -- the tuple shape gen_core and its
    callers have always used.

    The logic itself moved to engine.runtime, which is also what the
    image, sonic, lipsync and TTS paths use, so one device value means one
    thing across every modality. See runtime.resolve_device for the accepted
    values ("auto" / "all" / "cuda" / "cuda:<int>" / "xpu" / "cpu").
    """
    from ai_engine.runtime.device import resolve_device as _resolve
    plan = _resolve(device)
    return plan.gpu_dev, plan.use_gpu, plan.use_cuda


def load_model(model_dir: str, params: GenParams, *, verbose: bool = True
               ) -> ModelBundle:
    """Resolve the device, apply the CPU thread guard and the memory
    ceiling, load the packed model, fill params.ctx_size from the model
    config if the user did not set it, and return a ModelBundle. Identical
    for every interface; the only per-interface variation (what to print)
    is gated on `verbose`.

    Mutates `params` in place (ctx_size). A GenParams is per-run config,
    not shared state, so this is intentional -- but do not reuse one
    GenParams across models with different context windows.
    """
    import torch
    from ai_engine.runtime.memlimit import apply_unified_memory_limit

    pack = model_dir = model_dir.rstrip("/")
    if not os.path.isdir(pack):
        raise FileNotFoundError(f"pack folder does not exist: {pack}")
    # No manifest.json: a plain Hugging Face checkpoint (bf16, or the official FP8 one, whose
    # weights are dequantised while streaming) served by the same engine, so the dense references
    # and the packs are generated by one code path.
    dense = not os.path.isfile(os.path.join(pack, "manifest.json"))
    if dense and not os.path.isfile(os.path.join(pack, "config.json")):
        raise FileNotFoundError(
            f"no manifest.json or config.json in {pack} (neither a QV-band pack nor a checkpoint folder)")

    gpu_dev, use_gpu, use_cuda = resolve_device(params.device)
    if use_cuda:
        # One-launch RMSNorm / gated RMSNorm / SwiGLU in place of the eager op chains (about a
        # third of a prefill chunk); patches a transformers class only if its source is exactly
        # the one the kernels reproduce.
        from qv_band.decode.fused_pointwise import install as _install_fused_pointwise
        _install_fused_pointwise()

    if not use_gpu:
        # qv_anchored_cpu's own thread pool (kernel_cpu_c.py, sized to
        # os.cpu_count()) already saturates every core decoding the packed
        # weights; torch's default intra-op pool is the same size and only
        # contends with it for the small slice of ops still on torch.
        # Measured 1.62 -> 1.70 tok/s forcing torch down to 1. An explicit
        # params.cpu_threads (a positive count) overrides that default.
        torch.set_num_threads(params.cpu_threads if params.cpu_threads
                              and params.cpu_threads > 0 else 1)

    # Before the model loads, so the ceiling covers the weights too. Which
    # guard this becomes (host RLIMIT_AS, CUDA VRAM poll, or both) depends
    # on the resolved device -- see memlimit.py.
    apply_unified_memory_limit(params.max_memory_gb, params.device,
                               verbose=verbose)

    t0 = time.time()
    if use_gpu:
        from qv_band.run.run_inference import load_qv_model
        model, tok = load_qv_model(model_dir, None if dense else pack, device=gpu_dev, dtype=params.dtype)
        # Packs are calibrated for fp16 AND bf16 first (qv_band tech_doc/restore_precision_policy.md);
        # the dtype that runs is the user's --dtype (fp16 on a V100).
        # With bf16 activations, prefill linears (M > 8) of a bf16-certified pack take the bf16 GEMM.
        # Decode (M <= 8) is the GEMV kernels either way.
        from qv_band.formats.qvb8.module import enable_bf16_gemm
        n_bf16, n_lin = enable_bf16_gemm(model)
        if verbose and n_lin:
            if params.dtype == "fp16":
                print(f"[load] --dtype fp16: the fp16 path runs; the bf16 GEMM ({n_bf16}/{n_lin} linears are bf16-certified) "
                      f"is not used", flush=True)
            else:
                print(f"[load] bf16 GEMM on {n_bf16}/{n_lin} QVB8 linears", flush=True)
        if params.prefill_linear == "int8" and not dense:
            from ai_engine.generation.int8_prefill import install as _install_int8_prefill
            _install_int8_prefill(model, pack, verbose=verbose)
    else:
        from qv_band.run.run_inference_cpu import load_qv_model_cpu
        model, tok = load_qv_model_cpu(model_dir, pack, verbose=verbose)
    load_seconds = time.time() - t0

    params.resolve_ctx_size(model)

    return ModelBundle(model=model, tok=tok, pack=pack, gpu_dev=gpu_dev,
                       use_gpu=use_gpu, use_cuda=use_cuda,
                       load_seconds=load_seconds)


# ---------------------------------------------------------------------------
# Drafters: the ONE 4-way x CUDA/CPU switch.
# ---------------------------------------------------------------------------


class DrafterBuildError(RuntimeError):
    """A heavy drafter (mtp/dspark/n-gram_with_mtp) failed to build. Raised
    unless build_drafter(fallback_to_ngram=True), in which case the caller
    gets a plain n-gram drafter and this is logged instead of raised."""


def build_drafter(kind: str, bundle: ModelBundle, *,
                  max_ctx: int = DEFAULTS.DRAFT_CTX,
                  fallback_to_ngram: bool = False,
                  on_fallback: Callable[[str, BaseException], None] | None = None,
                  params: "GenParams | None" = None):
    """Build the drafter object for `kind` ("dflash" | "n-gram" | "mtp" |
    "dspark" | "n-gram_with_mtp"), or None when kind is falsy or draft is
    disabled.

    "dflash" reads `params.draft_model` -- the DFlash checkpoint directory,
    which is REQUIRED (no auto-discovery): a missing or wrong path raises
    DrafterBuildError rather than silently degrading to n-gram.
    The CUDA vs CPU builder split is handled here so no interface repeats
    it.

    `fallback_to_ngram`: when a heavy drafter fails to build, return a
    plain n-gram drafter instead of raising -- so a missing MTP head
    degrades a running server (llm_server.py) rather than killing it. False fails
    loudly at startup. The failure is never
    swallowed silently: `on_fallback(kind, exc)` is invoked when set (for
    the caller to log), otherwise the exception message is printed.

    An n-gram build failure is always raised -- there is nothing to fall
    back to, and it means the environment itself is broken.
    """
    if not kind:
        return None
    if kind not in DRAFTER_CHOICES:
        raise ValueError(
            f"drafter must be one of {DRAFTER_CHOICES}, got {kind!r}")

    model, pack, model_dir = bundle.model, bundle.pack, bundle.pack
    use_cuda, gpu_dev = bundle.use_cuda, bundle.gpu_dev

    def _ngram():
        from qv_band.decode.ngram_draft import NGramDrafter
        return NGramDrafter(max_ctx=max_ctx)

    if kind == "n-gram":
        return _ngram()

    if kind in ("dflash", "n-gram_with_dflash"):
        from qv_band.decode.dflash_draft import (build_dflash_drafter,
                                                 find_dflash_pack_for)
        draft_dir = getattr(params, "draft_model", DEFAULTS.DRAFT_MODEL)
        fallback = "n-gram"
        # The drafter checkpoint is EXPLICIT: no auto-discovery, no guessing.
        # A wrong path must fail loudly at startup rather than silently
        # degrade to n-gram (which costs ~40% throughput and looks like
        # "the server got slow" instead of "the drafter is missing").
        if draft_dir in (None, "", "auto"):
            raise DrafterBuildError(
                "--drafter dflash needs --draft-model DIR (the DFlash "
                "checkpoint to load); 'auto' is not accepted any more. "
                "llm_server.sh fills it in automatically.")
        draft_dir = os.path.abspath(os.path.expanduser(str(draft_dir)))
        if not os.path.isdir(draft_dir):
            # A mistyped path is the usual cause ("...-DFlash" for the
            # published "...-DFlash2"). Name the checkpoint that IS on disk
            # instead of only saying "not a directory" -- the caller can
            # then paste the right path.
            hint = ""
            try:
                from qv_band.decode.dflash_draft import suggest_dflash_draft
                alt = suggest_dflash_draft(draft_dir)
                if alt:
                    hint = (f"; did you mean {alt!r}? "
                            f"(pass --draft-model {alt})")
            except Exception:
                hint = ""
            raise DrafterBuildError(
                f"--draft-model {draft_dir!r} is not a directory{hint}")
        # CUDA only (CPU/XPU have never run DFlash).
        unsupported = None if use_cuda else "needs a CUDA launch"
        if unsupported:
            print(f"[gen_core] DFlash drafter {draft_dir} {unsupported}; "
                  f"using {fallback}", flush=True)
            return build_drafter(fallback, bundle, max_ctx=max_ctx,
                                 fallback_to_ngram=fallback_to_ngram,
                                 on_fallback=on_fallback, params=params)
        # A qvb8 pack of the drafter cuts its weights to ~52-62%, but the
        # drafter only ever proposes a BLOCK of 8-16 tokens, so it runs its
        # GEMMs at M=8..16 where the packed path's overhead dominates its
        # weight saving -- on the 27B/9B the packed drafter is SLOWER end to
        # end than the dense one. So the pack is opt-in (--draft-pack), not
        # automatic: dense is the default (see DEFAULTS.DRAFT_PACK).
        draft_pack_mode = getattr(params, "draft_pack",
                                 DEFAULTS.DRAFT_PACK)
        if draft_pack_mode in ("pack", "auto"):
            # The pack is looked up from the EXPLICIT drafter path, not from
            # the target's name: the caller said which drafter to load, so
            # the packed copy of THAT drafter is the one to use. (For a dense
            # drafter dir, target_name() yields its basename, e.g.
            # "Qwen3.8-27B-DFlash2", and the pack is published as
            # "<that>_qvb8_*".)
            draft_pack = find_dflash_pack_for(draft_dir)
        else:                                   # "dense"
            draft_pack = None
        try:
            d = build_dflash_drafter(model, draft_dir, device=gpu_dev or "cpu",
                                     pack_dir=draft_pack)
            if kind == "n-gram_with_dflash":
                from qv_band.decode.dflash_draft import NGramWithDFlashDrafter
                from qv_band.decode.ngram_draft import NGramDrafter
                d = NGramWithDFlashDrafter(NGramDrafter(max_ctx=max_ctx), d)
                print(f"n-gram_with_dflash: n-gram window {max_ctx} + DFlash tail", flush=True)
            return d
        except Exception as exc:
            if not fallback_to_ngram:
                raise DrafterBuildError(
                    f"dflash drafter failed to build: {exc!r}") from exc
            if on_fallback is not None:
                on_fallback(kind, exc)
            else:
                print(f"[gen_core] dflash drafter unavailable ({exc!r}); "
                      f"using {fallback}", flush=True)
            return build_drafter(fallback, bundle, max_ctx=max_ctx,
                                 fallback_to_ngram=True,
                                 on_fallback=on_fallback, params=params)

    try:
        if kind == "n-gram_with_mtp":
            if use_cuda:
                from qv_band.decode.spec_decode import build_n_gram_with_mtp_drafter_cuda
                return build_n_gram_with_mtp_drafter_cuda(
                    model, pack, model_dir, max_ctx=max_ctx, device=gpu_dev)
            from qv_band.decode.spec_decode import build_n_gram_with_mtp_drafter
            return build_n_gram_with_mtp_drafter(
                model, pack, model_dir, max_ctx=max_ctx)
        if kind == "dspark":
            if use_cuda:
                from qv_band.decode.dspark_draft import build_dspark_drafter_cuda
                return build_dspark_drafter_cuda(model, pack, model_dir,
                                                 device=gpu_dev)
            from qv_band.decode.dspark_draft import build_dspark_drafter
            return build_dspark_drafter(model, pack, model_dir)
        # kind == "mtp"
        if use_cuda:
            from qv_band.decode.spec_decode import build_drafter_cuda
            return build_drafter_cuda(model, pack, model_dir, device=gpu_dev)
        # NOTE: alias the import -- importing it under the name
        # "build_drafter" would shadow THIS function inside its own body and
        # make the recursive n-gram fallback below raise UnboundLocalError.
        from qv_band.decode.spec_decode import build_drafter as build_mtp_drafter
        return build_mtp_drafter(model, pack, model_dir)
    except Exception as exc:
        if not fallback_to_ngram:
            raise DrafterBuildError(
                f"{kind} drafter failed to build: {exc!r}") from exc
        if on_fallback is not None:
            on_fallback(kind, exc)
        else:
            print(f"[gen_core] {kind} drafter unavailable ({exc!r}); "
                  f"falling back to n-gram", flush=True)
        return _ngram()


# ---------------------------------------------------------------------------
# The single generation call.
# ---------------------------------------------------------------------------


def encode_prompt(bundle: ModelBundle, text: str, *, chat: bool,
                  think: bool, history: list | None = None):
    """text -> input_ids on the model's device. `chat` applies the instruct
    template (with `history` prepended and `think` toggling the reasoning
    block); otherwise the text is encoded raw for plain continuation."""
    tok = bundle.tok
    if chat:
        msgs = (history or []) + [{"role": "user", "content": text}]
        s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=think)
    else:
        s = text
    ids = tok(s, return_tensors="pt").input_ids
    if bundle.use_gpu:
        ids = ids.to(bundle.gpu_dev)
    return ids


def run_generate(bundle: ModelBundle, drafter, ids, params: GenParams, *,
                 stream: bool = False,
                 abort_check: Callable[[], bool] | None = None,
                 kv_stats: dict | None = None,
                 skip_special_tokens: bool = True,
                 prefix_cache=None,
                 return_cache: bool = False,
                 on_heartbeat: Callable[[], None] | None = None):
    """The one speculative_generate() call. Every KV-cache mode and every
    drafter reaches the model through here; the KV-cache preset -> kwargs mapping
    is GenParams.to_speculative_kwargs() and lives nowhere else.

    `abort_check`: a zero-arg predicate polled once per decode step; True
    stops generation cleanly. None (the default) uses the --max-memory-gb
    VRAM-ceiling poll.

    `prefix_cache` / `return_cache`: single-conversation KV-cache reuse
    (see qv_band.decode.spec_decode.PrefixCache) -- pass back a cache saved from a
    prior call (return_cache=True there) to skip re-prefilling whatever
    prompt text is identical, exact-prefix only (see PrefixCache's
    docstring for why a partial/cropped match is never attempted: the
    linear-attention layers' recurrent state cannot be rolled back). A
    stale or mismatched cache is silently ignored, never an error.

    Returns speculative_generate's own (out_ids, stats) tuple, or
    (out_ids, stats, PrefixCache | None) when return_cache=True.
    speculative_generate is already @torch.no_grad(), so no grad guard is
    added here.
    """
    from qv_band.decode.spec_decode import speculative_generate
    _adapter = getattr(bundle.model, "_qv_adapter", None)
    _own_loop = _adapter.generate_fn() if _adapter is not None else None
    if _own_loop is not None:                 # a model whose stack needs another loop kind (decode/hybrid_generate)
        speculative_generate = _own_loop

    if abort_check is None:
        from ai_engine.runtime.memlimit import vram_abort_check
        abort_check = vram_abort_check

    return speculative_generate(
        bundle.model, drafter, bundle.tok, ids,
        stream=stream, kv_stats=kv_stats, abort_check=abort_check,
        skip_special_tokens=skip_special_tokens,
        prefix_cache=prefix_cache, return_cache=return_cache,
        on_heartbeat=on_heartbeat,
        **params.to_speculative_kwargs())
