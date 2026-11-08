"""Run Ripple-65B (QVB8 pack) with its DFlash2 drafter. Ripple-only entry point: the Qwen / generic
run_inference.py and dflash_draft.py are NOT modified. This wrapper
  * defaults QV_ATTN_IMPL=sdpa (the shipped RicursiveForCausalLM refuses flash-attn),
  * routes drafter construction to RippleDFlashDrafter when the target is a `ricursive` model,
then hands over to run_inference.main() unchanged.

    python -m qv_band.models.ripple.run --model PACK --pack-dir PACK --draft-model SRC --draft-pack DPACK --spec-k 15
"""
import os

# Forced, not defaulted: the shipped RicursiveForCausalLM refuses flash-attn at construction, and launchers export
# QV_ATTN_IMPL=flash_attention_2 when it is available. runtime.install() then switches the attention (now ours)
# to the flash-attn registry entry after loading.
os.environ["QV_ATTN_IMPL"] = "sdpa"

from qv_band.decode import dflash_draft as _dd

_orig_build = _dd.build_dflash_drafter


def _build(model, draft_dir, device="cuda", pack_dir=None):
    if getattr(model.config, "model_type", "") == "ricursive":
        from qv_band.models.ripple.drafter import build_ripple_drafter
        return build_ripple_drafter(model, draft_dir, device=device, pack_dir=pack_dir)
    return _orig_build(model, draft_dir, device=device, pack_dir=pack_dir)


_dd.build_dflash_drafter = _build

from qv_band.run import run_inference as _ri

_orig_load = _ri._load_qv_model_impl


def _load(args):
    model, tok = _orig_load(args)
    if getattr(model.config, "model_type", "") == "ricursive":
        from qv_band.models.ripple.runtime import install
        install(model)
    return model, tok


_ri._load_qv_model_impl = _load

if __name__ == "__main__":
    _ri.main()
