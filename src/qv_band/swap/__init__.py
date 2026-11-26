"""Put a pack into an already-loaded model: replace its dense
weights with packed ones.

    replace_diffusers.py  273  nn.Linear in a diffusers pipeline -- image_gen.
                              Genuinely codec-agnostic: dispatches through
                              qv_band.common.formats.get_format(), so the
                              same code path serves every registered pack format.
    load_embedding.py     142  embedding + reranker HF models. Has no codec
                              logic of its own -- reuses replace_diffusers'
                              replace_qv_band() directly.

replace_unet.py (nn.Conv2d/Conv3d/Linear -- Sonic, any .pth) moved to
qv_band.formats.qvb8.replace_unet: unlike the two files
above, it is hardcoded to QVB8 only (function name replace_unet_qvb8,
imports qv_band.formats.qvb8.module_conv directly, no format dispatch),
so it belongs with qvb8's other inference-time code, not here.

These are what image_gen.py and sonic_qvb8.py call into. They do not load a
model; they operate on one the caller already has.
"""
