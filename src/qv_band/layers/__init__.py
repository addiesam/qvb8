"""Layer kinds of the framework, one module per kind (named by what the layer does, never by a model).

  sliding_attention   gated attention with value residual; sliding-window (RoPE) and global (NoPE) layers
  fused_pointwise     Triton kernels: clamped-SiLU MLP activation, RMSNorm, RoPE, sigmoid gate
  mamba2_hybrid       Mamba-2 / attention / latent-MoE hybrid stack: skeleton, forwards, CUDA graphs, rollback
  cradio_vision       C-RADIO vision tower + pixel-shuffle projector

A model adapter (qv_band/models/<name>/adapter.py) chooses among these; nothing here imports qv_band.models.
"""
