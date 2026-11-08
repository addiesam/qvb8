"""The synthetic manifest key for one slice of a routed-MoE expert stack.

A 3-D [E, out, in] expert stack is packed and addressed one [out, in] slice
at a time; the manifest names each slice "<base_tensor>#expert<e:03d>"
(e.g. "model.language_model.layers.0.mlp.experts.gate_up_proj#expert003").
The calibration driver that writes these keys (calibration/pack_model.py) and the
inference-side loader that parses them back (run/run_inference.py) need the
EXACT same separator, so it is defined once here rather than as three
independently-duplicated literals.

Importing this from run_inference.py does NOT create a calibration
dependency at inference time: this is a plain string constant with no
calibration logic attached, so a deployed server importing it never pulls
in any pack-time code.
"""
from __future__ import annotations

EXPERT_KEY_SEP = "#expert"
