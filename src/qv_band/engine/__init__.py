"""Model-agnostic engine layer: the adapter protocol and registry (adapter.py).

Layering (enforced by tests/test_layering.py):
    models/<name>  ->  engine  ->  core (formats/, common/, kv/)
Nothing here imports qv_band.models; a model plugs in by registering a ModelAdapter.
"""
from qv_band.engine.adapter import (  # noqa: F401
    Drafter, ModelAdapter, ServeDefaults, Target, find_adapter, register, registered,
)
