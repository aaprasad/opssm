"""Baseline adapters: each fits a model on an EvalContext's FIT split and returns a Result the eval-core
can score. `get_adapter(name)` -> adapter instance."""
from opssm.eval.baselines.opssm_adapter import OpssmAdapter

_REGISTRY = {"opssm": OpssmAdapter}


def get_adapter(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
