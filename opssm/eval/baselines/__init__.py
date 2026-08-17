"""Baseline adapters: each fits a model on an EvalContext's FIT split and returns a Result the eval-core
can score. `get_adapter(name)` -> adapter instance.

- opssm            : in-process (neuraloperator venv), loads a trained checkpoint.
- latent_sde       : torchsde sim-based latent SDE, subprocess in ~/venvs/baselines.
- visde            : Course & Nair sim-free latent SDE, subprocess in ~/venvs/baselines (Phase 2b).
- ekf / rslds / sing: JAX, subprocess in ~/venvs/jaxbaselines (Phase 3).
"""
import os

from opssm.eval.baselines.opssm_adapter import OpssmAdapter
from opssm.eval.baselines.subprocess_adapter import SubprocessAdapter

_WORKERS = os.path.join(os.path.dirname(__file__), "_workers")
_TORCH_PY = "~/venvs/baselines/bin/python"
_JAX_PY = "~/venvs/jaxbaselines/bin/python"

# name -> (venv python, worker file, posterior tag)  for subprocess baselines
_SUBPROC = {
    "latent_sde": (_TORCH_PY, "latent_sde_worker.py", "smoother"),
    "visde":      (_TORCH_PY, "visde_worker.py", "smoother"),
    "ekf":        (_JAX_PY, "ekf_worker.py", "filter"),
    "rslds":      (_JAX_PY, "rslds_worker.py", "smoother"),
    "sing":       (_JAX_PY, "sing_worker.py", "smoother"),
}


def get_adapter(name):
    if name == "opssm":
        return OpssmAdapter()
    if name in _SUBPROC:
        py, worker, post = _SUBPROC[name]
        return SubprocessAdapter(name, py, os.path.join(_WORKERS, worker), post)
    raise KeyError(f"unknown model {name!r}; have {['opssm', *_SUBPROC]}")
