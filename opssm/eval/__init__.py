"""Model-agnostic evaluation core for opssm and its baselines.

Any fitted model (opssm, EKF, SING, rSLDS, visde, latent-SDE) is reduced to a `Result`
(z_hat, y_hat, optional drift) and scored by the SAME metrics/figures, so comparisons are fair.
"""
from opssm.eval.core import (
    Result, EvalContext, gauge_aligned, decode, recon_metrics, data_flow,
)

__all__ = ["Result", "EvalContext", "gauge_aligned", "decode", "recon_metrics", "data_flow"]
