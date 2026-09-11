"""Model-neutral evaluation. Affine gauge fitted on validation, frozen on test."""
import numpy as np


def fit_alignment(z_hat, z_true):
    """Least-squares row-vector map z_true ~= z_hat @ A + b, fitted on validation."""
    x = z_hat.reshape(-1, z_hat.shape[-1])
    target = z_true.reshape(-1, z_true.shape[-1])
    return np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), target, rcond=None)[0]


def score_dynamics(learned_drift, true_drift, data, alignment, kind="continuous_drift"):
    """Compare vector fields at identical GT test states, in physical time units.

    For the row-vector alignment x=z@A+b, f_x(x)=f_z((x-b)@inv(A))@A.
    Translation changes the query location but must NOT be added to velocities.
    Inverting the frozen validation map isolates drift error from state-estimation
    error. A collapsed map cannot define this comparison; no pseudoinverse is used.
    """
    A, b = np.asarray(alignment[:-1], dtype=float), np.asarray(alignment[-1], dtype=float)
    target = np.asarray(data["z_test"], dtype=float)
    cond = float(np.linalg.cond(A))
    metrics = dict(dynamics_kind=kind, dynamics_rmse=None, dynamics_nrmse=None,
                   alignment_kind="affine_validation", alignment_condition=cond if np.isfinite(cond) else None)
    if A.shape[0] != A.shape[1] or not np.isfinite(cond) or cond > 1e10:
        metrics["dynamics_status"] = "noninvertible_or_ill_conditioned_alignment"
        return metrics, {}
    query = np.linalg.solve(A.T, (target - b).reshape(-1, target.shape[-1]).T).T.reshape(target.shape)
    estimate = np.asarray(learned_drift(query), dtype=float)
    reference = np.asarray(true_drift(target), dtype=float)
    if estimate.shape != query.shape or reference.shape != target.shape:
        raise ValueError("Dynamics must cover every test trajectory, time, and latent coordinate")
    aligned = estimate @ A
    if not np.isfinite(aligned).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Nonfinite dynamics evaluation")
    rmse = float(np.sqrt(np.mean((aligned - reference) ** 2)))
    scale = max(float(np.sqrt(np.mean(np.asarray(true_drift(data["z_train"]), dtype=float) ** 2))), 1e-12)
    metrics.update(dynamics_rmse=rmse, dynamics_nrmse=rmse / scale, dynamics_status="ok")
    arrays = dict(dynamics_query_model=query, dynamics_pred_aligned=aligned, dynamics_true=reference)
    return metrics, arrays


def score_posterior(result, data, alignment):
    z, cov, y = (np.asarray(result[k]) for k in ("mean", "cov", "reconstruction"))
    truth = data["z_test"]
    if z.shape[:2] != truth.shape[:2] or y.shape != data["signal_test"].shape:
        raise ValueError("Every method must score every test trajectory and time")
    A, b = alignment[:-1], alignment[-1]
    aligned = z @ A + b
    error = aligned - truth
    scale = data["z_train"].std((0, 1)).clip(1e-8)
    var = np.diagonal(A.T @ cov @ A, axis1=-2, axis2=-1).clip(1e-10)
    signal = data["signal_test"]
    return dict(latent_rmse=float(np.sqrt(np.mean(error ** 2))),
                latent_nrmse=float(np.sqrt(np.mean((error / scale) ** 2))),
                clean_recon_rmse=float(np.sqrt(np.mean((y - signal) ** 2))),
                clean_recon_r2=float(1 - np.sum((y - signal) ** 2) /
                                     max(np.sum((signal - signal.mean((0, 1))) ** 2), 1e-12)),
                marginal_95_coverage=float(np.mean(np.abs(error) <= 1.959963984540054 * np.sqrt(var))))


def score_benchmark(val, test, ahead, data, cfg, learned_drift, kind="continuous_drift", train_mean=None):
    """Shared scoring for simulated truth and real observations, without invented GT."""
    from .data import drift
    arrays = {}
    cut = len(data["ts"]) // 2
    if "z_test" in data:
        alignment = fit_alignment(val["mean"], data["z_val"])
        metrics = score_posterior(test, data, alignment)
        dynamics, fields = score_dynamics(learned_drift, lambda z: drift(z, cfg), data, alignment, kind)
        metrics.update(dynamics)
        arrays.update(fields, alignment=alignment)
        metrics["forecast_clean_rmse"] = float(np.sqrt(np.mean((ahead["forecast_mean"] - data["signal_test"][cut:]) ** 2)))
    else:
        obs = data["y_test"]
        error = np.asarray(test["reconstruction"]) - obs
        if error.shape != obs.shape or ahead["forecast_mean"].shape != obs[cut:].shape:
            raise ValueError("Every real-data test observation must be scored")
        metrics = dict(latent_rmse=None, latent_nrmse=None, dynamics_rmse=None, dynamics_nrmse=None,
                       dynamics_status="unavailable_no_ground_truth", alignment_kind="unavailable_no_ground_truth",
                       observation_recon_rmse=float(np.sqrt(np.mean(error ** 2))),
                       observation_recon_r2=float(1 - np.sum(error ** 2) /
                           max(np.sum((obs - obs.mean((0, 1))) ** 2), 1e-12)),
                       observation_recon_rmse_physical=float(np.sqrt(np.mean((error * data["obs_scale"]) ** 2))))
        forecast_error = ahead["forecast_mean"] - obs[cut:]
        metrics.update(forecast_observation_rmse=float(np.sqrt(np.mean(forecast_error ** 2))),
                       forecast_observation_rmse_physical=float(np.sqrt(np.mean((forecast_error * data["obs_scale"]) ** 2))),
                       forecast_persistence_rmse=float(np.sqrt(np.mean((obs[cut - 1] - obs[cut:]) ** 2))))
    if "states_train" in data:
        from .decoding import score_linear_decoding
        if train_mean is None:
            raise ValueError("Behavior decoding requires latents from the frozen model on training data")
        decoding, probe = score_linear_decoding(dict(train=train_mean, val=val["mean"], test=test["mean"]), data)
        metrics.update(decoding)
        arrays.update(probe)
    metrics["forecast_marginal_nll"] = float(-ahead["forecast_loglik"].mean() / cfg.obs_dim)
    return metrics, arrays


def aggregate(rows):
    """Across-seed mean and sample standard deviation, separated by information set."""
    groups = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["dataset"], row["model"], row["posterior_type"])
        groups.setdefault(key, []).append(row)
    out = []
    for (dataset, model, posterior), group in sorted(groups.items()):
        entry = dict(dataset=dataset, model=model, posterior_type=posterior, n_seeds=len(group))
        for key, value in group[0].items():
            if key in ("seed", "steps", "best_step") or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values = [g[key] for g in group if isinstance(g.get(key), (int, float))]
            entry[key + "_mean"] = float(np.mean(values))
            entry[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        out.append(entry)
    return out
