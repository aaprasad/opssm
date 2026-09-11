"""Temporal leakage, absent ground truth, and frozen behavior-probe checks."""
from dataclasses import replace
import json

import numpy as np
import pytest

from opssm.benchmarks.data import config_from_data, drift, fingerprint
from opssm.benchmarks.kato import KatoConfig, prepare_recording
from opssm.benchmarks.decoding import score_linear_decoding
from opssm.benchmarks.metrics import score_benchmark


def recording():
    rng = np.random.default_rng(42)
    return dict(traces=rng.normal(size=(600, 4)), states=np.arange(600) % 3,
                dt=.34, fps=1/.34, neuron_ids=["A", "B", "", "D"],
                state_names=["FWD", "REV", "NOSTATE"], name="WT_NoStim_worm0")


def test_kato_split_before_window_and_train_only_standardization():
    rec = recording()
    cfg = KatoConfig("fixture.mat", latent_dim=2, window=20, train_stride=10, gap_frames=5)
    resolved, data = prepare_recording(rec, cfg)
    changed = {**rec, "traces": rec["traces"].copy()}
    changed["traces"][360:] += 100
    _, other = prepare_recording(changed, cfg)
    for key in ("y_train", "obs_mean", "obs_scale"):
        np.testing.assert_array_equal(data[key], other[key])
    frames = {s: np.unique(data[f"frame_indices_{s}"]) for s in ("train", "val", "test")}
    assert frames["train"].max() + cfg.gap_frames < frames["val"].min()
    assert frames["val"].max() + cfg.gap_frames < frames["test"].min()
    for split in ("val", "test"):
        assert len(frames[split]) == data[f"frame_indices_{split}"].size
    np.testing.assert_allclose(data["ts"], np.arange(cfg.window) * rec["dt"])
    assert config_from_data(data) == resolved
    assert not any(k.startswith(("z_", "signal_")) for k in data)
    assert fingerprint(data) == fingerprint(prepare_recording(rec, cfg)[1])
    with pytest.raises(ValueError, match="No ground-truth"):
        drift(np.ones((3, 2)), resolved)
    with pytest.raises(ValueError, match="too short"):
        prepare_recording(rec, replace(cfg, window=200))


def separable_probe():
    rng = np.random.default_rng(4)
    data = {"metadata": np.array(json.dumps(dict(state_names=["FWD", "REV", "NOSTATE"])))}
    means = {}
    for i, s in enumerate(("train", "val", "test")):
        labels = np.tile([0, 1, 2], 30)[:, None]
        means[s] = (labels * 8 + rng.normal(0, .1, labels.shape))[..., None].astype(float)
        data[f"states_{s}"] = labels
        data[f"frame_indices_{s}"] = (np.arange(90) + i * 100)[:, None]
    return means, data


def test_linear_decoder_is_frozen_and_deduplicates_frames():
    pytest.importorskip("sklearn")
    means, data = separable_probe()
    metrics, arrays = score_linear_decoding(means, data)
    assert metrics["linear_decode_accuracy"] == 1
    assert metrics["linear_decode_balanced_accuracy"] == 1
    assert metrics["linear_decode_n_train"] == 60  # Ambiguous labels excluded.
    assert metrics["linear_decode_C"] == .001  # Ties choose stronger regularization.
    changed = dict(data)
    changed["states_test"] = np.where(data["states_test"] < 2, 1 - data["states_test"], 2)
    wrong, same = score_linear_decoding(means, changed)
    assert wrong["linear_decode_accuracy"] == 0
    for k in ("linear_decode_coef", "linear_decode_intercept", "linear_decode_feature_mean", "linear_decode_prediction"):
        np.testing.assert_array_equal(arrays[k], same[k])
    repeated = {**means, "train": np.repeat(means["train"], 2, axis=1)}
    for k in ("states_train", "frame_indices_train"):
        changed[k] = np.repeat(data[k], 2, axis=1)
    duplicate_metrics, duplicate_arrays = score_linear_decoding(repeated, changed)
    assert duplicate_metrics["linear_decode_n_train"] == 60
    np.testing.assert_allclose(arrays["linear_decode_coef"], duplicate_arrays["linear_decode_coef"])


def test_decoder_rejects_temporal_overlap_and_handles_missing_classes():
    pytest.importorskip("sklearn")
    means, data = separable_probe()
    with pytest.raises(ValueError, match="share recording frames"):
        score_linear_decoding(means, {**data, "frame_indices_test": data["frame_indices_train"]})
    metrics, arrays = score_linear_decoding(means, {**data, "states_train": np.zeros_like(data["states_train"])})
    assert metrics["linear_decode_status"] == "insufficient_training_classes"
    assert metrics["linear_decode_accuracy"] is None and not arrays
    changed = {**data, "states_test": np.full_like(data["states_test"], 3)}
    metrics, _ = score_linear_decoding(means, changed)
    assert metrics["linear_decode_unseen_test_fraction"] == 1
    assert metrics["linear_decode_accuracy"] == 0


def test_real_data_scoring_does_not_invent_latents_or_drift():
    pytest.importorskip("sklearn")
    cfg, data = prepare_recording(recording(), KatoConfig("fixture.mat", latent_dim=2, window=20, gap_frames=5, train_stride=10))
    val = dict(mean=data["y_val"][..., :2])
    test = dict(mean=data["y_test"][..., :2], reconstruction=data["y_test"])
    cut = len(data["ts"]) // 2
    ahead = dict(forecast_mean=data["y_test"][cut:] + 2, forecast_loglik=np.zeros(data["y_test"][cut:].shape[:2]))
    def forbidden(_):
        raise AssertionError("Cannot evaluate biological ground-truth drift")
    metrics, arrays = score_benchmark(val, test, ahead, data, cfg, forbidden, train_mean=data["y_train"][..., :2])
    assert metrics["observation_recon_rmse"] == 0
    assert metrics["forecast_observation_rmse"] == 2
    assert metrics["dynamics_rmse"] is None and metrics["latent_rmse"] is None
    assert "alignment" not in arrays and "forecast_clean_rmse" not in metrics
    assert metrics["linear_decode_status"] == "ok"
