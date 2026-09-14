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
    cfg = KatoConfig("fixture.mat", latent_dim=2, window=20, train_stride=10, gap_frames=5, folds=1)
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


def test_rotating_folds_cover_the_trace_without_leakage():
    """Every fold tests a different region, and no held-out frame sits within gap of training."""
    rec = recording()
    base = dict(latent_dim=2, window=20, train_stride=10, gap_frames=5, folds=5)
    tested = []
    for fold in range(base["folds"]):
        cfg = KatoConfig("fixture.mat", fold=fold, **base)
        resolved, data = prepare_recording(rec, cfg)
        frames = {s: np.unique(data[f"frame_indices_{s}"]) for s in ("train", "val", "test")}
        train = set(frames["train"].tolist())
        for split in ("val", "test"):
            held = set(frames[split].tolist())
            assert not (train & held), f"fold {fold}: {split} overlaps training"
            # the gap must actually separate them in time, not merely make them disjoint
            assert min(abs(h - t) for h in held for t in train) > cfg.gap_frames
        # standardization must not see a held-out frame
        fit = np.concatenate([rec["traces"][s:e] for s, e in
                              json.loads(str(data["metadata"]))["split_frame_bounds"]["train"]])
        np.testing.assert_allclose(data["obs_mean"], fit.mean(0))
        assert resolved.name.endswith(f"_fold{fold}of5")
        tested.append(frames["test"].min())
    assert len(set(tested)) == base["folds"], "folds must test different regions"


def test_fold_minus_one_fits_and_scores_the_whole_trace():
    """fold=-1 is the dLDS protocol: no split, every frame in every split, flagged in-sample."""
    rec = recording()
    cfg = KatoConfig("fixture.mat", latent_dim=2, window=20, train_stride=10, gap_frames=5, fold=-1)
    resolved, data = prepare_recording(rec, cfg)
    frames = {s: np.unique(data[f"frame_indices_{s}"]) for s in ("train", "val", "test")}
    n = len(rec["traces"])
    for split in ("train", "val", "test"):
        assert frames[split].min() == 0 and frames[split].max() >= n - cfg.window
    # test frames are a subset of training frames -- that is the point, and why it is in-sample
    assert set(frames["test"].tolist()) <= set(frames["train"].tolist())
    meta = json.loads(str(data["metadata"]))
    assert meta["protocol"] == "whole_trace_in_sample" and meta["held_out"] is False
    assert resolved.name.endswith("_full")
    # standardization sees the whole trace, as dLDS does
    np.testing.assert_allclose(data["obs_mean"], np.asarray(rec["traces"], float).mean(0))


def test_in_sample_protocol_still_holds_out_decoder_frames():
    """fold=-1 gives the probe its own split, so decoding is never fit and scored on one frame."""
    from opssm.benchmarks.decoding import score_linear_decoding
    rec = recording()
    cfg = KatoConfig("fixture.mat", latent_dim=2, window=20, train_stride=10, gap_frames=5, fold=-1)
    _, data = prepare_recording(rec, cfg)
    means = {s: np.random.default_rng(0).normal(size=(*data[f"states_{s}"].shape, 2))
             for s in ("train", "val", "test")}
    metrics, arrays = score_linear_decoding(means, data)
    assert metrics["linear_decode_status"] == "ok_in_sample_latents"
    # the probe's own splits must be disjoint even though the dataset's are identical
    assert metrics["linear_decode_n_train"] > 0 and metrics["linear_decode_n_test"] > 0
    total = sum(metrics[f"linear_decode_n_{s}"] for s in ("train", "val", "test"))
    assert total < data["frame_indices_train"].max() + 1, "gap frames must be dropped"
