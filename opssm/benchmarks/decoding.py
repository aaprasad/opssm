"""Frozen linear behavior probe: fit train, choose regularization on val, score test."""
import json

import numpy as np


def _unique_frames(mean, labels, frames, excluded):
    if mean.shape[:2] != labels.shape or labels.shape != frames.shape:
        raise ValueError("Latents, labels and recording frame indices must agree")
    # Overlapping training windows can infer a frame more than once. Average those
    # posterior means, then count each recording frame once in the probe.
    unique, first, inv = np.unique(frames.ravel(), return_index=True, return_inverse=True)
    target = labels.ravel()[first]
    if not np.array_equal(labels.ravel(), target[inv]):
        raise ValueError("Inconsistent labels for the same recording frame")
    features = np.zeros((len(unique), mean.shape[-1]), dtype=np.float64)
    np.add.at(features, inv, mean.reshape(-1, mean.shape[-1]))
    features /= np.bincount(inv)[:, None]
    keep = (target >= 0) & ~np.isin(target, excluded)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite decoder features")
    return features[keep], target[keep], unique[keep]


def _scores(y, pred):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    # Balanced accuracy = mean recall over classes present in this split.
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(recall_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def score_linear_decoding(means, data, c_grid=(.001, .01, .1, 1., 10., 100.)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    meta = json.loads(str(data["metadata"]))
    excluded = [i for i, name in enumerate(meta["state_names"]) if name.upper() == "NOSTATE"]
    splits = {s: _unique_frames(means[s], data[f"states_{s}"], data[f"frame_indices_{s}"], excluded)
              for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(splits[a][2], splits[b][2]).size:
            raise ValueError("Behavior decoder splits share recording frames")
    metrics = dict(linear_decode_accuracy=None, linear_decode_balanced_accuracy=None,
                   linear_decode_macro_f1=None, linear_decode_excluded_classes=excluded,
                   **{f"linear_decode_n_{s}": len(v[1]) for s, v in splits.items()})
    if any(not len(v[1]) for v in splits.values()):
        return {**metrics, "linear_decode_status": "empty_labeled_split"}, {}
    xtrain, ytrain, _ = splits["train"]
    if len(np.unique(ytrain)) < 2:
        return {**metrics, "linear_decode_status": "insufficient_training_classes"}, {}
    scaler = StandardScaler().fit(xtrain)
    features = {s: scaler.transform(v[0]) for s, v in splits.items()}
    best, best_score, selected_c = None, -np.inf, None
    for c in sorted(c_grid):
        probe = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, class_weight="balanced")
        probe.fit(features["train"], ytrain)
        value = _scores(splits["val"][1], probe.predict(features["val"]))["balanced_accuracy"]
        if value > best_score:  # Ties choose the stronger regularization.
            best, best_score, selected_c = probe, value, c
    target = splits["test"][1]
    pred = best.predict(features["test"])
    metrics.update({f"linear_decode_{k}": v for k, v in _scores(target, pred).items()})
    majority = np.unique(ytrain)[np.argmax(np.unique(ytrain, return_counts=True)[1])]
    metrics.update(linear_decode_status="ok", linear_decode_C=selected_c,
                   linear_decode_val_balanced_accuracy=best_score,
                   linear_decode_train_classes=best.classes_.tolist(),
                   linear_decode_unseen_test_fraction=float(np.mean(~np.isin(target, best.classes_))),
                   linear_decode_majority_accuracy=float(np.mean(target == majority)))
    arrays = dict(linear_decode_prediction=pred, linear_decode_target=target,
                  linear_decode_test_frames=splits["test"][2], linear_decode_classes=best.classes_,
                  linear_decode_coef=best.coef_, linear_decode_intercept=best.intercept_,
                  linear_decode_feature_mean=scaler.mean_, linear_decode_feature_scale=scaler.scale_)
    return metrics, arrays
