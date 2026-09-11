"""Resolve the repository's Hydra experiment model settings for the common data."""
import hashlib
import json
from pathlib import Path


EXPERIMENTS = dict(doublewell="duncker_dw", vanderpol="duncker_vdp",
                   lorenz="duncker_lorenz", kato="kato")


def resolve_opssm_config(system, *, experiment=None, config_dir=None, overrides=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    directory = Path(config_dir) if config_dir else Path(__file__).resolve().parents[2] / "configs"
    directory = directory.resolve()
    experiment = experiment or EXPERIMENTS[system]
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}", "backend=jax"])
        hp = OmegaConf.to_container(cfg.model, resolve=True)
    for key in ("_target_", "train_dir"):
        hp.pop(key, None)
    if overrides:
        hp.update(overrides)
    # Fail explicitly if an experiment requests a path absent from this JAX backend.
    if hp.get("encoder", "gru") != "gru" or hp.get("encoder_kwargs"):
        raise ValueError("The JAX OPSSM benchmark currently supports the default GRU encoder only")
    if any(hp.get(k, False) for k in ("learn_smoother", "joint_g", "g_net")):
        raise ValueError("The JAX OPSSM benchmark does not implement smoother/joint_g/g_net experiments")
    if hp.get("loss") != "zakai" or not hp.get("learn_dynamics", True) or hp.get("mean_method") != "mala":
        raise ValueError("The JAX OPSSM benchmark requires Zakai EM with MALA readout")
    if int(hp.get("chunk_size", 0)) < 1:
        raise ValueError("OPSSM chunk_size must be positive")
    files = [directory / "train.yaml", directory / "model/operator.yaml",
             directory / "experiment" / f"{experiment}.yaml"]
    provenance = dict(experiment=experiment, config_dir=str(directory),
                      source_sha256={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in files},
                      resolved_model_sha256=hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest(),
                      data_policy="shared benchmark data, dimensions, timing and preprocessing",
                      training_policy="benchmark step budget and validation cadence")
    return hp, provenance
