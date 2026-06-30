"""Hydra entry point for the neural mesh-free Zakai filter.

Examples:
    python scripts/train.py experiment=supervised
    python scripts/train.py experiment=em_1d
    python scripts/train.py experiment=em_highd
    python scripts/train.py experiment=em_highd model.pca_init=false trainer.max_steps=8000
"""
import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    pl.seed_everything(cfg.seed)
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
