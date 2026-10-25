"""Run FAMOS inference and the segmentation and motion evaluation."""

import logging
import os
import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from famos.utils.checkpoint import load_checkpoint_for_module

logger = logging.getLogger("Evaluate")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup(cfg: DictConfig):
    ckpt_path = cfg.get("ckpt_path", None)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Set FAMOS_CHECKPOINT or override ckpt_path."
        )

    # Point subsampling draws from the global RNGs; cameras are seeded per object.
    L.seed_everything(cfg.seed, workers=True, verbose=False)
    logger.info(f"Seed set to {cfg.seed} for sampling")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(
        model.articulation_transformer,
        ckpt_path,
        prefix_to_remove="articulation_transformer.",
    )
    model.eval()
    model.ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        enable_checkpointing=False,
        logger=False,
    )
    return model, datamodule, trainer


@hydra.main(version_base="1.3", config_path="./config", config_name="inference")
def main(cfg: DictConfig):
    model, datamodule, trainer = setup(cfg)
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    # Metrics are written to <log_dir>/<checkpoint-name>/<dataset>_eval/metrics.json;
    # aggregate them with scripts/eval/summarize.py.
    print(f"\nMetrics written under {cfg.log_dir}\n")


if __name__ == "__main__":
    main()
