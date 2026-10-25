"""Training utilities for FAMOS."""

import glob
import logging
import os
import hydra
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("Train")


def find_wandb_run_id(ckpt_path: str) -> str | None:
    """ID of the most recent W&B run saved next to `ckpt_path` (run dirs end in `-<id>`)."""
    run_dirs = glob.glob(os.path.join(os.path.dirname(ckpt_path), "wandb", "*run-*"))
    if not run_dirs:
        return None
    return os.path.basename(max(run_dirs, key=os.path.getmtime)).rsplit("-", 1)[-1]


def setup_wandb_resume(cfg: DictConfig) -> None:
    """Continue logging to the previous W&B run when resuming from a checkpoint."""
    if "wandb" not in (cfg.get("loggers") or {}):
        return
    run_id = find_wandb_run_id(cfg.ckpt_path)
    if run_id:
        cfg.loggers.wandb.id = run_id
        cfg.loggers.wandb.resume = "allow"
        rank_zero_info(f"Resuming W&B run {run_id}.")


def setup_loggers(cfg: DictConfig) -> list[Logger]:
    return [
        hydra.utils.instantiate(logger)
        for logger in cfg.get("loggers", dict()).values()
    ]


@rank_zero_only
def log_code_to_wandb(loggers: list[Logger]) -> None:
    """Upload the repository's .py and .yaml files to the W&B run."""
    for log in loggers:
        if isinstance(log, WandbLogger):
            original_cwd = hydra.utils.get_original_cwd()
            log.experiment.log_code(
                root=original_cwd,
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
            )


@rank_zero_only
def log_config_to_wandb(loggers: list[Logger], cfg: DictConfig):
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        for log in loggers:
            if isinstance(log, WandbLogger):
                run = log.experiment
                # Plot everything against the optimizer step (see FAMOS.on_before_optimizer_step).
                run.define_metric("train/optimizer_step")
                run.define_metric("*", step_metric="train/optimizer_step")
                run.config.update(config_dict, allow_val_change=True)
    except Exception as e:
        logger.warning(f"Failed to update wandb config: {e}")
