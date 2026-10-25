"""Train FAMOS from a Hydra configuration."""

import os
import warnings

import hydra
import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from omegaconf import DictConfig, open_dict

from famos.utils.training import (
    log_code_to_wandb,
    log_config_to_wandb,
    setup_loggers,
    setup_wandb_resume,
)

warnings.filterwarnings("ignore", module="lightning")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup_training(cfg: DictConfig):
    os.makedirs(cfg.log_dir, exist_ok=True)
    loggers = setup_loggers(cfg)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model, log_dir=cfg.log_dir)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)
    return model, datamodule, trainer, loggers


def _set_resume_step(cfg: DictConfig, ckpt_path: str, is_fresh_run: bool) -> None:
    """Tell the data module the resumed optimizer step, so it continues the seeded data stream."""
    resume_step = 0
    if not is_fresh_run:
        # mmap: only the step counter is read, not the weights.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
        resume_step = int(checkpoint.get("global_step", 0))
        rank_zero_info(f"Resuming from {ckpt_path} at global_step={resume_step}.")
    with open_dict(cfg.data):
        cfg.data.resume_global_step = resume_step


def _num_devices(devices) -> int:
    """Number of GPUs selected by Lightning's `devices` setting."""
    if devices in ("auto", -1, "-1"):
        return torch.cuda.device_count()
    if isinstance(devices, str):
        return len(devices.strip(",").split(",")) if "," in devices else int(devices)
    if isinstance(devices, int):
        return devices
    return len(devices)


def _set_gradient_accumulation(cfg: DictConfig) -> None:
    """Accumulate gradients so that every optimizer step uses `effective_batch_size` samples,
    whatever the number of GPUs."""
    with open_dict(cfg.trainer):
        effective = int(cfg.trainer.pop("effective_batch_size"))
        per_batch = _num_devices(cfg.trainer.devices) * int(cfg.trainer.num_nodes) * int(cfg.data.batch_size)
        if per_batch == 0 or effective % per_batch:
            raise ValueError(
                f"effective_batch_size {effective} is not a multiple of GPUs x nodes x batch_size "
                f"= {per_batch}; change trainer.devices or data.batch_size."
            )
        cfg.trainer.accumulate_grad_batches = effective // per_batch
    rank_zero_info(
        f"Effective batch size {effective} = {per_batch} per batch x "
        f"{cfg.trainer.accumulate_grad_batches} accumulated batches."
    )


def _set_val_check_interval(cfg: DictConfig) -> None:
    """Convert `val_check_optimizer_steps` to Lightning's `val_check_interval`, which counts batches."""
    with open_dict(cfg.trainer):
        val_steps = int(cfg.trainer.pop("val_check_optimizer_steps"))
        cfg.trainer.val_check_interval = val_steps * int(cfg.trainer.accumulate_grad_batches)


@hydra.main(version_base="1.3", config_path="./config", config_name="train_stage1")
def main(cfg: DictConfig):
    ckpt_path = cfg.get("ckpt_path")
    is_fresh_run = not (ckpt_path and os.path.exists(ckpt_path))
    if is_fresh_run:
        L.seed_everything(cfg.seed, workers=True, verbose=False)
    else:
        setup_wandb_resume(cfg)
    _set_resume_step(cfg, ckpt_path, is_fresh_run)
    _set_gradient_accumulation(cfg)
    _set_val_check_interval(cfg)

    model, datamodule, trainer, loggers = setup_training(cfg)
    log_config_to_wandb(loggers, cfg)
    log_code_to_wandb(loggers)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=None if is_fresh_run else ckpt_path,
    )


if __name__ == "__main__":
    main()
