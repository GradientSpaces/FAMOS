"""Optimizer-step progress and ETA logging."""

from __future__ import annotations

import time
from collections import deque

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class StepProgressCallback(Callback):
    """Report progress against ``trainer.max_steps``, since training is step-based, not epoch-based.

    The speed is averaged over the last `window_size` optimizer steps.
    """

    def __init__(self, log_every_n_steps: int = 50, window_size: int = 50) -> None:
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._history = deque(maxlen=max(2, int(window_size)))
        self._last_seen_step = -1

    def on_train_start(self, trainer, pl_module) -> None:
        step = trainer.global_step
        self._last_seen_step = step
        self._history.clear()
        self._history.append((step, time.monotonic()))
        rank_zero_info(
            f"Step progress: {step}/{trainer.max_steps} "
            f"({100.0 * min(1.0, step / trainer.max_steps):.2f}%)"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        # Called once per micro-batch; act once per optimizer step.
        step = trainer.global_step
        if step <= 0 or step == self._last_seen_step:
            return

        now = time.monotonic()
        self._last_seen_step = step
        self._history.append((step, now))
        first_step, first_time = self._history[0]
        steps_per_sec = max(0.0, (step - first_step) / max(1e-9, now - first_time))
        progress = min(1.0, step / trainer.max_steps)
        remaining = max(0, trainer.max_steps - step)
        eta_hours = remaining / max(steps_per_sec, 1e-9) / 3600.0

        # Progress bar only; the console line below reports the full step timing.
        for name, value in (("progress_pct", progress * 100.0), ("eta_hours", eta_hours)):
            pl_module.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
                rank_zero_only=True,
            )

        if step % self.log_every_n_steps == 0:
            rank_zero_info(
                f"Step progress: {step}/{trainer.max_steps} ({progress * 100.0:.2f}%), "
                f"remaining={remaining}, speed={steps_per_sec:.3f} opt steps/s, "
                f"ETA={eta_hours:.2f}h"
            )
