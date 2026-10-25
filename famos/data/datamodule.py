"""Lightning data module that builds the configured train/val/test datasets."""

import logging
import os
from functools import partial

import lightning as L
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset, DataLoader

from . import rendering
from .dataset import collate_fn
from .hdf5_dataset import HDF5Dataset
from .procedural_dataset import ProceduralDataset
from .sampling import MixedBatchSampler, SeededDataset

logger = logging.getLogger("Data")


def _physical_gpu_ids() -> list[int]:
    """Physical GPU ids from CUDA_VISIBLE_DEVICES (GPU 0 if unset or not numeric)."""
    try:
        ids = [
            int(x)
            for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if x.strip()
        ]
    except ValueError:
        ids = []
    return ids or [0]


# Read before any worker is started. Rank r renders on the r-th visible GPU.
_GPU_IDS = _physical_gpu_ids()


class _PinEGLDevice:
    """Worker init: render on this rank's GPU with a fresh EGL context."""

    def __init__(self, local_rank: int):
        self.gpu_id = _GPU_IDS[local_rank % len(_GPU_IDS)]

    def __call__(self, worker_id: int = 0):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_DEVICE_ID"] = str(self.gpu_id)
        rendering.reset_renderer()


class FAMOSDataModule(L.LightningDataModule):
    """Builds the datasets in `dataset_list` and their dataloaders.

    Each `dataset_list` entry names a source: `dataset_type: hdf5` for a release HDF5
    file (`dir`, `filenames`, and `states` outside training) or `procedural` for the
    procedural generator (`procedural_*` options). The remaining keyword arguments
    (camera and sensor options) are shared by every dataset (see `ObservationDataset`).

    Training batches come from `MixedBatchSampler`: one source per batch, chosen by
    each entry's `sample_ratio`.
    """

    def __init__(
        self,
        data_root: str,
        dataset_list: dict,
        batch_size: int,
        num_workers: int,
        num_points: int,
        max_parts: int,
        max_states: int,
        state_sampling_probs: list[float] | None = None,
        observed_articulation_span_min_visible_points: int = 5,
        val_num_workers: int = 0,
        sampling_seed: int = 0,
        resume_global_step: int = 0,
        **observation_kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_list = dataset_list
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_num_workers = int(val_num_workers)
        self.num_points = int(num_points)
        self.max_parts = int(max_parts)
        self.max_states = int(max_states)
        self.state_sampling_probs = state_sampling_probs
        self.observed_articulation_span_min_visible_points = int(
            observed_articulation_span_min_visible_points
        )
        self.sampling_seed = int(sampling_seed)
        self.resume_global_step = int(resume_global_step)
        self.observation_kwargs = observation_kwargs

    def _build_dataset(self, cfg: dict, split: str):
        common = dict(
            split=split,
            dataset_name=cfg["name"],
            num_points=self.num_points,
            up_axis=cfg["up_axis"],
            max_states=self.max_states,
            **self.observation_kwargs,
        )
        if cfg["dataset_type"] == "hdf5":
            return HDF5Dataset(
                data_path=os.path.join(self.data_root, cfg["dir"]),
                sample_list_path=cfg["filenames"],
                states_path=cfg.get("states"),
                **common,
            )
        if cfg["dataset_type"] == "procedural":
            procedural = {k: v for k, v in cfg.items() if k.startswith("procedural_")}
            return ProceduralDataset(max_parts=self.max_parts, **procedural, **common)
        raise ValueError(f"Unsupported dataset_type={cfg['dataset_type']!r}")

    def setup(self, stage: str):
        logger.info(f"| {'Dataset':<16} | {'Split':<8} | {'Length':<8} |")
        if stage == "fit":
            self.train_datasets = [
                self._build_dataset(cfg, "train") for cfg in self.dataset_list["train"]
            ]
            self.train_sample_ratios = [
                float(cfg["sample_ratio"]) for cfg in self.dataset_list["train"]
            ]
            if min(self.train_sample_ratios) <= 0.0:
                raise ValueError("Every train dataset needs a positive sample_ratio.")
        if stage in ("fit", "validate"):
            self.val_dataset = ConcatDataset(
                [self._build_dataset(cfg, "val") for cfg in self.dataset_list["val"]]
            )
        if stage in ("test", "predict"):
            self.test_datasets = [
                self._build_dataset(cfg, "test") for cfg in self.dataset_list["test"]
            ]

    def _collate(self, train: bool = False):
        return partial(
            collate_fn,
            max_parts=self.max_parts,
            state_sampling_probs=self.state_sampling_probs,
            observed_articulation_span_min_visible_points=(
                self.observed_articulation_span_min_visible_points
            ),
            train=train,
        )

    def _local_rank(self) -> int:
        return self.trainer.local_rank if self.trainer is not None else 0

    def _worker_kwargs(self, num_workers: int, persistent: bool, context: str) -> dict:
        if num_workers == 0:
            # Render in this process, on this rank's GPU.
            _PinEGLDevice(self._local_rank())()
            return {}
        return {
            "persistent_workers": persistent,
            "prefetch_factor": 2,
            "worker_init_fn": _PinEGLDevice(self._local_rank()),
            "multiprocessing_context": mp.get_context(context),
        }

    def train_dataloader(self):
        trainer = self.trainer
        accumulate = trainer.accumulate_grad_batches if trainer is not None else 1
        max_steps = trainer.max_steps if trainer is not None else -1
        train_dataset = ConcatDataset(self.train_datasets)
        if max_steps > 0:
            num_batches = max_steps * accumulate
        else:
            num_batches = max(1, len(train_dataset) // self.batch_size)
        sampler = MixedBatchSampler(
            dataset_lengths=[len(dataset) for dataset in self.train_datasets],
            sample_ratios=self.train_sample_ratios,
            batch_size=self.batch_size,
            num_batches=num_batches,
            seed=self.sampling_seed,
            rank=trainer.global_rank if trainer is not None else 0,
            world_size=trainer.world_size if trainer is not None else 1,
            start_batch=self.resume_global_step * accumulate,
        )
        logger.info(
            f"Training batches from {[cfg['name'] for cfg in self.dataset_list['train']]} "
            f"with ratios {self.train_sample_ratios}, starting at batch {sampler.start_batch}"
        )
        return DataLoader(
            SeededDataset(train_dataset),
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate(train=True),
            **self._worker_kwargs(
                self.num_workers, persistent=True, context="forkserver"
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.val_num_workers,
            collate_fn=self._collate(),
            **self._worker_kwargs(
                self.val_num_workers, persistent=True, context="fork"
            ),
        )

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=self._collate(),
                **self._worker_kwargs(
                    self.num_workers, persistent=False, context="fork"
                ),
            )
            for dataset in self.test_datasets
        ]
