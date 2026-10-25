"""Procedural objects generated on the fly (training only)."""

import logging
import random

import numpy as np

from .dataset import ObservationDataset
from .procedural_assets import ProceduralAssetFactory

logger = logging.getLogger("Data")


class ProceduralDataset(ObservationDataset):
    """An endless stream of procedural objects; index `i` always yields the same one.

    `procedural_family_pitch_ranges` overrides the camera pitch range for families that
    are hard to read from the default `camera_pitch_range` (e.g. flat scissors).
    """

    def __init__(
        self,
        max_parts: int,
        procedural_length: int = 10_000_000,
        procedural_seed: int = 0,
        procedural_families: list[str] | None = None,
        procedural_forbid_cabinet_side_doors: bool = True,
        procedural_family_pitch_ranges: dict[str, tuple[float, float]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.length = int(procedural_length)
        self.family_pitch_ranges = {
            str(family): tuple(float(x) for x in pitch_range)
            for family, pitch_range in (procedural_family_pitch_ranges or {}).items()
        }
        self.asset_factory = ProceduralAssetFactory(
            families=procedural_families,
            seed=int(procedural_seed),
            forbid_cabinet_side_doors=bool(procedural_forbid_cabinet_side_doors),
            max_joints=max(1, max_parts - 1),
        )
        logger.info(
            f"| {self.dataset_name:16s} | {self.split:8s} | {self.length:8d} | "
            f"families={','.join(self.asset_factory.families)}"
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        bundle = self.asset_factory.generate(index)
        seed = int(bundle.seed) % (2**32)
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed((seed + 1_337) % (2**32))
            return self._observe(
                key=bundle.model_id,
                object_id=bundle.model_id,
                rest_vertices=bundle.rest_vertices,
                faces=np.asarray(bundle.faces, dtype=np.int32),
                face_part_ids=bundle.face_part_ids,
                kinematics=bundle.kinematics,
                rest_center=np.asarray(bundle.rest_center, dtype=np.float64),
                rest_scale=float(bundle.rest_scale),
                canonical_yaw_offset=float(bundle.canonical_yaw_offset),
                base_seed=seed,
                pitch_range=self.family_pitch_ranges.get(bundle.family),
            )
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
