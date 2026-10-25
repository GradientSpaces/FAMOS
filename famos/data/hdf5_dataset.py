"""Objects from a release HDF5 file (built by dataset_process/)."""

import hashlib
import json
import logging

import h5py
import numpy as np

from .dataset import ObservationDataset

logger = logging.getLogger("Data")


class HDF5Dataset(ObservationDataset):
    """Loads the objects listed in `sample_list_path` from the HDF5 file `data_path`.

    Validation and test splits observe the fixed joint states in `states_path` from a
    fixed camera per object.
    """

    def __init__(
        self,
        data_path: str,
        sample_list_path: str,
        states_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        with open(sample_list_path) as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
        if self.split != "train":
            if not states_path:
                raise ValueError(
                    f"{self.dataset_name}: split '{self.split}' requires `states`."
                )
            with open(states_path) as f:
                self.fixed_states = json.load(f)
            missing = [key for key in self.sample_list if key not in self.fixed_states]
            if missing:
                raise KeyError(f"{states_path} has no states for {missing[:5]}")
        self._h5 = None  # opened lazily, once per DataLoader worker
        logger.info(f"| {self.dataset_name:16s} | {self.split:8s} | {len(self):8d} |")

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        key = self.sample_list[index]
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        group = self._h5[key]

        canonical_yaw_offset = float(group.attrs.get("canonical_yaw_offset", 0.0))
        if "grscenes" in self.dataset_name.lower():
            # GRScenes objects face +X after preprocessing; the camera arc faces -Y.
            canonical_yaw_offset += 90.0
        base_seed = None
        if self.split != "train":
            base_seed = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % (2**32)

        return self._observe(
            key=key,
            object_id=key.split("/", 1)[1],
            rest_vertices=np.asarray(group["rest_vertices"][:], dtype=np.float64),
            faces=np.asarray(group["faces"][:], dtype=np.int32),
            face_part_ids=group["face_part_ids"][:],
            kinematics=json.loads(group.attrs["kinematics"]),
            rest_center=np.array(group.attrs["rest_center"], dtype=np.float64),
            rest_scale=float(group.attrs["rest_scale"]),
            canonical_yaw_offset=canonical_yaw_offset,
            base_seed=base_seed,
        )
