"""Shared by the per-dataset preprocessors: read the split lists, write the release HDF5s."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args(dataset: str, input_help: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Build the {dataset} release HDF5 files.")
    parser.add_argument("--input_dir", type=Path, required=True, help=input_help)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help=f"Writes {dataset}_<split>.h5 here, e.g. $FAMOS_DATA_ROOT/{dataset}.")
    parser.add_argument("--split_dir", type=Path, default=REPO_ROOT / "data_split" / dataset,
                        help="Folder with the <split>.txt object lists.")
    return parser.parse_args()


def build_release(dataset: str, args: argparse.Namespace, process) -> None:
    """Process every object listed in `split_dir` and write it into each split file that lists it.

    `process(key)` returns a dict with `faces`, `face_part_ids`, `rest_vertices`,
    `kinematics` and optionally `canonical_yaw_offset`. Objects already written to all their
    split files are skipped, so an interrupted run can be resumed.
    """
    splits = {}
    for split in SPLITS:
        path = args.split_dir / f"{split}.txt"
        if path.exists():
            splits[split] = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not splits:
        raise FileNotFoundError(f"No split lists in {args.split_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: args.output_dir / f"{dataset}_{split}.h5" for split in splits}
    targets = {}
    for split, keys in splits.items():
        for key in keys:
            targets.setdefault(key, []).append(split)

    def written(key):
        for split in targets[key]:
            if not paths[split].exists():
                return False
            with h5py.File(paths[split], "r") as f:
                if key not in f:
                    return False
        return True

    pending = [key for key in sorted(targets) if not written(key)]
    print(f"{dataset}: {len(targets)} objects, {len(pending)} to process")
    for key in tqdm(pending):
        obj = process(key)
        for split in targets[key]:
            with h5py.File(paths[split], "a") as f:
                if key in f:
                    del f[key]
                write_object(f.create_group(key), **obj)


def write_object(group, faces, face_part_ids, rest_vertices, kinematics, canonical_yaw_offset=None):
    rest_vertices = np.asarray(rest_vertices, dtype=np.float32)
    group.create_dataset("faces", data=np.asarray(faces, dtype=np.int32))
    group.create_dataset("face_part_ids", data=np.asarray(face_part_ids, dtype=np.int32))
    group.create_dataset("rest_vertices", data=rest_vertices)
    bb_min, bb_max = rest_vertices.min(axis=0), rest_vertices.max(axis=0)
    group.attrs["rest_center"] = ((bb_min + bb_max) / 2.0).tolist()
    group.attrs["rest_scale"] = float(max((bb_max - bb_min).max(), 1e-8))
    group.attrs["kinematics"] = json.dumps(kinematics)
    if canonical_yaw_offset is not None:
        group.attrs["canonical_yaw_offset"] = float(canonical_yaw_offset)
