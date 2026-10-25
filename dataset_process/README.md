# Dataset preprocessing

We provide the data preprocessing scripts for the training and evaluation data. Each script
processes the objects listed in `data_split/<dataset>/{train,val,test}.txt` and writes one
`*.h5` file per split:

```text
$FAMOS_DATA_ROOT/
├── partnet_mobility/partnet_mobility_{train,val,test}.h5
├── grscenes/grscenes_{train,val}.h5
├── acd/acd_test.h5
└── articraft/articraft_test.h5
```

Evaluation needs only the `*_test.h5` files; training also needs the PartNet-Mobility and
GRScenes train/val files. In order to reproduce validation and evaluation metrics, we provide the sampled joint states in
`data_split/<dataset>/{val,test}_states.json`.

## Setup

Set the output root and install the extra dependencies:

```bash
export FAMOS_DATA_ROOT=data      # output root
pip install usd-core pygltflib   # extra dependencies for GRScenes and ACD
```

## PartNet-Mobility

Download PartNet-Mobility from [SAPIEN](https://sapien.ucsd.edu/downloads) (registration
required); it unpacks to one folder per model id.

```bash
python dataset_process/partnet_mobility.py \
  --input_dir /path/to/partnet-mobility \
  --output_dir "$FAMOS_DATA_ROOT/partnet_mobility"
```

## GRScenes

Download the three parts of `scenes/GRScenes-100/home_scenes.zip` (`.zip`, `.z01`, `.z02`)
from [InternRobotics/GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes).

```bash
(cd /path/to/GRScenes-100 && 7z x home_scenes.zip)
python dataset_process/grscenes.py \
  --input_dir /path/to/GRScenes-100/home_scenes/models/object/articulated \
  --output_dir "$FAMOS_DATA_ROOT/grscenes"
```

## ACD

ACD is part of the [S2O](https://github.com/3dlg-hcvc/s2o) release. Accept the terms of
[3dlg-hcvc/s2o](https://huggingface.co/datasets/3dlg-hcvc/s2o) on Hugging Face, then download
only the interior-completed GLBs.

```bash
hf download 3dlg-hcvc/s2o --repo-type dataset --include "acd/articulated/glb/*" \
  --local-dir /path/to/s2o
python dataset_process/acd.py \
  --input_dir /path/to/s2o/acd/articulated/glb \
  --output_dir "$FAMOS_DATA_ROOT/acd"
```

## ArtiCraft

Download [Articraft-10K](https://huggingface.co/datasets/camvsl/Articraft-10K) and extract
each record archive in place.

```bash
hf download camvsl/Articraft-10K --repo-type dataset \
  --revision 3c79d5a05bb7cb6bf7bfee5e090176636ee3ac65 --local-dir /path/to/Articraft-10K
(cd /path/to/Articraft-10K && for f in rec_*.tar.gz; do tar -xzf "$f"; done)
python dataset_process/articraft.py \
  --input_dir /path/to/Articraft-10K \
  --output_dir "$FAMOS_DATA_ROOT/articraft"
```

<details>
<summary><b>HDF5 format</b></summary>

Each object is stored under `<category>/<model_id>`:

| Field | Content |
| --- | --- |
| `rest_vertices`, `faces` | rest-pose triangle mesh |
| `face_part_ids` | rigid part of each face; `0` is the static base |
| `kinematics` (attribute) | JSON with, per joint, its `type`, `axis`, `origin`, `limits` and `controlled_part_ids` |
| `rest_center`, `rest_scale` (attributes) | bounding-box center and largest extent of the rest mesh |
| `canonical_yaw_offset` (attribute) | GRScenes and ArtiCraft only: camera yaw of the front view |

Meshes and joints keep the units and up axis of the raw dataset: PartNet-Mobility and ACD
are Y-up, GRScenes and ArtiCraft Z-up, and GRScenes is in centimeters.

</details>
