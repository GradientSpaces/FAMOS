"""Predict the movable parts and joints of objects from their partial point clouds.

    python predict.py                              # all demo objects
    python predict.py input_dir=demo/real_world    # the real-world captures of the paper
    python predict.py input_dir=demo/benchmark     # benchmark objects of the paper
    python predict.py input_dir=my_object          # your own capture, see scripts/reconstruct.py

Every object folder holds `reconstruction.npz` (the point cloud of every input state) and
optionally `view.json` with the input state shown in the figures (`state`, the first by
default), their up and viewing directions (`up`, `eye`), and the animated range of each part
(`motion`, see `save_gif`). Missing checkpoints are
downloaded from Hugging Face. Results go to `<output_dir>/<object>/`:

    prediction.json   one entry per movable part: joint type, axis, and origin
    prediction.npz    the points fed to the model and their part labels (0 = static)
    prediction.png    the predicted parts and joints on one input state
    prediction.gif    the predicted parts opening and closing along their joints

Coordinates are those of the input point clouds: the first camera's frame (x right, y up,
z toward the viewer).
"""

import json
import logging
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from PIL import Image
from pytorch3d.ops import estimate_pointcloud_normals
from scipy.spatial import cKDTree

from famos.utils import prediction_rendering as rendering
from famos.utils.checkpoint import load_checkpoint_for_module

logger = logging.getLogger("Predict")
logging.getLogger("OpenGL").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The reduce argument of torch.scatter")

FAMOS_REPO = "kevinqu7/famos-model"
PARTFIELD_REPO = "mikaelaangel/partfield-ckpt"
DEMO_DIR = Path("demo")
JOINT_TYPES = {1: "revolute", 2: "prismatic"}
# Without a view.json, the figure looks at the object from the first camera, slightly from
# above: `up` is the camera's up, `eye` points from the object toward the viewer.
DEFAULT_VIEW = {"up": [0.0, 1.0, 0.0], "eye": [0.0, 0.2079, 0.9781]}
# The GIF opens every predicted part along its predicted joint and closes it again. Parts with
# fewer points in the shown state are left static.
GIF_FRAMES = 24  # frames per opening
GIF_FRAME_MS = 60
GIF_HOLD_MS = 500  # pause when fully closed and fully open
MIN_PART_POINTS = 10
# Smallest opening shown, so a part that barely moves across the inputs still moves visibly.
MIN_OPENING = {"revolute": np.deg2rad(70.0), "prismatic": 0.25}  # radians, fraction of size


def download_if_missing(path, repo_id: str, filename: str):
    path = Path(path)
    if not path.exists():
        logger.info(f"Downloading {filename} from {repo_id} to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(hf_hub_download(repo_id, filename, local_dir=path.parent)).rename(path)


def setup(cfg: DictConfig):
    """Download missing checkpoints and load FAMOS."""
    download_if_missing(cfg.ckpt_path, FAMOS_REPO, "famos.safetensors")
    download_if_missing(cfg.model.partfield_checkpoint, PARTFIELD_REPO, "model_objaverse.ckpt")
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(
        model.articulation_transformer,
        cfg.ckpt_path,
        prefix_to_remove="articulation_transformer.",
    )
    return model.cuda().eval()


def sample(points, n, rng):
    """`n` random points, repeating points if there are fewer."""
    if len(points) >= n:
        return points[rng.permutation(len(points))[:n]]
    extra = rng.choice(len(points), size=n - len(points), replace=True)
    idx = np.concatenate([np.arange(len(points)), extra])
    rng.shuffle(idx)
    return points[idx]


def model_inputs(states, cfg):
    """Normalize all states jointly by their bounding box, as in training, and sample the
    points for PartField and for the model."""
    all_points = np.concatenate(states)
    lo, hi = all_points.min(0), all_points.max(0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    scale = float(max((hi - lo).max(), 1e-6))
    states = [(pts - center) / scale for pts in states]

    rng = np.random.default_rng(cfg.seed)
    partfield_points = np.stack([sample(pts, cfg.data.num_partfield_points, rng) for pts in states])
    points = np.stack([sample(pts, cfg.data.num_points, rng) for pts in states])
    return partfield_points, points, center, scale


@torch.inference_mode()
def run_model(model, partfield_points, points):
    point_clouds = torch.from_numpy(points).to(model.device)
    batch = {
        "point_clouds": point_clouds[None],
        "partfield_point_clouds": torch.from_numpy(partfield_points).to(model.device)[None],
        "surface_normals": estimate_pointcloud_normals(point_clouds, neighborhood_size=30)[None],
    }
    output = model(batch, model._encode_with_partfield(batch))
    return {k: v[0].float().cpu().numpy() for k, v in output.items()}


def decode(output, points, min_confidence):
    """Per-point part labels (0 = static) and one joint per movable part.

    The model often classifies queries that own no points as movable; only queries that own
    points, with mean probability of at least `min_confidence` on them, become parts.
    """
    probs = torch.from_numpy(output["segmentation_logits"]).softmax(-1).numpy()  # (S, N, Q)
    query = probs.argmax(-1)
    labels = np.zeros(query.shape, dtype=np.int64)
    joints = []
    for q in np.flatnonzero(output["joint_types"] > 0):
        assigned = query == q
        if not assigned.any() or probs[assigned, q].mean() < min_confidence:
            continue
        label = len(joints) + 1
        labels[assigned] = label
        # Joints are placed next to the part as seen in the first state it appears in.
        state = int(np.flatnonzero(assigned.any(1))[0])
        centroid = points[state][assigned[state]].mean(0)

        joint_type = JOINT_TYPES[int(output["joint_types"][q])]
        if joint_type == "revolute":
            d, m = np.split(output["revolute_plucker_coordinates"][q], 2)
            axis = d / np.linalg.norm(d)
            origin = np.cross(d, m) / np.dot(d, d)
            origin = origin + np.dot(centroid - origin, axis) * axis
        else:
            axis = output["prismatic_directions"][q]
            origin = centroid
        joints.append({"part_id": label, "type": joint_type, "axis": axis, "origin": origin})
    return labels, joints


def upright_rotation(up) -> np.ndarray:
    """Rotation from the reconstruction frame to a frame whose z axis is `up`."""
    z = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def render_prediction(points, labels, joints, view, bounds=None, size=800) -> Image.Image:
    """Shaded points colored by part, joint arrows, and a soft floor shadow, as in the paper.

    The camera and the floor are fit to `bounds` (the points by default).
    """
    rotation = upright_rotation(view["up"])
    points = points @ rotation.T
    bounds = points if bounds is None else bounds @ rotation.T
    center = (bounds.min(0) + bounds.max(0)) / 2
    radius = float(np.linalg.norm(bounds - center, axis=1).max())
    points = (points - center) / radius

    toward_camera = rotation @ np.asarray(view["eye"], dtype=np.float64)
    toward_camera /= np.linalg.norm(toward_camera)
    distance = rendering.FIT_MARGIN / np.sin(np.deg2rad(rendering.FOV_DEG) / 2)
    camera_pose = rendering._look_at(toward_camera * distance, np.zeros(3))

    joints = [
        dict(joint, axis=rotation @ joint["axis"], origin=(rotation @ joint["origin"] - center) / radius)
        for joint in joints
    ]
    floor_z = float((bounds[:, 2].min() - center[2]) / radius) - rendering.POINT_RADIUS
    return rendering.render_panel(
        points, labels, joints, camera_pose, toward_camera, floor_z, size * rendering.SUPERSAMPLE
    )


def rotate(points, origin, axis, angle):
    """Rotate `points` by `angle` about the line through `origin` along the unit `axis`."""
    p = points - origin
    cos, sin = np.cos(angle), np.sin(angle)
    return p * cos + np.cross(axis, p) * sin + np.outer(p @ axis, axis) * (1 - cos) + origin


def opening_motion(points, labels, joint, state):
    """The joint's axis, signed so that a positive value opens the part (moves it away from the
    object), its value in `state`, and the range of values the GIF sweeps.

    The model predicts no joint limits, so the range comes from the observed motion: the part's
    centroid in every input state, as a length along a prismatic axis or an angle about a
    revolute one. It starts at the least opened state and reaches half an observed span past
    the most opened one, or at least `MIN_OPENING`.
    """
    part = labels == joint["part_id"]
    visible = [s for s in range(len(points)) if part[s].sum() >= MIN_PART_POINTS]
    axis, origin = joint["axis"] / np.linalg.norm(joint["axis"]), joint["origin"]
    revolute = joint["type"] == "revolute"
    object_center = points[visible[0]].mean(0)
    reference = points[visible[0]][part[visible[0]]].mean(0)
    if revolute:
        opened = rotate(reference[None], origin, axis, 0.2)[0]
        opens = np.linalg.norm(opened - object_center) > np.linalg.norm(reference - object_center)
    else:
        opens = (reference - object_center) @ axis > 0
    axis = axis if opens else -axis

    values = {}
    for s in visible:
        centroid = points[s][part[s]].mean(0)
        if revolute:
            u, v = reference - origin, centroid - origin
            u, v = u - (u @ axis) * axis, v - (v @ axis) * axis
            values[s] = np.arctan2(np.cross(u, v) @ axis, u @ v)
        else:
            values[s] = (centroid - reference) @ axis
    low, high = min(values.values()), max(values.values())
    size = np.linalg.norm(points[state].max(0) - points[state].min(0))
    min_opening = MIN_OPENING["revolute"] if revolute else MIN_OPENING["prismatic"] * size
    return axis, values[state], low, max(high + 0.5 * (high - low), low + min_opening)


def save_gif(path, points, labels, joints, view, state, size=480):
    """Open all predicted parts of `state` together and close them again.

    A part moves over the range in `view["motion"]` (joint values relative to `state`, along
    the predicted axis) if given, and otherwise over the motion observed across the inputs.
    """
    shown = labels[state]
    ranges = view.get("motion", {})
    moving = []
    for joint in joints:
        if (shown == joint["part_id"]).sum() < MIN_PART_POINTS:
            continue
        if str(joint["part_id"]) in ranges:
            axis = joint["axis"] / np.linalg.norm(joint["axis"])
            moving.append((joint, axis, 0.0, *ranges[str(joint["part_id"])]))
        else:
            moving.append((joint, *opening_motion(points, labels, joint, state)))
    poses = []
    for t in np.linspace(0.0, 1.0, GIF_FRAMES):
        pose = points[state].copy()
        for joint, axis, value, low, high in moving:
            delta = low + (high - low) * (1 - np.cos(np.pi * t)) / 2 - value
            part = shown == joint["part_id"]
            if joint["type"] == "revolute":
                pose[part] = rotate(pose[part], joint["origin"], axis, delta)
            else:
                pose[part] += axis * delta
        poses.append(pose)
    bounds = np.concatenate(poses)
    shown_joints = [dict(j, axis=axis) for j, axis, *_ in moving]
    frames = [render_prediction(p, shown, shown_joints, view, bounds, size) for p in poses]

    # One palette for all frames, so that colors do not flicker, with the background exactly
    # white. Pillow's own palette mapping is approximate and would turn white into light gray.
    palette = Image.fromarray(np.concatenate([np.asarray(frames[0]), np.asarray(frames[-1])]))
    palette = palette.quantize(colors=255, dither=Image.Dither.NONE)
    colors = np.asarray(palette.getpalette()[: 3 * 255]).reshape(-1, 3)
    colors[colors.sum(1).argmax()] = 255
    nearest = cKDTree(colors)
    indexed = []
    for frame in frames:
        rgb = np.asarray(frame).reshape(-1, 3)
        image = Image.fromarray(nearest.query(rgb)[1].reshape(size, size).astype(np.uint8), "P")
        image.putpalette(colors.flatten().tolist())
        indexed.append(image)
    frames = indexed + indexed[-2:0:-1]
    durations = [GIF_FRAME_MS] * len(frames)
    durations[0] = durations[GIF_FRAMES - 1] = GIF_HOLD_MS
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )


def predict(model, cfg: DictConfig, object_dir: Path, output_dir: Path) -> dict:
    data = np.load(object_dir / "reconstruction.npz")
    states = np.split(data["points"].astype(np.float32), data["state_offsets"][1:-1])

    partfield_points, points, center, scale = model_inputs(states, cfg)
    output = run_model(model, partfield_points, points)
    labels, joints = decode(output, points, cfg.model.min_part_confidence)

    # Back to the frame of the input point clouds.
    points = points * scale + center
    for joint in joints:
        joint["origin"] = joint["origin"] * scale + center

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = {
        "num_states": len(states),
        "parts": [dict(j, axis=j["axis"].tolist(), origin=j["origin"].tolist()) for j in joints],
    }
    (output_dir / "prediction.json").write_text(json.dumps(prediction, indent=2))
    np.savez_compressed(output_dir / "prediction.npz", points=points, labels=labels)

    view_path = object_dir / "view.json"
    view = json.loads(view_path.read_text()) if view_path.exists() else DEFAULT_VIEW
    state = view.get("state", 0)
    visible_joints = [j for j in joints if (labels[state] == j["part_id"]).any()]
    render_prediction(points[state], labels[state], visible_joints, view).save(
        output_dir / "prediction.png"
    )
    save_gif(output_dir / "prediction.gif", points, labels, joints, view, state)
    return prediction


@hydra.main(version_base="1.3", config_path="./config", config_name="predict")
def main(cfg: DictConfig):
    model = setup(cfg)
    input_dir, output_dir = Path(cfg.input_dir), Path(cfg.output_dir)
    # Demo results keep the layout of demo/, however much of it is run.
    in_demo = DEMO_DIR.resolve() in (input_dir.resolve(), *input_dir.resolve().parents)
    root = DEMO_DIR if in_demo else input_dir
    if (root / "reconstruction.npz").exists():
        root = root.parent
    object_dirs = sorted(p.parent for p in input_dir.rglob("reconstruction.npz"))
    if not object_dirs:
        raise FileNotFoundError(f"No object folder with a reconstruction.npz in {input_dir}")
    for i, object_dir in enumerate(object_dirs, start=1):
        name = object_dir.resolve().relative_to(root.resolve())
        start = time.perf_counter()
        prediction = predict(model, cfg, object_dir, output_dir / name)
        parts = prediction["parts"]
        print(
            f"[{i}/{len(object_dirs)}] {name}: {prediction['num_states']} input states, "
            f"{len(parts)} movable part(s), {time.perf_counter() - start:.1f} s"
        )
        for part in parts:
            axis = ", ".join(f"{x:+.2f}" for x in part["axis"])
            print(f"    part {part['part_id']}: {part['type']:9s} axis ({axis})")
        print(f"    saved to {output_dir / name}/ (prediction.png, prediction.gif, ...)")
    print(f"\nResults written under {output_dir}\n")


if __name__ == "__main__":
    main()
