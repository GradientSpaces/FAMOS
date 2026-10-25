"""Shared observation pipeline: render articulated meshes into partial point clouds."""

import contextlib
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .rendering import (
    FaceIdRenderer,
    articulate_vertices,
    camera_pose_from_offsets,
    precompute_joint_vertex_ids,
    sample_or_pad_indices,
)
from .sampling import BATCH_SEED_KEY, rng_context

logger = logging.getLogger("Data")
logging.getLogger("trimesh").setLevel(logging.ERROR)

CAMERA_MODES = ("drift", "independent_uniform")
MAX_VIEW_RETRIES = 10
# A joint is supervised only if at least this many of its points are observed.
MIN_VALID_JOINT_POINTS = 5
# Y-up data are rendered Z-up: (x, y, z) -> (x, -z, y).
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class ObservationDataset(Dataset):
    """Observes an articulated mesh through a virtual depth camera, once per state.

    For each joint state the mesh is articulated, rendered from a camera in front of the
    object, and the visible pixels are backprojected into a partial point cloud labeled
    with joint slots (0 = static base, i = i-th joint). Everything is expressed in the
    frame of the first camera. Subclasses load the meshes and call `_observe`.

    Camera: each state is viewed from a yaw and pitch in `camera_yaw_range` and
    `camera_pitch_range` around the object's front. Per object, `camera_mode_probs`
    picks `independent_uniform` (a new viewpoint per state) or `drift` (each viewpoint
    moves from the previous one by up to `camera_*_drift_range`). A view is re-drawn (up
    to `MAX_VIEW_RETRIES` times) until every moving part is visible.

    Sensor augmentation (training only, per object with `sensor_aug_prob`) backprojects
    with a perturbed pose and intrinsics, adds depth noise, drops points, and adds
    outliers, which are labeled -1 and ignored by the losses.
    """

    def __init__(
        self,
        split: str,
        dataset_name: str,
        num_points: int,
        up_axis: str,
        max_states: int,
        num_partfield_points: int,
        render_resolution: int,
        camera_distance: float,
        camera_fov_deg: float,
        camera_mode_probs: dict[str, float],
        camera_yaw_range: tuple[float, float],
        camera_pitch_range: tuple[float, float],
        camera_yaw_drift_range: tuple[float, float],
        camera_pitch_drift_range: tuple[float, float],
        sensor_aug_prob: float = 0.0,
        sensor_aug_point_keep_range: tuple[float, float] = (1.0, 1.0),
        sensor_aug_depth_std: float = 0.0,
        sensor_aug_depth_rel_std: float = 0.0,
        sensor_aug_depth_max: float = 0.0,
        sensor_aug_focal_scale_std: float = 0.0,
        sensor_aug_focal_scale_max: float = 0.0,
        sensor_aug_principal_point_std_px: float = 0.0,
        sensor_aug_principal_point_max_px: float = 0.0,
        sensor_aug_pose_rot_std_deg: float = 0.0,
        sensor_aug_pose_rot_max_deg: float = 0.0,
        sensor_aug_pose_trans_std: float = 0.0,
        sensor_aug_pose_trans_max: float = 0.0,
        sensor_aug_outlier_ratio: float = 0.0,
        sensor_aug_outlier_near_surface_prob: float = 0.0,
        sensor_aug_outlier_near_surface_std: float = 0.0,
        sensor_aug_outlier_bbox_scale: float = 1.0,
    ):
        super().__init__()
        if up_axis not in ("y", "z"):
            raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")
        self.split = split
        self.dataset_name = dataset_name
        self.num_points = int(num_points)
        self.up_axis = up_axis
        self.max_states = int(max_states)
        self.fixed_states = None  # {key: {joint: [q_0, q_1, ...]}} outside training

        self.points_all = int(num_partfield_points)
        self.resolution = int(render_resolution)
        self.camera_dist = float(camera_distance)
        self.fov_deg = float(camera_fov_deg)
        probs = {m: float(w) for m, w in camera_mode_probs.items()}
        if any(m not in CAMERA_MODES for m in probs) or sum(probs.values()) <= 0.0:
            raise ValueError(f"Invalid camera_mode_probs={probs}")
        self.camera_modes = [m for m, w in probs.items() if w > 0.0]
        self.camera_mode_weights = [w for w in probs.values() if w > 0.0]
        self.yaw_range = tuple(float(x) for x in camera_yaw_range)
        self.pitch_range = tuple(float(x) for x in camera_pitch_range)
        self.yaw_drift_range = tuple(float(x) for x in camera_yaw_drift_range)
        self.pitch_drift_range = tuple(float(x) for x in camera_pitch_drift_range)

        self.sensor_aug_prob = float(sensor_aug_prob)
        self.aug_point_keep_range = tuple(float(x) for x in sensor_aug_point_keep_range)
        self.aug_depth_std = float(sensor_aug_depth_std)
        self.aug_depth_rel_std = float(sensor_aug_depth_rel_std)
        self.aug_depth_max = float(sensor_aug_depth_max)
        self.aug_focal_scale_std = float(sensor_aug_focal_scale_std)
        self.aug_focal_scale_max = float(sensor_aug_focal_scale_max)
        self.aug_principal_point_std_px = float(sensor_aug_principal_point_std_px)
        self.aug_principal_point_max_px = float(sensor_aug_principal_point_max_px)
        self.aug_pose_rot_std_deg = float(sensor_aug_pose_rot_std_deg)
        self.aug_pose_rot_max_deg = float(sensor_aug_pose_rot_max_deg)
        self.aug_pose_trans_std = float(sensor_aug_pose_trans_std)
        self.aug_pose_trans_max = float(sensor_aug_pose_trans_max)
        self.aug_outlier_ratio = float(sensor_aug_outlier_ratio)
        self.aug_outlier_near_surface_prob = float(sensor_aug_outlier_near_surface_prob)
        self.aug_outlier_near_surface_std = float(sensor_aug_outlier_near_surface_std)
        self.aug_outlier_bbox_scale = float(sensor_aug_outlier_bbox_scale)

    def _joint_states(self, key: str, kinematics: dict) -> dict[str, list[float]]:
        """Fixed states outside training; otherwise one random value per equal bin."""
        if self.fixed_states is not None:
            states = {}
            for joint_name in kinematics:
                values = self.fixed_states[key][joint_name]
                if len(values) < self.max_states:
                    raise ValueError(
                        f"{key}/{joint_name}: {len(values)} fixed states stored, "
                        f"but max_states={self.max_states}."
                    )
                states[joint_name] = [float(v) for v in values[: self.max_states]]
            return states
        return {
            joint_name: self._sample_n_states(*joint["limits"], self.max_states)
            for joint_name, joint in kinematics.items()
        }

    @staticmethod
    def _sample_n_states(lo: float, hi: float, n_states: int) -> list[float]:
        """One uniform value from each of `n_states` equal bins, in shuffled order."""
        lo, hi = sorted((float(lo), float(hi)))
        if abs(hi - lo) < 1e-8:
            vals = [lo] * n_states
        else:
            edges = np.linspace(lo, hi, n_states + 1, dtype=np.float64)
            vals = []
            for i in range(n_states):
                a, b = float(edges[i]), float(edges[i + 1])
                # Half-open bins, except the last one, which includes `hi`.
                vals.append(
                    float(
                        random.uniform(
                            a, b if i == n_states - 1 else np.nextafter(b, a)
                        )
                    )
                )
        random.shuffle(vals)
        return vals

    @staticmethod
    def _rng(base_seed: int | None, offset: int = 0):
        """Seeded RNG for evaluation and procedural objects; the global one otherwise."""
        return random if base_seed is None else random.Random(base_seed + offset)

    def _choose_camera_mode(self, base_seed: int | None) -> str:
        rng = self._rng(base_seed, 104729)
        return rng.choices(self.camera_modes, weights=self.camera_mode_weights, k=1)[0]

    @staticmethod
    def _offsets_from_yaw_pitch(yaw: float, pitch: float) -> tuple[float, float, float]:
        yaw_rad, pitch_rad = np.deg2rad(float(yaw)), np.deg2rad(float(pitch))
        return (
            float(np.cos(pitch_rad) * np.sin(yaw_rad)),
            float(-np.cos(pitch_rad) * np.cos(yaw_rad)),
            float(np.sin(pitch_rad)),
        )

    def _clamp_view(
        self, yaw: float, pitch: float, pitch_range
    ) -> tuple[float, float]:
        yaw = max(self.yaw_range[0], min(self.yaw_range[1], yaw))
        pitch = max(pitch_range[0], min(pitch_range[1], pitch))
        return float(yaw), float(pitch)

    def _view_offsets(
        self, camera_mode, state_idx, retry, base_seed, base_yaw, base_pitch, pitch_range
    ) -> tuple[float, float, float]:
        """Camera direction for one attempt at observing state `state_idx`."""
        rng = self._rng(base_seed, state_idx * 10007 + retry)
        if camera_mode == "drift":
            yaw, pitch = base_yaw, base_pitch
            if retry > 0:
                yaw += rng.uniform(*self.yaw_drift_range)
                pitch += rng.uniform(*self.pitch_drift_range)
            return self._offsets_from_yaw_pitch(
                *self._clamp_view(yaw, pitch, pitch_range)
            )
        yaw = float(rng.uniform(*self.yaw_range))
        pitch = float(rng.uniform(*pitch_range))
        return self._offsets_from_yaw_pitch(yaw, pitch)

    def _sensor_aug_active(self) -> bool:
        return self.split == "train" and random.random() < self.sensor_aug_prob

    @staticmethod
    def _clipped_normal(std: float, max_abs: float) -> float:
        if std <= 0.0:
            return 0.0
        val = float(np.random.normal(0.0, std))
        return max(-max_abs, min(max_abs, val)) if max_abs > 0.0 else val

    @staticmethod
    def _clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if max_norm > 0.0 and norm > max_norm and norm > 1e-12:
            return vec * (max_norm / norm)
        return vec

    def _backprojection_params(self, camera_pose: np.ndarray, active: bool) -> dict:
        """Pose, intrinsics and depth noise used to lift the rendered pixels."""
        if not active:
            return {"backproject_pose": camera_pose}

        pose = np.asarray(camera_pose, dtype=np.float64).copy()
        if self.aug_pose_rot_std_deg > 0.0:
            rot_deg = np.random.normal(0.0, self.aug_pose_rot_std_deg, size=3).astype(
                np.float64
            )
            rot_deg = self._clip_norm(rot_deg, self.aug_pose_rot_max_deg)
            pose[:3, :3] = (
                pose[:3, :3] @ Rotation.from_rotvec(np.deg2rad(rot_deg)).as_matrix()
            )
        if self.aug_pose_trans_std > 0.0:
            trans = np.random.normal(0.0, self.aug_pose_trans_std, size=3).astype(
                np.float64
            )
            trans = self._clip_norm(trans, self.aug_pose_trans_max)
            pose[:3, 3] = pose[:3, 3] + pose[:3, :3] @ trans

        focal_delta = self._clipped_normal(
            self.aug_focal_scale_std, self.aug_focal_scale_max
        )
        pp_x = self._clipped_normal(
            self.aug_principal_point_std_px, self.aug_principal_point_max_px
        )
        pp_y = self._clipped_normal(
            self.aug_principal_point_std_px, self.aug_principal_point_max_px
        )
        return {
            "backproject_pose": pose,
            "focal_scale": max(1e-6, 1.0 + focal_delta),
            "principal_point_px": (pp_x, pp_y),
            "depth_noise_std": self.aug_depth_std,
            "depth_noise_rel_std": self.aug_depth_rel_std,
            "depth_noise_max": self.aug_depth_max,
        }

    def _augment_points(self, points, labels, active):
        """Randomly drop points and add near-surface and uniform outliers (label -1)."""
        if not active or len(points) == 0:
            return points, labels

        keep_n = round(len(points) * random.uniform(*self.aug_point_keep_range))
        keep_n = max(1, min(len(points), keep_n))
        if keep_n < len(points):
            keep_idx = np.random.permutation(len(points))[:keep_n]
            points, labels = points[keep_idx], labels[keep_idx]

        n_outliers = round(len(points) * self.aug_outlier_ratio)
        if n_outliers <= 0:
            return points, labels
        n_near = round(n_outliers * self.aug_outlier_near_surface_prob)
        n_uniform = n_outliers - n_near
        outliers = []
        if n_near > 0:
            base_idx = np.random.choice(len(points), size=n_near, replace=True)
            noise = np.random.normal(
                0.0, self.aug_outlier_near_surface_std, size=(n_near, 3)
            )
            outliers.append(
                (points[base_idx] + noise.astype(np.float32)).astype(
                    points.dtype, copy=False
                )
            )
        if n_uniform > 0:
            bb_min, bb_max = points.min(axis=0), points.max(axis=0)
            center = (bb_min + bb_max) * 0.5
            half_extent = np.maximum(
                (bb_max - bb_min) * 0.5 * self.aug_outlier_bbox_scale, 1e-3
            )
            outliers.append(
                np.random.uniform(
                    low=center - half_extent,
                    high=center + half_extent,
                    size=(n_uniform, 3),
                ).astype(points.dtype, copy=False)
            )
        outliers = np.concatenate(outliers, axis=0)
        points = np.concatenate([points, outliers], axis=0)
        labels = np.concatenate(
            [labels, np.full(len(outliers), -1, dtype=labels.dtype)], axis=0
        )
        return points, labels

    def _observe(
        self,
        *,
        key: str,
        object_id: str,
        rest_vertices: np.ndarray,
        faces: np.ndarray,
        face_part_ids: np.ndarray,
        kinematics: dict,
        rest_center: np.ndarray,
        rest_scale: float,
        canonical_yaw_offset: float,
        base_seed: int | None,
        pitch_range: tuple[float, float] | None = None,
    ) -> dict:
        """Render all states of one object and assemble the sample."""
        pitch_range = pitch_range or self.pitch_range
        joint_names = list(kinematics)
        # Parts moved by the i-th joint get slot i.
        part_to_slot = {
            int(part_id): slot
            for slot, joint_name in enumerate(joint_names, start=1)
            for part_id in kinematics[joint_name]["controlled_part_ids"]
            if int(part_id) > 0
        }
        active_slots = sorted(set(part_to_slot.values()))
        joint_vertex_ids = precompute_joint_vertex_ids(faces, face_part_ids, kinematics)

        def slot_labels(point_face_idx: np.ndarray) -> np.ndarray:
            raw = np.zeros(len(point_face_idx), dtype=np.int64)
            ok = (point_face_idx >= 0) & (point_face_idx < len(face_part_ids))
            raw[ok] = face_part_ids[point_face_idx[ok]].astype(np.int64)
            labels = np.zeros(raw.shape, dtype=np.int64)
            for part_id, slot in part_to_slot.items():
                labels[raw == part_id] = slot
            return labels

        camera_mode = self._choose_camera_mode(base_seed)
        joint_states = self._joint_states(key, kinematics)

        safe_scale = max(float(rest_scale), 1e-8)
        if self.up_axis == "y":
            center_render = np.array(
                [rest_center[0], -rest_center[2], rest_center[1]], dtype=np.float64
            )
        else:
            center_render = rest_center.astype(np.float64, copy=False)

        renderer = FaceIdRenderer(faces, self.resolution, self.fov_deg)
        sensor_aug_active = self._sensor_aug_active()
        rng_walk = self._rng(base_seed)
        current_yaw = rng_walk.uniform(*self.yaw_range)
        current_pitch = rng_walk.uniform(*pitch_range)

        clouds, cloud_labels, camera_poses = [], [], []
        for state_idx in range(self.max_states):
            values = {name: joint_states[name][state_idx] for name in joint_names}
            # Render in a normalized, Z-up frame.
            vertices = articulate_vertices(
                rest_vertices, kinematics, values, joint_vertex_ids
            )
            vertices = (vertices - rest_center) / safe_scale
            if self.up_axis == "y":
                vertices = vertices @ Y_UP_TO_Z_UP.T

            base_yaw, base_pitch = current_yaw, current_pitch
            if camera_mode == "drift" and state_idx > 0:
                rng_state = self._rng(base_seed, state_idx * 10007)
                base_yaw += rng_state.uniform(*self.yaw_drift_range)
                base_pitch += rng_state.uniform(*self.pitch_drift_range)
                base_yaw, base_pitch = self._clamp_view(
                    base_yaw, base_pitch, pitch_range
                )

            # Keep the first view in which every moving part is visible, else the view
            # whose least-visible moving part has the most points.
            best, best_score = None, -1
            with renderer.posed(vertices):
                for retry in range(MAX_VIEW_RETRIES):
                    offsets = self._view_offsets(
                        camera_mode,
                        state_idx,
                        retry,
                        base_seed,
                        base_yaw,
                        base_pitch,
                        pitch_range,
                    )
                    camera_pose = camera_pose_from_offsets(
                        self.camera_dist, offsets, canonical_yaw_offset
                    )
                    params = self._backprojection_params(camera_pose, sensor_aug_active)
                    points, point_face_idx = renderer.render(camera_pose, **params)

                    labels = slot_labels(point_face_idx)
                    counts = [int((labels == slot).sum()) for slot in active_slots]
                    score = min(counts) if counts else len(points)
                    min_required = max(1, min(20, int(len(points) * 0.002)))
                    # Camera pose in the raw object frame (Z-up).
                    raw_pose = np.asarray(
                        params["backproject_pose"], dtype=np.float64
                    ).copy()
                    raw_pose[:3, 3] = raw_pose[:3, 3] * safe_scale + center_render
                    view = (points, point_face_idx, offsets, raw_pose)
                    if score > best_score:
                        best, best_score = view, score
                    if all(count >= min_required for count in counts):
                        best = view
                        break
            points, point_face_idx, offsets, raw_pose = best

            if camera_mode == "drift":
                off = np.asarray(offsets, dtype=np.float64)
                off_norm = np.linalg.norm(off)
                if off_norm > 1e-12:
                    current_yaw = float(np.rad2deg(np.arctan2(off[0], -off[1])))
                    current_pitch = float(
                        np.rad2deg(np.arcsin(np.clip(off[2] / off_norm, -1.0, 1.0)))
                    )
            camera_poses.append(raw_pose)

            if len(points) == 0:  # nothing visible: fall back to the mesh vertices
                verts = np.asarray(vertices, dtype=np.float32)
                points = verts[sample_or_pad_indices(len(verts), self.points_all)]
                point_face_idx = np.full(len(points), -1, dtype=np.int32)

            labels = slot_labels(point_face_idx)
            points, labels = self._augment_points(points, labels, sensor_aug_active)
            # Back to the raw object frame and units.
            if self.up_axis == "y":
                points = points @ Y_UP_TO_Z_UP.astype(np.float32)
            points = points * safe_scale + rest_center

            sel = sample_or_pad_indices(len(points), self.points_all)
            clouds.append(points[sel].astype(np.float32))
            cloud_labels.append(labels[sel].astype(np.int64))

        sample = self._assemble_sample(
            object_id=object_id,
            kinematics=kinematics,
            joint_states=joint_states,
            point_clouds=torch.from_numpy(np.stack(clouds, axis=0)).float(),
            segmentation_labels=torch.from_numpy(np.stack(cloud_labels, axis=0)).long(),
            camera_poses=camera_poses,
        )
        # Yaw of the object's front, for the visualizations.
        sample["canonical_yaw_offset"] = float(canonical_yaw_offset)
        return sample

    def _assemble_sample(
        self,
        object_id,
        kinematics,
        joint_states,
        point_clouds,
        segmentation_labels,
        camera_poses,
    ) -> dict:
        """Express everything in the normalized model frame and build the targets."""
        joint_names = list(kinematics)
        if self.up_axis == "y":
            point_clouds = self._y_to_z(point_clouds.reshape(-1, 3)).reshape(
                point_clouds.shape
            )

        # Visualizations use the object-aligned frame. This affine maps normalized
        # model-frame points back to normalized object-aligned coordinates.
        _, vis_center, vis_scale = self._normalize_states(point_clouds)

        cam0 = np.asarray(camera_poses[0], dtype=np.float64).copy()
        first_camera_R = torch.from_numpy(cam0[:3, :3]).to(point_clouds.dtype)
        first_camera_t = torch.from_numpy(cam0[:3, 3]).to(point_clouds.dtype)
        model_to_canonical_dir = first_camera_R.T
        # Row-vector points: p_cam = (p_world - t) @ R; poses become inv(T0) @ Ti.
        point_clouds = (point_clouds - first_camera_t.view(1, 1, 3)) @ first_camera_R
        cam0_inv = np.eye(4, dtype=np.float64)
        cam0_inv[:3, :3] = cam0[:3, :3].T
        cam0_inv[:3, 3] = -cam0[:3, :3].T @ cam0[:3, 3]
        camera_poses = [_compose_se3(cam0_inv, pose) for pose in camera_poses]

        point_clouds, center, scale = self._normalize_states(point_clouds)
        model_to_canonical_matrix = (scale / vis_scale) * model_to_canonical_dir
        model_to_canonical_offset = (
            center @ model_to_canonical_dir + first_camera_t - vis_center
        ) / vis_scale
        camera_poses = [pose.copy() for pose in camera_poses]
        for pose in camera_poses:
            pose[:3, 3] = (
                (torch.from_numpy(pose[:3, 3]).float() - center) / scale
            ).numpy()

        subsampled = [
            self._stratified_subsample(point_clouds[s], segmentation_labels[s])
            for s in range(len(point_clouds))
        ]
        points_sub = torch.stack([pts for pts, _ in subsampled])
        labels_sub = torch.stack([lbl for _, lbl in subsampled])
        joint_valid_mask = torch.ones(len(joint_names) + 1, dtype=torch.bool)
        for slot in range(1, len(joint_names) + 1):
            joint_valid_mask[slot] = (
                labels_sub == slot
            ).sum().item() >= MIN_VALID_JOINT_POINTS

        axes, origins, pluckers, limits = [], [], [], []
        for joint_name in joint_names:
            joint = kinematics[joint_name]
            axis = torch.tensor(joint["axis"], dtype=torch.float32)
            origin = torch.tensor(joint["origin"], dtype=torch.float32)
            if self.up_axis == "y":
                axis, origin = self._y_to_z(axis), self._y_to_z(origin)
            origin = (origin - first_camera_t) @ first_camera_R
            axis = F.normalize(axis @ first_camera_R, dim=0, eps=1e-6)
            origin = (origin - center) / scale
            axis = F.normalize(axis, dim=0, eps=1e-6)
            limit_scale = scale if joint["type"] == "prismatic" else 1.0
            axes.append(axis)
            origins.append(origin)
            pluckers.append(torch.cat([axis, torch.cross(origin, axis, dim=0)], dim=0))
            limits.append(
                torch.tensor(joint["limits"], dtype=torch.float32) / limit_scale
            )

        return {
            "partfield_point_clouds": point_clouds,
            "point_clouds": points_sub,
            "segmentation_labels": labels_sub,
            "joint_values": torch.tensor(
                [
                    [joint_states[j][s] for j in joint_names]
                    for s in range(len(point_clouds))
                ],
                dtype=torch.float32,
            ),
            "dataset_name": self.dataset_name,
            "object_id": object_id,
            "center": center,
            "scale": scale,
            "joint_type": [kinematics[j]["type"] for j in joint_names],
            "joint_axis": torch.stack(axes),
            "joint_origin": torch.stack(origins),
            "gt_plucker": torch.stack(pluckers),
            "joint_limits": torch.stack(limits),
            "joint_valid_mask": joint_valid_mask,
            "camera_pose": torch.stack(
                [torch.from_numpy(p).float() for p in camera_poses]
            ),
            "vis_model_to_canonical_matrix": model_to_canonical_matrix.float(),
            "vis_model_to_canonical_offset": model_to_canonical_offset.float(),
            "vis_model_to_canonical_dir_matrix": model_to_canonical_dir.float(),
        }

    @staticmethod
    def _y_to_z(pts: torch.Tensor) -> torch.Tensor:
        """Y-up -> Z-up for (N, 3) points or a (3,) vector."""
        out = pts.clone()
        out[..., 1] = -pts[..., 2]
        out[..., 2] = pts[..., 1]
        return out

    @staticmethod
    def _normalize_states(points: torch.Tensor):
        """Center and scale all states jointly by their combined bounding box."""
        flat = points.reshape(-1, 3)
        g_min, g_max = flat.amin(dim=0), flat.amax(dim=0)
        center = (g_min + g_max) / 2.0
        scale = (g_max - g_min).max().clamp(min=1e-6)
        return (points - center) / scale, center, scale

    def _stratified_subsample(self, points: torch.Tensor, labels: torch.Tensor):
        """Subsample to `num_points`: one point per part, the rest proportional to part
        size, leftovers to the parts with the most unused points."""
        n, m = points.shape[0], self.num_points
        if m >= n:
            return points, labels

        label_idx = [
            torch.where(labels == lbl)[0]
            for lbl in torch.unique(labels, sorted=True).tolist()
        ]
        counts = torch.tensor([len(idx) for idx in label_idx], dtype=torch.long)
        alloc = torch.zeros(len(counts), dtype=torch.long)
        remaining = m
        if m >= len(counts):
            alloc += 1
            remaining -= len(counts)
        if remaining > 0:
            weights = counts.float() / counts.float().sum().clamp(min=1.0)
            proportional = torch.floor(weights * remaining).to(torch.long)
            proportional = torch.minimum(
                proportional, torch.clamp(counts - alloc, min=0)
            )
            alloc += proportional
            remaining -= int(proportional.sum().item())
        if remaining > 0:
            available = torch.clamp(counts - alloc, min=0)
            for pos in torch.argsort(available, descending=True).tolist():
                if remaining <= 0:
                    break
                take = min(int(available[pos].item()), remaining)
                alloc[pos] += take
                remaining -= take

        chunks = [
            idx[torch.randperm(len(idx))[:k]]
            for idx, k in zip(label_idx, alloc.tolist(), strict=True)
            if k > 0
        ]
        sel = torch.cat(chunks, dim=0)
        sel = sel[torch.randperm(len(sel))]
        return points[sel], labels[sel]


def _compose_se3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = a[:3, :3] @ b[:3, :3]
    out[:3, 3] = a[:3, :3] @ b[:3, 3] + a[:3, 3]
    return out



def collate_fn(
    batch: list[dict],
    max_parts: int,
    state_sampling_probs: list[float] | None = None,
    observed_articulation_span_min_visible_points: int = 5,
    train: bool = False,
) -> dict:
    """Stack samples and pad the per-joint targets to `max_parts` slots (slot 0 = base).

    Training batches keep a random number of states, drawn from `state_sampling_probs`;
    evaluation batches keep all of them.
    """
    B = len(batch)
    max_states = min(int(sample["point_clouds"].shape[0]) for sample in batch)
    if train:
        probs = [float(p) for p in state_sampling_probs[:max_states]]
        batch_seed = batch[0].get(BATCH_SEED_KEY)
        with (
            rng_context(batch_seed)
            if batch_seed is not None
            else contextlib.nullcontext()
        ):
            S = random.choices(
                population=list(range(1, max_states + 1)),
                weights=[p / sum(probs) for p in probs],
                k=1,
            )[0]
    else:
        S = max_states

    for sample in batch:
        if len(sample["joint_type"]) > max_parts - 1:
            raise ValueError(
                f"{sample['object_id']} has {len(sample['joint_type'])} joints but "
                f"max_parts={max_parts}. Increase max_parts/num_part_queries."
            )

    def stack(key):
        return torch.stack([sample[key] for sample in batch], dim=0)

    def pad_joints(key, shape):
        # Joint i of a sample goes to slot i + 1; slot 0 is the static base.
        out = torch.zeros((B, max_parts, *shape), dtype=batch[0][key].dtype)
        for b, sample in enumerate(batch):
            out[b, 1 : 1 + len(sample[key])] = sample[key]
        return out

    joint_values = torch.zeros((B, S, max_parts), dtype=batch[0]["joint_values"].dtype)
    joint_valid_mask = torch.zeros((B, max_parts), dtype=torch.bool)
    joint_valid_mask[:, 0] = True
    joint_type_ids = torch.zeros((B, max_parts), dtype=torch.long)
    num_valid_parts = torch.zeros(B, dtype=torch.long)
    for b, sample in enumerate(batch):
        n_joints = len(sample["joint_type"])
        joint_values[b, :, 1 : 1 + n_joints] = sample["joint_values"][:S]
        joint_valid_mask[b, : n_joints + 1] = sample["joint_valid_mask"]
        num_valid_parts[b] = n_joints + 1
        for slot, joint_type in enumerate(sample["joint_type"], start=1):
            joint_type_ids[b, slot] = {"revolute": 1, "prismatic": 2}.get(joint_type, 0)

    out = {
        "partfield_point_clouds": stack("partfield_point_clouds")[:, :S],
        "point_clouds": stack("point_clouds")[:, :S],
        "segmentation_labels": stack("segmentation_labels")[:, :S],
        "center": stack("center"),
        "scale": stack("scale"),
        "vis_model_to_canonical_matrix": stack("vis_model_to_canonical_matrix"),
        "vis_model_to_canonical_offset": stack("vis_model_to_canonical_offset"),
        "vis_model_to_canonical_dir_matrix": stack("vis_model_to_canonical_dir_matrix"),
        "canonical_yaw_offset": torch.tensor([sample["canonical_yaw_offset"] for sample in batch]),
        "joint_values": joint_values,
        "joint_valid_mask": joint_valid_mask,
        "joint_axis": pad_joints("joint_axis", (3,)),
        "joint_origin": pad_joints("joint_origin", (3,)),
        "gt_plucker": pad_joints("gt_plucker", (6,)),
        "joint_limits": pad_joints("joint_limits", (2,)),
        "joint_type_ids": joint_type_ids,
        "num_valid_parts": num_valid_parts,
        "joint_type": [sample["joint_type"] for sample in batch],
        "camera_pose": [sample["camera_pose"] for sample in batch],
        "dataset_name": [sample["dataset_name"] for sample in batch],
        "object_id": [sample["object_id"] for sample in batch],
    }
    out["observed_articulation_spans"], out["observed_articulation_span_valid"] = (
        _observed_articulation_spans(
            out["segmentation_labels"],
            joint_values,
            joint_type_ids,
            num_valid_parts,
            out["scale"],
            observed_articulation_span_min_visible_points,
        )
    )
    return out


def _observed_articulation_spans(
    labels, joint_values, joint_type_ids, num_valid_parts, scale, min_visible_points
):
    """Range of each joint's values over the states in which its part is visible.

    A part counts as visible in a state with at least `min_visible_points` points;
    spans need two visible states. Prismatic spans are in normalized units.
    """
    B, max_parts = joint_values.shape[0], joint_values.shape[2]
    spans = torch.zeros((B, max_parts), dtype=joint_values.dtype)
    valid = torch.zeros((B, max_parts), dtype=torch.bool)
    scale = scale.to(joint_values.dtype).reshape(B)
    for b in range(B):
        for part in range(1, min(max_parts, int(num_valid_parts[b].item()))):
            visible = (labels[b] == part).sum(dim=1) >= min_visible_points
            if int(visible.sum().item()) < 2:
                continue
            vals = joint_values[b, visible, part]
            span = vals.max() - vals.min()
            if int(joint_type_ids[b, part].item()) == 2:
                span = span / scale[b].clamp_min(1e-6)
            spans[b, part] = span
            valid[b, part] = True
    return spans, valid
