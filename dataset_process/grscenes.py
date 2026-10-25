"""GRScenes-100 (home scenes) -> grscenes_{train,val}.h5.

Reads the raw articulated USD assets (`<input_dir>/<category>/<model_id>/instance.usd`).
Links connected by fixed joints, and parts smaller than 10 cm, are merged into rigid
parts. Flipped joint axes are corrected and prismatic travel is capped at 85% of the
part depth.
"""

import os
from typing import Any, Optional, Tuple, Union

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from common import build_release, parse_args

# Joint types that may move, per category. Other joints are treated as fixed.
CATEGORY_JOINT_TYPES = {
    "chestofdrawers": ["prismatic", "revolute"],
    "desk": ["prismatic", "revolute"],
    "dishwasher": ["revolute"],
    "microwave": ["revolute"],
    "nightstand": ["prismatic", "revolute"],
    "oven": ["revolute"],
    "refrigerator": ["revolute"],
    "shoecabinet": ["prismatic", "revolute"],
    "sideboardcabinet": ["prismatic", "revolute"],
    "table": ["prismatic", "revolute"],
    "teatable": ["prismatic", "revolute"],
    "toilet": ["revolute"],
    "trashcan": ["revolute"],
    "tvstand": ["prismatic", "revolute"],
    "washingmachine": ["revolute"],
}
MIN_PART_SIZE_METERS = 0.1
NUM_SURFACE_POINTS = 100_000


def _look_at_camera_pose(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target are identical")
    forward = forward / forward_norm

    up = up / (np.linalg.norm(up) + 1e-12)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-12)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_pos
    return pose


def _count_visible_moving_part_hits(
    points: np.ndarray,
    part_ids: np.ndarray,
    camera_pose: np.ndarray,
    resolution: int = 512,
    fov_deg: float = 60.0,
) -> int:
    if len(points) == 0:
        return 0

    r_cw = camera_pose[:3, :3]
    t_cw = camera_pose[:3, 3]
    points_cam = (points - t_cw) @ r_cw

    depth = -points_cam[:, 2]
    valid = depth > 1e-6
    if not np.any(valid):
        return 0

    tan_half_fov = np.tan(np.deg2rad(float(fov_deg)) / 2.0)
    x_ndc = points_cam[:, 0] / (depth * tan_half_fov + 1e-12)
    y_ndc = points_cam[:, 1] / (depth * tan_half_fov + 1e-12)
    valid &= np.abs(x_ndc) <= 1.0
    valid &= np.abs(y_ndc) <= 1.0
    if not np.any(valid):
        return 0

    valid_idx = np.flatnonzero(valid)
    depth_valid = depth[valid]
    x_valid = x_ndc[valid]
    y_valid = y_ndc[valid]

    px = np.clip(((x_valid + 1.0) * 0.5 * (resolution - 1)).astype(np.int32), 0, resolution - 1)
    py = np.clip(((1.0 - y_valid) * 0.5 * (resolution - 1)).astype(np.int32), 0, resolution - 1)
    pixel_ids = py * resolution + px

    order = np.argsort(depth_valid)
    sorted_pixels = pixel_ids[order]
    _, first_hits = np.unique(sorted_pixels, return_index=True)
    visible_point_idx = valid_idx[order[first_hits]]

    return int(np.sum(part_ids[visible_point_idx] > 0))


def detect_canonical_front_yaw(rest_pts: np.ndarray, point_ids: np.ndarray) -> float:
    """Yaw (0/90/180/270 deg) of the side view that sees the most moving-part points."""
    if len(rest_pts) == 0:
        return 0.0

    bbox_min = rest_pts.min(axis=0)
    bbox_max = rest_pts.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    max_extent = max(float(np.max(bbox_max - bbox_min)), 1e-3)
    camera_dist = max_extent * 1.5
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    view_results = []
    for yaw_deg in [0.0, 90.0, 180.0, 270.0]:
        yaw_rad = np.deg2rad(yaw_deg)
        camera_pos = bbox_center + camera_dist * np.array(
            [np.cos(yaw_rad), np.sin(yaw_rad), 0.0],
            dtype=np.float64,
        )
        camera_pose = _look_at_camera_pose(camera_pos, bbox_center, up)
        view_results.append((yaw_deg, _count_visible_moving_part_hits(rest_pts, point_ids, camera_pose)))

    return float(sorted(view_results, key=lambda item: (item[1], -item[0]), reverse=True)[0][0])


class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if rx > ry:
            rx, ry = ry, rx
        self.parent[ry] = rx


def _apply_transform(matrix_4x4: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_homogenous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out_4 = points_homogenous @ matrix_4x4.T
    return out_4[:, :3] / out_4[:, 3:]


def _axis_token_to_vec(token: Any) -> np.ndarray:
    if token in (UsdPhysics.Tokens.x, 'X', 'x'):
        return np.array([1.0, 0.0, 0.0])
    if token in (UsdPhysics.Tokens.y, 'Y', 'y'):
        return np.array([0.0, 1.0, 0.0])
    if token in (UsdPhysics.Tokens.z, 'Z', 'z'):
        return np.array([0.0, 0.0, 1.0])
    try:
        v = np.array(token, dtype=float).reshape(3)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
    except Exception:
        pass
    return np.array([1.0, 0.0, 0.0])


def _get_attr(prim: Usd.Prim, name: str, default: Any = None) -> Any:
    attr = prim.GetAttribute(name)
    if not attr:
        return default
    val = attr.Get()
    return val if val is not None else default


def _quat_to_mat(quaternion: Optional[Union[Gf.Quatf, Gf.Quatd]]) -> np.ndarray:
    if quaternion is None:
        return np.eye(3)
    try:
        real = float(quaternion.GetReal())
        x, y, z = list(quaternion.GetImaginary())
        w = real
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)
    except Exception:
        return np.eye(3)


def safe_float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return default
        value = value[0]
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return default
        return float_val
    except (TypeError, ValueError):
        return default


def compute_joint_axis_anchor(xf_cache: UsdGeom.XformCache,
                              parent_prim: Usd.Prim,
                              joint_prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    """World-space axis direction and anchor point of a joint, from its body-0 frame."""
    local_pos0 = _get_attr(joint_prim, 'physics:localPos0', Gf.Vec3d(0, 0, 0))
    local_rot0 = _get_attr(joint_prim, 'physics:localRot0', None)
    R0 = _quat_to_mat(local_rot0)

    axis_attr = joint_prim.GetAttribute('physics:axis')
    axis_val = axis_attr.Get() if axis_attr else UsdPhysics.Tokens.z
    axis_local = _axis_token_to_vec(axis_val)

    parent_to_world = np.array(xf_cache.GetLocalToWorldTransform(parent_prim), dtype=float).T

    pos0_array = np.array([[local_pos0[0], local_pos0[1], local_pos0[2]]], dtype=float)
    anchor_world = _apply_transform(parent_to_world, pos0_array)[0]

    R_parent = parent_to_world[:3, :3]
    axis_world = (R_parent @ (R0 @ axis_local.reshape(3, 1))).reshape(3)

    norm = np.linalg.norm(axis_world)
    if norm > 0:
        axis_world = axis_world / norm
    else:
        axis_world = np.array([0.0, 0.0, 1.0], float)

    return axis_world, anchor_world


class USDPointProcessor:
    def __init__(self, usd_path, allowed_joint_types, seed=0):
        self.target_num_points = NUM_SURFACE_POINTS
        self.allowed_joint_types = allowed_joint_types
        self.rng = np.random.default_rng(seed)

        self.stage = Usd.Stage.Open(usd_path)
        if not self.stage:
            raise ValueError(f"Failed to open USD: {usd_path}")

        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        self.size_threshold = MIN_PART_SIZE_METERS / self.meters_per_unit

        self.xf_cache = UsdGeom.XformCache()
        self.bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        self.joints = {}
        self.descendants_map = {}

        self._parse_usd()

    def _parse_usd(self):
        raw_joints = []
        links_set = set()

        for prim in self.stage.TraverseAll():
            if not prim.IsA(UsdPhysics.Joint):
                continue

            jtype = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                usd_joint = UsdPhysics.RevoluteJoint(prim)
                jtype = 'revolute'
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                usd_joint = UsdPhysics.PrismaticJoint(prim)
                jtype = 'prismatic'
            elif prim.IsA(UsdPhysics.FixedJoint):
                usd_joint = UsdPhysics.FixedJoint(prim)
                jtype = 'fixed'
            else:
                continue

            rel0 = usd_joint.GetBody0Rel()
            rel1 = usd_joint.GetBody1Rel()
            targets0 = rel0.GetTargets() if rel0 else []
            targets1 = rel1.GetTargets() if rel1 else []

            if targets0:
                links_set.add(targets0[0].pathString)
            if targets1:
                links_set.add(targets1[0].pathString)
            if not (targets0 and targets1):
                continue

            parent_path = targets0[0].pathString
            child_path = targets1[0].pathString

            if jtype != 'fixed':
                if jtype not in self.allowed_joint_types:
                    jtype = 'fixed'
                else:
                    child_prim = self.stage.GetPrimAtPath(child_path)
                    if child_prim:
                        bound = self.bbox_cache.ComputeWorldBound(child_prim)
                        b_range = bound.ComputeAlignedRange()

                        if not b_range.IsEmpty():
                            b_min = np.array(b_range.GetMin())
                            b_max = np.array(b_range.GetMax())
                            max_dim = np.max(b_max - b_min)

                            if max_dim < self.size_threshold:
                                jtype = 'fixed'

            raw_joints.append({
                'prim': prim,
                'name': prim.GetName(),
                'type': jtype,
                'parent': parent_path,
                'child': child_path,
                'usd_joint': usd_joint
            })

        self.link_paths = sorted(links_set)

        self.uf = UnionFind(self.link_paths)
        for j in raw_joints:
            if j['type'] == 'fixed':
                self.uf.union(j['parent'], j['child'])

        merged_link_names = sorted(set(self.uf.find(name) for name in self.link_paths))
        self.merged_link_index_map = {name: i for i, name in enumerate(merged_link_names)}

        def get_merged_index(link_name: str) -> int:
            return self.merged_link_index_map[self.uf.find(link_name)]

        active_joints = [j for j in raw_joints if j['type'] != 'fixed']
        child_to_joint = {}
        joint_children = {}

        for j in active_joints:
            parent_idx = get_merged_index(j['parent'])
            child_idx = get_merged_index(j['child'])

            if parent_idx == child_idx:
                continue

            prim = j['prim']
            usd_joint = j['usd_joint']
            parent_prim = self.stage.GetPrimAtPath(j['parent'])

            axis_world, anchor_world = compute_joint_axis_anchor(self.xf_cache, parent_prim, prim)

            lo_attr = usd_joint.GetLowerLimitAttr().Get() if usd_joint.GetLowerLimitAttr() else None
            hi_attr = usd_joint.GetUpperLimitAttr().Get() if usd_joint.GetUpperLimitAttr() else None

            if j['type'] == 'revolute':
                lo = safe_float_or_default(lo_attr, -180.0) * np.pi / 180.0
                hi = safe_float_or_default(hi_attr, 180.0) * np.pi / 180.0
            else:
                lo = safe_float_or_default(lo_attr, 0.0)
                hi = safe_float_or_default(hi_attr, 0.0)

            if lo < -1e-6 and np.abs(hi) < 1e-6:
                lo, hi, axis_world = 0, -lo, -axis_world

            self.joints[j['name']] = {
                'type': j['type'],
                'child_part_id': child_idx,
                'origin_trans': anchor_world,
                'axis': axis_world,
                'limit': {'lower': lo, 'upper': hi}
            }

            child_to_joint[child_idx] = j['name']
            if j['name'] not in joint_children:
                joint_children[j['name']] = []

            for other_j in active_joints:
                if get_merged_index(other_j['child']) == parent_idx:
                    joint_children[other_j['name']].append(j['name'])

        for j_name in self.joints.keys():
            descendants = []
            queue = joint_children.get(j_name, [])[:]
            while queue:
                curr = queue.pop(0)
                descendants.append(curr)
                queue.extend(joint_children.get(curr, []))
            self.descendants_map[j_name] = list(set(descendants))

    def _fix_joint_directions(self):
        """Flip axes whose positive motion pushes the part into the base, and cap prismatic travel.

        An axis is flipped if moving forward brings more base points inside the moving part's
        bounding box than moving backward (and than at rest).
        """
        if len(self.rest_points) == 0:
            return

        for j_name, j_info in self.joints.items():
            moving_part_ids = [j_info['child_part_id']]
            for d in self.descendants_map.get(j_name, []):
                if d in self.joints:
                    moving_part_ids.append(self.joints[d]['child_part_id'])

            moving_mask = np.isin(self.point_ids, moving_part_ids)
            base_mask = self.point_ids == 0

            if not np.any(moving_mask) or not np.any(base_mask):
                continue

            base_pts = self.rest_points[base_mask]
            moving_pts = self.rest_points[moving_mask]

            # Bounding box of the moving part
            m_center = moving_pts.mean(axis=0)
            m_extents = (moving_pts.max(axis=0) - moving_pts.min(axis=0)) / 2.0
            m_min = m_center - m_extents
            m_max = m_center + m_extents

            def count_static_inside(pts):
                return np.sum(np.all((pts >= m_min) & (pts <= m_max), axis=1))

            rest_collisions = count_static_inside(base_pts)

            limit_range = j_info['limit']['upper'] - j_info['limit']['lower']
            test_val = limit_range * 0.3 if limit_range != 0 else (0.5 if j_info['type'] == 'prismatic' else np.pi/4)

            if test_val > 0:
                base_pts_homog = np.hstack([base_pts, np.ones((base_pts.shape[0], 1))])

                tf_fwd = np.eye(4)
                if j_info['type'] == 'revolute':
                    tf_fwd = trimesh.transformations.rotation_matrix(test_val, j_info['axis'], j_info['origin_trans'])
                else:
                    tf_fwd[:3, 3] = j_info['axis'] * test_val

                tf_fwd_inv = np.linalg.inv(tf_fwd)
                fwd_collisions = count_static_inside((tf_fwd_inv @ base_pts_homog.T).T[:, :3])

                tf_bwd = np.eye(4)
                if j_info['type'] == 'revolute':
                    tf_bwd = trimesh.transformations.rotation_matrix(-test_val, j_info['axis'], j_info['origin_trans'])
                else:
                    tf_bwd[:3, 3] = j_info['axis'] * -test_val

                tf_bwd_inv = np.linalg.inv(tf_bwd)
                bwd_collisions = count_static_inside((tf_bwd_inv @ base_pts_homog.T).T[:, :3])

                if fwd_collisions > rest_collisions and bwd_collisions < fwd_collisions:
                    self.joints[j_name]['axis'] *= -1.0

            # Clamp prismatic travel to 85% of the part's depth along the axis
            if j_info['type'] == 'prismatic':
                current_axis = self.joints[j_name]['axis']
                projections = np.dot(moving_pts, current_axis)
                safe_travel = (projections.max() - projections.min()) * 0.85

                current_lower = self.joints[j_name]['limit']['lower']
                current_upper = self.joints[j_name]['limit']['upper']
                if current_upper - current_lower > safe_travel and safe_travel > 0.01:
                    self.joints[j_name]['limit']['upper'] = current_lower + safe_travel

    def process_and_sample(self):
        part_meshes_lists = {}

        for link_path in self.link_paths:
            merged_name = self.uf.find(link_path)
            part_id = self.merged_link_index_map[merged_name]

            prim = self.stage.GetPrimAtPath(link_path)
            if not prim:
                continue

            for descendant in Usd.PrimRange(prim):
                if descendant.IsA(UsdGeom.Mesh):
                    mesh_prim = UsdGeom.Mesh(descendant)
                    points = mesh_prim.GetPointsAttr().Get()
                    if not points:
                        continue

                    xf = self.xf_cache.GetLocalToWorldTransform(descendant)
                    world_points = np.array([xf.Transform(pt) for pt in points], dtype=float)

                    face_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
                    face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()

                    if not face_counts or not face_indices:
                        continue

                    triangles = []
                    idx = 0
                    for count in face_counts:
                        for i in range(1, count - 1):
                            triangles.append([face_indices[idx], face_indices[idx+i], face_indices[idx+i+1]])
                        idx += count

                    if triangles:
                        trimesh_obj = trimesh.Trimesh(vertices=world_points, faces=np.array(triangles))
                        if part_id not in part_meshes_lists:
                            part_meshes_lists[part_id] = []
                        part_meshes_lists[part_id].append(trimesh_obj)

        part_meshes = {}
        for part_id, mesh_list in part_meshes_lists.items():
            if mesh_list:
                part_meshes[part_id] = trimesh.util.concatenate(mesh_list)

        total_area = sum(m.area for m in part_meshes.values())
        all_points, all_ids = [], []
        all_faces, all_face_ids = [], []
        all_verts = []
        vert_offset = 0

        points_sampled = 0
        part_items = list(part_meshes.items())

        for i, (part_id, mesh) in enumerate(part_items):
            if mesh.is_empty or total_area == 0:
                continue
            faces = mesh.faces.astype(np.int32)
            all_faces.append((faces + vert_offset).astype(np.int32))
            all_face_ids.append(np.full((len(faces),), part_id, dtype=np.int32))
            all_verts.append(mesh.vertices.astype(np.float32))
            vert_offset += len(mesh.vertices)

            if i == len(part_items) - 1:
                num_samples = self.target_num_points - points_sampled
            else:
                num_samples = int((mesh.area / total_area) * self.target_num_points)

            points_sampled += num_samples

            if num_samples > 0:
                points, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=self.rng)
                all_points.append(points)
                all_ids.append(np.full((num_samples, 1), part_id))

        self.rest_points = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float32)
        self.point_ids = np.vstack(all_ids).flatten() if all_ids else np.empty((0,), dtype=np.int32)
        self.faces = np.concatenate(all_faces, axis=0)
        self.face_ids = np.concatenate(all_face_ids, axis=0)
        self.rest_vertices = np.concatenate(all_verts, axis=0)

        self._fix_joint_directions()

    def get_kinematics_metadata(self):
        kinematics = {}
        for joint_name, j_data in self.joints.items():
            controlled_ids = [j_data['child_part_id']]
            for d_joint in self.descendants_map.get(joint_name, []):
                controlled_ids.append(self.joints[d_joint]['child_part_id'])

            kinematics[joint_name] = {
                "type": j_data['type'],
                "axis": j_data['axis'].tolist(),
                "origin": j_data['origin_trans'].tolist(),
                "limits": [j_data['limit']['lower'], j_data['limit']['upper']],
                "controlled_part_ids": list(set(controlled_ids))
            }
        return kinematics


if __name__ == "__main__":
    args = parse_args("grscenes", "GRScenes-100/home_scenes/models/object/articulated")

    def process(key):
        category, model_id = key.split("/")
        processor = USDPointProcessor(
            os.path.join(args.input_dir, category, model_id, "instance.usd"),
            allowed_joint_types=CATEGORY_JOINT_TYPES[category],
        )
        processor.process_and_sample()
        return {
            "faces": processor.faces,
            "face_part_ids": processor.face_ids,
            "rest_vertices": processor.rest_vertices,
            "kinematics": processor.get_kinematics_metadata(),
            # GRScenes assets have no canonical front: use the side that shows the most moving parts.
            "canonical_yaw_offset": detect_canonical_front_yaw(processor.rest_points, processor.point_ids),
        }

    build_release("grscenes", args, process)
