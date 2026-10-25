"""Procedural articulated assets built in memory for training.

Objects come from the builders in ``generate_procedural_data/generate.py`` and are
converted to the mesh and kinematics format of the release HDF5 files, without
textures or file export.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from generate_procedural_data import generate as gen

MAX_RETRIES = 128


@dataclass(frozen=True)
class ProceduralAssetBundle:
    model_id: str
    family: str
    seed: int
    rest_vertices: np.ndarray
    faces: np.ndarray
    face_part_ids: np.ndarray
    kinematics: dict[str, dict[str, Any]]
    canonical_yaw_offset: float
    rest_center: np.ndarray
    rest_scale: float


def available_families() -> list[str]:
    return list(gen.BUILDERS.keys())


def _metadata_has_side_mode(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                str(key) == "mode"
                and isinstance(child, str)
                and child.startswith("side_")
            ):
                return True
            if _metadata_has_side_mode(child):
                return True
    elif isinstance(value, list):
        return any(_metadata_has_side_mode(child) for child in value)
    return False


def _build_urdf(
    family: str, seed: int, *, forbid_cabinet_side_doors: bool
) -> tuple[str, dict[str, Any]]:
    if family not in gen.BUILDERS:
        raise ValueError(
            f"Unknown procedural family={family!r}. Available: {', '.join(available_families())}"
        )

    original_modes = gen._door_modes_for_pos

    def front_only_modes(pos, bay_count, forbid_open_up=False):
        modes = original_modes(pos, bay_count, forbid_open_up=forbid_open_up)
        return [mode for mode in modes if not str(mode).startswith("side_")]

    gen.FAMOS_TRAINING = True
    if forbid_cabinet_side_doors:
        gen._door_modes_for_pos = front_only_modes
    try:
        return gen.BUILDERS[family](seed)
    finally:
        gen.FAMOS_TRAINING = False
        gen._door_modes_for_pos = original_modes


def _joint_limit(joint: ET.Element) -> tuple[float, float]:
    limit = joint.find("limit")
    if limit is None:
        return 0.0, 1.0
    lower = float(limit.attrib.get("lower", "0.0"))
    upper = float(limit.attrib.get("upper", "1.0"))
    if upper < lower:
        lower, upper = upper, lower
    return lower, upper


def _joint_axis(joint: ET.Element) -> np.ndarray:
    axis_elem = joint.find("axis")
    if axis_elem is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = np.array(
            gen._parse_vec3(axis_elem.attrib.get("xyz", "0 0 1")), dtype=np.float64
        )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _children_by_parent(
    joints: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = {}
    for joint in joints.values():
        children.setdefault(joint["parent"], []).append(joint)
    for child_list in children.values():
        child_list.sort(key=lambda item: item["elem"].attrib["name"])
    return children


def _assign_link_parts(
    root_link: str,
    joints: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    children = _children_by_parent(joints)
    link_part_ids = {root_link: 0}
    joint_part_ids: dict[str, int] = {}
    joint_order: list[str] = []
    next_part_id = 1

    stack: list[tuple[str, int]] = [(root_link, 0)]
    while stack:
        link_name, current_part_id = stack.pop()
        for joint in reversed(children.get(link_name, [])):
            joint_name = joint["elem"].attrib["name"]
            child = joint["child"]
            if joint["type"] == "fixed":
                child_part_id = current_part_id
            else:
                child_part_id = next_part_id
                next_part_id += 1
                joint_part_ids[joint_name] = child_part_id
                joint_order.append(joint_name)
            link_part_ids[child] = child_part_id
            stack.append((child, child_part_id))

    return link_part_ids, joint_part_ids, joint_order


def _mesh_for_link(link: ET.Element, transform: np.ndarray):
    visual_geom = gen._get_visual_geom(link)
    if visual_geom is None:
        return None
    spec = dict(visual_geom)
    spec["T"] = transform @ gen._T_from_xyzrpy(spec["xyz"], spec["rpy"])
    return gen._trimesh_for_spec(spec)


def _urdf_to_bundle(
    *,
    family: str,
    seed: int,
    urdf: str,
    metadata: dict[str, Any],
) -> ProceduralAssetBundle:
    root = ET.fromstring(urdf)
    links_by_name = {link.attrib["name"]: link for link in root.findall("link")}
    joints, parents = gen._collect_joint_maps(root)
    root_link = gen._urdf_find_root_link(root)
    if root_link is None:
        raise RuntimeError(
            f"Procedural family={family!r} produced a URDF without a root link."
        )

    link_transforms = gen._forward_kin_rest(root, joints, parents, root_link)
    link_part_ids, joint_part_ids, joint_order = _assign_link_parts(root_link, joints)

    vertices_chunks = []
    faces_chunks = []
    face_part_chunks = []
    vertex_offset = 0
    for link_name in sorted(links_by_name.keys()):
        mesh = _mesh_for_link(
            links_by_name[link_name], link_transforms.get(link_name, np.eye(4))
        )
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        part_id = int(link_part_ids.get(link_name, 0))
        vertices_chunks.append(vertices)
        faces_chunks.append(faces + vertex_offset)
        face_part_chunks.append(np.full((len(faces),), part_id, dtype=np.int32))
        vertex_offset += len(vertices)

    if not vertices_chunks:
        raise RuntimeError(
            f"Procedural family={family!r} produced no visible geometry."
        )

    rest_vertices = np.concatenate(vertices_chunks, axis=0).astype(
        np.float64, copy=False
    )
    faces = np.concatenate(faces_chunks, axis=0).astype(np.int32, copy=False)
    face_part_ids = np.concatenate(face_part_chunks, axis=0).astype(
        np.int32, copy=False
    )

    kinematics: dict[str, dict[str, Any]] = {}
    for joint_name in joint_order:
        joint_info = joints[joint_name]
        joint_elem = joint_info["elem"]
        joint_type = joint_info["type"]
        if joint_type not in {"revolute", "prismatic", "continuous"}:
            continue
        parent_transform = link_transforms.get(joint_info["parent"], np.eye(4))
        joint_transform = parent_transform @ gen._T_from_xyzrpy(
            joint_info["xyz"], joint_info["rpy"]
        )
        axis = joint_transform[:3, :3] @ _joint_axis(joint_elem)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = axis / axis_norm
        lower, upper = _joint_limit(joint_elem)
        part_id = int(joint_part_ids[joint_name])
        kinematics[joint_name] = {
            "type": "revolute" if joint_type == "continuous" else joint_type,
            "axis": axis.astype(float).tolist(),
            "origin": joint_transform[:3, 3].astype(float).tolist(),
            "limits": [float(lower), float(upper)],
            "controlled_part_ids": [part_id],
        }

    if not kinematics:
        raise RuntimeError(
            f"Procedural family={family!r} produced no articulated joints."
        )

    bb_min = rest_vertices.min(axis=0)
    bb_max = rest_vertices.max(axis=0)
    rest_center = ((bb_min + bb_max) * 0.5).astype(np.float64, copy=False)
    rest_scale = float(max((bb_max - bb_min).max(), 1e-8))
    model_id = f"{family}_seed_{int(seed)}"

    return ProceduralAssetBundle(
        model_id=model_id,
        family=family,
        seed=int(seed),
        rest_vertices=rest_vertices,
        faces=faces,
        face_part_ids=face_part_ids,
        kinematics=kinematics,
        canonical_yaw_offset=float(metadata.get("canonical_yaw_offset", 0.0)),
        rest_center=rest_center,
        rest_scale=rest_scale,
    )


class ProceduralAssetFactory:
    """Stream of procedural assets with one fixed object per index.

    Each index draws a family uniformly from `families` (the list order affects which
    family is drawn) and retries with a new seed until the build succeeds and has at
    most `max_joints` joints.
    """

    def __init__(
        self,
        *,
        families: Optional[list[str]] = None,
        seed: int = 0,
        forbid_cabinet_side_doors: bool = True,
        max_joints: int = 15,
    ) -> None:
        self.families = list(families) if families else available_families()
        unknown = [family for family in self.families if family not in gen.BUILDERS]
        if unknown:
            raise ValueError(
                f"Unknown procedural families={unknown}. Available: {', '.join(available_families())}"
            )
        self.seed = int(seed)
        self.forbid_cabinet_side_doors = bool(forbid_cabinet_side_doors)
        self.max_joints = int(max_joints)

    def _choose_family(self, index_seed: int) -> str:
        return random.Random(index_seed).choices(self.families, k=1)[0]

    def generate(self, index: int) -> ProceduralAssetBundle:
        index = int(index)
        base_seed = self.seed + index * 1_000_003
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            seed = base_seed + attempt * 9_176 + 17
            chosen_family = self._choose_family(seed)
            try:
                urdf, metadata = _build_urdf(
                    chosen_family,
                    seed,
                    forbid_cabinet_side_doors=self.forbid_cabinet_side_doors,
                )
                if self.forbid_cabinet_side_doors and _metadata_has_side_mode(metadata):
                    continue
                bundle = _urdf_to_bundle(
                    family=chosen_family,
                    seed=seed,
                    urdf=urdf,
                    metadata=metadata,
                )
                if len(bundle.kinematics) > self.max_joints:
                    continue
                return bundle
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"Failed to generate a valid procedural asset after {MAX_RETRIES} retries."
        ) from last_error
