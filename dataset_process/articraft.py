"""Articraft-10K -> articraft_test.h5.

Reads the extracted Articraft-10K records (`<input_dir>/<record_id>/model.urdf` plus its
meshes). Revolute, prismatic and continuous joints move; small controls (buttons, knobs,
...) and internal parts (shelves, trays, ...) stay fixed. Each active joint moves its child
link and every link below it up to the next active joint. The camera side of every object
is chosen so that the default camera faces the moving parts.
"""

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from common import build_release, parse_args

ACTIVE_JOINT_TYPES = {"revolute", "prismatic", "continuous"}
EXCLUDE_JOINT_NAME_REGEX = (
    r"(button|knob|dial|switch|key|keycap|control|selector)"
    r"|((^|_)(rack|shelf|tray|turntable|roller|spit|rotisserie|grill|plate|platter|carousel|agitator|tub)(_|$|[0-9]))"
)
PRIMITIVE_SECTIONS = 48
CAMERA_SIDE_TO_YAW = {
    "-y": 0.0,
    "+x": 90.0,
    "+y": 180.0,
    "-x": -90.0,
}
CAMERA_SIDE_DIRECTIONS = {
    "-x": np.array([-1.0, 0.0], dtype=np.float64),
    "+x": np.array([1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0], dtype=np.float64),
    "+y": np.array([0.0, 1.0], dtype=np.float64),
}


@dataclass
class VisualSpec:
    geometry_type: str
    transform: np.ndarray
    filename: Optional[str] = None
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    size: Optional[np.ndarray] = None
    radius: Optional[float] = None
    length: Optional[float] = None


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_transform: np.ndarray
    axis_local: np.ndarray
    lower: float
    upper: float


def _parse_vec(text: Optional[str], default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(elem: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if elem is None:
        return transform
    transform[:3, :3] = _rpy_to_matrix(_parse_vec(elem.attrib.get("rpy"), (0.0, 0.0, 0.0)))
    transform[:3, 3] = _parse_vec(elem.attrib.get("xyz"), (0.0, 0.0, 0.0))
    return transform


class ArticraftProcessor:
    def __init__(self, record_dir: Path):
        self.record_dir = Path(record_dir)
        self.record_id = self.record_dir.name
        self.exclude_joint_name_pattern = re.compile(EXCLUDE_JOINT_NAME_REGEX, re.IGNORECASE)

        self.links: dict[str, list[VisualSpec]] = {}
        self.joints: dict[str, JointSpec] = {}
        self.child_joint_by_link: dict[str, JointSpec] = {}
        self.child_joints_by_parent_link: dict[str, list[JointSpec]] = {}
        self.root_links: list[str] = []
        self.link_world: dict[str, np.ndarray] = {}
        self.active_joint_names: list[str] = []
        self.part_id_by_joint: dict[str, int] = {}
        self.part_id_by_link: dict[str, int] = {}
        self.descendant_joints: dict[str, list[str]] = {}
        self.joint_world_origin: dict[str, np.ndarray] = {}
        self.joint_world_axis: dict[str, np.ndarray] = {}

        self._parse_urdf()
        self._compute_rest_kinematics()
        self._assign_part_ids()
        self.mesh_entries = self._build_meshes()

    def _parse_urdf(self) -> None:
        root = ET.parse(self.record_dir / "model.urdf").getroot()

        for link in root.findall("link"):
            link_name = link.attrib["name"]
            visuals = []
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is None or len(list(geom)) == 0:
                    continue
                visual_tf = _origin_transform(visual.find("origin"))
                child = list(geom)[0]
                tag = child.tag
                if tag == "mesh":
                    scale = _parse_vec(child.attrib.get("scale"), (1.0, 1.0, 1.0))
                    visuals.append(
                        VisualSpec(
                            geometry_type="mesh",
                            transform=visual_tf,
                            filename=child.attrib.get("filename"),
                            scale=scale,
                        )
                    )
                elif tag == "box":
                    visuals.append(
                        VisualSpec(
                            geometry_type="box",
                            transform=visual_tf,
                            size=_parse_vec(child.attrib.get("size"), (1.0, 1.0, 1.0)),
                        )
                    )
                elif tag == "cylinder":
                    visuals.append(
                        VisualSpec(
                            geometry_type="cylinder",
                            transform=visual_tf,
                            radius=float(child.attrib.get("radius", 0.5)),
                            length=float(child.attrib.get("length", 1.0)),
                        )
                    )
                elif tag == "sphere":
                    visuals.append(
                        VisualSpec(
                            geometry_type="sphere",
                            transform=visual_tf,
                            radius=float(child.attrib.get("radius", 0.5)),
                        )
                    )
            self.links[link_name] = visuals

        child_links = set()
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            if parent_elem is None or child_elem is None:
                continue
            parent_link = parent_elem.attrib["link"]
            child_link = child_elem.attrib["link"]
            child_links.add(child_link)

            joint_type = joint.attrib.get("type", "fixed")
            axis = _parse_vec(
                joint.find("axis").attrib.get("xyz") if joint.find("axis") is not None else None,
                (0.0, 0.0, 1.0),
            )
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axis = axis / norm

            limit = joint.find("limit")
            if joint_type == "continuous":
                lower = -math.pi
                upper = math.pi
            elif limit is not None:
                lower = float(limit.attrib.get("lower", 0.0))
                upper = float(limit.attrib.get("upper", 0.0))
            else:
                lower = 0.0
                upper = 0.0

            spec = JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent_link,
                child=child_link,
                origin_transform=_origin_transform(joint.find("origin")),
                axis_local=axis,
                lower=lower,
                upper=upper,
            )
            self.joints[name] = spec
            self.child_joint_by_link[child_link] = spec
            self.child_joints_by_parent_link.setdefault(parent_link, []).append(spec)

        self.root_links = sorted(set(self.links) - child_links)
        if not self.root_links and self.links:
            self.root_links = [sorted(self.links)[0]]

    def _compute_rest_kinematics(self) -> None:
        queue: list[tuple[str, np.ndarray]] = [
            (root_link, np.eye(4, dtype=np.float64)) for root_link in self.root_links
        ]
        seen = set()
        while queue:
            link_name, link_tf = queue.pop(0)
            if link_name in seen:
                continue
            seen.add(link_name)
            self.link_world[link_name] = link_tf
            for joint in self.child_joints_by_parent_link.get(link_name, []):
                joint_tf = link_tf @ joint.origin_transform
                self.joint_world_origin[joint.name] = joint_tf[:3, 3].copy()
                axis_world = joint_tf[:3, :3] @ joint.axis_local
                norm = float(np.linalg.norm(axis_world))
                self.joint_world_axis[joint.name] = (
                    axis_world / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
                )
                queue.append((joint.child, joint_tf))

        for link_name in self.links:
            self.link_world.setdefault(link_name, np.eye(4, dtype=np.float64))

    def _active_joint_depth(self, joint_name: str) -> int:
        joint = self.joints[joint_name]
        depth = 0
        link = joint.parent
        while link in self.child_joint_by_link:
            depth += 1
            link = self.child_joint_by_link[link].parent
        return depth

    def _assign_part_ids(self) -> None:
        active_joint_names = []
        for name, joint in self.joints.items():
            if joint.joint_type not in ACTIVE_JOINT_TYPES or abs(joint.upper - joint.lower) <= 1e-8:
                continue
            if self.exclude_joint_name_pattern.search(name):
                continue
            active_joint_names.append(name)
        self.active_joint_names = active_joint_names
        self.active_joint_names.sort(key=lambda name: (self._active_joint_depth(name), name))
        self.part_id_by_joint = {name: idx + 1 for idx, name in enumerate(self.active_joint_names)}

        def nearest_active_joint_for_link(link_name: str) -> Optional[str]:
            current = link_name
            while current in self.child_joint_by_link:
                joint = self.child_joint_by_link[current]
                if joint.name in self.part_id_by_joint:
                    return joint.name
                current = joint.parent
            return None

        for link_name in self.links:
            joint_name = nearest_active_joint_for_link(link_name)
            self.part_id_by_link[link_name] = (
                self.part_id_by_joint[joint_name] if joint_name is not None else 0
            )

        for joint_name in self.active_joint_names:
            joint = self.joints[joint_name]
            descendants = []
            queue = [joint.child]
            while queue:
                link_name = queue.pop(0)
                for child_joint in self.child_joints_by_parent_link.get(link_name, []):
                    if child_joint.name in self.part_id_by_joint:
                        descendants.append(child_joint.name)
                    queue.append(child_joint.child)
            self.descendant_joints[joint_name] = descendants

    def _visual_to_mesh(self, visual: VisualSpec, world_transform: np.ndarray) -> Optional[trimesh.Trimesh]:
        if visual.geometry_type == "mesh":
            if not visual.filename:
                return None
            try:
                mesh = trimesh.load(self.record_dir / visual.filename, force="mesh", process=False)
            except Exception as exc:
                print(f"  [Warn] {self.record_id}: skipping unreadable mesh {visual.filename}: {exc}")
                return None
            if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                return None
            if not np.allclose(visual.scale, np.ones(3)):
                mesh.apply_scale(visual.scale)
        elif visual.geometry_type == "box":
            mesh = trimesh.creation.box(extents=visual.size)
        elif visual.geometry_type == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=float(visual.radius),
                height=float(visual.length),
                sections=PRIMITIVE_SECTIONS,
            )
        elif visual.geometry_type == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(visual.radius))
        else:
            return None
        mesh.apply_transform(world_transform @ visual.transform)
        return mesh

    def _build_meshes(self) -> list[tuple[trimesh.Trimesh, int]]:
        entries = []
        for link_name, visuals in self.links.items():
            part_id = self.part_id_by_link.get(link_name, 0)
            link_tf = self.link_world.get(link_name, np.eye(4, dtype=np.float64))
            for visual in visuals:
                mesh = self._visual_to_mesh(visual, link_tf)
                if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
                    continue
                entries.append((mesh, int(part_id)))
        return entries

    def _bbox_for_entries(self, moving_only: bool = False) -> Optional[tuple[np.ndarray, np.ndarray]]:
        vertices = []
        for mesh, part_id in self.mesh_entries:
            if moving_only and int(part_id) <= 0:
                continue
            mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if len(mesh_vertices) > 0:
                vertices.append(mesh_vertices)
        if not vertices:
            return None
        stacked = np.vstack(vertices)
        return stacked.min(axis=0), stacked.max(axis=0)

    @staticmethod
    def _side_from_xy_vector(vector_xy: np.ndarray) -> Optional[str]:
        norm = float(np.linalg.norm(vector_xy))
        if norm <= 1e-12:
            return None
        direction = vector_xy / norm
        return max(CAMERA_SIDE_DIRECTIONS, key=lambda side: float(np.dot(direction, CAMERA_SIDE_DIRECTIONS[side])))

    def _prismatic_camera_side(self) -> Optional[str]:
        vectors = []
        for joint_name in self.active_joint_names:
            joint = self.joints[joint_name]
            if joint.joint_type != "prismatic":
                return None
            lo, hi = float(joint.lower), float(joint.upper)
            rest = min(max(0.0, lo), hi)
            target = hi if abs(hi - rest) >= abs(lo - rest) else lo
            motion = self.joint_world_axis[joint_name] * (target - rest)
            if np.linalg.norm(motion[:2]) > 1e-12:
                vectors.append(motion[:2])
        if not vectors:
            return None
        return self._side_from_xy_vector(np.mean(vectors, axis=0))

    def canonical_yaw_offset(self) -> float:
        """Camera yaw facing the moving parts.

        Objects with only prismatic joints are viewed from the side the parts slide out
        to; otherwise from the side of the bounding box the moving parts sit on.
        """
        side = self._prismatic_camera_side()
        if side is None:
            object_bbox = self._bbox_for_entries()
            moving_bbox = self._bbox_for_entries(moving_only=True)
            if object_bbox is not None and moving_bbox is not None:
                object_min, object_max = object_bbox
                moving_min, moving_max = moving_bbox
                extent = np.maximum(object_max - object_min, 1e-9)
                moving_extent = np.maximum(moving_max - moving_min, 1e-9)
                object_center = (object_min + object_max) * 0.5
                moving_center = (moving_min + moving_max) * 0.5
                center_delta = (moving_center - object_center) / extent
                gaps = {
                    "-x": abs(moving_min[0] - object_min[0]) / extent[0],
                    "+x": abs(moving_max[0] - object_max[0]) / extent[0],
                    "-y": abs(moving_min[1] - object_min[1]) / extent[1],
                    "+y": abs(moving_max[1] - object_max[1]) / extent[1],
                }
                thickness_preference = {
                    "-x": 1.0 - min(1.0, moving_extent[0] / extent[0]),
                    "+x": 1.0 - min(1.0, moving_extent[0] / extent[0]),
                    "-y": 1.0 - min(1.0, moving_extent[1] / extent[1]),
                    "+y": 1.0 - min(1.0, moving_extent[1] / extent[1]),
                }
                signed_offsets = {
                    "-x": -center_delta[0],
                    "+x": center_delta[0],
                    "-y": -center_delta[1],
                    "+y": center_delta[1],
                }
                scores = {
                    name: (1.0 - min(1.0, gap)) + 0.6 * thickness_preference[name] + 0.2 * signed_offsets[name]
                    for name, gap in gaps.items()
                }
                side = max(scores, key=scores.get)
        return CAMERA_SIDE_TO_YAW[side or "-y"]

    def get_kinematics_metadata(self) -> dict[str, dict]:
        kinematics = {}
        for joint_name in self.active_joint_names:
            joint = self.joints[joint_name]
            controlled = [self.part_id_by_joint[joint_name]]
            for descendant_name in self.descendant_joints.get(joint_name, []):
                controlled.append(self.part_id_by_joint[descendant_name])
            joint_type = "revolute" if joint.joint_type == "continuous" else joint.joint_type
            kinematics[joint_name] = {
                "type": joint_type,
                "axis": self.joint_world_axis[joint_name].tolist(),
                "origin": self.joint_world_origin[joint_name].tolist(),
                "limits": [float(joint.lower), float(joint.upper)],
                "controlled_part_ids": sorted(set(int(x) for x in controlled)),
            }
        return kinematics

    def mesh_arrays(self):
        faces, face_part_ids, vertices, offset = [], [], [], 0
        for mesh, part_id in self.mesh_entries:
            faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
            face_part_ids.append(np.full(len(mesh.faces), part_id))
            vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
            offset += len(mesh.vertices)
        return np.concatenate(faces), np.concatenate(face_part_ids), np.concatenate(vertices)


if __name__ == "__main__":
    args = parse_args("articraft", "Articraft-10K folder with the extracted rec_* records")

    def process(key):
        processor = ArticraftProcessor(args.input_dir / key.split("/")[1])
        faces, face_part_ids, rest_vertices = processor.mesh_arrays()
        return {
            "faces": faces,
            "face_part_ids": face_part_ids,
            "rest_vertices": rest_vertices,
            "kinematics": processor.get_kinematics_metadata(),
            "canonical_yaw_offset": processor.canonical_yaw_offset(),
        }

    build_release("articraft", args, process)
