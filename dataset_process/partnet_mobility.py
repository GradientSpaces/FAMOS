"""PartNet-Mobility -> partnet_mobility_{train,val,test}.h5.

Reads the raw PartNet-Mobility download (`<input_dir>/<model_id>/mobility.urdf`,
`mobility_v2.json`, `meta.json`, `textured_objs/`). Each selected joint moves one rigid
part: its child link plus all links below it in the kinematic tree; everything else is
merged into the static base (part 0).
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from common import build_release, parse_args

# Categories outside the SINGAPO split: only joints with these semantic tags move.
EXTRA_CATEGORY_TAGS = {
    "Box": ["lid_surface", "drawer", "rotation_lid", "countertop"],
    "Bucket": ["handle"],
    "Display": ["frame", "display_screen"],
    "Door": ["glass", "fixed_part", "hinge", "surface_board"],
    "Eyeglasses": ["leg"],
    "FoldingChair": ["seat"],
    "Knife": ["translation_blade"],
    "Laptop": ["screen", "base_side", "other_leaf", "shaft", "screen_side"],
    "Lighter": ["lid"],
    "Phone": ["rotation_lid", "flipping_lid", "translation_lid"],
    "Pliers": ["leg"],
    "Printer": ["drawer"],
    "Safe": ["door"],
    "Scissors": ["handle", "blade_handle_set"],
    "Stapler": ["lid"],
    "Suitcase": ["lid", "pull-out_handle"],
    "Toilet": ["lid"],
    "TrashCan": ["opener", "lid"],
    "USB": ["cap", "rotation"],
}
# SINGAPO categories: every joint moves except small controls and internal parts.
EXCLUDED_TAG_WORDS = ("knob", "button", "board", "tray")


def keep_joint(category, joint_type, tag):
    if joint_type not in ("revolute", "prismatic") or tag is None:
        return False
    if category in EXTRA_CATEGORY_TAGS:
        return tag in EXTRA_CATEGORY_TAGS[category]
    if category == "Dishwasher" and "shelf" in tag:
        return False
    return not any(word in tag for word in EXCLUDED_TAG_WORDS)


def process_object(model_dir, category):
    root = ET.parse(os.path.join(model_dir, "mobility.urdf")).getroot()
    with open(os.path.join(model_dir, "meta.json")) as f:
        assert json.load(f)["model_cat"] == category, model_dir
    tags = {}
    with open(os.path.join(model_dir, "mobility_v2.json")) as f:
        for entry in json.load(f):
            tags.setdefault(entry["id"], entry["name"])

    joints = [j for j in root.findall("joint")
              if j.find("child") is not None and j.find("parent") is not None]
    child_link = {j.get("name"): j.find("child").get("link") for j in joints}
    parent_link = {j.get("name"): j.find("parent").get("link") for j in joints}

    def descendants(name):
        found, queue = [], [name]
        while queue:
            link = child_link[queue.pop(0)]
            for other, parent in parent_link.items():
                if parent == link and other not in found:
                    found.append(other)
                    queue.append(other)
        return found

    selected = [j.get("name") for j in joints
                if keep_joint(category, j.get("type"), tags.get(int(j.get("name").split("_")[-1])))]
    nested = {d for name in selected for d in descendants(name)}
    root_joints = [name for name in selected if name not in nested]

    # Part ids: 0 = base, then one id per moving root joint (child link + all links below it).
    links = {link.get("name"): link for link in root.findall("link")}
    part_ids = {name: 0 for name in links}
    for part_id, name in enumerate(root_joints, start=1):
        for joint in [name] + descendants(name):
            if child_link[joint] in part_ids:
                part_ids[child_link[joint]] = part_id

    kinematics = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in root_joints:
            continue
        origin, axis, limit = joint.find("origin"), joint.find("axis"), joint.find("limit")
        part_id = part_ids.get(child_link[name], 0)
        kinematics[name] = {
            "type": joint.get("type"),
            "axis": [float(x) for x in axis.get("xyz").split()] if axis is not None else [0.0] * 3,
            "origin": ([float(x) for x in origin.get("xyz").split()]
                       if origin is not None and "xyz" in origin.attrib else [0.0] * 3),
            "limits": [float(limit.get("lower", 0)), float(limit.get("upper", 0))]
                      if limit is not None else [0.0, 0.0],
            "controlled_part_ids": [part_id] if part_id > 0 else [],
        }

    # Rest mesh, grouped by part id. The textured_objs are stored in object coordinates at the
    # rest pose, so the URDF visual origins are not applied.
    meshes_by_part = {}
    for name, link in links.items():
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            path = os.path.join(model_dir, mesh.get("filename")) if mesh is not None else None
            if path is not None and os.path.exists(path):
                meshes_by_part.setdefault(part_ids[name], []).append(path)

    faces, face_part_ids, vertices, offset = [], [], [], 0
    for part_id in sorted(meshes_by_part):
        for path in meshes_by_part[part_id]:
            mesh = trimesh.load(path, force="mesh", process=False)
            faces.append(mesh.faces + offset)
            face_part_ids.append(np.full(len(mesh.faces), part_id))
            vertices.append(mesh.vertices.astype(np.float32))
            offset += len(mesh.vertices)

    return {
        "faces": np.concatenate(faces),
        "face_part_ids": np.concatenate(face_part_ids),
        "rest_vertices": np.concatenate(vertices),
        "kinematics": kinematics,
    }


if __name__ == "__main__":
    args = parse_args("partnet_mobility", "Raw PartNet-Mobility folder with one directory per model id.")

    def process(key):
        category, model_id = key.split("/")
        return process_object(os.path.join(args.input_dir, model_id), category)

    build_release("partnet_mobility", args, process)
