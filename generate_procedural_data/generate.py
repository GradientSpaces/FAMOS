"""Procedural articulated asset generator.

Builds URDF articulated objects from analytic primitives (boxes, cylinders, hollow
cylinders, half-tori, spheres, holed panels), then normalizes them, exports textured
GLB meshes, and writes PartNet-Mobility-style metadata (mobility_v2.json, result.json).

Families and variants:
  hinged_cabinet    – single-bay cabinet with hinged doors (front / top / side)
  drawer_cabinet    – single-bay cabinet with sliding drawers
  combo_cabinet     – 2 bays, each hinged doors or drawers
  triple_combo_cabinet – 3 bays, each hinged doors or drawers
  stacked_cabinets  – two equally sized cabinet units stacked vertically

  door              – single_swing        single panel, left or right hinge
                      double_swing        two panels, hinged on opposite sides
                      sliding             two half-width panels, one slides on X
                      sliding_up          two half-height panels, top slides on Z

  table             – drop_leaf_single    full top + 1 leaf extending beyond edge
                      drop_leaf_double    full top + 2 asymmetric leaves
                      drawer              full top + drawer sliding out front (-Y)
                      fold_up             top folds up from back hinge

  box               – split_lid           two lid halves hinged front/back
                      lid_tray            back-hinged lid + inner tray slides up (Z)

  laptop            – standard            base + screen hinged at back edge
                      book                base + cover hinged on left/right (spine)

  oven              – side_door_rack      side-hinged door + sliding rack
                      drop_door_rack      bottom-hinged drop door + sliding rack

  bucket            – standard            open-top bin + swinging handle

  eyeglasses        – standard            front frame + two folding arms

  fan               – standard            base + pole + spinning blade panel

  scissors          – standard            two blades pivoting around center

  trash_can         – revolute            cylindrical bin + revolving lid
                      lift                cylindrical bin + lid slides up (Z)

  trolley           – 4_wheel            slim box body + telescoping U-handle + 4 wheels

  washing_machine   – side_hinge_left    cube body + inner drum + round door (left hinge)
                      side_hinge_right   cube body + inner drum + round door (right hinge)

Usage:
  python generate_procedural_data/generate.py --out procedural_assets --per_family 10 \\
      --materials materials/

FAMOS training builds the same objects in memory (famos/data/procedural_assets.py)
with FAMOS_TRAINING enabled and without the fan family.
"""
import argparse
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

_DRAWER_ROOT_RE = re.compile(r"(?:^|_)drawer_\d+$")  # matches bay prefixes too (e.g., A_bay1_drawer_2)

def _is_drawer_root(name: str) -> bool:
    return bool(name and _DRAWER_ROOT_RE.search(name))

_DOOR_LINK_RE = re.compile(r'(?:^|_)door(?:$|_)')  # matches 'door', '..._door', '..._door_seg0', etc.

def _is_door_link(name: str) -> bool:
    return bool(name and _DOOR_LINK_RE.search(name))

_IMAGE_CACHE = {}
def _get_image(path: Path):
    p = str(path)
    img = _IMAGE_CACHE.get(p)
    if img is None:
        if not path.exists():
            raise FileNotFoundError(f"Material texture not found: {path}")
        try:
            _IMAGE_CACHE[p] = Image.open(p).convert("RGBA")
            img = _IMAGE_CACHE[p]
        except Exception as e:
            raise RuntimeError(f"Failed to load material texture {path}: {e}")
    return img

def _make_pbr_visual(uvs, material_set):
    """Build a trimesh TextureVisuals with full PBR (basecolor + metallicRoughness + normal)."""
    basecolor_img = _get_image(material_set["basecolor"])
    kwargs = {"baseColorTexture": basecolor_img}

    rough_path = material_set.get("roughness")
    metal_path = material_set.get("metallic")
    if rough_path and rough_path.exists() and metal_path and metal_path.exists():
        # Pack into glTF metallicRoughnessTexture: G=roughness, B=metallic
        rough_img = _get_image(rough_path).convert("L")
        metal_img = _get_image(metal_path).convert("L")
        size = basecolor_img.size
        rough_resized = rough_img.resize(size)
        metal_resized = metal_img.resize(size)
        mr = Image.merge("RGBA", [
            Image.new("L", size, 0),       # R (unused/occlusion)
            rough_resized,                  # G = roughness
            metal_resized,                  # B = metallic
            Image.new("L", size, 255),      # A
        ])
        kwargs["metallicRoughnessTexture"] = mr
        kwargs["metallicFactor"] = 1.0
        kwargs["roughnessFactor"] = 1.0

    normal_path = material_set.get("normal")
    if normal_path and normal_path.exists():
        kwargs["normalTexture"] = _get_image(normal_path).resize(basecolor_img.size)

    pbr_mat = trimesh.visual.material.PBRMaterial(**kwargs)
    return trimesh.visual.TextureVisuals(uv=uvs, material=pbr_mat)

def rnd(a, b): return float(random.uniform(a, b))
def rnd_color(): return [rnd(0.15, 0.95), rnd(0.15, 0.95), rnd(0.15, 0.95), 1.0]

def inertia_box(m, sx, sy, sz):
    ixx = m*(sy*sy + sz*sz)/12.0
    iyy = m*(sx*sx + sz*sz)/12.0
    izz = m*(sx*sx + sy*sy)/12.0
    return ixx, iyy, izz

def inertia_cylinder(m, radius, length):
    """Solid cylinder aligned along Z axis."""
    r2 = radius * radius
    l2 = length * length
    ixx = m * (3*r2 + l2) / 12.0
    iyy = ixx
    izz = m * r2 / 2.0
    return ixx, iyy, izz

def inertia_hollow_cylinder(m, outer_r, inner_r, length):
    """Thin-walled hollow cylinder (tube) aligned along Z."""
    r2o = outer_r * outer_r
    r2i = inner_r * inner_r
    l2 = length * length
    ixx = m * (3*(r2o + r2i) + l2) / 12.0
    iyy = ixx
    izz = m * (r2o + r2i) / 2.0
    return ixx, iyy, izz

def inertia_half_torus(m, arc_radius, tube_radius):
    """Approximate inertia for a semicircular arc bar (half torus)."""
    # Treat as thin ring approximation
    r2 = arc_radius * arc_radius
    t2 = tube_radius * tube_radius
    ixx = m * (r2 + t2 * 0.75) / 2.0
    iyy = m * (r2 + t2 * 0.75) / 2.0
    izz = m * (r2 + t2 * 0.5)
    return ixx, iyy, izz

def inertia_sphere(m, radius):
    i = 2.0 * m * radius * radius / 5.0
    return i, i, i

def inertia_panel_with_hole(m, width, height, thickness):
    """Approximate as solid box (hole is small relative to panel)."""
    return inertia_box(m, width, height, thickness)

def _clamp_inertia(ixx, iyy, izz, floor=1e-6):
    return max(floor, ixx), max(floor, iyy), max(floor, izz)

def material_xml(name, rgba):
    r,g,b,a = rgba
    return f'''
  <material name="{name}">
    <color rgba="{r:.4f} {g:.4f} {b:.4f} {a:.4f}"/>
  </material>'''

def link_box_xml(name, size_xyz, origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    sx, sy, sz = size_xyz; ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_box(max(1e-4, mass), sx, sy, sz))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><box size="{sx:.4f} {sy:.4f} {sz:.4f}"/></geometry>
    </collision>
  </link>'''

def link_cylinder_xml(name, radius, length, origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    """URDF cylinder: aligned along Z, centred at origin. length = height along Z."""
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_cylinder(max(1e-4, mass), radius, length))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
    </collision>
  </link>'''

def link_hollow_cylinder_xml(name, radius, inner_radius, length,
                             origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    """Hollow cylinder (tube): Z-aligned, outer radius / inner radius / length.
    Uses custom <hollow_cylinder> tag recognised by our pipeline."""
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_hollow_cylinder(max(1e-4, mass), radius, inner_radius, length))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><hollow_cylinder radius="{radius:.4f}" inner_radius="{inner_radius:.4f}" length="{length:.4f}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><cylinder radius="{radius:.4f}" length="{length:.4f}"/></geometry>
    </collision>
  </link>'''

def link_half_torus_xml(name, arc_radius, tube_radius,
                        origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    """Half-torus (semicircular arc bar): arches in XZ plane from +X through +Z to -X.
    arc_radius = centre-line radius of the arc, tube_radius = bar thickness."""
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_half_torus(max(1e-4, mass), arc_radius, tube_radius))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><half_torus arc_radius="{arc_radius:.4f}" tube_radius="{tube_radius:.4f}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><sphere radius="{arc_radius + tube_radius:.4f}"/></geometry>
    </collision>
  </link>'''

def link_sphere_xml(name, radius, origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_sphere(max(1e-4, mass), radius))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><sphere radius="{radius:.4f}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><sphere radius="{radius:.4f}"/></geometry>
    </collision>
  </link>'''

def link_panel_with_hole_xml(name, width, height, thickness, hole_radius,
                             hole_cx=0.0, hole_cy=0.0, sections=32,
                             origin_xyz=(0,0,0), rpy=(0,0,0), mass=1.0, material="mat"):
    """Rectangular panel (X=width, Z=height, Y=thickness) with a circular hole.
    hole_cx/hole_cy are the hole center in the panel's XZ plane."""
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    ixx, iyy, izz = _clamp_inertia(*inertia_panel_with_hole(max(1e-4, mass), width, height, thickness))
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
    <visual>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><panel_with_hole width="{width:.4f}" height="{height:.4f}" thickness="{thickness:.4f}" hole_radius="{hole_radius:.4f}" hole_cx="{hole_cx:.4f}" hole_cy="{hole_cy:.4f}" sections="{sections}"/></geometry>
      <material name="{material}"/>
    </visual>
    <collision>
      <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
      <geometry><box size="{width:.4f} {thickness:.4f} {height:.4f}"/></geometry>
    </collision>
  </link>'''

def fixed_joint_xml(name, parent, child, origin_xyz=(0,0,0), rpy=(0,0,0)):
    ox, oy, oz = origin_xyz; rr, rp, ry = rpy
    return f'''
  <joint name="{name}" type="fixed">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="{rr:.4f} {rp:.4f} {ry:.4f}"/>
  </joint>'''

def revolute_joint_xml(name, parent, child, origin_xyz, axis_xyz, lower, upper, effort=40, vel=3):
    ox, oy, oz = origin_xyz; ax, ay, az = axis_xyz
    return f'''
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="0 0 0"/>
    <axis xyz="{ax:.4f} {ay:.4f} {az:.4f}"/>
    <limit lower="{lower:.6f}" upper="{upper:.6f}" effort="{effort}" velocity="{vel}"/>
  </joint>'''

def prismatic_joint_xml(name, parent, child, origin_xyz, axis_xyz, lower, upper, effort=300, vel=1.5):
    ox, oy, oz = origin_xyz; ax, ay, az = axis_xyz
    return f'''
  <joint name="{name}" type="prismatic">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{ox:.4f} {oy:.4f} {oz:.4f}" rpy="0 0 0"/>
    <axis xyz="{ax:.4f} {ay:.4f} {az:.4f}"/>
    <limit lower="{lower:.6f}" upper="{upper:.6f}" effort="{effort}" velocity="{vel}"/>
  </joint>'''

def link_empty_xml(name, mass=1e-3):
    """A link with inertial only (no visual/collision) to act as a kinematic root."""
    return f'''
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>'''

FLUSH_EPS = 0.0005  # tiny pull-in to avoid Z-fighting when closed/flush
DOOR_MAX_OPEN_DEG = 90.0
DOOR_MAX_OPEN_RAD = math.radians(DOOR_MAX_OPEN_DEG)

# Training variant, enabled by famos/data/procedural_assets.py: no box tray, no oven
# rack, revolute-only trash-can lid, wider scissors opening, eyeglasses arms capped at 90 deg.
FAMOS_TRAINING = False

def add_hollow_drawer(links, joints, name, w, d, h, t, center_xyz=(0.0, 0.0, 0.0),
                      material="mat_drawer", mat_interior=None):
    if mat_interior is None:
        mat_interior = material
    t = max(0.004, min(t, 0.45*h, 0.25*w, 0.25*d))
    inner_w = max(0.001, w - 2*t)
    inner_d = max(0.001, d - 2*t)

    cx, cy, cz = center_xyz
    links.append(link_empty_xml(name, mass=0.01))

    vol_front  = w * t * h
    vol_back   = w * t * h
    vol_left   = t * inner_d * h
    vol_right  = t * inner_d * h
    vol_bottom = inner_w * inner_d * t
    vol_sum = max(1e-6, vol_front + vol_back + vol_left + vol_right + vol_bottom)
    total_m = 1.1
    def alloc(vol): return total_m * (vol/vol_sum)

    links.append(link_box_xml(f"{name}_front", (w, t, h),
                              origin_xyz=(cx, cy - d/2 + t/2, cz),
                              mass=alloc(vol_front), material=material))
    joints.append(fixed_joint_xml(f"fix_{name}_front", name, f"{name}_front"))

    links.append(link_box_xml(f"{name}_back", (w, t, h),
                              origin_xyz=(cx, cy + d/2 - t/2, cz),
                              mass=alloc(vol_back), material=mat_interior))
    joints.append(fixed_joint_xml(f"fix_{name}_back", name, f"{name}_back"))

    links.append(link_box_xml(f"{name}_left", (t, inner_d, h),
                              origin_xyz=(cx - w/2 + t/2, cy, cz),
                              mass=alloc(vol_left), material=mat_interior))
    joints.append(fixed_joint_xml(f"fix_{name}_left", name, f"{name}_left"))

    links.append(link_box_xml(f"{name}_right", (t, inner_d, h),
                              origin_xyz=(cx + w/2 - t/2, cy, cz),
                              mass=alloc(vol_right), material=mat_interior))
    joints.append(fixed_joint_xml(f"fix_{name}_right", name, f"{name}_right"))

    links.append(link_box_xml(f"{name}_bottom", (inner_w, inner_d, t),
                              origin_xyz=(cx, cy, cz - h/2 + t/2),
                              mass=alloc(vol_bottom), material=mat_interior))
    joints.append(fixed_joint_xml(f"fix_{name}_bottom", name, f"{name}_bottom"))

def add_cabinet_panels(links, joints, W, D, H, wall, mat="mat_body",
                       skip_left=False, skip_right=False, skip_top=False,
                       mat_interior=None):
    if mat_interior is None:
        mat_interior = mat
    links.append(link_empty_xml("base_inertial_dummy", mass=0.6))
    if not skip_left:
        links.append(link_box_xml("side_L",(wall,D,H), origin_xyz=(-W/2+wall/2,0,0), mass=1.0, material=mat))
        joints.append(fixed_joint_xml("fix_side_L","base_inertial_dummy","side_L"))
    if not skip_right:
        links.append(link_box_xml("side_R",(wall,D,H), origin_xyz=(+W/2-wall/2,0,0), mass=1.0, material=mat))
        joints.append(fixed_joint_xml("fix_side_R","base_inertial_dummy","side_R"))
    if not skip_top:
        links.append(link_box_xml("top",(W,D,wall), origin_xyz=(0,0,+H/2-wall/2), mass=1.0, material=mat))
        joints.append(fixed_joint_xml("fix_top","base_inertial_dummy","top"))
    links.append(link_box_xml("bottom",(W,D,wall), origin_xyz=(0,0,-H/2+wall/2), mass=1.0, material=mat_interior))
    joints.append(fixed_joint_xml("fix_bottom","base_inertial_dummy","bottom"))
    links.append(link_box_xml("back",(W,wall,H), origin_xyz=(0,+D/2-wall/2,0), mass=1.0, material=mat_interior))
    joints.append(fixed_joint_xml("fix_back","base_inertial_dummy","back"))

def add_front_panel_span(links, joints, W, D, H, wall, span_w, x_center, mat="mat_body", name="front_span", parent="base_inertial_dummy"):
    links.append(link_box_xml(name, (span_w, wall, H),
                              origin_xyz=(x_center, -D/2 + wall/2, 0.0),
                              mass=1.0, material=mat))
    joints.append(fixed_joint_xml(f"fix_{name}", parent, name))

def add_top_panel_span(links, joints, W, D, H, wall, span_w, x_center, mat="mat_body", name="top_span", parent="base_inertial_dummy"):
    links.append(link_box_xml(name, (span_w, D, wall),
                              origin_xyz=(x_center, 0.0, H/2 - wall/2),
                              mass=1.0, material=mat))
    joints.append(fixed_joint_xml(f"fix_{name}", parent, name))

def place_front_door(W,D,H,wall,gap,door_thk, theta_cap, axis):
    hinge_out = rnd(0.004, 0.010)
    hinge_y   = -D/2 - hinge_out
    y_center  = (-D/2 + FLUSH_EPS) + door_thk/2
    y_offset  = y_center - hinge_y
    return hinge_out, hinge_y, y_offset

def place_top_lid(W,D,H,wall,gap,door_thk):
    hinge_out = rnd(0.004, 0.010)
    hinge_z   = H/2 - wall/2 + hinge_out
    hinge_y   = +D/2 - wall - gap
    z_center  = (H/2 - FLUSH_EPS) - door_thk/2
    z_offset  = z_center - hinge_z
    door_d    = D - 2*wall - 2*gap
    y_offset  = -door_d/2.0
    return hinge_out, hinge_z, hinge_y, z_offset, y_offset

def place_side_door(side, W, D, H, wall, gap, door_thk):
    sgn = -1 if side == "left" else +1
    hinge_out = rnd(0.004, 0.010)
    hinge_x   = sgn*(W/2 - wall/2) + sgn*hinge_out
    x_center  = sgn*((W/2 - FLUSH_EPS) - door_thk/2)
    x_offset  = x_center - hinge_x
    return sgn, hinge_out, hinge_x, x_offset

def plan_drawer_closed_front(D):
    front_margin = rnd(0.0015, 0.0030)
    drawer_d = rnd(0.35*D, 0.55*D)
    y_closed = -D/2 + front_margin + drawer_d/2.0
    ext_limit = max(0.010, drawer_d)
    return drawer_d, y_closed, front_margin, ext_limit


def _bay_skip_flags(door_modes):
    """Compute shell panel skips from a list of door modes (strings or None)."""
    skip_left  = any(m and m.startswith("side_left")  for m in door_modes)
    skip_right = any(m and m.startswith("side_right") for m in door_modes)
    skip_top   = any(m == "top_lid" for m in door_modes)
    return skip_left, skip_right, skip_top

def _add_drawer_column(links, joints, base_link, prefix, *,
                       x_center, bay_w, W, D, H, wall, gap, mat_drawer,
                       mat_knob="mat_knob", knob_radius=None,
                       mat_interior=None):
    n_drawers = random.choice([1,2,3,4])
    gap_v = rnd(0.0015, 0.0035)
    usable_h = H - 2*wall - (n_drawers+1)*gap_v
    slot_h = usable_h / n_drawers
    drawer_d, y_closed, front_margin, ext_limit = plan_drawer_closed_front(D)
    drawer_w = bay_w - rnd(0.004, 0.010)
    z0 = -H/2 + wall + gap_v + slot_h/2.0
    n_knobs = random.choice([1, 2])  # consistent across all drawers in this column
    for i in range(n_drawers):
        zc = z0 + i*(slot_h + gap_v)
        drawer_h = max(0.02, slot_h - gap_v)
        wall_t   = rnd(0.006, 0.012)
        name = f"{prefix}drawer_{i}"
        add_hollow_drawer(links, joints, name,
                          w=drawer_w, d=drawer_d, h=drawer_h, t=wall_t,
                          center_xyz=(0.0, 0.0, 0.0), material=mat_drawer,
                          mat_interior=mat_interior)
        # Joint origin placed at the drawer's actual center in cabinet frame
        joints.append(prismatic_joint_xml(f"{prefix}slide_{i}",
                                          base_link, name,
                                          origin_xyz=(x_center, y_closed, zc),
                                          axis_xyz=(0,-1,0), lower=0.0, upper=ext_limit))
        # Sphere knob(s) on front face of drawer
        kr = knob_radius if knob_radius is not None else rnd(0.012, 0.025)
        for ki in range(n_knobs):
            if n_knobs == 1:
                kx = 0.0
            else:
                kx = (-0.25 + 0.5 * ki) * drawer_w  # quarter-width apart
            ky = -drawer_d / 2 - kr  # protrude from front face
            kz = 0.0
            kname = f"{prefix}knob_{i}_{ki}"
            links.append(link_sphere_xml(kname, kr, origin_xyz=(kx, ky, kz), mass=0.02, material=mat_knob))
            joints.append(fixed_joint_xml(f"fix_{kname}", name, kname))
    return {
        "n": n_drawers, "slot_h": slot_h,
        "drawer_d": drawer_d, "y_closed": y_closed,
        "ext_limit": ext_limit, "front_margin": front_margin
    }

def _add_door_in_bay(links, joints, base_link, prefix, *,
                     x_center, bay_w, W, D, H, wall, gap,
                     mat_body, mat_door, mat_knob="mat_knob",
                     knob_radius=None,
                     theta_cap=DOOR_MAX_OPEN_RAD,
                     forbid_open_up=False,
                     mode=None,
                     split=None):
    """
    mode in:
      - "front_top", "front_bottom" (front hinged)
      - "top_lid"
      - "side_left_top", "side_left_bottom", "side_right_top", "side_right_bottom"

    split:
      - None (default): maybe split randomly inside this function
      - "same": two panels open the same way (both top-hinged or both bottom-hinged, following `mode`)
      - "opposite": upper panel opens opposite to lower (e.g., upper flips up, lower flips down)
    """
    door_thk  = rnd(0.015, 0.025)
    kr = knob_radius if knob_radius is not None else rnd(0.008, 0.018)

    # If an upward opening is disallowed, remap to a downward/front-bottom alternative
    if forbid_open_up and mode in ("front_top", "top_lid"):
        mode = "front_bottom"

    if mode is None:
        # choose something reasonable if not provided
        mode = random.choice(["front_top", "front_bottom", "top_lid",
                              "side_left_top","side_left_bottom",
                              "side_right_top","side_right_bottom"])

    # Front doors hinged at the top or bottom, optionally split into two leaves.
    if mode in ("front_top", "front_bottom"):
        door_w = bay_w - rnd(0.0010, 0.0030)
        # height equals inner opening height minus tiny reveal (gap was tightened above)
        door_h = H - 2*wall - 2*gap

        # decide split if not specified
        if split is None:
            # ~50% chance to split, pattern chosen uniformly if split
            if random.random() < 0.5:
                split = None
            else:
                split = random.choice(["same", "opposite"])

        # Helper to place a single front door leaf (full height or half height)
        def _place_front_leaf(name_suffix, hinge_style, leaf_h, hinge_z, leaf_axis_sign):
            """
            hinge_style: "top" or "bottom" (where hinge sits on the leaf)
            leaf_axis_sign: +1 for front_top style (axis=(-1,0,0)), -1 for front_bottom style (axis=(+1,0,0))
            hinge_z: absolute z of the hinge line for this leaf (in base frame)
            """
            axis = (-leaf_axis_sign, 0, 0)
            # same y placement for all front leaves
            _, hinge_y, y_offset = place_front_door(W,D,H,wall,gap,door_thk,theta_cap,axis)

            if hinge_style == "top":
                z_offset = -leaf_h/2.0
                # Top-hinged: axis=(-1,0,0) needs positive rotation to open down/out
                lower, upper = 0.0, theta_cap
            else:  # "bottom"
                z_offset = +leaf_h/2.0
                # Bottom-hinged: axis=(+1,0,0) needs positive rotation to open up/out
                lower, upper = 0.0, theta_cap

            lname = f"{prefix}door{name_suffix}"
            jname = f"{prefix}hinge{name_suffix}"
            links.append(link_box_xml(lname, (door_w, door_thk, leaf_h),
                                      origin_xyz=(0.0, y_offset, z_offset),
                                      mass=0.6, material=mat_door))
            joints.append(revolute_joint_xml(jname, base_link, lname,
                                             origin_xyz=(x_center, hinge_y, hinge_z),
                                             axis_xyz=axis, lower=lower, upper=upper))
            # Sphere knob on front face, near the free edge (opposite hinge)
            knob_z = z_offset + (-leaf_h * 0.35 if hinge_style == "top" else +leaf_h * 0.35)
            kn = f"{prefix}knob{name_suffix}"
            links.append(link_sphere_xml(kn, kr,
                         origin_xyz=(0.0, y_offset - door_thk/2 - kr, knob_z),
                         mass=0.02, material=mat_knob))
            joints.append(fixed_joint_xml(f"fix_{kn}", lname, kn))

        if split in ("same", "opposite"):
            # Split into two stacked leaves separated by a tiny seam
            seam = rnd(0.0010, 0.0025)
            half_h = max(0.02, 0.5*(door_h - seam))
            z_top_edge    = +H/2 - wall/2 - gap
            z_bottom_edge = -H/2 + wall/2 + gap
            z_seam_top_of_lower = +seam/2.0
            z_seam_bottom_of_upper = -seam/2.0

            if split == "same":
                if mode == "front_top":
                    # Both leaves are top-hinged
                    _place_front_leaf("_seg0", "top", half_h, z_top_edge, +1)             # upper, hinge at cabinet top
                    _place_front_leaf("_seg1", "top", half_h, z_seam_top_of_lower, +1)    # lower, hinge at seam
                else:
                    # Both leaves are bottom-hinged
                    _place_front_leaf("_seg0", "bottom", half_h, z_seam_bottom_of_upper, -1)  # upper, hinge at seam
                    _place_front_leaf("_seg1", "bottom", half_h, z_bottom_edge, -1)          # lower, hinge at cabinet bottom
            else:
                # split == "opposite"
                if mode == "front_top":
                    # Upper flips up (top-hinged), lower flips down (bottom-hinged)
                    _place_front_leaf("_seg0", "top",    half_h, z_top_edge, +1)             # upper
                    _place_front_leaf("_seg1", "bottom", half_h, z_bottom_edge, -1)          # lower
                else:
                    # mode == "front_bottom": Upper flips up, lower flips down (opposite to bottom mode)
                    _place_front_leaf("_seg0", "top",    half_h, z_top_edge, +1)             # upper
                    _place_front_leaf("_seg1", "bottom", half_h, z_bottom_edge, -1)          # lower

            return {"mode": mode, "split": split}

        # Not splitting: a single leaf.
        if mode == "front_top":
            axis=(-1,0,0)
            _, hinge_y, y_offset = place_front_door(W,D,H,wall,gap,door_thk,theta_cap,axis)
            z_hinge = +H/2 - wall/2 - gap; z_offset = -door_h/2.0
            # Top-hinged: needs positive rotation to open down/out
            lower, upper = 0.0, theta_cap
        else:
            axis=(1,0,0)
            _, hinge_y, y_offset = place_front_door(W,D,H,wall,gap,door_thk,theta_cap,axis)
            z_hinge = -H/2 + wall/2 + gap; z_offset = +door_h/2.0
            # Bottom-hinged: needs positive rotation to open up/out
            lower, upper = 0.0, theta_cap

        links.append(link_box_xml(f"{prefix}door",(door_w,door_thk,door_h),
                                  origin_xyz=(0.0, y_offset, z_offset), mass=1.2, material=mat_door))
        joints.append(revolute_joint_xml(f"{prefix}hinge", base_link, f"{prefix}door",
                                         origin_xyz=(x_center, hinge_y, z_hinge),
                                         axis_xyz=axis, lower=lower, upper=upper))
        # Sphere knob on front face, near the free edge
        knob_z = z_offset + (-door_h * 0.35 if mode == "front_top" else +door_h * 0.35)
        kn = f"{prefix}knob"
        links.append(link_sphere_xml(kn, kr,
                     origin_xyz=(0.0, y_offset - door_thk/2 - kr, knob_z),
                     mass=0.02, material=mat_knob))
        joints.append(fixed_joint_xml(f"fix_{kn}", f"{prefix}door", kn))
        return {"mode": mode, "split": None}

    # Front doors hinged on the left or right.
    if mode in ("front_left", "front_right"):
        door_w = bay_w - rnd(0.0010, 0.0030)
        door_h = H - 2*wall - 2*gap

        axis_z = -1 if mode == "front_left" else +1
        axis = (0, 0, axis_z)

        # front placement (same as other front doors)
        _, hinge_y, y_offset = place_front_door(W, D, H, wall, gap, door_thk, theta_cap, axis)

        # hinge line along left or right edge of the door leaf
        if mode == "front_left":
            x_hinge = x_center - door_w*0.5
            x_vis_offset = +door_w*0.5   # leaf center to the right of hinge
        else:  # "front_right"
            x_hinge = x_center + door_w*0.5
            x_vis_offset = -door_w*0.5   # leaf center to the left of hinge

        z_hinge = 0.0
        z_offset = 0.0

        links.append(link_box_xml(f"{prefix}door",
                                  (door_w, door_thk, door_h),
                                  origin_xyz=(x_vis_offset, y_offset, z_offset),
                                  mass=1.2, material=mat_door))
        joints.append(revolute_joint_xml(f"{prefix}hinge", base_link, f"{prefix}door",
                                         origin_xyz=(x_hinge, hinge_y, z_hinge),
                                         axis_xyz=axis, lower=0.0, upper=theta_cap))
        # Sphere knob on front face, opposite the hinge side
        kx_sign = +1 if mode == "front_left" else -1  # opposite hinge
        knob_x = x_vis_offset + kx_sign * door_w * 0.35
        kn = f"{prefix}knob"
        links.append(link_sphere_xml(kn, kr,
                     origin_xyz=(knob_x, y_offset - door_thk/2 - kr, 0.0),
                     mass=0.02, material=mat_knob))
        joints.append(fixed_joint_xml(f"fix_{kn}", f"{prefix}door", kn))
        return {"mode": mode, "split": None}

    # Top lid.
    if mode == "top_lid":
        door_w = bay_w - rnd(0.0010, 0.0030)
        door_d = D - 2*wall - 2*gap
        _, hinge_z, hinge_y, z_offset, y_offset = place_top_lid(W,D,H,wall,gap,door_thk)
        links.append(link_box_xml(f"{prefix}door",(door_w,door_d,door_thk),
                                  origin_xyz=(0.0, y_offset, z_offset), mass=1.2, material=mat_door))
        joints.append(revolute_joint_xml(f"{prefix}hinge", base_link, f"{prefix}door",
                                         origin_xyz=(x_center, hinge_y, hinge_z),
                                         axis_xyz=(1,0,0), lower=-theta_cap, upper=0.0))
        # Sphere knob on top face, near front edge of lid
        kn = f"{prefix}knob"
        links.append(link_sphere_xml(kn, kr,
                     origin_xyz=(0.0, y_offset - door_d * 0.35, z_offset + door_thk/2 + kr),
                     mass=0.02, material=mat_knob))
        joints.append(fixed_joint_xml(f"fix_{kn}", f"{prefix}door", kn))
        # close front
        add_front_panel_span(links, joints, W,D,H, wall, door_w, x_center,
                             mat=mat_body, name=f"{prefix}front_panel", parent=base_link)
        return {"mode": mode, "split": None}

    # Side doors.
    side = "left" if "side_left" in mode else "right"
    door_d = D - 2*wall - 2*gap
    door_h = H - 2*wall - 2*gap
    sgn, _, hinge_x, x_offset = place_side_door(side, W,D,H,wall,gap,door_thk)
    if mode.endswith("top"):
        z_hinge = +H/2 - wall/2 - gap; z_offset = -door_h/2.0; axis = (0, sgn, 0)
        # Top-hinged side doors need negative rotation to open upward/out
        lower, upper = -theta_cap, 0.0
    else:
        z_hinge = -H/2 + wall/2 + gap; z_offset = +door_h/2.0; axis = (0, -sgn, 0)
        # Bottom-hinged side doors need negative rotation to open downward/out
        lower, upper = -theta_cap, 0.0
    links.append(link_box_xml(f"{prefix}door",(door_thk,door_d,door_h),
                              origin_xyz=(x_offset, 0.0, z_offset),
                              mass=1.2, material=mat_door))
    joints.append(revolute_joint_xml(f"{prefix}hinge", base_link, f"{prefix}door",
                                     origin_xyz=(hinge_x, 0.0, z_hinge),
                                     axis_xyz=axis, lower=lower, upper=upper))
    # Sphere knob on outer face of side door
    knob_x_sign = -1 if side == "left" else +1  # protrude outward
    kn = f"{prefix}knob"
    links.append(link_sphere_xml(kn, kr,
                 origin_xyz=(x_offset + knob_x_sign * (door_thk/2 + kr), 0.0, z_offset),
                 mass=0.02, material=mat_knob))
    joints.append(fixed_joint_xml(f"fix_{kn}", f"{prefix}door", kn))
    # solid front panel
    add_front_panel_span(links, joints, W,D,H, wall, bay_w - rnd(0.0010, 0.0030), x_center,
                         mat=mat_body, name=f"{prefix}front_panel", parent=base_link)
    return {"mode": mode, "split": None}


def _door_modes_for_pos(pos, bay_count, forbid_open_up=False):
    """Return allowed door modes list based on bay position (outer bays may side hinge)."""
    # front_*: includes flip up/down + regular swing left/right
    opts = ["front_top", "front_bottom", "front_left", "front_right", "top_lid"]
    if pos == 0:                  # left outer may also hinge on left side panel
        opts += ["side_left_top", "side_left_bottom"]
    if pos == bay_count - 1:      # right outer may also hinge on right side panel
        opts += ["side_right_top", "side_right_bottom"]
    if forbid_open_up:
        # disallow modes that open upward
        opts = [m for m in opts if m not in ("front_top", "top_lid")]
    return opts


def _build_multi_bay(*, seed=None, bay_count=1,
                     bay_types=None,      # list of "door"/"drawer" or None for random
                     door_modes=None,     # list of mode strings or None for random (applies where type=="door")
                     forbid_open_up_any=False,  # for the whole cabinet (e.g., bottom unit in stacked)
                     name="cabinet",
                     fixed_dims=None,     # (W, D, H) to force outer size
                     mat_body_name="mat_body",   # material names to reference
                     mat_door_name="mat_door",
                     mat_drawer_name="mat_drawer"):
    """
    Returns (urdf_xml, meta) for bay_count in {1,2,3}
    """
    random.seed(seed)
    # Base dims (optionally fixed)
    if fixed_dims is None:
        W, D, H = rnd(0.55, 1.50), rnd(0.38, 0.85), rnd(0.55, 1.45)
    else:
        W, D, H = fixed_dims
    wall = rnd(0.016, 0.028); gap  = rnd(0.0015, 0.0035)
    knob_r = rnd(0.015, 0.030)  # consistent knob size for entire cabinet

    # Materials (emit with the provided names)
    mat_interior_name = "mat_interior"
    body_col, door_col, drawer_col = rnd_color(), rnd_color(), rnd_color()
    mats = (material_xml(mat_body_name,   body_col) +
            material_xml(mat_door_name,   door_col) +
            material_xml(mat_drawer_name, drawer_col) +
            material_xml(mat_interior_name, rnd_color()))

    # Randomize bay types if not specified
    if bay_types is None:
        if bay_count == 1:
            bay_types = [random.choice(["door","drawer"])]
        else:
            bay_types = [random.choice(["door","drawer"]) for _ in range(bay_count)]
    assert len(bay_types) == bay_count

    # Pick door modes per bay where needed.
    if door_modes is None:
        door_modes = [None]*bay_count
    for i in range(bay_count):
        if bay_types[i] == "door" and door_modes[i] is None:
            door_modes[i] = random.choice(_door_modes_for_pos(i, bay_count, forbid_open_up=forbid_open_up_any))

    # Apply remap here so the rest of the build sees final modes
    def _remap_for_forbid(m):
        if not m: return m
        if forbid_open_up_any and m in ("front_top", "top_lid"):
            return "front_bottom"
        return m

    door_modes = [_remap_for_forbid(m) for m in door_modes]

    # Panel skips depend on the final modes.
    skip_left, skip_right, skip_top = _bay_skip_flags(door_modes)


    links, joints = [], []
    add_cabinet_panels(links, joints, W, D, H, wall, mat=mat_body_name,
                       skip_left=skip_left, skip_right=skip_right, skip_top=skip_top,
                       mat_interior=mat_interior_name)

    # Add dividers & compute bay widths/centers
    part_gap = rnd(0.006, 0.012) if bay_count > 1 else 0.0
    W_in = W - 2*wall
    if bay_count == 1:
        widths = [W_in]
    elif bay_count == 2:
        split = rnd(0.4, 0.6)
        widths = [split*W_in, W_in - split*W_in - part_gap]
    else:
        usable_W = W_in - 2*part_gap
        r = np.array([rnd(0.20, 0.52), rnd(0.20, 0.52), rnd(0.20, 0.52)])
        r = r / max(1e-9, r.sum())
        widths = [usable_W * r[0], usable_W * r[1], usable_W * r[2]]

    centers = []
    x_acc = -W/2 + wall
    for i, w in enumerate(widths):
        xc = x_acc + w/2.0
        centers.append(xc)
        x_acc += w
        if i < bay_count - 1:
            # add divider and gap
            x_div = x_acc + part_gap/2.0
            links.append(link_box_xml(f"divider_{i}", (wall, D, H-2*wall),
                                      origin_xyz=(x_div,0,0), mass=0.8, material=mat_interior_name))
            joints.append(fixed_joint_xml(f"fix_divider_{i}", "base_inertial_dummy", f"divider_{i}"))
            x_acc += part_gap

    # If any bay uses a top lid, add top spans above the other bays
    if skip_top and bay_count > 1:
        for i in range(bay_count):
            if not (bay_types[i] == "door" and door_modes[i] == "top_lid"):
                add_top_panel_span(links, joints, W,D,H, wall, widths[i], centers[i],
                                   mat=mat_body_name, name=f"top_span_{i}", parent="base_inertial_dummy")

    # Build bays
    drawers_meta = None
    bays_meta = []
    for i in range(bay_count):
        prefix = ("bay{}_".format(i)) if bay_count > 1 else ""
        if bay_types[i] == "drawer":
            drawers_meta = _add_drawer_column(
                links, joints, "base_inertial_dummy", prefix,
                x_center=centers[i], bay_w=widths[i],
                W=W, D=D, H=H, wall=wall, gap=gap, mat_drawer=mat_drawer_name,
                mat_knob=mat_drawer_name, knob_radius=knob_r,
                mat_interior=mat_interior_name
            )
            bays_meta.append({"index": i, "type": "drawer", "mode": None, "split": None,
                              "width": widths[i], "x_center": centers[i]})
        else:
            ret = _add_door_in_bay(
                links, joints, "base_inertial_dummy", prefix,
                x_center=centers[i], bay_w=widths[i],
                W=W, D=D, H=H, wall=wall, gap=gap,
                mat_body=mat_body_name, mat_door=mat_door_name,
                mat_knob=mat_door_name, knob_radius=knob_r,
                forbid_open_up=forbid_open_up_any, mode=door_modes[i],
                split=None  # auto/random inside; set to "same"/"opposite" to force
            )
            bays_meta.append({"index": i, "type": "door", "mode": ret.get("mode"),
                              "split": ret.get("split"),
                              "width": widths[i], "x_center": centers[i]})


    family_name = {
        1: ("hinged_cabinet" if bay_types[0]=="door" else "drawer_cabinet"),
        2: "combo_cabinet",
        3: "triple_combo_cabinet"
    }[bay_count]

    meta = {
        "family": family_name,
        "dims":{"W":W,"D":D,"H":H,"wall":wall},
        "bays": bays_meta,
        "drawers": drawers_meta or {},
        "invariant":"q=0 closed (flush); doors ≤90°; drawers extend ≤ depth; side doors only on outer bays; top lids get front/top spans as needed. Front doors may be split into two stacked leaves (same or opposite opening)."
    }


    robot = f'<robot name="{name}">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    return robot, meta

def build_hinged_cabinet(seed=None):
    # force a single-bay DOOR build
    return _build_multi_bay(seed=seed, bay_count=1, bay_types=["door"], name="hinged_cabinet")

def build_drawer_cabinet(seed=None):
    # force a single-bay DRAWER build
    return _build_multi_bay(seed=seed, bay_count=1, bay_types=["drawer"], name="drawer_cabinet")

def build_combo_cabinet(seed=None):
    # 2-bay combo (random mix of doors/drawers)
    return _build_multi_bay(seed=seed, bay_count=2, name="combo_cabinet")

def build_triple_combo_cabinet(seed=None):
    # 3-bay combo (random mix)
    return _build_multi_bay(seed=seed, bay_count=3, name="triple_combo_cabinet")

def build_stacked_cabinets(seed=None):
    random.seed(seed)
    # shared W,D,H across units for stacking posture
    W, D, H = rnd(0.60, 1.40), rnd(0.40, 0.85), rnd(0.65, 1.25)

    links, joints = [], []
    links.append(link_empty_xml("stack_root", mass=0.02))
    mats = (
        material_xml("A_mat_body",   rnd_color()) +
        material_xml("A_mat_door",   rnd_color()) +
        material_xml("A_mat_drawer", rnd_color()) +
        material_xml("B_mat_body",   rnd_color()) +
        material_xml("B_mat_door",   rnd_color()) +
        material_xml("B_mat_drawer", rnd_color())
    )

    # pick families via bay_count and mixes
    families = ["hinged","drawer","combo","triple"]
    fam_bottom = random.choice(families)
    fam_top    = random.choice(families)

    def build_unit(prefix, base_xyz, fam, forbid_up=False):
        # choose bay_count by fam
        bay_count = {"hinged":1, "drawer":1, "combo":2, "triple":3}[fam]
        # force type when needed
        if fam == "hinged":
            bay_types = ["door"]
        elif fam == "drawer":
            bay_types = ["drawer"]
        else:
            bay_types = None  # random

        # Same outer dims for both units, distinct material names per unit.
        urdf, meta = _build_multi_bay(
            seed=random.randint(0,10**9),
            bay_count=bay_count,
            bay_types=bay_types,
            door_modes=None,
            forbid_open_up_any=forbid_up,
            name=f"{prefix.rstrip('_')}unit",
            fixed_dims=(W, D, H),
            mat_body_name=f"{prefix}mat_body",
            mat_door_name=f"{prefix}mat_door",
            mat_drawer_name=f"{prefix}mat_drawer",
        )

        # parse and transplant under stack_root with fixed offset
        base = f"{prefix}base"
        links.append(link_empty_xml(base, mass=0.6))
        joints.append(fixed_joint_xml(f"{prefix}fix_to_stack", "stack_root", base, origin_xyz=base_xyz))

        root = ET.fromstring(urdf)

        # 1) Emit all links with prefix
        for L in root.findall("link"):
            L.attrib["name"] = f"{prefix}{L.attrib['name']}"
            links.append(ET.tostring(L, encoding="unicode"))

        # 2) Emit all joints with prefix (NO reparenting here)
        for J in root.findall("joint"):
            J.attrib["name"] = f"{prefix}{J.attrib['name']}"
            J.find("parent").attrib["link"] = f"{prefix}{J.find('parent').attrib['link']}"
            J.find("child").attrib["link"]  = f"{prefix}{J.find('child').attrib['link']}"
            joints.append(ET.tostring(J, encoding="unicode"))

        # 3) Anchor the sub-URDF root to our unit base.
        pref_sub_root = f"{prefix}base_inertial_dummy"
        anchor = ET.Element("joint", attrib={"name": f"{prefix}fix_subroot_to_base", "type":"fixed"})
        ET.SubElement(anchor, "parent", attrib={"link": base})
        ET.SubElement(anchor, "child",  attrib={"link": pref_sub_root})
        ET.SubElement(anchor, "origin", attrib={"xyz":"0 0 0", "rpy":"0 0 0"})
        joints.append(ET.tostring(anchor, encoding="unicode"))

        return meta

    # bottom: forbid opening upward
    metaA = build_unit("A_", (0.0,0.0,0.0), fam_bottom, forbid_up=True)
    # top: free
    metaB = build_unit("B_", (0.0,0.0,H),    fam_top,    forbid_up=False)

    all_xml = ['<robot name="stacked_cabinets">', mats, *links, *(j for j in joints if j), "</robot>"]
    robot = "\n".join(all_xml)

    meta = {
        "family": "stacked_cabinets",
        "dims": {"W":W,"D":D,"H":H},
        "units": [{"prefix":"A_", "family":fam_bottom}, {"prefix":"B_", "family":fam_top}],
        "invariant": "Two cabinets share identical W,D,H. Top unit fixed directly above bottom. Bottom unit never opens upward.",
        "sub_meta": {"A":metaA, "B":metaB}
    }
    return robot, meta

def _suggest_tile_uv(size_xyz, base_tile=1.0, max_tile=8.0):
    """
    Pick a UV tiling factor based on the largest physical extent of the box.
    - base_tile: ~1 repeat per meter (tweak to taste)
    - max_tile:  cap to avoid extreme tiling on very large parts
    """
    sx, sy, sz = size_xyz
    smax = max(float(sx), float(sy), float(sz))
    return float(min(max_tile, max(1.0, smax) * base_tile))


def _scan_material_sets(materials_root: Path):
    sets = []
    subs = [p for p in materials_root.iterdir() if p.is_dir()]
    if not subs:
        subs = [materials_root]
    for sub in subs:
        bc = sub / "basecolor.png"
        if bc.exists():
            sets.append({
                "name": sub.name,
                "basecolor": bc,
                "metallic": sub / "metallic.png",
                "normal": sub / "normal.png",
                "roughness": sub / "roughness.png",
                "root": sub
            })
    if not sets:
        raise RuntimeError(f"No material sets with basecolor.png found under {materials_root}")
    return sets

def _create_textured_box_glb(size_xyz, material_set: dict, out_glb: Path, tile_uv=None):
    """Create a GLB box with planar UVs and PBR textures."""
    sx, sy, sz = size_xyz
    if tile_uv is None:
        tile_uv = _suggest_tile_uv((sx, sy, sz))

    base = trimesh.creation.box(extents=(sx, sy, sz))
    tri = np.asarray(base.triangles)
    n = tri.shape[0]

    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)

    # make it two-sided: add reverse-wound faces (share same vertices/UVs)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    v0 = tri[:, 0]; v1 = tri[:, 1]; v2 = tri[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)

    # Create UVs for vertices (one UV per vertex, not per face)
    # Since we're using the same vertices for two-sided faces, we only need n*3 UVs
    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    for i in range(n):
        nrm = normals[i]
        axis = int(np.argmax(np.abs(nrm)))
        if axis == 0:       # ±X => (y,z)
            a, b, ea, eb = 1, 2, sy, sz
        elif axis == 1:     # ±Y => (x,z)
            a, b, ea, eb = 0, 2, sx, sz
        else:               # ±Z => (x,y)
            a, b, ea, eb = 0, 1, sx, sy
        for j in range(3):
            v = tri[i, j]
            uvs[i*3+j, 0] = ((v[a] / ea) + 0.5) * tile_uv
            uvs[i*3+j, 1] = ((v[b] / eb) + 0.5) * tile_uv

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))  # GLB by extension


def _fix_uv_seam(uvs, faces, u_col=0):
    """Fix atan2 wraparound seam: for each triangle, if the u-spread > 0.5,
    vertices with u < 0.5 get +1.0 so the triangle doesn't stretch across
    the whole texture.  Works in-place."""
    for f in faces:
        us = [uvs[f[j], u_col] for j in range(3)]
        if max(us) - min(us) > 0.5:
            for j in range(3):
                if uvs[f[j], u_col] < 0.5:
                    uvs[f[j], u_col] += 1.0

def _create_textured_cylinder_glb(radius, length, material_set: dict, out_glb: Path, sections=32):
    """Create a GLB cylinder (Z-aligned) with cylindrical UVs and PBR textures."""
    base = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    tri = np.asarray(base.triangles)
    n = tri.shape[0]
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    # Hybrid UVs: planar XY for cap faces, cylindrical for side faces
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len < 1e-12] = 1.0
    normals = normals / norms_len
    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    for i in range(n):
        is_cap = abs(normals[i, 2]) > 0.9  # face normal mostly along Z
        for j in range(3):
            x, y, z = vertices[i * 3 + j]
            if is_cap:
                # planar projection for caps
                uvs[i * 3 + j] = [x / (2 * radius) + 0.5, y / (2 * radius) + 0.5]
            else:
                u = (math.atan2(y, x) / (2 * math.pi)) % 1.0
                v = (z / length) + 0.5
                uvs[i * 3 + j] = [u, v]
    _fix_uv_seam(uvs, faces)

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _create_textured_sphere_glb(radius, material_set: dict, out_glb: Path, subdivisions=3):
    """Create a GLB sphere with spherical UVs and PBR textures."""
    base = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    tri = np.asarray(base.triangles)
    n = tri.shape[0]
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    # spherical UVs
    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    for i in range(n * 3):
        x, y, z = vertices[i]
        u = (math.atan2(y, x) / (2 * math.pi)) % 1.0
        v = (math.asin(np.clip(z / max(radius, 1e-9), -1, 1)) / math.pi) + 0.5
        uvs[i] = [u, v]
    _fix_uv_seam(uvs, faces)

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _create_hollow_cylinder_mesh(outer_r, inner_r, height, sections=32):
    """Build a tube mesh: outer wall, inner wall, top ring, bottom ring."""
    hz = height / 2.0
    angles = np.linspace(0, 2*np.pi, sections, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)

    verts = []
    faces = []

    # Helper: add a ring of quads between two rings of vertices
    def add_ring(ring_top, ring_bot):
        n = len(ring_top)
        base = len(verts)
        verts.extend(ring_top)
        verts.extend(ring_bot)
        for i in range(n):
            j = (i + 1) % n
            ti, tj = base + i, base + j
            bi, bj = base + n + i, base + n + j
            faces.append([ti, bj, tj])   # upper triangle
            faces.append([ti, bi, bj])   # lower triangle

    # Outer wall (normals point outward naturally)
    outer_top = [(outer_r*ca[i], outer_r*sa[i], hz) for i in range(sections)]
    outer_bot = [(outer_r*ca[i], outer_r*sa[i], -hz) for i in range(sections)]
    add_ring(outer_top, outer_bot)

    # Inner wall (reversed winding so normals point inward)
    inner_top = [(inner_r*ca[i], inner_r*sa[i], hz) for i in range(sections)]
    inner_bot = [(inner_r*ca[i], inner_r*sa[i], -hz) for i in range(sections)]
    add_ring(inner_bot, inner_top)  # swap top/bot to flip normals

    # Top ring (annulus connecting outer_top to inner_top)
    top_outer = [(outer_r*ca[i], outer_r*sa[i], hz) for i in range(sections)]
    top_inner = [(inner_r*ca[i], inner_r*sa[i], hz) for i in range(sections)]
    add_ring(top_outer, top_inner)

    # Bottom ring (annulus connecting inner_bot to outer_bot)
    bot_inner = [(inner_r*ca[i], inner_r*sa[i], -hz) for i in range(sections)]
    bot_outer = [(outer_r*ca[i], outer_r*sa[i], -hz) for i in range(sections)]
    add_ring(bot_inner, bot_outer)

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def _create_textured_hollow_cylinder_glb(outer_r, inner_r, length,
                                          material_set: dict, out_glb: Path, sections=32):
    """Create a GLB hollow cylinder (tube) with cylindrical UVs and PBR textures."""
    base = _create_hollow_cylinder_mesh(outer_r, inner_r, length, sections)
    tri = np.asarray(base.triangles)
    n = tri.shape[0]
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_len[norms_len < 1e-12] = 1.0
    normals = normals / norms_len
    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    for i in range(n):
        is_cap = abs(normals[i, 2]) > 0.9
        for j in range(3):
            x, y, z = vertices[i * 3 + j]
            if is_cap:
                uvs[i * 3 + j] = [x / (2 * outer_r) + 0.5, y / (2 * outer_r) + 0.5]
            else:
                u = (math.atan2(y, x) / (2 * math.pi)) % 1.0
                v = (z / max(length, 1e-9)) + 0.5
                uvs[i * 3 + j] = [u, v]
    _fix_uv_seam(uvs, faces)

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _create_half_torus_mesh(arc_radius, tube_radius, arc_sections=32, tube_sections=12):
    """Build a half-torus mesh: semicircular arc from +X through +Z to -X."""
    verts = []
    faces = []
    # theta: arc angle 0..pi, phi: tube cross-section 0..2pi
    for i in range(arc_sections + 1):
        theta = math.pi * i / arc_sections
        ct, st = math.cos(theta), math.sin(theta)
        for j in range(tube_sections):
            phi = 2 * math.pi * j / tube_sections
            cp, sp = math.cos(phi), math.sin(phi)
            r = arc_radius + tube_radius * cp
            x = r * ct
            z = r * st
            y = tube_radius * sp
            verts.append([x, y, z])

    # Build quads between adjacent arc rings
    for i in range(arc_sections):
        for j in range(tube_sections):
            j_next = (j + 1) % tube_sections
            v00 = i * tube_sections + j
            v01 = i * tube_sections + j_next
            v10 = (i + 1) * tube_sections + j
            v11 = (i + 1) * tube_sections + j_next
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

def _create_textured_half_torus_glb(arc_radius, tube_radius, material_set: dict, out_glb: Path):
    """Create a GLB half-torus with cylindrical UVs and PBR textures."""
    base = _create_half_torus_mesh(arc_radius, tube_radius)
    tri = np.asarray(base.triangles)
    n = tri.shape[0]
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    for i in range(n * 3):
        x, y, z = vertices[i]
        # u = arc progress (0 at +X, 1 at -X)
        u = math.atan2(z, x) / math.pi
        u = np.clip(u, 0, 1)
        # v = tube angle
        v = (math.atan2(y, math.sqrt(x*x + z*z) - arc_radius) / (2 * math.pi)) % 1.0
        uvs[i] = [u, v]
    _fix_uv_seam(uvs, faces, u_col=1)  # seam is on v (tube angle)

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _create_panel_with_hole_mesh(width, height, thickness, hole_radius,
                                  hole_cx=0.0, hole_cy=0.0, sections=32):
    """Rectangular panel (X=width, Z=height, Y=thickness) with a circular hole.
    hole_cx/hole_cy are the hole center in the panel's XZ plane.
    Returns a trimesh.Trimesh."""
    hw, hh, ht = width / 2, height / 2, thickness / 2
    # Outer rectangle corners (in XZ plane)
    rect = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    # Hole circle points (in XZ plane)
    angles = [2 * math.pi * i / sections for i in range(sections)]
    hole_pts = [(hole_cx + hole_radius * math.cos(a),
                 hole_cy + hole_radius * math.sin(a)) for a in angles]

    verts = []
    faces = []

    # Front face (Y = +ht) and back face (Y = -ht).
    # Delaunay triangulation of outer rect + hole points, then filter out
    # triangles whose centroid falls inside the hole (circular check).
    from scipy.spatial import Delaunay
    coords_2d = np.array(rect, dtype=np.float64)
    hole_2d = np.array(hole_pts, dtype=np.float64)
    all_2d = np.vstack([coords_2d, hole_2d])
    tri_2d = Delaunay(all_2d)
    face_tris = []
    for simplex in tri_2d.simplices:
        cx = np.mean(all_2d[simplex, 0])
        cz = np.mean(all_2d[simplex, 1])
        # Centroid must be inside the rectangle AND outside the hole
        if (-hw <= cx <= hw and -hh <= cz <= hh and
                math.sqrt((cx - hole_cx)**2 + (cz - hole_cy)**2) > hole_radius):
            face_tris.append(simplex)

    # Front face (Y = +ht)
    v_off = len(verts)
    for x, z in all_2d:
        verts.append((x, ht, z))
    for tri in face_tris:
        faces.append([v_off + tri[0], v_off + tri[1], v_off + tri[2]])

    # Back face (Y = -ht) — reversed winding
    v_off = len(verts)
    for x, z in all_2d:
        verts.append((x, -ht, z))
    for tri in face_tris:
        faces.append([v_off + tri[2], v_off + tri[1], v_off + tri[0]])

    # Outer edges (4 rectangular sides).
    for i in range(4):
        x0, z0 = rect[i]
        x1, z1 = rect[(i + 1) % 4]
        v_off = len(verts)
        verts.extend([
            (x0, ht, z0), (x1, ht, z1),    # front edge
            (x0, -ht, z0), (x1, -ht, z1),  # back edge
        ])
        faces.append([v_off, v_off + 1, v_off + 3])
        faces.append([v_off, v_off + 3, v_off + 2])

    # Hole inner wall (cylinder connecting the front and back hole rims).
    for i in range(sections):
        j = (i + 1) % sections
        x0, z0 = hole_pts[i]
        x1, z1 = hole_pts[j]
        v_off = len(verts)
        verts.extend([
            (x0, ht, z0), (x1, ht, z1),    # front rim
            (x0, -ht, z0), (x1, -ht, z1),  # back rim
        ])
        # Winding inward (normals point into the hole = toward center)
        faces.append([v_off, v_off + 2, v_off + 3])
        faces.append([v_off, v_off + 3, v_off + 1])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _create_textured_panel_with_hole_glb(width, height, thickness, hole_radius,
                                          hole_cx, hole_cy, material_set: dict,
                                          out_glb: Path, sections=32):
    """Create a GLB panel-with-hole with planar UVs on faces, cylindrical on hole wall, PBR textures."""
    base = _create_panel_with_hole_mesh(width, height, thickness, hole_radius,
                                         hole_cx, hole_cy, sections)
    tri = np.asarray(base.triangles)
    n = tri.shape[0]
    vertices = tri.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])

    uvs = np.zeros((n * 3, 2), dtype=np.float32)
    tri_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nl = np.linalg.norm(tri_normals, axis=1, keepdims=True); nl[nl < 1e-12] = 1.0
    tri_normals = tri_normals / nl

    for i in range(n):
        is_face = abs(tri_normals[i, 1]) > 0.9  # front/back face normal along Y
        for j in range(3):
            x, y, z = vertices[i * 3 + j]
            if is_face:
                # Planar XZ mapping
                uvs[i * 3 + j] = [x / width + 0.5, z / height + 0.5]
            else:
                # Check if it's the hole wall or outer edge
                dx = x - hole_cx
                dz = z - hole_cy
                dist = math.sqrt(dx * dx + dz * dz)
                if abs(dist - hole_radius) < hole_radius * 0.5:
                    # Hole wall: cylindrical UV
                    u = (math.atan2(dz, dx) / (2 * math.pi)) % 1.0
                    v = (y / thickness) + 0.5
                    uvs[i * 3 + j] = [u, v]
                else:
                    # Outer edge: planar projection
                    # Use the dominant axis of the edge
                    nx, ny, nz = tri_normals[i]
                    if abs(nx) > abs(nz):
                        uvs[i * 3 + j] = [y / thickness + 0.5, z / height + 0.5]
                    else:
                        uvs[i * 3 + j] = [x / width + 0.5, y / thickness + 0.5]
    _fix_uv_seam(uvs, faces)

    visual = _make_pbr_visual(uvs, material_set)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _parse_vec3(s: str):
    vals = [float(v) for v in s.strip().split()] if s else [0.0,0.0,0.0]
    while len(vals) < 3: vals.append(0.0)
    return tuple(vals[:3])

def _rpy_to_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    R = Rz @ Ry @ Rx
    M = np.eye(4); M[:3,:3] = R
    return M

def _T_from_xyzrpy(xyz, rpy):
    M = _rpy_to_mat(*rpy)
    M[:3,3] = np.array(xyz)
    return M

def _get_visual_geom(link_elem):
    """Unified visual geometry extractor — returns dict with 'type' key:
       'box':      size=(sx,sy,sz)
       'cylinder': radius, length  (aligned along Z)
       'sphere':   radius
    Also includes xyz, rpy, vis, geom, shape_elem (the XML element).
    """
    vis = link_elem.find("visual")
    if vis is None: return None
    geom = vis.find("geometry")
    if geom is None: return None
    ori = vis.find("origin")
    xyz = _parse_vec3(ori.attrib.get("xyz","0 0 0")) if ori is not None else (0.0,0.0,0.0)
    rpy = _parse_vec3(ori.attrib.get("rpy","0 0 0")) if ori is not None else (0.0,0.0,0.0)
    base = {"xyz": xyz, "rpy": rpy, "vis": vis, "geom": geom}

    box = geom.find("box")
    if box is not None:
        sx, sy, sz = _parse_vec3(box.attrib.get("size","0 0 0"))
        return {**base, "type": "box", "size": (sx,sy,sz), "shape_elem": box}

    hcyl = geom.find("hollow_cylinder")
    if hcyl is not None:
        r = float(hcyl.attrib.get("radius","0"))
        ir = float(hcyl.attrib.get("inner_radius","0"))
        l = float(hcyl.attrib.get("length","0"))
        return {**base, "type": "hollow_cylinder", "radius": r, "inner_radius": ir,
                "length": l, "shape_elem": hcyl}

    cyl = geom.find("cylinder")
    if cyl is not None:
        r = float(cyl.attrib.get("radius","0"))
        l = float(cyl.attrib.get("length","0"))
        return {**base, "type": "cylinder", "radius": r, "length": l, "shape_elem": cyl}

    sph = geom.find("sphere")
    if sph is not None:
        r = float(sph.attrib.get("radius","0"))
        return {**base, "type": "sphere", "radius": r, "shape_elem": sph}

    ht = geom.find("half_torus")
    if ht is not None:
        ar = float(ht.attrib.get("arc_radius","0"))
        tr = float(ht.attrib.get("tube_radius","0"))
        return {**base, "type": "half_torus", "arc_radius": ar, "tube_radius": tr, "shape_elem": ht}

    pwh = geom.find("panel_with_hole")
    if pwh is not None:
        w = float(pwh.attrib.get("width","0"))
        h = float(pwh.attrib.get("height","0"))
        t = float(pwh.attrib.get("thickness","0"))
        hr = float(pwh.attrib.get("hole_radius","0"))
        hcx = float(pwh.attrib.get("hole_cx","0"))
        hcy = float(pwh.attrib.get("hole_cy","0"))
        sec = int(pwh.attrib.get("sections","32"))
        return {**base, "type": "panel_with_hole", "width": w, "height": h,
                "thickness": t, "hole_radius": hr, "hole_cx": hcx, "hole_cy": hcy,
                "sections": sec, "shape_elem": pwh}

    return None

def _trimesh_for_spec(sp):
    """Create a trimesh primitive from a geometry spec, apply its transform."""
    T = sp["T"]
    gtype = sp.get("type", "box")
    if gtype == "box":
        sx, sy, sz = sp["size"]
        m = trimesh.creation.box(extents=(sx, sy, sz))
    elif gtype == "cylinder":
        m = trimesh.creation.cylinder(radius=sp["radius"], height=sp["length"], sections=32)
    elif gtype == "sphere":
        m = trimesh.creation.icosphere(radius=sp["radius"], subdivisions=3)
    elif gtype == "hollow_cylinder":
        m = _create_hollow_cylinder_mesh(sp["radius"], sp["inner_radius"], sp["length"])
    elif gtype == "half_torus":
        m = _create_half_torus_mesh(sp["arc_radius"], sp["tube_radius"])
    elif gtype == "panel_with_hole":
        m = _create_panel_with_hole_mesh(sp["width"], sp["height"], sp["thickness"],
                                          sp["hole_radius"], sp.get("hole_cx", 0.0),
                                          sp.get("hole_cy", 0.0), sp.get("sections", 32))
    else:
        m = trimesh.creation.box(extents=sp.get("size", (0.01,0.01,0.01)))
    m.apply_transform(T)
    return m

def _uv_for_triangles(tri, normals, sp):
    """Compute planar/cylindrical UVs for triangle soup depending on geometry type."""
    n = tri.shape[0]
    gtype = sp.get("type", "box")
    uvs = np.zeros((n * 3, 2), dtype=np.float32)

    if gtype == "box":
        sx, sy, sz = sp["size"]
        local_tile = _suggest_tile_uv((sx, sy, sz))
        for i in range(n):
            nrm = normals[i]
            axis = int(np.argmax(np.abs(nrm)))
            if axis == 0: a,bax,ea,eb = 1,2, sy,sz
            elif axis == 1: a,bax,ea,eb = 0,2, sx,sz
            else: a,bax,ea,eb = 0,1, sx,sy
            for j in range(3):
                v = tri[i,j]
                uvs[i*3+j,0] = ((v[a]/max(ea,1e-9))+0.5) * local_tile
                uvs[i*3+j,1] = ((v[bax]/max(eb,1e-9))+0.5) * local_tile
    elif gtype == "cylinder":
        length = sp["length"]
        radius = sp["radius"]
        tri_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nl = np.linalg.norm(tri_normals, axis=1, keepdims=True); nl[nl < 1e-12] = 1.0
        tri_normals = tri_normals / nl
        for i in range(n):
            is_cap = abs(tri_normals[i, 2]) > 0.9
            for j in range(3):
                x, y, z = tri[i, j]
                if is_cap:
                    uvs[i*3+j] = [x / (2*radius) + 0.5, y / (2*radius) + 0.5]
                else:
                    uvs[i*3+j, 0] = (math.atan2(y, x) / (2 * math.pi)) % 1.0
                    uvs[i*3+j, 1] = (z / max(length, 1e-9)) + 0.5
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        _fix_uv_seam(uvs, faces)
    elif gtype == "sphere":
        radius = sp["radius"]
        for i in range(n * 3):
            x, y, z = tri.reshape(-1,3)[i]
            uvs[i,0] = (math.atan2(y, x) / (2 * math.pi)) % 1.0
            uvs[i,1] = (math.asin(np.clip(z / max(radius, 1e-9), -1, 1)) / math.pi) + 0.5
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        _fix_uv_seam(uvs, faces)
    elif gtype == "hollow_cylinder":
        length = sp["length"]
        outer_r = sp["radius"]
        tri_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nl = np.linalg.norm(tri_normals, axis=1, keepdims=True); nl[nl < 1e-12] = 1.0
        tri_normals = tri_normals / nl
        for i in range(n):
            is_cap = abs(tri_normals[i, 2]) > 0.9
            for j in range(3):
                x, y, z = tri[i, j]
                if is_cap:
                    uvs[i*3+j] = [x / (2*outer_r) + 0.5, y / (2*outer_r) + 0.5]
                else:
                    uvs[i*3+j, 0] = (math.atan2(y, x) / (2 * math.pi)) % 1.0
                    uvs[i*3+j, 1] = (z / max(length, 1e-9)) + 0.5
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        _fix_uv_seam(uvs, faces)
    elif gtype == "half_torus":
        arc_r = sp["arc_radius"]
        for i in range(n * 3):
            x, y, z = tri.reshape(-1,3)[i]
            uvs[i,0] = np.clip(math.atan2(z, x) / math.pi, 0, 1)
            uvs[i,1] = (math.atan2(y, math.sqrt(x*x+z*z) - arc_r) / (2*math.pi)) % 1.0
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        _fix_uv_seam(uvs, faces, u_col=1)
    elif gtype == "panel_with_hole":
        pw = sp["width"]; ph = sp["height"]; pt = sp["thickness"]
        p_hr = sp["hole_radius"]; p_hcx = sp.get("hole_cx", 0.0); p_hcy = sp.get("hole_cy", 0.0)
        tri_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nl = np.linalg.norm(tri_normals, axis=1, keepdims=True); nl[nl < 1e-12] = 1.0
        tri_normals = tri_normals / nl
        for i in range(n):
            is_face = abs(tri_normals[i, 1]) > 0.9
            for j in range(3):
                x, y, z = tri[i, j]
                if is_face:
                    uvs[i*3+j] = [x / max(pw, 1e-9) + 0.5, z / max(ph, 1e-9) + 0.5]
                else:
                    dx = x - p_hcx; dz = z - p_hcy
                    dist = math.sqrt(dx*dx + dz*dz)
                    if abs(dist - p_hr) < p_hr * 0.5:
                        uvs[i*3+j, 0] = (math.atan2(dz, dx) / (2*math.pi)) % 1.0
                        uvs[i*3+j, 1] = (y / max(pt, 1e-9)) + 0.5
                    else:
                        nx, ny, nz = tri_normals[i]
                        if abs(nx) > abs(nz):
                            uvs[i*3+j] = [y / max(pt, 1e-9) + 0.5, z / max(ph, 1e-9) + 0.5]
                        else:
                            uvs[i*3+j] = [x / max(pw, 1e-9) + 0.5, y / max(pt, 1e-9) + 0.5]
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        _fix_uv_seam(uvs, faces)

    return uvs

def _create_merged_boxes_glb(specs, material_set: dict, out_glb: Path, tile_uv=1.0):
    if not specs: return
    tris = []; uvs = []
    for sp in specs:
        m = _trimesh_for_spec(sp)
        tri = np.asarray(m.triangles)

        v0, v1, v2 = tri[:,0], tri[:,1], tri[:,2]
        normals = np.cross(v1 - v0, v2 - v0)
        normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)

        face_uvs = _uv_for_triangles(tri, normals, sp)
        tris.append(tri); uvs.append(face_uvs)

    tri = np.concatenate(tris, axis=0)
    uv  = np.concatenate(uvs,  axis=0)
    verts = tri.reshape(-1,3)
    faces = np.arange(verts.shape[0], dtype=np.int64).reshape(-1,3)
    faces_twosided = np.vstack([faces, faces[:, ::-1]])
    visual = _make_pbr_visual(uv, material_set)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces_twosided, visual=visual, process=False)
    mesh.export(str(out_glb))


def _urdf_find_root_link(root):
    links = {L.attrib["name"] for L in root.findall("link")}
    children = {J.find("child").attrib["link"] for J in root.findall("joint")}
    roots = list(links - children)
    return roots[0] if roots else None

def _to_mat(xyz=(0,0,0), rpy=(0,0,0)):
    return _T_from_xyzrpy(xyz, rpy)

def _corner_points(size):
    sx, sy, sz = size
    hx, hy, hz = sx*0.5, sy*0.5, sz*0.5
    corners = []
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                corners.append(np.array([dx, dy, dz, 1.0], dtype=np.float64))
    return corners

def _bounding_points_cylinder(radius, length):
    """Axis-aligned bounding points for a Z-aligned cylinder."""
    hz = length * 0.5
    pts = []
    for dx in (-radius, radius):
        for dy in (-radius, radius):
            for dz in (-hz, hz):
                pts.append(np.array([dx, dy, dz, 1.0], dtype=np.float64))
    return pts

def _bounding_points_sphere(radius):
    pts = []
    for dx in (-radius, radius):
        for dy in (-radius, radius):
            for dz in (-radius, radius):
                pts.append(np.array([dx, dy, dz, 1.0], dtype=np.float64))
    return pts

def _bounding_points_for_geom(vg):
    """Return bounding-box sample points for any geometry type."""
    if vg["type"] == "box":
        return _corner_points(vg["size"])
    elif vg["type"] in ("cylinder", "hollow_cylinder"):
        return _bounding_points_cylinder(vg["radius"], vg["length"])
    elif vg["type"] == "sphere":
        return _bounding_points_sphere(vg["radius"])
    elif vg["type"] == "half_torus":
        # Arches in XZ from +X to -X; bounding box is [-R,R] x [-tr,tr] x [0,R]
        R = vg["arc_radius"] + vg["tube_radius"]
        tr = vg["tube_radius"]
        pts = []
        for dx in (-R, R):
            for dy in (-tr, tr):
                for dz in (-tr, R):  # bottom can dip slightly below 0
                    pts.append(np.array([dx, dy, dz, 1.0], dtype=np.float64))
        return pts
    elif vg["type"] == "panel_with_hole":
        hw = vg["width"] / 2; hh = vg["height"] / 2; ht = vg["thickness"] / 2
        pts = []
        for dx in (-hw, hw):
            for dy in (-ht, ht):
                for dz in (-hh, hh):
                    pts.append(np.array([dx, dy, dz, 1.0], dtype=np.float64))
        return pts
    return []

def _collect_joint_maps(root):
    joints = {}
    parents = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        typ  = j.attrib.get("type","fixed")
        p = j.find("parent").attrib["link"]; c = j.find("child").attrib["link"]
        ori = j.find("origin")
        xyz = _parse_vec3(ori.attrib.get("xyz","0 0 0")) if ori is not None else (0.0,0.0,0.0)
        rpy = _parse_vec3(ori.attrib.get("rpy","0 0 0")) if ori is not None else (0.0,0.0,0.0)
        joints[name] = {"type":typ, "parent":p, "child":c, "xyz":xyz, "rpy":rpy, "elem":j}
        parents[c] = name
    return joints, parents

def _forward_kin_rest(root, joints, parents, root_link):
    T_link = {root_link: np.eye(4)}
    child_by_parent = {}
    for j in joints.values():
        child_by_parent.setdefault(j["parent"], []).append(j)
    stack = [root_link]
    while stack:
        L = stack.pop()
        for j in child_by_parent.get(L, []):
            T_pc = _to_mat(j["xyz"], j["rpy"])
            T_link[j["child"]] = T_link[L] @ T_pc
            stack.append(j["child"])
    return T_link

def _compute_rest_aabb(root):
    links_by_name = {L.attrib["name"]: L for L in root.findall("link")}
    joints, parents = _collect_joint_maps(root)
    root_link = _urdf_find_root_link(root)
    if root_link is None:
        raise RuntimeError("Cannot find URDF root link")
    T_link = _forward_kin_rest(root, joints, parents, root_link)
    pmin = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float64)
    pmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for lname, L in links_by_name.items():
        vg = _get_visual_geom(L)
        if vg is None:
            continue
        T = T_link.get(lname, np.eye(4)) @ _to_mat(vg["xyz"], vg["rpy"])
        pts = _bounding_points_for_geom(vg)
        for c in pts:
            pw = (T @ c)[:3]
            pmin = np.minimum(pmin, pw)
            pmax = np.maximum(pmax, pw)
    return pmin, pmax, root_link

def _scale_all_geometry_and_origins(root, scale):
    for j in root.findall("joint"):
        ori = j.find("origin")
        if ori is not None and "xyz" in ori.attrib:
            x,y,z = _parse_vec3(ori.attrib.get("xyz","0 0 0"))
            ori.set("xyz", f"{x*scale:.6f} {y*scale:.6f} {z*scale:.6f}")

        jtype = j.attrib.get("type", "fixed")
        lim = j.find("limit")
        if lim is not None and jtype == "prismatic":
            if "lower" in lim.attrib:
                lv = float(lim.attrib["lower"])
                lim.set("lower", f"{lv*scale:.6f}")
            if "upper" in lim.attrib:
                uv = float(lim.attrib["upper"])
                lim.set("upper", f"{uv*scale:.6f}")

    for L in root.findall("link"):
        for tag in ("visual","collision"):
            v = L.find(tag)
            if v is None: continue
            ori = v.find("origin")
            if ori is not None and "xyz" in ori.attrib:
                x,y,z = _parse_vec3(ori.attrib.get("xyz","0 0 0"))
                ori.set("xyz", f"{x*scale:.6f} {y*scale:.6f} {z*scale:.6f}")
            geom = v.find("geometry")
            if geom is None: continue
            box = geom.find("box")
            if box is not None and "size" in box.attrib:
                sx,sy,sz = _parse_vec3(box.attrib.get("size","0 0 0"))
                box.set("size", f"{sx*scale:.6f} {sy*scale:.6f} {sz*scale:.6f}")
            hcyl = geom.find("hollow_cylinder")
            if hcyl is not None:
                if "radius" in hcyl.attrib:
                    hcyl.set("radius", f"{float(hcyl.attrib['radius'])*scale:.6f}")
                if "inner_radius" in hcyl.attrib:
                    hcyl.set("inner_radius", f"{float(hcyl.attrib['inner_radius'])*scale:.6f}")
                if "length" in hcyl.attrib:
                    hcyl.set("length", f"{float(hcyl.attrib['length'])*scale:.6f}")
            cyl = geom.find("cylinder")
            if cyl is not None:
                if "radius" in cyl.attrib:
                    cyl.set("radius", f"{float(cyl.attrib['radius'])*scale:.6f}")
                if "length" in cyl.attrib:
                    cyl.set("length", f"{float(cyl.attrib['length'])*scale:.6f}")
            sph = geom.find("sphere")
            if sph is not None and "radius" in sph.attrib:
                sph.set("radius", f"{float(sph.attrib['radius'])*scale:.6f}")
            ht = geom.find("half_torus")
            if ht is not None:
                if "arc_radius" in ht.attrib:
                    ht.set("arc_radius", f"{float(ht.attrib['arc_radius'])*scale:.6f}")
                if "tube_radius" in ht.attrib:
                    ht.set("tube_radius", f"{float(ht.attrib['tube_radius'])*scale:.6f}")
            pwh = geom.find("panel_with_hole")
            if pwh is not None:
                for attr in ("width", "height", "thickness", "hole_radius", "hole_cx", "hole_cy"):
                    if attr in pwh.attrib:
                        pwh.set(attr, f"{float(pwh.attrib[attr])*scale:.6f}")

def _recenter_in_place(root: ET.Element, root_link_name: str, offset_xyz):
    """
    Apply a global translation to the model without introducing a new node:
    - shift all joints whose parent == root_link by 'offset_xyz'
    - shift the root_link's visual/collision/inertial origins by 'offset_xyz'
    This is equivalent to placing a fixed joint with that offset *above* the root.
    """
    ox, oy, oz = map(float, offset_xyz)

    # 1) Shift all joints that have the root as parent
    for J in root.findall("joint"):
        parent = J.find("parent").attrib["link"]
        if parent != root_link_name:
            continue
        ori = J.find("origin")
        if ori is None:
            ori = ET.SubElement(J, "origin", attrib={"xyz":"0 0 0", "rpy":"0 0 0"})
        x, y, z = _parse_vec3(ori.attrib.get("xyz", "0 0 0"))
        ori.set("xyz", f"{x+ox:.6f} {y+oy:.6f} {z+oz:.6f}")

    # 2) Shift the root link's visuals/collisions/inertial origins
    links = {L.attrib["name"]: L for L in root.findall("link")}
    Lroot = links.get(root_link_name)
    if Lroot is not None:
        # inertial origin
        inertial = Lroot.find("inertial")
        if inertial is not None:
            io = inertial.find("origin")
            if io is None:
                io = ET.SubElement(inertial, "origin", attrib={"xyz":"0 0 0", "rpy":"0 0 0"})
            x, y, z = _parse_vec3(io.attrib.get("xyz", "0 0 0"))
            io.set("xyz", f"{x+ox:.6f} {y+oy:.6f} {z+oz:.6f}")

        # visual / collision origins
        for tag in ("visual", "collision"):
            for v in Lroot.findall(tag):
                ori = v.find("origin")
                if ori is None:
                    ori = ET.SubElement(v, "origin", attrib={"xyz":"0 0 0", "rpy":"0 0 0"})
                x, y, z = _parse_vec3(ori.attrib.get("xyz", "0 0 0"))
                ori.set("xyz", f"{x+ox:.6f} {y+oy:.6f} {z+oz:.6f}")


def _normalize_urdf_geometry(urdf_path: Path, target_max=1.0, recenter=True):
    """
    Scale geometry to fit target_max and optionally recenter the whole model
    *without* inserting a new root link.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    # compute AABB in rest pose
    pmin, pmax, root_link = _compute_rest_aabb(root)
    size = pmax - pmin
    max_side = float(max(size[0], size[1], size[2]))
    if max_side <= 0:
        raise RuntimeError("Degenerate geometry: zero size for normalization")

    # scale everything
    scale = target_max / max_side
    _scale_all_geometry_and_origins(root, scale)

    # recompute center after scale
    center = (pmin + pmax) * 0.5 * scale

    if recenter and root_link is not None:
        # apply global translation in-place (no extra node)
        offset = -center
        _recenter_in_place(root, root_link, offset)

    tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)


def _strip_visual_material(vis: ET.Element):
    """Remove an inline <material> so the GLB’s embedded texture is used."""
    m = vis.find("material")
    if m is not None:
        vis.remove(m)


def _meshify_urdf_boxes(urdf_path: Path, materials_root: Path, seed=None):
    # Per-asset material seed (a str, since hash() changes between Python processes).
    random.seed(f"{seed}/{urdf_path.parent.name}")
    sets = _scan_material_sets(materials_root)
    if len(sets) >= 3:
        pair = random.sample(range(len(sets)), 3)
    elif len(sets) >= 2:
        pair = random.sample(range(len(sets)), 2)
        pair.append(pair[0])  # reuse first as interior fallback
    else:
        pair = [0, 0, 0]

    used_material_indices = set()

    def choose_tex():
        idx = random.choice(pair)
        used_material_indices.add(idx)
        return sets[idx]

    def choose_interior_tex():
        """The interior material (index 2 of the sampled sets)."""
        used_material_indices.add(pair[2])
        return sets[pair[2]]

    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    out_dir = urdf_path.parent
    textured_objs_dir = out_dir / "textured_objs"
    textured_objs_dir.mkdir(parents=True, exist_ok=True)

    links = {L.attrib["name"]: L for L in root.findall("link")}
    joints = list(root.findall("joint"))

    # Local helper: treat doors and only the drawer ROOT as dynamic
    def _is_dynamic(name: str) -> bool:
        if not name:
            return False
        if _is_door_link(name):
            return True
        if _is_drawer_root(name):      # existing drawer pattern
            return True
        return False

    # Fixed-joint maps.
    fixed_map = {}          # child -> (parent, T_parent_to_child)
    fixed_children = {}     # parent -> [children]
    fixed_T = {}            # (parent, child) -> T
    for j in joints:
        if j.attrib.get("type") != "fixed":
            continue
        parent = j.find("parent").attrib["link"]
        child  = j.find("child").attrib["link"]
        ori    = j.find("origin")
        xyz = _parse_vec3(ori.attrib.get("xyz", "0 0 0")) if ori is not None else (0.0, 0.0, 0.0)
        rpy = _parse_vec3(ori.attrib.get("rpy", "0 0 0")) if ori is not None else (0.0, 0.0, 0.0)
        T = _T_from_xyzrpy(xyz, rpy)
        fixed_map[child] = (parent, T)
        fixed_children.setdefault(parent, []).append(child)
        fixed_T[(parent, child)] = T

    def _spec_from_vg(vg, T):
        """Build a geometry spec dict from a visual-geom dict + world transform."""
        sp = {"type": vg["type"], "T": T}
        if vg["type"] == "box": sp["size"] = vg["size"]
        elif vg["type"] == "cylinder": sp["radius"] = vg["radius"]; sp["length"] = vg["length"]
        elif vg["type"] == "hollow_cylinder":
            sp["radius"] = vg["radius"]; sp["inner_radius"] = vg["inner_radius"]; sp["length"] = vg["length"]
        elif vg["type"] == "sphere": sp["radius"] = vg["radius"]
        elif vg["type"] == "half_torus":
            sp["arc_radius"] = vg["arc_radius"]; sp["tube_radius"] = vg["tube_radius"]
        elif vg["type"] == "panel_with_hole":
            sp["width"] = vg["width"]; sp["height"] = vg["height"]; sp["thickness"] = vg["thickness"]
            sp["hole_radius"] = vg["hole_radius"]; sp["hole_cx"] = vg["hole_cx"]; sp["hole_cy"] = vg["hole_cy"]
            sp["sections"] = vg.get("sections", 32)
        return sp

    def _collect_static_specs_from_base(base_name: str):
        """
        DFS over *fixed* joints; gather all non-dynamic boxes as specs in base frame.
        Also return prunable link names (all visited non-dynamic links except the base itself).
        """
        if base_name not in links:
            return [], [], set()
        exterior_specs = []
        interior_specs = []
        prunable = set()
        stack = [(base_name, np.eye(4))]
        visited = set()

        while stack:
            lname, Tacc = stack.pop()
            if lname in visited:
                continue
            visited.add(lname)

            le = links.get(lname)
            if le is not None:
                vg = _get_visual_geom(le)
                # If this link hosts a visual *and* is not dynamic, absorb it and mark prunable
                if vg is not None and not _is_dynamic(lname):
                    T_vis = _T_from_xyzrpy(vg["xyz"], vg["rpy"])
                    sp = _spec_from_vg(vg, Tacc @ T_vis)
                    # Check material name to separate interior/exterior
                    mat_elem = vg["vis"].find("material")
                    mat_name = mat_elem.attrib.get("name", "") if mat_elem is not None else ""
                    if mat_name == "mat_interior":
                        interior_specs.append(sp)
                    else:
                        exterior_specs.append(sp)
                    # remove so it won't be exported twice
                    le.remove(vg["vis"])
                    # Protect unit kinematic roots: *_base_inertial_dummy must stay
                    if lname != base_name and not lname.endswith("_base_inertial_dummy"):
                        prunable.add(lname)

            # traverse fixed children but do not descend into dynamic subtrees
            for ch in fixed_children.get(lname, []):
                if _is_dynamic(ch):
                    continue
                T_next = Tacc @ fixed_T[(lname, ch)]
                stack.append((ch, T_next))

        return exterior_specs, interior_specs, prunable

    def _collect_drawer_specs(root_link_name: str, links, fixed_children, fixed_T):
        """
        Collect all box visuals under a drawer root (in the drawer root frame),
        following only fixed joints. Returns (exterior_specs, interior_specs).
        Also removes the original <visual> nodes so they won't be processed again.
        """
        if root_link_name not in links:
            return [], []

        exterior_specs = []
        interior_specs = []
        stack = [(root_link_name, np.eye(4))]  # (link_name, T_root_to_link)
        visited = set()

        while stack:
            lname, Tacc = stack.pop()
            if lname in visited:
                continue
            visited.add(lname)

            le = links.get(lname)
            if le is not None:
                vg = _get_visual_geom(le)
                if vg is not None:
                    T_vis = _T_from_xyzrpy(vg["xyz"], vg["rpy"])
                    sp = _spec_from_vg(vg, Tacc @ T_vis)
                    mat_elem = vg["vis"].find("material")
                    mat_name = mat_elem.attrib.get("name", "") if mat_elem is not None else ""
                    if mat_name == "mat_interior":
                        interior_specs.append(sp)
                    else:
                        exterior_specs.append(sp)
                    # remove so it won't be exported again
                    le.remove(vg["vis"])

            for ch in fixed_children.get(lname, []):
                # We *do* want to traverse all fixed children within the drawer,
                # even if their names contain "_drawer_" etc.
                T_next = Tacc @ fixed_T[(lname, ch)]
                stack.append((ch, T_next))

        return exterior_specs, interior_specs

    # bases that receive merged static meshes
    base_names = []
    if "base_inertial_dummy" in links:
        base_names.append("base_inertial_dummy")
    base_names += [n for n in links if n.endswith("_base")]
    # Fallback: if no known base found, use URDF root link
    if not base_names and "stack_root" not in links:
        root_link = _urdf_find_root_link(root)
        if root_link and root_link in links:
            base_names.append(root_link)

    # Helper: accumulate fixed transform up to an ancestor (parent frame -> ancestor frame)
    def _T_to_ancestor(child_name: str, ancestor_name: str):
        T = np.eye(4)
        cur = child_name
        # Walk up via fixed joints only
        while cur in fixed_map:
            parent, Tpc = fixed_map[cur]            # T_parent->child
            T = Tpc @ T                              # compose up the chain
            if parent == ancestor_name:
                return T
            cur = parent
        return None

    did_stack_merge = False
    if ("stack_root" in links) and ("A_base" in links) and ("B_base" in links):
        # Collect static pieces from both bases in their local base frames
        ext_A, int_A, prunable_A = _collect_static_specs_from_base("A_base")
        ext_B, int_B, prunable_B = _collect_static_specs_from_base("B_base")

        # Lift both sets into the stack_root frame
        TA = _T_to_ancestor("A_base", "stack_root")
        if TA is None:
            TA = np.eye(4)

        TB = _T_to_ancestor("B_base", "stack_root")
        if TB is None:
            TB = np.eye(4)

        def _lift_specs(specs, T):
            return [{**sp, "T": T @ sp["T"]} for sp in specs]

        ext_specs = _lift_specs(ext_A, TA) + _lift_specs(ext_B, TB)
        int_specs = _lift_specs(int_A, TA) + _lift_specs(int_B, TB)

        # Attach merged meshes on stack_root
        for v in list(links["stack_root"].findall("visual")):
            links["stack_root"].remove(v)

        if ext_specs:
            glb_name = "stack_static.glb"
            glb_path = textured_objs_dir / glb_name
            _create_merged_boxes_glb(ext_specs, choose_tex(), glb_path)
            vis = ET.SubElement(links["stack_root"], "visual")
            ET.SubElement(vis, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", attrib={"filename": f"textured_objs/{glb_name}"})
            _strip_visual_material(vis)

        if int_specs:
            glb_name_int = "stack_static_interior.glb"
            glb_path_int = textured_objs_dir / glb_name_int
            _create_merged_boxes_glb(int_specs, choose_interior_tex(), glb_path_int)
            vis_int = ET.SubElement(links["stack_root"], "visual")
            ET.SubElement(vis_int, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom_int = ET.SubElement(vis_int, "geometry")
            ET.SubElement(geom_int, "mesh", attrib={"filename": f"textured_objs/{glb_name_int}"})
            _strip_visual_material(vis_int)

        # Prune the fixed-only static children we merged
        protected = {
            "A_base", "B_base",
            "A_base_inertial_dummy", "B_base_inertial_dummy",
        }
        to_prune = (prunable_A | prunable_B) - protected
        _prune_fixed_subtree(root, to_prune, _is_dynamic)

        did_stack_merge = True

    # Fallback: per-base merged statics only if we did NOT do the stack merge
    if not did_stack_merge:
        for base in base_names:
            ext_specs, int_specs, prunable = _collect_static_specs_from_base(base)
            if not ext_specs and not int_specs:
                continue

            # wipe existing visuals on base
            for v in list(links[base].findall("visual")):
                links[base].remove(v)

            if base == "base_inertial_dummy":
                glb_base = "static"
            else:
                glb_base = base.rstrip('_').replace("_base", "") + "static"

            # Exterior mesh
            if ext_specs:
                glb_name = f"{glb_base}.glb"
                glb_path = textured_objs_dir / glb_name
                _create_merged_boxes_glb(ext_specs, choose_tex(), glb_path)
                vis = ET.SubElement(links[base], "visual")
                ET.SubElement(vis, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
                geom = ET.SubElement(vis, "geometry")
                ET.SubElement(geom, "mesh", attrib={"filename": f"textured_objs/{glb_name}"})
                _strip_visual_material(vis)

            # Interior mesh (different texture)
            if int_specs:
                glb_name_int = f"{glb_base}_interior.glb"
                glb_path_int = textured_objs_dir / glb_name_int
                _create_merged_boxes_glb(int_specs, choose_interior_tex(), glb_path_int)
                vis_int = ET.SubElement(links[base], "visual")
                ET.SubElement(vis_int, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
                geom_int = ET.SubElement(vis_int, "geometry")
                ET.SubElement(geom_int, "mesh", attrib={"filename": f"textured_objs/{glb_name_int}"})
                _strip_visual_material(vis_int)

            _prune_fixed_subtree(root, prunable, _is_dynamic)

    links = {L.attrib["name"]: L for L in root.findall("link")}
    # Merge each drawer subtree into one GLB attached to its root.
    for lname in list(links.keys()):
        if not _is_drawer_root(lname):
            continue

        # collect all simple boxes in the drawer subtree (in drawer root frame)
        ext_specs, int_specs = _collect_drawer_specs(lname, links, fixed_children, fixed_T)
        if not ext_specs and not int_specs:
            continue

        # wipe existing visuals on the drawer root (if any) and attach merged meshes
        for v in list(links[lname].findall("visual")):
            links[lname].remove(v)

        if ext_specs:
            glb_name = f"{lname}.glb"
            glb_path = textured_objs_dir / glb_name
            _create_merged_boxes_glb(ext_specs, choose_tex(), glb_path)
            vis = ET.SubElement(links[lname], "visual")
            ET.SubElement(vis, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", attrib={"filename": f"textured_objs/{glb_name}"})
            _strip_visual_material(vis)

        if int_specs:
            glb_name_int = f"{lname}_interior.glb"
            glb_path_int = textured_objs_dir / glb_name_int
            _create_merged_boxes_glb(int_specs, choose_interior_tex(), glb_path_int)
            vis_int = ET.SubElement(links[lname], "visual")
            ET.SubElement(vis_int, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom_int = ET.SubElement(vis_int, "geometry")
            ET.SubElement(geom_int, "mesh", attrib={"filename": f"textured_objs/{glb_name_int}"})
            _strip_visual_material(vis_int)

        _, _, drawer_fixed_children, _ = _build_fixed_maps(root)
        prunable_drawer = set()
        stack = [lname]
        seen = set([lname])
        while stack:
            p = stack.pop()
            for j, c in drawer_fixed_children.get(p, []):
                if c in seen: 
                    continue
                seen.add(c)
                prunable_drawer.add(c)
                stack.append(c)
        # remove those children unless they’re dynamic (there shouldn’t be)
        _prune_fixed_subtree(root, prunable_drawer, _is_dynamic)
    
    # Refresh after the drawer merges and pruning.
    links = {L.attrib["name"]: L for L in root.findall("link")}


    # Merge the fixed children of door links into the door's GLB.
    # For door links that have fixed children (e.g. fan cross blade),
    # collect all child specs and merge into one GLB with the same texture.
    door_merged = set()  # track links already merged into a door
    for lname, link in list(links.items()):
        if not _is_door_link(lname):
            continue
        vg = _get_visual_geom(link)
        if vg is None:
            continue
        # Check if this door link has any fixed children with visuals
        child_specs = []
        child_stack = [(lname, np.eye(4))]
        child_visited = {lname}
        while child_stack:
            cur, Tacc = child_stack.pop()
            for ch in fixed_children.get(cur, []):
                if ch in child_visited:
                    continue
                child_visited.add(ch)
                T_ch = Tacc @ fixed_T[(cur, ch)]
                ch_le = links.get(ch)
                if ch_le is not None:
                    ch_vg = _get_visual_geom(ch_le)
                    if ch_vg is not None:
                        T_cv = _T_from_xyzrpy(ch_vg["xyz"], ch_vg["rpy"])
                        child_specs.append((ch_le, ch_vg, _spec_from_vg(ch_vg, T_ch @ T_cv)))
                child_stack.append((ch, T_ch))

        if child_specs:
            # Build merged spec list: door link + its fixed children
            T_vis = _T_from_xyzrpy(vg["xyz"], vg["rpy"])
            specs = [_spec_from_vg(vg, T_vis)]
            link.remove(vg["vis"])  # remove only when we're merging
            for ch_le, ch_vg, ch_sp in child_specs:
                specs.append(ch_sp)
                ch_le.remove(ch_vg["vis"])
                door_merged.add(ch_le.attrib.get("name", ""))

            glb_name = f"{lname}.glb"
            glb_path = textured_objs_dir / glb_name
            tex = choose_tex()
            _create_merged_boxes_glb(specs, tex, glb_path)
            # Re-add mesh ref to the door link
            vis_new = ET.SubElement(link, "visual")
            ET.SubElement(vis_new, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom_new = ET.SubElement(vis_new, "geometry")
            ET.SubElement(geom_new, "mesh", attrib={"filename": f"textured_objs/{glb_name}"})
            door_merged.add(lname)

    # Texture the remaining primitives (dynamic links etc.).
    for lname, link in links.items():
        if lname in door_merged:
            continue
        vg = _get_visual_geom(link)
        if vg is None:
            continue
        glb_name = f"{lname}.glb"
        glb_path = textured_objs_dir / glb_name
        tex = choose_tex()
        if vg["type"] == "box":
            _create_textured_box_glb(vg["size"], tex, glb_path)
        elif vg["type"] == "cylinder":
            _create_textured_cylinder_glb(vg["radius"], vg["length"], tex, glb_path)
        elif vg["type"] == "hollow_cylinder":
            _create_textured_hollow_cylinder_glb(vg["radius"], vg["inner_radius"], vg["length"], tex, glb_path)
        elif vg["type"] == "sphere":
            _create_textured_sphere_glb(vg["radius"], tex, glb_path)
        elif vg["type"] == "half_torus":
            _create_textured_half_torus_glb(vg["arc_radius"], vg["tube_radius"], tex, glb_path)
        elif vg["type"] == "panel_with_hole":
            _create_textured_panel_with_hole_glb(
                vg["width"], vg["height"], vg["thickness"],
                vg["hole_radius"], vg["hole_cx"], vg["hole_cy"], tex, glb_path,
                vg.get("sections", 32))
        else:
            continue
        vg["geom"].remove(vg["shape_elem"])
        ET.SubElement(vg["geom"], "mesh", attrib={"filename": f"textured_objs/{glb_name}"})
        _strip_visual_material(vg["vis"])

    tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)
    # Only save materials that were actually used
    meta_tex = {
        "materials_root": str(materials_root),
        "available_pair": [{"index": i, "name": sets[i]["name"], "path": str(sets[i]["root"])} for i in pair],
        "used_materials": [{"index": i, "name": sets[i]["name"], "path": str(sets[i]["root"])} for i in sorted(used_material_indices)]
    }
    (out_dir / "texture_manifest.json").write_text(json.dumps(meta_tex, indent=2))


def _write_bounding_box_json(urdf_path: Path, out_dir: Path):
    # compute AABB *after* normalization, while visuals are still boxes
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    pmin, pmax, _ = _compute_rest_aabb(root)
    bb = {"min": list(map(float, pmin.tolist())), "max": list(map(float, pmax.tolist()))}
    (out_dir / "bounding_box.json").write_text(json.dumps(bb, indent=2))

def _write_meta_json(out_dir: Path, family: str):
    # model_id = folder name; the other fields are placeholders.
    model_id = out_dir.name
    meta = {
        "model_id":   model_id,
        "anno_id":    "000",
        "user_id":    "datagen",
        "time_in_sec":"1",
        "version":    "1",
        "model_cat":  family,     # e.g., "hinged_cabinet", "drawer_cabinet", ...
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

def _build_fixed_maps(root):
    """Return links dict, joints list, fixed_children map, and by_child joint map for *fixed* joints only."""
    links = {L.attrib["name"]: L for L in root.findall("link")}
    joints = list(root.findall("joint"))

    fixed_children = {}
    fixed_by_child = {}
    for j in joints:
        if j.attrib.get("type") != "fixed":
            continue
        p = j.find("parent").attrib["link"]
        c = j.find("child").attrib["link"]
        fixed_children.setdefault(p, []).append((j, c))
        fixed_by_child[c] = j
    return links, joints, fixed_children, fixed_by_child

def _prune_fixed_subtree(root, prunable_roots, is_dynamic):
    """
    Remove links reachable via *fixed* joints from each name in prunable_roots,
    stopping whenever we hit anything dynamic (doors/drawers) or a joint that is not fixed.
    Also removes the fixed joints that attach the deleted links.
    """
    links, joints, fixed_children, fixed_by_child = _build_fixed_maps(root)

    to_delete_links = set()
    to_delete_joints = set()

    stack = list(prunable_roots)
    seen = set()

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)

        # Never delete dynamic leaves (door links or drawer root)
        if is_dynamic(node):
            continue

        L = links.get(node)
        if L is None:
            continue

        # Mark link + the fixed joint that attached it (if any)
        to_delete_links.add(node)
        j = fixed_by_child.get(node)
        if j is not None:
            to_delete_joints.add(j)

        # Descend only along fixed joints
        for j_child, child in fixed_children.get(node, []):
            stack.append(child)

    # Physically remove from the XML tree
    for j in to_delete_joints:
        root.remove(j)
    for name in to_delete_links:
        root.remove(links[name])

def _export_result_json(urdf_path: Path, out_dir: Path, family: str):
    # 1) read mobility_v2 to get stable ids that dataloader uses
    mob_path = out_dir / "mobility_v2.json"
    mob = json.loads(mob_path.read_text())
    name_to_id = {node["name"]: int(node["id"]) for node in mob}

    # 2) decide the body mesh basename actually written on disk
    #    (single unit => static.glb, stacked => stack_static.glb)
    body_glb = "stack_static" if (out_dir / "textured_objs/stack_static.glb").exists() else "static"

    # 3) collect door/drawer link names directly from mobility (by joint type)
    doors   = [n["name"] for n in mob if n.get("joint") == "hinge"]
    drawers = [n["name"] for n in mob if n.get("joint") == "slider"]

    # 4) Build the taxonomy tree with ids that MATCH mobility_v2.json
    # Body uses the mobility id of the link that hosts the main static frame.
    # Generic root detection: prefer stack_root > base_inertial_dummy > URDF root
    if "stack_root" in name_to_id:
        body_link_name = "stack_root"
    elif "base_inertial_dummy" in name_to_id:
        body_link_name = "base_inertial_dummy"
    else:
        # fallback: the static root from mobility_v2
        body_link_name = next(n["name"] for n in mob if n["parent"] == -1)

    frame_node = {
        "text": "Frame",
        "objs": [body_glb],                       # <- GLB basename, not link name
        "id":   name_to_id[body_link_name],       # <- mobility id
        "name": "frame"
    }

    door_nodes = [{
        "text": "Door",
        "objs": [dn],                             # GLB basename equals link name
        "id":   name_to_id[dn],                   # mobility id of the door link
        "name": dn
    } for dn in sorted(doors)]

    drawer_nodes = [{
        "text": "Drawer",
        "objs": [drn],                            # GLB basename equals link name
        "id":   name_to_id[drn],                  # mobility id of the drawer root
        "name": drn
    } for drn in sorted(drawers)]

    body_node   = {"text": "Body",    "children": [frame_node], "id": name_to_id[body_link_name], "name":"body"}
    doors_node  = {"text": "Doors",   "children": door_nodes,   "id": (name_to_id[doors[0]] if doors else 0),   "name":"doors"} if doors else None
    drawers_node= {"text": "Drawers", "children": drawer_nodes, "id": (name_to_id[drawers[0]] if drawers else 0),"name":"drawers"} if drawers else None

    children = [body_node] + ([doors_node] if doors_node else []) + ([drawers_node] if drawers_node else [])
    # Use the family name for the root taxonomy label (generic, not hardcoded to "Cabinet")
    root_label = family.replace("_", " ").title()
    tree = [{
        "text": root_label,
        "children": [{
            "text": root_label,
            "children": children,
            "id": name_to_id[body_link_name], "name": family
        }],
        "id": name_to_id[body_link_name], "name": family
    }]

    (out_dir / "result.json").write_text(json.dumps(tree, indent=2))

def _export_mobility_v2_json(urdf_path: Path, out_dir: Path):
    """
    Export a compact mobility_v2.json:
    - Keep only: root + dynamic leaves (doors, drawer roots)
    - Reparent through fixed scaffolding and transform joint origins accordingly
    """
    def _rad2deg(x):
        return float(x) * 180.0 / math.pi

    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    links = {L.attrib["name"]: L for L in root.findall("link")}
    joints_xml = list(root.findall("joint"))

    # joint dicts
    joints = []
    parent_of = {}
    for J in joints_xml:
        jtype = J.attrib.get("type", "fixed")
        name  = J.attrib["name"]
        parent = J.find("parent").attrib["link"]
        child  = J.find("child").attrib["link"]
        ori    = J.find("origin")
        xyz = _parse_vec3(ori.attrib.get("xyz","0 0 0")) if ori is not None else (0.0,0.0,0.0)
        rpy = _parse_vec3(ori.attrib.get("rpy","0 0 0")) if ori is not None else (0.0,0.0,0.0)
        axis_el = J.find("axis"); axis = (0.0,0.0,0.0)
        if axis_el is not None:
            axis = _parse_vec3(axis_el.attrib.get("xyz","0 0 0"))
        lim = J.find("limit")
        lower = upper = None
        if lim is not None:
            lower = float(lim.attrib.get("lower","0"))
            upper = float(lim.attrib.get("upper","0"))
        joints.append({
            "name": name, "type": jtype, "parent": parent, "child": child,
            "origin_xyz": xyz, "origin_rpy": rpy, "axis": axis, "lower": lower, "upper": upper
        })
        parent_of[child] = parent

    # URDF root link: link with no parent joint
    link_names = set(links.keys())
    children = set(parent_of.keys())
    urdf_root_link = (list(link_names - children)[0] if (link_names - children) else next(iter(link_names)))

    # Prefer these as the exported root if present
    exported_root = "stack_root" if "stack_root" in links else (
        "base_inertial_dummy" if "base_inertial_dummy" in links else urdf_root_link
    )

    # Fixed-joint transforms (parent -> child).
    fixed_children = {}
    fixed_T = {}  # (parent, child) -> 4x4
    fixed_parent = {}  # child -> parent (fixed only)
    for j in joints:
        if j["type"] != "fixed":
            continue
        p, c = j["parent"], j["child"]
        T = _T_from_xyzrpy(j["origin_xyz"], j["origin_rpy"])
        fixed_children.setdefault(p, []).append(c)
        fixed_T[(p, c)] = T
        fixed_parent[c] = p

    def T_to_ancestor(start_child: str, ancestor: str):
        """
        Compose transforms along FIXED joints from 'ancestor' -> 'start_child'.
        Returns T_ancestor_to_child, or None if chain doesn't exist purely via fixed joints.
        """
        T = np.eye(4)
        cur = start_child
        while cur in fixed_parent:
            p = fixed_parent[cur]
            T = fixed_T[(p, cur)] @ T   # parent->child pre-multiplies
            if p == ancestor:
                return T
            cur = p
        return None

    def is_dynamic_link(name: str) -> bool:
        return _is_door_link(name) or _is_drawer_root(name)

    kept_links = {exported_root} | {j["child"] for j in joints if is_dynamic_link(j["child"])}

    nodes = []
    name_to_nodeid = {}

    def add_node(name, joint_kind, parent_idx, parts, axis=None, origin=None, lower=None, upper=None):
        nid = len(nodes)
        name_to_nodeid[name] = nid
        entry = {
            "id": nid,
            "parent": int(parent_idx) if parent_idx is not None else -1,
            "joint": joint_kind,
            "name": name,
            "parts": [{"id": nid, "name": parts, "children": []}],
            "jointData": {}
        }
        if joint_kind != "static":
            entry["jointData"]["axis"] = {
                "origin": list(map(float, origin)) if origin is not None else [0.0,0.0,0.0],
                "direction": list(map(float, axis)) if axis is not None else [0.0,0.0,0.0]
            }
            no_lim = (lower is None) or (upper is None)
            if joint_kind == "hinge":
                a = _rad2deg(lower or 0.0); b = _rad2deg(upper or 0.0)
            else:  # "slider"
                a = float(lower or 0.0); b = float(upper or 0.0)
            entry["jointData"]["limit"] = {"a": a, "b": b, "noLimit": bool(no_lim)}
        nodes.append(entry)
        return nid

    # root first
    root_idx = add_node(exported_root, "static", None, parts=exported_root)

    def kind_of(jt):
        if jt in ("revolute", "continuous"):
            return "hinge"
        if jt == "prismatic":
            return "slider"
        return "static"

    # index joints by child for quick lookup
    joint_by_child = {j["child"]: j for j in joints}

    # Emit only dynamic children, reparenting through fixed scaffolding if needed
    for child in sorted(kept_links):
        if child == exported_root:
            continue
        if child not in joint_by_child:
            # orphan dynamic link (shouldn't happen), attach to root with zero origin
            add_node(child, "hinge", root_idx, parts=child, axis=(0,0,1), origin=(0,0,0), lower=0, upper=0)
            continue

        J = joint_by_child[child]
        if not is_dynamic_link(child):
            continue  # skip scaffolding

        # find nearest kept ancestor as parent
        parent = J["parent"]
        parent_kept = parent in kept_links
        parent_idx = name_to_nodeid.get(parent if parent_kept else exported_root, root_idx)

        # If parent is not kept, climb up via fixed joints until we hit a kept ancestor (or root)
        if not parent_kept:
            # walk fixed parents
            cur = parent
            target = exported_root
            while cur in fixed_parent:
                p = fixed_parent[cur]
                if p in kept_links:
                    target = p
                    break
                cur = p
            parent_idx = name_to_nodeid.get(target, root_idx)
            # Transform the joint origin from original parent frame into 'target' frame
            Tpa = T_to_ancestor(J["parent"], target)
            if Tpa is None:
                Tpa = np.eye(4)
            T_joint = _T_from_xyzrpy(J["origin_xyz"], J["origin_rpy"])
            T_new = Tpa @ T_joint
            origin_xyz = tuple(T_new[:3, 3].tolist())
            # rotate axis into new parent frame
            R_new = T_new[:3, :3]
            axis = (R_new @ np.array(J["axis"], dtype=float)).tolist()
        else:
            origin_xyz = J["origin_xyz"]
            axis = J["axis"]

        add_node(
            child,
            kind_of(J["type"]),
            parent_idx,
            parts=child,
            axis=axis,
            origin=origin_xyz,
            lower=J["lower"],
            upper=J["upper"],
        )

    (out_dir / "mobility_v2.json").write_text(json.dumps(nodes, indent=2))


def _normalize_joint_ranges(urdf_path: Path):
    """
    Normalize all joint ranges to use positive values by flipping axes if needed.
    - Revolute joints with negative ranges: flip axis and invert range to positive
    - Prismatic joints with negative ranges: flip axis and invert range to positive
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    
    for joint in root.findall("joint"):
        jtype = joint.attrib.get("type", "fixed")
        if jtype not in ("revolute", "prismatic", "continuous"):
            continue
        
        limit = joint.find("limit")
        axis = joint.find("axis")
        
        if limit is None or axis is None:
            continue
        
        lower = float(limit.attrib.get("lower", "0"))
        upper = float(limit.attrib.get("upper", "0"))
        
        # Check if range is negative (lower < 0 and upper <= 0)
        if lower < 0 and upper <= 0:
            # Flip the axis
            axis_xyz = axis.attrib.get("xyz", "0 0 0")
            ax, ay, az = [float(v) for v in axis_xyz.split()]
            axis.set("xyz", f"{-ax:.4f} {-ay:.4f} {-az:.4f}")
            
            # Invert the range
            limit.set("lower", f"{-upper:.6f}")
            limit.set("upper", f"{-lower:.6f}")
    
    tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)


def build_door(seed=None):
    """Standalone room door with frame.
    Variants: single_swing, double_swing, sliding, sliding_up."""
    random.seed(seed)
    W = rnd(0.70, 1.80)   # frame width
    H = rnd(1.80, 2.60)   # frame height
    D = rnd(0.10, 0.20)   # frame depth
    bar = rnd(0.04, 0.08)
    gap = rnd(0.002, 0.005)
    panel_thk = rnd(0.030, 0.050)

    body_col = rnd_color()
    panel_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", panel_col)

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.5))

    # Frame: top + left + right (no bottom bar — it's a doorway)
    links.append(link_box_xml("frame_top", (W, D, bar), origin_xyz=(0, 0, +H/2 - bar/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_frame_top", "base_inertial_dummy", "frame_top"))
    links.append(link_box_xml("frame_left", (bar, D, H - bar), origin_xyz=(-W/2 + bar/2, 0, -bar/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_frame_left", "base_inertial_dummy", "frame_left"))
    links.append(link_box_xml("frame_right", (bar, D, H - bar), origin_xyz=(+W/2 - bar/2, 0, -bar/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_frame_right", "base_inertial_dummy", "frame_right"))

    inner_w = W - 2*bar - 2*gap
    inner_h = H - bar - 2*gap
    theta_cap = math.radians(rnd(80, 110))

    variant = random.choice(["single_swing", "double_swing", "sliding", "sliding_up"])

    kr = rnd(0.015, 0.035)  # knob radius for door

    if variant == "single_swing":
        side = random.choice(["left", "right"])
        if side == "left":
            x_hinge = -inner_w/2
            x_vis = +inner_w/2
            axis_z = -1
        else:
            x_hinge = +inner_w/2
            x_vis = -inner_w/2
            axis_z = 1
        links.append(link_box_xml("panel_door", (inner_w, panel_thk, inner_h),
                                  origin_xyz=(x_vis, 0, 0), mass=1.5, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_panel", "base_inertial_dummy", "panel_door",
                                         origin_xyz=(x_hinge, 0, -gap - bar/2),
                                         axis_xyz=(0, 0, axis_z), lower=0.0, upper=theta_cap))
        # Knob on front face, opposite hinge side
        knob_x_sign = -1 if side == "left" else +1  # toward hinge side in vis coords = opposite in world
        knob_x = x_vis + knob_x_sign * inner_w * 0.35
        links.append(link_sphere_xml("knob", kr,
                     origin_xyz=(knob_x, -panel_thk/2 - kr, 0.0),
                     mass=0.02, material="mat_door"))
        joints.append(fixed_joint_xml("fix_knob", "panel_door", "knob"))
    elif variant == "double_swing":
        half_w = (inner_w - gap) / 2.0
        links.append(link_box_xml("panel_left_door", (half_w, panel_thk, inner_h),
                                  origin_xyz=(half_w/2, 0, 0), mass=1.0, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_left", "base_inertial_dummy", "panel_left_door",
                                         origin_xyz=(-inner_w/2, 0, -gap - bar/2),
                                         axis_xyz=(0, 0, -1), lower=0.0, upper=theta_cap))
        # Left panel knob — near free edge (right side of panel, +X in vis)
        links.append(link_sphere_xml("knob_left", kr,
                     origin_xyz=(half_w * 0.85, -panel_thk/2 - kr, 0.0),
                     mass=0.02, material="mat_door"))
        joints.append(fixed_joint_xml("fix_knob_left", "panel_left_door", "knob_left"))
        links.append(link_box_xml("panel_right_door", (half_w, panel_thk, inner_h),
                                  origin_xyz=(-half_w/2, 0, 0), mass=1.0, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_right", "base_inertial_dummy", "panel_right_door",
                                         origin_xyz=(+inner_w/2, 0, -gap - bar/2),
                                         axis_xyz=(0, 0, 1), lower=0.0, upper=theta_cap))
        # Right panel knob — near free edge (left side of panel, -X in vis)
        links.append(link_sphere_xml("knob_right", kr,
                     origin_xyz=(-half_w * 0.85, -panel_thk/2 - kr, 0.0),
                     mass=0.02, material="mat_door"))
        joints.append(fixed_joint_xml("fix_knob_right", "panel_right_door", "knob_right"))
    elif variant == "sliding":
        slide_range = inner_w * rnd(0.5, 0.9)
        # Fixed panel (right half)
        half_w = inner_w / 2.0
        links.append(link_box_xml("panel_fixed", (half_w, panel_thk, inner_h),
                                  origin_xyz=(+half_w/2, D*0.1, 0), mass=1.0, material="mat_door"))
        joints.append(fixed_joint_xml("fix_panel_fixed", "base_inertial_dummy", "panel_fixed",
                                      origin_xyz=(0, 0, -gap - bar/2)))
        # Sliding panel (left half)
        links.append(link_box_xml("slide_drawer_0", (half_w, panel_thk, inner_h),
                                  origin_xyz=(0, -D*0.1, 0), mass=1.0, material="mat_door"))
        joints.append(prismatic_joint_xml("slide_panel", "base_inertial_dummy", "slide_drawer_0",
                                          origin_xyz=(-half_w/2, 0, -gap - bar/2),
                                          axis_xyz=(1, 0, 0), lower=0.0, upper=slide_range))
    else:  # sliding_up
        # Two panels stacked vertically — top slides up, staggered in Y so they don't collide
        half_h = inner_h / 2.0
        slide_range = half_h * rnd(0.5, 0.9)
        y_off = panel_thk/2 + gap
        z_base = -gap - bar/2  # Z center of inner opening
        # Fixed bottom panel (sits in front)
        links.append(link_box_xml("panel_fixed", (inner_w, panel_thk, half_h),
                                  origin_xyz=(0, -y_off, -half_h/2), mass=1.0, material="mat_door"))
        joints.append(fixed_joint_xml("fix_panel_fixed", "base_inertial_dummy", "panel_fixed",
                                      origin_xyz=(0, 0, z_base)))
        # Sliding top panel (sits behind) — slides up
        links.append(link_box_xml("slide_drawer_0", (inner_w, panel_thk, half_h),
                                  origin_xyz=(0, 0, 0), mass=1.0, material="mat_door"))
        joints.append(prismatic_joint_xml("slide_panel", "base_inertial_dummy", "slide_drawer_0",
                                          origin_xyz=(0, +y_off, z_base + half_h/2),
                                          axis_xyz=(0, 0, 1), lower=0.0, upper=slide_range))

    robot = '<robot name="door">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "door", "variant": variant, "dims": {"W": W, "H": H, "D": D, "bar": bar}}
    return robot, meta


def build_table(seed=None):
    """Table with legs and an articulated top part.
    Variants: drop_leaf_single, drop_leaf_double, drawer, fold_up."""
    random.seed(seed)
    W = rnd(0.80, 1.80)   # width (X)
    De = rnd(0.50, 1.20)  # depth (Y)
    H = rnd(0.65, 0.85)   # total height (Z)
    top_thk = rnd(0.025, 0.045)
    leg_w = rnd(0.035, 0.065)
    gap = rnd(0.002, 0.005)  # unused, but drawn to keep the random stream unchanged

    body_col = rnd_color()
    leaf_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", leaf_col)

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.3))

    leg_h = H - top_thk
    leg_inset_x = rnd(0.02, 0.06)
    leg_inset_y = rnd(0.02, 0.06)

    # 4 legs
    for i, (sx, sy) in enumerate([(-1, -1), (-1, +1), (+1, -1), (+1, +1)]):
        lx = sx * (W/2 - leg_inset_x - leg_w/2)
        ly = sy * (De/2 - leg_inset_y - leg_w/2)
        lz = -top_thk/2 - leg_h/2
        links.append(link_box_xml(f"leg_{i}", (leg_w, leg_w, leg_h),
                                  origin_xyz=(lx, ly, lz), mass=0.5, material="mat_body"))
        joints.append(fixed_joint_xml(f"fix_leg_{i}", "base_inertial_dummy", f"leg_{i}"))

    # Apron rails connecting the legs (just below tabletop)
    apron_h = rnd(0.04, 0.08)
    apron_thk = rnd(0.015, 0.025)
    apron_z = -top_thk/2 - apron_h/2 - 0.005
    lx_pos = W/2 - leg_inset_x - leg_w/2
    ly_pos = De/2 - leg_inset_y - leg_w/2
    apron_len_x = 2 * lx_pos - leg_w
    apron_len_y = 2 * ly_pos - leg_w
    links.append(link_box_xml("apron_front", (apron_len_x, apron_thk, apron_h),
                              origin_xyz=(0, -ly_pos, apron_z), mass=0.3, material="mat_body"))
    joints.append(fixed_joint_xml("fix_apron_front", "base_inertial_dummy", "apron_front"))
    links.append(link_box_xml("apron_back", (apron_len_x, apron_thk, apron_h),
                              origin_xyz=(0, +ly_pos, apron_z), mass=0.3, material="mat_body"))
    joints.append(fixed_joint_xml("fix_apron_back", "base_inertial_dummy", "apron_back"))
    links.append(link_box_xml("apron_left", (apron_thk, apron_len_y, apron_h),
                              origin_xyz=(-lx_pos, 0, apron_z), mass=0.3, material="mat_body"))
    joints.append(fixed_joint_xml("fix_apron_left", "base_inertial_dummy", "apron_left"))
    links.append(link_box_xml("apron_right", (apron_thk, apron_len_y, apron_h),
                              origin_xyz=(+lx_pos, 0, apron_z), mass=0.3, material="mat_body"))
    joints.append(fixed_joint_xml("fix_apron_right", "base_inertial_dummy", "apron_right"))

    variant = random.choice(["drop_leaf_single", "drop_leaf_double", "drawer", "fold_up"])

    if variant in ("drop_leaf_single", "drop_leaf_double"):
        # Main top always covers the full width (all legs)
        links.append(link_box_xml("tabletop", (W, De, top_thk),
                                  origin_xyz=(0, 0, 0), mass=1.5, material="mat_body"))
        joints.append(fixed_joint_xml("fix_tabletop", "base_inertial_dummy", "tabletop"))

        # Leaves extend BEYOND the main top
        n_leaves = 2 if variant == "drop_leaf_double" else 1
        if n_leaves == 2:
            leaf_w_left = rnd(0.08, W * 0.35)
            leaf_w_right = rnd(0.08, W * 0.35)
        else:
            side = random.choice(["left", "right"])
            leaf_w_single = rnd(0.08, W * 0.40)

        if n_leaves == 2 or side == "left":
            lw = leaf_w_left if n_leaves == 2 else leaf_w_single
            links.append(link_box_xml("leaf_left_door", (lw, De, top_thk),
                                      origin_xyz=(-lw/2, 0, 0), mass=0.5, material="mat_door"))
            joints.append(revolute_joint_xml("hinge_leaf_left", "base_inertial_dummy", "leaf_left_door",
                                             origin_xyz=(-W/2, 0, 0),
                                             axis_xyz=(0, 1, 0), lower=0.0, upper=math.radians(90)))
        if n_leaves == 2 or side == "right":
            lw = leaf_w_right if n_leaves == 2 else leaf_w_single
            links.append(link_box_xml("leaf_right_door", (lw, De, top_thk),
                                      origin_xyz=(lw/2, 0, 0), mass=0.5, material="mat_door"))
            joints.append(revolute_joint_xml("hinge_leaf_right", "base_inertial_dummy", "leaf_right_door",
                                             origin_xyz=(+W/2, 0, 0),
                                             axis_xyz=(0, -1, 0), lower=0.0, upper=math.radians(90)))

    elif variant == "drawer":
        # Full tabletop + drawer underneath that slides out front
        links.append(link_box_xml("tabletop", (W, De, top_thk),
                                  origin_xyz=(0, 0, 0), mass=1.5, material="mat_body"))
        joints.append(fixed_joint_xml("fix_tabletop", "base_inertial_dummy", "tabletop"))

        inner_w = apron_len_x  # space between legs
        drawer_w = inner_w * rnd(0.30, 0.70)
        drawer_d = De * rnd(0.50, 0.80)
        drawer_h = min(rnd(0.08, 0.18), apron_h - 0.01)  # fit within apron
        slide_range = drawer_d * rnd(0.6, 0.9)
        x_off = 0.0  # centered along the long side
        knob_r_table = min(rnd(0.015, 0.030), drawer_h * 0.4)
        n_knobs_table = random.choice([1, 2])
        links.append(link_box_xml("table_drawer_0", (drawer_w, drawer_d, drawer_h),
                                  origin_xyz=(0, 0, 0), mass=0.4, material="mat_door"))
        joints.append(prismatic_joint_xml("slide_drawer", "base_inertial_dummy", "table_drawer_0",
                                          origin_xyz=(x_off, -De/2 + drawer_d/2, -top_thk/2 - drawer_h/2 - 0.005),
                                          axis_xyz=(0, -1, 0), lower=0.0, upper=slide_range))
        # Sphere knobs on drawer front face
        for ki in range(n_knobs_table):
            if n_knobs_table == 1:
                kx = 0.0
            else:
                kx = (-0.25 + 0.5 * ki) * drawer_w
            kn = f"knob_0_{ki}"
            links.append(link_sphere_xml(kn, knob_r_table,
                         origin_xyz=(kx, -drawer_d/2 - knob_r_table, 0.0),
                         mass=0.02, material="mat_door"))
            joints.append(fixed_joint_xml(f"fix_{kn}", "table_drawer_0", kn))
    else:  # fold_up
        # Wall-mount style: entire top folds up from back hinge
        links.append(link_box_xml("fold_door", (W, De, top_thk),
                                  origin_xyz=(0, -De/2, 0), mass=1.5, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_fold", "base_inertial_dummy", "fold_door",
                                         origin_xyz=(0, De/2, 0),
                                         axis_xyz=(-1, 0, 0), lower=0.0, upper=math.radians(90)))

    robot = '<robot name="table">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "table", "variant": variant,
            "dims": {"W": W, "D": De, "H": H, "top_thk": top_thk}}
    return robot, meta


def build_box(seed=None):
    """Storage box/chest with hinged lid.
    Variants: split_lid, lid_tray."""
    random.seed(seed)
    W = rnd(0.30, 0.90)   # width
    De = rnd(0.25, 0.70)  # depth
    H = rnd(0.25, 0.60)   # body height (without lid)
    wall = rnd(0.015, 0.030)
    lid_thk = rnd(0.020, 0.040)
    gap = rnd(0.001, 0.003)

    body_col = rnd_color()
    lid_col = rnd_color()
    mats = (material_xml("mat_body", body_col) + material_xml("mat_door", lid_col) +
            material_xml("mat_interior", rnd_color()))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.3))

    # Box body: 5 walls (no top — that's the lid)
    links.append(link_box_xml("wall_bottom", (W, De, wall), origin_xyz=(0, 0, -H/2 + wall/2), mass=0.5, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_wall_bottom", "base_inertial_dummy", "wall_bottom"))
    links.append(link_box_xml("wall_front", (W, wall, H - wall), origin_xyz=(0, -De/2 + wall/2, wall/2), mass=0.4, material="mat_body"))
    joints.append(fixed_joint_xml("fix_wall_front", "base_inertial_dummy", "wall_front"))
    links.append(link_box_xml("wall_back", (W, wall, H - wall), origin_xyz=(0, +De/2 - wall/2, wall/2), mass=0.4, material="mat_body"))
    joints.append(fixed_joint_xml("fix_wall_back", "base_inertial_dummy", "wall_back"))
    links.append(link_box_xml("wall_left", (wall, De - 2*wall, H - wall), origin_xyz=(-W/2 + wall/2, 0, wall/2), mass=0.4, material="mat_body"))
    joints.append(fixed_joint_xml("fix_wall_left", "base_inertial_dummy", "wall_left"))
    links.append(link_box_xml("wall_right", (wall, De - 2*wall, H - wall), origin_xyz=(+W/2 - wall/2, 0, wall/2), mass=0.4, material="mat_body"))
    joints.append(fixed_joint_xml("fix_wall_right", "base_inertial_dummy", "wall_right"))

    theta_cap = math.radians(rnd(90, 120))
    variant = random.choice(["split_lid", "lid_tray"])

    if variant == "split_lid":
        # Two halves, each hinged on opposite sides
        half_de = (De - gap) / 2.0
        # Back half — hinged at back
        links.append(link_box_xml("lid_back_door", (W, half_de, lid_thk),
                                  origin_xyz=(0, -half_de/2, 0), mass=0.3, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_lid_back", "base_inertial_dummy", "lid_back_door",
                                         origin_xyz=(0, +De/2 - wall, +H/2 + lid_thk/2),
                                         axis_xyz=(-1, 0, 0), lower=0.0, upper=theta_cap))
        # Front half — hinged at front (axis +X so +Y swings toward +Z = outward)
        links.append(link_box_xml("lid_front_door", (W, half_de, lid_thk),
                                  origin_xyz=(0, half_de/2, 0), mass=0.3, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_lid_front", "base_inertial_dummy", "lid_front_door",
                                         origin_xyz=(0, -De/2 + wall, +H/2 + lid_thk/2),
                                         axis_xyz=(1, 0, 0), lower=0.0, upper=theta_cap))

    else:  # lid_tray
        # Lid + inner tray that lifts out (prismatic Z)
        links.append(link_box_xml("lid_door", (W, De, lid_thk),
                                  origin_xyz=(0, -De/2, 0), mass=0.5, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_lid", "base_inertial_dummy", "lid_door",
                                         origin_xyz=(0, +De/2 - wall, +H/2 + lid_thk/2),
                                         axis_xyz=(-1, 0, 0), lower=0.0, upper=theta_cap))
        # Inner tray
        if not FAMOS_TRAINING:
            tray_h = H * rnd(0.25, 0.40)
            tray_inset = wall + rnd(0.005, 0.010)
            tray_w = W - 2*tray_inset
            tray_d = De - 2*tray_inset
            links.append(link_box_xml("tray_drawer_0", (tray_w, tray_d, tray_h),
                                      origin_xyz=(0, 0, 0), mass=0.3, material="mat_door"))
            joints.append(prismatic_joint_xml("slide_tray", "base_inertial_dummy", "tray_drawer_0",
                                              origin_xyz=(0, 0, H/2 - tray_h/2),
                                              axis_xyz=(0, 0, 1), lower=0.0, upper=H * 0.6))

    robot = '<robot name="box">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "box", "variant": variant,
            "dims": {"W": W, "D": De, "H": H, "wall": wall}}
    return robot, meta


def build_laptop(seed=None):
    """Laptop / book-like hinge: base + screen.
    Variants: standard, book."""
    random.seed(seed)
    W = rnd(0.28, 0.40)   # width
    De = rnd(0.20, 0.30)  # depth
    base_thk = rnd(0.012, 0.022)
    screen_thk = rnd(0.005, 0.012)
    gap = rnd(0.001, 0.003)

    body_col = rnd_color()
    screen_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", screen_col)

    links, joints = [], []
    links.append(link_box_xml("base_inertial_dummy", (W, De, base_thk),
                              origin_xyz=(0, 0, 0), mass=1.0, material="mat_body"))

    variant = random.choice(["standard", "book"])

    theta_cap = math.radians(rnd(120, 145))

    if variant == "standard":
        # Screen — hinged at back edge of base (laptop style)
        screen_h = De * rnd(0.90, 1.0)
        hinge_y = De/2 - gap
        links.append(link_box_xml("screen_door", (W, screen_h, screen_thk),
                                  origin_xyz=(0, -screen_h/2, 0), mass=0.5, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_screen", "base_inertial_dummy", "screen_door",
                                         origin_xyz=(0, hinge_y, base_thk/2 + screen_thk/2 + gap),
                                         axis_xyz=(-1, 0, 0), lower=0.0, upper=theta_cap))
    else:  # book — hinged on left or right edge (spine), opens upward
        side = random.choice(["left", "right"])
        if side == "left":
            hinge_x = -W/2 + gap
            vis_x = W/2
            axis_y = -1
        else:
            hinge_x = W/2 - gap
            vis_x = -W/2
            axis_y = 1
        links.append(link_box_xml("screen_door", (W, De, screen_thk),
                                  origin_xyz=(vis_x, 0, 0), mass=0.5, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_screen", "base_inertial_dummy", "screen_door",
                                         origin_xyz=(hinge_x, 0, base_thk/2 + screen_thk/2 + gap),
                                         axis_xyz=(0, axis_y, 0), lower=0.0, upper=theta_cap))

    robot = '<robot name="laptop">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "laptop", "variant": variant,
            "dims": {"W": W, "D": De, "base_thk": base_thk, "screen_thk": screen_thk}}
    return robot, meta


def build_oven(seed=None):
    """Oven/microwave/appliance with front door and an inner rack.
    Variants: side_door_rack, drop_door_rack."""
    random.seed(seed)
    W = rnd(0.45, 0.80)
    De = rnd(0.40, 0.70)
    H = rnd(0.35, 0.70)
    wall = rnd(0.020, 0.035)
    gap = rnd(0.001, 0.003)
    door_thk = rnd(0.020, 0.035)

    body_col = rnd_color()
    door_col = rnd_color()
    mats = (material_xml("mat_body", body_col) + material_xml("mat_door", door_col) +
            material_xml("mat_interior", rnd_color()))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.5))

    # Shell: top, bottom, left, right, back (open front)
    links.append(link_box_xml("shell_top",    (W, De, wall), origin_xyz=(0, 0, +H/2 - wall/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_top", "base_inertial_dummy", "shell_top"))
    links.append(link_box_xml("shell_bottom", (W, De, wall), origin_xyz=(0, 0, -H/2 + wall/2), mass=0.8, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_shell_bottom", "base_inertial_dummy", "shell_bottom"))
    links.append(link_box_xml("shell_left",   (wall, De, H - 2*wall), origin_xyz=(-W/2 + wall/2, 0, 0), mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_left", "base_inertial_dummy", "shell_left"))
    links.append(link_box_xml("shell_right",  (wall, De, H - 2*wall), origin_xyz=(+W/2 - wall/2, 0, 0), mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_right", "base_inertial_dummy", "shell_right"))
    links.append(link_box_xml("shell_back",   (W, wall, H - 2*wall), origin_xyz=(0, +De/2 - wall/2, 0), mass=0.6, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_shell_back", "base_inertial_dummy", "shell_back"))

    inner_w = W - 2*wall - 2*gap
    inner_h = H - 2*wall - 2*gap
    theta_cap = math.radians(rnd(80, 100))

    variant = random.choice(["side_door_rack", "drop_door_rack"])

    # Bar handle dimensions (consistent across variants)
    bar_r = rnd(0.006, 0.012)       # bar cylinder radius
    standoff_r = rnd(0.005, 0.010)  # standoff cylinder radius
    standoff_len = rnd(0.015, 0.030)  # gap between bar and door face

    if variant == "side_door_rack":
        # Side-hinged door + inner rack that slides out
        hinge_y = -De/2 - rnd(0.004, 0.010)
        y_offset = door_thk/2 + FLUSH_EPS
        side = random.choice(["left", "right"])
        if side == "left":
            x_hinge = -inner_w/2
            x_vis = +inner_w/2
            axis_z = -1
        else:
            x_hinge = +inner_w/2
            x_vis = -inner_w/2
            axis_z = 1
        links.append(link_box_xml("rack_door", (inner_w, door_thk, inner_h),
                                  origin_xyz=(x_vis, y_offset, 0), mass=0.8, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_door", "base_inertial_dummy", "rack_door",
                                         origin_xyz=(x_hinge, hinge_y, 0),
                                         axis_xyz=(0, 0, axis_z), lower=0.0, upper=theta_cap))
        # Bar handle — vertical bar near the free edge (opposite hinge)
        bar_len = inner_h * rnd(0.25, 0.45)
        free_x_sign = +1 if side == "left" else -1  # opposite hinge in vis coords
        bar_x = x_vis + free_x_sign * inner_w * 0.35
        bar_y_center = y_offset - door_thk/2 - standoff_len - bar_r
        # Main bar (vertical cylinder, rotated to align with Z)
        links.append(link_cylinder_xml("handle_bar", radius=bar_r, length=bar_len,
                     origin_xyz=(bar_x, bar_y_center, 0),
                     mass=0.05, material="mat_door"))
        joints.append(fixed_joint_xml("fix_handle_bar", "rack_door", "handle_bar"))
        # Two standoffs (horizontal cylinders connecting bar to door face)
        for si, sz in enumerate([+bar_len/2 - standoff_r*2, -bar_len/2 + standoff_r*2]):
            links.append(link_cylinder_xml(f"handle_standoff_{si}", radius=standoff_r,
                         length=standoff_len,
                         origin_xyz=(bar_x, y_offset - door_thk/2 - standoff_len/2, sz),
                         rpy=(math.pi/2, 0, 0),
                         mass=0.02, material="mat_door"))
            joints.append(fixed_joint_xml(f"fix_standoff_{si}", "rack_door", f"handle_standoff_{si}"))
        # Inner rack slides out front
        if not FAMOS_TRAINING:
            rack_thk = rnd(0.008, 0.015)
            rack_w = inner_w
            rack_d = De - 2*wall - 0.02
            links.append(link_box_xml("rack_drawer_0", (rack_w, rack_d, rack_thk),
                                      origin_xyz=(0, 0, 0), mass=0.3, material="mat_body"))
            joints.append(prismatic_joint_xml("slide_rack", "base_inertial_dummy", "rack_drawer_0",
                                              origin_xyz=(0, 0, -H*0.1),
                                              axis_xyz=(0, -1, 0), lower=0.0, upper=rack_d * 0.8))
    else:  # drop_door_rack
        # Bottom-hinged drop door + inner rack that slides out
        hinge_y = -De/2 - rnd(0.004, 0.010)
        y_offset = door_thk/2 + FLUSH_EPS
        links.append(link_box_xml("oven_door", (inner_w, door_thk, inner_h),
                                  origin_xyz=(0, y_offset, +inner_h/2), mass=0.8, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_door", "base_inertial_dummy", "oven_door",
                                         origin_xyz=(0, hinge_y, -H/2 + wall + gap),
                                         axis_xyz=(1, 0, 0), lower=0.0, upper=theta_cap))
        # Bar handle — horizontal bar near top of door (free edge, opposite bottom hinge)
        bar_len = inner_w * rnd(0.50, 0.85)
        bar_z = inner_h * 0.85  # near top of door visual (vis origin offsets door upward)
        bar_y_center = y_offset - door_thk/2 - standoff_len - bar_r
        # Main bar (horizontal cylinder, rotated to align with X)
        links.append(link_cylinder_xml("handle_bar", radius=bar_r, length=bar_len,
                     origin_xyz=(0, bar_y_center, bar_z),
                     rpy=(0, math.pi/2, 0),
                     mass=0.05, material="mat_door"))
        joints.append(fixed_joint_xml("fix_handle_bar", "oven_door", "handle_bar"))
        # Two standoffs at each end
        for si, sx in enumerate([+bar_len/2 - standoff_r*2, -bar_len/2 + standoff_r*2]):
            links.append(link_cylinder_xml(f"handle_standoff_{si}", radius=standoff_r,
                         length=standoff_len,
                         origin_xyz=(sx, y_offset - door_thk/2 - standoff_len/2, bar_z),
                         rpy=(math.pi/2, 0, 0),
                         mass=0.02, material="mat_door"))
            joints.append(fixed_joint_xml(f"fix_standoff_{si}", "oven_door", f"handle_standoff_{si}"))
        # Inner rack slides out front
        if not FAMOS_TRAINING:
            rack_thk = rnd(0.008, 0.015)
            rack_w = inner_w
            rack_d = De - 2*wall - 0.02
            links.append(link_box_xml("rack_drawer_0", (rack_w, rack_d, rack_thk),
                                      origin_xyz=(0, 0, 0), mass=0.3, material="mat_body"))
            joints.append(prismatic_joint_xml("slide_rack", "base_inertial_dummy", "rack_drawer_0",
                                              origin_xyz=(0, 0, -H*0.1),
                                              axis_xyz=(0, -1, 0), lower=0.0, upper=rack_d * 0.8))

    robot = '<robot name="oven">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "oven", "variant": variant,
            "dims": {"W": W, "D": De, "H": H, "wall": wall}}
    return robot, meta


def build_bucket(seed=None):
    """Open-top bin with a swinging U-handle.
    Variants: rectangular (box walls), round (hollow cylinder)."""
    random.seed(seed)
    variant = random.choice(["rectangular", "round"])
    H = rnd(0.20, 0.45)
    wall = rnd(0.008, 0.018)
    handle_thk = rnd(0.012, 0.022)
    handle_w = rnd(0.015, 0.028)
    gap = rnd(0.001, 0.003)

    body_col = rnd_color()
    handle_col = rnd_color()
    mats = (material_xml("mat_body", body_col) + material_xml("mat_door", handle_col) +
            material_xml("mat_interior", rnd_color()))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.3))

    if variant == "round":
        radius = rnd(0.08, 0.18)
        inner_r = radius - wall
        W = radius * 2  # for handle span

        # Hollow cylinder shell
        links.append(link_hollow_cylinder_xml("body_shell", radius=radius, inner_radius=inner_r,
                                              length=H, origin_xyz=(0, 0, 0), mass=0.5, material="mat_body"))
        joints.append(fixed_joint_xml("fix_body", "base_inertial_dummy", "body_shell"))

        # Bottom disk
        links.append(link_cylinder_xml("bin_bottom", radius=inner_r, length=wall,
                                       origin_xyz=(0, 0, -H/2 + wall/2), mass=0.3, material="mat_interior"))
        joints.append(fixed_joint_xml("fix_bin_bottom", "base_inertial_dummy", "bin_bottom"))

    else:  # rectangular
        W = rnd(0.12, 0.30)
        De = rnd(0.12, 0.30)

        # 5-wall open-top bin
        links.append(link_box_xml("bin_bottom", (W, De, wall), origin_xyz=(0, 0, -H/2 + wall/2), mass=0.4, material="mat_interior"))
        joints.append(fixed_joint_xml("fix_bin_bottom", "base_inertial_dummy", "bin_bottom"))
        links.append(link_box_xml("bin_front",  (W, wall, H - wall), origin_xyz=(0, -De/2 + wall/2, wall/2), mass=0.3, material="mat_body"))
        joints.append(fixed_joint_xml("fix_bin_front", "base_inertial_dummy", "bin_front"))
        links.append(link_box_xml("bin_back",   (W, wall, H - wall), origin_xyz=(0, +De/2 - wall/2, wall/2), mass=0.3, material="mat_body"))
        joints.append(fixed_joint_xml("fix_bin_back", "base_inertial_dummy", "bin_back"))
        links.append(link_box_xml("bin_left",   (wall, De - 2*wall, H - wall), origin_xyz=(-W/2 + wall/2, 0, wall/2), mass=0.3, material="mat_body"))
        joints.append(fixed_joint_xml("fix_bin_left", "base_inertial_dummy", "bin_left"))
        links.append(link_box_xml("bin_right",  (wall, De - 2*wall, H - wall), origin_xyz=(+W/2 - wall/2, 0, wall/2), mass=0.3, material="mat_body"))
        joints.append(fixed_joint_xml("fix_bin_right", "base_inertial_dummy", "bin_right"))

    # Handle — either curved (half-torus) or straight (U-shaped boxes)
    handle_style = random.choice(["curved", "straight"])
    handle_span = W - 2*wall

    if handle_style == "curved":
        # Semicircular arc handle spanning left-to-right, arching upward
        arc_r = handle_span / 2  # arc centre-line radius = half the span
        tube_r = handle_thk / 2
        # Half-torus arches in XZ from +X to -X; hinge at left rim
        links.append(link_half_torus_xml("handle_arc_door", arc_radius=arc_r, tube_radius=tube_r,
                                         origin_xyz=(arc_r, 0, 0), mass=0.08, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_handle", "base_inertial_dummy", "handle_arc_door",
                                         origin_xyz=(-W/2 + wall, 0, H/2 + gap),
                                         axis_xyz=(1, 0, 0), lower=0.0, upper=math.radians(90)))
    else:
        # U-shaped arch (left riser + crossbar + right riser)
        rise_h = rnd(0.10, 0.22)

        # Left riser — the dynamic link, hinged at left rim top edge
        links.append(link_box_xml("handle_riser_door", (handle_thk, handle_w, rise_h),
                                  origin_xyz=(0, 0, rise_h/2), mass=0.04, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_handle", "base_inertial_dummy", "handle_riser_door",
                                         origin_xyz=(-W/2 + wall, 0, H/2 + gap),
                                         axis_xyz=(1, 0, 0), lower=0.0, upper=math.radians(90)))

        # Crossbar — fixed to left riser top, spans in +X
        links.append(link_box_xml("handle_cross", (handle_span, handle_w, handle_thk),
                                  origin_xyz=(handle_span/2, 0, 0), mass=0.04, material="mat_door"))
        joints.append(fixed_joint_xml("fix_handle_cross", "handle_riser_door",
                                      "handle_cross",
                                      origin_xyz=(0, 0, rise_h)))

        # Right riser — fixed to crossbar end, extends downward
        links.append(link_box_xml("handle_riser_r", (handle_thk, handle_w, rise_h),
                                  origin_xyz=(0, 0, -rise_h/2), mass=0.04, material="mat_door"))
        joints.append(fixed_joint_xml("fix_handle_riser_r", "handle_cross",
                                      "handle_riser_r",
                                      origin_xyz=(handle_span, 0, 0)))

    robot = '<robot name="bucket">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    De_out = radius * 2 if variant == "round" else De
    meta = {"family": "bucket", "variant": variant,
            "dims": {"W": W, "D": De_out, "H": H, "wall": wall}}
    return robot, meta


def build_eyeglasses(seed=None):
    """Eyeglasses: front frame + two folding arms.
    Variants: standard (rectangular frame), round (two disk lenses + bridge)."""
    random.seed(seed)
    variant = random.choice(["standard", "round"])

    frame_thk = rnd(0.004, 0.008)
    bridge_w = rnd(0.010, 0.020)
    arm_len = rnd(0.10, 0.15)
    arm_w = rnd(0.004, 0.008)
    arm_thk = rnd(0.003, 0.006)
    gap = rnd(0.001, 0.002)

    body_col = rnd_color()
    arm_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", arm_col)

    links, joints = [], []

    if variant == "round":
        # Two disk lenses + bridge bar
        lens_r = rnd(0.020, 0.035)
        lens_thk = rnd(0.003, 0.006)
        lens_gap = bridge_w  # space between lenses
        lens_cx = lens_r + lens_gap / 2  # x offset of each lens centre
        frame_w = 2 * lens_cx + 2 * lens_r  # total span

        # Invisible dummy at centre for kinematic root
        links.append(link_empty_xml("base_inertial_dummy", mass=0.001))

        # Left lens disk
        links.append(link_cylinder_xml("lens_left", radius=lens_r, length=lens_thk,
                                       origin_xyz=(-lens_cx, 0, 0),
                                       rpy=(math.pi/2, 0, 0),
                                       mass=0.01, material="mat_body"))
        joints.append(fixed_joint_xml("fix_lens_left", "base_inertial_dummy", "lens_left"))

        # Right lens disk
        links.append(link_cylinder_xml("lens_right", radius=lens_r, length=lens_thk,
                                       origin_xyz=(lens_cx, 0, 0),
                                       rpy=(math.pi/2, 0, 0),
                                       mass=0.01, material="mat_body"))
        joints.append(fixed_joint_xml("fix_lens_right", "base_inertial_dummy", "lens_right"))

        # Bridge bar connecting the two lenses
        links.append(link_box_xml("bridge", (lens_gap, frame_thk, arm_thk),
                                  origin_xyz=(0, 0, 0), mass=0.005, material="mat_body"))
        joints.append(fixed_joint_xml("fix_bridge", "base_inertial_dummy", "bridge"))

    else:
        # Standard rectangular frame
        frame_w = rnd(0.12, 0.16)
        frame_h = rnd(0.03, 0.05)
        links.append(link_box_xml("base_inertial_dummy", (frame_w, frame_thk, frame_h),
                                  origin_xyz=(0, 0, 0), mass=0.03, material="mat_body"))

    arm_theta_cap = math.radians(90 if FAMOS_TRAINING else 110)

    # Left arm — hinged at left edge, folds inward (+X)
    links.append(link_box_xml("arm_left_door", (arm_w, arm_len, arm_thk),
                              origin_xyz=(0, arm_len/2, 0), mass=0.01, material="mat_door"))
    joints.append(revolute_joint_xml("hinge_arm_left", "base_inertial_dummy", "arm_left_door",
                                     origin_xyz=(-frame_w/2 + arm_w/2, frame_thk/2 + gap, 0),
                                     axis_xyz=(0, 0, -1), lower=0.0, upper=arm_theta_cap))

    # Right arm — hinged at right edge, folds inward (-X)
    links.append(link_box_xml("arm_right_door", (arm_w, arm_len, arm_thk),
                              origin_xyz=(0, arm_len/2, 0), mass=0.01, material="mat_door"))
    joints.append(revolute_joint_xml("hinge_arm_right", "base_inertial_dummy", "arm_right_door",
                                     origin_xyz=(+frame_w/2 - arm_w/2, frame_thk/2 + gap, 0),
                                     axis_xyz=(0, 0, 1), lower=0.0, upper=arm_theta_cap))

    robot = '<robot name="eyeglasses">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "eyeglasses", "variant": variant,
            "dims": {"W": frame_w, "D": frame_thk, "arm_len": arm_len}}
    return robot, meta


def build_fan(seed=None):
    """Standing fan: base/stand + blade panel that rotates.
    Variants: standard (horizontal axis spin)."""
    random.seed(seed)
    base_w = rnd(0.10, 0.35)
    base_d = rnd(0.08, 0.25)
    base_h = rnd(0.015, 0.06)
    pole_w = rnd(0.02, 0.05)
    pole_h = rnd(0.25, 0.50)
    blade_w = rnd(0.20, 0.50)   # blade panel width (wider)
    max_blade_h = pole_h - 0.04  # keep blade well above base
    blade_h = rnd(0.06, min(0.25, max_blade_h * 0.65))  # shorter → more rectangular
    blade_thk = rnd(0.005, 0.025)
    gap = rnd(0.001, 0.003)

    body_col = rnd_color()
    blade_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", blade_col)

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.3))

    # Base platform
    links.append(link_box_xml("base_plate", (base_w, base_d, base_h),
                              origin_xyz=(0, 0, -pole_h/2 - base_h/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_base", "base_inertial_dummy", "base_plate"))

    # Pole
    links.append(link_box_xml("pole", (pole_w, pole_w, pole_h),
                              origin_xyz=(0, 0, 0), mass=0.4, material="mat_body"))
    joints.append(fixed_joint_xml("fix_pole", "base_inertial_dummy", "pole"))

    # Blade panel — rotates around Y axis (horizontal spin facing front)
    blade_y = -pole_w/2 - blade_thk/2 - gap
    blade_z = pole_h/2 - blade_h/2
    links.append(link_box_xml("blade_door", (blade_w, blade_thk, blade_h),
                              origin_xyz=(0, 0, 0), mass=0.3, material="mat_door"))
    joints.append(revolute_joint_xml("hinge_blade", "base_inertial_dummy", "blade_door",
                                     origin_xyz=(0, blade_y, blade_z),
                                     axis_xyz=(0, -1, 0), lower=0.0, upper=math.radians(360)))

    # Cross blade — second panel perpendicular, forms + shape with first
    # First blade: wide in X (blade_w), tall in Z (blade_h)
    # Cross blade: wide in Z (blade_h → now X), tall in X (blade_w → now Z)
    links.append(link_box_xml("blade_cross", (blade_h, blade_thk, blade_w),
                              origin_xyz=(0, 0, 0), mass=0.15, material="mat_door"))
    joints.append(fixed_joint_xml("fix_blade_cross", "blade_door", "blade_cross"))

    variant = "standard"
    robot = '<robot name="fan">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "fan", "variant": variant,
            "dims": {"W": base_w, "D": base_d, "pole_h": pole_h, "blade_w": blade_w, "blade_h": blade_h}}
    return robot, meta


def build_scissors(seed=None):
    """Scissors: each half = blade (box in +Y) + finger loop (half-torus U in -Y),
    both lying flat in XY. Two halves pivot around Z at the origin.
    Variants: standard."""
    random.seed(seed)
    blade_len = rnd(0.15, 0.25)
    blade_w = rnd(0.010, 0.020)
    blade_thk = rnd(0.002, 0.005)
    # Finger loop dimensions
    ring_arc_r = rnd(0.015, 0.030)   # half-circle radius (finger hole size)
    ring_tube_r = blade_thk / 2        # tube diameter matches bar thickness
    gap = rnd(0.001, 0.002)
    open_angle = rnd(35, 80) if FAMOS_TRAINING else rnd(20, 45)  # opening angle in degrees
    # Pivot offset: always toward blade tip (away from circles)
    pivot_shift = rnd(0.01, 0.04)

    body_col = rnd_color()
    blade_col = rnd_color()
    mats = material_xml("mat_body", body_col) + material_xml("mat_door", blade_col)

    links, joints = [], []

    # Each half is a P lying flat in XY, pivoting about Z at the origin. Its bar runs
    # from +blade_len to -(2R + tube_r); the loop is centered at (0,-R,0) with endpoints
    # at y=0 and y=-2R, and the bar extends tube_r past it so the edges match.
    total_len = blade_len + 2 * ring_arc_r + ring_tube_r
    bar_vis_y = (blade_len - 2 * ring_arc_r - ring_tube_r) / 2

    # pivot_shift moves geometry relative to pivot (negative = toward handle)
    vis_off = -pivot_shift  # shift visuals so pivot appears moved toward blade

    # Bottom half (base).
    links.append(link_box_xml("base_inertial_dummy", (blade_w, total_len, blade_thk),
                              origin_xyz=(0, bar_vis_y + vis_off, 0), mass=0.04, material="mat_body"))
    # Circle centered at y=-R, bump in +X, shifted to bar's outer edge
    links.append(link_half_torus_xml("handle_bottom", arc_radius=ring_arc_r, tube_radius=ring_tube_r,
                                     origin_xyz=(blade_w/2, -ring_arc_r + vis_off, 0),
                                     rpy=(math.pi/2, 0, math.pi/2),
                                     mass=0.02, material="mat_body"))
    joints.append(fixed_joint_xml("fix_handle_bottom", "base_inertial_dummy", "handle_bottom"))

    # Top half (moving).
    top_thk = blade_thk
    links.append(link_box_xml("blade_top_door", (blade_w, total_len, top_thk),
                              origin_xyz=(0, bar_vis_y + vis_off, 0), mass=0.04, material="mat_door"))
    joints.append(revolute_joint_xml("hinge_pivot", "base_inertial_dummy", "blade_top_door",
                                     origin_xyz=(0, 0, blade_thk/2 + top_thk/2 + gap),
                                     axis_xyz=(0, 0, 1), lower=-math.radians(open_angle), upper=0.0))
    # Circle bump in -X (mirror), shifted to bar's outer edge
    links.append(link_half_torus_xml("handle_top", arc_radius=ring_arc_r, tube_radius=ring_tube_r,
                                     origin_xyz=(-blade_w/2, -ring_arc_r + vis_off, 0),
                                     rpy=(math.pi/2, 0, -math.pi/2),
                                     mass=0.02, material="mat_door"))
    joints.append(fixed_joint_xml("fix_handle_top", "blade_top_door", "handle_top"))

    variant = "standard"
    robot = '<robot name="scissors">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "scissors", "variant": variant,
            "dims": {"W": blade_w + ring_arc_r, "D": blade_len + ring_arc_r * 2,
                     "blade_len": blade_len, "ring_arc_r": ring_arc_r}}
    return robot, meta


def build_trash_can(seed=None):
    """Round trash can: hollow cylinder body, disk bottom, disk lid.
    Variants: revolute (hinged at back edge), lift (lid slides straight up)."""
    random.seed(seed)
    radius = rnd(0.10, 0.25)
    H = rnd(0.25, 0.50)
    wall = rnd(0.005, 0.012)    # shell thickness
    lid_thk = rnd(0.008, 0.015)
    gap = rnd(0.001, 0.003)
    knob_r = rnd(0.012, 0.025)

    body_col = rnd_color()
    lid_col = rnd_color()
    mats = (material_xml("mat_body", body_col) + material_xml("mat_door", lid_col) +
            material_xml("mat_interior", rnd_color()))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.01))

    # Hollow cylinder shell (body) — true tube with wall thickness
    inner_r = radius - wall
    links.append(link_hollow_cylinder_xml("body_shell", radius=radius, inner_radius=inner_r,
                                          length=H, origin_xyz=(0, 0, 0), mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_body", "base_inertial_dummy", "body_shell"))

    # Bottom disk — thin cylinder closing the bottom (interior-facing)
    links.append(link_cylinder_xml("bottom_disk", radius=inner_r, length=wall,
                                   origin_xyz=(0, 0, -H/2 + wall/2), mass=0.2, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_bottom", "base_inertial_dummy", "bottom_disk"))

    variant = random.choice(["revolute"] if FAMOS_TRAINING else ["revolute", "lift"])

    if variant == "revolute":
        # Lid disk — hinged at the back edge of the rim, revolute around X axis
        links.append(link_cylinder_xml("lid_door", radius=radius, length=lid_thk,
                                       origin_xyz=(0, radius, 0), mass=0.15, material="mat_door"))
        joints.append(revolute_joint_xml("hinge_lid", "base_inertial_dummy", "lid_door",
                                         origin_xyz=(0, -radius, H/2 + gap),
                                         axis_xyz=(1, 0, 0), lower=0.0, upper=math.radians(120)))
        # Knob on lid — front edge (opposite hinge), on top face
        links.append(link_sphere_xml("knob", knob_r,
                     origin_xyz=(0, radius * 1.7, lid_thk/2 + knob_r),
                     mass=0.02, material="mat_door"))
        joints.append(fixed_joint_xml("fix_knob", "lid_door", "knob"))
    else:  # lift
        # Lid slides straight up
        lift_range = H * rnd(0.3, 0.6)
        links.append(link_cylinder_xml("lid_drawer_0", radius=radius, length=lid_thk,
                                       origin_xyz=(0, 0, 0), mass=0.15, material="mat_door"))
        joints.append(prismatic_joint_xml("slide_lid", "base_inertial_dummy", "lid_drawer_0",
                                          origin_xyz=(0, 0, H/2 + gap + lid_thk/2),
                                          axis_xyz=(0, 0, 1), lower=0.0, upper=lift_range))
        # Knob on top center of lid
        links.append(link_sphere_xml("knob", knob_r,
                     origin_xyz=(0, 0, lid_thk/2 + knob_r),
                     mass=0.02, material="mat_door"))
        joints.append(fixed_joint_xml("fix_knob", "lid_drawer_0", "knob"))

    robot = '<robot name="trash_can">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "trash_can", "variant": variant,
            "dims": {"W": radius*2, "D": radius*2, "H": H, "radius": radius}}
    return robot, meta


def build_trolley(seed=None):
    """Trolley/suitcase: slim box body, telescoping U-handle (prismatic Z),
    small cylinder wheels at the bottom.
    Variants: 4_wheel."""
    random.seed(seed)
    W = rnd(0.30, 0.50)       # width (left-right)
    De = rnd(0.15, 0.28)      # depth (front-back, slim)
    H = rnd(0.45, 0.70)       # height
    wall = rnd(0.010, 0.020)  # unused, but drawn to keep the random stream unchanged
    handle_thk = rnd(0.010, 0.018)  # handle bar thickness
    handle_w = rnd(0.012, 0.020)    # handle bar width
    gap = rnd(0.001, 0.003)

    # Wheels
    wheel_r = rnd(0.020, 0.040)
    wheel_thk = rnd(0.012, 0.025)

    body_col = rnd_color()
    handle_col = rnd_color()
    wheel_col = rnd_color()
    mats = (material_xml("mat_body", body_col) +
            material_xml("mat_door", handle_col) +
            material_xml("mat_wheel", wheel_col))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.5))

    # Body — solid box
    links.append(link_box_xml("body", (W, De, H),
                              origin_xyz=(0, 0, 0), mass=3.0, material="mat_body"))
    joints.append(fixed_joint_xml("fix_body", "base_inertial_dummy", "body"))

    # Telescoping handle — U-shape (left pole + crossbar + right pole)
    # Slides up from the back face (+Y side)
    handle_span = W * rnd(0.50, 0.80)
    pole_h = H * rnd(0.50, 0.80)  # how tall the handle extends
    slide_range = pole_h + handle_thk  # full retraction: nothing sticks up

    # Left pole — the dynamic link (slider, so use "drawer" naming)
    links.append(link_box_xml("handle_drawer_0", (handle_thk, handle_w, pole_h),
                              origin_xyz=(-handle_span/2, 0, pole_h/2),
                              mass=0.1, material="mat_door"))
    joints.append(prismatic_joint_xml("slide_handle", "base_inertial_dummy", "handle_drawer_0",
                                      origin_xyz=(0, De/2 + handle_w/2 + gap, H/2),
                                      axis_xyz=(0, 0, -1), lower=0.0, upper=slide_range))

    # Right pole — fixed to left pole
    links.append(link_box_xml("handle_pole_right", (handle_thk, handle_w, pole_h),
                              origin_xyz=(handle_span/2, 0, pole_h/2),
                              mass=0.1, material="mat_door"))
    joints.append(fixed_joint_xml("fix_pole_right", "handle_drawer_0", "handle_pole_right"))

    # Crossbar at top connecting the two poles
    links.append(link_box_xml("handle_cross", (handle_span, handle_w, handle_thk),
                              origin_xyz=(0, 0, pole_h),
                              mass=0.05, material="mat_door"))
    joints.append(fixed_joint_xml("fix_handle_cross", "handle_drawer_0", "handle_cross"))

    # Wheels — 4 small cylinders at bottom corners, axis along X (roll in Y)
    variant = "4_wheel"
    wheel_positions = []
    for sx in [-1, +1]:
        for sy in [-1, +1]:
            wx = sx * (W/2 - wheel_thk/2)
            wy = sy * (De/2 - wheel_thk/2)
            wheel_positions.append((wx, wy))

    for wi, (wx, wy) in enumerate(wheel_positions):
        wname = f"wheel_{wi}_door"
        links.append(link_cylinder_xml(wname, radius=wheel_r, length=wheel_thk,
                     origin_xyz=(0, 0, 0),
                     rpy=(0, math.pi/2, 0),
                     mass=0.05, material="mat_wheel"))
        joints.append(revolute_joint_xml(f"spin_{wname}", "base_inertial_dummy", wname,
                                         origin_xyz=(wx, wy, -H/2 - wheel_r),
                                         axis_xyz=(1, 0, 0),
                                         lower=0.0, upper=math.radians(360)))

    robot = '<robot name="trolley">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "trolley", "variant": variant,
            "dims": {"W": W, "D": De, "H": H, "wheel_r": wheel_r}}
    return robot, meta


def build_washing_machine(seed=None):
    """Front-loading washing machine: box shell with circular front opening,
    inner cylindrical drum, and a round hinged door.
    Variants: side_hinge_left, side_hinge_right."""
    random.seed(seed)
    W = rnd(0.45, 0.65)
    De = rnd(0.45, 0.65)
    H = rnd(0.50, 0.75)
    wall = rnd(0.020, 0.035)
    gap = rnd(0.001, 0.003)
    door_thk = rnd(0.015, 0.030)

    # Inner drum — radius fits inside the box, axis along Y (front-back)
    drum_r = min(W, H) / 2 - wall - gap - 0.01
    drum_inner_r = drum_r - rnd(0.005, 0.012)
    drum_len = De - 2 * wall  # spans from front panel to back wall

    # Front hole and door all match the drum outer radius exactly
    hole_r = drum_r

    body_col = rnd_color()
    door_col = rnd_color()
    drum_col = rnd_color()
    mats = (material_xml("mat_body", body_col) +
            material_xml("mat_door", door_col) +
            material_xml("mat_drum", drum_col) +
            material_xml("mat_interior", rnd_color()))

    links, joints = [], []
    links.append(link_empty_xml("base_inertial_dummy", mass=0.5))

    # Shell: top, bottom, left, right, back (open front)
    links.append(link_box_xml("shell_top",    (W, De, wall), origin_xyz=(0, 0, +H/2 - wall/2), mass=0.8, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_top", "base_inertial_dummy", "shell_top"))
    links.append(link_box_xml("shell_bottom", (W, De, wall), origin_xyz=(0, 0, -H/2 + wall/2), mass=0.8, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_shell_bottom", "base_inertial_dummy", "shell_bottom"))
    links.append(link_box_xml("shell_left",   (wall, De, H - 2*wall), origin_xyz=(-W/2 + wall/2, 0, 0), mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_left", "base_inertial_dummy", "shell_left"))
    links.append(link_box_xml("shell_right",  (wall, De, H - 2*wall), origin_xyz=(+W/2 - wall/2, 0, 0), mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_shell_right", "base_inertial_dummy", "shell_right"))
    links.append(link_box_xml("shell_back",   (W, wall, H - 2*wall), origin_xyz=(0, +De/2 - wall/2, 0), mass=0.6, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_shell_back", "base_inertial_dummy", "shell_back"))

    # Front panel with circular hole
    front_w = W
    front_h = H - 2 * wall
    front_y = -De / 2 + wall / 2
    links.append(link_panel_with_hole_xml("front_panel", front_w, front_h, wall,
                                           hole_r, hole_cx=0.0, hole_cy=0.0,
                                           origin_xyz=(0, front_y, 0),
                                           mass=0.6, material="mat_body"))
    joints.append(fixed_joint_xml("fix_front_panel", "base_inertial_dummy", "front_panel"))

    # Inner drum (hollow cylinder, axis along Y → rpy=(pi/2,0,0))
    # Center so front rim is flush with front panel inner face
    drum_y = -De/2 + wall + drum_len/2
    links.append(link_hollow_cylinder_xml("drum", radius=drum_r, inner_radius=drum_inner_r,
                                           length=drum_len,
                                           origin_xyz=(0, drum_y, 0),
                                           rpy=(math.pi/2, 0, 0),
                                           mass=0.5, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_drum", "base_inertial_dummy", "drum"))

    # Back disk of drum (circular cap)
    drum_cap_thk = rnd(0.005, 0.012)
    links.append(link_cylinder_xml("drum_back_cap", radius=drum_inner_r, length=drum_cap_thk,
                                    origin_xyz=(0, drum_y + drum_len/2 - drum_cap_thk/2, 0),
                                    rpy=(math.pi/2, 0, 0),
                                    mass=0.2, material="mat_interior"))
    joints.append(fixed_joint_xml("fix_drum_cap", "base_inertial_dummy", "drum_back_cap"))

    # Round door — a thin cylinder (disk) that covers the hole
    # Hinged on one side (revolute joint)
    theta_cap = math.radians(rnd(90, 130))
    side = random.choice(["left", "right"])
    variant = f"side_hinge_{side}"

    door_y = -De/2 - rnd(0.002, 0.005)  # just in front of front panel

    if side == "left":
        hinge_x = -hole_r
        vis_x = +hole_r
        axis_z = -1
    else:
        hinge_x = +hole_r
        vis_x = -hole_r
        axis_z = 1

    links.append(link_cylinder_xml("washer_door", radius=hole_r, length=door_thk,
                                    origin_xyz=(vis_x, door_thk/2 + FLUSH_EPS, 0),
                                    rpy=(math.pi/2, 0, 0),
                                    mass=0.4, material="mat_door"))
    joints.append(revolute_joint_xml("hinge_door", "base_inertial_dummy", "washer_door",
                                      origin_xyz=(hinge_x, door_y, 0),
                                      axis_xyz=(0, 0, axis_z), lower=0.0, upper=theta_cap))

    robot = '<robot name="washing_machine">\n' + mats + "".join(links) + "".join(joints) + "\n</robot>"
    meta = {"family": "washing_machine", "variant": variant,
            "dims": {"W": W, "D": De, "H": H, "wall": wall, "drum_r": drum_r, "hole_r": hole_r}}
    return robot, meta


BUILDERS = {
    "hinged_cabinet":        build_hinged_cabinet,
    "drawer_cabinet":        build_drawer_cabinet,
    "combo_cabinet":         build_combo_cabinet,
    "triple_combo_cabinet":  build_triple_combo_cabinet,
    "stacked_cabinets":      build_stacked_cabinets,
    "door":                  build_door,
    "table":                 build_table,
    "box":                   build_box,
    "laptop":                build_laptop,
    "oven":                  build_oven,
    "bucket":                build_bucket,
    "eyeglasses":            build_eyeglasses,
    "fan":                   build_fan,
    "scissors":              build_scissors,
    "trash_can":             build_trash_can,
    "trolley":               build_trolley,
    "washing_machine":       build_washing_machine,
}

def save_asset(root, idx, family, seed, materials_root):
    urdf, meta = BUILDERS[family](seed)
    adir = Path(root) / f"asset_{idx:05d}_{family}"
    adir.mkdir(parents=True, exist_ok=True)
    urdf_path = adir / "mobility.urdf"
    urdf_path.write_text(urdf)
    (adir/"manifest.json").write_text(json.dumps(meta, indent=2))

    # Normalize joint ranges to always use positive values
    _normalize_joint_ranges(urdf_path)

    _normalize_urdf_geometry(urdf_path, target_max=1.0, recenter=True)

    # Write bounding box + meta
    _write_bounding_box_json(urdf_path, adir)
    _write_meta_json(adir, family)

    _meshify_urdf_boxes(urdf_path, Path(materials_root), seed=seed)

    # Write result.json (taxonomy) and mobility_v2.json (joint graph)
    _export_mobility_v2_json(urdf_path, adir)
    _export_result_json(urdf_path, adir, meta.get("family", family))
    
    return str(adir)

def main(out_dir, per_family, seed, materials_root):
    random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fams = list(BUILDERS.keys())
    n = per_family * len(fams)

    # One seed per asset
    jobs = [(out_dir, i, fams[i % len(fams)], seed+i, materials_root) for i in range(n)]

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 16) as ex:
        futs = [ex.submit(save_asset, *job) for job in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            fut.result()

    print(f"Wrote {n} assets to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="procedural_assets")
    p.add_argument("--per_family", type=int, default=10, help="Number of assets per family.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--materials", type=str, required=True,
                   help="Folder containing material sets (each set has basecolor.png; others optional). "
                        "See download_materials.py.")
    args = p.parse_args()
    main(args.out, args.per_family, args.seed, Path(args.materials))
