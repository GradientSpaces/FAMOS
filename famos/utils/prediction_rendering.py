"""Paper-style figures of predictions: shaded point clouds, joint arrows, soft floor shadow.

All panels of an object share one canonical, Z-up camera, so the articulation across the
input states reads as a sequence.
"""

from functools import lru_cache

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..data.rendering import _ensure_pyrender

STATIC_COLOR = (153, 153, 153)
PART_COLORS = [
    (72, 36, 117), (33, 145, 140), (189, 223, 38), (153, 80, 8),
    (12, 12, 242), (242, 12, 150), (12, 242, 150), (12, 150, 242),
]
AXIS_COLORS = {"revolute": (230, 26, 26), "prismatic": (255, 217, 13)}
UNLABELED_COLOR = (215, 215, 215)

VIEW_ELEV_DEG = 20.0
VIEW_AZIM_OFFSET_DEG = 45.0  # 3/4 view: straight-on, slide axes would point at the camera
FOV_DEG = 32.0
FIT_MARGIN = 1.08
POINT_RADIUS = 0.016        # in units of the object's bounding-sphere radius
SHADE_MIN = 0.55            # side of a point facing away from the light
SHADE_LIGHT = np.array([-3.0, -4.0, 6.0]) / np.linalg.norm([-3.0, -4.0, 6.0])
SHADOW_LIGHT = np.array([0.15, 0.25, 1.0]) / np.linalg.norm([0.15, 0.25, 1.0])
SHADOW_OPACITY = 0.16
AXIS_RADIUS = 0.018
AXIS_EXT = 0.12             # extension past the part, fraction of its span along the axis
AXIS_MIN_EXT = 0.06
AXIS_PART_FRAC = 0.5        # revolute floor, as a fraction of the part's own radius
PRISM_HALF = 0.30
PRISM_POKE = 0.14           # a slide arrow pokes this far out of its part
AXIS_LIFT = 0.12            # arrows are moved toward the camera so points do not hide them
SUPERSAMPLE = 2


def _label_color(label: int):
    if label < 0:
        return UNLABELED_COLOR
    if label == 0:
        return STATIC_COLOR
    return PART_COLORS[(label - 1) % len(PART_COLORS)]


@lru_cache(maxsize=None)
def _shaded_sphere(color: tuple, radius: float) -> trimesh.Trimesh:
    """Sphere whose vertex colors carry a fixed, light-independent N.L gradient."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    shade = SHADE_MIN + (1.0 - SHADE_MIN) * (sphere.vertex_normals @ SHADE_LIGHT + 1.0) / 2.0
    rgb = np.clip(np.asarray(color, dtype=np.float64)[None] * shade[:, None], 0, 255)
    sphere.visual.vertex_colors = np.concatenate([rgb, np.full((len(rgb), 1), 255)], 1).astype(np.uint8)
    return sphere


def _arrow(start, end, radius, color) -> trimesh.Trimesh:
    direction = end - start
    length = float(np.linalg.norm(direction))
    direction = direction / length
    tip_radius = radius * 1.9
    tip_length = min(max(min(0.16 * length, radius * 7), tip_radius * 1.3), 0.6 * length)
    body_length = max(length - tip_length, 1e-4)
    shaft = trimesh.creation.cylinder(radius=radius, height=body_length, sections=20)
    shaft.apply_translation([0, 0, body_length / 2])
    head = trimesh.creation.cone(radius=tip_radius, height=tip_length, sections=24)
    head.apply_translation([0, 0, body_length])
    arrow = trimesh.util.concatenate([shaft, head])
    arrow.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))
    arrow.apply_translation(start)
    arrow.visual.vertex_colors = np.tile([*color, 255], (len(arrow.vertices), 1)).astype(np.uint8)
    return arrow


def _axis_arrow(joint: dict, part_points: np.ndarray, toward_camera: np.ndarray):
    """Arrow for one joint, sized to its part (units: object bounding-sphere radius)."""
    direction = np.asarray(joint["axis"], dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    origin = np.asarray(joint["origin"], dtype=np.float64)
    has_part = len(part_points) > 0

    def extent(values):
        # Percentiles, so a few stray points labeled as the part do not move the arrow.
        return np.percentile(values, [5, 95], axis=0)

    if joint["type"] == "prismatic":
        # A slide origin is arbitrary along the axis, so centre the arrow on the part.
        centre = extent(part_points).mean(0) if has_part else origin
        t_min, t_max = extent((part_points - centre) @ direction) if has_part else (0.0, 0.0)
        half = max(PRISM_HALF, (t_max - t_min) / 2 + PRISM_POKE)
        start, end = centre - half * direction, centre + half * direction
    else:
        # Bounded to the hinged part and extended past it on both ends.
        t_min, t_max = extent((part_points - origin) @ direction) if has_part else (-PRISM_HALF, PRISM_HALF)
        ext = max(AXIS_EXT * (t_max - t_min), AXIS_MIN_EXT)
        part_radius = (
            extent(np.linalg.norm(part_points - np.median(part_points, 0), axis=1))[1]
            if has_part else AXIS_MIN_EXT
        )
        centre = origin + direction * (t_min + t_max) / 2
        half = max((t_max - t_min) / 2 + ext, AXIS_PART_FRAC * part_radius)
        start, end = centre - half * direction, centre + half * direction
        # A rotation axis has no inherent direction: point the head along its dominant component.
        if (end - start)[np.argmax(np.abs(end - start))] < 0:
            start, end = end, start
    lift = AXIS_LIFT * toward_camera
    return _arrow(start + lift, end + lift, AXIS_RADIUS, AXIS_COLORS.get(joint["type"], AXIS_COLORS["revolute"]))


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world pose (OpenGL convention, -Z forward) with world Z up."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -forward, eye
    return pose


_RENDERERS = {}


def _render(parts, camera_pose, size):
    """Render (vertices, faces, RGBA uint8 colors) arrays, unlit, as one primitive."""
    _ensure_pyrender()  # headless EGL; MSAA off, since its resolve is not bitwise reproducible
    import pyrender

    offsets = np.cumsum([0] + [len(v) for v, _, _ in parts[:-1]])
    primitive = pyrender.Primitive(
        positions=np.concatenate([v for v, _, _ in parts]).astype(np.float32),
        indices=np.concatenate([f + o for (_, f, _), o in zip(parts, offsets)]).astype(np.uint32),
        color_0=np.concatenate([c for _, _, c in parts]).astype(np.float32) / 255.0,
        mode=pyrender.constants.GLTF.TRIANGLES,
    )
    # A single primitive is drawn in one call, so overlapping points always resolve the
    # same way (pyrender orders separate meshes by iterating a set).
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh([primitive]))
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(FOV_DEG), znear=0.05, zfar=20.0), pose=camera_pose)
    if size not in _RENDERERS:
        _RENDERERS[size] = pyrender.OffscreenRenderer(size, size)
    color, depth = _RENDERERS[size].render(scene, flags=pyrender.RenderFlags.FLAT)
    return color, depth


def _spheres(points: np.ndarray, sphere: trimesh.Trimesh):
    """Vertex, face, and color arrays of one copy of `sphere` at every point."""
    n, nv = len(points), len(sphere.vertices)
    vertices = (sphere.vertices[None] + points[:, None]).reshape(-1, 3)
    faces = (sphere.faces[None] + (np.arange(n) * nv)[:, None, None]).reshape(-1, 3)
    return vertices, faces, np.tile(sphere.visual.vertex_colors, (n, 1))


def _arrays(mesh: trimesh.Trimesh):
    return mesh.vertices, mesh.faces, mesh.visual.vertex_colors


def render_panel(points, labels, joints, camera_pose, toward_camera, floor_z, size):
    """One panel: shaded points per label, joint arrows, and a soft contact shadow."""
    parts = [
        _spheres(points[labels == label], _shaded_sphere(_label_color(int(label)), POINT_RADIUS))
        for label in np.unique(labels)
    ]
    for joint in joints or []:
        part = points[labels == joint.get("part_id", -1)]
        parts.append(_arrays(_axis_arrow(joint, part, toward_camera)))
    color, depth = _render(parts, camera_pose, size)

    # Contact shadow: project the points along the light onto the floor, render their
    # silhouette from the same camera, blur it, and darken the background under it.
    flat = points - np.outer((points[:, 2] - floor_z) / SHADOW_LIGHT[2], SHADOW_LIGHT)
    flat[:, 2] = floor_z
    black = _shaded_sphere((0, 0, 0), POINT_RADIUS * 1.2)
    _, shadow_depth = _render([_spheres(flat, black)], camera_pose, size)
    mask = Image.fromarray(((shadow_depth > 0) * 255).astype(np.uint8))
    mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(size * 0.022)), dtype=np.float64) / 255.0
    background = 255.0 * (1.0 - SHADOW_OPACITY * mask)[..., None] * np.ones(3)
    out = np.where((depth > 0)[..., None], color[..., :3].astype(np.float64), background)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize(
        (size // SUPERSAMPLE, size // SUPERSAMPLE), Image.LANCZOS
    )


@lru_cache(maxsize=None)
def _font(size: int):
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def render_prediction_figure(
    point_clouds: np.ndarray,
    segmentation_labels_gt: np.ndarray,
    segmentation_labels_pred: np.ndarray,
    gt_joints: list,
    pred_joints: list,
    front_azim_deg: float,
    caption: str = "",
    panel_size: int = 480,
) -> Image.Image:
    """Two rows, ground truth on top and prediction below, with one column per input state.

    Inputs are in the canonical, Z-up object frame: points (S, N, 3), labels (S, N) with
    0 = static base, and joint dicts with origin, axis, type, and part_id. The camera looks
    at the object from 45 deg off its front (azimuth `front_azim_deg`), 20 deg up.
    """
    S = point_clouds.shape[0]
    all_points = point_clouds.reshape(-1, 3)
    center = (all_points.min(0) + all_points.max(0)) / 2
    radius = max(float(np.linalg.norm(all_points - center, axis=1).max()), 1e-6)

    def normalize(pts):
        return (np.asarray(pts, dtype=np.float64) - center) / radius

    def normalize_joints(joints):
        return [dict(j, origin=normalize(j["origin"])) for j in (joints or []) if j]

    elev, azim = np.deg2rad(VIEW_ELEV_DEG), np.deg2rad(front_azim_deg + VIEW_AZIM_OFFSET_DEG)
    toward_camera = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    camera_pose = _look_at(toward_camera * FIT_MARGIN / np.sin(np.deg2rad(FOV_DEG) / 2), np.zeros(3))
    floor_z = float(normalize(all_points)[:, 2].min()) - POINT_RADIUS
    size = panel_size * SUPERSAMPLE

    rows = [("Ground truth", segmentation_labels_gt, normalize_joints(gt_joints)),
            ("Prediction", segmentation_labels_pred, normalize_joints(pred_joints))]
    label_w, header = int(panel_size * 0.3), int(panel_size * 0.08)
    footer = int(panel_size * 0.09) if caption else 0
    width = label_w + panel_size * S
    figure = Image.new("RGB", (width, header + 2 * panel_size + footer), "white")
    draw = ImageDraw.Draw(figure)
    label_font, caption_font = _font(int(panel_size * 0.042)), _font(int(panel_size * 0.034))
    gray = (110, 107, 122)
    for s in range(S):
        x = label_w + s * panel_size
        draw.text((x + panel_size / 2, header * 0.6), f"Input {s + 1}", fill=gray, font=label_font, anchor="mm")
        for r, (_, labels, joints) in enumerate(rows):
            image = render_panel(normalize(point_clouds[s]), np.asarray(labels[s]), joints,
                                  camera_pose, toward_camera, floor_z, size)
            figure.paste(image, (x, header + r * panel_size))
    for r, (title, _, _) in enumerate(rows):
        draw.text((label_w * 0.55, header + (r + 0.5) * panel_size), title, fill=gray, font=label_font, anchor="mm")
    if caption:
        draw.text((width / 2, header + 2 * panel_size + footer * 0.45), caption,
                  fill=gray, font=caption_font, anchor="mm")
    return figure
