"""Headless face-id rendering, backprojection, and mesh articulation."""

import os

import numpy as np
import trimesh

# One renderer and scene per process, created on first use (see `reset_renderer`).
_RENDERER = None
_RENDERER_RESOLUTION = None
_SCENE = None
_CAMERA_NODE = None
_SCENE_FOV = None
_PYRENDER_READY = False


def pin_egl_vendor() -> None:
    """Prefer NVIDIA's EGL driver for headless rendering when it is available.

    Otherwise GLVND may try Mesa first, which fails without access to /dev/dri. Set
    FAMOS_EGL_VENDOR_JSON to use a vendor file in a non-standard location.
    """
    candidates = [
        os.environ.get("FAMOS_EGL_VENDOR_JSON"),
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "/etc/glvnd/egl_vendor.d/10_nvidia.json",
    ]
    vendor_json = next((p for p in candidates if p and os.path.exists(p)), None)
    if vendor_json is None:
        return

    current = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "").lower()
    if not current or "mesa" in current or current.endswith(".so") or ".so." in current:
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = vendor_json


def reset_renderer() -> None:
    """Drop the cached renderer so the next render creates a fresh EGL context.

    Called in every DataLoader worker, which may inherit the parent's state on fork.
    """
    global _RENDERER, _RENDERER_RESOLUTION, _SCENE, _CAMERA_NODE, _SCENE_FOV
    global _PYRENDER_READY
    pin_egl_vendor()
    _RENDERER = _RENDERER_RESOLUTION = _SCENE = _CAMERA_NODE = _SCENE_FOV = None
    _PYRENDER_READY = False


def _ensure_pyrender():
    """Import pyrender on first use and disable MSAA.

    OpenGL is only loaded in the process that renders, so forked or forkserver
    workers never inherit a half-initialized GL state. MSAA would blend the face-id
    colors at triangle edges, so it is turned off.
    """
    global _PYRENDER_READY
    if _PYRENDER_READY:
        return

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    pin_egl_vendor()

    import pyrender.renderer

    original_gl_enable = pyrender.renderer.glEnable

    def gl_enable_without_msaa(cap):
        if cap == pyrender.renderer.GL_MULTISAMPLE:
            pyrender.renderer.glDisable(pyrender.renderer.GL_MULTISAMPLE)
            return
        original_gl_enable(cap)

    pyrender.renderer.glEnable = gl_enable_without_msaa
    _PYRENDER_READY = True


def _get_renderer(resolution: int):
    global _RENDERER, _RENDERER_RESOLUTION
    _ensure_pyrender()
    import pyrender

    if _RENDERER is None or _RENDERER_RESOLUTION != resolution:
        if _RENDERER is not None:
            _RENDERER.delete()
        try:
            _RENDERER = pyrender.OffscreenRenderer(resolution, resolution)
        except ValueError as exc:
            raise ValueError(
                "Failed to initialize the headless pyrender EGL renderer. "
                f"EGL_DEVICE_ID={os.environ.get('EGL_DEVICE_ID')!r}, "
                "__EGL_VENDOR_LIBRARY_FILENAMES="
                f"{os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')!r}. "
                "If __EGL_VENDOR_LIBRARY_FILENAMES is set, it must reference a GLVND "
                "JSON ICD file (for example 10_nvidia.json), not libEGL_nvidia.so.0."
            ) from exc
        _RENDERER_RESOLUTION = resolution
    return _RENDERER


def _get_scene(fov_deg: float):
    global _SCENE, _CAMERA_NODE, _SCENE_FOV
    _ensure_pyrender()
    import pyrender

    if _SCENE is None or _SCENE_FOV != fov_deg:
        _SCENE = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        camera = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(fov_deg), aspectRatio=1.0, znear=0.01, zfar=10.0
        )
        _CAMERA_NODE = _SCENE.add(camera, pose=np.eye(4, dtype=np.float64))
        _SCENE_FOV = fov_deg
    return _SCENE, _CAMERA_NODE


def camera_pose_from_offsets(
    camera_dist: float, offsets: tuple[float, float, float], yaw_deg: float = 0.0
) -> np.ndarray:
    """Camera-to-world pose looking at the origin from `offsets` rotated by `yaw_deg`."""
    yaw_rad = np.deg2rad(yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    rot_yaw = np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    cam_pos = rot_yaw @ np.array(offsets, dtype=np.float64)

    cam_pos = cam_pos / (np.linalg.norm(cam_pos) + 1e-12) * float(camera_dist)
    forward = -cam_pos
    forward /= np.linalg.norm(forward) + 1e-12
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


class FaceIdRenderer:
    """Renders a mesh with one flat color per face and backprojects visible pixels.

    The face-id colors are built once per mesh; `posed` places the mesh for one
    articulation state, after which `render` can be called for several cameras.
    """

    def __init__(self, faces: np.ndarray, resolution: int, fov_deg: float):
        self.faces = np.asarray(faces, dtype=np.int64)
        n_faces = len(self.faces)
        if n_faces >= (1 << 24):
            raise ValueError(f"Mesh has {n_faces} faces, exceeding 24-bit face ids")
        self.resolution = int(resolution)
        self.fov_deg = float(fov_deg)

        # Every face gets its own three vertices colored with its 24-bit index.
        self.unwelded_faces = np.arange(n_faces * 3, dtype=np.uint32).reshape(-1, 3)
        face_ids = np.arange(n_faces, dtype=np.int64)
        rgb = np.stack(
            [(face_ids >> 16) & 0xFF, (face_ids >> 8) & 0xFF, face_ids & 0xFF], axis=1
        ).astype(np.uint8)
        rgba = np.concatenate(
            [np.repeat(rgb, 3, axis=0), np.full((n_faces * 3, 1), 255, dtype=np.uint8)],
            axis=1,
        )
        self.vertex_rgba = rgba.astype(np.float32) / 255.0
        self._scene = None
        self._mesh_node = None

    def posed(self, vertices: np.ndarray) -> "FaceIdRenderer":
        _ensure_pyrender()
        import pyrender

        self._renderer = _get_renderer(self.resolution)
        self._scene, self._camera_node = _get_scene(self.fov_deg)
        primitive = pyrender.Primitive(
            positions=np.asarray(vertices, dtype=np.float32)[self.faces].reshape(-1, 3),
            color_0=self.vertex_rgba,
            indices=self.unwelded_faces,
        )
        self._mesh_node = self._scene.add(pyrender.Mesh(primitives=[primitive]))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._scene.remove_node(self._mesh_node)
        self._mesh_node = None
        return False

    def render(
        self,
        camera_pose: np.ndarray,
        backproject_pose: np.ndarray,
        focal_scale: float = 1.0,
        principal_point_px: tuple[float, float] = (0.0, 0.0),
        depth_noise_std: float = 0.0,
        depth_noise_rel_std: float = 0.0,
        depth_noise_max: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render from `camera_pose` and backproject the visible pixels.

        The pixels are lifted with `backproject_pose` and the (possibly perturbed)
        intrinsics and depth noise, which simulates a miscalibrated sensor. Returns
        world points (N, 3) and the id of the face each point lies on (N,).
        """
        import pyrender

        self._scene.set_pose(self._camera_node, pose=camera_pose)
        color, depth = self._renderer.render(
            self._scene,
            flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES,
        )

        valid = depth > 0
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        h, w = depth.shape
        ys, xs = np.where(valid)
        f = (1.0 / np.tan(np.deg2rad(self.fov_deg) / 2.0)) * max(focal_scale, 1e-6)
        cx = (w * 0.5) + principal_point_px[0]
        cy = (h * 0.5) + principal_point_px[1]
        nx = ((xs.astype(np.float64) - cx) / (w * 0.5)) / f
        ny = ((cy - ys.astype(np.float64)) / (h * 0.5)) / f
        pts_cam = np.stack([nx, ny, -np.ones(len(xs), dtype=np.float64)], axis=-1)

        depths = depth[valid].astype(np.float64, copy=True)
        if depth_noise_std > 0.0 or depth_noise_rel_std > 0.0:
            depth_noise = np.random.normal(
                0.0, depth_noise_std + np.abs(depths) * depth_noise_rel_std
            )
            if depth_noise_max > 0.0:
                depth_noise = np.clip(depth_noise, -depth_noise_max, depth_noise_max)
            depths = np.maximum(depths + depth_noise, 1e-6)

        pts_cam *= depths[:, None]
        pts_world = pts_cam @ backproject_pose[:3, :3].T + backproject_pose[:3, 3]

        rgb = color[valid, :3].astype(np.int64)
        face_ids = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        return pts_world.astype(np.float32), face_ids.astype(np.int32)


def precompute_joint_vertex_ids(
    faces: np.ndarray, face_part_ids: np.ndarray, kinematics: dict
) -> dict[str, np.ndarray]:
    """Indices of the vertices each joint moves (vertices of its controlled parts)."""
    return {
        joint_name: np.unique(
            faces[np.isin(face_part_ids, joint["controlled_part_ids"])].reshape(-1)
        ).astype(np.int64, copy=False)
        for joint_name, joint in kinematics.items()
    }


def articulate_vertices(
    vertices: np.ndarray,
    kinematics: dict,
    joint_values: dict[str, float],
    joint_vertex_ids: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a copy of `vertices` with every joint set to its value in `joint_values`."""
    verts = np.array(vertices, dtype=np.float64)
    for joint_name in sorted(joint_values):
        vertex_ids = joint_vertex_ids[joint_name]
        if vertex_ids.size == 0:
            continue
        joint = kinematics[joint_name]
        value = float(joint_values[joint_name])
        axis = np.asarray(joint["axis"], dtype=np.float64)
        if joint["type"] == "revolute":
            origin = np.asarray(joint["origin"], dtype=np.float64)
            tf = trimesh.transformations.rotation_matrix(value, axis, origin)
        else:
            tf = np.eye(4, dtype=np.float64)
            tf[:3, 3] = axis * value
        verts[vertex_ids] = verts[vertex_ids] @ tf[:3, :3].T + tf[:3, 3]
    return verts


def sample_or_pad_indices(num_src: int, target_n: int) -> np.ndarray:
    """`target_n` random indices into `num_src` items, repeating items if too few."""
    if num_src >= target_n:
        return np.random.permutation(num_src)[:target_n]
    extra = np.random.choice(num_src, size=target_n - num_src, replace=True)
    out = np.concatenate([np.arange(num_src, dtype=np.int64), extra], axis=0)
    np.random.shuffle(out)
    return out
