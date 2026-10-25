"""ACD (from the S2O release) -> acd_test.h5.

Reads the interior-completed GLBs of the S2O download (`<input_dir>/<model_id>.glb`, i.e.
`acd/articulated/glb`), whose glTF extras store the part ids and articulations. Static and
unarticulated parts are merged into the base, handles attached to a moving part move with
it, and revolute axes are flipped so that doors open outwards.
"""

import os
import tempfile

import numpy as np
import pygltflib
import trimesh

from common import build_release, parse_args

CATEGORY = "s2o_openables"
NUM_SURFACE_POINTS = 100_000


class ArticulatedGLBProcessor:
    def __init__(self, glb_path, seed=0):
        self.glb_path = glb_path
        self.target_num_points = NUM_SURFACE_POINTS
        self.rng = np.random.default_rng(seed)

        self.submeshes = {}
        self.kinematics = {}

        self._parse_embedded_data()

    def _parse_embedded_data(self):
        """Loads kinematics and part connectivity directly from the GLB."""
        gltf = pygltflib.GLTF2().load(self.glb_path)

        scene_extras = gltf.scenes[0].extras if gltf.scenes and gltf.scenes[0].extras else {}
        parts_conn = scene_extras.get("partsConnectivity", {})
        parts_list = parts_conn.get("parts", [])
        conn_graph = parts_conn.get("connectivityGraph", [])

        pid_to_idx = {}
        pid_to_name = {}

        for i, p_info in enumerate(parts_list):
            if not isinstance(p_info, dict):
                continue
            pid = int(p_info.get("pid", -1))
            if pid != -1:
                pid_to_idx[pid] = i
                raw_label = p_info.get("label")
                if not raw_label:
                    raw_label = "unknown"
                pid_to_name[pid] = str(raw_label).split('_')[0].lower()

        root_extras = gltf.extras if gltf.extras else {}
        articulations = root_extras.get("articulations", [])
        if not articulations and scene_extras:
            articulations = scene_extras.get("articulations", [])

        for i, art in enumerate(articulations):
            pid = art.get("pid")
            if pid is None:
                continue
            pid = int(pid)
            joint_name = f"joint_{i}"
            motion_type = str(art.get("type", "")).lower()
            if motion_type in {"translation", "prismatic"}:
                motion_type = "prismatic"
            elif motion_type in {"rotation", "revolute"}:
                motion_type = "revolute"

            self.kinematics[joint_name] = {
                "type": motion_type,
                "axis": art.get("axis", [0, 0, 1]),
                "origin": art.get("origin", [0, 0, 0]),
                "limits": art.get("limit", [0.0, art.get("rangeMax", 1.0)]),
                "controlled_part_ids": [pid]
            }

        # Unarticulated parts join the base, except handles and knobs attached to a moving part.
        articulated_pids = set()
        for j_info in self.kinematics.values():
            articulated_pids.update(j_info["controlled_part_ids"])

        pid_alias_map = {}
        for pid, idx in pid_to_idx.items():
            if pid > 1 and pid not in articulated_pids:
                part_name = pid_to_name.get(pid, "unknown")
                if "handle" in part_name or "knob" in part_name:
                    fused = False
                    for art_pid in articulated_pids:
                        art_idx = pid_to_idx.get(art_pid, -1)
                        if art_idx != -1 and art_idx < len(conn_graph):
                            if idx in conn_graph[art_idx]:
                                pid_alias_map[pid] = art_pid
                                fused = True
                                break
                    if not fused:
                        pid_alias_map[pid] = 0
                else:
                    pid_alias_map[pid] = 0

        # Tag every node with its part id; trimesh keeps node names, so the id survives the reload.
        node_pids = {}

        def assign_pids_and_rename(node_idx, current_pid):
            if node_idx in node_pids:
                return
            node = gltf.nodes[node_idx]
            node_pid = current_pid
            if node.extras:
                raw_pid = node.extras.get("pid", node.extras.get("belongToPart"))
                if raw_pid is not None:
                    node_pid = int(raw_pid)
                    if node_pid in pid_alias_map:
                        node_pid = pid_alias_map[node_pid]
                    if node_pid > 0 and node_pid not in articulated_pids:
                        node_pid = 0
            node_pids[node_idx] = node_pid
            original_name = node.name if node.name else f"node_{node_idx}"
            node.name = f"S2OPID_{node_pid}_{original_name}"
            if node.children:
                for child_idx in node.children:
                    assign_pids_and_rename(child_idx, node_pid)

        roots = []
        if gltf.scenes:
            for s in gltf.scenes:
                if s.nodes:
                    roots.extend(s.nodes)
        for r in roots:
            assign_pids_and_rename(r, 0)
        for i in range(len(gltf.nodes)):
            if i not in node_pids:
                assign_pids_and_rename(i, 0)

        fd, tmp_path = tempfile.mkstemp(suffix=".glb")
        os.close(fd)
        try:
            gltf.save(tmp_path)
            scene = trimesh.load(tmp_path, force='scene')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        for node_name in scene.graph.nodes_geometry:
            pid = 0
            if "S2OPID_" in node_name:
                try:
                    pid = int(node_name.split("S2OPID_")[1].split("_")[0])
                except Exception:
                    pass
            transform, geometry_name = scene.graph.get(node_name)
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            if pid not in self.submeshes:
                self.submeshes[pid] = []
            self.submeshes[pid].append(mesh)
        del scene

        for pid, mesh_list in self.submeshes.items():
            self.submeshes[pid] = trimesh.util.concatenate(mesh_list)

        self._drop_tiny_parts()
        self._prune_and_reindex_parts()

    def _drop_tiny_parts(self):
        """Empty moving parts too small to receive a single surface sample."""
        part_ids_ordered = [pid for pid, m in self.submeshes.items() if not m.is_empty and pid > 0]
        if not part_ids_ordered:
            return

        areas = np.array([float(self.submeshes[pid].area) for pid in part_ids_ordered], dtype=np.float64)
        total_area = areas.sum()
        if total_area > 0:
            raw_counts = np.round(areas / total_area * self.target_num_points).astype(np.int64)
            for idx, pid in enumerate(part_ids_ordered):
                if raw_counts[idx] == 0:
                    self.submeshes[pid] = trimesh.Trimesh()

    def _prune_and_reindex_parts(self):
        """Drop empty moving parts and renumber the part ids contiguously from 0."""
        removed_pids = []
        for pid, mesh in list(self.submeshes.items()):
            if pid == 0:
                continue
            if mesh.is_empty:
                removed_pids.append(pid)
                del self.submeshes[pid]

        if removed_pids:
            active_pids = set(self.submeshes.keys())
            for j_name in list(self.kinematics.keys()):
                kept = [pid for pid in self.kinematics[j_name]["controlled_part_ids"] if pid in active_pids]
                if not kept:
                    del self.kinematics[j_name]
                else:
                    self.kinematics[j_name]["controlled_part_ids"] = kept

        unique_pids = sorted(
            [pid for pid, mesh in self.submeshes.items() if (not mesh.is_empty) or pid == 0]
        )
        pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

        new_submeshes = {}
        for pid, mesh in self.submeshes.items():
            if not mesh.is_empty or pid == 0:
                new_submeshes[pid_to_idx[pid]] = mesh
        self.submeshes = new_submeshes

        for j_name in self.kinematics.keys():
            self.kinematics[j_name]["controlled_part_ids"] = [
                pid_to_idx[pid] for pid in self.kinematics[j_name]["controlled_part_ids"] if pid in pid_to_idx
            ]

    def _fix_joint_directions(self):
        """Flip revolute axes whose positive rotation moves the part into the base."""
        base_mask = self.point_ids == 0
        if not np.any(base_mask):
            return

        base_pts = self.rest_points[base_mask]
        base_center = (base_pts.max(axis=0) + base_pts.min(axis=0)) / 2.0

        for j_name, j_info in self.kinematics.items():
            if j_info['type'] != 'revolute':
                continue
            moving_mask = np.isin(self.point_ids, j_info['controlled_part_ids'])
            if not np.any(moving_mask):
                continue

            moving_pts = self.rest_points[moving_mask]
            door_center = moving_pts.mean(axis=0)
            outward_vec = door_center - base_center
            norm = np.linalg.norm(outward_vec)
            if norm < 1e-6:
                continue
            outward_vec = outward_vec / norm

            test_val = np.pi / 6
            origin = np.array(j_info['origin'])
            axis = np.array(j_info['axis'])

            def apply_rotation(pts, angle):
                tf = trimesh.transformations.rotation_matrix(angle, axis, origin)
                homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
                return (tf @ homog.T).T[:, :3]

            fwd_center = apply_rotation(moving_pts, test_val).mean(axis=0)
            bwd_center = apply_rotation(moving_pts, -test_val).mean(axis=0)

            if np.dot(bwd_center - door_center, outward_vec) > np.dot(fwd_center - door_center, outward_vec):
                self.kinematics[j_name]['axis'] = (-axis).tolist()

    def process_and_sample(self):
        """Build the rest mesh arrays, sample surface points and fix the joint directions."""
        part_ids_ordered = [pid for pid, m in self.submeshes.items() if not m.is_empty]
        areas = np.array([float(self.submeshes[pid].area) for pid in part_ids_ordered], dtype=np.float64)
        raw_counts = np.round(areas / areas.sum() * self.target_num_points).astype(np.int64)
        raw_counts[-1] += self.target_num_points - raw_counts.sum()   # correct rounding drift

        faces, face_ids, vertices, points, point_ids = [], [], [], [], []
        v_cursor = 0
        for pid, n_samples in zip(part_ids_ordered, raw_counts):
            mesh = self.submeshes[pid]
            faces.append(mesh.faces + v_cursor)
            face_ids.append(np.full(len(mesh.faces), pid))
            vertices.append(mesh.vertices.astype(np.float32))
            v_cursor += len(mesh.vertices)
            if n_samples > 0:
                sampled, _ = trimesh.sample.sample_surface(mesh, int(n_samples), seed=self.rng)
                points.append(sampled.astype(np.float32))
                point_ids.append(np.full(len(sampled), pid, dtype=np.int32))

        self.faces = np.concatenate(faces)
        self.face_ids = np.concatenate(face_ids)
        self.rest_vertices = np.concatenate(vertices)
        self.rest_points = np.concatenate(points)
        self.point_ids = np.concatenate(point_ids)

        self._fix_joint_directions()


if __name__ == "__main__":
    args = parse_args("acd", "S2O download folder acd/articulated/glb")

    def process(key):
        category, model_id = key.split("/")
        assert category == CATEGORY, key
        processor = ArticulatedGLBProcessor(os.path.join(args.input_dir, f"{model_id}.glb"))
        processor.process_and_sample()
        return {
            "faces": processor.faces,
            "face_part_ids": processor.face_ids,
            "rest_vertices": processor.rest_vertices,
            "kinematics": processor.kinematics,
        }

    build_release("acd", args, process)
