"""Browse assets exported by generate.py in a local web viewer.

Each asset is shown twice with a shared camera: the textured GLBs, and a copy
colored by movable part with its joint axes. A slider drives every joint across
its limits.

    python generate_procedural_data/visualize.py procedural_assets
"""
import argparse
import functools
import http.server
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

import generate as gen

VIEWER_HTML = Path(__file__).with_name("viewer.html")


def describe_asset(asset_dir: Path) -> dict:
    """Rest-pose GLB transforms and joints (in the URDF root frame) of one asset."""
    root = ET.parse(asset_dir / "mobility.urdf").getroot()
    joints, parents = gen._collect_joint_maps(root)
    root_link = gen._urdf_find_root_link(root)
    link_T = gen._forward_kin_rest(root, joints, parents, root_link)

    # Every movable joint starts a new part; fixed joints stay in the parent's part.
    children = {}
    for j in joints.values():
        children.setdefault(j["parent"], []).append(j)
    part_of = {root_link: 0}
    parts, hierarchy = [], []
    stack = [root_link]
    while stack:
        link = stack.pop()
        for j in children.get(link, []):
            child = j["child"]
            if j["type"] == "fixed":
                part_of[child] = part_of[link]
            else:
                label = len(parts) + 1
                part_of[child] = label
                hierarchy.append([part_of[link], label])
                T = link_T[j["parent"]] @ gen._T_from_xyzrpy(j["xyz"], j["rpy"])
                axis = T[:3, :3] @ np.array(gen._parse_vec3(j["elem"].find("axis").attrib["xyz"]))
                limit = j["elem"].find("limit")
                parts.append({
                    "label": label,
                    "type": j["type"],
                    "axis": (axis / np.linalg.norm(axis)).round(6).tolist(),
                    "origin": T[:3, 3].round(6).tolist(),
                    "limits": [float(limit.attrib["lower"]), float(limit.attrib["upper"])],
                })
            stack.append(child)

    visuals = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            xyz = gen._parse_vec3(origin.attrib.get("xyz", "0 0 0")) if origin is not None else (0, 0, 0)
            rpy = gen._parse_vec3(origin.attrib.get("rpy", "0 0 0")) if origin is not None else (0, 0, 0)
            T = link_T.get(name, np.eye(4)) @ gen._T_from_xyzrpy(xyz, rpy)
            visuals.append({
                "path": f"{asset_dir.name}/{mesh.attrib['filename']}",
                "part": part_of.get(name, 0),
                "matrix": T.T.reshape(-1).round(7).tolist(),  # column-major for three.js
            })

    manifest = json.loads((asset_dir / "manifest.json").read_text())
    return {"name": asset_dir.name, "family": manifest["family"], "variant": manifest.get("variant"),
            "parts": parts, "hierarchy": hierarchy, "visuals": visuals}


def build_index(assets_root: Path) -> dict:
    by_family = {}
    for d in sorted(assets_root.iterdir()):
        if (d / "mobility.urdf").exists() and (d / "textured_objs").is_dir():
            family = json.loads((d / "manifest.json").read_text())["family"]
            by_family.setdefault(family, []).append(d.name)
    order = [f for f in gen.BUILDERS if f in by_family] + sorted(set(by_family) - set(gen.BUILDERS))
    return {"families": [{"id": f, "assets": by_family[f]} for f in order]}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves viewer.html, the asset index and descriptions, and the asset files."""

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/index.json":
            return self._send(json.dumps(build_index(Path(self.directory))).encode(), "application/json")
        if path.startswith("/describe/"):
            asset_dir = Path(self.directory) / Path(path).stem
            if not (asset_dir / "mobility.urdf").exists():
                return self.send_error(404)
            return self._send(json.dumps(describe_asset(asset_dir)).encode(), "application/json")
        return super().do_GET()

    def _send(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main(assets_root: Path, port: int):
    index = build_index(assets_root)
    if not index["families"]:
        raise SystemExit(f"No exported assets found in {assets_root}. Run generate.py first.")
    handler = functools.partial(Handler, directory=str(assets_root.resolve()))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    n = sum(len(f["assets"]) for f in index["families"])
    print(f"Serving {n} assets from {assets_root} at http://localhost:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("assets", type=Path, help="Output directory of generate.py.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    main(args.assets, args.port)
