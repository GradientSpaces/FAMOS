"""Download MatSynth PBR materials in the folder layout expected by generate.py.

Each material is written to <out>/<name>/{basecolor,metallic,normal,roughness}.png.
Requires `pip install datasets`.
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

MAPS = ["basecolor", "metallic", "normal", "roughness"]


def main(out_dir, limit, resolution):
    out_dir = Path(out_dir)
    ds = load_dataset("gvecchio/MatSynth", split="train", streaming=True)
    ds = ds.select_columns(["name"] + MAPS)
    for i, x in enumerate(tqdm(ds, total=limit, unit="mat")):
        if limit is not None and i >= limit:
            break
        mat_dir = out_dir / x["name"]
        if all((mat_dir / f"{k}.png").exists() for k in MAPS):
            continue
        mat_dir.mkdir(parents=True, exist_ok=True)
        for k in MAPS:
            x[k].resize((resolution, resolution)).save(mat_dir / f"{k}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="materials")
    p.add_argument("--limit", type=int, default=None, help="Download only the first N materials.")
    p.add_argument("--resolution", type=int, default=512)
    args = p.parse_args()
    main(args.out, args.limit, args.resolution)
