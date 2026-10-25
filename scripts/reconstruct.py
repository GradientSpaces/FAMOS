"""Reconstruct the masked point cloud of every input state with VGGT-Omega.

    python scripts/reconstruct.py path/to/object --vggt_checkpoint checkpoints/vggt_omega_1b_512.pt

`<object>/images/` holds one photo per articulation state, `<object>/masks/` one binary
object mask per photo (PNG with the same file name stem, nonzero = object). Photos are used
in file name order, and the first one defines the coordinate frame. The result,
`<object>/reconstruction.npz`, is the input of `python predict.py input_dir=<object>`.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import binary_erosion
from torchvision.transforms.functional import to_tensor
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio
from vggt_omega.utils.pose_enc import encoding_to_camera

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_RESOLUTION = 512
PATCH_SIZE = 16
# Point filtering, applied per state: erode the mask to drop depth bleeding across the object
# boundary, then remove the least confident and the farthest depths, and finally floaters far
# from the object.
MASK_EROSION_PIXELS = 4
MIN_CONFIDENCE_PERCENTILE = 1.0
MAX_DEPTH_PERCENTILE = 95.0
MAX_RADIUS_PERCENTILE = 97.0
# VGGT-Omega cameras are OpenCV (y down, z forward); FAMOS uses y up, z toward the viewer.
OPENCV_TO_OPENGL = np.array([1.0, -1.0, -1.0])


def load_inputs(object_dir: Path):
    """Photos and masks, preprocessed exactly like VGGT-Omega's `load_and_preprocess_images`."""
    image_paths = sorted(
        p for p in (object_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    images, masks = [], []
    for image_path in image_paths:
        mask_path = object_dir / "masks" / f"{image_path.stem}.png"
        # Phone photos are often stored sideways with an EXIF tag that rotates them for
        # display. VGGT-Omega ignores the tag, so it is applied here, to the mask as well.
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        mask = ImageOps.exif_transpose(Image.open(mask_path)).convert("L")
        if mask.size != image.size:
            raise ValueError(
                f"{mask_path} is {mask.size[0]}x{mask.size[1]} but {image_path} is "
                f"{image.size[0]}x{image.size[1]}; masks must be drawn on the upright photo."
            )
        image = _crop_to_supported_aspect_ratio(image)
        mask = _crop_to_supported_aspect_ratio(mask)
        height, width = _balanced_target_shape(
            image.height / image.width, IMAGE_RESOLUTION, PATCH_SIZE
        )
        images.append(to_tensor(image.resize((width, height), Image.Resampling.BICUBIC)))
        masks.append(np.asarray(mask.resize((width, height), Image.Resampling.NEAREST)) > 0)
    if len({tuple(image.shape) for image in images}) > 1:
        raise ValueError("All photos of an object must have the same resolution and orientation.")
    return torch.stack(images), np.stack(masks)


@torch.inference_mode()
def run_vggt(checkpoint: Path, images: torch.Tensor):
    device = torch.device("cuda")
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    predictions = model.to(device)(images.to(device))
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
    return (
        predictions["depth"][0, ..., 0].float().cpu().numpy(),
        predictions["depth_conf"][0].float().cpu().numpy(),
        extrinsics[0].float().cpu().numpy(),
        intrinsics[0].float().cpu().numpy(),
    )


def masked_point_cloud(depth, confidence, mask, colors, extrinsic, intrinsic, first_extrinsic):
    """Points of the masked object, and their colors, in the first camera's frame."""
    valid = (
        mask & np.isfinite(depth) & (depth > 1e-5) & np.isfinite(confidence) & (confidence > 1e-5)
    )
    valid &= confidence >= np.percentile(confidence[valid], MIN_CONFIDENCE_PERCENTILE)
    valid &= depth <= np.percentile(depth[valid], MAX_DEPTH_PERCENTILE)

    v, u = np.nonzero(valid)
    z = depth[valid]
    camera = np.stack(
        [
            (u - intrinsic[0, 2]) / intrinsic[0, 0] * z,
            (v - intrinsic[1, 2]) / intrinsic[1, 1] * z,
            z,
        ],
        -1,
    )
    # Extrinsics map world to camera (OpenCV convention): this camera -> world -> first camera.
    world = (camera - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    points = (world @ first_extrinsic[:3, :3].T + first_extrinsic[:3, 3]) * OPENCV_TO_OPENGL

    radius = np.linalg.norm(points - np.median(points, 0), axis=1)
    keep = radius <= np.percentile(radius, MAX_RADIUS_PERCENTILE)
    return points[keep].astype(np.float32), colors[valid][keep]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("object_dir", type=Path, help="Folder with images/ and masks/.")
    parser.add_argument(
        "--vggt_checkpoint", type=Path, default=Path("checkpoints/vggt_omega_1b_512.pt")
    )
    args = parser.parse_args()

    images, masks = load_inputs(args.object_dir)
    masks = np.stack([binary_erosion(m, iterations=MASK_EROSION_PIXELS) for m in masks])
    depth, confidence, extrinsics, intrinsics = run_vggt(args.vggt_checkpoint, images)

    pixel_colors = np.round(images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    states, colors = [], []
    for s in range(len(images)):
        points, point_colors = masked_point_cloud(
            depth[s],
            confidence[s],
            masks[s],
            pixel_colors[s],
            extrinsics[s],
            intrinsics[s],
            extrinsics[0],
        )
        states.append(points)
        colors.append(point_colors)

    output_path = args.object_dir / "reconstruction.npz"
    np.savez_compressed(
        output_path,
        points=np.concatenate(states),
        colors=np.concatenate(colors),
        state_offsets=np.cumsum([0] + [len(p) for p in states]),
    )
    print(f"{len(states)} states with {[len(p) for p in states]} points. Wrote {output_path}")


if __name__ == "__main__":
    main()
