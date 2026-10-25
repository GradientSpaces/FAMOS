"""Joint representation and per-sample joint errors for the validation images."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class Joint:
    """Axis direction and a point on the axis of a revolute or prismatic joint."""

    VALID_TYPES = ("revolute", "prismatic")

    def __init__(
        self,
        joint_type: str,
        axis_dir: Union[np.ndarray, torch.Tensor, List],
        axis_orig: Union[np.ndarray, torch.Tensor, List],
    ):
        if joint_type not in self.VALID_TYPES:
            raise ValueError(
                f"joint_type must be one of {self.VALID_TYPES}, got '{joint_type}'"
            )
        self.type = joint_type
        self.axis_dir = np.asarray(_to_numpy(axis_dir), dtype=np.float64).reshape(3)
        self.axis_orig = np.asarray(_to_numpy(axis_orig), dtype=np.float64).reshape(3)

    def apply_translation(self, translation: Union[np.ndarray, List]) -> "Joint":
        """Shift the axis origin by *translation* (in-place)."""
        self.axis_orig += np.asarray(translation, dtype=np.float64).reshape(3)
        return self

    def apply_scale(self, scale: float) -> "Joint":
        """Scale the axis origin by *scale* (in-place)."""
        self.axis_orig *= scale
        return self


def _to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().float().numpy()
    return obj


def length_unit_factor(data_dict: dict, b: int) -> float:
    """Factor that converts sample b's lengths to meters."""
    # GRScenes is stored in centimeters, all other datasets in meters.
    return 0.01 if "grscenes" in str(data_dict["dataset_name"][b]).lower() else 1.0


def _joint_from_plucker(joint_type: str, plucker: torch.Tensor) -> Optional[Joint]:
    """Build a Joint from predicted Plucker coordinates [d, m]."""
    d = plucker[:3].detach().cpu().double()
    m = plucker[3:6].detach().cpu().double()
    d_sq = torch.dot(d, d)
    if d_sq < 1e-12:
        return None
    # For Plucker with moment m = p x d, the point closest to the origin is
    # p0 = (d x m) / ||d||^2.
    p = torch.cross(d, m, dim=0) / d_sq
    axis = d / torch.sqrt(d_sq)
    return Joint(joint_type=joint_type, axis_dir=axis.numpy(), axis_orig=p.numpy())


def eval_joint(pred: Joint, gt: Joint) -> Dict[str, float]:
    """Angle between the (undirected) axes and, for revolute joints, the axis-line distance.

    Returns ``joint_axis_err_deg`` and ``joint_orig_err`` (NaN for prismatic joints).
    """
    u = pred.axis_dir / np.linalg.norm(pred.axis_dir)
    v = gt.axis_dir / np.linalg.norm(gt.axis_dir)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    axis_err_rad = float(np.arccos(abs(dot)))
    result = {
        "joint_axis_err_deg": float(np.rad2deg(axis_err_rad)),
        "joint_orig_err": float("nan"),
    }

    if gt.type == "revolute":
        w0 = pred.axis_orig - gt.axis_orig
        n = np.cross(u, v)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-10:
            # Parallel lines: point-to-line distance.
            result["joint_orig_err"] = float(np.linalg.norm(np.cross(w0, u)))
        else:
            # Skew lines.
            result["joint_orig_err"] = float(abs(np.dot(w0, n)) / n_norm)
    return result


def compute_per_sample_articulation_metrics(
    data_dict: dict, output_dict: dict
) -> Dict[str, torch.Tensor]:
    """Per-sample mean axis error (deg) and origin error (m) of the matched parts.

    `output_dict` is query-sorted (query k predicts GT part k). A part is scored if it is
    matched and its joint type is predicted correctly.
    """
    B = int(data_dict["point_clouds"].shape[0])
    device = data_dict["point_clouds"].device
    axis_err_deg = torch.full((B,), float("nan"), device=device)
    orig_err = torch.full((B,), float("nan"), device=device)

    for b in range(B):
        scale_world = data_dict["scale"][b].item() * length_unit_factor(data_dict, b)
        center = data_dict["center"][b].cpu().numpy()
        b_axis_err_deg, b_orig_err = [], []
        for k in range(1, int(data_dict["num_valid_parts"][b].item())):
            joint_type = data_dict["joint_type"][b][k - 1]
            pred_type = {1: "revolute", 2: "prismatic"}.get(
                int(output_dict["joint_types"][b][k].item()), "fixed"
            )
            if pred_type == "revolute":
                pred_joint = _joint_from_plucker(
                    "revolute", output_dict["revolute_plucker_coordinates"][b][k]
                )
            elif pred_type == "prismatic":
                # A prismatic axis has no distinguished origin.
                pred_joint = Joint(
                    joint_type="prismatic",
                    axis_dir=output_dict["prismatic_directions"][b][k],
                    axis_orig=np.zeros(3, dtype=np.float64),
                )
            else:
                continue
            if pred_joint is None:
                continue
            gt_joint = Joint(
                joint_type=joint_type,
                axis_dir=data_dict["joint_axis"][b, k],
                axis_orig=data_dict["joint_origin"][b, k],
            )
            for j in (pred_joint, gt_joint):
                j.apply_scale(scale_world)
                j.apply_translation(center)

            is_matched = (output_dict["inference_matching"][b] == k).any().item()
            if is_matched and pred_joint.type == gt_joint.type:
                res = eval_joint(pred_joint, gt_joint)
                b_axis_err_deg.append(res["joint_axis_err_deg"])
                if not math.isnan(res["joint_orig_err"]):
                    b_orig_err.append(res["joint_orig_err"])

        if b_axis_err_deg:
            axis_err_deg[b] = np.mean(b_axis_err_deg)
        if b_orig_err:
            orig_err[b] = np.mean(b_orig_err)

    return {"joint_axis_err_deg": axis_err_deg, "joint_orig_err": orig_err}


def get_visual_pivot(
    axis: torch.Tensor,
    pivot: torch.Tensor,
    object_points: torch.Tensor,
) -> torch.Tensor:
    """Slide `pivot` along the unit `axis` to the point closest to the object centroid."""
    centroid = object_points.mean(dim=0)
    t = torch.dot(centroid - pivot, axis)
    return pivot + t * axis
