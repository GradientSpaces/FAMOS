"""Joint-geometry errors and the motion evaluation used during validation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


def _finite_values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _safe_mean(values: Iterable[float]) -> float:
    vals = _finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def _safe_median(values: Iterable[float]) -> float:
    vals = _finite_values(values)
    return float(np.median(vals)) if vals else float("nan")


def _normalize(vec) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 3:
        return None
    arr = arr[:3]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


def axis_error_deg(pred_axis, gt_axis) -> float:
    """Angle between two undirected axes, in degrees."""
    pred = _normalize(pred_axis)
    gt = _normalize(gt_axis)
    if pred is None or gt is None:
        return float("nan")
    dot = float(np.clip(np.dot(pred, gt), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(abs(dot))))


def plucker_to_axis_origin(plucker) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Unit axis and line point of Plucker coordinates (d, m) with moment m = origin x d.

    The point on the line closest to the world origin is d x m / ||d||^2.
    """
    arr = np.asarray(plucker, dtype=np.float64).reshape(-1)
    direction = arr[:3]
    moment = arr[3:6]
    d_sq = float(np.dot(direction, direction))
    if d_sq < 1e-12:
        return None, None
    axis = direction / math.sqrt(d_sq)
    origin = np.cross(direction, moment) / d_sq
    return axis, origin


def revolute_origin_error(pred_axis, pred_origin, gt_axis, gt_origin) -> float:
    """Distance between the predicted and GT axis lines."""
    pred_axis = _normalize(pred_axis)
    gt_axis = _normalize(gt_axis)
    if pred_axis is None or gt_axis is None or pred_origin is None or gt_origin is None:
        return float("nan")
    pred_origin = np.asarray(pred_origin, dtype=np.float64).reshape(3)
    gt_origin = np.asarray(gt_origin, dtype=np.float64).reshape(3)
    w0 = pred_origin - gt_origin
    normal = np.cross(pred_axis, gt_axis)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-10:
        return float(np.linalg.norm(np.cross(w0, pred_axis)))
    return float(abs(np.dot(w0, normal)) / normal_norm)


def _axis_for_pred_type(
    pred_type: int, q_idx: int, revolute_plucker_coordinates, prismatic_directions
):
    if pred_type == 1:
        axis, _ = plucker_to_axis_origin(revolute_plucker_coordinates[q_idx])
        return axis
    if pred_type == 2:
        return _normalize(prismatic_directions[q_idx])
    return None


def evaluate_visible_part_motion(
    visible_active_result: dict,
    gt_joint_axis,
    gt_joint_origin,
    joint_types,
    revolute_plucker_coordinates,
    prismatic_directions,
    origin_length_scale: float = 1.0,
) -> dict:
    """Joint-type accuracy and axis/origin errors on the visible query/GT matches.

    Axis and origin errors are measured only for matches with the correct joint type;
    origin errors (revolute only) are multiplied by `origin_length_scale`.
    """
    gt_axes = _to_numpy(gt_joint_axis)
    gt_origins = _to_numpy(gt_joint_origin)
    pred_types = _to_numpy(joint_types).astype(np.int64).reshape(-1)
    pred_rev = _to_numpy(revolute_plucker_coordinates)
    pred_pri = _to_numpy(prismatic_directions)
    origin_length_scale = float(origin_length_scale)

    type_total = 0
    type_correct = 0
    axis_errors = []
    origin_errors = []
    for match in visible_active_result["visible_active_joint_matches"]:
        gt_part_id = int(match["gt_part_id"])
        q_idx = int(match["pred_query_id"])
        gt_type = int(match["gt_joint_type"])
        pred_type = int(pred_types[q_idx])
        type_total += 1
        if pred_type != gt_type:
            continue
        type_correct += 1

        axis_err = axis_error_deg(
            _axis_for_pred_type(pred_type, q_idx, pred_rev, pred_pri),
            gt_axes[gt_part_id],
        )
        if np.isfinite(axis_err):
            axis_errors.append(axis_err)
        if gt_type == 1:
            pred_axis_rev, pred_origin = plucker_to_axis_origin(pred_rev[q_idx])
            origin_err = revolute_origin_error(
                pred_axis_rev,
                pred_origin,
                gt_axes[gt_part_id],
                gt_origins[gt_part_id],
            )
            if np.isfinite(origin_err):
                origin_errors.append(origin_err * origin_length_scale)

    return {
        "visible_joint_type_correct": int(type_correct),
        "visible_joint_type_total": int(type_total),
        "visible_joint_axis_error_deg_values": [float(v) for v in axis_errors],
        "visible_joint_origin_error_values": [float(v) for v in origin_errors],
    }
