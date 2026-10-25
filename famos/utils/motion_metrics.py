"""Motion metrics: the cumulative M / MA / MAO ladder, axis error, and origin error.

Origin errors are distances in the normalized model frame, i.e. fractions of the object
scale.
"""

from __future__ import annotations

import numpy as np

from .motion_evaluation import (
    _axis_for_pred_type,
    _finite_values,
    _safe_mean,
    _safe_median,
    _to_numpy,
    axis_error_deg,
    plucker_to_axis_origin,
    revolute_origin_error,
)

REVOLUTE = 1
PRISMATIC = 2


def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def _pr_f1(correct: float, total_pred: float, total_gt: float) -> tuple[float, float, float]:
    p = float(correct / total_pred) if total_pred > 0 else 0.0
    r = float(correct / total_gt) if total_gt > 0 else 0.0
    return p, r, _f1(p, r)


def evaluate_motion(
    active_eval_res: dict,
    gt_joint_type_ids,
    gt_joint_axis,
    gt_joint_origin,
    joint_types,
    revolute_plucker_coordinates,
    prismatic_directions,
    origin_norm_scale: float = 1.0,
    axis_threshold_deg: float = 5.0,
    origin_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    object_id: str = "",
    dataset_name: str = "",
) -> dict:
    """Compute the M/MA/MAO ladder for one object from its visible-active-joint matches.

    A match counts for the ladder if its IoU reaches `iou_threshold`. M: correct joint
    type; MA: also axis error < `axis_threshold_deg`; MAO: also origin error <
    `origin_threshold` (revolute only; prismatic parts have MAO == MA).
    """
    matches = active_eval_res["visible_active_joint_matches"]
    n_gt = int(active_eval_res["visible_active_joint_total_gt"])
    n_pred = int(active_eval_res["visible_active_joint_total_pred"])
    gt_part_ids = active_eval_res["visible_active_joint_gt_part_ids"]

    gt_types = _to_numpy(gt_joint_type_ids).astype(np.int64).reshape(-1)
    gt_axes = _to_numpy(gt_joint_axis)
    gt_origins = _to_numpy(gt_joint_origin)
    pred_types = _to_numpy(joint_types).astype(np.int64).reshape(-1)
    pred_rev = _to_numpy(revolute_plucker_coordinates)
    pred_pri = _to_numpy(prismatic_directions)
    origin_norm_scale = float(origin_norm_scale)

    n_gt_rev = sum(1 for part_id in gt_part_ids if int(gt_types[int(part_id)]) == REVOLUTE)
    n_pred_rev = sum(
        1 for t in active_eval_res["visible_active_joint_pred_types"] if t == REVOLUTE
    )

    correct_M = 0
    correct_MA = 0
    correct_MAO = 0
    n_matched = 0
    matched_iou_values: list[float] = []
    axis_err_values: list[float] = []
    origin_frac_values: list[float] = []
    match_records: list[dict] = []

    for match in matches:
        gt_part_id = int(match["gt_part_id"])
        q_idx = int(match["pred_query_id"])
        # Matches are captured at a low IoU floor; the reporting threshold is applied here.
        iou = float(match["iou"])
        report_match = iou >= iou_threshold
        gt_type = int(match["gt_joint_type"])
        pred_type = int(pred_types[q_idx])
        eval_M = pred_type == gt_type

        # Errors are kept for every captured match, so the IoU threshold can be changed offline.
        axis_err = float("nan")
        eval_A = False
        origin_frac = float("nan")
        eval_O = False
        if eval_M:
            pred_axis = _axis_for_pred_type(pred_type, q_idx, pred_rev, pred_pri)
            axis_err = axis_error_deg(pred_axis, gt_axes[gt_part_id])
            if np.isfinite(axis_err):
                eval_A = axis_err < axis_threshold_deg
            if gt_type == REVOLUTE:
                pred_axis_rev, pred_origin = plucker_to_axis_origin(pred_rev[q_idx])
                origin_err = revolute_origin_error(
                    pred_axis_rev,
                    pred_origin,
                    gt_axes[gt_part_id],
                    gt_origins[gt_part_id],
                )
                if np.isfinite(origin_err):
                    origin_frac = float(origin_err * origin_norm_scale)
                    eval_O = eval_A and (origin_frac < origin_threshold)

        if report_match:
            n_matched += 1
            matched_iou_values.append(iou)
            if eval_M:
                correct_M += 1
            if np.isfinite(axis_err):
                axis_err_values.append(axis_err)
            if np.isfinite(origin_frac):
                origin_frac_values.append(origin_frac)
            if eval_M and eval_A:
                correct_MA += 1
            if gt_type == REVOLUTE:
                if eval_M and eval_A and eval_O:
                    correct_MAO += 1
            else:
                if eval_M and eval_A:
                    correct_MAO += 1

        # Per-match record for re-scoring without inference; origin_frac is null for prismatic parts.
        match_records.append(
            {
                "iou": iou,
                "gt_type": gt_type,
                "type_correct": bool(eval_M),
                "axis_err_deg": float(axis_err) if np.isfinite(axis_err) else None,
                "origin_frac": float(origin_frac) if np.isfinite(origin_frac) else None,
            }
        )

    p_M, r_M, f1_M = _pr_f1(correct_M, n_pred, n_gt)
    p_MA, r_MA, f1_MA = _pr_f1(correct_MA, n_pred, n_gt)
    p_MAO, r_MAO, f1_MAO = _pr_f1(correct_MAO, n_pred, n_gt)

    return {
        "object_id": str(object_id),
        "dataset_name": str(dataset_name),
        "motion_n_gt": n_gt,
        "motion_n_pred": n_pred,
        "motion_n_gt_rev": int(n_gt_rev),
        "motion_n_pred_rev": int(n_pred_rev),
        "motion_n_matched": int(n_matched),
        "motion_mean_match_iou": _safe_mean(matched_iou_values),
        "motion_correct_M": int(correct_M),
        "motion_correct_MA": int(correct_MA),
        "motion_correct_MAO": int(correct_MAO),
        "motion_precision_M": p_M,
        "motion_recall_M": r_M,
        "motion_f1_M": f1_M,
        "motion_precision_MA": p_MA,
        "motion_recall_MA": r_MA,
        "motion_f1_MA": f1_MA,
        "motion_precision_MAO": p_MAO,
        "motion_recall_MAO": r_MAO,
        "motion_f1_MAO": f1_MAO,
        "motion_matches": match_records,
        "motion_axis_error_deg_values": [float(v) for v in axis_err_values],
        "motion_origin_frac_values": [float(v) for v in origin_frac_values],
        "motion_axis_error_deg_mean": _safe_mean(axis_err_values),
        "motion_axis_error_deg_median": _safe_median(axis_err_values),
        "motion_origin_frac_mean": _safe_mean(origin_frac_values),
        "motion_origin_frac_median": _safe_median(origin_frac_values),
        "motion_axis_threshold_deg": float(axis_threshold_deg),
        "motion_origin_threshold": float(origin_threshold),
        "motion_iou_threshold": float(iou_threshold),
        "motion_origin_norm_scale": origin_norm_scale,
    }


def aggregate_motion_metrics(results: list[dict]) -> dict:
    """Micro-averaged segmentation and M/MA/MAO scores and mean errors over objects.

    Segmentation P/R/F1 counts a movable part matched at IoU >= threshold as a true
    positive, regardless of its motion.
    """

    def _sum(key: str) -> float:
        return float(sum(float(r[key]) for r in results))

    n_gt = _sum("motion_n_gt")
    n_pred = _sum("motion_n_pred")
    n_matched = _sum("motion_n_matched")
    seg_p, seg_r, seg_f1 = _pr_f1(n_matched, n_pred, n_gt)
    axis_values = [
        v for r in results for v in _finite_values(r["motion_axis_error_deg_values"])
    ]
    origin_values = [
        v for r in results for v in _finite_values(r["motion_origin_frac_values"])
    ]
    return {
        "segmentation": {"precision": seg_p, "recall": seg_r, "f1": seg_f1},
        "motion": {
            "matched_parts": int(n_matched),
            "f1_M": _pr_f1(_sum("motion_correct_M"), n_pred, n_gt)[2],
            "f1_MA": _pr_f1(_sum("motion_correct_MA"), n_pred, n_gt)[2],
            "f1_MAO": _pr_f1(_sum("motion_correct_MAO"), n_pred, n_gt)[2],
            "axis_error_deg": _safe_mean(axis_values),
            "origin_error": _safe_mean(origin_values),
        },
    }
