"""Visible-part segmentation matching and the validation totals."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .motion_evaluation import _to_numpy


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _filter_pred_ids_by_confidence(
    logits: np.ndarray, min_part_confidence: float
) -> np.ndarray:
    """Per-point argmax query; queries whose mean point probability is below the threshold become -1."""
    pred_ids = np.argmax(logits, axis=-1).astype(np.int64)
    if min_part_confidence <= 0.0:
        return pred_ids

    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.maximum(exp_logits.sum(axis=-1, keepdims=True), 1e-12)

    keep = np.zeros(logits.shape[-1], dtype=bool)
    for q_idx in range(logits.shape[-1]):
        q_points = pred_ids == q_idx
        if q_points.any():
            keep[q_idx] = probs[q_points, q_idx].mean() >= min_part_confidence
    return np.where(keep[pred_ids], pred_ids, -1).astype(np.int64)


def _iou_matrix(pred_flat, pred_candidates, gt_flat, gt_candidates) -> np.ndarray:
    iou_matrix = np.zeros((len(pred_candidates), len(gt_candidates)), dtype=np.float32)
    for row, q_idx in enumerate(pred_candidates):
        pred_mask = pred_flat == q_idx
        for col, part_id in enumerate(gt_candidates):
            gt_mask = gt_flat == part_id
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou_matrix[row, col] = float(intersection / union) if union > 0 else 0.0
    return iou_matrix


def run_visible_active_joint_evaluation(
    segmentation_labels,
    segmentation_logits,
    num_valid_parts: int,
    joint_types,
    gt_joint_type_ids,
    joint_valid_mask,
    min_part_confidence: float = 0.0,
    iou_threshold: float = 0.10,
    min_points: int = 1,
) -> dict:
    """Match predicted movable-part queries to the visible GT movable parts.

    One Hungarian assignment on point IoU is found over all observed states together
    (`segmentation_labels` (S, N), `segmentation_logits` (S, N, Q)); matches below
    `iou_threshold` are dropped. Only queries predicted as revolute or prismatic and GT
    parts with a valid revolute or prismatic joint take part.
    """
    segmentation_labels = _to_numpy(segmentation_labels).astype(np.int64)
    logits = _to_numpy(segmentation_logits)
    num_part_queries = int(logits.shape[-1])
    pred_ids = _filter_pred_ids_by_confidence(logits, float(min_part_confidence))
    pred_types = _to_numpy(joint_types).astype(np.int64).reshape(-1)
    gt_joint_type_ids = _to_numpy(gt_joint_type_ids).astype(np.int64).reshape(-1)
    joint_valid_mask = _to_numpy(joint_valid_mask).astype(bool).reshape(-1)

    gt_flat = segmentation_labels.reshape(-1)
    gt_candidates = [
        part_id
        for part_id in range(1, int(num_valid_parts))
        if np.any(gt_flat == part_id)
        and joint_valid_mask[part_id]
        and int(gt_joint_type_ids[part_id]) in (1, 2)
    ]
    pred_flat = pred_ids.reshape(-1)
    pred_candidates = [
        q_idx
        for q_idx in range(num_part_queries)
        if int(pred_types[q_idx]) in (1, 2)
        and int(np.sum(pred_flat == q_idx)) >= int(min_points)
    ]

    iou_matrix = _iou_matrix(pred_flat, pred_candidates, gt_flat, gt_candidates)
    matches = []
    if iou_matrix.size > 0:
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
        for row, col in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            iou = float(iou_matrix[row, col])
            if iou < float(iou_threshold):
                continue
            q_idx = int(pred_candidates[row])
            part_id = int(gt_candidates[col])
            matches.append(
                {
                    "gt_part_id": part_id,
                    "pred_query_id": q_idx,
                    "iou": iou,
                    "gt_joint_type": int(gt_joint_type_ids[part_id]),
                    "joint_types": int(pred_types[q_idx]),
                }
            )

    tp = len(matches)
    return {
        "visible_active_joint_total_gt": len(gt_candidates),
        "visible_active_joint_total_pred": len(pred_candidates),
        "visible_active_joint_tp": tp,
        "visible_active_joint_fp": len(pred_candidates) - tp,
        "visible_active_joint_fn": len(gt_candidates) - tp,
        "visible_active_joint_precision": _safe_div(tp, len(pred_candidates)),
        "visible_active_joint_recall": _safe_div(tp, len(gt_candidates)),
        "visible_active_joint_gt_part_ids": gt_candidates,
        "visible_active_joint_pred_types": [int(pred_types[q]) for q in pred_candidates],
        "visible_active_joint_matches": matches,
    }


def run_visible_part_iou_evaluation(
    segmentation_labels,
    segmentation_logits,
    num_valid_parts: int,
    min_part_confidence: float = 0.0,
    iou_threshold: float = 0.10,
    min_points: int = 1,
) -> dict:
    """Match raw queries to the visible GT parts (base included) by point IoU.

    One assignment is found over all observed states. Returns the query -> GT part map
    (`visible_part_query_to_gt`, -1 if unmatched), the mean IoU over visible GT parts with
    missed parts counted as 0 (`visible_part_mIoU_all`, and its per-state average
    `avg_state_visible_part_mIoU_all`), and per-point predicted labels in GT ids, where
    unmatched queries get new ids from `num_valid_parts` on.
    """
    segmentation_labels = _to_numpy(segmentation_labels).astype(np.int64)
    logits = _to_numpy(segmentation_logits)
    num_valid_parts = int(num_valid_parts)
    min_points = int(min_points)
    num_part_queries = int(logits.shape[-1])
    pred_ids = _filter_pred_ids_by_confidence(logits, float(min_part_confidence))

    gt_flat_all = segmentation_labels.reshape(-1)
    valid_flat = (gt_flat_all >= 0) & (gt_flat_all < num_valid_parts)
    gt_flat = gt_flat_all[valid_flat]
    pred_flat = pred_ids.reshape(-1)[valid_flat]
    gt_candidates = [
        part_id
        for part_id in range(num_valid_parts)
        if int(np.sum(gt_flat == part_id)) >= min_points
    ]
    pred_candidates = [
        q_idx
        for q_idx in range(num_part_queries)
        if int(np.sum(pred_flat == q_idx)) >= min_points
    ]

    iou_matrix = _iou_matrix(pred_flat, pred_candidates, gt_flat, gt_candidates)
    query_to_gt = [-1 for _ in range(num_part_queries)]
    gt_to_query: dict[int, int] = {}
    gt_iou = {int(part_id): 0.0 for part_id in gt_candidates}
    if iou_matrix.size > 0:
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
        for row, col in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            iou = float(iou_matrix[row, col])
            if iou < float(iou_threshold):
                continue
            q_idx = int(pred_candidates[row])
            part_id = int(gt_candidates[col])
            query_to_gt[q_idx] = part_id
            gt_to_query[part_id] = q_idx
            gt_iou[part_id] = iou

    predicted_labels = np.full(pred_ids.shape, -1, dtype=np.int64)
    next_extra_id = num_valid_parts
    for q_idx in range(num_part_queries):
        q_mask = pred_ids == q_idx
        if not q_mask.any():
            continue
        if query_to_gt[q_idx] >= 0:
            predicted_labels[q_mask] = query_to_gt[q_idx]
        else:
            predicted_labels[q_mask] = next_extra_id
            next_extra_id += 1

    state_mious = []
    for gt_s, pred_s in zip(segmentation_labels, pred_ids, strict=True):
        valid_s = (gt_s >= 0) & (gt_s < num_valid_parts)
        part_ious = []
        for part_id in gt_candidates:
            gt_mask = (gt_s == part_id) & valid_s
            if int(np.sum(gt_mask)) < min_points:
                continue
            q_idx = gt_to_query.get(part_id)
            pred_mask = (
                (pred_s == q_idx) & valid_s
                if q_idx is not None
                else np.zeros_like(gt_mask, dtype=bool)
            )
            union = np.logical_or(gt_mask, pred_mask).sum()
            part_ious.append(
                float(np.logical_and(gt_mask, pred_mask).sum() / union)
                if union > 0
                else 0.0
            )
        if part_ious:
            state_mious.append(float(np.mean(part_ious)))

    return {
        "visible_part_mIoU_all": (
            float(np.mean([gt_iou[part_id] for part_id in gt_candidates]))
            if gt_candidates
            else float("nan")
        ),
        "avg_state_visible_part_mIoU_all": (
            float(np.mean(state_mious)) if state_mious else float("nan")
        ),
        "visible_part_query_to_gt": query_to_gt,
        "predicted_segmentation_labels": predicted_labels,
    }


# Validation metric key -> name logged as val/<dataset>/<name>.
VALIDATION_VISIBLE_METRIC_KEYS = {
    "avg_state_visible_part_mIoU_all": "part_mIoU",
    "visible_active_joint_micro_recall": "detection_recall",
    "visible_active_joint_micro_precision": "detection_precision",
    "visible_joint_type_accuracy_micro": "joint_type_acc",
    "visible_joint_origin_error_mean": "origin_err_m",
    "visible_joint_axis_error_deg_mean": "axis_err_deg",
}

# Per-dataset sums accumulated over validation and reduced over DDP ranks.
VALIDATION_VISIBLE_TOTAL_KEYS = (
    "avg_state_visible_part_mIoU_all_sum",
    "avg_state_visible_part_mIoU_all_count",
    "visible_active_joint_tp",
    "visible_active_joint_fp",
    "visible_active_joint_fn",
    "visible_joint_type_correct",
    "visible_joint_type_total",
    "visible_joint_origin_error_sum",
    "visible_joint_origin_error_count",
    "visible_joint_axis_error_deg_sum",
    "visible_joint_axis_error_deg_count",
)


def empty_validation_visible_eval_totals() -> dict[str, float]:
    return {key: 0.0 for key in VALIDATION_VISIBLE_TOTAL_KEYS}


def _accumulate_finite_values(
    totals: dict[str, float], sum_key: str, count_key: str, values
) -> None:
    for value in values:
        if np.isfinite(float(value)):
            totals[sum_key] += float(value)
            totals[count_key] += 1.0


def accumulate_validation_visible_eval_result(
    totals: dict[str, float], result: dict
) -> None:
    miou = float(result["avg_state_visible_part_mIoU_all"])
    if np.isfinite(miou):
        totals["avg_state_visible_part_mIoU_all_sum"] += miou
        totals["avg_state_visible_part_mIoU_all_count"] += 1.0
    for key in (
        "visible_active_joint_tp",
        "visible_active_joint_fp",
        "visible_active_joint_fn",
        "visible_joint_type_correct",
        "visible_joint_type_total",
    ):
        totals[key] += float(result[key])
    _accumulate_finite_values(
        totals,
        "visible_joint_origin_error_sum",
        "visible_joint_origin_error_count",
        result["visible_joint_origin_error_values"],
    )
    _accumulate_finite_values(
        totals,
        "visible_joint_axis_error_deg_sum",
        "visible_joint_axis_error_deg_count",
        result["visible_joint_axis_error_deg_values"],
    )


def compute_validation_visible_metrics_from_totals(
    totals_by_dataset: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    def _mean(totals: dict[str, float], sum_key: str, count_key: str) -> float:
        count = float(totals.get(count_key, 0.0))
        return float(totals.get(sum_key, 0.0) / count) if count > 0.0 else float("nan")

    metrics_by_dataset = {}
    for dataset_name, totals in totals_by_dataset.items():
        tp = float(totals.get("visible_active_joint_tp", 0.0))
        fp = float(totals.get("visible_active_joint_fp", 0.0))
        fn = float(totals.get("visible_active_joint_fn", 0.0))
        type_total = float(totals.get("visible_joint_type_total", 0.0))

        metrics_by_dataset[str(dataset_name)] = {
            "avg_state_visible_part_mIoU_all": _mean(
                totals,
                "avg_state_visible_part_mIoU_all_sum",
                "avg_state_visible_part_mIoU_all_count",
            ),
            "visible_active_joint_micro_recall": _safe_div(tp, tp + fn),
            "visible_active_joint_micro_precision": _safe_div(tp, tp + fp),
            "visible_joint_type_accuracy_micro": (
                float(totals.get("visible_joint_type_correct", 0.0) / type_total)
                if type_total > 0.0
                else float("nan")
            ),
            "visible_joint_origin_error_mean": _mean(
                totals,
                "visible_joint_origin_error_sum",
                "visible_joint_origin_error_count",
            ),
            "visible_joint_axis_error_deg_mean": _mean(
                totals,
                "visible_joint_axis_error_deg_sum",
                "visible_joint_axis_error_deg_count",
            ),
        }
    return metrics_by_dataset
