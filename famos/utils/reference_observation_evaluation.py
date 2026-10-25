"""Evaluation on the first observation, or jointly on all observations.

By default the first input observation is the reference shared by every input-set size:
this module slices the per-sample tensors to that observation and reuses the standard
segmentation and motion metrics. With `all_states`, parts are matched and scored jointly
over the points of all input observations instead.
"""

from __future__ import annotations

from .motion_evaluation import _to_numpy
from .motion_metrics import evaluate_motion
from .visible_part_evaluation import run_visible_active_joint_evaluation


def evaluate_reference_observation(
    segmentation_labels,
    segmentation_logits,
    num_valid_parts: int,
    object_id: str,
    dataset_name: str,
    joint_types,
    gt_joint_type_ids,
    joint_valid_mask,
    gt_joint_axis,
    gt_joint_origin,
    revolute_plucker_coordinates,
    prismatic_directions,
    min_part_confidence: float = 0.0,
    iou_threshold: float = 0.1,
    min_points: int = 1,
    motion_axis_threshold_deg: float = 5.0,
    motion_origin_threshold: float = 0.05,
    motion_iou_threshold: float = 0.5,
    all_states: bool = False,
) -> dict:
    """Run the segmentation and motion evaluation on the first observation.

    Returns the same dict as ``evaluate_motion``. The prediction still uses the full
    input set; only the evaluation target is fixed to the first state, or to all states
    together if `all_states` is set.
    """
    states = slice(None) if all_states else slice(0, 1)
    active0 = run_visible_active_joint_evaluation(
        segmentation_labels=_to_numpy(segmentation_labels)[states],
        segmentation_logits=_to_numpy(segmentation_logits)[states],
        num_valid_parts=num_valid_parts,
        joint_types=joint_types,
        gt_joint_type_ids=gt_joint_type_ids,
        joint_valid_mask=joint_valid_mask,
        min_part_confidence=min_part_confidence,
        iou_threshold=iou_threshold,
        min_points=min_points,
    )
    return evaluate_motion(
        active0,
        gt_joint_type_ids=gt_joint_type_ids,
        gt_joint_axis=gt_joint_axis,
        gt_joint_origin=gt_joint_origin,
        joint_types=joint_types,
        revolute_plucker_coordinates=revolute_plucker_coordinates,
        prismatic_directions=prismatic_directions,
        axis_threshold_deg=motion_axis_threshold_deg,
        origin_threshold=motion_origin_threshold,
        iou_threshold=motion_iou_threshold,
        object_id=object_id,
        dataset_name=dataset_name,
    )
