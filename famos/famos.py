"""FAMOS Lightning module: forward pass, training objective, and evaluation.

Frozen PartField features are encoded together with coordinates and normals, the
Multi-State Articulation Transformer predicts one set of part queries shared by all
observed states, and Hungarian matching aligns those queries with ground-truth parts
before the loss and the segmentation/motion metrics are computed.
"""

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from pytorch3d.ops import estimate_pointcloud_normals

from .model.matching import HungarianPartMatcher
from .utils.art_utils import (
    compute_per_sample_articulation_metrics,
    get_visual_pivot,
    length_unit_factor,
)
from .utils.articulation_visualization import render_articulation_comparison
from .utils.checkpoint import load_checkpoint_for_module
from .utils.motion_evaluation import evaluate_visible_part_motion
from .utils.motion_metrics import aggregate_motion_metrics
from .utils.prediction_rendering import render_prediction_figure
from .utils.reference_observation_evaluation import evaluate_reference_observation
from .utils.visible_part_evaluation import (
    VALIDATION_VISIBLE_METRIC_KEYS,
    VALIDATION_VISIBLE_TOTAL_KEYS,
    accumulate_validation_visible_eval_result,
    compute_validation_visible_metrics_from_totals,
    empty_validation_visible_eval_totals,
    run_visible_active_joint_evaluation,
    run_visible_part_iou_evaluation,
)

logger = logging.getLogger("Modeling")

# Objective terms logged to W&B at every training step.
TRAIN_LOG_KEYS = (
    "loss",
    "segmentation_loss",
    "revolute_loss",
    "prismatic_loss",
    "type_loss",
    "span_loss",
)

# Per-query model outputs, (B, Q, ...).
QUERY_OUTPUT_KEYS = (
    "revolute_plucker_coordinates",
    "prismatic_directions",
    "joint_type_logits",
    "joint_types",
    "observed_articulation_spans",
)

# Number of validation samples rendered to W&B per validation run.
NUM_VALIDATION_IMAGES = 4


def _clean_nans(obj):
    """Recursively replace NaN floats with None so JSON uses null."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


PARTFIELD_ROOT = Path(__file__).resolve().parents[1] / "PartField"
sys.path.insert(0, str(PARTFIELD_ROOT))

# PartField is vendored as a minimal subset rather than an installable package.
import partfield.model_trainer_pvcnn_only_demo as partfield_demo  # noqa: E402
from partfield.config import setup  # noqa: E402
from partfield.model.PVCNN.encoder_pc import sample_triplane_feat  # noqa: E402


def _load_partfield_backbone_cpu(partfield_checkpoint: str, partfield_config):
    """Load PartField without touching CUDA before Lightning assigns rank devices."""
    original_encoder_cls = partfield_demo.TriPlanePC2Encoder

    def cpu_encoder(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_encoder_cls(*args, **kwargs)

    partfield_demo.TriPlanePC2Encoder = cpu_encoder
    try:
        return partfield_demo.Model.load_from_checkpoint(
            partfield_checkpoint,
            map_location="cpu",
            cfg=partfield_config,
        )
    finally:
        partfield_demo.TriPlanePC2Encoder = original_encoder_cls


class FAMOS(L.LightningModule):
    """Feed-forward 3D articulation modeling from sparse observations."""

    def __init__(
        self,
        articulation_transformer: nn.Module,
        partfield_checkpoint: str,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        initialization_checkpoint: str | None = None,
        lambda_revolute: float = 1.0,
        lambda_prismatic: float = 1.0,
        lambda_type: float = 1.0,
        lambda_segmentation: float = 1.0,
        lambda_span: float = 1.0,
        min_part_confidence: float = 0.0,
        num_part_queries: int = 16,
        log_dir: str = "./outputs",
        visible_active_iou_threshold: float = 0.10,
        visible_active_min_points: int = 1,
        export_visualizations: bool = False,
        motion_axis_threshold_deg: float = 5.0,
        motion_origin_threshold: float = 0.05,
        motion_iou_threshold: float = 0.5,
        eval_all_states: bool = False,
    ):
        """Initialize the FAMOS training and evaluation module.

        Parts are matched at `visible_active_iou_threshold` (a low floor, so errors are
        recorded for loose matches too); the reported metrics use `motion_iou_threshold`.
        The benchmark scores the first input state, or all input states jointly if
        `eval_all_states` is set.
        """
        super().__init__()

        self.articulation_transformer = articulation_transformer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lambda_revolute = float(lambda_revolute)
        self.lambda_prismatic = float(lambda_prismatic)
        self.lambda_type = float(lambda_type)
        self.lambda_segmentation = float(lambda_segmentation)
        self.lambda_span = float(lambda_span)
        self.min_part_confidence = float(min_part_confidence)
        self.part_matcher = HungarianPartMatcher()
        self.log_dir = log_dir
        self.visible_active_iou_threshold = float(visible_active_iou_threshold)
        self.visible_active_min_points = int(visible_active_min_points)
        self.export_visualizations = bool(export_visualizations)
        self.motion_axis_threshold_deg = float(motion_axis_threshold_deg)
        self.motion_origin_threshold = float(motion_origin_threshold)
        self.motion_iou_threshold = float(motion_iou_threshold)
        self.eval_all_states = bool(eval_all_states)
        # Outputs go to <log_dir>/<ckpt_name>/; evaluate.py sets the checkpoint name.
        self.ckpt_name = "outputs"
        self._val_image_round = 0
        self._val_image_batches = set()
        self._val_images = []
        self._val_visible_eval_totals: dict[str, dict[str, float]] = {}

        partfield_config = setup(
            argparse.Namespace(
                config_file=str(PARTFIELD_ROOT / "configs" / "final" / "demo.yaml"),
                opts=[],
            ),
            freeze=False,
        )
        logger.info(f"Loading PartField backbone from {partfield_checkpoint}")
        self.partfield_backbone = _load_partfield_backbone_cpu(
            partfield_checkpoint, partfield_config
        )
        if initialization_checkpoint is not None:
            load_checkpoint_for_module(
                self.articulation_transformer,
                initialization_checkpoint,
                prefix_to_remove="articulation_transformer.",
            )
        self._freeze_partfield()

    def _freeze_partfield(self):
        """Keep PartField frozen and in eval mode (Lightning's train() would switch it back)."""
        self.partfield_backbone.eval()
        for param in self.partfield_backbone.parameters():
            param.requires_grad = False

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._freeze_partfield()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._freeze_partfield()
        self._val_visible_eval_totals = {}
        self._val_images = []
        if self.trainer.sanity_checking:
            self._val_image_batches = set()
            return

        # Render evenly spaced batches, shifted by one every validation run.
        num_batches = max(int(self.trainer.num_val_batches[0]), 1)
        num_images = min(NUM_VALIDATION_IMAGES, num_batches)
        self._val_image_batches = {
            int(((i * num_batches) // num_images + self._val_image_round) % num_batches)
            for i in range(num_images)
        }
        self._val_image_round += 1

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._freeze_partfield()
        self._reference_observation_eval_results = []

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        batch = super().on_before_batch_transfer(batch, dataloader_idx)
        # Camera poses are only needed by the validation and test visualizations.
        if self.training and not self.trainer.sanity_checking:
            batch.pop("camera_pose", None)
        return batch

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        # Normals are estimated on the GPU, only for the points the model consumes.
        B, S, N, _ = batch["point_clouds"].shape
        batch["surface_normals"] = estimate_pointcloud_normals(
            batch["point_clouds"].view(B * S, N, 3), neighborhood_size=30
        ).view(B, S, N, 3)
        return batch

    def _encode_with_partfield(self, data_dict: dict) -> dict:
        """Encode every observation independently with PartField."""
        partfield_point_clouds = data_dict["partfield_point_clouds"]  # (B, S, N_all, 3)
        point_clouds = data_dict["point_clouds"]  # (B, S, N, 3)

        B, S = partfield_point_clouds.shape[:2]
        flat_points_all = partfield_point_clouds.reshape(
            B * S, partfield_point_clouds.shape[2], 3
        )
        flat_points = point_clouds.reshape(B * S, point_clouds.shape[2], 3)

        def normalize_points(points_all, points_subset):
            """Normalize point clouds to the PartField input range [-0.9, 0.9]."""
            bbmin = points_all.min(dim=-2, keepdim=True)[0]
            bbmax = points_all.max(dim=-2, keepdim=True)[0]
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * 0.9 / (bbmax - bbmin).max(dim=-1, keepdim=True)[0]

            points_all_norm = (points_all - center) * scale
            points_subset_norm = (points_subset - center) * scale

            return points_all_norm, points_subset_norm

        points_enc, points_dec = normalize_points(flat_points_all, flat_points)

        with torch.inference_mode():
            pc_feat = self.partfield_backbone.pvcnn(points_enc, points_enc)
            planes = self.partfield_backbone.triplane_transformer(pc_feat)
            _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
            point_feat = sample_triplane_feat(part_planes, points_dec)

        partfield_features = point_feat.reshape(
            B, S, point_feat.shape[1], point_feat.shape[2]
        )
        return {"partfield_features": partfield_features}

    def forward(self, data_dict: dict, feat_dict: dict) -> dict:
        """Predict segmentation and articulation from sparse observations."""
        return self.articulation_transformer(feat_dict=feat_dict, data_dict=data_dict)

    @staticmethod
    def _valid_parts_mask(
        num_valid_parts: torch.Tensor, num_part_queries: int
    ) -> torch.Tensor:
        return torch.arange(num_part_queries, device=num_valid_parts.device).unsqueeze(
            0
        ) < num_valid_parts.unsqueeze(1)

    @staticmethod
    def _sorted_indices_from_matching(matching: torch.Tensor) -> torch.Tensor:
        """Place matched GT part k strictly at index k, and unmatched queries in remaining slots."""
        batch_size, num_part_queries = matching.shape
        sorted_indices = torch.full(
            (batch_size, num_part_queries), -1, dtype=torch.long, device=matching.device
        )
        for b in range(batch_size):
            valid_mask = matching[b] >= 0
            matched_cols = []
            if valid_mask.any():
                valid_cols = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                gt_ids = matching[b, valid_cols]
                for q_idx, gt_id in zip(valid_cols, gt_ids, strict=True):
                    if gt_id < num_part_queries:
                        sorted_indices[b, gt_id] = q_idx
                        matched_cols.append(q_idx.item())

            unmatched_cols = [
                q for q in range(num_part_queries) if q not in matched_cols
            ]
            unmatched_ptr = 0
            for i in range(num_part_queries):
                if sorted_indices[b, i] == -1:
                    sorted_indices[b, i] = torch.tensor(
                        unmatched_cols[unmatched_ptr],
                        dtype=torch.long,
                        device=matching.device,
                    )
                    unmatched_ptr += 1
        return sorted_indices

    @staticmethod
    def _sort_query_dim(
        tensor: torch.Tensor, sorted_indices: torch.Tensor, query_dim: int = 1
    ) -> torch.Tensor:
        if query_dim < 0:
            query_dim += tensor.ndim
        if query_dim != 1:
            dims = list(range(tensor.ndim))
            dims[1], dims[query_dim] = dims[query_dim], dims[1]
            tensor = tensor.permute(*dims)
            inv_dims = [0] * len(dims)
            for i, d in enumerate(dims):
                inv_dims[d] = i
            need_permute_back = True
        else:
            need_permute_back = False
            inv_dims = None

        idx_shape = [sorted_indices.shape[0], sorted_indices.shape[1]] + [1] * (
            tensor.ndim - 2
        )
        idx = sorted_indices.view(*idx_shape).expand(-1, -1, *tensor.shape[2:])
        sorted_tensor = torch.gather(tensor, dim=1, index=idx)
        if need_permute_back:
            sorted_tensor = sorted_tensor.permute(*inv_dims)
        return sorted_tensor

    def _visible_iou_sorted_outputs(self, data_dict: dict, output_dict: dict) -> dict:
        """Match queries to GT parts by visible-point IoU and sort the query dimension so
        that GT part k is predicted by query k. `inference_matching` maps raw queries to
        GT parts (-1 if unmatched)."""
        segmentation_logits = output_dict["segmentation_logits"]
        matching = torch.stack(
            [
                torch.as_tensor(
                    run_visible_part_iou_evaluation(
                        segmentation_labels=data_dict["segmentation_labels"][b],
                        segmentation_logits=segmentation_logits[b],
                        num_valid_parts=int(data_dict["num_valid_parts"][b].item()),
                        min_part_confidence=self.min_part_confidence,
                        iou_threshold=self.visible_active_iou_threshold,
                        min_points=self.visible_active_min_points,
                    )["visible_part_query_to_gt"],
                    device=segmentation_logits.device,
                    dtype=torch.long,
                )
                for b in range(segmentation_logits.shape[0])
            ],
            dim=0,
        )
        sorted_indices = self._sorted_indices_from_matching(matching)
        sorted_outputs = {
            "inference_matching": matching,
            "segmentation_logits": self._sort_query_dim(
                segmentation_logits, sorted_indices, query_dim=3
            ),
        }
        for key in QUERY_OUTPUT_KEYS:
            sorted_outputs[key] = self._sort_query_dim(
                output_dict[key], sorted_indices, query_dim=1
            )
        return sorted_outputs

    def training_objective(self, output_dict: dict, data_dict: dict) -> dict:
        """Compute Eq. (3): segmentation, type, revolute, prismatic, and span losses."""
        segmentation_logits = output_dict["segmentation_logits"]
        device = segmentation_logits.device
        dtype = segmentation_logits.dtype

        segmentation_labels = data_dict["segmentation_labels"].to(
            device=device, dtype=torch.long
        )
        num_valid_parts = data_dict["num_valid_parts"].to(device=device, dtype=torch.long)

        B, S, N, Q = segmentation_logits.shape
        matching = self.part_matcher(
            segmentation_logits.reshape(B, S * N, Q),
            segmentation_labels.reshape(B, S * N),
            num_valid_parts,
        )
        sorted_indices = self._sorted_indices_from_matching(matching)
        sorted_logits = self._sort_query_dim(
            segmentation_logits, sorted_indices, query_dim=3
        )
        sorted_outputs = {
            key: self._sort_query_dim(output_dict[key], sorted_indices, query_dim=1)
            for key in (
                "revolute_plucker_coordinates",
                "prismatic_directions",
                "joint_type_logits",
                "observed_articulation_spans",
            )
        }

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        segmentation_loss = F.cross_entropy(
            sorted_logits.reshape(-1, Q),
            segmentation_labels.reshape(-1),
            ignore_index=-1,
        )
        total_loss = total_loss + (self.lambda_segmentation * segmentation_loss)
        loss_dict["segmentation_loss"] = segmentation_loss

        # Joint targets are padded to Q slots; slot k belongs to GT part k (0 = base).
        gt_plucker = data_dict["gt_plucker"].to(device=device, dtype=dtype)
        gt_axis = F.normalize(
            data_dict["joint_axis"].to(device=device, dtype=dtype), dim=-1, eps=1e-6
        )
        gt_joint_type = data_dict["joint_type_ids"].to(device=device, dtype=torch.long)
        joint_valid_mask = data_dict["joint_valid_mask"].to(device=device, dtype=torch.bool)
        gt_limits = data_dict["joint_limits"].to(device=device, dtype=dtype)
        supervised_parts_mask = (
            self._valid_parts_mask(num_valid_parts, Q) & joint_valid_mask
        )

        # Losses of absent joint types are zeros that stay attached to the graph.
        pred_rev_plucker = sorted_outputs["revolute_plucker_coordinates"]
        rev_mask = supervised_parts_mask & gt_joint_type.eq(1)
        revolute_loss = pred_rev_plucker.sum() * 0.0
        if rev_mask.any():
            revolute_loss = F.l1_loss(pred_rev_plucker[rev_mask], gt_plucker[rev_mask])
        total_loss = total_loss + (self.lambda_revolute * revolute_loss)
        loss_dict["revolute_loss"] = revolute_loss

        predicted_prismatic_directions = sorted_outputs["prismatic_directions"]
        pri_mask = supervised_parts_mask & gt_joint_type.eq(2)
        prismatic_loss = predicted_prismatic_directions.sum() * 0.0
        if pri_mask.any():
            prismatic_loss = F.l1_loss(
                predicted_prismatic_directions[pri_mask], gt_axis[pri_mask]
            )
        total_loss = total_loss + (self.lambda_prismatic * prismatic_loss)
        loss_dict["prismatic_loss"] = prismatic_loss

        jt_logits = sorted_outputs["joint_type_logits"]
        type_loss = jt_logits.sum() * 0.0
        if supervised_parts_mask.any():
            type_loss = F.cross_entropy(
                jt_logits[supervised_parts_mask], gt_joint_type[supervised_parts_mask]
            )
        total_loss = total_loss + (self.lambda_type * type_loss)
        loss_dict["type_loss"] = type_loss

        pred_span = sorted_outputs["observed_articulation_spans"]
        target_span = data_dict["observed_articulation_spans"].to(
            device=device, dtype=dtype
        )
        target_valid = data_dict["observed_articulation_span_valid"].to(
            device=device, dtype=torch.bool
        )
        # USB continuous joints are encoded as zero-limit revolutes; their axis/type are
        # valid, but they have no canonical span target.
        zero_limit_revolute = gt_joint_type.eq(1) & gt_limits.abs().amax(dim=-1).le(1e-8)
        span_mask = (
            supervised_parts_mask
            & target_valid
            & gt_joint_type.gt(0)
            & ~zero_limit_revolute
        )
        span_loss = pred_span.sum() * 0.0
        if span_mask.any():
            span_loss = F.l1_loss(pred_span[span_mask], target_span[span_mask])
            loss_dict["span_loss"] = span_loss
        total_loss = total_loss + (self.lambda_span * span_loss)

        loss_dict["loss"] = total_loss
        return loss_dict

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        feat_dict = self._encode_with_partfield(data_dict)
        output_dict = self.forward(data_dict, feat_dict)
        loss_dict = self.training_objective(output_dict, data_dict)

        # Log each objective term explicitly to keep the W&B step axis consistent.
        batch_size = int(data_dict["point_clouds"].shape[0])
        for k, v in loss_dict.items():
            if k not in TRAIN_LOG_KEYS:
                continue
            metric = v.detach()
            if not torch.isfinite(metric):
                continue
            self.log(
                f"train/{k}",
                metric,
                on_step=True,
                on_epoch=False,
                prog_bar=(k == "loss"),
                logger=True,
                rank_zero_only=True,
                batch_size=batch_size,
            )

        return loss_dict["loss"]

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        if not self.trainer.training:
            return
        # The W&B x-axis for every metric (see log_config_to_wandb).
        self.log(
            "train/optimizer_step",
            float(self.trainer.global_step + 1),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            rank_zero_only=True,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler(optimizer),
                "interval": "step",
                "frequency": 1,
            },
        }

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        feat_dict = self._encode_with_partfield(data_dict)
        output_dict = self.forward(data_dict, feat_dict)
        loss_dict = self.training_objective(output_dict, data_dict)
        batch_size = int(data_dict["point_clouds"].shape[0])

        if not self.trainer.sanity_checking:
            self._accumulate_validation_metrics(data_dict, output_dict)
            if (
                self.trainer.global_rank == 0
                and batch_idx in self._val_image_batches
                and isinstance(self.logger, WandbLogger)
            ):
                sample_idx = (self._val_image_round + len(self._val_images)) % batch_size
                self._val_images.append(
                    self._build_articulation_wandb_image(
                        data_dict,
                        self._visible_iou_sorted_outputs(data_dict, output_dict),
                        sample_idx=sample_idx,
                    )
                )

        self.log(
            "val/loss",
            loss_dict["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss_dict["loss"]

    def _accumulate_validation_metrics(self, data_dict: dict, output_dict: dict) -> None:
        """Add each sample's segmentation and motion results to its dataset's totals."""
        for b in range(output_dict["segmentation_logits"].shape[0]):
            segmentation_labels = (
                data_dict["segmentation_labels"][b].detach().cpu().numpy()
            )
            num_valid_parts = int(data_dict["num_valid_parts"][b].item())
            dataset_name = str(data_dict["dataset_name"][b])
            part_eval = run_visible_part_iou_evaluation(
                segmentation_labels=segmentation_labels,
                segmentation_logits=output_dict["segmentation_logits"][b],
                num_valid_parts=num_valid_parts,
                min_part_confidence=self.min_part_confidence,
                iou_threshold=self.visible_active_iou_threshold,
                min_points=self.visible_active_min_points,
            )
            active_eval = run_visible_active_joint_evaluation(
                segmentation_labels=segmentation_labels,
                segmentation_logits=output_dict["segmentation_logits"][b],
                num_valid_parts=num_valid_parts,
                joint_types=output_dict["joint_types"][b],
                gt_joint_type_ids=data_dict["joint_type_ids"][b],
                joint_valid_mask=data_dict["joint_valid_mask"][b],
                min_part_confidence=self.min_part_confidence,
                iou_threshold=self.visible_active_iou_threshold,
                min_points=self.visible_active_min_points,
            )
            motion_eval = evaluate_visible_part_motion(
                active_eval,
                gt_joint_axis=data_dict["joint_axis"][b],
                gt_joint_origin=data_dict["joint_origin"][b],
                joint_types=output_dict["joint_types"][b],
                revolute_plucker_coordinates=output_dict["revolute_plucker_coordinates"][b],
                prismatic_directions=output_dict["prismatic_directions"][b],
                # Origin errors in meters.
                origin_length_scale=float(data_dict["scale"][b].detach().cpu().item())
                * length_unit_factor(data_dict, b),
            )
            totals = self._val_visible_eval_totals.setdefault(
                dataset_name, empty_validation_visible_eval_totals()
            )
            accumulate_validation_visible_eval_result(
                totals, {**part_eval, **active_eval, **motion_eval}
            )

    def _reduce_validation_visible_eval_totals(self) -> dict[str, dict[str, float]]:
        """Sum the per-dataset validation totals over all DDP ranks."""
        local_totals = self._val_visible_eval_totals
        dataset_names = sorted(local_totals.keys())
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if world_size > 1:
            gathered_dataset_names = [None] * world_size
            dist.all_gather_object(gathered_dataset_names, dataset_names)
            dataset_names = sorted(
                {str(name) for names in gathered_dataset_names for name in (names or [])}
            )
        if not dataset_names:
            return {}

        flat_totals = torch.tensor(
            [
                float(local_totals.get(dataset_name, {}).get(key, 0.0))
                for dataset_name in dataset_names
                for key in VALIDATION_VISIBLE_TOTAL_KEYS
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if world_size > 1:
            dist.all_reduce(flat_totals, op=dist.ReduceOp.SUM)

        values = flat_totals.detach().cpu().tolist()
        num_keys = len(VALIDATION_VISIBLE_TOTAL_KEYS)
        return {
            dataset_name: dict(
                zip(VALIDATION_VISIBLE_TOTAL_KEYS, values[i * num_keys : (i + 1) * num_keys])
            )
            for i, dataset_name in enumerate(dataset_names)
        }

    def on_validation_epoch_end(self):
        """Log the per-dataset validation metrics and the W&B images."""
        metrics_by_dataset = compute_validation_visible_metrics_from_totals(
            self._reduce_validation_visible_eval_totals()
        )
        if self.trainer.global_rank == 0:
            for dataset_name, metrics in metrics_by_dataset.items():
                for metric_name, log_name in VALIDATION_VISIBLE_METRIC_KEYS.items():
                    self.log(
                        f"val/{dataset_name}/{log_name}",
                        torch.tensor(
                            metrics[metric_name], dtype=torch.float32, device=self.device
                        ),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=False,
                        rank_zero_only=True,
                    )
            if (
                not self.trainer.sanity_checking
                and isinstance(self.logger, WandbLogger)
                and self._val_images
            ):
                self.logger.experiment.log({"val_images/examples": self._val_images})
        self._val_images = []
        self._val_visible_eval_totals = {}

    def _gt_joints(self, data_dict: dict, b: int) -> list:
        """GT joints {origin, axis, type, part_id} of sample b; revolute origins are
        moved along the axis next to their part."""
        joints = []
        part_ids = data_dict["segmentation_labels"][b, 0]
        src_points_for_joint = data_dict["point_clouds"][b, 0]
        for k in range(1, int(data_dict["num_valid_parts"][b].item())):
            j_type = data_dict["joint_type"][b][k - 1]
            axis = data_dict["joint_axis"][b, k].cpu().numpy()
            origin = data_dict["joint_origin"][b, k].cpu().numpy()
            if j_type == "revolute":
                moving_mask = part_ids == k
                if moving_mask.any():
                    obj_points = src_points_for_joint[moving_mask]
                elif (part_ids != 0).any():
                    obj_points = src_points_for_joint[part_ids != 0]
                else:
                    obj_points = src_points_for_joint
                axis_for_vis = F.normalize(
                    torch.as_tensor(
                        axis, device=obj_points.device, dtype=obj_points.dtype
                    ),
                    dim=0,
                    eps=1e-6,
                )
                origin_for_vis = torch.as_tensor(
                    origin, device=obj_points.device, dtype=obj_points.dtype
                )
                origin = (
                    get_visual_pivot(axis_for_vis, origin_for_vis, obj_points)
                    .detach()
                    .cpu()
                    .numpy()
                )
            joints.append({"origin": origin, "axis": axis, "type": j_type, "part_id": k})
        return joints

    def _pred_joints(self, data_dict: dict, output_dict: dict, b: int) -> list:
        """Predicted joints of the matched parts of sample b, with the predicted joint type.

        `output_dict` is query-sorted (see `_visible_iou_sorted_outputs`).
        """
        joints = []
        part_ids = data_dict["segmentation_labels"][b, 0]
        src_points_for_joint = data_dict["point_clouds"][b, 0]
        num_q = int(output_dict["revolute_plucker_coordinates"].shape[1])
        for k in range(1, int(data_dict["num_valid_parts"][b].item())):
            if k >= num_q or not (output_dict["inference_matching"][b] == k).any():
                continue
            moving_mask = part_ids == k
            if not moving_mask.any():
                moving_mask = part_ids != 0
            obj_points = src_points_for_joint[moving_mask]
            if obj_points.numel() == 0:
                obj_points = src_points_for_joint

            # 0 = static, 1 = revolute, 2 = prismatic.
            pred_type = int(output_dict["joint_type_logits"][b][k].argmax().item())
            if pred_type == 1:
                plucker = output_dict["revolute_plucker_coordinates"][b][k]
                d = plucker[:3].reshape(3).to(obj_points.device, dtype=obj_points.dtype)
                m = plucker[3:6].reshape(3).to(obj_points.device, dtype=obj_points.dtype)
                d_sq = torch.dot(d, d)
                if d_sq > 1e-12:
                    axis = d / torch.sqrt(d_sq)
                    pivot = get_visual_pivot(axis, torch.cross(d, m, dim=0) / d_sq, obj_points)
                    joints.append(
                        {
                            "origin": pivot.detach().cpu().numpy(),
                            "axis": axis.detach().cpu().numpy(),
                            "type": "revolute",
                            "part_id": k,
                        }
                    )
            elif pred_type == 2:
                axis = F.normalize(
                    output_dict["prismatic_directions"][b][k].reshape(1, 3),
                    dim=-1,
                    eps=1e-6,
                ).reshape(3)
                joints.append(
                    {
                        "origin": obj_points.mean(dim=0).detach().cpu().numpy(),
                        "axis": axis.detach().cpu().numpy(),
                        "type": "prismatic",
                        "part_id": k,
                    }
                )
        return joints

    @staticmethod
    def _get_visual_canonical_transform(data_dict: dict, b: int):
        """Row-vector affine (matrix, offset, direction matrix) from the model frame to the
        object-aligned frame used by the visualizations."""
        return tuple(
            data_dict[key][b].detach().cpu().numpy().astype(np.float64)
            for key in (
                "vis_model_to_canonical_matrix",
                "vis_model_to_canonical_offset",
                "vis_model_to_canonical_dir_matrix",
            )
        )

    @staticmethod
    def _transform_visual_points(points: np.ndarray, transform):
        matrix, offset, _ = transform
        return np.asarray(points, dtype=np.float64) @ matrix + offset

    @staticmethod
    def _transform_visual_joints(joints: list, transform) -> list:
        matrix, offset, dir_matrix = transform
        out = []
        for joint in joints:
            new_joint = dict(joint)
            origin = np.asarray(joint["origin"], dtype=np.float64).reshape(3)
            new_joint["origin"] = origin @ matrix + offset
            axis = np.asarray(joint["axis"], dtype=np.float64).reshape(3) @ dir_matrix
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis = axis / norm
            new_joint["axis"] = axis
            out.append(new_joint)
        return out

    def _canonical_camera_view_angles(self, data_dict: dict, b: int, transform, state_idx: int):
        """Matplotlib (elev, azim) of the capture camera of a state, in the visualization frame."""
        cam_pose = data_dict["camera_pose"][b][state_idx].detach().cpu().numpy()
        cam_pos = np.asarray(cam_pose[:3, 3], dtype=np.float64).reshape(1, 3)
        cam_pos = self._transform_visual_points(cam_pos, transform).reshape(3)
        azim = float(np.rad2deg(np.arctan2(cam_pos[1], cam_pos[0])))
        elev = float(
            np.rad2deg(np.arctan2(cam_pos[2], np.hypot(cam_pos[0], cam_pos[1])))
        )
        return elev, azim

    def _visible_labels_from_sorted_logits(
        self, logits_nq: torch.Tensor, matching: torch.Tensor, num_valid_parts: int
    ) -> np.ndarray:
        """Per-point labels from query-sorted logits: matched queries keep their GT id,
        other confident queries get new ids from `num_valid_parts` on, the rest -1."""
        probs = logits_nq.softmax(dim=-1)
        pred_ids = logits_nq.argmax(dim=-1)
        keep = torch.zeros(
            logits_nq.shape[-1], dtype=torch.bool, device=logits_nq.device
        )
        for q_idx in range(logits_nq.shape[-1]):
            q_points = pred_ids == q_idx
            if q_points.any():
                keep[q_idx] = probs[q_points, q_idx].mean() >= self.min_part_confidence
        pred_ids = torch.where(keep[pred_ids], pred_ids, torch.full_like(pred_ids, -1))
        pred_np = pred_ids.detach().cpu().numpy().astype(np.int64)

        matched_gt = {int(gt_id) for gt_id in matching.tolist() if int(gt_id) >= 0}
        remapped = np.full_like(pred_np, -1)
        next_extra_id = int(num_valid_parts)
        for label in np.unique(pred_np):
            label = int(label)
            if label < 0:
                continue
            if label in matched_gt:
                remapped[pred_np == label] = label
            else:
                remapped[pred_np == label] = next_extra_id
                next_extra_id += 1
        return remapped

    def _build_articulation_wandb_image(
        self, data_dict: dict, output_dict: dict, sample_idx: int
    ) -> wandb.Image:
        """Render one validation sample (query-sorted `output_dict`) as a W&B image."""
        b = int(sample_idx)
        point_clouds = data_dict["point_clouds"][b].detach().cpu().numpy()
        segmentation_labels_gt = (
            data_dict["segmentation_labels"][b].detach().cpu().numpy()
        )
        S, N = point_clouds.shape[:2]
        num_valid_parts = int(data_dict["num_valid_parts"][b].detach().cpu().item())
        segmentation_labels_pred = np.zeros((S, N), dtype=np.int64)
        for s in range(S):
            segmentation_labels_pred[s] = self._visible_labels_from_sorted_logits(
                output_dict["segmentation_logits"][b, s],
                output_dict["inference_matching"][b],
                num_valid_parts,
            )

        gt_joint = self._gt_joints(data_dict, b)
        pred_joint = self._pred_joints(data_dict, output_dict, b)

        dataset_name = str(data_dict["dataset_name"][b])
        object_id = str(data_dict["object_id"][b])
        vis_metrics = {"obj_id": f"{dataset_name}/{object_id}"}
        if gt_joint:
            vis_metrics["j_type"] = gt_joint[0]["type"]
        art_m = compute_per_sample_articulation_metrics(data_dict, output_dict)
        vis_metrics["axis_err"] = float(art_m["joint_axis_err_deg"][b].item())
        vis_metrics["orig_err"] = float(art_m["joint_orig_err"][b].item())

        visual_transform = self._get_visual_canonical_transform(data_dict, b)
        view_angles = [
            self._canonical_camera_view_angles(data_dict, b, visual_transform, s)
            for s in range(S)
        ]
        img = render_articulation_comparison(
            point_clouds=self._transform_visual_points(point_clouds, visual_transform),
            segmentation_labels_gt=segmentation_labels_gt,
            segmentation_labels_pred=segmentation_labels_pred,
            gt_joint=self._transform_visual_joints(gt_joint, visual_transform),
            pred_joint=self._transform_visual_joints(pred_joint, visual_transform),
            vis_metrics=vis_metrics,
            view_angles=view_angles,
        )
        return wandb.Image(
            img,
            caption=(
                f"Canonical geometry, camera-view render | Epoch {self.current_epoch}  "
                f"step {self.global_step}  {dataset_name}/{object_id}"
            ),
        )

    @staticmethod
    def _moving_part_bucket(num_moving_parts: int) -> str:
        if num_moving_parts <= 0:
            return "0_parts"
        if num_moving_parts == 1:
            return "1_part"
        if 2 <= num_moving_parts <= 4:
            return "2_4_parts"
        return "more_than_4_parts"

    def test_step(
        self,
        data_dict: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Evaluate every sample on its first observation (or all, see `eval_all_states`); optionally export figures."""
        feat_dict = self._encode_with_partfield(data_dict)
        output_dict = self.forward(data_dict, feat_dict)
        # Number of observations fed to the model, reported alongside the metrics.
        self._num_input_states_seen = int(data_dict["point_clouds"].shape[1])

        for b in range(data_dict["point_clouds"].shape[0]):
            num_valid_parts = int(data_dict["num_valid_parts"][b].detach().cpu().item())
            result = evaluate_reference_observation(
                segmentation_labels=data_dict["segmentation_labels"][b].detach().cpu().numpy(),
                segmentation_logits=output_dict["segmentation_logits"][b],
                num_valid_parts=num_valid_parts,
                object_id=str(data_dict["object_id"][b]),
                dataset_name=str(data_dict["dataset_name"][b]),
                joint_types=output_dict["joint_types"][b],
                gt_joint_type_ids=data_dict["joint_type_ids"][b],
                joint_valid_mask=data_dict["joint_valid_mask"][b],
                gt_joint_axis=data_dict["joint_axis"][b],
                gt_joint_origin=data_dict["joint_origin"][b],
                revolute_plucker_coordinates=output_dict["revolute_plucker_coordinates"][b],
                prismatic_directions=output_dict["prismatic_directions"][b],
                min_part_confidence=self.min_part_confidence,
                iou_threshold=self.visible_active_iou_threshold,
                min_points=self.visible_active_min_points,
                motion_axis_threshold_deg=self.motion_axis_threshold_deg,
                motion_origin_threshold=self.motion_origin_threshold,
                motion_iou_threshold=self.motion_iou_threshold,
                all_states=self.eval_all_states,
            )
            num_moving = max(num_valid_parts - 1, 0)
            result["num_valid_parts"] = num_valid_parts
            result["num_moving_parts"] = num_moving
            result["moving_part_bucket"] = self._moving_part_bucket(num_moving)
            self._reference_observation_eval_results.append(result)

        if self.export_visualizations:
            self._export_observation_figures(data_dict, output_dict)

    def _export_observation_figures(self, data_dict: dict, output_dict: dict) -> None:
        """Save one figure per object: GT and prediction on every input observation."""
        eval_output_dict = self._visible_iou_sorted_outputs(data_dict, output_dict)
        vis_out_base = os.path.join(self.log_dir, self.ckpt_name, "visualization")
        for b in range(data_dict["point_clouds"].shape[0]):
            states_pts = data_dict["point_clouds"][b].detach().cpu().numpy()
            segmentation_labels = (
                data_dict["segmentation_labels"][b].detach().cpu().numpy()
            )
            dataset_name = str(data_dict["dataset_name"][b])
            obj_id = data_dict["object_id"][b]
            num_valid_parts = int(data_dict["num_valid_parts"][b].detach().cpu().item())

            part_eval = run_visible_part_iou_evaluation(
                segmentation_labels=segmentation_labels,
                segmentation_logits=output_dict["segmentation_logits"][b],
                num_valid_parts=num_valid_parts,
                min_part_confidence=self.min_part_confidence,
                iou_threshold=self.visible_active_iou_threshold,
                min_points=self.visible_active_min_points,
            )
            active_eval = run_visible_active_joint_evaluation(
                segmentation_labels=segmentation_labels,
                segmentation_logits=output_dict["segmentation_logits"][b],
                num_valid_parts=num_valid_parts,
                joint_types=output_dict["joint_types"][b],
                gt_joint_type_ids=data_dict["joint_type_ids"][b],
                joint_valid_mask=data_dict["joint_valid_mask"][b],
                min_part_confidence=self.min_part_confidence,
                iou_threshold=self.visible_active_iou_threshold,
                min_points=self.visible_active_min_points,
            )
            gt_joint = self._gt_joints(data_dict, b)
            pred_joint = self._pred_joints(data_dict, eval_output_dict, b)
            visual_transform = self._get_visual_canonical_transform(data_dict, b)

            gt_types = {j["part_id"]: j["type"] for j in gt_joint}
            n_correct = sum(gt_types.get(j.get("part_id")) == j["type"] for j in pred_joint)
            caption = (
                f"{dataset_name}/{obj_id}   ·   "
                f"mIoU {100 * part_eval['visible_part_mIoU_all']:.1f}   ·   "
                f"P/R {100 * active_eval['visible_active_joint_precision']:.0f}/"
                f"{100 * active_eval['visible_active_joint_recall']:.0f}   ·   "
                f"joint types {n_correct}/{len(gt_types)}"
            )
            img = render_prediction_figure(
                point_clouds=self._transform_visual_points(states_pts, visual_transform),
                segmentation_labels_gt=segmentation_labels,
                segmentation_labels_pred=part_eval["predicted_segmentation_labels"],
                gt_joints=self._transform_visual_joints(gt_joint, visual_transform),
                pred_joints=self._transform_visual_joints(pred_joint, visual_transform),
                front_azim_deg=float(data_dict["canonical_yaw_offset"][b]) - 90.0,
                caption=caption,
            )
            safe_obj = str(obj_id).replace("/", "_").replace(" ", "_")
            safe_ds = dataset_name.replace("/", "_").replace(" ", "_")
            png_path = os.path.join(vis_out_base, safe_ds, f"{safe_obj}.png")
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            img.save(png_path)

    def on_test_epoch_end(self):
        """Write per_object.json and metrics.json for each dataset."""
        by_dataset = {}
        for entry in self._reference_observation_eval_results:
            by_dataset.setdefault(entry["dataset_name"], []).append(entry)
        # Per-state error lists feed the dataset means but are not written per object.
        drop_keys = {"motion_axis_error_deg_values", "motion_origin_frac_values"}
        for ds_name, ds_results in by_dataset.items():
            eval_out_dir = Path(self.log_dir) / self.ckpt_name / f"{ds_name}_eval"
            eval_out_dir.mkdir(parents=True, exist_ok=True)

            per_object = [
                {k: v for k, v in entry.items() if k not in drop_keys}
                for entry in sorted(ds_results, key=lambda x: str(x["object_id"]))
            ]
            with open(eval_out_dir / "per_object.json", "w") as f:
                json.dump(
                    _clean_nans(
                        {
                            "dataset": ds_name,
                            "num_objects": len(ds_results),
                            "objects": per_object,
                        }
                    ),
                    f,
                    indent=4,
                    default=float,
                )

            # Segmentation P/R/F1, matched parts, the M/MA/MAO ladder, and mean axis/origin error.
            metrics_payload = {
                "dataset": ds_name,
                "num_objects": len(ds_results),
                "num_input_states": self._num_input_states_seen,
                "evaluated_states": "all" if self.eval_all_states else "first",
                **aggregate_motion_metrics(ds_results),
                "thresholds": {
                    "iou": self.motion_iou_threshold,
                    "axis_deg": self.motion_axis_threshold_deg,
                    "origin": self.motion_origin_threshold,
                    "min_part_confidence": self.min_part_confidence,
                },
            }
            with open(eval_out_dir / "metrics.json", "w") as f:
                json.dump(_clean_nans(metrics_payload), f, indent=4, default=float)

            seg = metrics_payload["segmentation"]
            mot = metrics_payload["motion"]
            print(
                f"\n  {ds_name}: {len(ds_results)} objects | "
                f"Seg P/R/F1={100 * seg['precision']:.1f}/{100 * seg['recall']:.1f}/"
                f"{100 * seg['f1']:.1f} | #={mot['matched_parts']} "
                f"M/MA/MAO={100 * mot['f1_M']:.1f}/{100 * mot['f1_MA']:.1f}/"
                f"{100 * mot['f1_MAO']:.1f} | "
                f"AE={mot['axis_error_deg']:.1f} OE={mot['origin_error']:.3f}"
            )

        self._reference_observation_eval_results = []
