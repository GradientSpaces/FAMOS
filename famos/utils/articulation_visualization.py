"""Matplotlib figure of the validation predictions logged to W&B.

One column per observed state: GT segmentation and joint axes on top, the prediction
below. Moving parts are colored by id, the static base is grey.
"""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def _axis_bounds(points, axis_origin, axis_dir, extension=0.1):
    """Endpoints of the axis segment that spans `points` projected onto the axis, plus `extension`."""
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-6)
    projections = np.dot(points - axis_origin, axis_dir)
    s_min = np.min(projections) - extension
    s_max = np.max(projections) + extension
    return axis_origin + s_min * axis_dir, axis_origin + s_max * axis_dir


def _part_colors(labels: np.ndarray) -> np.ndarray:
    """Map part IDs to RGB colors. 0 is always grey."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float64)
    if labels.shape[0] == 0:
        return colors
    cmap = plt.get_cmap("tab20")
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = labels == lab
        if lab == 0:
            colors[idx] = np.array([0.68, 0.68, 0.68], dtype=np.float64)
        else:
            colors[idx] = np.asarray(cmap(int(lab) % 20)[:3], dtype=np.float64)
    return colors


def render_articulation_comparison(
    point_clouds: np.ndarray,  # (S, N, 3)
    segmentation_labels_gt: np.ndarray,  # (S, N)
    segmentation_labels_pred: np.ndarray,  # (S, N)
    gt_joint: list,
    pred_joint: list,
    vis_metrics: dict,
    view_angles: list[tuple[float, float]],
) -> Image.Image:
    """Render the 2 x S grid (GT row, prediction row) as a PIL image.

    Joints are dicts with `origin` and `axis`; `view_angles` holds one matplotlib
    (elev, azim) per state.
    """
    S = point_clouds.shape[0]

    fig, ax = plt.subplots(
        nrows=2, ncols=S, figsize=(4 * S, 8), subplot_kw={"projection": "3d"}
    )
    if S == 1:
        ax = np.expand_dims(ax, axis=1)

    fig.patch.set_facecolor("#1a1a2e")

    # Shared axis limits from all rendered points/joints so translated outputs stay in frame.
    cloud_all = point_clouds.reshape(-1, 3)
    mins = cloud_all.min(axis=0)
    maxs = cloud_all.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * np.max(maxs - mins)
    half = max(half * 1.15, 0.55)

    for joint in gt_joint + pred_joint:
        origin = np.asarray(joint["origin"], dtype=float).reshape(3)
        mins = np.minimum(mins, origin)
        maxs = np.maximum(maxs, origin)

    center = 0.5 * (mins + maxs)
    half = max(0.5 * np.max(maxs - mins) * 1.15, half)

    def _setup_ax(ax_sub, title: str, state_idx: int):
        elev_i, azim_i = view_angles[state_idx]
        ax_sub.set_facecolor("#1a1a2e")
        ax_sub.set_title(title, color="white", fontsize=10, pad=4)
        ax_sub.set_xlim(center[0] - half, center[0] + half)
        ax_sub.set_ylim(center[1] - half, center[1] + half)
        ax_sub.set_zlim(center[2] - half, center[2] + half)
        ax_sub.view_init(elev=elev_i, azim=azim_i)
        ax_sub.set_axis_off()

    def _scatter_by_colors(ax_sub, pts, colors, size=2.0, alpha=0.9):
        if pts.shape[0] == 0:
            return
        ax_sub.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=size, alpha=alpha, linewidths=0
        )

    def _draw_joints(ax_sub, joints, color: str, all_points: np.ndarray, mask: np.ndarray):
        """Draw the joint axes, each bounded by the points of its part."""
        for joint in joints:
            part_pts = all_points[mask == joint["part_id"]]
            if len(part_pts) == 0:
                part_pts = all_points[mask != 0]
                if len(part_pts) == 0:
                    continue

            origin = np.asarray(joint["origin"], dtype=float)
            axis = np.asarray(joint["axis"], dtype=float)
            p0, p1 = _axis_bounds(part_pts, origin, axis, extension=0.05)
            ax_sub.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=color,
                linewidth=3.0,
                zorder=10,
            )
            ax_sub.scatter(*origin, c=color, s=60, marker="x", zorder=11)

    for i in range(S):
        ax_gt = ax[0, i]
        _setup_ax(ax_gt, f"State {i} (GT)", i)
        pts = point_clouds[i]
        mask_gt = segmentation_labels_gt[i]
        _scatter_by_colors(ax_gt, pts, _part_colors(mask_gt), size=1.5, alpha=0.9)
        _draw_joints(ax_gt, gt_joint, "#ffd700", pts, mask_gt)

        ax_pred = ax[1, i]
        mask_pred = segmentation_labels_pred[i]
        acc = float((mask_pred == mask_gt).mean())
        _setup_ax(ax_pred, f"State {i} (Pred) | Acc: {acc:.1%}", i)
        _scatter_by_colors(ax_pred, pts, _part_colors(mask_pred), size=1.5, alpha=0.9)
        _draw_joints(ax_pred, pred_joint, "#cc5de8", pts, mask_pred)

    title_str = (
        f"Obj: {vis_metrics['obj_id']} | Joint: {vis_metrics.get('j_type', 'unknown')} | "
        f"Axis Err: {vis_metrics['axis_err']:.2f}° | Orig Err: {vis_metrics['orig_err']:.4f}m"
    )
    fig.suptitle(title_str, color="white", fontsize=14, fontweight="bold")

    plt.subplots_adjust(
        top=0.85, bottom=0.05, left=0.02, right=0.98, wspace=0.1, hspace=0.1
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
