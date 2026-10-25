"""Hungarian matching of part queries to ground-truth parts."""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianPartMatcher(nn.Module):
    """Match part queries to ground-truth parts from segmentation logits."""

    @torch.no_grad()
    def forward(
        self,
        segmentation_logits: torch.Tensor,
        gt_part_ids: torch.Tensor,
        num_valid_parts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            segmentation_logits: (B, N, Q) logits.
            gt_part_ids: (B, N) integer part IDs.
            num_valid_parts: (B,) number of valid GT parts (base included).

        Returns:
            matched_part_ids: (B, Q) mapped GT part ID for each query column.
                Unmatched queries are set to -1.
        """
        batch_size, _, num_part_queries = segmentation_logits.shape
        device = segmentation_logits.device
        log_probs = torch.log_softmax(segmentation_logits, dim=-1)

        out = []
        for b in range(batch_size):
            n_valid = int(num_valid_parts[b].item())
            if n_valid <= 0:
                out.append(
                    torch.full((num_part_queries,), -1, dtype=torch.long, device=device)
                )
                continue

            part_masks = (
                gt_part_ids[b].unsqueeze(0)
                == torch.arange(n_valid, device=device).unsqueeze(1)
            ).float()  # (P, N)
            log_prob_sums = part_masks @ log_probs[b]  # (P, Q)
            cost = (-log_prob_sums).detach().cpu().float().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)

            mapped = torch.full(
                (num_part_queries,), -1, dtype=torch.long, device=device
            )
            mapped[torch.as_tensor(col_ind, device=device, dtype=torch.long)] = (
                torch.as_tensor(row_ind, device=device, dtype=torch.long)
            )
            out.append(mapped)

        return torch.stack(out, dim=0)
