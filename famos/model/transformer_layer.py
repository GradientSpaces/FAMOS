"""Alternating-attention transformer layer used by FAMOS."""

import flash_attn
import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

from .norm import MultiHeadRMSNorm


class MultiStateArticulationTransformerLayer(nn.Module):
    """Per-state self-attention + global joint self-attention layer.

    Architecture per layer:
        1. Per-state self-attention on points (B*S, N, D).
        2. Query self-attention (B, Q, D).
        3. Global joint self-attention over concatenated [queries; points]
           sequence (B, Q + S*N, D).
        4. Feed-forward on queries and points.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0,
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        assert dim == attention_head_dim * num_attention_heads, (
            "dim must be equal to attention_head_dim * num_attention_heads"
        )

        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.attn_dtype = attn_dtype

        self.statewise_prenorm = nn.LayerNorm(dim)
        self.statewise_qkv_projection = nn.Linear(dim, dim * 3, bias=False)
        self.statewise_output_projection = nn.Linear(dim, dim)
        self.statewise_query_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
        self.statewise_key_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        self.part_query_prenorm = nn.LayerNorm(dim)
        self.part_query_qkv_projection = nn.Linear(dim, dim * 3, bias=False)
        self.part_query_output_projection = nn.Linear(dim, dim)
        self.part_query_query_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
        self.part_query_key_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        self.global_prenorm = nn.LayerNorm(dim)
        self.global_qkv_projection = nn.Linear(dim, dim * 3, bias=False)
        self.global_output_projection = nn.Linear(dim, dim)
        self.global_query_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
        self.global_key_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        self.part_query_feed_forward_norm = nn.LayerNorm(dim)
        self.point_token_feed_forward_norm = nn.LayerNorm(dim)
        self.part_query_feed_forward = FeedForward(
            dim, dropout=dropout, activation_fn="geglu"
        )
        self.point_token_feed_forward = FeedForward(
            dim, dropout=dropout, activation_fn="geglu"
        )

    @staticmethod
    def _qk_norm(
        qkv: torch.Tensor, q_norm: nn.Module, k_norm: nn.Module
    ) -> torch.Tensor:
        q, k, v = qkv.unbind(dim=2)
        q, k = q_norm(q).to(v.dtype), k_norm(k).to(v.dtype)
        return torch.stack([q, k, v], dim=2)

    def _flash_self_attn(
        self,
        x: torch.Tensor,
        qkv_proj,
        out_proj,
        q_norm,
        k_norm,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = qkv_proj(x).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = self._qk_norm(qkv, q_norm, k_norm)
        out = flash_attn.flash_attn_qkvpacked_func(qkv.to(self.attn_dtype)).to(x.dtype)
        return out_proj(out.reshape(B, N, C))

    def forward(
        self,
        queries: torch.Tensor,
        x_points: torch.Tensor,
        num_states: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with alternating local and global attention."""
        B, SN, D = x_points.shape
        Q = queries.shape[1]
        S = num_states
        N = SN // S

        # 1) Per-state point self-attention: (B, S*N, D) -> (B*S, N, D)
        x_reshaped = x_points.view(B * S, N, D)
        x_normed = self.statewise_prenorm(x_reshaped)
        state_sa_out = self._flash_self_attn(
            x_normed,
            self.statewise_qkv_projection,
            self.statewise_output_projection,
            self.statewise_query_norm,
            self.statewise_key_norm,
        )
        x_reshaped = x_reshaped + state_sa_out

        # 2) Query self-attention.
        if Q > 1:
            q_normed = self.part_query_prenorm(queries)
            q_sa_out = self._flash_self_attn(
                q_normed,
                self.part_query_qkv_projection,
                self.part_query_output_projection,
                self.part_query_query_norm,
                self.part_query_key_norm,
            )
            queries = queries + q_sa_out

        # 3) Global joint self-attention over [queries; all points].
        x_points_global = x_reshaped.view(B, S * N, D)
        joint_seq = torch.cat([queries, x_points_global], dim=1)  # (B, Q + S*N, D)
        joint_sa_out = self._flash_self_attn(
            self.global_prenorm(joint_seq),
            self.global_qkv_projection,
            self.global_output_projection,
            self.global_query_norm,
            self.global_key_norm,
        )
        joint_seq = joint_seq + joint_sa_out
        queries = joint_seq[:, :Q]
        x_points = joint_seq[:, Q:]

        # 4) Feed-forward.
        queries = queries + self.part_query_feed_forward(
            self.part_query_feed_forward_norm(queries)
        )
        x_points = x_points + self.point_token_feed_forward(
            self.point_token_feed_forward_norm(x_points)
        )
        return queries, x_points
