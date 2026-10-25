"""The Multi-State Articulation Transformer from the FAMOS paper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_encoding import PointEncoder
from .transformer_layer import MultiStateArticulationTransformerLayer


class PartQueryEmbedding(nn.Module):
    """Embed fractional query indices with sinusoidal features and an MLP."""

    def __init__(self, embed_dim: int, freq_dim: int = 64):
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _fourier(self, x: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = x.float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, fractions: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._fourier(fractions))


class MultiStateArticulationTransformer(nn.Module):
    """FAMOS transformer with alternating state-wise and global attention."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        attn_dtype: str = "float16",
        num_part_queries: int = 16,
        part_query_frequency_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_part_queries = int(num_part_queries)

        if attn_dtype in {"float16", "fp16"}:
            attention_dtype = torch.float16
        elif attn_dtype in {"bfloat16", "bf16"}:
            attention_dtype = torch.bfloat16
        elif attn_dtype in {"float32", "fp32"}:
            attention_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported attn_dtype: {attn_dtype}")

        self.point_encoder = PointEncoder(in_dim=in_dim, embed_dim=embed_dim)
        self.part_query_embedding = PartQueryEmbedding(
            embed_dim=embed_dim,
            freq_dim=part_query_frequency_dim,
        )
        self.layers = nn.ModuleList(
            [
                MultiStateArticulationTransformerLayer(
                    dim=embed_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=embed_dim // num_heads,
                    dropout=dropout_rate,
                    attn_dtype=attention_dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.segmentation_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, 1),
        )
        self.revolute_plucker_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, 6),
        )
        self.prismatic_direction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, 3),
        )
        self.joint_type_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, 3),
        )
        self.observed_articulation_span_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, 1),
        )

    def _build_query_tokens(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        fractions = torch.arange(
            self.num_part_queries, device=device, dtype=torch.float32
        ) / float(self.num_part_queries)
        query_tokens = self.part_query_embedding(fractions.unsqueeze(-1))
        return query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _decode_pairwise_point_query_logits(
        head: nn.Sequential,
        point_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the mask MLP to point/query pairs without materializing concatenations."""
        first_linear = head[0]
        point_dim = point_tokens.shape[-1]
        point_hidden = F.linear(
            point_tokens,
            first_linear.weight[:, :point_dim],
            first_linear.bias,
        )
        query_hidden = F.linear(
            query_tokens,
            first_linear.weight[:, point_dim:],
            None,
        )
        pair_hidden = head[1](point_hidden.unsqueeze(2) + query_hidden.unsqueeze(1))
        return head[2](pair_hidden).squeeze(-1)

    def forward(self, feat_dict: dict, data_dict: dict) -> dict:
        """Predict per-state segmentation and one shared articulation per part query.

        Inputs are ``(B, S, N, ...)`` tensors for B objects, S observed states, and N
        points per state. The S states of one object share a single set of Q part
        queries, so the motion heads below are state-independent by construction and
        the joint prediction is a property of the object, not of an observation.

        Returns:
            segmentation_logits: (B, S, N, Q) point-to-query assignment logits.
            revolute_plucker_coordinates: (B, Q, 6) revolute axis in Plucker form.
            prismatic_directions: (B, Q, 3) unit translation direction.
            joint_type_logits / joint_types: (B, Q, 3) / (B, Q) over static,
                revolute, prismatic.
            observed_articulation_spans: (B, Q) non-negative motion range covered by
                the observed states, used by the span loss.
        """
        partfield_features = feat_dict["partfield_features"]
        batch_size, num_states, num_points, _ = partfield_features.shape

        point_tokens = self.point_encoder(feat_dict, data_dict).reshape(
            batch_size,
            num_states * num_points,
            self.embed_dim,
        )
        part_queries = self._build_query_tokens(batch_size, point_tokens.device)

        for layer in self.layers:
            part_queries, point_tokens = layer(
                part_queries,
                point_tokens,
                num_states=num_states,
            )

        statewise_point_tokens = point_tokens.view(
            batch_size,
            num_states,
            num_points,
            self.embed_dim,
        )
        part_queries = part_queries.to(dtype=point_tokens.dtype)

        flattened_point_tokens = statewise_point_tokens.reshape(
            batch_size * num_states,
            num_points,
            self.embed_dim,
        )
        repeated_part_queries = (
            part_queries.unsqueeze(1)
            .expand(
                batch_size,
                num_states,
                self.num_part_queries,
                self.embed_dim,
            )
            .reshape(
                batch_size * num_states,
                self.num_part_queries,
                self.embed_dim,
            )
        )
        segmentation_logits = self._decode_pairwise_point_query_logits(
            self.segmentation_head,
            flattened_point_tokens,
            repeated_part_queries,
        ).reshape(
            batch_size,
            num_states,
            num_points,
            self.num_part_queries,
        )

        revolute_plucker_coordinates = self.revolute_plucker_head(part_queries)
        prismatic_directions_raw = self.prismatic_direction_head(part_queries)
        with torch.amp.autocast(point_tokens.device.type, enabled=False):
            revolute_plucker_coordinates = revolute_plucker_coordinates.float()
            prismatic_directions = F.normalize(
                prismatic_directions_raw.float(),
                dim=-1,
                eps=1e-6,
            )

        joint_type_logits = self.joint_type_head(part_queries)
        observed_articulation_spans = F.softplus(
            self.observed_articulation_span_head(part_queries)
        ).squeeze(-1)

        return {
            "segmentation_logits": segmentation_logits,
            "revolute_plucker_coordinates": revolute_plucker_coordinates,
            "prismatic_directions": prismatic_directions,
            "joint_types": joint_type_logits.argmax(dim=-1),
            "joint_type_logits": joint_type_logits,
            "observed_articulation_spans": observed_articulation_spans,
        }
