"""Point-cloud token embeddings used by FAMOS."""

import torch
import torch.nn as nn


class SinusoidalEncoding:
    """Embed 3D vectors as [x, sin(2^k x), cos(2^k x)] for k = 0, ..., num_freqs - 1."""

    def __init__(self, num_freqs: int, input_dims: int = 3):
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs)
        self.out_dim = input_dims * (1 + 2 * num_freqs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [inputs]
        for frequency in self.freq_bands.to(inputs.device):
            outputs.append(torch.sin(inputs * frequency))
            outputs.append(torch.cos(inputs * frequency))
        return torch.cat(outputs, dim=-1)


class PointEncoder(nn.Module):
    """Fuse PartField features with point coordinates and normals."""

    def __init__(self, in_dim: int, embed_dim: int, num_freqs: int = 10):
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.encoding = SinusoidalEncoding(num_freqs)
        self.projection = nn.Linear(
            self.in_dim + 2 * self.encoding.out_dim, self.embed_dim
        )

    def forward(
        self,
        feat_dict: dict,
        data_dict: dict,
    ) -> torch.Tensor:
        point_clouds = data_dict["point_clouds"]
        surface_normals = data_dict["surface_normals"]
        partfield_features = feat_dict["partfield_features"]
        batch_size, num_states, num_points, _ = point_clouds.shape

        flat_points = point_clouds.reshape(batch_size * num_states, num_points, 3)
        flat_normals = surface_normals.reshape(batch_size * num_states, num_points, 3)
        flat_features = partfield_features.reshape(
            batch_size * num_states,
            num_points,
            self.in_dim,
        )
        embedded = torch.cat(
            [
                flat_features,
                self.encoding.embed(flat_points),
                self.encoding.embed(flat_normals),
            ],
            dim=-1,
        )
        return self.projection(embedded).reshape(
            batch_size,
            num_states,
            num_points,
            self.embed_dim,
        )
