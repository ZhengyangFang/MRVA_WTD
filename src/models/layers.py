from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import GCNConv


class HorizonEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, embed_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForcingSummaryEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, embed_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopographyEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, embed_dim: int = 8) -> None:
        super().__init__()
        hidden_dim = max(8, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class FutureForcingSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        embed_dim: int = 16,
        kind: str = "gru",
        d_model: int = 32,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.kind = str(kind).lower()
        if self.kind == "gru":
            self.gru = nn.GRU(
                input_dim,
                embed_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=(dropout if num_layers > 1 else 0.0),
            )
            self.input_proj = None
            self.positional_encoding = None
            self.encoder = None
            self.output_proj = None
        elif self.kind == "transformer":
            self.gru = None
            self.input_proj = nn.Linear(input_dim, d_model)
            self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=max(64, d_model * 4),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, embed_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unsupported future forcing sequence encoder: {kind}")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.clamp(lengths.view(-1), min=1)
        if self.kind == "gru":
            packed = pack_padded_sequence(
                x,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            return hidden[-1]

        x = self.input_proj(x)
        x = self.positional_encoding(x)
        time_index = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        padding_mask = time_index >= lengths.unsqueeze(1)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return self.output_proj(pooled)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WeightedGCNStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        residual_proj = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(GCNConv(dims[i], dims[i + 1], add_self_loops=True, normalize=True))
            if use_residual:
                if dims[i] == dims[i + 1]:
                    residual_proj.append(nn.Identity())
                else:
                    residual_proj.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        self.layers = nn.ModuleList(layers)
        self.use_residual = bool(use_residual)
        self.residual_proj = nn.ModuleList(residual_proj)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
        for i, conv in enumerate(self.layers):
            residual = self.residual_proj[i](x) if self.use_residual else None
            x = conv(x, edge_index, edge_weight=edge_weight)
            if residual is not None:
                x = x + residual
            x = self.activation(x)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x
