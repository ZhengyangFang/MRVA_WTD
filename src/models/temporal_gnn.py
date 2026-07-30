from __future__ import annotations

import torch
from torch import nn

from ..aem_encoder import MLPAEMEncoder
from .layers import (
    ForcingSummaryEncoder,
    FutureForcingSequenceEncoder,
    HorizonEncoder,
    MLPHead,
    TopographyEncoder,
    WeightedGCNStack,
)


class TemporalGNNModel(nn.Module):
    def __init__(
        self,
        aem_profile_dim: int,
        topography_dim: int,
        config: dict,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        features_cfg = config["features"]

        explicit_seq_cols = features_cfg.get("forcing_sequence_columns")
        if explicit_seq_cols is not None:
            if not isinstance(explicit_seq_cols, (list, tuple)):
                raise TypeError("features.forcing_sequence_columns must be a list/tuple.")
            seq_input_dim = len([str(v).strip() for v in explicit_seq_cols if str(v).strip()])
            if seq_input_dim <= 0:
                raise ValueError("features.forcing_sequence_columns must contain at least one valid column.")
        else:
            seq_input_dim = 2

        hidden_dim = int(model_cfg["hidden_dim"])
        static_embed_dim = int(model_cfg["static_embed_dim"])
        topography_embed_dim = int(model_cfg.get("topography_embed_dim", 8))
        horizon_embed_dim = int(model_cfg["horizon_embed_dim"])
        forcing_embed_dim = int(model_cfg["forcing_embed_dim"])
        dropout = float(model_cfg["dropout"])
        forcing_sequence_encoder = str(model_cfg.get("forcing_sequence_encoder", "gru"))
        forcing_sequence_d_model = int(model_cfg.get("forcing_sequence_d_model", 32))
        forcing_sequence_layers = int(model_cfg.get("forcing_sequence_num_layers", 1))
        forcing_sequence_heads = int(model_cfg.get("forcing_sequence_num_heads", 4))
        forcing_sequence_max_len = int(model_cfg.get("forcing_sequence_max_len", 64))

        self.use_aem_profile = bool(features_cfg.get("use_aem_profile", True))
        self.use_topography = bool(features_cfg.get("use_topography", False)) and int(topography_dim) > 0
        self.topography_mode = str(model_cfg.get("topography_mode", "encoded")).lower()
        self.use_horizon_forcing_summary = bool(features_cfg.get("use_horizon_forcing_summary", True))
        self.use_horizon_forcing_sequence = bool(features_cfg.get("use_horizon_forcing_sequence", False))
        self.use_gnn_residual = bool(model_cfg.get("use_gnn_residual", False))

        self.aem_encoder = MLPAEMEncoder(
            input_dim=aem_profile_dim,
            embed_dim=static_embed_dim,
            dropout=dropout,
        )
        if self.use_topography and self.topography_mode == "encoded":
            self.topography_encoder = TopographyEncoder(
                input_dim=topography_dim,
                embed_dim=topography_embed_dim,
            )
            topography_out_dim = topography_embed_dim
        elif self.use_topography and self.topography_mode == "raw":
            self.topography_encoder = None
            topography_out_dim = int(topography_dim)
        elif self.use_topography:
            raise ValueError(f"Unsupported model.topography_mode: {self.topography_mode}")
        else:
            self.topography_encoder = None
            topography_out_dim = 0
        self.horizon_encoder = HorizonEncoder(input_dim=3, embed_dim=horizon_embed_dim)
        self.forcing_encoder = ForcingSummaryEncoder(input_dim=2, embed_dim=forcing_embed_dim)
        self.forcing_sequence_encoder = FutureForcingSequenceEncoder(
            input_dim=seq_input_dim,
            embed_dim=forcing_embed_dim,
            kind=forcing_sequence_encoder,
            d_model=forcing_sequence_d_model,
            num_layers=forcing_sequence_layers,
            num_heads=forcing_sequence_heads,
            dropout=dropout,
            max_len=forcing_sequence_max_len,
        )

        node_input_dim = horizon_embed_dim
        if self.use_aem_profile:
            node_input_dim += static_embed_dim
        if self.use_topography:
            node_input_dim += topography_out_dim
        if self.use_horizon_forcing_summary:
            node_input_dim += forcing_embed_dim
        if self.use_horizon_forcing_sequence:
            node_input_dim += forcing_embed_dim

        self.gnn = WeightedGCNStack(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg["gnn_layers"]),
            dropout=dropout,
            use_residual=self.use_gnn_residual,
        )
        self.head = MLPHead(input_dim=hidden_dim + 1, hidden_dim=64, dropout=dropout)

    def forward(self, data):
        parts = [self.horizon_encoder(data.horizon_features)]
        if self.use_aem_profile:
            parts.append(self.aem_encoder(data.aem_profile))
        if self.use_topography:
            if self.topography_mode == "encoded":
                parts.append(self.topography_encoder(data.topography_features))
            else:
                parts.append(data.topography_features)
        if self.use_horizon_forcing_summary:
            parts.append(self.forcing_encoder(data.forcing_summary))
        if self.use_horizon_forcing_sequence:
            parts.append(self.forcing_sequence_encoder(data.forcing_sequence, data.forcing_sequence_length))
        node_features = torch.cat(parts, dim=-1)
        x = self.gnn(node_features, data.edge_index, getattr(data, "edge_weight", None))
        root_x = x[data.root_mask]
        root_wtd = data.root_wtd.view(-1, 1)
        out = self.head(torch.cat([root_x, root_wtd], dim=-1))
        return out.view(-1)
