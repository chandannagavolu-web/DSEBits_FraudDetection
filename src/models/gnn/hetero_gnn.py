"""Heterogeneous GNN for IEEE-CIS fraud node classification.

Operates on the ``HeteroData`` graph produced by
``src/graph/ieee_cis_graph.py`` (a ``transaction`` node type plus one node
type per shared entity — ``card1``, ``addr1``, ``P_emaildomain``,
``DeviceInfo``). Each node type is linearly projected to a common hidden
dimension, then ``num_layers`` rounds of heterogeneous message passing
(GraphSAGE or GAT convolutions per edge type) are applied before a binary
classification head on the ``transaction`` nodes.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear, SAGEConv

Metadata = Tuple[list, list]


class HeteroGNN(nn.Module):
    def __init__(
        self,
        metadata: Metadata,
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        model_type: str = "hetero_sage",
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout

        self.input_proj = nn.ModuleDict(
            {node_type: Linear(in_channels_dict[node_type], hidden_dim) for node_type in node_types}
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                if model_type == "hetero_sage":
                    conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
                elif model_type == "hetero_gat":
                    conv_dict[edge_type] = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
            # "mean" (not "sum") keeps activation scale comparable across node
            # types with different in-degrees -- transaction nodes receive 4
            # relation types (one per entity) while entity nodes receive 1,
            # so summing would make transaction activations grow ~4x per
            # layer relative to entity activations.
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.norms.append(
                nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
            )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        x_dict = {node_type: self.input_proj[node_type](x).relu() for node_type, x in x_dict.items()}

        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            # LayerNorm bounds activation magnitude per node type, preventing
            # the unbounded growth across layers that otherwise produces
            # exploding logits / BCE loss for high-degree entity nodes.
            x_dict = {key: norm[key](x) for key, x in x_dict.items()}
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()
            }

        logits = self.classifier(x_dict["transaction"]).squeeze(-1)
        return logits
