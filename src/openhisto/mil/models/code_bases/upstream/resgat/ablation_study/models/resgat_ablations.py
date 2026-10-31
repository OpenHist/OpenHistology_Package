"""
ResGAT ablation variants: configurable convolution type, number of layers,
and normalisation layer.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GCNConv, GINConv, SAGEConv,
    GraphNorm, BatchNorm, InstanceNorm, global_mean_pool,
)
from torch_geometric.nn.norm import LayerNorm


def _build_conv(conv_type, in_dim, out_dim, heads, dropout):
    conv_type = conv_type.lower()
    if conv_type == "gatv2":
        return GATv2Conv(in_dim, out_dim // heads, heads=heads,
                         concat=True, dropout=dropout)
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim)
    if conv_type == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp)
    if conv_type == "sage":
        return SAGEConv(in_dim, out_dim)
    raise ValueError(f"Unknown conv_type '{conv_type}'. "
                     "Choose from: gatv2, gcn, gin, sage.")


def _build_norm(dim, norm_type):
    norm_type = norm_type.lower()
    if norm_type == "graph":
        return GraphNorm(dim)
    if norm_type == "batch":
        return BatchNorm(dim)
    if norm_type == "instance":
        return InstanceNorm(dim)
    if norm_type == "layer":
        return LayerNorm(dim)
    raise ValueError(f"Unknown norm_type '{norm_type}'. "
                     "Choose from: graph, batch, layer, instance.")


class _ResidualBlock(nn.Module):
    """Residual block with configurable conv and norm.

    The skip branch always takes the block *input* (pre-convolution)
    and projects it to the output dimension.

    Args:
        in_dim:    Input feature dimension.
        out_dim:   Output feature dimension (must be divisible by *heads*
                   when using GATv2).
        heads:     Number of attention heads (only used by GATv2).
        dropout:   Dropout probability inside the convolution layer.
        conv_type: ``'gatv2'`` | ``'gcn'`` | ``'gin'`` | ``'sage'``.
        norm_type: ``'graph'`` | ``'batch'`` | ``'layer'`` | ``'instance'``.
    """

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.3,
                 conv_type="gatv2", norm_type="graph"):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.conv = _build_conv(self.conv_type, in_dim, out_dim, heads, dropout)
        self.bn = _build_norm(out_dim, norm_type)

        self.skip = nn.Linear(in_dim, out_dim, bias=True)
        self.skip_norm = _build_norm(out_dim, norm_type)

    def forward(self, x, edge_index, batch):
        x_main = self.conv(x, edge_index)
        x_main = self.bn(x_main, batch)

        x_skip = self.skip_norm(self.skip(x), batch)

        return F.elu(x_main + x_skip)


class ResGATsAblation(nn.Module):
    """ResGAT backbone with configurable conv type, depth, and norm.

    Args:
        input_dim: Node feature dimension.
        hidden_dim: List ``[h0, h1, h2]`` of hidden dimensions.
        conv_type: ``'gatv2'`` | ``'gcn'`` | ``'gin'`` | ``'sage'``.
        num_layers: ``2`` | ``3`` | ``4``.
        norm_type: ``'graph'`` | ``'batch'`` | ``'layer'`` | ``'instance'``.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim, hidden_dim=None, first_layer_head=8,
                 heads=2, dropout=0.3, conv_type="gatv2",
                 num_layers=3, norm_type="graph", n_classes=2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256]
        h0, h1, h2 = hidden_dim
        assert num_layers in (2, 3, 4)
        self.num_layers = num_layers

        kw = dict(dropout=dropout, conv_type=conv_type, norm_type=norm_type)
        self.block1 = _ResidualBlock(input_dim, h0, heads=first_layer_head, **kw)
        self.block2 = _ResidualBlock(h0, h1, heads=heads, **kw)
        self.block3 = _ResidualBlock(h1, h2, heads=4, **kw)
        self.block4 = _ResidualBlock(h2, h2, heads=4, **kw)

        fc_in = h1 if num_layers == 2 else h2
        self.fc1 = nn.Linear(fc_in, 512)
        self.drop = nn.Dropout(0.13)
        self.fc2 = nn.Linear(512, int(n_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.block1(x, edge_index, batch)
        x = self.block2(x, edge_index, batch)
        if self.num_layers >= 3:
            x = self.block3(x, edge_index, batch)
        if self.num_layers >= 4:
            x = self.block4(x, edge_index, batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)
        return logits, F.log_softmax(logits, dim=1)
