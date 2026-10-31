"""
ResGAT: Residual Graph Attention Network for WSI classification.

Architecture:
    3 cascaded ResidualBlockGAT layers (GATv2 + linear skip + GraphNorm),
    followed by global mean pooling and a 2-layer MLP classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, SAGPooling, global_mean_pool


class ResidualBlockGAT(nn.Module):
    """GATv2 convolution with a parallel linear skip branch and GraphNorm.

    The skip branch projects the *block input* (pre-convolution) to the
    output dimension via a linear layer + GraphNorm.  Both paths are
    independently normalised before summation, which stabilises training
    when input and output dimensions differ.

    Args:
        in_dim:  Input feature dimension.
        out_dim: Output feature dimension (must be divisible by *heads*).
        heads:   Number of attention heads.
        dropout: Dropout probability inside GATv2Conv.
    """

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.3):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_dim // heads, heads=heads,
                              concat=True, dropout=dropout)
        self.bn = GraphNorm(out_dim)

        self.skip = nn.Linear(in_dim, out_dim, bias=True)
        self.skip_norm = GraphNorm(out_dim)

        self._pool = SAGPooling(out_dim, ratio=0.9)

    def forward(self, x, edge_index, batch):
        x_main = self.conv(x, edge_index)
        x_main = self.bn(x_main, batch)

        x_skip = self.skip_norm(self.skip(x), batch)

        return F.elu(x_main + x_skip)


class ResGATs(nn.Module):
    """Residual Graph Attention Network.

    Three ``ResidualBlockGAT`` layers progressively reduce the hidden
    dimension (``hidden_dim = [1024, 512, 256]`` by default).  A global
    mean-pool readout feeds into a two-layer MLP classifier.

    Args:
        input_dim: Dimension of input node features.
        hidden_dim: List of three hidden dimensions for the three blocks.
        heads: List of three head counts for each GATv2 layer.
        dropout: Dropout probability for GATv2Conv.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], heads=[8, 2, 4],
                 dropout=0.3, n_classes: int = 2):
        super().__init__()

        h0, h1, h2 = hidden_dim

        self.block1 = ResidualBlockGAT(input_dim, h0, heads=heads[0],
                                       dropout=dropout)
        self.block2 = ResidualBlockGAT(h0, h1, heads=heads[1],
                                       dropout=dropout)
        self.block3 = ResidualBlockGAT(h1, h2, heads=heads[2],
                                       dropout=dropout)

        self.fc1 = nn.Linear(h2, 512)
        self.drop = nn.Dropout(0.13)
        self.fc2 = nn.Linear(512, int(n_classes))

        self._pool = SAGPooling(h0, ratio=0.9)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.block1(x, edge_index, batch)
        x = self.block2(x, edge_index, batch)
        x = self.block3(x, edge_index, batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)

        return logits, F.log_softmax(logits, dim=1)
