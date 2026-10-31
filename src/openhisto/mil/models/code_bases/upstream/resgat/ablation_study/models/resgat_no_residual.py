"""
ResGAT with the residual/skip path ablated (structure ablation).

Each block is simply: GATv2Conv -> GraphNorm -> ELU (no skip branch).
This isolates the contribution of the residual connection.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool


class _PlainBlockGAT(nn.Module):
    """GATv2 block without a residual skip branch.

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

    def forward(self, x, edge_index, batch):
        x_main = self.conv(x, edge_index)
        return F.elu(self.bn(x_main, batch))


class ResGATsNoResidual(nn.Module):
    """Same topology as ResGATs but without the skip/residual path.

    Args:
        input_dim: Dimension of input node features.
        hidden_dim: List of three hidden dimensions for the three blocks.
        heads: List of three head counts for each GATv2 layer.
        dropout: Dropout probability for GATv2Conv.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim, hidden_dim=None, heads=None,
                 dropout=0.3, n_classes=2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [1024, 512, 256]
        if heads is None:
            heads = [8, 2, 4]
        h0, h1, h2 = hidden_dim

        self.block1 = _PlainBlockGAT(input_dim, h0, heads=heads[0],
                                     dropout=dropout)
        self.block2 = _PlainBlockGAT(h0, h1, heads=heads[1],
                                     dropout=dropout)
        self.block3 = _PlainBlockGAT(h1, h2, heads=heads[2],
                                     dropout=dropout)

        self.fc1 = nn.Linear(h2, 512)
        self.drop = nn.Dropout(0.13)
        self.fc2 = nn.Linear(512, int(n_classes))

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
