"""Portable operations used by the pinned graph implementations.

These implement representation/reduction operations, not replacement networks.
No optional extension modules are installed into global import namespaces.
"""

import inspect
from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.utils import scatter as pyg_scatter


def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
    if out is not None:
        raise ValueError("The portable scatter path does not support an out buffer")
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


class COOAdj:
    """Small adjacency representation for TAD's native coo()/t() interface.

    Row order is retained on transpose so edge attributes stay aligned.
    Caller edge topology is never inferred from features or coordinates.
    """

    def __init__(self, row, col, size):
        self.row, self.col, self.size = row, col, int(size)

    def coo(self):
        return self.row, self.col, None

    def t(self):
        return type(self)(self.col, self.row, self.size)


def set_diag(adj):
    if not isinstance(adj, COOAdj):
        raise TypeError("Expected the adapter COO adjacency representation")
    keep = adj.row != adj.col
    diagonal = torch.arange(adj.size, device=adj.row.device)
    row = torch.cat((adj.row[keep], diagonal))
    col = torch.cat((adj.col[keep], diagonal))
    order = (row * adj.size + col).argsort(stable=True)
    return COOAdj(row[order], col[order], adj.size)


def transpose(edge_index, values, m=None, n=None, coalesced=False):
    # reorder_like in the native model explicitly realigns this output.
    return edge_index.flip(0), values


class SparseTensor:
    """Sentinel for unavailable optional SparseTensor branches.

    The adapters use native PyG COO edge_index tensors. SparseTensor inputs
    must be converted to COO by their caller, rather than silently densified.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("Pass a prepared PyG graph with COO edge_index tensors")


class NativeForwardBase(nn.Module):
    """TAD forward-only replacement for its training-framework base class.

    All native graph modules, samplers and losses remain in upstream code.
    Trainer, optimizer, dataloader and logging integrations are out of scope.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(**kwargs)
        self.current_epoch = 0
        self.logged_metrics = {}

    def save_hyperparameters(self):
        values = inspect.currentframe().f_back.f_locals
        for key, value in values.items():
            if key not in {"self", "__class__", "args", "kwargs"}:
                setattr(self.hparams, key, value)
        for key, value in values.get("kwargs", {}).items():
            setattr(self.hparams, key, value)

    def log(self, name, value, **kwargs):
        self.logged_metrics[name] = value
