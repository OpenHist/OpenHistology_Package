"""Width/device fixes retaining H2's native RAConv and IHPool hierarchy."""

from .loader import load_native


def module():
    common = "from openhisto.mil.models.code_bases.compat.graph_ops import "
    patches = {
        "code/main.py": [
            (
                "os.environ['CUDA_VISIBLE_DEVICES'] = '1'",
                "# Adapter does not mutate CUDA visibility",
            ),
            ("import matplotlib.pyplot as plt", "# Plotting is not imported by the model adapter"),
            (
                "self.norm = LayerNorm(in_feats)",
                "self.input_norm = LayerNorm(in_feats)\n        self.norm = LayerNorm(out_classes)\n        self.head_norm = LayerNorm(out_classes//2)",
            ),
            (
                "x = self.norm(x)\n        \n        x = self.conv1",
                "x = self.input_norm(x)\n        \n        x = self.conv1",
            ),
            (
                "x = self.lin1(x)\n        x = self.relu(x)  \n        x = self.norm(x)",
                "x = self.lin1(x)\n        x = self.relu(x)  \n        x = self.head_norm(x)",
            ),
        ],
        "code/RAConv.py": [
            ("from torch_scatter import scatter", common + "scatter"),
            ("from torch_sparse import SparseTensor, set_diag", common + "SparseTensor, set_diag"),
            (
                "H, C = self.heads, self.out_channels",
                "device = x.device\n        H, C = self.heads, self.out_channels",
            ),
            ("size_i = torch.max(soft_index)+1", "size_i = int(torch.max(soft_index).item())+1"),
        ],
        "code/IHPool.py": [
            ("from torch_sparse import spspmm", "# Unused sparse multiplication import omitted"),
            ("from torch_scatter import scatter", common + "scatter"),
            ("from torch_sparse import SparseTensor", common + "SparseTensor"),
            (
                "from torch_geometric.nn.pool.topk_pool import topk",
                "# Unused old private topk import omitted",
            ),
            ("N = x.size(0)", "device = x.device\n        N = x.size(0)"),
        ],
    }
    result = load_native("h2_mil", "main", roots=("code",), replacements=patches)
    from torch_geometric.nn import MessagePassing

    conv = load_native("h2_mil", "RAConv")
    conv.RAConv.propagate = MessagePassing.propagate
    return result
