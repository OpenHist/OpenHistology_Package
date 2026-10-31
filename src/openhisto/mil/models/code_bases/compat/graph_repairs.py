"""Exact, fail-closed source patches for graph/multiscale repositories."""

from .loader import load_native


def wikg():
    return load_native(
        "wikg",
        "model",
        replacements={
            "model.py": [
                (
                    "torch.autograd.set_detect_anomaly(True)",
                    "# Adapter leaves global anomaly detection unchanged",
                ),
            ]
        },
    )


def patch_gcn():
    return load_native(
        "patch_gcn",
        "models.model_graph_mil",
        replacements={
            "models/model_graph_mil.py": [
                ("nn.Linear(1024, 256)", "nn.Linear(num_features, hidden_dim)"),
                ("nn.Linear(1024, 128)", "nn.Linear(num_features, hidden_dim)"),
                ("hidden_dim*4", "hidden_dim*num_layers"),
                ("data.edge_attr.type(torch.cuda.FloatTensor)", "data.edge_attr.float()"),
            ]
        },
    )


def slidegraph():
    return load_native(
        "slidegraph",
        "GNN_pr",
        replacements={
            "utils.py": [
                (
                    "import matplotlib.pyplot as plt",
                    "# Plotting is not imported by the model adapter",
                )
            ],
        },
    )


def zoommil():
    return load_native(
        "zoommil",
        "zoommil.models.zoommil",
        replacements={
            "zoommil/models/zoommil.py": [
                ("device=self.device, requires_grad=True", "device=x1.device, dtype=x1.dtype"),
            ],
            "zoommil/utils/perturbedtopk.py": [
                ("(grad_input,) + tuple([None] * 5)", "(grad_input,) + tuple([None] * 3)"),
                ("tuple([None] * 5)", "tuple([None] * 4)"),
            ],
        },
    )


def cs_mil():
    path = "CS-MIL_Docker/src/DeepAttnMISL_CS_MIL.py"
    return load_native(
        "cs_mil",
        "DeepAttnMISL_CS_MIL",
        roots=("CS-MIL_Docker/src",),
        replacements={
            path: [
                (
                    'x[i].type(torch.FloatTensor).to("cuda")',
                    "x[i].to(device=self.embedding_net[0].weight.device, dtype=self.embedding_net[0].weight.dtype)",
                ),
                # Keep the native sigmoid output, expose configurable channel/head widths.
                (
                    "def __init__(self, cluster_num):",
                    "def __init__(self, cluster_num, input_dim=2048, num_classes=1):",
                ),
                ("nn.Conv2d(2048, 64, 1)", "nn.Conv2d(input_dim, 64, 1)"),
                (
                    "nn.Linear(32, 1),\n            nn.Sigmoid()",
                    "nn.Linear(32, num_classes),\n            nn.Sigmoid()",
                ),
                (
                    "x_exp = (x - x_max.unsqueeze(-1)).exp()",
                    "x_exp = (x_masked - x_max.unsqueeze(-1)).exp()",
                ),
                (
                    "return x_exp / x_exp.sum(1).unsqueeze(-1)",
                    "return x_exp / x_exp.sum(1).unsqueeze(-1).clamp_min(torch.finfo(x_exp.dtype).tiny)",
                ),
            ]
        },
    )


def tad_graph():
    prefix = "src/tadgraph/models/"
    common = "from openhisto.mil.models.code_bases.compat.graph_ops import "
    patches = {
        prefix + "tad_graph_module.py": [
            ("import ipdb", "# Optional debugger omitted"),
            (
                "from tadgraph.models.base_trainer import BaseLightningModule",
                common + "NativeForwardBase as BaseLightningModule",
            ),
            ("from torch_sparse import transpose", common + "transpose"),
            ("from tadgraph.paths import *", "# No dataset paths needed for model forward"),
            (
                "graph_data, node_atten=causal_atts[0], edge_atten=causal_edge_atts)",
                "graph_data, node_atten=None if self.hparams.learn_edge_att else causal_atts[0], edge_atten=causal_edge_atts)",
            ),
            (
                "graph_data, node_atten=1 - causal_atts[0], edge_atten=trivial_edge_atts)",
                "graph_data, node_atten=None if self.hparams.learn_edge_att else 1 - causal_atts[0], edge_atten=trivial_edge_atts)",
            ),
        ],
        prefix + "gnn.py": [
            ("import ipdb", "# Optional debugger omitted"),
            (
                "graph_data.x = graph_data.x * node_atten",
                "graph_data = graph_data.clone()\n            graph_data.x = graph_data.x * node_atten",
            ),
            (
                "out_channels=initial_dim,\n                    edge_dim=2*initial_dim)",
                "out_channels=initial_dim,\n                    heads=attention_head_num, edge_dim=2*initial_dim)",
            ),
            (
                "in_channels=initial_dim,\n                    out_channels=initial_dim,",
                "in_channels=attention_head_num * initial_dim,\n                    out_channels=initial_dim,",
            ),
            ("F.dropout(x, p=0.2)", "F.dropout(x, p=0.2, training=self.training)"),
            ("F.dropout(x_pool, p=0.2)", "F.dropout(x_pool, p=0.2, training=self.training)"),
        ],
        prefix + "utils.py": [
            ("import ipdb", "# Optional debugger omitted"),
            ("from torch_sparse import set_diag", common + "set_diag"),
            (
                "torch.tensor(drop_edge_attr_diag).float()",
                "torch.as_tensor(drop_edge_attr_diag, device=x.device, dtype=x.dtype)",
            ),
        ],
        prefix + "conv_layers.py": [
            ("import ipdb", "# Optional debugger omitted"),
            (
                "import torch_sparse",
                "from openhisto.mil.models.code_bases.compat import graph_ops as torch_sparse",
            ),
            ("from torch_sparse import SparseTensor, set_diag", common + "SparseTensor, set_diag"),
            ("from torch_scatter import scatter", common + "scatter"),
        ],
        prefix + "clam_model.py": [("import ipdb", "# Optional debugger omitted")],
    }
    module = load_native(
        "tad_graph", "tadgraph.models.tad_graph_module", roots=("src",), replacements=patches
    )
    # Modern PyG generates a base-specific propagate signature; the native
    # subclass has an additional edge_atten argument. Use its public dispatcher.
    from torch_geometric.nn import MessagePassing

    conv = load_native("tad_graph", "tadgraph.models.conv_layers")
    conv.GATv2Conv.propagate = MessagePassing.propagate
    return module
