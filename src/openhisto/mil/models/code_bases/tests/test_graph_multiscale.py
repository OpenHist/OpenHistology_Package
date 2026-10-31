"""Synthetic-only acceptance of real upstream graph/multiscale implementations."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from ..adapters.graph_multiscale import NAMES, build


def _features(*shape):
    return torch.randn(*shape, requires_grad=True)


def _graph(dim=8, n=8):
    source = torch.arange(n)
    target = (source + 1) % n
    edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    return Data(x=_features(n, dim), edge_index=edges, batch=torch.zeros(n, dtype=torch.long))


def make_case(name, root: Path):
    """Return factory options, native synthetic inputs and forward options."""
    torch.manual_seed(193)
    options = {"input_dim": 8, "num_classes": 3}
    forward = {}
    if name == "wikg":
        options.update(dim_hidden=16, topk=3, dropout=0.0)
        inputs = _features(8, 8)
    elif name == "resgat":
        options.update(hidden_dim=[16, 16, 8], heads=[2, 2, 2], dropout=0.0)
        inputs = _graph()
    elif name == "patch_gcn":
        options.update(hidden_dim=8, num_layers=3, dropout=0.0)
        inputs = _graph()
    elif name == "slidegraph_plus":
        options.update(layers=[16, 16, 8], dropout=0.0, conv="EdgeConv", pooling="mean", aggr="max")
        inputs = _graph()
    elif name == "tad_graph":
        options.update(initial_dim=8, num_layers=1, attention_head_num=1)
        inputs = _graph()
        # Native edge descriptors are normalized distance and angle, not WSI metadata.
        inputs.edge_attr = torch.full((inputs.edge_index.shape[1], 2), 0.2)
    elif name == "cs_mil":
        options.update(cluster_num=2, num_classes=1)
        inputs = {
            "clusters": [_features(3, 8, 3, 1), _features(5, 8, 3, 1)],
            "mask": torch.ones(1, 8),
        }
    elif name == "zoommil":
        options.update(hidden_feat_dim=8, out_feat_dim=16, k_sample=2)
        inputs = {"levels": tuple(_features(n, 8) for n in (3, 12, 48))}
    elif name == "hag_mil":
        options.update(hidden_dims=[16, 16], select_k=2, dropout=0.0, k_sample=2)
        inputs = {"levels": tuple(_features(n, 8) for n in (3, 12, 48))}
    elif name == "gfr_mil":
        options.update(hidden_dim=32, dropout=0.0)
        inputs = {
            "features": _features(4, 8),
            "high_features": _features(16, 8),
            "coords": torch.stack((torch.arange(4) * 3584, torch.zeros(4)), 1),
            "high_coords": torch.stack((torch.arange(16) * 448, torch.zeros(16)), 1),
            "region_labels": torch.arange(4).repeat_interleave(4),
        }
    elif name == "h2_mil":
        options.update(hidden_dim=8, pool1_ratio=0.5, pool2_ratio=2, drop_out_ratio=0.0)
        inputs = _graph(n=13)
        inputs.edge_index_tree_8nb = inputs.edge_index
        inputs.node_type = torch.tensor([0] + [1] * 4 + [2] * 8)
        inputs.node_tree = torch.tensor([-1] + [0] * 4 + [1, 1, 2, 2, 3, 3, 4, 4])
        children = torch.arange(1, 13)
        parents = inputs.node_tree[children]
        hierarchy_edges = torch.stack((parents, children))
        inputs.edge_index_tree_8nb = torch.cat(
            (inputs.edge_index_tree_8nb, hierarchy_edges, hierarchy_edges.flip(0)), dim=1
        )
        inputs.x_y_index = torch.rand(13, 2)
    elif name == "hipt":
        options.update(input_dim=384, variant="lgp")
        inputs = {"features": _features(2, 256, 384)}
    else:
        raise ValueError(name)
    return options, inputs, forward


class NativeGraphAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_supported_edgeless_graphs_preserve_outputs_and_gradients(self):
        for name in ("resgat", "patch_gcn"):
            options, _, _ = make_case(name, Path("."))
            model = build(name, **options).eval()
            for count in (1, 4):
                with self.subTest(profile=name, nodes=count):
                    graph = Data(
                        x=_features(count, options["input_dim"]),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                    )
                    native_graph = graph.clone()
                    native_graph.batch = torch.zeros(count, dtype=torch.long)
                    expected = (
                        model.native(x_path=native_graph)
                        if name == "patch_gcn"
                        else model.native(native_graph)
                    )
                    output = model(graph)
                    actual = output.aux["hazards"] if name == "patch_gcn" else output.logits
                    torch.testing.assert_close(actual, expected[0])
                    self.assertTrue(torch.isfinite(output.logits).all())
                    output.logits.sum().backward()
                    self.assertIsNotNone(graph.x.grad)
                    self.assertTrue(torch.isfinite(graph.x.grad).all())
            for edges in (torch.empty((3, 0), dtype=torch.long), torch.empty((2, 0))):
                with self.assertRaisesRegex(ValueError, "long tensor"):
                    model(Data(x=torch.randn(1, options["input_dim"]), edge_index=edges))

    def test_all_profiles_cpu_forward_backward_state(self):
        for name in NAMES:
            with self.subTest(name=name):
                options, inputs, forward = make_case(name, Path("."))
                model = build(name, **options).cpu().train()
                result = model(inputs, **forward)
                self.assertIsNotNone(result.native_output)
                self.assertEqual(result.logits.shape, (1, options["num_classes"]))
                self.assertTrue(torch.isfinite(result.logits).all())
                # Include all native HAG stage heads: hard hierarchy selection
                # intentionally does not differentiate through parent indices.
                loss = result.logits.square().mean()
                if name == "hag_mil":
                    loss = loss + sum(
                        stage[0].square().mean() for stage in result.aux["stages"][:2]
                    )
                if name == "tad_graph":
                    loss = (
                        loss
                        + result.aux["info_loss"]
                        + result.aux["trivial_logits"].square().mean()
                    )
                if name == "gfr_mil":
                    loss = loss + result.aux["aux_loss"]
                loss.backward()
                grads = [
                    parameter.grad for parameter in model.parameters() if parameter.grad is not None
                ]
                self.assertTrue(grads, name)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in grads), name)
                self.assertTrue(any(grad.abs().sum() > 0 for grad in grads), name)
                restored = build(name, **options)
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                # Exercise native inference selection branches as well.
                model.eval()
                with torch.no_grad():
                    prediction = model(inputs, **forward)
                self.assertTrue(torch.isfinite(prediction.logits).all())

    def test_hipt_gp_none_and_checkpoints(self):
        for variant, dim, shape in [("gp", 192, (4, 192)), ("none", 8, (3, 5, 8))]:
            with self.subTest(variant=variant):
                model = build("hipt", input_dim=dim, num_classes=3, variant=variant)
                features = _features(*shape)
                output = model(features)
                output.logits.square().sum().backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all())
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "synthetic.pt"
                    torch.save(model.native.state_dict(), checkpoint)
                    restored = build(
                        "hipt", input_dim=dim, num_classes=3, variant=variant, checkpoint=checkpoint
                    )
                    self.assertEqual(restored.metadata["weights"], "provided_checkpoint")
        with self.assertRaises(FileNotFoundError):
            build("hipt", input_dim=192, variant="gp", checkpoint="/missing/synthetic_weights.pt")

    def test_required_native_inputs_fail_clearly(self):
        for name in (
            "resgat",
            "patch_gcn",
            "slidegraph_plus",
            "h2_mil",
            "tad_graph",
            "gfr_mil",
            "cs_mil",
            "zoommil",
            "hag_mil",
        ):
            with self.subTest(name=name):
                options, _, _ = make_case(name, Path("."))
                with self.assertRaises((ValueError, TypeError)):
                    build(name, **options)(_features(8, 8))
        with self.assertRaisesRegex(ValueError, "raster-ordered"):
            build("hipt", variant="lgp")(_features(1, 32, 384))

    def test_tad_does_not_mutate_input_features(self):
        options, graph, _ = make_case("tad_graph", Path("."))
        before = graph.x.detach().clone()
        build("tad_graph", **options)(graph)
        torch.testing.assert_close(graph.x.detach(), before)

    def test_tad_multihead_edge_attention_and_eval(self):
        options, graph, _ = make_case("tad_graph", Path("."))
        options.update(attention_head_num=2, num_layers=2, learn_edge_att=True)
        model = build("tad_graph", **options)
        result = model(graph)
        (result.logits.square().sum() + result.aux["info_loss"]).backward()
        self.assertIsNotNone(graph.x.grad)
        model.eval()
        with torch.no_grad():
            first = model(graph).logits
            second = model(graph).logits
        torch.testing.assert_close(first, second)

    def test_tad_survival_semantics_and_backward(self):
        options, graph, _ = make_case("tad_graph", Path("."))
        model = build("tad_graph", **options, task="survival_prediction")
        result = model(graph)
        self.assertIsNone(result.probabilities)
        torch.testing.assert_close(result.aux["hazards"], result.logits.sigmoid())
        torch.testing.assert_close(
            result.aux["survival"], torch.cumprod(1 - result.aux["hazards"], dim=1)
        )
        torch.testing.assert_close(result.aux["risk"], -result.aux["survival"].sum(dim=1))
        torch.testing.assert_close(
            result.aux["trivial_hazards"], result.aux["trivial_logits"].sigmoid()
        )
        (result.aux["risk"].square().mean() + result.aux["info_loss"]).backward()
        self.assertIsNotNone(graph.x.grad)
        self.assertTrue(torch.isfinite(graph.x.grad).all())

    def test_graph_batch_labels_are_explicit_integer_zeros(self):
        options, graph, _ = make_case("resgat", Path("."))
        model = build("resgat", **options)
        for batch in (torch.full((len(graph.x),), 5, dtype=torch.long), torch.zeros(len(graph.x))):
            graph.batch = batch
            with self.assertRaisesRegex(ValueError, "long vector of zeros"):
                model(graph)

    def test_h2_unequal_widths_and_hierarchy(self):
        options, graph, _ = make_case("h2_mil", Path("."))
        options["hidden_dim"] = 12
        model = build("h2_mil", **options)
        result = model(graph)
        result.logits.square().sum().backward()
        self.assertIsNotNone(graph.x.grad)
        self.assertTrue(torch.isfinite(graph.x.grad).all())
        for types, tree, xy, edges in zip(
            result.aux["node_types"],
            result.aux["trees"],
            result.aux["coordinates"],
            result.aux["edges"],
        ):
            self.assertEqual(len(types), len(tree))
            self.assertEqual(xy.shape, (len(types), 2))
            self.assertLess(int(edges.max()), len(types))
            fine = types == 2
            self.assertTrue((types[tree[fine]] == 1).all())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "h2_synthetic.pt"
            torch.save(model.native.state_dict(), checkpoint)
            build("h2_mil", **options, checkpoint=checkpoint)
            state = model.native.state_dict()
            state.pop("head_norm.weight")
            torch.save(state, checkpoint)
            with self.assertRaises(RuntimeError):
                build("h2_mil", **options, checkpoint=checkpoint)

    def test_gfr_survival_retains_hazards(self):
        options, inputs, _ = make_case("gfr_mil", Path("."))
        model = build("gfr_mil", **options, task="survival")
        result = model(inputs)
        torch.testing.assert_close(result.logits.sigmoid(), result.aux["hazards"])
        torch.testing.assert_close(
            torch.cumprod(1 - result.aux["hazards"], dim=1), result.aux["survival"]
        )
        result.logits.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
