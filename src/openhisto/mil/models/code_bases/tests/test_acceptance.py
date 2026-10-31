"""Every public profile must pass the same real CPU execution contract."""

import importlib
import io
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

from openhisto.mil.models.code_bases import MILOutput, create_model, list_models


def _seed():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


def _prediction(output):
    for value in (output.logits, output.probabilities, output.embedding):
        if isinstance(value, torch.Tensor):
            return value
    raise AssertionError("Adapter did not expose a prediction or representation")


class AcceptanceTests(unittest.TestCase):
    verified_profiles = []

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_public_profiles_cpu_backward_and_state_roundtrip(self):
        verified = []
        with tempfile.TemporaryDirectory(prefix="cpath-mil-acceptance-") as temporary:
            for profile in list_models():
                name, family = profile["name"], profile["family"]
                with self.subTest(model=name):
                    _seed()
                    if family == "baselines":
                        options, inputs, kwargs = (
                            {"input_dim": 8, "num_classes": 2},
                            torch.randn(5, 8),
                            {},
                        )
                    else:
                        tests = importlib.import_module(
                            "openhisto.mil.models.code_bases.tests.test_"
                            + ("graph_multiscale" if family == "graph_multiscale" else family)
                        )
                        options, inputs, kwargs = tests.make_case(name, Path(temporary))
                    before_path, before_cuda = (
                        list(sys.path),
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                    )
                    model = create_model(name, **options)
                    # PyTorch itself registers a temporary distributed-JIT
                    # directory on first import. It is not a repository alias.
                    jit = sys.modules.get("torch.distributed.nn.jit.instantiator")
                    jit_path = getattr(jit, "INSTANTIATED_TEMPLATE_DIR_PATH", None)
                    self.assertEqual(
                        [p for p in sys.path if p != jit_path],
                        [p for p in before_path if p != jit_path],
                        "Native imports changed sys.path",
                    )
                    self.assertEqual(
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                        before_cuda,
                        "Native imports changed CUDA routing",
                    )
                    self.assertTrue(
                        all(parameter.device.type == "cpu" for parameter in model.parameters())
                    )
                    model.train()
                    output = model(inputs, **kwargs)
                    self.assertIsInstance(output, MILOutput)
                    prediction = _prediction(output)
                    self.assertEqual(prediction.device.type, "cpu")
                    self.assertTrue(torch.isfinite(prediction).all())
                    loss = prediction.float().square().mean()
                    loss.backward()
                    grads = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    self.assertTrue(grads, "No native trainable parameter received gradients")
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in grads))
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    clone = create_model(name, **options)
                    clone.load_state_dict(
                        torch.load(buffer, map_location="cpu", weights_only=True), strict=True
                    )
                    model.eval()
                    clone.eval()
                    with torch.no_grad():
                        _seed()
                        original = _prediction(model(inputs, **kwargs))
                        _seed()
                        restored = _prediction(clone(inputs, **kwargs))
                    torch.testing.assert_close(original, restored, rtol=1e-4, atol=1e-5)
                    verified.append(name)
        self.assertEqual(len(verified), 34)
        type(self).verified_profiles = verified


if __name__ == "__main__":
    unittest.main()
