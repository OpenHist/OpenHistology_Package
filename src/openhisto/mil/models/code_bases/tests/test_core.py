import io
import json
import subprocess
import sys
import unittest

import torch

from openhisto._common import worker_environment
from openhisto.mil.models.code_bases import MILOutput, create_model, list_models, model_info
from openhisto.mil.models.code_bases.sources import ROOT, REPOSITORIES, _source_file
from openhisto.mil.models.code_bases.snapshot import validate_source_tree


class CoreTests(unittest.TestCase):
    def test_catalogue_is_complete_and_lazy(self):
        self.assertEqual(len(list_models()), 34)
        self.assertEqual(len(REPOSITORIES), 29)  # 28 origins; two MHIM revisions.
        code = "from openhisto.mil.models.code_bases import list_models; import sys; assert len(list_models())==34; assert 'torch' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code], check=True, timeout=20, env=worker_environment()
        )
        self.assertEqual(model_info("R²T-MIL")["name"], "rrt_mil")

    def test_all_pinned_sources_exist_and_are_code_only(self):
        for slug in REPOSITORIES:
            with self.subTest(source=slug):
                record = json.loads((ROOT / "compat" / "sources" / (slug + ".json")).read_text())
                path = ROOT / record["checkout"]
                self.assertEqual(validate_source_tree(ROOT, slug, record), record["commit"])
                for file in path.rglob("*"):
                    if ".git" in file.parts or "__pycache__" in file.parts or not file.is_file():
                        continue
                    self.assertTrue(
                        _source_file(str(file.relative_to(path))), "Non-source artifact checked out"
                    )

    def test_pooling_matches_exact_statistics_and_preserves_gradients(self):
        values = torch.tensor([[1.0, 7.0, 3.0], [5.0, 1.0, 9.0]], requires_grad=True)
        expected = {
            "mean_mil": [[3.0, 4.0, 6.0]],
            "max_mil": [[5.0, 7.0, 9.0]],
            "mean_std_mil": [[3.0, 4.0, 6.0, 2.0, 3.0, 3.0]],
        }
        for name in expected:
            model = create_model(name, input_dim=3, num_classes=2)
            output = model(values)
            self.assertIsInstance(output, MILOutput)
            torch.testing.assert_close(output.embedding, torch.tensor(expected[name]))
            output.logits.square().mean().backward()
            self.assertTrue(torch.isfinite(values.grad).all())
            state = io.BytesIO()
            torch.save(model.state_dict(), state)
            state.seek(0)
            other = create_model(name, input_dim=3, num_classes=2)
            other.load_state_dict(torch.load(state, weights_only=True), strict=True)
            torch.testing.assert_close(other(values).logits, output.logits)

    def test_single_patch_std_and_invalid_inputs(self):
        model = create_model("MeanStdMIL", input_dim=3)
        output = model(torch.ones(1, 3, requires_grad=True))
        torch.testing.assert_close(output.embedding[:, 3:], torch.zeros(1, 3))
        output.logits.sum().backward()
        for values in [
            torch.empty(0, 3),
            torch.ones(2, 4),
            torch.ones(2, 2, 3),
            torch.full((2, 3), float("nan")),
        ]:
            with self.assertRaises(ValueError):
                model(values)
        for name, value in [("input_dim", 0), ("num_classes", True)]:
            with self.assertRaises(ValueError):
                create_model("mean_mil", **{name: value})
        with self.assertRaisesRegex(ValueError, 'device="cpu"'):
            create_model("mean_mil", device="cuda")

    def test_data_and_lfs_payloads_are_not_selected(self):
        for value in [
            "Data/slides.svs",
            "data/clinical.csv",
            "features/slide.h5",
            "models/weights.pt",
            "splits/fold.json",
            "dataset_csv/slide.json",
        ]:
            self.assertFalse(_source_file(value))
        for value in [
            "datasets/dataset.py",
            "src/configs/default/config.json",
            "LICENSE",
            "requirements.txt",
        ]:
            self.assertTrue(_source_file(value))


if __name__ == "__main__":
    unittest.main()
