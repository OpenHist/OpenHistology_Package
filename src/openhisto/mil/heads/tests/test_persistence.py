import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from openhisto.mil.classification import create_neural_head
from openhisto.mil.heads import (
    save_head,
    load_head,
    attach_head,
    get_representation_spec,
    representation_dimension,
)
from openhisto.mil.heads.provenance import state_fingerprint
from openhisto.mil.models.code_bases import create_model


class PersistenceTests(unittest.TestCase):
    def test_semantic_attention_configuration_must_match_saved_head(self):
        cases = (
            ("varpool", {"n_var_pools": 2}, {"act_func": "sqrt"}, {"act_func": "log"}),
            (
                "dtfd_mil",
                {"hidden_dim": 8, "num_groups": 2, "distill": "MaxS"},
                {"instances_per_group": 1},
                {"instances_per_group": 3},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, common, first, second in cases:
                with self.subTest(profile=name):
                    torch.manual_seed(5)
                    a = create_model(name, input_dim=4, num_classes=2, **common, **first).eval()
                    b = create_model(name, input_dim=4, num_classes=2, **common, **second).eval()
                    b.load_state_dict(a.state_dict())
                    self.assertEqual(state_fingerprint(a), state_fingerprint(b))
                    self.assertNotEqual(get_representation_spec(a), get_representation_spec(b))
                    features = torch.randn(16, 4)
                    self.assertFalse(torch.allclose(a(features).embedding, b(features).embedding))
                    path = Path(directory) / (name + ".milhead")
                    head = create_neural_head("linear", representation_dimension(a), 2)
                    save_head(attach_head(a, head), path)
                    load_head(path, expected_backbone=a)
                    with self.assertRaisesRegex(ValueError, "representation/configuration"):
                        load_head(path, expected_backbone=b)

    def test_explicit_and_implicit_attention_defaults_have_the_same_specification(self):
        implicit = create_model("varpool", input_dim=4, n_var_pools=2)
        explicit = create_model(
            "varpool",
            input_dim=4,
            n_var_pools=2,
            act_func="sqrt",
            separate_attn=np.bool_(False),
            gated=np.bool_(True),
        )
        self.assertEqual(get_representation_spec(implicit), get_representation_spec(explicit))

    def test_checksum_unknown_members_and_size_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.milhead"
            save_head(create_neural_head("linear", 4, 2), path)
            with self.assertRaisesRegex(ValueError, "size limit"):
                load_head(path, max_uncompressed_bytes=1)
            for value in (True, 0, -1, "100"):
                with self.assertRaises(ValueError):
                    load_head(path, max_uncompressed_bytes=value)
            with zipfile.ZipFile(path) as archive:
                manifest, weights = archive.read("manifest.json"), archive.read("weights.pt")
            damaged = Path(directory) / "damaged.milhead"
            with zipfile.ZipFile(damaged, "w") as archive:
                archive.writestr("manifest.json", manifest)
                archive.writestr("weights.pt", weights + b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_head(damaged)
            with zipfile.ZipFile(damaged, "w") as archive:
                archive.writestr("manifest.json", manifest)
                archive.writestr("weights.pt", weights)
                archive.writestr("unregistered.py", "raise RuntimeError()")
            with self.assertRaisesRegex(ValueError, "members"):
                load_head(damaged)

    def test_manifest_constructor_allowlist_and_publication_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.milhead"
            head = create_neural_head("linear", 3, 2)
            save_head(head, path)
            with self.assertRaises(TypeError):
                save_head(head, path, overwrite="False")
            link = Path(directory) / "alias.milhead"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                save_head(head, link, overwrite=True)
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                weights = archive.read("weights.pt")
            manifest["config"]["family"] = "unregistered_python_module"
            bad = Path(directory) / "bad.milhead"
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                archive.writestr("weights.pt", weights)
            with self.assertRaisesRegex(ValueError, "Unregistered"):
                load_head(bad)
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr("manifest.json", "[]")
                archive.writestr("weights.pt", weights)
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_head(bad)


if __name__ == "__main__":
    unittest.main()
