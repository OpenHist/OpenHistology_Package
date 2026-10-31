"""Model/auth tests use synthetic adapters and metadata; no model downloads."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np

from openhisto.feature_encoder.auth import AuthConfig
from openhisto.feature_encoder.base import AccessError, EncoderError, ModelDescriptor
from openhisto.feature_encoder.config import EncodeConfig, EncoderSpec
from openhisto.feature_encoder.registry import (
    check_access,
    get_descriptor,
    list_models,
    model_info,
    prepare_model,
    register_encoder,
)
from openhisto.feature_encoder.locked.HF.local import (
    Conch15Encoder,
    TorchEncoder,
    UNIEncoder,
    device_and_precision,
    load_weights,
    snapshot,
)
from openhisto.feature_encoder.public.phikon import PhikonEncoder


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class AuthTests(unittest.TestCase):
    def test_explicit_and_environment_credentials_are_runtime_only(self):
        secret = "synthetic-private-value"
        auth = AuthConfig(hf_token=secret, gemini_api_key_env="TEST_GEMINI_KEY")
        with patch.dict(
            "os.environ",
            {
                "TEST_GEMINI_KEY": "synthetic-google-value",
                "COHERE_API_KEY": "synthetic-cohere-value",
            },
            clear=True,
        ):
            self.assertEqual(auth.resolve("hf"), secret)
            self.assertEqual(auth.resolve("google"), "synthetic-google-value")
            self.assertEqual(auth.resolve("gemini"), "synthetic-google-value")
            self.assertEqual(auth.resolve("cohere"), "synthetic-cohere-value")
        self.assertNotIn(secret, repr(auth))
        self.assertIn("<redacted>", repr(auth))
        self.assertIn("TEST_GEMINI_KEY", repr(auth))

    def test_hf_cache_lookup_and_no_implicit_prompt(self):
        module = SimpleNamespace(get_token=lambda: "synthetic-cached-value")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"huggingface_hub": module}),
        ):
            self.assertEqual(AuthConfig().resolve("hf"), "synthetic-cached-value")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("openhisto.feature_encoder.auth.getpass.getpass") as prompt,
        ):
            self.assertIsNone(AuthConfig(prompt=True).resolve("cohere", required=False))
            prompt.assert_not_called()
            with self.assertRaises(AccessError):
                AuthConfig().resolve("cohere")

    def test_opt_in_masked_prompt_is_cached_in_memory(self):
        auth = AuthConfig(
            prompt_credentials=True, aws_profile="synthetic-profile", aws_region="us-west-2"
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "openhisto.feature_encoder.auth.getpass.getpass",
                return_value="synthetic-prompt-value",
            ) as prompt,
        ):
            self.assertEqual(auth.resolve("google"), "synthetic-prompt-value")
            self.assertEqual(auth.resolve("google"), "synthetic-prompt-value")
            self.assertEqual(prompt.call_count, 1)
        self.assertEqual(auth.aws_profile, "synthetic-profile")
        self.assertEqual(auth.aws_region, "us-west-2")

    def test_invalid_environment_names_and_empty_explicit_secrets(self):
        with self.assertRaises(ValueError):
            AuthConfig(hf_token_env="not a variable")
        with self.assertRaises(ValueError):
            AuthConfig(hf_token="")


class RegistryTests(unittest.TestCase):
    def test_catalogue_is_lazy_and_contains_immutable_local_defaults(self):
        with patch(
            "openhisto.feature_encoder.registry.importlib.import_module",
            side_effect=AssertionError("No adapter imports"),
        ):
            profiles = {item["model_id"]: item for item in list_models()}
            self.assertEqual(profiles["phikon"]["dimension"], 768)
            self.assertEqual(profiles["phikon_v2"]["dimension"], 1024)
            self.assertEqual(profiles["conch15_titan"]["dimension"], 768)
            self.assertEqual(profiles["gemini_embedding_2"]["dimension"], 3072)
            self.assertEqual(
                profiles["gemini_embedding_2"]["cache_status"]["status"], "not_applicable"
            )
            for name in ("phikon", "phikon_v2", "uni_v1", "conch15_titan"):
                self.assertEqual(len(profiles[name]["default_revision"]), 40)
            json.dumps(profiles)

    def test_catalogue_reports_unverified_complete_and_partial_cache_without_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            descriptor = get_descriptor("phikon")
            cached = (
                Path(directory)
                / "models--owkin--phikon"
                / "snapshots"
                / descriptor.preprocessing["default_revision"]
            )
            cached.mkdir(parents=True)
            with (
                patch.dict("os.environ", {"HF_HUB_CACHE": directory}),
                patch(
                    "openhisto.feature_encoder.auth.getpass.getpass",
                    side_effect=AssertionError("No prompt"),
                ),
                patch(
                    "openhisto.feature_encoder.registry.importlib.import_module",
                    side_effect=AssertionError("No model imports"),
                ),
            ):
                self.assertEqual(model_info("phikon")["cache_status"]["status"], "not_cached")
                (cached / "config.json").write_text("{}")
                self.assertEqual(
                    model_info("phikon")["cache_status"]["status"], "partial_unverified"
                )
                (cached / "preprocessor_config.json").write_text("{}")
                (cached / "pytorch_model.bin").write_bytes(b"synthetic cached weights")
                status = model_info("phikon")["cache_status"]
                self.assertEqual(status["status"], "present_unverified")
                self.assertFalse(status["verified"])

    def test_custom_factory_contract_and_dimension_validation(self):
        name = "synthetic_" + uuid.uuid4().hex
        descriptor = ModelDescriptor(name, "Synthetic", "fixture", "public", 3, "synthetic-model")
        closed = []

        class Dummy:
            dimension = 3
            metadata = {"resolved_revision": "synthetic-revision", "dimension": 3}
            usage = {}

            def encode(self, images):
                return np.ones((len(images), 3), dtype=np.float32)

            def close(self):
                closed.append(True)

        factory = Mock(side_effect=lambda spec, *, config, auth: Dummy())
        register_encoder(descriptor, factory)
        encoder = prepare_model(name)
        self.assertEqual(encoder.encode([np.zeros((2, 2, 3), dtype=np.uint8)]).shape, (1, 3))
        self.assertIsInstance(factory.call_args.kwargs["auth"], AuthConfig)
        self.assertEqual(get_descriptor(name), descriptor)
        self.assertEqual(check_access(name)["status"], "custom_adapter_unverified")
        with self.assertRaises(ValueError):
            register_encoder(descriptor, factory)
        with self.assertRaises(ValueError):
            prepare_model(EncoderSpec(name, dimension=4))

    def test_secret_metadata_is_rejected_and_encoder_closed(self):
        name = "synthetic_" + uuid.uuid4().hex
        encoder = SimpleNamespace(dimension=2, metadata={"token": "synthetic-value"}, close=Mock())
        register_encoder(
            ModelDescriptor(name, "Synthetic", "fixture", "public", 2, "synthetic"),
            lambda *args, **kwargs: encoder,
        )
        with self.assertRaises(ValueError):
            prepare_model(name)
        encoder.close.assert_called_once()

    def test_cloud_presence_check_makes_no_inference_or_network_calls(self):
        auth = AuthConfig(
            gemini_api_key="synthetic-google-value",
            cohere_api_key="synthetic-cohere-value",
            aws_region="us-west-2",
        )
        with patch("requests.head", side_effect=AssertionError("No network")):
            result = check_access("gemini_embedding_2", auth=auth)
            self.assertTrue(result["available"])
            self.assertFalse(result["verified"])
            self.assertNotIn("synthetic-google-value", json.dumps(result))
            self.assertEqual(
                check_access("nova_multimodal", auth=auth)["status"],
                "sdk_credential_chain_unverified",
            )

    def test_gated_hf_check_uses_head_only_and_redacts_auth_errors(self):
        response = SimpleNamespace(status_code=302, close=Mock())
        auth = AuthConfig(hf_token="synthetic-private-value")
        with patch("requests.head", return_value=response) as head:
            result = check_access("uni_v1", auth=auth)
            self.assertEqual(result["status"], "granted")
            self.assertTrue(result["verified"])
            self.assertFalse(head.call_args.kwargs["allow_redirects"])
            self.assertEqual(
                head.call_args.kwargs["headers"]["Authorization"], "Bearer synthetic-private-value"
            )
            self.assertNotIn("synthetic-private-value", json.dumps(result))
            response.close.assert_called_once()
        with patch("requests.head", side_effect=RuntimeError("synthetic-private-value")):
            result = check_access("uni_v1", auth=auth)
            self.assertEqual(result["status"], "access_check_failed")
            self.assertNotIn("synthetic-private-value", json.dumps(result))

    def test_explicit_access_prompt_is_honored_for_gated_hf_and_cloud(self):
        response = SimpleNamespace(status_code=302, close=Mock())
        hub = SimpleNamespace(get_token=lambda: None)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"HF_HUB_CACHE": directory}, clear=True),
            patch.dict("sys.modules", {"huggingface_hub": hub}),
        ):
            with (
                patch(
                    "openhisto.feature_encoder.auth.getpass.getpass",
                    return_value="synthetic-prompt-value",
                ) as prompt,
                patch("requests.head", return_value=response) as head,
            ):
                result = check_access("uni_v1", auth=AuthConfig(prompt=True))
                self.assertEqual(result["status"], "granted")
                prompt.assert_called_once()
                head.assert_called_once()
                self.assertNotIn("synthetic-prompt-value", json.dumps(result))
            with (
                patch(
                    "openhisto.feature_encoder.auth.getpass.getpass",
                    return_value="synthetic-cloud-value",
                ) as prompt,
                patch("requests.head", side_effect=AssertionError("No cloud request")),
            ):
                result = check_access("gemini_embedding_2", auth=AuthConfig(prompt=True))
                self.assertEqual(result["status"], "credentials_present_unverified")
                prompt.assert_called_once()
                self.assertNotIn("synthetic-cloud-value", json.dumps(result))

    def test_access_prompt_stays_disabled_for_public_and_offline_profiles(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"HF_HUB_CACHE": directory}, clear=True),
            patch(
                "openhisto.feature_encoder.auth.getpass.getpass",
                side_effect=AssertionError("No prompt"),
            ) as prompt,
        ):
            self.assertEqual(
                check_access("phikon", auth=AuthConfig(prompt=True))["status"],
                "public_access_unverified",
            )
            self.assertEqual(
                check_access(
                    "uni_v1", auth=AuthConfig(prompt=True), config=EncodeConfig(offline=True)
                )["status"],
                "checkpoint_not_cached",
            )
            prompt.assert_not_called()

    def test_local_access_and_offline_mode_need_no_hf_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No credential lookup")))
            result = check_access(EncoderSpec("uni_v1", checkpoint=directory), auth=auth)
            self.assertEqual(result["status"], "local_checkpoint_present")
        with tempfile.TemporaryDirectory() as directory:
            descriptor = get_descriptor("uni_v1")
            cached = (
                Path(directory)
                / "models--MahmoodLab--UNI"
                / "snapshots"
                / descriptor.preprocessing["default_revision"]
            )
            cached.mkdir(parents=True)
            (cached / "config.json").write_text("{}")
            (cached / "pytorch_model.bin").write_bytes(b"synthetic cached weights")
            auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No credential lookup")))
            with patch("requests.head", side_effect=AssertionError("No network")):
                self.assertTrue(
                    check_access(
                        "uni_v1", auth=auth, config=EncodeConfig(offline=True, cache_dir=directory)
                    )["available"]
                )


class LocalAdapterTests(unittest.TestCase):
    def test_unused_local_options_are_rejected_before_dependencies_or_auth(self):
        auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No credential lookup")))
        profiles = (
            ("phikon", PhikonEncoder),
            ("phikon_v2", PhikonEncoder),
            ("uni_v1", UNIEncoder),
            ("conch15_titan", Conch15Encoder),
        )
        for model, factory in profiles:
            for options in ({"unused": True}, {"cache_tag": 4}):
                with (
                    self.subTest(model=model, options=options),
                    patch("openhisto.feature_encoder.locked.HF.local.dependency") as imports,
                ):
                    with self.assertRaises(ValueError):
                        factory(
                            EncoderSpec(model, options=options), config=EncodeConfig(), auth=auth
                        )
                    imports.assert_not_called()
        auth.resolve.assert_not_called()

    def test_local_cache_tag_is_the_only_supported_optional_setting(self):
        encoder = TorchEncoder.__new__(TorchEncoder)
        with (
            patch(
                "openhisto.feature_encoder.locked.HF.local.dependency",
                return_value=SimpleNamespace(),
            ),
            patch(
                "openhisto.feature_encoder.locked.HF.local.device_and_precision",
                return_value=(SimpleNamespace(type="cpu"), "float32"),
            ),
        ):
            encoder._initialize(
                EncoderSpec("phikon", options={"cache_tag": "synthetic"}),
                get_descriptor("phikon"),
                EncodeConfig(),
            )
        self.assertEqual(encoder.spec.options, {"cache_tag": "synthetic"})

    def test_complete_gated_cache_is_used_before_any_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots" / ("a" * 40)
            root.mkdir(parents=True)
            for name in ("config.json", "pytorch_model.bin"):
                (root / name).write_text("synthetic")

            def lookup(repo, name, **kwargs):
                path = root / name
                return str(path) if path.is_file() else None

            hub = SimpleNamespace(
                try_to_load_from_cache=lookup,
                hf_hub_download=Mock(side_effect=AssertionError("No network")),
            )
            auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No credential lookup")))
            with patch("openhisto.feature_encoder.locked.HF.local.dependency", return_value=hub):
                result = snapshot(
                    EncoderSpec("uni_v1"),
                    get_descriptor("uni_v1"),
                    EncodeConfig(),
                    auth,
                    [("model.safetensors", "pytorch_model.bin")],
                )
            self.assertEqual(result, (root, "a" * 40, None))
            auth.resolve.assert_not_called()
            hub.hf_hub_download.assert_not_called()

    def test_partial_offline_bundle_never_requests_credentials(self):
        hub = SimpleNamespace(
            try_to_load_from_cache=lambda *args, **kwargs: None, hf_hub_download=Mock()
        )
        auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No credential lookup")))
        with (
            patch("openhisto.feature_encoder.locked.HF.local.dependency", return_value=hub),
            self.assertRaisesRegex(EncoderError, "not cached"),
        ):
            snapshot(
                EncoderSpec("uni_v1"),
                get_descriptor("uni_v1"),
                EncodeConfig(offline=True),
                auth,
                ["pytorch_model.bin"],
            )
        auth.resolve.assert_not_called()
        hub.hf_hub_download.assert_not_called()

    def test_safe_first_weights_fall_back_only_for_missing_file(self):
        class EntryNotFoundError(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots" / ("a" * 40)
            root.mkdir(parents=True)
            calls = []

            def download(**kwargs):
                calls.append(kwargs["filename"])
                if kwargs["filename"] == "model.safetensors":
                    raise EntryNotFoundError("synthetic missing artifact")
                return str(root / kwargs["filename"])

            hub = SimpleNamespace(
                try_to_load_from_cache=lambda *args, **kwargs: None, hf_hub_download=download
            )
            with patch("openhisto.feature_encoder.locked.HF.local.dependency", return_value=hub):
                snapshot(
                    EncoderSpec("phikon"),
                    get_descriptor("phikon"),
                    EncodeConfig(),
                    AuthConfig(hf_token="synthetic-value"),
                    ["preprocessor_config.json", ("model.safetensors", "pytorch_model.bin")],
                )
            self.assertEqual(
                calls,
                [
                    "config.json",
                    "preprocessor_config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                ],
            )

            # Authentication/network failures are not reinterpreted as a missing
            # safe checkpoint and do not trigger a different weight format.
            def unauthorized(**kwargs):
                if kwargs["filename"] == "model.safetensors":
                    raise RuntimeError("synthetic denied")
                return str(root / kwargs["filename"])

            hub.hf_hub_download = Mock(side_effect=unauthorized)
            with (
                patch("openhisto.feature_encoder.locked.HF.local.dependency", return_value=hub),
                self.assertRaises(EncoderError),
            ):
                snapshot(
                    EncoderSpec("phikon"),
                    get_descriptor("phikon"),
                    EncodeConfig(),
                    AuthConfig(hf_token="synthetic-value"),
                    [("model.safetensors", "pytorch_model.bin")],
                )
            self.assertNotIn(
                "pytorch_model.bin",
                [call.kwargs["filename"] for call in hub.hf_hub_download.call_args_list],
            )

    def test_snapshot_resolves_metadata_then_pins_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots" / ("a" * 40)
            root.mkdir(parents=True)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                return str(root / kwargs["filename"])

            hub = SimpleNamespace(
                try_to_load_from_cache=lambda *args, **kwargs: None, hf_hub_download=download
            )
            descriptor = get_descriptor("phikon")
            with patch("openhisto.feature_encoder.locked.HF.local.dependency", return_value=hub):
                path, revision, explicit = snapshot(
                    EncoderSpec("phikon", revision="synthetic-tag"),
                    descriptor,
                    EncodeConfig(),
                    AuthConfig(hf_token="synthetic-value"),
                    ["preprocessor_config.json", "model.safetensors"],
                )
            self.assertEqual(path, root)
            self.assertEqual(revision, "a" * 40)
            self.assertIsNone(explicit)
            self.assertEqual(
                [call["revision"] for call in calls], ["synthetic-tag", "a" * 40, "a" * 40]
            )
            self.assertEqual(
                [call["filename"] for call in calls],
                ["config.json", "preprocessor_config.json", "model.safetensors"],
            )

    def test_local_bundle_bypasses_hub_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            auth = SimpleNamespace(resolve=Mock(side_effect=AssertionError("No auth lookup")))
            with patch(
                "openhisto.feature_encoder.locked.HF.local.dependency",
                side_effect=AssertionError("No hub import"),
            ):
                path, revision, explicit = snapshot(
                    EncoderSpec("phikon", checkpoint=directory),
                    get_descriptor("phikon"),
                    EncodeConfig(offline=True),
                    auth,
                    [],
                )
            self.assertEqual(path, Path(directory).resolve())
            self.assertIsNone(revision)
            self.assertIsNone(explicit)

    def test_checkpoint_loader_never_falls_back_to_unsafe_pickle(self):
        torch = SimpleNamespace(load=Mock(return_value={"state_dict": {"synthetic_tensor": 1}}))
        self.assertEqual(load_weights(torch, Path("synthetic.bin")), {"synthetic_tensor": 1})
        self.assertTrue(torch.load.call_args.kwargs["weights_only"])
        torch.load.side_effect = TypeError("old API")
        with self.assertRaisesRegex(EncoderError, "weights_only"):
            load_weights(torch, Path("synthetic.bin"))
        self.assertEqual(torch.load.call_count, 2)

    def test_device_precision_defaults_and_cpu_restriction(self):
        def fake_torch(cuda=False, mps=False, bf16=False):
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                device=lambda name: SimpleNamespace(type=name.split(":")[0]),
            )

        device, precision = device_and_precision(fake_torch(), EncodeConfig())
        self.assertEqual((device.type, precision), ("cpu", "float32"))
        self.assertEqual(device_and_precision(fake_torch(mps=True), EncodeConfig())[0].type, "mps")
        self.assertEqual(device_and_precision(fake_torch(cuda=True), EncodeConfig())[1], "float16")
        self.assertEqual(
            device_and_precision(fake_torch(cuda=True, bf16=True), EncodeConfig())[1], "bfloat16"
        )
        with self.assertRaises(EncoderError):
            device_and_precision(fake_torch(), EncodeConfig(precision="float16"))

    def test_phikon_uses_cls_not_pooler_output(self):
        encoder = PhikonEncoder.__new__(PhikonEncoder)
        tokens = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        encoder.model = lambda **kwargs: SimpleNamespace(
            last_hidden_state=tokens, pooler_output=np.zeros((2, 4))
        )
        np.testing.assert_array_equal(encoder._forward({"pixel_values": object()}), tokens[:, 0, :])

    def test_batch_wrapper_preserves_singletons_and_validates_rgb(self):
        encoder = TorchEncoder.__new__(TorchEncoder)
        encoder.dimension = 3
        encoder.device = SimpleNamespace(type="cpu")
        encoder.precision = "float32"
        encoder.torch = SimpleNamespace(
            stack=lambda values: FakeTensor(np.stack([value.value for value in values])),
            inference_mode=nullcontext,
        )
        encoder.model = lambda batch: FakeTensor(np.ones((len(batch.value), 3), dtype=np.float32))
        encoder.processor = None
        encoder.transform = lambda image: FakeTensor(np.asarray(image))
        images = [np.zeros((4, 4, 3), dtype=np.uint8)]
        self.assertEqual(encoder.encode(images).shape, (1, 3))
        self.assertEqual(encoder.encode(images * 2).shape, (2, 3))
        self.assertEqual(encoder.encode([]).shape, (0, 3))
        with self.assertRaises(ValueError):
            encoder.encode([np.zeros((4, 4, 3), dtype=np.float32)])
        encoder.close()
        with self.assertRaises(EncoderError):
            encoder.encode(images)


if __name__ == "__main__":
    unittest.main()
