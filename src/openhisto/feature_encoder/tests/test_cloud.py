"""Cloud adapter contracts exercised entirely with synthetic arrays and SDK mocks."""

import base64
import importlib.util
import inspect
from io import BytesIO
import json
import sys
import threading
import time
import traceback
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from openhisto.feature_encoder.base import AccessError, EncoderError
from openhisto.feature_encoder.config import EncodeConfig, EncoderSpec
from openhisto.feature_encoder.non_path.common import CloudBase, png_bytes
from openhisto.feature_encoder.non_path.gemini import GeminiEncoder
from openhisto.feature_encoder.non_path.cohere import CohereEncoder
from openhisto.feature_encoder.non_path.nova import NovaEncoder


class FakeAuth:
    aws_profile = "synthetic-profile"
    aws_region = "us-east-1"

    def __init__(self):
        self.providers = []

    def resolve(self, provider, *, required):
        self.providers.append((provider, required))
        return "synthetic-token-not-a-real-key"


class HTTPError(Exception):
    def __init__(self, status, headers=None):
        super().__init__("synthetic-token-not-a-real-key data:image/png;base64,DO_NOT_ECHO")
        self.status_code = status
        self.response = types.SimpleNamespace(headers=headers or {})


def synthetic_images(count=3):
    return [np.full((4, 5, 3), index + 1, dtype=np.uint8) for index in range(count)]


def available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def image_value(data):
    with Image.open(BytesIO(data)) as image:
        return float(np.asarray(image)[0, 0, 0])


def google_modules(client):
    google = types.ModuleType("google")
    google.__path__ = []
    genai = types.ModuleType("google.genai")
    genai.__version__ = "synthetic-sdk"
    genai.Client = lambda **kwargs: client.configure(kwargs)
    api_types = types.ModuleType("google.genai.types")
    api_types.HttpOptions = lambda **kwargs: types.SimpleNamespace(**kwargs)
    api_types.HttpRetryOptions = lambda **kwargs: types.SimpleNamespace(**kwargs)
    api_types.EmbedContentConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    api_types.Content = lambda **kwargs: types.SimpleNamespace(**kwargs)
    api_types.Part = types.SimpleNamespace(
        from_bytes=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    genai.types = api_types
    google.genai = genai
    return {"google": google, "google.genai": genai, "google.genai.types": api_types}


class GoogleClient:
    def __init__(self):
        self.models = self
        self.calls = []
        self.closed = False

    def configure(self, kwargs):
        self.options = kwargs
        return self

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        self_test = kwargs["contents"]
        if len(self_test) != 1 or len(self_test[0].parts) != 1:
            raise AssertionError("Each request must carry one independent image")
        value = image_value(self_test[0].parts[0].data)
        return types.SimpleNamespace(
            embeddings=[
                types.SimpleNamespace(values=[value] * kwargs["config"].output_dimensionality)
            ],
            usage_metadata=types.SimpleNamespace(prompt_token_count=7),
        )

    def close(self):
        self.closed = True


class CohereClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    def configure(self, kwargs):
        self.options = kwargs
        return self

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        vectors = []
        for item in kwargs["inputs"]:
            if len(item["content"]) != 1:
                raise AssertionError("Images must not be fused into one input")
            url = item["content"][0]["image_url"]["url"]
            value = image_value(base64.b64decode(url.split(",", 1)[1]))
            vectors.append([value] * kwargs["output_dimension"])
        return types.SimpleNamespace(
            embeddings=types.SimpleNamespace(float_=vectors),
            meta=types.SimpleNamespace(
                billed_units=types.SimpleNamespace(input_tokens=len(vectors) * 2)
            ),
        )

    def close(self):
        self.closed = True


def cohere_modules(client):
    module = types.ModuleType("cohere")
    module.__version__ = "synthetic-sdk"
    module.ClientV2 = lambda **kwargs: client.configure(kwargs)
    return {"cohere": module}


class NovaClient:
    def __init__(self):
        self.meta = types.SimpleNamespace(region_name="us-east-1")
        self.calls = []
        self.closed = False

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        body = json.loads(kwargs["body"])
        params = body["singleEmbeddingParams"]
        value = image_value(base64.b64decode(params["image"]["source"]["bytes"]))
        response = {
            "embeddings": [
                {"embeddingType": "IMAGE", "embedding": [value] * params["embeddingDimension"]}
            ]
        }
        return {
            "body": BytesIO(json.dumps(response).encode()),
            "ResponseMetadata": {"RequestId": "synthetic-id"},
        }

    def close(self):
        self.closed = True


def nova_modules(client):
    boto3 = types.ModuleType("boto3")
    boto3.__version__ = "synthetic-sdk"

    class Session:
        def __init__(self, **kwargs):
            client.session_options = kwargs

        def client(self, service, **kwargs):
            client.client_options = kwargs
            client.service = service
            return client

    boto3.Session = Session
    botocore = types.ModuleType("botocore")
    botocore.__path__ = []
    config = types.ModuleType("botocore.config")
    config.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    botocore.config = config
    return {"boto3": boto3, "botocore": botocore, "botocore.config": config}


class CloudProviderTests(unittest.TestCase):
    def test_google_has_one_content_one_image_and_independent_ordered_vectors(self):
        client = GoogleClient()
        auth = FakeAuth()
        with patch.dict(sys.modules, google_modules(client)):
            encoder = GeminiEncoder(
                EncoderSpec("gemini_embedding_2", dimension=128),
                config=EncodeConfig(cloud_concurrency=2),
                auth=auth,
            )
            output = encoder.encode(synthetic_images())
            self.assertEqual(output.shape, (3, 128))
            np.testing.assert_array_equal(output[:, 0], [1, 2, 3])
            self.assertEqual(output.dtype, np.float32)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(client.options["http_options"].retry_options.attempts, 1)
            self.assertFalse(client.options["vertexai"])
            self.assertEqual(auth.providers, [("google", True)])
            self.assertEqual(encoder.usage["prompt_tokens"], 21)
            self.assertIsNone(encoder.metadata["resolved_revision"])
            self.assertNotIn("synthetic-token", json.dumps(encoder.metadata))
            encoder.close()
            self.assertTrue(client.closed)

    def test_google_active_requests_are_bounded_by_concurrency(self):
        client = GoogleClient()
        original = client.embed_content
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def slow_request(**kwargs):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            try:
                time.sleep(0.005)
                return original(**kwargs)
            finally:
                with lock:
                    state["active"] -= 1

        client.embed_content = slow_request
        with patch.dict(sys.modules, google_modules(client)):
            encoder = GeminiEncoder(
                EncoderSpec("gemini_embedding_2", dimension=128),
                config=EncodeConfig(cloud_concurrency=2),
                auth=FakeAuth(),
            )
            encoder.encode(synthetic_images(8))
            self.assertEqual(state["maximum"], 2)
            encoder.close()

    def test_google_rejects_combined_or_wrong_shape_output(self):
        client = GoogleClient()
        client.embed_content = lambda **kwargs: {
            "embeddings": [{"values": [0] * 128}, {"values": [0] * 128}]
        }
        with patch.dict(sys.modules, google_modules(client)):
            encoder = GeminiEncoder(
                EncoderSpec("gemini_embedding_2", dimension=128),
                config=EncodeConfig(),
                auth=FakeAuth(),
            )
            with self.assertRaisesRegex(EncoderError, "number"):
                encoder.encode(synthetic_images(1))
            encoder.close()

    def test_cohere_batches_separate_objects_and_disables_sdk_retry(self):
        client = CohereClient()
        with patch.dict(sys.modules, cohere_modules(client)):
            encoder = CohereEncoder(
                EncoderSpec("cohere_embed_5_fast", dimension=256),
                config=EncodeConfig(batch_size=2),
                auth=FakeAuth(),
            )
            output = encoder.encode(synthetic_images(5))
            np.testing.assert_array_equal(output[:, 0], [1, 2, 3, 4, 5])
            self.assertEqual([len(call["inputs"]) for call in client.calls], [2, 2, 1])
            self.assertTrue(
                all(call["request_options"]["max_retries"] == 0 for call in client.calls)
            )
            self.assertEqual(client.calls[0]["model"], "embed-v5.0-fast")
            self.assertEqual(encoder.usage["billed_input_tokens"], 10)
            encoder.close()

    def test_cohere_payload_limit_and_response_count_are_enforced(self):
        from openhisto.feature_encoder.non_path import cohere as adapter

        client = CohereClient()
        with patch.dict(sys.modules, cohere_modules(client)):
            encoder = CohereEncoder(
                EncoderSpec("cohere_embed_5_pro", dimension=256),
                config=EncodeConfig(batch_size=200),
                auth=FakeAuth(),
            )
            single = list(encoder._chunks(synthetic_images(1)))[0]
            limit = len(json.dumps(encoder._body(single)).encode()) + 5
            with patch.object(adapter, "MAX_PAYLOAD_BYTES", limit):
                output = encoder.encode(synthetic_images(3))
            self.assertEqual(output.shape, (3, 256))
            self.assertTrue(all(len(call["inputs"]) == 1 for call in client.calls))
            with patch.object(adapter, "MAX_PAYLOAD_BYTES", 1):
                with self.assertRaisesRegex(EncoderError, "payload limit"):
                    encoder.encode(synthetic_images(1))
            client.embed = lambda **kwargs: {"embeddings": {"float": [[1] * 256]}}
            with self.assertRaisesRegex(EncoderError, "shape"):
                encoder.encode(synthetic_images(2))
            encoder.close()

    def test_nova_uses_sdk_profile_and_one_single_image_request(self):
        client = NovaClient()
        auth = FakeAuth()
        with patch.dict(sys.modules, nova_modules(client)):
            encoder = NovaEncoder(
                EncoderSpec("nova_multimodal", dimension=384),
                config=EncodeConfig(cloud_concurrency=2),
                auth=auth,
            )
            output = encoder.encode(synthetic_images())
            np.testing.assert_array_equal(output[:, 0], [1, 2, 3])
            self.assertEqual(output.shape, (3, 384))
            self.assertEqual(client.service, "bedrock-runtime")
            self.assertEqual(
                client.session_options,
                {"profile_name": "synthetic-profile", "region_name": "us-east-1"},
            )
            self.assertEqual(client.client_options["config"].retries["total_max_attempts"], 1)
            self.assertEqual(auth.providers, [])
            for call in client.calls:
                body = json.loads(call["body"])
                self.assertEqual(body["taskType"], "SINGLE_EMBEDDING")
                self.assertEqual(
                    set(body["singleEmbeddingParams"]) & {"image", "text", "audio", "video"},
                    {"image"},
                )
            self.assertEqual(encoder.metadata["region"], "us-east-1")
            encoder.close()

    def test_offline_invalid_options_and_dimensions_fail_before_credentials(self):
        for cls, profile in (
            (GeminiEncoder, "gemini_embedding_2"),
            (CohereEncoder, "cohere_embed_5_pro"),
            (NovaEncoder, "nova_multimodal"),
        ):
            auth = FakeAuth()
            with self.assertRaises(AccessError):
                cls(EncoderSpec(profile), config=EncodeConfig(offline=True), auth=auth)
            self.assertEqual(auth.providers, [])
            with self.assertRaises(EncoderError):
                cls(EncoderSpec(profile, dimension=17), config=EncodeConfig(), auth=auth)
            with self.assertRaises(EncoderError):
                cls(
                    EncoderSpec(profile, options={"endpoint": "https://example.invalid"}),
                    config=EncodeConfig(),
                    auth=auth,
                )


class CloudRetryTests(unittest.TestCase):
    def encoder(self, retries=2):
        encoder = CloudBase(
            EncoderSpec("synthetic"),
            EncodeConfig(max_retries=retries),
            provider="synthetic",
            model_name="synthetic",
            dimension=1,
            task=None,
            sdk=types.SimpleNamespace(__version__="synthetic"),
            sdk_name="synthetic",
        )
        encoder._client = types.SimpleNamespace()
        return encoder

    def test_transient_retry_honors_header_and_stops_at_budget(self):
        encoder = self.encoder()
        attempts = []

        def request():
            attempts.append(1)
            if len(attempts) < 3:
                raise HTTPError(429, {"Retry-After": "0"})
            return "synthetic-result"

        with patch("openhisto.feature_encoder.non_path.common.time.sleep") as sleep:
            self.assertEqual(encoder._request(request), "synthetic-result")
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(sleep.call_args.args[0], 0)
        self.assertEqual(encoder.usage["request_attempts"], 3)
        self.assertEqual(encoder.usage["retries"], 2)

    def test_auth_failure_is_not_retried_and_exception_payload_is_suppressed(self):
        encoder = self.encoder()
        with patch("openhisto.feature_encoder.non_path.common.time.sleep") as sleep:
            try:
                encoder._request(lambda: (_ for _ in ()).throw(HTTPError(403)))
            except AccessError as error:
                rendered = "".join(traceback.format_exception(error))
                self.assertNotIn("synthetic-token", rendered)
                self.assertNotIn("DO_NOT_ECHO", rendered)
            else:
                self.fail("Expected AccessError")
            sleep.assert_not_called()
        self.assertEqual(encoder.usage["request_attempts"], 1)

    def test_retry_exhaustion_long_delay_and_nonfinite_features(self):
        encoder = self.encoder(retries=1)
        with patch("openhisto.feature_encoder.non_path.common.time.sleep") as sleep:
            with self.assertRaises(EncoderError):
                encoder._request(lambda: (_ for _ in ()).throw(HTTPError(503)))
            self.assertEqual(sleep.call_count, 1)
        encoder = self.encoder()
        with patch("openhisto.feature_encoder.non_path.common.time.sleep") as sleep:
            with self.assertRaisesRegex(EncoderError, "60 seconds"):
                encoder._request(
                    lambda: (_ for _ in ()).throw(HTTPError(429, {"Retry-After": "61"}))
                )
            sleep.assert_not_called()
        from openhisto.feature_encoder.non_path.common import validated

        with self.assertRaisesRegex(EncoderError, "nonfinite"):
            validated([[float("nan")]], 1, 1)
        with self.assertRaisesRegex(EncoderError, "float32"):
            validated([[1e300]], 1, 1)
        with self.assertRaises(EncoderError):
            png_bytes(np.zeros((4, 4), dtype=np.uint8))


class OptionalSDKSchemaTests(unittest.TestCase):
    """Validate installed SDK schemas locally; never invoke a remote method."""

    @unittest.skipUnless(available("google.genai"), "optional google-genai unavailable")
    def test_actual_google_request_and_response_types(self):
        from google.genai import types as real_types

        content = real_types.Content(
            role="user",
            parts=[real_types.Part.from_bytes(data=b"SYNTHETIC", mime_type="image/png")],
        )
        options = real_types.HttpOptions(
            base_url="https://generativelanguage.googleapis.com",
            api_version="v1beta",
            timeout=60000,
            retry_options=real_types.HttpRetryOptions(attempts=1),
        )
        config = real_types.EmbedContentConfig(output_dimensionality=128)
        response = real_types.EmbedContentResponse(
            embeddings=[
                real_types.ContentEmbedding(values=[1.0] * 128, statistics={"token_count": 7})
            ],
            metadata={"billable_character_count": 10},
        )
        self.assertEqual(content.parts[0].inline_data.mime_type, "image/png")
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertEqual(config.output_dimensionality, 128)
        client = GoogleClient()
        client.embed_content = lambda **kwargs: response
        with patch.dict(sys.modules, google_modules(client)):
            encoder = GeminiEncoder(
                EncoderSpec("gemini_embedding_2", dimension=128),
                config=EncodeConfig(),
                auth=FakeAuth(),
            )
            encoder.encode(synthetic_images(1))
            self.assertEqual(encoder.usage["billable_characters"], 10)
            self.assertEqual(encoder.usage["embedding_tokens"], 7)
            encoder.close()

    @unittest.skipUnless(available("cohere"), "optional cohere unavailable")
    def test_actual_cohere_typed_input_options_and_response(self):
        import cohere
        from cohere.core.request_options import RequestOptions
        from cohere.types.embed_input import EmbedInput
        from cohere.types.embed_by_type_response import EmbedByTypeResponse
        from pydantic import TypeAdapter

        item = {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,U1lOVEhFVElD"}}
            ]
        }
        TypeAdapter(EmbedInput).validate_python(item)
        TypeAdapter(RequestOptions).validate_python({"timeout": 60.0, "max_retries": 0})
        for name in (
            "inputs",
            "model",
            "input_type",
            "embedding_types",
            "output_dimension",
            "request_options",
        ):
            self.assertIn(name, inspect.signature(cohere.ClientV2.embed).parameters)
        response = EmbedByTypeResponse.model_validate(
            {
                "id": "synthetic",
                "embeddings": {"float": [[0.0] * 256]},
                "meta": {"billed_units": {"input_tokens": 2}},
            }
        )
        self.assertEqual(len(response.embeddings.float_[0]), 256)
        self.assertEqual(response.meta.billed_units.input_tokens, 2)

    @unittest.skipUnless(available("botocore.config"), "optional botocore unavailable")
    def test_actual_botocore_retry_and_endpoint_configuration(self):
        from botocore.config import Config

        config = Config(
            ignore_configured_endpoint_urls=True,
            retries={"mode": "standard", "total_max_attempts": 1},
            connect_timeout=60,
            read_timeout=60,
        )
        self.assertTrue(config.ignore_configured_endpoint_urls)
        self.assertEqual(config.retries["total_max_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
