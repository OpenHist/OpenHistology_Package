"""Cohere Embed 5 with separate image inputs and bounded request payloads."""

import base64
import json

import numpy as np

from ..base import AccessError, EncoderError
from .common import CloudBase, check_options, field, png_bytes, safe_encode, validated


MODELS = {"cohere_embed_5_pro": "embed-v5.0-pro", "cohere_embed_5_fast": "embed-v5.0-fast"}
MAX_PAYLOAD_BYTES = 20_000_000


class CohereEncoder(CloudBase):
    def __init__(self, spec, *, config, auth):
        if config.offline:
            raise AccessError("Cloud embeddings are unavailable in offline mode")
        if spec.model not in MODELS:
            raise EncoderError(
                "CohereEncoder requires a cohere_embed_5_pro or cohere_embed_5_fast profile"
            )
        dimension = check_options(
            spec,
            allowed={"task", "cache_tag"},
            dimensions={256, 512, 768, 1024, 1536, 2048},
            default_dimension=2048,
        )
        task = spec.options.get("task", "classification")
        if task not in {"classification", "clustering", "search_document", "search_query"}:
            raise EncoderError(
                "Cohere task must be classification, clustering, search_document, or search_query"
            )
        try:
            import cohere
        except ImportError:
            raise EncoderError(
                "Install the optional cohere dependency to use Cohere embeddings"
            ) from None
        token = auth.resolve("cohere", required=True)
        super().__init__(
            spec,
            config,
            provider="cohere",
            model_name=MODELS[spec.model],
            dimension=dimension,
            task=task,
            sdk=cohere,
            sdk_name="cohere",
        )
        try:
            self._client = cohere.ClientV2(
                api_key=token,
                base_url="https://api.cohere.com",
                timeout=config.timeout,
                max_retries=0,
                log_warning_experimental_features=False,
            )
        except Exception:
            raise AccessError(
                "Cohere client setup failed; check the credential reference and optional cohere SDK installation"
            ) from None
        self.metadata["batching"] = (
            "one_image_per_input_object; at_most_96_inputs_and_20MB_per_request"
        )
        self.metadata["endpoint"] = "https://api.cohere.com/v2/embed"

    def _body(self, inputs):
        return {
            "model": self.metadata["model_name"],
            "input_type": self.metadata["task"],
            "embedding_types": ["float"],
            "output_dimension": self.dimension,
            "inputs": inputs,
        }

    def _chunks(self, images):
        base_size = len(json.dumps(self._body([])).encode("utf-8"))
        inputs = []
        size = base_size
        for image in images:
            url = "data:image/png;base64," + base64.b64encode(png_bytes(image)).decode("ascii")
            item = {"content": [{"type": "image_url", "image_url": {"url": url}}]}
            item_size = len(json.dumps(item).encode("utf-8"))
            if base_size + item_size > MAX_PAYLOAD_BYTES:
                raise EncoderError(
                    "An encoded image exceeds Cohere's request payload limit; choose smaller patches"
                )
            addition = item_size + (2 if inputs else 0)
            if inputs and (
                len(inputs) >= min(96, self.config.batch_size)
                or size + addition > MAX_PAYLOAD_BYTES
            ):
                yield inputs
                inputs = []
                size = base_size
                addition = item_size
            inputs.append(item)
            size += addition
        if inputs:
            yield inputs

    @safe_encode
    def encode(self, images):
        if self._closed:
            raise EncoderError("Cloud encoder is closed")
        output = []
        for inputs in self._chunks(images):
            response = self._request(
                lambda: self._client.embed(
                    **self._body(inputs),
                    request_options={"max_retries": 0, "timeout": self.config.timeout},
                )
            )
            embeddings = field(response, "embeddings", {})
            values = field(embeddings, "float_")
            if values is None:
                values = field(embeddings, "float")
            output.append(validated(values, len(inputs), self.dimension))
            billed = field(field(response, "meta", {}), "billed_units", {})
            self._add_usage(
                {
                    "images": len(inputs),
                    "billed_input_tokens": field(billed, "input_tokens"),
                    "billed_output_tokens": field(billed, "output_tokens"),
                    "billed_images": field(billed, "images"),
                }
            )
        if not output:
            return np.empty((0, self.dimension), dtype=np.float32)
        return validated(np.concatenate(output, axis=0), len(images), self.dimension)
