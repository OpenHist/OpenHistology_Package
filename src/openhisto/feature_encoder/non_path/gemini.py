"""Gemini Embedding 2: one explicit image Content per independent request."""

import numpy as np

from ..base import AccessError, EncoderError
from .common import CloudBase, check_options, field, png_bytes, safe_encode, validated


class GeminiEncoder(CloudBase):
    def __init__(self, spec, *, config, auth):
        if config.offline:
            raise AccessError("Cloud embeddings are unavailable in offline mode")
        if spec.model != "gemini_embedding_2":
            raise EncoderError("GeminiEncoder requires the gemini_embedding_2 profile")
        dimension = check_options(spec, allowed={"task", "cache_tag"}, default_dimension=3072)
        if not 128 <= dimension <= 3072:
            raise EncoderError("Gemini Embedding 2 supports dimensions from 128 to 3072")
        if spec.options.get("task") is not None:
            raise EncoderError(
                "Gemini Embedding 2 image inputs do not accept task_type; omit the task option"
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise EncoderError(
                "Install the optional google-genai dependency to use Gemini embeddings"
            ) from None
        token = auth.resolve("google", required=True)
        super().__init__(
            spec,
            config,
            provider="google",
            model_name="gemini-embedding-2",
            dimension=dimension,
            task=None,
            sdk=genai,
            sdk_name="google-genai",
        )
        self._types = types
        try:
            options = types.HttpOptions(
                base_url="https://generativelanguage.googleapis.com",
                api_version="v1beta",
                timeout=max(1, int(config.timeout * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
            )
            self._client = genai.Client(api_key=token, vertexai=False, http_options=options)
        except Exception:
            raise AccessError(
                "Gemini client setup failed; check the credential reference and install a current google-genai SDK with retry configuration support"
            ) from None
        self.metadata["batching"] = "one_image_per_request"
        self.metadata["endpoint"] = "https://generativelanguage.googleapis.com/v1beta"

    def _one(self, image):
        payload = png_bytes(image)
        content = self._types.Content(
            role="user", parts=[self._types.Part.from_bytes(data=payload, mime_type="image/png")]
        )
        response = self._request(
            lambda: self._client.models.embed_content(
                model="gemini-embedding-2",
                contents=[content],
                config=self._types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
        )
        embeddings = field(response, "embeddings")
        if embeddings is None or len(embeddings) != 1:
            raise EncoderError("Gemini returned an unexpected number of image embeddings")
        array = validated([field(embeddings[0], "values")], 1, self.dimension)
        usage = field(response, "usage_metadata") or field(response, "usageMetadata", {})
        metadata = field(response, "metadata", {})
        statistics = field(embeddings[0], "statistics", {})
        self._add_usage(
            {
                "images": 1,
                "prompt_tokens": field(
                    usage, "prompt_token_count", field(usage, "promptTokenCount")
                ),
                "billable_characters": field(metadata, "billable_character_count"),
                "embedding_tokens": field(statistics, "token_count"),
            }
        )
        return array[0]

    @safe_encode
    def encode(self, images):
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        return validated(self._parallel(self._one, images), len(images), self.dimension)
