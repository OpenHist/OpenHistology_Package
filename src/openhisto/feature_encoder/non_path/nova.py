"""Amazon Nova Multimodal Embeddings through the standard AWS SDK chain."""

import base64
import json

import numpy as np

from ..base import AccessError, EncoderError
from .common import CloudBase, check_options, field, png_bytes, safe_encode, validated


MODEL_NAME = "amazon.nova-2-multimodal-embeddings-v1:0"


class NovaEncoder(CloudBase):
    def __init__(self, spec, *, config, auth):
        if config.offline:
            raise AccessError("Cloud embeddings are unavailable in offline mode")
        if spec.model != "nova_multimodal":
            raise EncoderError("NovaEncoder requires the nova_multimodal profile")
        dimension = check_options(
            spec,
            allowed={"task", "cache_tag", "region", "profile"},
            dimensions={256, 384, 1024, 3072},
            default_dimension=3072,
        )
        task = spec.options.get("task", "CLASSIFICATION")
        if not isinstance(task, str) or task.upper() not in {
            "GENERIC_INDEX",
            "GENERIC_RETRIEVAL",
            "TEXT_RETRIEVAL",
            "IMAGE_RETRIEVAL",
            "VIDEO_RETRIEVAL",
            "DOCUMENT_RETRIEVAL",
            "AUDIO_RETRIEVAL",
            "CLASSIFICATION",
            "CLUSTERING",
        }:
            raise EncoderError("Unsupported Nova embedding purpose")
        task = task.upper()
        region = spec.options.get("region", getattr(auth, "aws_region", None))
        profile = spec.options.get("profile", getattr(auth, "aws_profile", None))
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (region, profile)
        ):
            raise EncoderError("AWS region/profile must be nonempty strings when supplied")
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise EncoderError(
                "Install the optional boto3 dependency to use Amazon Nova embeddings"
            ) from None
        super().__init__(
            spec,
            config,
            provider="aws_bedrock",
            model_name=MODEL_NAME,
            dimension=dimension,
            task=task,
            sdk=boto3,
            sdk_name="boto3",
        )
        try:
            session_options = {}
            if profile is not None:
                session_options["profile_name"] = profile
            if region is not None:
                session_options["region_name"] = region
            session = boto3.Session(**session_options)
            self._client = session.client(
                "bedrock-runtime",
                config=Config(
                    connect_timeout=config.timeout,
                    read_timeout=config.timeout,
                    ignore_configured_endpoint_urls=True,
                    retries={"mode": "standard", "total_max_attempts": 1},
                ),
            )
        except Exception:
            raise AccessError(
                "AWS client setup failed; configure an AWS SDK profile/role and a supported Bedrock region through AuthConfig"
            ) from None
        self.metadata["region"] = field(field(self._client, "meta", {}), "region_name", region)
        self.metadata["endpoint"] = field(field(self._client, "meta", {}), "endpoint_url")
        self.metadata["preprocessing"]["detail_level"] = "STANDARD_IMAGE"
        self.metadata["batching"] = "one_image_per_SINGLE_EMBEDDING_request"

    def _one(self, image):
        payload = base64.b64encode(png_bytes(image)).decode("ascii")
        body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": self.metadata["task"],
                "embeddingDimension": self.dimension,
                "image": {
                    "format": "png",
                    "detailLevel": "STANDARD_IMAGE",
                    "source": {"bytes": payload},
                },
            },
        }
        if len(json.dumps(body).encode("utf-8")) > 25_000_000:
            raise EncoderError(
                "An encoded image exceeds Nova's inline request limit; choose smaller patches"
            )
        response = self._request(
            lambda: self._client.invoke_model(
                modelId=MODEL_NAME,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
        )
        try:
            stream = response["body"]
            try:
                result = json.loads(stream.read())
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        except Exception:
            raise EncoderError("Nova returned an unreadable embedding response") from None
        embeddings = field(result, "embeddings")
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != 1
            or field(embeddings[0], "embeddingType") != "IMAGE"
        ):
            raise EncoderError("Nova returned an unexpected image embedding response")
        array = validated([field(embeddings[0], "embedding")], 1, self.dimension)
        self._add_usage({"images": 1})
        return array[0]

    @safe_encode
    def encode(self, images):
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        return validated(self._parallel(self._one, images), len(images), self.dimension)
