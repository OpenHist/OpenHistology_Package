"""Runtime credential lookup. This object is never part of run provenance."""

import getpass
import os
import re

from .base import AccessError


class AuthConfig:
    """Resolve explicit secrets, named environment variables, or the HF cache.

    Secret values are excluded from ``repr``. Masked prompting is opt-in and is
    used only when a required credential is actually requested. AWS adapters use
    the standard SDK credential chain with optional profile and region names.
    """

    def __init__(
        self,
        *,
        hf_token_env="HF_TOKEN",
        gemini_api_key_env="GEMINI_API_KEY",
        cohere_api_key_env="COHERE_API_KEY",
        hf_token=None,
        gemini_api_key=None,
        cohere_api_key=None,
        prompt=False,
        prompt_credentials=None,
        aws_profile=None,
        aws_region=None,
    ):
        self.hf_token_env = self._env_name(hf_token_env)
        self.gemini_api_key_env = self._env_name(gemini_api_key_env)
        self.cohere_api_key_env = self._env_name(cohere_api_key_env)
        self.prompt = prompt if prompt_credentials is None else prompt_credentials
        if not isinstance(self.prompt, bool):
            raise ValueError("Credential prompting must be boolean")
        for value in (aws_profile, aws_region):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("AWS profile and region must be nonempty strings or None")
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self._values = {}
        for provider, value in (
            ("hf", hf_token),
            ("google", gemini_api_key),
            ("cohere", cohere_api_key),
        ):
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Runtime credentials must be nonempty strings")
                self._values[provider] = value.strip()

    @staticmethod
    def _env_name(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("Credential environment-variable names must be valid identifiers")
        return value

    @staticmethod
    def _provider(provider):
        aliases = {
            "hf": "hf",
            "huggingface": "hf",
            "google": "google",
            "gemini": "google",
            "cohere": "cohere",
        }
        if provider not in aliases:
            raise ValueError("Credential provider must be hf, google/gemini, or cohere")
        return aliases[provider]

    def env_name(self, provider):
        provider = self._provider(provider)
        return {
            "hf": self.hf_token_env,
            "google": self.gemini_api_key_env,
            "cohere": self.cohere_api_key_env,
        }[provider]

    def resolve(self, provider, *, required=True):
        provider = self._provider(provider)
        value = self._values.get(provider) or os.environ.get(self.env_name(provider), "").strip()
        if not value and provider == "hf":
            try:
                from huggingface_hub import get_token

                value = get_token()
            except Exception:
                value = None
        if not value and required and self.prompt:
            try:
                value = getpass.getpass(f"Enter {self.env_name(provider)} (hidden): ").strip()
            except (EOFError, KeyboardInterrupt):
                raise AccessError("Credential entry was cancelled") from None
            if value:
                self._values[provider] = value
        if not value and required:
            suffix = " or sign in with the Hugging Face CLI" if provider == "hf" else ""
            raise AccessError(
                f"Set {self.env_name(provider)} or supply AuthConfig explicitly{suffix}"
            )
        return value or None

    def __repr__(self):
        return (
            f"AuthConfig(hf_token_env={self.hf_token_env!r}, "
            f"gemini_api_key_env={self.gemini_api_key_env!r}, "
            f"cohere_api_key_env={self.cohere_api_key_env!r}, prompt={self.prompt!r}, "
            f"aws_profile={self.aws_profile!r}, aws_region={self.aws_region!r}, "
            "credentials=<redacted>)"
        )
