"""Phikon v1/v2 CLS features using their checkpoint-specific image processors."""

from ..base import EncoderError
from ..locked.HF.local import TorchEncoder, checkpoint_file, dependency, read_json, sha256, snapshot


class PhikonEncoder(TorchEncoder):
    def __init__(self, spec, *, config, auth):
        from ..registry import get_descriptor

        descriptor = get_descriptor(spec)
        self._initialize(spec, descriptor, config)
        transformers = dependency("transformers")
        directory, revision, explicit = snapshot(
            spec,
            descriptor,
            config,
            auth,
            ["preprocessor_config.json", ("model.safetensors", "pytorch_model.bin")],
        )
        config_data = read_json(directory / "config.json")
        expected_type = "vit" if spec.model == "phikon" else "dinov2"
        if (
            config_data.get("model_type") != expected_type
            or config_data.get("hidden_size") != self.dimension
        ):
            raise EncoderError(
                "The local checkpoint architecture/dimension does not match this Phikon profile"
            )
        processor_path = directory / "preprocessor_config.json"
        processor_config = read_json(processor_path)
        weight_path = checkpoint_file(directory, explicit)
        if weight_path.name not in {"model.safetensors", "pytorch_model.bin"}:
            raise EncoderError(
                "Phikon bundles must use the standard model.safetensors or pytorch_model.bin filename"
            )
        try:
            self.processor = transformers.AutoImageProcessor.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False
            )
            kwargs = {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": weight_path.suffix == ".safetensors",
                "weights_only": True,
            }
            if spec.model == "phikon":
                kwargs["add_pooling_layer"] = False
            self.model = transformers.AutoModel.from_pretrained(str(directory), **kwargs)
        except Exception:
            raise EncoderError(
                "Could not load the Phikon checkpoint and processor; verify compatible Transformers/PyTorch dependencies and matching safe weight files"
            ) from None
        allowed_keys = {
            "image_processor_type",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
        }
        self._finish(
            directory,
            revision,
            weight_path,
            {
                "kind": "AutoImageProcessor",
                "class": type(self.processor).__name__,
                "config": {
                    key: value for key, value in processor_config.items() if key in allowed_keys
                },
                "config_sha256": sha256(processor_path),
            },
        )

    def _forward(self, batch):
        return self.model(**batch).last_hidden_state[:, 0, :]
