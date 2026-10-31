"""Native torchvision/Transformers models behind one aligned sample contract.

The engine resizes/augments RGB and labels together. These adapters only scale
uint8 RGB to [0,1], normalize, and pad. Dataset class zero remains background;
ADE label-reduction and processor resizing are deliberately disabled.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .base import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationAdapter,
    preprocess_batch,
    validate_targets,
)


def _backend(directory):
    return Path(directory) / "backend"


def _array(tensor):
    return tensor.detach().float().cpu().numpy()


class _StandardAdapter(SegmentationAdapter):
    def _normalization(self, processor=None, load_dir=None):
        self.mean = list(getattr(processor, "image_mean", IMAGENET_MEAN))
        self.std = list(getattr(processor, "image_std", IMAGENET_STD))
        if load_dir is not None:
            path = _backend(load_dir) / "preprocessing.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing backend preprocessing: {path}")
            data = json.loads(path.read_text())
            self.mean, self.std = data["mean"], data["std"]
        else:
            self.mean = list(self.config.options.get("image_mean", self.mean))
            self.std = list(self.config.options.get("image_std", self.std))

    def _batch(self, samples, training=False):
        result = preprocess_batch(
            samples,
            device=self.device,
            mean=self.mean,
            std=self.std,
            ignore_index=self.config.ignore_index,
            require_labels=training,
        )
        if training:
            validate_targets(result[1], self.config)
        return result

    @contextmanager
    def _batch_norm_mode(self, batch_size):
        """Keep singleton pooled features trainable without discarding samples.

        Auto uses running BatchNorm statistics for singleton DeepLab/UPerNet
        training forwards. Parameter gradients remain enabled. Every module's
        previous mode is restored even when the forward fails.
        """
        setting = self.config.options.get("freeze_batch_norm", "auto")
        if not isinstance(setting, bool) and setting != "auto":
            raise ValueError('freeze_batch_norm must be true, false, or "auto"')
        freeze = setting is True or (
            setting == "auto"
            and self.model.training
            and batch_size == 1
            and isinstance(self, (DeepLabV3Adapter, UPerNetAdapter))
        )
        states = []
        if freeze:
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    states.append((module, module.training))
                    module.eval()
        try:
            yield
        finally:
            for module, training in states:
                module.training = training

    @contextmanager
    def _inference_mode(self):
        states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            yield
        finally:
            for module, training in states:
                module.training = training

    def _save_normalization(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preprocessing.json").write_text(
            json.dumps(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "rescale_factor": 1 / 255,
                    "resize": False,
                    "reduce_labels": False,
                },
                indent=2,
            )
            + "\n"
        )

    def _dense_logits(self, pixel_values):
        output = self.model(pixel_values=pixel_values)
        return output.logits

    def loss(self, samples):
        images, labels, _, _ = self._batch(samples, training=True)
        with self._batch_norm_mode(len(samples)):
            logits = self._dense_logits(images)
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(logits, labels, ignore_index=self.config.ignore_index)

    @torch.no_grad()
    def infer(self, samples):
        images, _, sizes, _ = self._batch(samples)
        with self._inference_mode():
            logits = self._dense_logits(images)
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return [
            {
                "probabilities": _array(value[:, :h, :w].softmax(0)),
                "scores": _array(value[:, :h, :w]),
                "logits": _array(value[:, :h, :w]),
                "score_semantics": self.score_semantics,
            }
            for value, (h, w) in zip(logits, sizes)
        ]


class DeepLabV3Adapter(_StandardAdapter):
    """Torchvision DeepLabV3; checkpoint reload never requests pretrained weights."""

    def __init__(self, config, device="cpu", load_dir=None):
        super().__init__(config, device)
        from torchvision.models import segmentation as seg

        backbone = config.options.get("backbone", "resnet50")
        choices = {
            "resnet50": (seg.deeplabv3_resnet50, seg.DeepLabV3_ResNet50_Weights),
            "resnet101": (seg.deeplabv3_resnet101, seg.DeepLabV3_ResNet101_Weights),
            "mobilenet_v3_large": (
                seg.deeplabv3_mobilenet_v3_large,
                seg.DeepLabV3_MobileNet_V3_Large_Weights,
            ),
        }
        if backbone not in choices:
            raise ValueError(f"Unsupported DeepLabV3 backbone {backbone!r}")
        factory, weight_type = choices[backbone]
        aux_loss = bool(config.options.get("aux_loss", False))
        if load_dir is None and config.pretrained:
            self.model = factory(weights=weight_type.DEFAULT, weights_backbone=None)
            self.model.classifier[-1] = nn.Conv2d(
                self.model.classifier[-1].in_channels, config.num_classes, 1
            )
            if not aux_loss:
                self.model.aux_classifier = None
            elif self.model.aux_classifier is not None:
                self.model.aux_classifier[-1] = nn.Conv2d(
                    self.model.aux_classifier[-1].in_channels, config.num_classes, 1
                )
        else:
            self.model = factory(
                weights=None,
                weights_backbone=None,
                num_classes=config.num_classes,
                aux_loss=aux_loss,
            )
        if load_dir is not None:
            payload = torch.load(
                _backend(load_dir) / "model.pt", map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(payload["state_dict"], strict=True)
        self._normalization(load_dir=load_dir)
        self.to(device)

    def _dense_logits(self, pixel_values):
        return self.model(pixel_values)["out"]

    def loss(self, samples):
        images, labels, _, _ = self._batch(samples, training=True)
        with self._batch_norm_mode(len(samples)):
            outputs = self.model(images)
        logits = F.interpolate(
            outputs["out"], size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = F.cross_entropy(logits, labels, ignore_index=self.config.ignore_index)
        if "aux" in outputs:
            aux = F.interpolate(
                outputs["aux"], size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss + float(self.config.options.get("aux_loss_weight", 0.4)) * F.cross_entropy(
                aux, labels, ignore_index=self.config.ignore_index
            )
        return loss

    def save(self, directory):
        target = _backend(directory)
        self._save_normalization(target)
        torch.save({"state_dict": self.model.state_dict()}, target / "model.pt")


class _HFAdapter(_StandardAdapter):
    model_class = ""
    config_class = ""
    processor_class = ""
    default_pretrained = ""

    def __init__(self, config, device="cpu", load_dir=None):
        super().__init__(config, device)
        try:
            import transformers as hf
        except ImportError as exc:
            raise ImportError(
                'This backend requires the segmentation-transformers extra: pip install "openhisto[segmentation-transformers]"'
            ) from exc
        model_cls = getattr(hf, self.model_class)
        config_cls = getattr(hf, self.config_class)
        processor_cls = getattr(hf, self.processor_class)
        io_options = {
            key: config.options[key]
            for key in ("cache_dir", "revision", "local_files_only")
            if key in config.options
        }
        if load_dir is not None:
            target = _backend(load_dir)
            self.model = model_cls.from_pretrained(target, local_files_only=True)
            self.processor = processor_cls.from_pretrained(target, local_files_only=True)
            if self.model.config.num_labels != config.num_classes:
                raise ValueError(
                    "Backend checkpoint class count disagrees with bundle configuration"
                )
        elif config.pretrained and not (
            self.model_class == "UperNetForSemanticSegmentation"
            and config.model_name_or_path is None
        ):
            source = config.model_name_or_path or self.default_pretrained
            self.processor = hf.AutoImageProcessor.from_pretrained(source, **io_options)
            self.model = model_cls.from_pretrained(
                source, num_labels=config.num_classes, ignore_mismatched_sizes=True, **io_options
            )
        else:
            architecture = dict(config.options.get("architecture_config", {}))
            if config.model_name_or_path:
                # Random initialization never performs a configuration download.
                # A named architecture can be used after explicitly preparing it
                # in the HF cache, or the native JSON config can be supplied.
                local_options = {**io_options, "local_files_only": True}
                native_config = config_cls.from_pretrained(
                    config.model_name_or_path, **local_options
                )
                for key, value in architecture.items():
                    setattr(native_config, key, value)
            else:
                native_config = config_cls(**architecture)
            native_config.num_labels = config.num_classes
            native_config.id2label = {i: str(i) for i in range(config.num_classes)}
            native_config.label2id = {str(i): i for i in range(config.num_classes)}
            if getattr(native_config, "use_pretrained_backbone", False):
                raise ValueError("pretrained=False requires use_pretrained_backbone=False")
            self.model = model_cls(native_config)
            self.processor = processor_cls(do_resize=False, do_reduce_labels=False)
        # Processors are saved for provenance/interoperability, but only their
        # normalization is applied. Label reduction must never erase class zero.
        self.processor.do_resize = False
        self.processor.do_reduce_labels = False
        if hasattr(self.processor, "reduce_labels"):
            self.processor.reduce_labels = False
        self._normalization(self.processor, load_dir)
        self.to(device)

    def save(self, directory):
        target = _backend(directory)
        self._save_normalization(target)
        self.model.save_pretrained(target, safe_serialization=True)
        self.processor.save_pretrained(target)


class SegFormerAdapter(_HFAdapter):
    model_class = "SegformerForSemanticSegmentation"
    config_class = "SegformerConfig"
    processor_class = "SegformerImageProcessor"
    default_pretrained = "nvidia/segformer-b5-finetuned-ade-640-640"


class UPerNetAdapter(_HFAdapter):
    model_class = "UperNetForSemanticSegmentation"
    config_class = "UperNetConfig"
    processor_class = "SegformerImageProcessor"
    default_pretrained = "openmmlab/upernet-swin-tiny"


def _masked_query_loss(classes, masks, labels, config):
    """Hungarian class/BCE/Dice objective evaluated only at labeled pixels.

    Unlike HF's native criterion, this excludes ignore and padded pixels both
    from matching costs and mask losses. Uniform valid-pixel samples replace
    unrestricted continuous point sampling for this partial-annotation path.
    """
    from scipy.optimize import linear_sum_assignment

    query_classes, query_targets = [], []
    total_bce = masks.sum() * 0
    total_dice = masks.sum() * 0
    count = 0
    no_object = config.num_labels
    weights = classes.new_ones(no_object + 1)
    weights[-1] = config.no_object_weight
    for cls, pred, target in zip(classes, masks, labels):
        valid = target != config.ignore_value
        if not valid.any():
            continue
        ids = torch.unique(target[valid])
        locations = valid.flatten().nonzero().flatten()
        if locations.numel() > config.train_num_points:
            locations = locations[
                torch.randperm(locations.numel(), device=locations.device)[
                    : config.train_num_points
                ]
            ]
        height, width = target.shape
        ys = torch.div(locations, width, rounding_mode="floor")
        xs = locations % width
        coordinates = torch.stack(((xs + 0.5) / width, (ys + 0.5) / height), dim=-1)
        # Border padding matches align_corners=False interpolation, including
        # pixel centers close to the low-resolution query mask's outer edges.
        grid = (coordinates.to(pred.dtype) * 2 - 1)[None, None]
        pred = F.grid_sample(
            pred[None], grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, 0]
        truth = (target.flatten()[locations][None] == ids[:, None]).float()
        with torch.no_grad():
            positive, negative = F.softplus(-pred), F.softplus(pred)
            bce_cost = (positive @ truth.T + negative @ (1 - truth).T) / locations.numel()
            probs = pred.sigmoid()
            dice_cost = 1 - (2 * (probs @ truth.T) + 1) / (
                probs.sum(1)[:, None] + truth.sum(1)[None] + 1
            )
            cost = (
                -cls.softmax(-1)[:, ids]
                + config.mask_weight * bce_cost
                + config.dice_weight * dice_cost
            )
            source, destination = linear_sum_assignment(cost.float().cpu().numpy())
            source = torch.as_tensor(source, device=pred.device, dtype=torch.long)
            destination = torch.as_tensor(destination, device=pred.device, dtype=torch.long)
        targets = torch.full((cls.shape[0],), no_object, device=cls.device, dtype=torch.long)
        targets[source] = ids[destination]
        query_classes.append(cls)
        query_targets.append(targets)
        matched, truth = pred[source], truth[destination]
        total_bce = (
            total_bce
            + F.binary_cross_entropy_with_logits(matched, truth, reduction="none").mean(1).sum()
        )
        probs = matched.sigmoid()
        total_dice = (
            total_dice
            + (1 - (2 * (probs * truth).sum(1) + 1) / (probs.sum(1) + truth.sum(1) + 1)).sum()
        )
        count += len(ids)
    # The native criterion reduces over every query in the batch, with one
    # shared weighted denominator. Averaging per-image weighted CE would change
    # class/no-object weighting when images contain different class counts.
    total_class = (
        F.cross_entropy(
            torch.stack(query_classes).transpose(1, 2), torch.stack(query_targets), weight=weights
        )
        if query_classes
        else classes.sum() * 0
    )
    return (
        config.class_weight * total_class
        + config.mask_weight * total_bce / max(count, 1)
        + config.dice_weight * total_dice / max(count, 1)
    )


class Mask2FormerAdapter(_HFAdapter):
    model_class = "Mask2FormerForUniversalSegmentation"
    config_class = "Mask2FormerConfig"
    processor_class = "Mask2FormerImageProcessor"
    default_pretrained = "facebook/mask2former-swin-tiny-ade-semantic"
    supports_logits = False
    score_semantics = "summed_query_class_mask_scores"

    def loss(self, samples):
        images, labels, _, pixel_mask = self._batch(samples, training=True)
        # All-ignore samples contribute no query-class or mask supervision.
        keep = (labels != self.config.ignore_index).flatten(1).any(1)
        images, labels, pixel_mask = images[keep], labels[keep], pixel_mask[keep]
        native = self.model.config
        native.ignore_value = self.config.ignore_index
        if (labels == self.config.ignore_index).any():
            output = self.model(
                pixel_values=images,
                pixel_mask=pixel_mask,
                output_auxiliary_logits=bool(native.use_auxiliary_loss),
            )
            loss = _masked_query_loss(
                output.class_queries_logits, output.masks_queries_logits, labels, native
            )
            for auxiliary in output.auxiliary_logits or []:
                loss = loss + _masked_query_loss(
                    auxiliary["class_queries_logits"],
                    auxiliary["masks_queries_logits"],
                    labels,
                    native,
                )
            return loss
        classes = [torch.unique(label) for label in labels]
        masks = [(label[None] == ids[:, None, None]).float() for label, ids in zip(labels, classes)]
        return self.model(
            pixel_values=images, pixel_mask=pixel_mask, mask_labels=masks, class_labels=classes
        ).loss

    @torch.no_grad()
    def infer(self, samples):
        images, _, sizes, pixel_mask = self._batch(samples)
        with self._inference_mode():
            output = self.model(pixel_values=images, pixel_mask=pixel_mask)
        classes = output.class_queries_logits.softmax(-1)[..., :-1]
        chunk_size = self.config.options.get("query_chunk_size", 16)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        scores = images.new_zeros((len(samples), self.config.num_classes, *images.shape[-2:]))
        # Preserve the native upsample-logits -> sigmoid -> query summation
        # ordering without allocating a full [B,Q,H,W] high-resolution tensor.
        for start in range(0, classes.shape[1], chunk_size):
            stop = start + chunk_size
            masks = F.interpolate(
                output.masks_queries_logits[:, start:stop],
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            scores.add_(torch.einsum("bqc,bqhw->bchw", classes[:, start:stop], masks))
        result = []
        for value, (h, w) in zip(scores, sizes):
            value = value[:, :h, :w]
            denominator = value.sum(0, keepdim=True)
            probabilities = torch.where(
                denominator > 0,
                value / denominator.clamp_min(1e-20),
                torch.full_like(value, 1 / self.config.num_classes),
            )
            result.append(
                {
                    "probabilities": _array(probabilities),
                    "scores": _array(value),
                    "logits": None,
                    "score_semantics": self.score_semantics,
                }
            )
        return result
