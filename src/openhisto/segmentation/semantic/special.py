"""Optional SAM2 and local embedding-conditioned SegDiff adapters."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

from .base import SegmentationAdapter
from ..config import CapabilityError


SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def load_embedding(sample_or_spec, *, expected_dim=None):
    """Read exactly one finite embedding, with explicit identity/coordinates.

    Specifications: a vector; NPY/NPZ ``path``; ``kind=feature_store`` with a
    FeatureReader model/mag and row (global position or observation ID); or
    ``kind=legacy`` HDF5 with explicit row or xy. Never guess from filenames.
    """
    sample = sample_or_spec if hasattr(sample_or_spec, "embedding") else None
    spec = sample.embedding if sample is not None else sample_or_spec
    if spec is None:
        raise ValueError(
            "SegDiff requires one explicit embedding or feature-store reference for every sample"
        )
    if isinstance(spec, dict):
        if "vector" in spec:
            value = spec["vector"]
        else:
            path = Path(spec.get("path", spec.get("feature_store", "")))
            if not path.is_file():
                raise FileNotFoundError(f"Embedding file does not exist: {path}")
            kind = spec.get(
                "kind",
                "feature_store"
                if path.suffix == ".h5ad"
                else "legacy"
                if path.suffix in {".h5", ".hdf5"}
                else "array",
            )
            if kind == "feature_store":
                try:
                    from openhisto.feature_encoder import FeatureReader
                except ImportError as exc:
                    raise ImportError(
                        'Feature-store embeddings require the feature-encoder extra: pip install "openhisto[feature-encoder]"'
                    ) from exc
                if "model" not in spec or "mag" not in spec:
                    raise ValueError("feature_store embedding requires model and mag")
                with FeatureReader(path) as reader:
                    row = spec.get("row")
                    if row is None:
                        xy = spec.get("xy", sample.xy_level0 if sample is not None else None)
                        if xy is None:
                            raise ValueError(
                                "feature_store embedding requires an explicit row or xy"
                            )
                        meta = reader.patch_metadata(mag=spec["mag"])
                        matches = np.isclose(
                            meta["x_level0"].to_numpy(), xy[0], atol=1e-6, rtol=0
                        ) & np.isclose(meta["y_level0"].to_numpy(), xy[1], atol=1e-6, rtol=0)
                        if int(matches.sum()) != 1:
                            raise ValueError(
                                "Embedding coordinates must identify exactly one feature-store row"
                            )
                        row = str(meta.index[np.flatnonzero(matches)[0]])
                    value = reader.features(
                        spec["model"],
                        mag=spec["mag"],
                        run_key=spec.get("run_key"),
                        rows=[row],
                        require_complete=True,
                    )
            elif kind == "legacy":
                try:
                    import h5py
                except ImportError as exc:
                    raise ImportError("Legacy HDF5 embeddings require h5py") from exc
                with h5py.File(path, "r") as handle:
                    features = handle[spec.get("features_key", "features/20x")]
                    row = spec.get("row")
                    if row is None:
                        xy = spec.get("xy", sample.xy_level0 if sample is not None else None)
                        if xy is None:
                            raise ValueError(
                                "Legacy embeddings require explicit row or xy coordinates"
                            )
                        coordinates = np.asarray(handle[spec.get("coords_key", "coords/20x")])
                        matches = np.flatnonzero(
                            np.isclose(coordinates[:, :2], np.asarray(xy), rtol=0, atol=1e-6).all(1)
                        )
                        if len(matches) != 1:
                            raise ValueError(
                                "Embedding coordinates must identify exactly one legacy feature row"
                            )
                        row = int(matches[0])
                    if (
                        isinstance(row, bool)
                        or not isinstance(row, (int, np.integer))
                        or not 0 <= row < len(features)
                    ):
                        raise ValueError("Legacy embedding row must be a valid nonnegative integer")
                    value = features[int(row)]
            elif kind == "array":
                if path.suffix == ".npy":
                    value = np.load(path, allow_pickle=False)
                elif path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        key = spec.get("key")
                        if key is None and len(archive.files) == 1:
                            key = archive.files[0]
                        if key not in archive:
                            raise ValueError("NPZ embedding requires an unambiguous key")
                        value = archive[key]
                else:
                    raise ValueError("Array embedding files must be NPY or NPZ")
                if "row" in spec:
                    value = value[spec["row"]]
            else:
                raise ValueError(f"Unknown embedding kind: {kind}")
    else:
        value = spec.detach().cpu().numpy() if torch.is_tensor(spec) else spec
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("An embedding must be one nonempty finite feature vector")
    if expected_dim is not None and value.size != expected_dim:
        raise ValueError(
            f"Embedding dimension {value.size} differs from configured feature_dim {expected_dim}"
        )
    return np.ascontiguousarray(value)


class SegDiffAdapter(SegmentationAdapter):
    supports_logits = True
    supports_training = True
    score_semantics = "Final diffusion logits; probabilities are their class-axis softmax, conditional on the supplied patch embedding."

    def __init__(self, config, device="cpu", load_dir=None):
        super().__init__(config, device)
        from .segdiff import SegDiffModel

        self.model = SegDiffModel(config.num_classes, config.options)
        if load_dir is not None:
            self.model.load_state_dict(
                torch.load(Path(load_dir) / "segdiff.pt", map_location="cpu", weights_only=True)[
                    "model"
                ]
            )
        self.to(device)

    def _features(self, sample):
        return (
            torch.from_numpy(
                load_embedding(sample, expected_dim=self.config.options["feature_dim"])
            )
            .unsqueeze(0)
            .to(self.device)
        )

    def _size(self, image):
        h, w = np.asarray(image).shape[:2]
        stride = self.model.stride
        return h, w, math.ceil(h / stride) * stride, math.ceil(w / stride) * stride

    def loss(self, samples):
        targets, features, sizes = [], [], []
        for sample in samples:
            if sample.mask is None:
                raise ValueError("SegDiff loss requires labeled masks")
            h, w, hp, wp = self._size(sample.image)
            y = torch.as_tensor(np.asarray(sample.mask), dtype=torch.long, device=self.device)
            if tuple(y.shape) != (h, w):
                raise ValueError("SegDiff image and mask sizes differ")
            if not y.ne(self.config.ignore_index).any():
                raise ValueError("SegDiff training mask contains no valid labeled pixels")
            targets.append(y)
            features.append(self._features(sample)[0])
            sizes.append((hp, wp))
        if not targets:
            raise ValueError("SegDiff loss requires a nonempty batch")
        hp, wp = max(h for h, _ in sizes), max(w for _, w in sizes)
        targets = torch.stack(
            [
                F.pad(y, (0, wp - y.shape[1], 0, hp - y.shape[0]), value=self.config.ignore_index)
                for y in targets
            ]
        )
        # Preserve the source's batch-global CE/MSE and class-wise Dice reduction.
        return self.model.loss(targets, torch.stack(features), self.config.ignore_index)

    @torch.no_grad()
    def infer(self, samples):
        results = []
        for sample in samples:
            h, w, hp, wp = self._size(sample.image)
            generator = None
            if "inference_seed" in self.config.options:
                generator = torch.Generator(
                    device=self.device if self.device.type == "cuda" else "cpu"
                ).manual_seed(int(self.config.options["inference_seed"]))
            logits = self.model.sample(self._features(sample), (hp, wp), generator=generator)[
                0, :, :h, :w
            ]
            results.append(
                {
                    "logits": logits.float().cpu().numpy(),
                    "scores": logits.float().cpu().numpy(),
                    "probabilities": logits.softmax(0).float().cpu().numpy(),
                    "score_semantics": self.score_semantics,
                }
            )
        return results

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, directory / "segdiff.pt")


def _sam2_training_class():
    try:
        from training.model.sam2 import SAM2Train
    except ImportError as exc:
        raise ImportError(
            "SAM2 training requires the segmentation-sam2 extra and the pinned source installation: "
            "pip install --no-build-isolation -r src/openhisto/segmentation/requirements-sam2.txt"
        ) from exc

    class PromptedSAM2Train(SAM2Train):
        def prepare_prompt_inputs(self, backbone_out, inputs, start_frame_idx=0):
            out = super().prepare_prompt_inputs(backbone_out, inputs, start_frame_idx)
            explicit = getattr(self, "_segmentation_prompts", None)
            if explicit is not None:
                out["use_pt_input"] = True
                out["mask_inputs_per_frame"] = {}
                out["point_inputs_per_frame"] = {0: explicit}
                out["frames_to_add_correction_pt"] = []
            return out

    return PromptedSAM2Train


class SAM2Adapter(SegmentationAdapter):
    supports_logits = True
    supports_training = True
    score_semantics = "Binary foreground sigmoid score from the best mask per supplied prompt group, merged by maximum; class probabilities are [1-p,p]."

    def __init__(self, config, device="cpu", load_dir=None):
        super().__init__(config, device)
        if config.num_classes != 2:
            raise CapabilityError(
                "SAM2 supports binary prompted semantic segmentation; num_classes must be 2"
            )
        try:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 requires the segmentation-sam2 extra and the pinned source installation: "
                "pip install --no-build-isolation -r src/openhisto/segmentation/requirements-sam2.txt"
            ) from exc
        train_cls = _sam2_training_class()
        checkpoint = None
        if load_dir is not None:
            self.architecture = json.loads((Path(load_dir) / "sam2_architecture.json").read_text())
            checkpoint = Path(load_dir) / "sam2.pt"
        elif "architecture" in config.options:
            architecture = config.options["architecture"]
            self.architecture = dict(architecture.get("model", architecture))
            checkpoint = config.options.get("checkpoint")
        else:
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

            name = config.model_name_or_path or "facebook/sam2-hiera-large"
            selected = config.options.get("sam2_config")
            checkpoint = config.options.get("checkpoint")
            if "::" in name:
                checkpoint, selected = name.split("::", 1)
            elif name in HF_MODEL_ID_TO_FILENAMES:
                default_config, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[name]
                selected = selected or default_config
                if config.pretrained and checkpoint is None:
                    from huggingface_hub import hf_hub_download

                    checkpoint = hf_hub_download(name, checkpoint_name)
            elif Path(name).is_file():
                checkpoint = checkpoint or name
            if selected is None:
                raise ValueError(
                    "Local SAM2 weights require options.sam2_config (official config name or YAML path)"
                )
            selected_path = Path(selected)
            if selected_path.is_file():
                cfg = OmegaConf.load(selected_path)
            else:
                resource = resources.files("sam2").joinpath(selected)
                if not resource.is_file():
                    raise ValueError(f"SAM2 architecture config does not exist: {selected}")
                cfg = OmegaConf.create(resource.read_text())
            OmegaConf.resolve(cfg)
            self.architecture = OmegaConf.to_container(cfg.get("model", cfg), resolve=True)
            if config.pretrained and checkpoint is None:
                raise ValueError(
                    "SAM2 pretrained initialization requires a checkpoint or known official model ID"
                )
        # Keep a fully resolved, portable model configuration. The custom subclass
        # only injects explicit prompts; all differentiable model computation is official.
        if load_dir is None and config.pretrained and checkpoint is None:
            raise ValueError(
                "SAM2 pretrained initialization requires weights; set pretrained=False for a random architecture"
            )
        self.architecture["_target_"] = "training.model.sam2.SAM2Train"
        self.architecture.update(
            num_correction_pt_per_frame=0,
            rand_init_cond_frames_for_train=False,
            num_init_cond_frames_for_train=1,
            num_init_cond_frames_for_eval=1,
            num_frames_to_correct_for_train=1,
            num_frames_to_correct_for_eval=1,
            freeze_image_encoder=bool(config.options.get("freeze_image_encoder", False)),
        )
        model_cfg = dict(self.architecture)
        model_cfg.pop("_target_", None)
        self.model = train_cls(**instantiate(model_cfg, _recursive_=True, _convert_="all"))
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        resolution = int(self.model.image_size)
        if resolution < 32 or resolution % 16:
            raise ValueError("SAM2 image_size must be >=32 and divisible by16")
        self.to(device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.predictor._bb_feat_sizes = [
            (resolution // 4, resolution // 4),
            (resolution // 8, resolution // 8),
            (resolution // 16, resolution // 16),
        ]

    def _groups(self, sample, training=False):
        prompts = sample.prompts
        h, w = np.asarray(sample.image).shape[:2]
        if not prompts:
            if not training:
                raise CapabilityError(
                    "SAM2 prediction requires explicit point or box prompts for every image"
                )
            mask = np.asarray(sample.mask)
            valid = mask != self.config.ignore_index
            foreground = (mask == 1) & valid
            ys, xs = np.where(foreground if foreground.any() else valid)
            if not len(xs):
                raise ValueError("SAM2 training mask contains no valid pixels")
            chosen = int(torch.randint(len(xs), ()).item()) if self.model.training else len(xs) // 2
            prompts = {
                "point_coords": [[float(xs[chosen]), float(ys[chosen])]],
                "point_labels": [int(foreground.any())],
            }
            if foreground.any():
                prompts["box"] = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ]
        if "groups" in prompts:
            groups = prompts["groups"]
            if not isinstance(groups, list) or not groups:
                raise ValueError("SAM2 prompt groups must be a nonempty list")
        else:
            boxes = prompts.get("boxes", prompts.get("box"))
            if boxes is not None and np.asarray(boxes).ndim == 2:
                if not len(boxes):
                    raise ValueError("SAM2 boxes must be nonempty")
                groups = [
                    {**{k: v for k, v in prompts.items() if k not in {"box", "boxes"}}, "box": b}
                    for b in boxes
                ]
            else:
                groups = [prompts]
        cleaned = []
        for group in groups:
            box = group.get("box", group.get("boxes"))
            points = group.get("point_coords", group.get("points"))
            labels = group.get("point_labels", group.get("labels"))
            if box is None and points is None:
                raise ValueError("Each SAM2 prompt group requires points or a box")
            if box is not None:
                box = np.asarray(box, dtype=np.float32)
                if (
                    box.shape != (4,)
                    or not np.isfinite(box).all()
                    or not (0 <= box[0] <= box[2] <= w and 0 <= box[1] <= box[3] <= h)
                ):
                    raise ValueError("SAM2 box must be finite XYXY coordinates inside the image")
            if points is not None:
                points, labels = np.asarray(points, dtype=np.float32), np.asarray(labels)
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or labels.shape != (len(points),)
                    or not len(points)
                    or not np.isfinite(points).all()
                    or not np.isin(labels, [0, 1]).all()
                ):
                    raise ValueError(
                        "SAM2 points require finite Nx2 coordinates and N binary point_labels"
                    )
                if not (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < w)
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < h)
                ).all():
                    raise ValueError("SAM2 point coordinates must lie inside the image")
                labels = labels.astype(np.int32)
            cleaned.append({"box": box, "point_coords": points, "point_labels": labels})
        return cleaned

    def loss(self, samples):
        from training.utils.data_utils import Object, Frame, VideoDatapoint, collate_fn

        losses = []
        for sample in samples:
            if sample.mask is None:
                raise ValueError("SAM2 fine-tuning requires binary ground-truth masks")
            mask = torch.as_tensor(np.asarray(sample.mask), device=self.device, dtype=torch.long)
            valid = mask.ne(self.config.ignore_index)
            if (
                not valid.any()
                or not torch.isin(mask[valid], torch.tensor([0, 1], device=self.device)).all()
            ):
                raise ValueError("SAM2 training masks must contain valid binary class IDs")
            r = int(self.model.image_size)
            image = self.predictor._transforms(np.asarray(sample.image)).to(self.device)
            target = F.interpolate((mask == 1)[None, None].float(), size=(r, r), mode="nearest")[
                0, 0
            ].bool()
            valid_r = F.interpolate(valid[None, None].float(), size=(r, r), mode="nearest")[
                0, 0
            ].bool()
            if not valid_r.any():
                raise ValueError("No valid SAM2 training pixels remain at model resolution")
            batch = collate_fn(
                [
                    VideoDatapoint(
                        frames=[
                            Frame(
                                data=image,
                                objects=[Object(object_id=1, frame_index=0, segment=target)],
                            )
                        ],
                        video_id=0,
                        size=(r, r),
                    )
                ],
                dict_key="segmentation",
            )
            h, w = np.asarray(sample.image).shape[:2]
            union = None
            try:
                for prompt in self._groups(sample, training=True):
                    coords, labels = [], []
                    if prompt["box"] is not None:
                        coords.extend(np.asarray(prompt["box"]).reshape(2, 2).tolist())
                        labels.extend([2, 3])
                    if prompt["point_coords"] is not None:
                        coords.extend(prompt["point_coords"].tolist())
                        labels.extend(prompt["point_labels"].tolist())
                    coords = torch.tensor(coords, device=self.device, dtype=torch.float32)[None]
                    coords[..., 0] *= r / w
                    coords[..., 1] *= r / h
                    self.model._segmentation_prompts = {
                        "point_coords": coords,
                        "point_labels": torch.tensor(labels, device=self.device, dtype=torch.int32)[
                            None
                        ],
                    }
                    output = self.model(batch)[0]["pred_masks_high_res"]
                    union = output if union is None else torch.maximum(union, output)
            finally:
                self.model._segmentation_prompts = None
            truth = target.float()[None, None]
            v = valid_r[None, None].float()
            bce = (
                F.binary_cross_entropy_with_logits(union, truth, reduction="none") * v
            ).sum() / v.sum()
            probs = union.sigmoid() * v
            dice = 1 - (2 * (probs * truth).sum() + 1e-6) / (probs.sum() + (truth * v).sum() + 1e-6)
            losses.append(bce + dice)
        if not losses:
            raise ValueError("SAM2 loss requires a nonempty batch")
        return torch.stack(losses).mean()

    @torch.no_grad()
    def infer(self, samples):
        results = []
        for sample in samples:
            groups = self._groups(sample)
            self.predictor.set_image(np.asarray(sample.image))
            foreground_logits = None
            for prompt in groups:
                masks, qualities, _ = self.predictor.predict(
                    **prompt, multimask_output=True, return_logits=True
                )
                logits = np.asarray(masks[np.argmax(qualities)], dtype=np.float32)
                foreground_logits = (
                    logits if foreground_logits is None else np.maximum(foreground_logits, logits)
                )
            foreground = torch.from_numpy(foreground_logits).sigmoid().numpy()
            probs = np.stack((1 - foreground, foreground)).astype(np.float32)
            logits = np.stack((np.zeros_like(foreground_logits), foreground_logits))
            results.append(
                {
                    "probabilities": probs,
                    "scores": probs,
                    "logits": logits,
                    "score_semantics": self.score_semantics,
                }
            )
        return results

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sam2_architecture.json").write_text(json.dumps(self.architecture, indent=2))
        torch.save({"model": self.model.state_dict()}, directory / "sam2.pt")
