"""Shared training, evaluation, prediction and self-contained checkpoint bundles."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import os
import random
import shutil
import tempfile
import uuid
import numpy as np
from PIL import Image
from openhisto import _common
from .config import (
    ModelConfig,
    TrainConfig,
    PredictConfig,
    MaskSpec,
    Prediction,
    TrainingResult,
    CapabilityError,
    json_safe,
)


def _torch():
    import torch

    return torch


def resolve_device(device="auto"):
    return str(_common.resolve_device(_torch(), device))


def _nearest(mask, size):
    h, w = size
    ys = np.minimum((np.arange(h) * mask.shape[0] / h).astype(int), mask.shape[0] - 1)
    xs = np.minimum((np.arange(w) * mask.shape[1] / w).astype(int), mask.shape[1] - 1)
    return mask[ys[:, None], xs[None, :]]


def transform_sample(sample, image_size, *, augment=False):
    """Resize and optionally augment one aligned sample, including pixel prompts."""
    old_h, old_w = sample.image.shape[:2]
    h, w = image_size
    image = np.asarray(
        Image.fromarray(sample.image).resize((w, h), Image.Resampling.BILINEAR)
    ).copy()
    mask = None if sample.mask is None else _nearest(sample.mask, (h, w)).copy()
    prompts = None if sample.prompts is None else json.loads(json.dumps(json_safe(sample.prompts)))

    def update_prompts(value, flip=False):
        if not value:
            return
        for group in value.get("groups", []):
            update_prompts(group, flip=flip)
        for key in ("box", "boxes"):
            if value.get(key) is not None:
                boxes = np.asarray(value[key], dtype=float)
                if flip:
                    old = boxes.copy()
                    boxes[..., 0], boxes[..., 2] = w - 1 - old[..., 2], w - 1 - old[..., 0]
                else:
                    boxes *= np.array([w / old_w, h / old_h, w / old_w, h / old_h])
                value[key] = boxes.tolist()
        for key in ("points", "point_coords"):
            if value.get(key) is not None:
                points = np.asarray(value[key], dtype=float)
                if flip:
                    points[..., 0] = w - 1 - points[..., 0]
                else:
                    points *= [w / old_w, h / old_h]
                value[key] = points.tolist()

    update_prompts(prompts)
    if augment and np.random.random() < 0.5:
        image = image[:, ::-1].copy()
        if mask is not None:
            mask = mask[:, ::-1].copy()
        update_prompts(prompts, flip=True)
    if augment and np.random.random() < 0.2:
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-0.2, 0.2) * 255
        image = np.clip(image.astype(float) * alpha + beta, 0, 255).astype(np.uint8)
    return replace(sample, image=image, mask=mask, prompts=prompts)


class _Dataset:
    def __init__(self, items, config, augment=False):
        self.items, self.config, self.augment = items, config, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from .data import read_sample

        sample = read_sample(
            self.items[index],
            num_classes=self.config.num_classes,
            ignore_index=self.config.ignore_index,
        )
        if sample.mask is None:
            raise ValueError(f"Training/evaluation sample {sample.sample_id!r} has no annotation")
        if not np.any(sample.mask != self.config.ignore_index):
            raise ValueError(f"Sample {sample.sample_id!r} has no valid target pixels")
        transformed = transform_sample(sample, self.config.image_size, augment=self.augment)
        if not np.any(transformed.mask != self.config.ignore_index):
            raise ValueError(
                f"Sample {sample.sample_id!r} has no valid target pixels after resizing"
            )
        return transformed


def _collate(samples):
    return samples


def _worker_seed(_):
    seed = _torch().initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(items, config, settings, *, training=False):
    torch = _torch()
    augment = training and settings.augment and config.model.lower() != "segdiff"
    return torch.utils.data.DataLoader(
        _Dataset(items, config, augment),
        batch_size=settings.batch_size,
        shuffle=training,
        num_workers=settings.num_workers,
        collate_fn=_collate,
        worker_init_fn=_worker_seed,
    )


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(json_safe(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _publish(path, writer, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not overwrite):
        raise FileExistsError(path)
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + "-", dir=path.parent))
    backup = None
    try:
        writer(temporary)
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
            backup = path.with_name("." + path.name + "-old-" + uuid.uuid4().hex)
            os.replace(path, backup)
        try:
            os.replace(temporary, path)
        except BaseException:
            if backup is not None:
                os.replace(backup, path)
                backup = None
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def save_model(model, directory, *, training_state=None, overwrite=False):
    def write(path):
        model.save(path)
        spec = getattr(model, "mask_spec", None)
        _json_write(
            path / "bundle.json",
            {
                "schema_version": 1,
                "model_config": model.config.to_dict(),
                "mask_spec": None if spec is None else spec.to_dict(),
            },
        )
        if training_state is not None:
            _torch().save(training_state, path / "training_state.pt")

    _publish(directory, write, overwrite)
    return Path(directory)


def load_model(directory, *, device="auto"):
    from .semantic.registry import create_adapter

    path = Path(directory)
    metadata = json.loads((path / "bundle.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported segmentation checkpoint schema")
    model = create_adapter(
        ModelConfig(**metadata["model_config"]), device=resolve_device(device), load_dir=path
    )
    model.mask_spec = MaskSpec(**metadata["mask_spec"]) if metadata.get("mask_spec") else None
    return model


def prepare_model(config=None, destination=None, *, device="cpu", overwrite=False):
    """Explicitly construct/download a backend and optionally publish a local bundle."""
    from .semantic.registry import create_adapter

    config = (
        ModelConfig()
        if config is None
        else ModelConfig(**config)
        if isinstance(config, dict)
        else config
    )
    model = create_adapter(config, device=resolve_device(device))
    if destination is not None:
        save_model(model, destination, overwrite=overwrite)
    return model


def _items(data, config, mask_spec=None):
    from .data import prepare_dataset

    return prepare_dataset(
        data, mask_spec=mask_spec, num_classes=config.num_classes, ignore_index=config.ignore_index
    )


def _restore_array(values, shape):
    torch = _torch()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Backend score output must be finite [classes,height,width]")
    if values.shape[1:] != tuple(shape):
        values = torch.nn.functional.interpolate(
            torch.from_numpy(values)[None], size=shape, mode="bilinear", align_corners=False
        )[0].numpy()
    return values


def _prediction(model, native, result, settings):
    shape = native.image.shape[:2]
    scores = None if result.get("scores") is None else _restore_array(result["scores"], shape)
    logits = None if result.get("logits") is None else _restore_array(result["logits"], shape)
    if settings.logits and logits is None:
        raise CapabilityError(f"{model.config.model} does not expose native semantic logits")
    if settings.scores and scores is None:
        raise CapabilityError(f"{model.config.model} does not expose semantic scores")
    # Restore native scores before nonlinear normalization. Interpolating
    # probabilities independently can disagree with exported logit argmax.
    if logits is not None:
        probs = np.exp(logits - logits.max(axis=0, keepdims=True))
    elif scores is not None:
        probs = np.clip(scores, 0, None)
    else:
        probs = np.clip(_restore_array(result["probabilities"], shape), 0, None)
    if probs.shape[0] != model.config.num_classes:
        raise ValueError("Backend probability class count disagrees with model configuration")
    sums = probs.sum(axis=0, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Backend returned a pixel with no probability mass")
    probs /= sums
    labels = probs.argmax(0).astype(np.int64)
    return Prediction(
        labels=labels,
        sample_id=native.sample_id,
        probabilities=probs if settings.probabilities else None,
        scores=scores if settings.scores else None,
        logits=logits if settings.logits else None,
        image=native.image,
        slide_id=native.slide_id,
        xy_level0=native.xy_level0,
        footprint_level0=native.footprint_level0,
        valid_bounds=native.valid_bounds,
        class_names=model.config.class_names or [str(i) for i in range(model.config.num_classes)],
        score_semantics=result.get("score_semantics", ""),
    )


def predict(model, data, *, config=None, mask_spec=None):
    """Yield predictions incrementally; annotations are never passed to inference."""
    from .data import read_sample

    torch = _torch()
    settings = (
        PredictConfig()
        if config is None
        else PredictConfig(**config)
        if isinstance(config, dict)
        else config
    )
    if isinstance(model, (str, Path)):
        model = load_model(model, device=settings.device)
    else:
        model.to(resolve_device(settings.device))
    items = _items(data, model.config, mask_spec or getattr(model, "mask_spec", None))
    training_modes = [(module, module.training) for module in model.model.modules()]
    model.model.eval()
    try:
        for start in range(0, len(items), settings.batch_size):
            # Discard targets before adapter inference: SAM2 cannot derive GT prompts.
            native = [
                read_sample(
                    replace(item, mask=None),
                    num_classes=model.config.num_classes,
                    ignore_index=model.config.ignore_index,
                )
                for item in items[start : start + settings.batch_size]
            ]
            samples = [
                transform_sample(replace(s, mask=None), model.config.image_size) for s in native
            ]
            with torch.no_grad():
                results = model.infer(samples)
            if len(results) != len(samples):
                raise RuntimeError("Backend dropped one or more batch items")
            for sample, result in zip(native, results):
                yield _prediction(model, sample, result, settings)
    finally:
        for module, was_training in training_modes:
            module.training = was_training


class _Metrics:
    def __init__(self, num_classes, ignore_index):
        self.counts = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes, self.ignore_index = num_classes, ignore_index

    def update(self, labels, target):
        valid = target != self.ignore_index
        true, pred = np.asarray(target)[valid], np.asarray(labels)[valid]
        if not len(true):
            raise ValueError("No valid pixels for evaluation")
        if np.any((true < 0) | (true >= self.num_classes)) or np.any(
            (pred < 0) | (pred >= self.num_classes)
        ):
            raise ValueError("Evaluation labels are outside the class range")
        self.counts += np.bincount(
            true * self.num_classes + pred, minlength=self.num_classes**2
        ).reshape(self.counts.shape)

    def result(self):
        tp = np.diag(self.counts).astype(float)
        true, pred = self.counts.sum(1), self.counts.sum(0)
        union = true + pred - tp
        dice = np.divide(2 * tp, true + pred, out=np.full_like(tp, np.nan), where=(true + pred) > 0)
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)

        def mean(values):
            return float(np.nanmean(values)) if np.isfinite(values).any() else None

        return {
            "pixels": int(self.counts.sum()),
            "accuracy": float(tp.sum() / self.counts.sum()) if self.counts.sum() else None,
            "macro_dice": mean(dice),
            "macro_iou": mean(iou),
            "foreground_dice": mean(dice[1:]),
            "per_class": [
                {
                    "class_id": i,
                    "dice": None if np.isnan(dice[i]) else float(dice[i]),
                    "iou": None if np.isnan(iou[i]) else float(iou[i]),
                    "target_pixels": int(true[i]),
                }
                for i in range(self.num_classes)
            ],
        }


def evaluate(model, data, *, config=None, mask_spec=None):
    from .data import read_sample

    settings = (
        PredictConfig()
        if config is None
        else PredictConfig(**config)
        if isinstance(config, dict)
        else config
    )
    if isinstance(model, (str, Path)):
        model = load_model(model, device=settings.device)
    items = _items(data, model.config, mask_spec or getattr(model, "mask_spec", None))
    if not items:
        raise ValueError("Evaluation data is empty")
    metrics = _Metrics(model.config.num_classes, model.config.ignore_index)
    for item, prediction in zip(items, predict(model, items, config=settings)):
        sample = read_sample(
            item, num_classes=model.config.num_classes, ignore_index=model.config.ignore_index
        )
        if sample.mask is None:
            raise ValueError(f"Evaluation sample {sample.sample_id!r} has no annotation")
        metrics.update(prediction.labels, sample.mask)
    return metrics.result()


def _rng_state():
    torch = _torch()
    state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [state[0], torch.tensor(state[1].astype(np.int64)), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _set_rng(state):
    torch = _torch()
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state(
        (value[0], value[1].numpy().astype(np.uint32), value[2], value[3], value[4])
    )
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(
    train_data,
    val_data,
    output_dir,
    *,
    model_config=None,
    train_config=None,
    mask_spec=None,
    resume=None,
    model=None,
):
    torch = _torch()
    settings = (
        TrainConfig()
        if train_config is None
        else TrainConfig(**train_config)
        if isinstance(train_config, dict)
        else train_config
    )
    config = (
        ModelConfig()
        if model_config is None
        else ModelConfig(**model_config)
        if isinstance(model_config, dict)
        else model_config
    )
    mask_spec = MaskSpec(**mask_spec) if isinstance(mask_spec, dict) else mask_spec
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(
            f"Training output is not empty: {output}; use resume with a resumable checkpoint"
        )
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    device = resolve_device(settings.device)
    state = None
    if resume is not None:
        model = load_model(resume, device=device)
        config = model.config
        state = torch.load(
            Path(resume) / "training_state.pt", map_location="cpu", weights_only=True
        )
        saved = state["train_config"]
        for key in (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "lr_gamma",
            "seed",
            "augment",
            "num_workers",
        ):
            if saved[key] != getattr(settings, key):
                raise ValueError(
                    f"Resume changes {key}; only epoch limit, device and save cadence may change"
                )
        mask_spec = mask_spec or getattr(model, "mask_spec", None)
    if model is not None:
        config = model.config
        mask_spec = mask_spec or getattr(model, "mask_spec", None)
    train_items = _items(train_data, config, mask_spec)
    val_items = _items(val_data, config, mask_spec)
    if not train_items or not val_items:
        raise ValueError("Training and validation datasets must both be nonempty")
    # Validate every annotation before allocating a model or producing output.
    from .data import read_sample

    for item in train_items + val_items:
        sample = read_sample(item, num_classes=config.num_classes, ignore_index=config.ignore_index)
        if sample.mask is None or not np.any(sample.mask != config.ignore_index):
            raise ValueError(f"Sample {sample.sample_id!r} needs a nonempty valid target")
    if model is None:
        if config.model.lower() == "segdiff":
            from .semantic.special import load_embedding

            dims = {
                len(
                    load_embedding(
                        read_sample(
                            item, num_classes=config.num_classes, ignore_index=config.ignore_index
                        )
                    )
                )
                for item in train_items + val_items
            }
            if len(dims) != 1:
                raise ValueError("SegDiff embeddings must have one consistent dimension")
            dimension = dims.pop()
            if config.options.get("feature_dim", dimension) != dimension:
                raise ValueError("Configured feature_dim differs from supplied embeddings")
            config = replace(config, options={**config.options, "feature_dim": dimension})
        from .semantic.registry import create_adapter

        model = create_adapter(config, device=device)
    else:
        model.to(device)
    config = model.config
    model.mask_spec = mask_spec
    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings.lr_gamma)
    history, start, best = [], 0, float("inf")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer tensors follow parameters on the selected resume device.
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        scheduler.load_state_dict(state["scheduler"])
        history, start, best = state["history"], state["epoch"], state["best_val_loss"]
        _set_rng(state["rng"])
        if settings.num_epochs <= start:
            raise ValueError("num_epochs must exceed the completed resume epoch")
    train_loader = _loader(train_items, config, settings, training=True)
    val_loader = _loader(val_items, config, settings)
    output.mkdir(parents=True, exist_ok=True)
    _json_write(
        output / "run.json",
        {
            "model_config": config.to_dict(),
            "train_config": settings.to_dict(),
            "mask_spec": None if mask_spec is None else mask_spec.to_dict(),
        },
    )
    for epoch in range(start, settings.num_epochs):
        model.model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("Backend returned a nonfinite or nonscalar training loss")
            if not loss.requires_grad:
                raise RuntimeError("Backend training loss is detached from model parameters")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        model.model.eval()
        val_total, val_count = 0.0, 0
        metrics = _Metrics(config.num_classes, config.ignore_index)
        metrics_available = True
        with torch.no_grad():
            for batch in val_loader:
                loss = model.loss(batch)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite validation loss")
                val_total += float(loss.cpu()) * len(batch)
                val_count += len(batch)
                # A simulated GT prompt may measure loss, never segmentation accuracy.
                if config.model.lower() == "sam2" and any(not sample.prompts for sample in batch):
                    metrics_available = False
                    continue
                results = model.infer([replace(sample, mask=None) for sample in batch])
                if len(results) != len(batch):
                    raise RuntimeError("Backend dropped validation batch items")
                for sample, result in zip(batch, results):
                    prediction = np.asarray(result["probabilities"]).argmax(0)
                    metrics.update(prediction, sample.mask)
        val_loss = val_total / val_count
        record = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metrics": metrics.result() if metrics_available else None,
        }
        if not metrics_available:
            record["metrics_note"] = (
                "SAM2 validation accuracy requires explicit prompts; loss uses training prompt simulation."
            )
        history.append(record)
        scheduler.step()
        improved = val_loss < best
        best = min(best, val_loss)
        checkpoint = {
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_val_loss": best,
            "train_config": settings.to_dict(),
            "rng": _rng_state(),
        }
        save_model(model, output / "latest", training_state=checkpoint, overwrite=True)
        if improved:
            save_model(model, output / "best", training_state=checkpoint, overwrite=True)
        if (epoch + 1) % settings.save_every == 0:
            save_model(
                model,
                output / f"epoch_{epoch + 1}",
                training_state=checkpoint,
                overwrite=resume is not None,
            )
        _json_write(output / "history.json", history)
        print(
            f"Epoch {epoch + 1}/{settings.num_epochs}: train={total / count:.6f} val={val_loss:.6f}",
            flush=True,
        )
    save_model(model, output / "final", overwrite=resume is not None)
    return TrainingResult(str(output), history, settings.num_epochs, best)
