"""CycleGAN and paired Pix2Pix training with portable, resumable checkpoints."""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import TrainConfig, TrainReport
from .datasets import PairedDataset, UnpairedDataset
from .gan import checkpoint_metadata, grayscale_tensor, resolve_device
from .networks import discriminator, generator, initialize, spatial_requirements
from .networks.cyclegan import NNUNetStyleUNet

FORMAT = "stain_transfer.training.v1"
ALLOWED_OPTIONS = {
    "norm",
    "dropout",
    "pool_size",
    "color_jitter",
    "brightness",
    "contrast",
    "saturation",
    "beta1",
    "amp",
}


class ImagePool:
    def __init__(self, size=50):
        self.size = size
        self.images = []

    def query(self, batch):
        if not self.size:
            return batch.detach()
        result = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image.cpu().clone())
                result.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.size)
                previous = self.images[index].to(image.device)
                self.images[index] = image.cpu().clone()
                result.append(previous)
            else:
                result.append(image)
        return torch.cat(result)


def _rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _load(path):
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _validate_resume(state):
    message = "resume requires a full stain_transfer training checkpoint; use warm_start for legacy weights"
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError(message)
    required = {
        "config",
        "metadata",
        "networks",
        "optimizers",
        "schedulers",
        "scalers",
        "rng",
        "pools",
        "epochs_completed",
        "current_epoch",
        "next_batch",
        "global_step",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"Incomplete training checkpoint; missing: {', '.join(sorted(missing))}")
    for name in (
        "config",
        "metadata",
        "networks",
        "optimizers",
        "schedulers",
        "scalers",
        "rng",
        "pools",
    ):
        if not isinstance(state[name], dict):
            raise ValueError(f"Invalid training checkpoint: {name} must be a dictionary")
    for name in ("epochs_completed", "current_epoch", "next_batch", "global_step"):
        value = state[name]
        minimum = 1 if name == "current_epoch" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid training checkpoint counter: {name}")
    try:
        saved = TrainConfig(**state["config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training checkpoint configuration: {exc}") from exc
    network_names = {"G_A", "G_B", "D_A", "D_B"} if saved.method == "cyclegan" else {"G", "D"}
    optimizer_names = {"G", "D"}
    if saved.method == "cyclegan" and saved.lambda_mask:
        network_names.update({"S_A", "S_B"})
        optimizer_names.add("S")
    for group, expected in (
        ("networks", network_names),
        ("optimizers", optimizer_names),
        ("schedulers", optimizer_names),
        ("scalers", optimizer_names),
    ):
        if not expected.issubset(state[group]):
            raise ValueError(
                f"Incomplete training checkpoint {group}; expected {', '.join(sorted(expected))}"
            )
        if any(not isinstance(state[group][name], dict) for name in expected):
            raise ValueError(f"Invalid training checkpoint state dictionaries: {group}")
    if not {"python", "numpy", "torch", "cuda"}.issubset(state["rng"]):
        raise ValueError("Incomplete training checkpoint RNG state")
    if saved.method == "cyclegan" and not {"A", "B"}.issubset(state["pools"]):
        raise ValueError("Incomplete training checkpoint replay pools")
    return state


def _validate_warm_start(state, config):
    if not isinstance(state, dict):
        raise ValueError(
            "warm_start requires a generator state dictionary or a training checkpoint"
        )
    if state.get("format") != FORMAT:
        return state
    metadata, saved = state.get("metadata", {}), state.get("config", {})
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        raise ValueError("Invalid full warm_start checkpoint metadata/configuration")
    method = metadata.get("method", saved.get("method"))
    if not method or (
        metadata.get("method") and saved.get("method") and metadata["method"] != saved["method"]
    ):
        raise ValueError("Invalid full warm_start checkpoint method metadata")
    if method != config.method:
        raise ValueError(
            f"warm_start method {method!r} does not match training method {config.method!r}"
        )
    networks = state.get("networks")
    generators = {"G_A", "G_B"} if config.method == "cyclegan" else {"G"}
    if not isinstance(networks, dict) or not generators.intersection(networks):
        raise ValueError("Full warm_start checkpoint does not contain a compatible generator")
    return state


def _resolve(config, resumed=None):
    if resumed:
        saved = dict(resumed["config"])
        incoming, defaults = config.to_dict(), TrainConfig().to_dict()
        mutable = {
            "output",
            "device",
            "workers",
            "epochs",
            "decay_epochs",
            "max_steps",
            "sample_every",
            "checkpoint_every",
        }
        for name, value in incoming.items():
            if name in {"resume", "warm_start"}:
                continue
            if name in mutable:
                if (
                    name == "max_steps"
                    or name in config._specified_resume_fields
                    or (value != defaults[name] and value is not None)
                ):
                    saved[name] = value
            elif value != defaults[name] and value != saved.get(name):
                raise ValueError(
                    f"Resume configuration differs for {name}; use warm_start to change training settings"
                )
        saved.update(resume=config.resume, warm_start=None)
        config = TrainConfig(**saved)
    cyclegan = config.method == "cyclegan"
    values = config.to_dict()
    values["epochs"] = (50 if cyclegan else 30) if config.epochs is None else config.epochs
    values["decay_epochs"] = (
        (50 if cyclegan else 30) if config.decay_epochs is None else config.decay_epochs
    )
    values["batch_size"] = (
        (1 if cyclegan else 4) if config.batch_size is None else config.batch_size
    )
    values["load_size"] = (
        (config.crop_size if cyclegan else max(286, config.crop_size))
        if config.load_size is None
        else config.load_size
    )
    values["ndf"] = (32 if cyclegan else 64) if config.ndf is None else config.ndf
    values["lr"] = (1e-5 if cyclegan else 2e-4) if config.lr is None else config.lr
    values["architecture"] = config.architecture or ("unet_256" if cyclegan else "pix2pix_unet_256")
    values["input_mode"] = config.input_mode or (
        "grayscale" if cyclegan or config.grayscale_source else "rgb"
    )
    resolved = TrainConfig(**values)
    if resolved.seg_depth < 2:
        raise ValueError("seg_depth must be >= 2")
    unknown = set(resolved.options) - ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"Unknown training options: {', '.join(sorted(unknown))}")
    if resolved.load_size < resolved.crop_size:
        raise ValueError("load_size must be >= crop_size")
    stride, minimum = spatial_requirements(resolved.method, resolved.architecture)
    if resolved.crop_size < minimum or resolved.crop_size % stride:
        raise ValueError(
            f"crop_size must be >= {minimum} and divisible by {stride} for {resolved.architecture}"
        )
    if resolved.epochs + resolved.decay_epochs == 0:
        raise ValueError("At least one training epoch is required")
    if cyclegan and (resolved.pairs or resolved.paired_ab or resolved.grayscale_source):
        raise ValueError(
            "CycleGAN uses unpaired source/target inputs; pairing options belong to Pix2Pix"
        )
    if not cyclegan and resolved.lambda_mask:
        raise ValueError("Auxiliary mask consistency is supported by CycleGAN only")
    for name in ("dropout", "color_jitter", "amp"):
        if name in resolved.options and not isinstance(resolved.options[name], bool):
            raise ValueError(f"Training option {name} must be boolean")
    for name in ("brightness", "contrast", "saturation"):
        value = resolved.options.get(name, 0.75 if name != "contrast" else 0.5)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Training option {name} must be finite and nonnegative")
    pool_size = resolved.options.get("pool_size", 50)
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 0:
        raise ValueError("pool_size must be a nonnegative integer")
    beta = resolved.options.get("beta1", 0.5)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < 1:
        raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
    return resolved


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _metadata(config, direction="AtoB"):
    reverse = direction == "BtoA"
    return {
        "format": "stain_transfer.generator.v1",
        "method": config.method,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "architecture": config.architecture,
        "input_mode": config.input_mode,
        "direction": direction,
        "ngf": config.ngf,
        "norm": config.options.get("norm", "batch" if config.method == "pix2pix" else "instance"),
        "dropout": config.options.get("dropout", config.method == "pix2pix"),
        "source_name": config.target_name if reverse else config.source_name,
        "target_name": config.source_name if reverse else config.target_name,
        "training_crop_size": config.crop_size,
    }


def _export(output, config, nets):
    exports = []
    directions = (
        (("AtoB", "G_A"), ("BtoA", "G_B")) if config.method == "cyclegan" else (("AtoB", "G"),)
    )
    for direction, key in directions:
        path = output / f"generator_{direction}.pth"
        _save(path, _cpu_state(nets[key]))
        path.with_suffix(".json").write_text(
            json.dumps(_metadata(config, direction), indent=2) + "\n"
        )
        exports.append(path)
    return tuple(exports)


def _freeze(models, frozen):
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad_(not frozen)


def _gan(prediction, real):
    return torch.nn.functional.mse_loss(
        prediction, torch.ones_like(prediction) if real else torch.zeros_like(prediction)
    )


def _masked_dice(logits, masks, available):
    weights = available.reshape(-1).float()
    if not bool(weights.any()):
        return logits.sum() * 0
    probabilities = logits.sigmoid().flatten(1)
    truth = masks.flatten(1)
    loss = 1 - (2 * (probabilities * truth).sum(1) + 1) / (probabilities.sum(1) + truth.sum(1) + 1)
    return (loss * weights).sum() / weights.sum()


def _source_tensor(tensor, config, augment=False):
    if config.input_mode == "rgb":
        return tensor
    if augment and config.options.get("color_jitter", True):
        # Jitter in [0,1], then restore the generator's documented [-1,1] range.
        rgb = (tensor + 1) / 2
        for op in torch.randperm(3).tolist():
            amount = config.options.get(
                ("brightness", "contrast", "saturation")[op], (0.75, 0.5, 0.75)[op]
            )
            factor = float(torch.empty(()).uniform_(max(0, 1 - amount), 1 + amount))
            if op == 0:
                rgb = rgb * factor
            elif op == 1:
                mean = grayscale_tensor(rgb).mean((2, 3), keepdim=True)
                rgb = rgb * factor + mean * (1 - factor)
            else:
                gray = grayscale_tensor(rgb)
                rgb = rgb * factor + gray * (1 - factor)
            rgb = rgb.clamp(0, 1)
        tensor = rgb * 2 - 1
    return grayscale_tensor(tensor)


def _cycle_step(batch, nets, optimizers, scalers, config, context, pools):
    a, b = batch["A"], batch["B"]
    discriminators = [nets["D_A"], nets["D_B"]]
    segments = [nets[key] for key in ("S_A", "S_B") if key in nets]
    losses = {}
    if segments:
        _freeze(segments, False)
        optimizers["S"].zero_grad(set_to_none=True)
        with context():
            seg_loss = config.lambda_real_seg * (
                _masked_dice(nets["S_A"](a), batch["A_mask"], batch["A_has_mask"])
                + _masked_dice(nets["S_B"](b), batch["B_mask"], batch["B_has_mask"])
            )
        scalers["S"].scale(seg_loss).backward()
        scalers["S"].step(optimizers["S"])
        scalers["S"].update()
        losses["segmentation"] = float(seg_loss.detach())
        _freeze(segments, True)
    _freeze(discriminators, True)
    optimizers["G"].zero_grad(set_to_none=True)
    with context():
        fake_b = nets["G_A"](_source_tensor(a, config, True))
        fake_a = nets["G_B"](_source_tensor(b, config, True))
        rec_a = nets["G_B"](_source_tensor(fake_b, config, True))
        rec_b = nets["G_A"](_source_tensor(fake_a, config, True))
        gan_loss = _gan(nets["D_A"](fake_b), True) + _gan(nets["D_B"](fake_a), True)
        cycle_loss = config.lambda_cycle * (
            torch.nn.functional.l1_loss(rec_a, a) + torch.nn.functional.l1_loss(rec_b, b)
        )
        identity_loss = a.new_zeros(())
        if config.lambda_identity:
            identity_loss = (
                config.lambda_cycle
                * config.lambda_identity
                * (
                    torch.nn.functional.l1_loss(nets["G_A"](_source_tensor(b, config)), b)
                    + torch.nn.functional.l1_loss(nets["G_B"](_source_tensor(a, config)), a)
                )
            )
        mask_loss = a.new_zeros(())
        if segments:
            mask_loss = config.lambda_mask * (
                _masked_dice(nets["S_B"](fake_b), batch["A_mask"], batch["A_has_mask"])
                + _masked_dice(nets["S_A"](fake_a), batch["B_mask"], batch["B_has_mask"])
            )
        g_loss = gan_loss + cycle_loss + identity_loss + mask_loss
    if not torch.isfinite(g_loss):
        raise RuntimeError("Non-finite CycleGAN generator loss")
    scalers["G"].scale(g_loss).backward()
    scalers["G"].step(optimizers["G"])
    scalers["G"].update()
    _freeze(discriminators, False)
    optimizers["D"].zero_grad(set_to_none=True)
    with context():
        d_a = (_gan(nets["D_A"](b), True) + _gan(nets["D_A"](pools["B"].query(fake_b)), False)) / 2
        d_b = (_gan(nets["D_B"](a), True) + _gan(nets["D_B"](pools["A"].query(fake_a)), False)) / 2
        d_loss = d_a + d_b
    if not torch.isfinite(d_loss):
        raise RuntimeError("Non-finite CycleGAN discriminator loss")
    scalers["D"].scale(d_loss).backward()
    scalers["D"].step(optimizers["D"])
    scalers["D"].update()
    losses.update(
        generator=float(g_loss.detach()),
        discriminator=float(d_loss.detach()),
        cycle=float(cycle_loss.detach()),
        identity=float(identity_loss.detach()),
        mask=float(mask_loss.detach()),
    )
    return losses, fake_b.detach()


def _pix_step(batch, nets, optimizers, scalers, config, context):
    a, b = _source_tensor(batch["A"], config), batch["B"]
    optimizers["D"].zero_grad(set_to_none=True)
    with context():
        with torch.no_grad():
            fake = nets["G"](a)
        d_loss = (_gan(nets["D"](a, b), True) + _gan(nets["D"](a, fake), False)) / 2
    scalers["D"].scale(d_loss).backward()
    scalers["D"].step(optimizers["D"])
    scalers["D"].update()
    _freeze([nets["D"]], True)
    optimizers["G"].zero_grad(set_to_none=True)
    with context():
        fake = nets["G"](a)
        l1 = config.lambda_l1 * torch.nn.functional.l1_loss(fake, b)
        g_loss = _gan(nets["D"](a, fake), True) + l1
    if not torch.isfinite(g_loss + d_loss):
        raise RuntimeError("Non-finite Pix2Pix loss")
    scalers["G"].scale(g_loss).backward()
    scalers["G"].step(optimizers["G"])
    scalers["G"].update()
    _freeze([nets["D"]], False)
    return {
        "generator": float(g_loss.detach()),
        "discriminator": float(d_loss.detach()),
        "l1": float(l1.detach()),
    }, fake.detach()


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def train(config):
    if isinstance(config, dict):
        config = TrainConfig(**config)
    if not isinstance(config, TrainConfig):
        raise TypeError("train expects TrainConfig or a dictionary")
    resumed = _validate_resume(_load(config.resume)) if config.resume else None
    config = _resolve(config, resumed)
    warm_weights = (
        _validate_warm_start(_load(config.warm_start), config) if config.warm_start else None
    )
    device = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.method == "cyclegan":
        dataset = UnpairedDataset(config.source, config.target, config.load_size, config.crop_size)
        if config.lambda_mask and not dataset.has_masks:
            raise ValueError(
                "lambda_mask > 0 requires usable mask_path entries in at least one training manifest"
            )
    else:
        dataset = PairedDataset(
            config.source,
            config.target,
            config.pairs,
            config.paired_ab,
            config.grayscale_source,
            config.load_size,
            config.crop_size,
        )
    output = Path(config.output).expanduser().resolve()
    if output.is_dir() and not config.resume:
        known_artifacts = {
            "training_config.json",
            "losses.jsonl",
            "training_latest.pth",
            "samples",
            "generator_AtoB.pth",
            "generator_AtoB.json",
            "generator_BtoA.pth",
            "generator_BtoA.json",
        }
        if any((output / name).exists() for name in known_artifacts):
            raise FileExistsError(
                f"Training output already contains training artifacts; choose another output or resume: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)
    norm = config.options.get("norm", "batch" if config.method == "pix2pix" else "instance")
    if config.method == "pix2pix" and norm != "batch":
        raise ValueError("Pix2Pix uses batch normalization; norm must be batch")

    def make_generator():
        return initialize(
            generator(
                config.method,
                config.architecture,
                config.ngf,
                norm,
                config.options.get("dropout", config.method == "pix2pix"),
            )
        ).to(device)

    def make_discriminator():
        return initialize(discriminator(config.method, config.ndf, norm)).to(device)

    if config.method == "cyclegan":
        nets = {
            "G_A": make_generator(),
            "G_B": make_generator(),
            "D_A": make_discriminator(),
            "D_B": make_discriminator(),
        }
        if config.lambda_mask:
            nets.update(
                {
                    name: initialize(
                        NNUNetStyleUNet(3, 1, config.seg_channels, config.seg_depth)
                    ).to(device)
                    for name in ("S_A", "S_B")
                }
            )
        parameters = {
            "G": itertools.chain(nets["G_A"].parameters(), nets["G_B"].parameters()),
            "D": itertools.chain(nets["D_A"].parameters(), nets["D_B"].parameters()),
        }
        if config.lambda_mask:
            parameters["S"] = itertools.chain(nets["S_A"].parameters(), nets["S_B"].parameters())
    else:
        nets = {"G": make_generator(), "D": make_discriminator()}
        parameters = {name: net.parameters() for name, net in nets.items()}
    optimizers = {
        name: torch.optim.Adam(
            params, lr=config.lr, betas=(config.options.get("beta1", 0.5), 0.999)
        )
        for name, params in parameters.items()
    }

    def schedule(epoch):
        return 1 - max(0, epoch - config.epochs) / max(1, config.decay_epochs)

    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        for name, optimizer in optimizers.items()
    }
    amp = device.type == "cuda" and config.options.get("amp", True)
    scalers = {name: torch.amp.GradScaler("cuda", enabled=amp) for name in optimizers}

    def context():
        return torch.autocast(device_type="cuda", enabled=True) if amp else contextlib.nullcontext()

    pools = {name: ImagePool(config.options.get("pool_size", 50)) for name in ("A", "B")}
    completed, global_step, next_batch, start_epoch = 0, 0, 0, 1
    if resumed:
        for name, net in nets.items():
            net.load_state_dict(resumed["networks"][name])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(resumed["optimizers"][name])
            schedulers[name].load_state_dict(resumed["schedulers"][name])
            scalers[name].load_state_dict(resumed["scalers"][name])
        for name, images in resumed.get("pools", {}).items():
            pools[name].images = images
        completed, global_step = resumed["epochs_completed"], resumed["global_step"]
        next_batch = resumed.get("next_batch", 0)
        start_epoch = resumed.get("current_epoch", completed + 1) if next_batch else completed + 1
        if next_batch and config.workers:
            raise ValueError(
                "Mid-epoch max_steps checkpoints can resume only with workers=0; epoch checkpoints support any worker count"
            )
        _restore_rng(resumed["rng"])
    elif config.warm_start:
        weights = warm_weights
        if weights.get("format") == FORMAT:
            loaded_generators = 0
            for name, state in weights["networks"].items():
                if name in nets:
                    nets[name].load_state_dict(state)
                    loaded_generators += int(name in {"G_A", "G_B", "G"})
            if not loaded_generators:
                raise ValueError("Full warm_start checkpoint did not load a compatible generator")
        else:
            metadata = checkpoint_metadata(config.warm_start)
            key = (
                "G_B"
                if config.method == "cyclegan" and metadata.get("direction") == "BtoA"
                else "G_A"
                if config.method == "cyclegan"
                else "G"
            )
            if weights and all(name.startswith("module.") for name in weights):
                weights = {name[7:]: value for name, value in weights.items()}
            nets[key].load_state_dict(weights)
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    latest = output / "training_latest.pth"
    loss_path = output / "losses.jsonl"
    updates, current_epoch, last_fake, last_batch = 0, start_epoch, None, None

    def save_checkpoint():
        state = {
            "format": FORMAT,
            "config": config.to_dict(),
            "metadata": _metadata(config),
            "epochs_completed": completed,
            "current_epoch": current_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "networks": {name: _cpu_state(net) for name, net in nets.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
            "rng": _rng_state(),
            "pools": {name: pool.images for name, pool in pools.items()},
        }
        _save(latest, state)

    total_epochs = config.epochs + config.decay_epochs
    for current_epoch in range(start_epoch, total_epochs + 1):
        loader_seed = torch.Generator().manual_seed(config.seed + current_epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            drop_last=False,
            generator=loader_seed,
            worker_init_fn=_seed_worker,
        )
        skip = next_batch
        restore_after_skip = _rng_state() if skip else None
        for index, batch in enumerate(loader):
            if index < skip:
                if index + 1 == skip:
                    _restore_rng(restore_after_skip)
                continue
            batch = {name: value.to(device) for name, value in batch.items()}
            if config.method == "cyclegan":
                losses, fake = _cycle_step(batch, nets, optimizers, scalers, config, context, pools)
            else:
                losses, fake = _pix_step(batch, nets, optimizers, scalers, config, context)
            global_step += 1
            updates += 1
            next_batch = index + 1
            last_fake, last_batch = fake, batch
            with loss_path.open("a") as stream:
                stream.write(
                    json.dumps({"epoch": current_epoch, "step": global_step, **losses}) + "\n"
                )
            if config.max_steps is not None and updates >= config.max_steps:
                break
        epoch_finished = next_batch >= len(loader)
        if epoch_finished:
            completed = current_epoch
            next_batch = 0
            for scheduler in schedulers.values():
                scheduler.step()
            if (
                config.sample_every
                and current_epoch % config.sample_every == 0
                and last_fake is not None
            ):
                samples = output / "samples"
                samples.mkdir(exist_ok=True)
                triptych = torch.cat((last_batch["A"][0], last_fake[0], last_batch["B"][0]), dim=2)
                pixels = (
                    ((triptych.detach().cpu().clamp(-1, 1) + 1) * 127.5)
                    .round()
                    .byte()
                    .permute(1, 2, 0)
                    .numpy()
                )
                Image.fromarray(pixels).save(samples / f"epoch_{current_epoch:04d}.png")
            if current_epoch % config.checkpoint_every == 0:
                save_checkpoint()
        if config.max_steps is not None and updates >= config.max_steps:
            break
    save_checkpoint()
    exports = _export(output, config, nets)
    return TrainReport(
        output=output,
        epochs_completed=completed,
        global_step=global_step,
        checkpoint=latest,
        exports=exports,
    )
