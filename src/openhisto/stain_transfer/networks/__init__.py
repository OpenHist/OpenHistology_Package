"""Checkpoint-compatible network definitions; imported only by GAN workflows."""
import torch


def generator(method, architecture, ngf=64, norm="instance", dropout=False):
    if method == "pix2pix":
        if architecture not in {"unet_256", "pix2pix_unet_256", "pix2pix_unet"}:
            raise ValueError("Pix2Pix architecture must be pix2pix_unet_256 (or unet_256)")
        from .pix2pix import UNetGenerator
        return UNetGenerator(3, 3, ngf=ngf, use_dropout=dropout)
    from .cyclegan import UnetGenerator, ResnetGenerator, get_norm_layer
    normalization = get_norm_layer(norm)
    if architecture in {"unet_128", "unet_256"}:
        return UnetGenerator(3, 3, 7 if architecture == "unet_128" else 8, ngf,
                             norm_layer=normalization, use_dropout=dropout)
    if architecture in {"resnet_6blocks", "resnet_9blocks"}:
        return ResnetGenerator(3, 3, ngf, norm_layer=normalization,
                               use_dropout=dropout,
                               n_blocks=6 if architecture == "resnet_6blocks" else 9)
    raise ValueError(f"Unsupported CycleGAN architecture: {architecture}")


def spatial_requirements(method, architecture):
    if method == "pix2pix" or architecture == "unet_256":
        return 256, 256
    if architecture == "unet_128":
        return 128, 128
    # Instance normalization needs more than one spatial element at the bottleneck.
    return 4, 8


def discriminator(method, ndf=32, norm="instance"):
    if method == "pix2pix":
        from .pix2pix import PatchDiscriminator
        return PatchDiscriminator(3, 3, ndf=ndf)
    from .cyclegan import NLayerDiscriminator, get_norm_layer
    return NLayerDiscriminator(3, ndf=ndf, norm_layer=get_norm_layer(norm))


def initialize(network):
    """Apply the existing normal(0,.02) initialization without console output."""
    for module in network.modules():
        name = module.__class__.__name__
        if ("Conv" in name or "Linear" in name) and hasattr(module, "weight"):
            weight = getattr(module, "weight_orig", module.weight)
            torch.nn.init.normal_(weight, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif "BatchNorm2d" in name:
            torch.nn.init.normal_(module.weight, 1.0, 0.02)
            torch.nn.init.zeros_(module.bias)
    return network
