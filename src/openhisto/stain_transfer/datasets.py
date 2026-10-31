"""Portable image datasets with synchronized paired/mask geometry."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .imageio import load_rgb

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    mask: Path | None = None


def _path(value, base):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def image_records(source):
    if not source:
        raise ValueError("An image directory or CSV manifest is required")
    path = Path(source).expanduser()
    if path.is_dir():
        records = [
            ImageRecord(p.resolve())
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    elif path.is_file() and path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Empty image manifest: {path}")
            if "path" in reader.fieldnames:
                records = [
                    ImageRecord(
                        _path(row["path"], path.parent),
                        _path(row["mask_path"], path.parent)
                        if row.get("mask_path", "").strip()
                        else None,
                    )
                    for row in reader
                ]
            elif {"slide_dir", "mask_dir"}.issubset(reader.fieldnames):
                records = []
                for row in reader:
                    slide_dir, mask_dir = (
                        Path(row["slide_dir"]).expanduser(),
                        Path(row["mask_dir"]).expanduser(),
                    )
                    slide_dir = slide_dir if slide_dir.is_absolute() else path.parent / slide_dir
                    mask_dir = mask_dir if mask_dir.is_absolute() else path.parent / mask_dir
                    if not slide_dir.is_dir() or not mask_dir.is_dir():
                        raise FileNotFoundError(
                            f"Manifest slide_dir/mask_dir missing: {slide_dir}, {mask_dir}"
                        )
                    images = {
                        image.stem: image
                        for image in sorted(slide_dir.iterdir())
                        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                    }
                    for mask in sorted(mask_dir.iterdir()):
                        if (
                            mask.is_file()
                            and mask.suffix.lower() in IMAGE_EXTENSIONS
                            and mask.stem.endswith("_mask")
                        ):
                            image = images.get(mask.stem[:-5])
                            if image is not None:
                                records.append(ImageRecord(image.resolve(), mask.resolve()))
            else:
                raise ValueError(
                    f"Image manifest requires path or slide_dir/mask_dir columns: {path}"
                )
    elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        records = [ImageRecord(path.resolve())]
    else:
        raise FileNotFoundError(f"Image directory or CSV manifest not found: {path}")
    if not records:
        raise ValueError(f"No supported images found in {path}")
    return records


def _read(record):
    rgb = Image.fromarray(load_rgb(record.path))
    mask = None
    if record.mask:
        with Image.open(record.mask) as image:
            mask = image.convert("L")
        if rgb.size != mask.size:
            raise ValueError(f"Image and mask dimensions differ: {record.path}, {record.mask}")
    return rgb, mask


def _tensor(image):
    array = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1


def geometry(images, masks, load_size, crop_size):
    """One random spatial transform for all images/masks in a registered group."""
    images = [image.resize((load_size, load_size), Image.Resampling.BICUBIC) for image in images]
    masks = [
        mask.resize((load_size, load_size), Image.Resampling.NEAREST) if mask is not None else None
        for mask in masks
    ]
    x = random.randint(0, load_size - crop_size)
    y = random.randint(0, load_size - crop_size)
    box = (x, y, x + crop_size, y + crop_size)
    images = [image.crop(box) for image in images]
    masks = [mask.crop(box) if mask is not None else None for mask in masks]
    if random.random() < 0.5:
        images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
        masks = [
            mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mask is not None else None
            for mask in masks
        ]
    image_tensors = [_tensor(image) for image in images]
    mask_tensors = [
        torch.from_numpy((np.array(mask, copy=True) > 0).astype(np.float32)).unsqueeze(0)
        if mask is not None
        else torch.zeros((1, crop_size, crop_size))
        for mask in masks
    ]
    return image_tensors, mask_tensors


class UnpairedDataset(Dataset):
    def __init__(self, source, target, load_size=256, crop_size=256):
        self.source = image_records(source)
        self.target = image_records(target)
        self.load_size, self.crop_size = load_size, crop_size
        self.has_masks = any(record.mask for record in self.source + self.target)

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, index):
        a = self.source[index % len(self.source)]
        b = self.target[random.randrange(len(self.target))]
        image_a, mask_a = _read(a)
        image_b, mask_b = _read(b)
        (image_a,), (mask_a,) = geometry([image_a], [mask_a], self.load_size, self.crop_size)
        (image_b,), (mask_b,) = geometry([image_b], [mask_b], self.load_size, self.crop_size)
        return {
            "A": image_a,
            "B": image_b,
            "A_mask": mask_a,
            "B_mask": mask_b,
            "A_has_mask": float(a.mask is not None),
            "B_has_mask": float(b.mask is not None),
        }


class PairedDataset(Dataset):
    def __init__(
        self,
        source=None,
        target=None,
        pairs=None,
        paired_ab=None,
        grayscale_source=False,
        load_size=286,
        crop_size=256,
    ):
        self.load_size, self.crop_size = load_size, crop_size
        self.grayscale_source = grayscale_source
        self.ab = bool(paired_ab)
        if grayscale_source:
            if source or pairs or paired_ab:
                raise ValueError(
                    "grayscale_source uses only target images; do not also provide paired/source inputs"
                )
            self.pairs = [(record, record) for record in image_records(target)]
        elif paired_ab:
            if source or target or pairs:
                raise ValueError("paired_ab cannot be combined with other training image inputs")
            self.pairs = [(record, record) for record in image_records(paired_ab)]
        elif pairs:
            if source or target:
                raise ValueError("Provide a paired CSV or source/target directories, not both")
            manifest = Path(pairs).expanduser()
            with manifest.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not {"source_path", "target_path"}.issubset(
                    reader.fieldnames
                ):
                    raise ValueError("Paired CSV requires source_path and target_path columns")
                self.pairs = [
                    (
                        ImageRecord(_path(row["source_path"], manifest.parent)),
                        ImageRecord(_path(row["target_path"], manifest.parent)),
                    )
                    for row in reader
                ]
        else:
            source_root, target_root = Path(source or ""), Path(target or "")
            if not source or not target or not source_root.is_dir() or not target_root.is_dir():
                raise ValueError(
                    "Pix2Pix requires source/target directories, a paired CSV, AB images, or grayscale_source"
                )
            left = {
                record.path.relative_to(source_root.resolve()): record
                for record in image_records(source_root)
            }
            right = {
                record.path.relative_to(target_root.resolve()): record
                for record in image_records(target_root)
            }
            if left.keys() != right.keys():
                missing_source = sorted(str(key) for key in right.keys() - left.keys())
                missing_target = sorted(str(key) for key in left.keys() - right.keys())
                raise ValueError(
                    f"Unmatched Pix2Pix files; missing source={missing_source[:5]}, missing target={missing_target[:5]}"
                )
            self.pairs = [(left[key], right[key]) for key in sorted(left)]
        if not self.pairs:
            raise ValueError("No paired training images found")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b = self.pairs[index]
        image_a, _ = _read(a)
        if self.ab:
            width, height = image_a.size
            if width % 2:
                raise ValueError(f"AB images must have an even width: {a.path}")
            image_b = image_a.crop((width // 2, 0, width, height))
            image_a = image_a.crop((0, 0, width // 2, height))
        else:
            image_b, _ = _read(b)
        if image_a.size != image_b.size:
            raise ValueError(f"Paired images must have matching dimensions: {a.path}, {b.path}")
        if self.grayscale_source:
            image_a = image_b.convert("L").convert("RGB")
        (image_a, image_b), _ = geometry(
            [image_a, image_b], [None, None], self.load_size, self.crop_size
        )
        return {"A": image_a, "B": image_b}
