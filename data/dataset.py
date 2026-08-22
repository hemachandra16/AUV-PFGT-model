from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


class UIEBDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for paired raw/target underwater images from the UIEB dataset.

    The dataset expects two folders containing matching image filenames:
    - raw images under ``raw_dir``
    - reference images under ``reference_dir``

    Each image is loaded with Pillow, converted to RGB, resized to 256x256,
    converted to a PyTorch tensor, and normalized to the range [0, 1].
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        raw_dir: Optional[Union[str, Path]] = None,
        reference_dir: Optional[Union[str, Path]] = None,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.image_size = image_size
        self.augment = augment

        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[1]
        else:
            root_dir = Path(root_dir)

        if raw_dir is None:
            raw_dir = root_dir / "datasets" / "UIEB" / "raw-890"
        else:
            raw_dir = Path(raw_dir)

        if reference_dir is None:
            reference_dir = root_dir / "datasets" / "UIEB" / "reference-890"
        else:
            reference_dir = Path(reference_dir)

        self.raw_dir = raw_dir
        self.reference_dir = reference_dir

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw image directory was not found: {self.raw_dir}")
        if not self.reference_dir.exists():
            raise FileNotFoundError(
                f"Reference image directory was not found: {self.reference_dir}"
            )

        raw_files = sorted(path for path in self.raw_dir.iterdir() if path.is_file())
        reference_files = sorted(path for path in self.reference_dir.iterdir() if path.is_file())

        if not raw_files:
            raise FileNotFoundError(f"No image files were found in {self.raw_dir}")
        if not reference_files:
            raise FileNotFoundError(f"No image files were found in {self.reference_dir}")

        raw_names = {path.name for path in raw_files}
        reference_names = {path.name for path in reference_files}

        common_names = raw_names & reference_names

        self.pairs = [
            (raw_dir / name, reference_dir / name)
            for name in sorted(common_names)
        ]

        if not self.pairs:
            raise ValueError(
                "No matching image pairs were found between the raw and reference directories."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_path, reference_path = self.pairs[index]

        input_image = self._load_image(raw_path)
        target_image = self._load_image(reference_path)

        if self.augment:
            # Synchronized random horizontal flip
            if np.random.rand() > 0.5:
                input_image = torch.flip(input_image, dims=[2])
                target_image = torch.flip(target_image, dims=[2])
            # Synchronized random vertical flip
            if np.random.rand() > 0.5:
                input_image = torch.flip(input_image, dims=[1])
                target_image = torch.flip(target_image, dims=[1])

        return input_image, target_image

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), resample=self._get_resample_mode())
            image_array = np.array(image, dtype=np.float32)

        tensor = torch.from_numpy(image_array).permute(2, 0, 1) / 255.0
        return tensor

    @staticmethod
    def _get_resample_mode() -> int:
        if hasattr(Image, "Resampling"):
            return Image.Resampling.BILINEAR
        return Image.BILINEAR


def create_dataloader(
    root_dir: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
    reference_dir: Optional[Union[str, Path]] = None,
    image_size: int = 256,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for UIEB image pairs."""

    dataset = UIEBDataset(
        root_dir=root_dir,
        raw_dir=raw_dir,
        reference_dir=reference_dir,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_splits(
    config: Optional[dict] = None,
    root_dir: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
    reference_dir: Optional[Union[str, Path]] = None,
    image_size: Optional[int] = None,
    train_split: Optional[float] = None,
    seed: int = 42,
    augment_train: bool = True,
    extra_train_sources: Optional[Sequence[dict]] = None,
) -> Tuple[object, Subset]:
    """Return the project's canonical (train, validation) split of the UIEB dataset.

    This is the SINGLE definition of the split. It reproduces exactly what ``train.py``
    used to do inline: a ``torch.randperm`` over all pairs under a generator seeded with
    ``seed`` (42), taking the first ``train_split`` fraction as train and the remainder
    as validation.

    Why this exists: ``validate.py`` and ``test.py --mode dataset`` previously
    instantiated ``UIEBDataset`` over all 890 pairs and scored every one of them, which
    silently included the ~801 images the model had trained on. That leakage is what
    inflated the reported test PSNR (27.24 dB in logs/test.log) well above the true
    held-out validation PSNR observed during training (24.96 dB). Routing every script
    through this function means evaluation can only ever see the held-out 10%.

    Args:
        config: Optional full training config dict; ``dataset`` and ``dataloader``
            sections are read from it when the explicit arguments are not given.
        root_dir / raw_dir / reference_dir / image_size: Dataset location and sizing.
        train_split: Fraction of pairs used for training (default 0.9).
        seed: Split RNG seed. Must stay 42 to match previously trained checkpoints.
        augment_train: Whether the training subset applies augmentation. The validation
            subset is never augmented.

        extra_train_sources: Optional list of ``{raw_dir, reference_dir}`` dicts appended to
            the TRAINING pool only (session 6: LSUI). The validation subset is built solely
            from the UIEB split above and is never touched by these, so held-out numbers
            stay comparable to every prior session. Each extra source is paired by filename
            exactly like UIEB.

    Returns:
        ``(train_pool, val_subset)``. ``train_pool`` is a ``Subset`` when no extra sources
        are given, or a ``ConcatDataset`` when they are; ``val_subset`` is always a
        ``Subset`` of UIEB.
    """
    config = config or {}
    ds_cfg = config.get("dataset", {}) or {}
    dl_cfg = config.get("dataloader", {}) or {}

    root_dir = root_dir if root_dir is not None else ds_cfg.get("root_dir")
    raw_dir = raw_dir if raw_dir is not None else ds_cfg.get("raw_dir")
    reference_dir = reference_dir if reference_dir is not None else ds_cfg.get("reference_dir")
    image_size = image_size if image_size is not None else ds_cfg.get("image_size", 256)
    train_split = train_split if train_split is not None else dl_cfg.get("train_split", 0.9)
    if extra_train_sources is None:
        extra_train_sources = ds_cfg.get("extra_train_sources") or []

    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}.")

    common = dict(
        root_dir=root_dir,
        raw_dir=raw_dir,
        reference_dir=reference_dir,
        image_size=image_size,
    )
    # Two dataset instances so the training subset can augment while validation does not.
    train_base = UIEBDataset(augment=augment_train, **common)
    val_base = UIEBDataset(augment=False, **common)

    n_total = len(train_base)
    n_train = int(n_total * train_split)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices, val_indices = indices[:n_train], indices[n_train:]

    train_pool = Subset(train_base, train_indices)
    val_subset = Subset(val_base, val_indices)

    if extra_train_sources:
        parts = [train_pool]
        for src in extra_train_sources:
            extra = UIEBDataset(
                raw_dir=src["raw_dir"],
                reference_dir=src["reference_dir"],
                image_size=image_size,
                augment=augment_train,
            )
            parts.append(extra)
        train_pool = ConcatDataset(parts)

    return train_pool, val_subset


def pool_pair_names(pool) -> list[str]:
    """Every filename in a training pool, whether it is a Subset or a ConcatDataset.

    Used to prove programmatically that no held-out image has entered the training pool.
    """
    if isinstance(pool, ConcatDataset):
        names: list[str] = []
        for d in pool.datasets:
            names.extend(pool_pair_names(d))
        return names
    if isinstance(pool, Subset):
        return [pool.dataset.pairs[i][0].name for i in pool.indices]
    return [p[0].name for p in pool.pairs]


def subset_pair_names(subset) -> list[str]:
    """Filenames backing a split from :func:`get_splits`, in iteration order.

    Delegates to :func:`pool_pair_names` so it keeps working if a ConcatDataset training
    pool is passed in (callers such as validate.py --split train do this).
    """
    return pool_pair_names(subset)
