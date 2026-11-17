from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image as PILImage

from config import DatasetConfig, ALL_DATASETS
from attacks import TriggerSampler
from pathlib import Path


class AuxiliaryImageDataset(Dataset):
    """A user-provided training split, separate from every downstream task.

    Labels and directory class names are ignored. The caller must prepare
    disjoint train/validation/test base-image splits before synthesizing pairs.
    """

    def __init__(self, root: str, image_size: int = 32):
        directory = Path(root).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"Auxiliary image directory not found: {directory}")
        self.paths = sorted(p for p in directory.rglob('*')
                            if p.is_file() and p.suffix.lower() in
                            {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        if not self.paths:
            raise ValueError("The auxiliary training directory contains no images")
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from brifi import canonicalize_detector
        with PILImage.open(self.paths[index]) as source:
            tensor = T.ToTensor()(source.convert('RGB'))
        image = canonicalize_detector(tensor.unsqueeze(0), self.image_size)[0]
        return image, 0

class _RepeatChannels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1)

    def __repr__(self) -> str:
        return "RepeatChannels(n=3)"

class TinyImageNet(Dataset):

    def __init__(self, root: str, train: bool = True):
        base = os.path.join(root, "tiny-imagenet-200")
        wnids_path = os.path.join(base, "wnids.txt")
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(
                f"TinyImageNet not found at {base}.\n"
                "Download: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                f"then unzip into {root}")

        with open(wnids_path) as f:
            wnids = sorted(l.strip() for l in f if l.strip())
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples: List[Tuple[str, int]] = []

        if train:
            for wnid in wnids:
                img_dir = os.path.join(base, "train", wnid, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fn in sorted(os.listdir(img_dir)):
                    if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append(
                            (os.path.join(img_dir, fn), self.class_to_idx[wnid]))
        else:
            ann_file = os.path.join(base, "val", "val_annotations.txt")
            img_dir  = os.path.join(base, "val", "images")
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        fn, wnid = parts[0], parts[1]
                        self.samples.append(
                            (os.path.join(img_dir, fn),
                             self.class_to_idx.get(wnid, 0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[PILImage.Image, int]:
        path, label = self.samples[i]
        return PILImage.open(path).convert("RGB"), label

def get_raw_dataset(dcfg: DatasetConfig, train: bool = True) -> Dataset:
    root = dcfg.root
    if dcfg.name == "cifar10":
        return torchvision.datasets.CIFAR10(root, train=train, download=True)
    elif dcfg.name == "cifar100":
        return torchvision.datasets.CIFAR100(root, train=train, download=True)
    elif dcfg.name == "mnist":
        return torchvision.datasets.MNIST(root, train=train, download=True)
    elif dcfg.name == "tinyimagenet":
        return TinyImageNet(root, train=train)
    else:
        raise ValueError(f"Unknown dataset: {dcfg.name!r}")

def _make_to_tensor(dcfg: DatasetConfig) -> T.Compose:
    tfs: list = [T.ToTensor()]
    if dcfg.to_rgb:
        tfs.append(_RepeatChannels())
    return T.Compose(tfs)

def _make_aug(dcfg: DatasetConfig, use_aug: bool) -> Optional[T.Compose]:
    if not use_aug:
        return None
    if dcfg.use_affine:
        return T.Compose([
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.RandomCrop(dcfg.image_size, padding=dcfg.aug_padding),
        ])
    return T.Compose([
        T.RandomCrop(dcfg.image_size, padding=dcfg.aug_padding),
        T.RandomHorizontalFlip(p=0.5),
    ])

_DETECTOR_REF_SIZE = 32
_DETECTOR_REF_DCFG = DatasetConfig(
    name="_ref", root="",
    num_classes=2,
    image_size=_DETECTOR_REF_SIZE, channels=3, to_rgb=False,
    trigger_sz_min=1, trigger_sz_max=8,
    aug_padding=4, use_affine=False,
)

class MixedBaseDataset(Dataset):

    def __init__(self, dataset_names: List[str], root: str,
                 n_per_dataset: int = 5_000, seed: int = 42,
                 always_include_names: Optional[List[str]] = None):
        rng = np.random.RandomState(seed)
        self._refs: List[Tuple[Dataset, int]] = []
        always_include_names = always_include_names or []

        def _add(name: str, n: int) -> None:
            dcfg = ALL_DATASETS[name](root)
            raw  = get_raw_dataset(dcfg, train=True)
            k    = min(n, len(raw))
            idx  = rng.choice(len(raw), k, replace=False)
            for i in idx:
                self._refs.append((raw, int(i)))

        for name in dataset_names:
            _add(name, n_per_dataset)

        for name in always_include_names:
            if name not in dataset_names:
                _add(name, max(n_per_dataset // 4, 500))

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, i: int) -> Tuple[PILImage.Image, int]:
        raw, idx = self._refs[i]
        pil, _   = raw[idx]
        pil      = pil.convert("RGB")
        if pil.size != (_DETECTOR_REF_SIZE, _DETECTOR_REF_SIZE):
            pil = pil.resize((_DETECTOR_REF_SIZE, _DETECTOR_REF_SIZE),
                              PILImage.BILINEAR)
        return pil, 0

class PoisonedDataset(Dataset):

    def __init__(self, base_ds: Dataset, dcfg: DatasetConfig,
                 poison_rate: float = 0.0, trigger_sampler: Optional[TriggerSampler] = None,
                 aug: bool = True, seed: int = 42):
        self.base_ds          = base_ds
        self.dcfg             = dcfg
        self.trigger_sampler  = trigger_sampler
        self.to_tensor        = _make_to_tensor(dcfg)
        self.aug_tf           = _make_aug(dcfg, aug)

        self.is_poisoned = np.zeros(len(base_ds), dtype=bool)
        if poison_rate > 0:
            n_p = int(len(base_ds) * poison_rate)
            idx = np.random.RandomState(seed).choice(len(base_ds), n_p, replace=False)
            self.is_poisoned[idx] = True

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        img, lbl = self.base_ds[i]
        if self.aug_tf is not None:
            img = self.aug_tf(img)
        img = self.to_tensor(img)
        if self.is_poisoned[i] and self.trigger_sampler is not None:
            img = self.trigger_sampler(img)
            lbl = self.dcfg.target_label
        return img, lbl, i

class FreqRandTrigger:

    def __init__(self, amplitude_range: Tuple[float, float] = (0.08, 0.25),
                 n_freq_bins: int = 4):
        self.amplitude_range = amplitude_range
        self.n_freq_bins     = n_freq_bins

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        C, H, W = img.shape
        import os
        seed = int.from_bytes(os.urandom(4), "big")
        rng  = torch.Generator().manual_seed(seed)

        amp = (self.amplitude_range[0] +
               torch.rand(1, generator=rng).item() *
               (self.amplitude_range[1] - self.amplitude_range[0]))

        u = torch.randint(1, max(H // 4, 2), (self.n_freq_bins,), generator=rng)
        v = torch.randint(1, max(W // 4, 2), (self.n_freq_bins,), generator=rng)
        phase = torch.rand(self.n_freq_bins, generator=rng) * 2 * 3.14159

        ys = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        xs = torch.arange(W, dtype=torch.float32).unsqueeze(0)
        noise = torch.zeros(H, W)
        for k in range(self.n_freq_bins):
            noise += torch.sin(2 * 3.14159 * (u[k] * ys / H +
                                               v[k] * xs / W) + phase[k])
        noise = noise / (self.n_freq_bins * 1.4142)

        return (img + amp * noise.unsqueeze(0)).clamp(0.0, 1.0)

class DetectorDataset(Dataset):

    def __init__(self, base_ds: Dataset, dcfg: DatasetConfig,
                 ts: TriggerSampler, n_samples: int, seed: int = 123,
                 hard_neg_aug=None, hardneg_ratio: float = 0.0,
                 freq_rand_prob: float = 0.0):
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(base_ds), min(n_samples, len(base_ds)), replace=False)
        self.pil_imgs  = [base_ds[i][0] for i in sel]
        self.ts        = ts
        self.aug_tf    = _make_aug(dcfg, use_aug=True)
        self.to_tensor = _make_to_tensor(dcfg)
        self.n         = len(sel)
        self.freq_rand_prob   = freq_rand_prob
        self._freq_rand_stamp = FreqRandTrigger() if freq_rand_prob > 0 else None

        self.hard_neg_aug = hard_neg_aug
        n_hard = int(self.n * hardneg_ratio) if (hard_neg_aug is not None and hardneg_ratio > 0) else 0
        if n_hard > 0:
            rng2 = np.random.RandomState(seed + 7)
            hard_sel = rng2.choice(len(base_ds), min(n_hard, len(base_ds)), replace=False)
            self.hard_pil_imgs = [base_ds[i][0] for i in hard_sel]
        else:
            self.hard_pil_imgs = []
        self.n_hard = len(self.hard_pil_imgs)

    def __len__(self) -> int:
        return 2 * self.n + self.n_hard

    def _as_tensor(self, image):
        return image if isinstance(image, torch.Tensor) else self.to_tensor(image)

    def __getitem__(self, i: int):
        if i < self.n:
            pil = self.pil_imgs[i]
            aug = self.aug_tf(pil) if self.aug_tf is not None else pil
            return self._as_tensor(aug), torch.tensor(0.0)
        elif i < 2 * self.n:
            pil = self.pil_imgs[i - self.n]
            aug = self.aug_tf(pil) if self.aug_tf is not None else pil
            t   = self._as_tensor(aug)
            if (self._freq_rand_stamp is not None and
                    torch.rand(1).item() < self.freq_rand_prob):
                return self._freq_rand_stamp(t), torch.tensor(1.0)
            return self.ts(t), torch.tensor(1.0)
        else:
            j   = (i - 2 * self.n) % max(self.n_hard, 1)
            pil = self.hard_pil_imgs[j]
            aug = self.aug_tf(pil) if self.aug_tf is not None else pil
            t   = self._as_tensor(aug)
            return self.hard_neg_aug(t), torch.tensor(0.0)

class TriggeredTestSet(Dataset):

    def __init__(self, base_ds: Dataset, dcfg: DatasetConfig,
                 trigger_fn: callable):
        self.base_ds    = base_ds
        self.to_tensor  = _make_to_tensor(dcfg)
        self.trigger_fn = trigger_fn

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        img, lbl = self.base_ds[i]
        t = self.trigger_fn(self.to_tensor(img))
        return t, lbl

class CleanTestSet(Dataset):

    def __init__(self, base_ds: Dataset, dcfg: DatasetConfig):
        self.base_ds   = base_ds
        self.to_tensor = _make_to_tensor(dcfg)

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        img, lbl = self.base_ds[i]
        return self.to_tensor(img), lbl
