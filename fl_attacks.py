from __future__ import annotations
import math
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from attacks import TriggerSampler
from config import DatasetConfig
from datasets import _make_to_tensor, _make_aug

class LGAManipulator:

    def __init__(self, param_sizes: List[int],
                 min_scale: float = 0.01, max_scale: float = 1.0):
        self.param_sizes = param_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prev_agg_update: Optional[torch.Tensor] = None

    @classmethod
    def from_model(cls, model: nn.Module, **kw) -> "LGAManipulator":
        sizes = [p.numel() for p in model.parameters()]
        return cls(sizes, **kw)

    def update_reference(self, agg_grad: torch.Tensor) -> None:
        self.prev_agg_update = agg_grad.detach().clone()

    def manipulate(self, mal_grad: torch.Tensor) -> torch.Tensor:
        if self.prev_agg_update is None:
            return mal_grad

        aligned = torch.zeros_like(mal_grad)
        offset = 0
        for sz in self.param_sizes:
            ref_l = self.prev_agg_update[offset:offset + sz]
            mal_l = mal_grad[offset:offset + sz]

            ref_norm = ref_l.norm().item()
            mal_norm = mal_l.norm().item()

            if mal_norm > 1e-10:
                scale = min(self.max_scale, ref_norm / (mal_norm + 1e-12))
                scale = max(self.min_scale, scale)
            else:
                scale = 1.0

            aligned[offset:offset + sz] = mal_l * scale
            offset += sz
        return aligned

class SoDAManipulator:

    def __init__(self, proj_lambda: float = 0.3, norm_lambda: float = 0.7):
        self.proj_lambda = proj_lambda
        self.norm_lambda = norm_lambda

    def manipulate(self, mal_grad: torch.Tensor,
                   benign_avg: torch.Tensor) -> torch.Tensor:
        mal_dir = mal_grad / (mal_grad.norm() + 1e-12)
        ben_dir = benign_avg / (benign_avg.norm() + 1e-12)

        blended = (1 - self.proj_lambda) * mal_dir + self.proj_lambda * ben_dir
        blended = blended / (blended.norm() + 1e-12)

        mal_norm = mal_grad.norm().item()
        ben_norm = benign_avg.norm().item()
        target_norm = (1 - self.norm_lambda) * mal_norm + self.norm_lambda * ben_norm

        return blended * target_norm

def _benign_updates(clean_grad: torch.Tensor, n: int) -> List[torch.Tensor]:
    base = clean_grad / max(n, 1)
    noise_scale = 0.01 * (clean_grad.norm().item() + 1e-8)
    return [base + torch.randn_like(base) * noise_scale for _ in range(n)]

def make_updates_lga(
    clean_grad: torch.Tensor,
    poison_grad: torch.Tensor,
    n_benign: int,
    n_malicious: int,
    manipulator: LGAManipulator,
) -> List[torch.Tensor]:
    benign = _benign_updates(clean_grad, n_benign)
    base_m = poison_grad / max(n_malicious, 1)
    noise_s = 0.005 * (poison_grad.norm().item() + 1e-8)
    malicious = [
        manipulator.manipulate(base_m + torch.randn_like(base_m) * noise_s)
        for _ in range(n_malicious)
    ]
    return benign + malicious

def make_updates_soda(
    clean_grad: torch.Tensor,
    poison_grad: torch.Tensor,
    n_benign: int,
    n_malicious: int,
    manipulator: SoDAManipulator,
) -> List[torch.Tensor]:
    benign = _benign_updates(clean_grad, n_benign)
    benign_avg = clean_grad / max(n_benign, 1)
    base_m = poison_grad / max(n_malicious, 1)
    noise_s = 0.005 * (poison_grad.norm().item() + 1e-8)
    malicious = [
        manipulator.manipulate(
            base_m + torch.randn_like(base_m) * noise_s, benign_avg)
        for _ in range(n_malicious)
    ]
    return benign + malicious

def make_updates_nba(
    clean_grad: torch.Tensor,
    poison_grads: List[torch.Tensor],
    n_benign: int,
    n_malicious: int,
    scale_factor: float = 1.5,
) -> List[torch.Tensor]:
    benign = _benign_updates(clean_grad, n_benign)
    n_groups = len(poison_grads)
    malicious = []
    for i in range(n_malicious):
        pg = poison_grads[i % n_groups]
        base_m = pg / max(n_malicious, 1) * scale_factor
        noise_s = 0.005 * (pg.norm().item() + 1e-8)
        malicious.append(base_m + torch.randn_like(base_m) * noise_s)
    return benign + malicious

class NBADataset(Dataset):

    def __init__(
        self,
        base_ds: Dataset,
        dcfg: DatasetConfig,
        poison_rate: float,
        trigger_configs: List[Dict],
        aug: bool = True,
        seed: int = 42,
    ):
        self.base_ds = base_ds
        self.dcfg = dcfg
        self.to_tensor = _make_to_tensor(dcfg)
        self.aug_tf = _make_aug(dcfg, aug)
        self.trigger_configs = trigger_configs
        self.n_groups = len(trigger_configs)

        self.samplers = [
            TriggerSampler(dcfg, families=[tc["family"]], weights=[1.0])
            for tc in trigger_configs
        ]

        self.is_poisoned = np.zeros(len(base_ds), dtype=bool)
        self.poison_group = np.full(len(base_ds), -1, dtype=int)

        if poison_rate > 0:
            n_p = int(len(base_ds) * poison_rate)
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(base_ds), n_p, replace=False)
            self.is_poisoned[idx] = True
            groups = rng.randint(0, self.n_groups, size=n_p)
            self.poison_group[idx] = groups

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        img, lbl = self.base_ds[i]
        if self.aug_tf is not None:
            img = self.aug_tf(img)
        img = self.to_tensor(img)

        if self.is_poisoned[i]:
            g = self.poison_group[i]
            img = self.samplers[g](img)
            lbl = self.trigger_configs[g]["target_class"]

        return img, lbl, i
