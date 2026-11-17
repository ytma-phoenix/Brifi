from __future__ import annotations
import math, random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import DatasetConfig, CORE_FAMILIES, AGNOSTIC_FAMILIES

class TriggerSampler:

    _SODA_SEED = 1337

    _DEFAULTS: Dict[str, float] = {
        "badnets":       0.25,
        "blended":       0.17,
        "dba":           0.13,
        "pattern":       0.13,
        "noise":         0.09,
        "wanet":         0.09,
        "soda":          0.05,
        "freq_band":     0.05,
        "edge_patch":    0.05,
        "grid_warp":     0.04,
        "fourier_shuffle": 0.07,
        "narrowband":      0.07,
        "cutout_blend":    0.06,
        "phase_jitter":  0.05,
        "color_distort": 0.04,
        "smooth_noise":  0.04,
    }

    def __init__(self, dcfg: DatasetConfig,
                 families: Optional[List[str]] = None,
                 weights: Optional[List[float]] = None):
        self.H = dcfg.image_size
        self.W = dcfg.image_size
        self.C = 3 if dcfg.to_rgb else dcfg.channels
        self.sz_min = dcfg.trigger_sz_min
        self.sz_max = dcfg.trigger_sz_max

        if families is None:
            families = list(CORE_FAMILIES)
        self.families = families
        raw = [self._DEFAULTS[f] for f in families] if weights is None else list(weights)
        total = sum(raw)
        self.weights = [w / total for w in raw]
        self._cdf = list(np.cumsum(self.weights))

    def __repr__(self) -> str:
        parts = [f"{f}={w:.2f}" for f, w in zip(self.families, self.weights)]
        return f"TriggerSampler({', '.join(parts)})"

    def __call__(self, img_t: torch.Tensor) -> torch.Tensor:
        r = random.random()
        for fam, cdf_val in zip(self.families, self._cdf):
            if r < cdf_val:
                return getattr(self, f"_{fam}")(img_t)
        return getattr(self, f"_{self.families[-1]}")(img_t)

    def _rand_sz(self) -> int:
        return random.randint(self.sz_min, self.sz_max)

    def _rand_pos(self, sz: int) -> tuple:
        return random.randint(0, self.H - sz), random.randint(0, self.W - sz)

    def _badnets(self, img_t: torch.Tensor,
                 sz_range: Optional[tuple] = None) -> torch.Tensor:
        sz = random.randint(*(sz_range or (self.sz_min, self.sz_max)))
        y0, x0 = self._rand_pos(sz)
        col = torch.tensor([random.uniform(0.7, 1.0) for _ in range(self.C)]).view(self.C, 1, 1)
        out = img_t.clone()
        out[:, y0:y0 + sz, x0:x0 + sz] = col
        return out

    def _blended(self, img_t: torch.Tensor) -> torch.Tensor:
        sz = self._rand_sz()
        y0, x0 = self._rand_pos(sz)
        alpha = random.uniform(0.35, 0.55)
        pat   = torch.ones(self.C, sz, sz)
        out   = img_t.clone()
        out[:, y0:y0 + sz, x0:x0 + sz] = (
            (1.0 - alpha) * out[:, y0:y0 + sz, x0:x0 + sz] + alpha * pat)
        return out

    def _dba(self, img_t: torch.Tensor) -> torch.Tensor:
        out = img_t.clone()
        for _ in range(random.randint(2, 4)):
            out = self._badnets(out, sz_range=(self.sz_min, min(4, self.sz_max)))
        return out

    def _pattern(self, img_t: torch.Tensor) -> torch.Tensor:
        sz    = self._rand_sz()
        y0, x0 = self._rand_pos(sz)
        ptype = random.choice(["checkerboard", "stripes_h", "stripes_v", "cross"])
        patch = torch.zeros(self.C, sz, sz)
        if ptype == "checkerboard":
            for i in range(sz):
                for j in range(sz):
                    patch[:, i, j] = 0.9 if (i + j) % 2 == 0 else 0.1
        elif ptype == "stripes_h":
            for i in range(sz):
                patch[:, i, :] = 0.9 if i % 2 == 0 else 0.1
        elif ptype == "stripes_v":
            for j in range(sz):
                patch[:, :, j] = 0.9 if j % 2 == 0 else 0.1
        else:
            mid = sz // 2
            patch[:, mid, :] = random.uniform(0.8, 1.0)
            patch[:, :, mid] = random.uniform(0.8, 1.0)
        out = img_t.clone()
        out[:, y0:y0 + sz, x0:x0 + sz] = patch
        return out

    _NOISE_FREQ = 6.0

    def _noise(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp  = random.uniform(0.15, 0.25)
        phase = random.uniform(0, 2 * math.pi)
        ys   = torch.arange(H, dtype=torch.float32)
        wave = amp * torch.sin(2 * math.pi * self._NOISE_FREQ / H * ys + phase)
        wave = wave.view(1, H, 1).expand(C, H, W)
        return (img_t + wave).clamp(0.0, 1.0)

    def _wanet(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        strength = random.uniform(0.2, 0.6)
        freq     = random.randint(2, 5)
        disp     = strength * 2.0 / min(H, W)

        gy = torch.linspace(-1, 1, H)
        gx = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        dx = disp * torch.sin(2 * math.pi * freq * grid_y) + 0.01 * torch.randn(H, W)
        dy = disp * torch.sin(2 * math.pi * freq * grid_x) + 0.01 * torch.randn(H, W)
        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)

        out = F.grid_sample(img_t.unsqueeze(0), grid,
                            mode="bilinear", padding_mode="border",
                            align_corners=True)
        return out.squeeze(0)

    def _freq_band(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp     = random.uniform(0.15, 0.35)
        fx = random.uniform(H / 8, H / 3)
        fy = random.uniform(W / 8, W / 3)
        phase = random.uniform(0, 2 * math.pi)
        ys = torch.arange(H).float()
        xs = torch.arange(W).float()
        wave = amp * torch.sin(
            2 * math.pi * (ys.unsqueeze(1) / H * fx
                           + xs.unsqueeze(0) / W * fy) + phase)
        out = (img_t + wave.unsqueeze(0)).clamp(0.0, 1.0)
        return out

    def _edge_patch(self, img_t: torch.Tensor) -> torch.Tensor:
        sz = self._rand_sz()
        y0, x0 = self._rand_pos(sz)
        patch = img_t[:, y0:y0 + sz, x0:x0 + sz].clone()

        blurred = F.avg_pool2d(patch.unsqueeze(0),
                               kernel_size=3, stride=1, padding=1).squeeze(0)
        strength = random.uniform(1.5, 3.0)
        sharpened = (patch + strength * (patch - blurred)).clamp(0.0, 1.0)

        out = img_t.clone()
        out[:, y0:y0 + sz, x0:x0 + sz] = sharpened
        return out

    def _grid_warp(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        grid_n  = random.randint(3, 6)
        strength = random.uniform(0.05, 0.15)

        gy = torch.linspace(-1, 1, H)
        gx = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        noise_h = torch.randn(1, 1, grid_n, grid_n) * strength
        noise_w = torch.randn(1, 1, grid_n, grid_n) * strength
        dy = F.interpolate(noise_h, size=(H, W), mode="bicubic",
                           align_corners=True).squeeze()
        dx = F.interpolate(noise_w, size=(H, W), mode="bicubic",
                           align_corners=True).squeeze()

        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
        out  = F.grid_sample(img_t.unsqueeze(0), grid,
                             mode="bilinear", padding_mode="border",
                             align_corners=True)
        return out.squeeze(0)

    def _soda(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        sz = self._rand_sz()
        y0, x0 = self._rand_pos(sz)

        rng = np.random.RandomState(self._SODA_SEED)
        ood_raw = rng.randn(C, self.sz_max, self.sz_max).astype(np.float32)
        ood_full = torch.from_numpy(ood_raw)
        ood_full = (ood_full - ood_full.min()) / (ood_full.max() - ood_full.min() + 1e-8)
        ood = ood_full[:, :sz, :sz]

        alpha = random.uniform(0.35, 0.55)
        out = img_t.clone()
        out[:, y0:y0 + sz, x0:x0 + sz] = (
            (1.0 - alpha) * out[:, y0:y0 + sz, x0:x0 + sz] + alpha * ood)
        return out.clamp(0.0, 1.0)

    def _fourier_shuffle(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp   = random.uniform(0.10, 0.30)
        f_lo  = random.uniform(0.05, 0.4)
        f_hi  = f_lo + random.uniform(0.1, 0.4)

        gray = img_t.mean(0)
        F_c  = torch.fft.rfft2(gray)
        H_f, W2 = F_c.shape
        W_f = (W2 - 1) * 2

        freq_row = torch.arange(H_f, dtype=torch.float32)
        freq_row = torch.minimum(freq_row, H_f - freq_row) / (H_f / 2.0)
        freq_col = torch.arange(W2,  dtype=torch.float32) / (W_f / 2.0)
        freq_r   = (freq_row.unsqueeze(1).pow(2) + freq_col.pow(2)).sqrt()
        band     = (freq_r >= f_lo) & (freq_r < f_hi)

        mag   = F_c.abs()
        phase = torch.angle(F_c)
        band_idx = band.nonzero(as_tuple=False)
        if band_idx.numel() > 0:
            perm = torch.randperm(len(band_idx))
            shuffled_phase = phase.clone()
            for i, j in zip(range(len(band_idx)), perm.tolist()):
                r0, c0 = band_idx[i]
                r1, c1 = band_idx[j]
                shuffled_phase[r0, c0] = phase[r1, c1]
            F_out = mag * torch.exp(1j * shuffled_phase)
            out_gray = torch.fft.irfft2(F_out, s=(H_f, W_f)).clamp(0, 1)
            out = img_t.clone()
            for c in range(C):
                out[c] = (1 - amp) * img_t[c] + amp * out_gray
            return out.clamp(0.0, 1.0)
        return img_t

    def _narrowband(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp   = random.uniform(0.12, 0.25)
        phase = random.uniform(0, 2 * math.pi)
        freq  = random.uniform(2.0, H / 3.0)
        ys    = torch.arange(H, dtype=torch.float32)
        wave  = amp * torch.sin(2 * math.pi * freq / H * ys + phase)
        wave  = wave.view(1, H, 1).expand(C, H, W)
        return (img_t + wave).clamp(0.0, 1.0)

    def _cutout_blend(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        sz    = self._rand_sz()
        y0, x0 = self._rand_pos(sz)
        alpha = random.uniform(0.4, 1.0)
        fill_type = random.random()
        if fill_type < 0.5:
            fill = torch.full((C, sz, sz), random.uniform(0.0, 1.0))
        else:
            fill = torch.rand(C, sz, sz)
        out = img_t.clone()
        patch = out[:, y0:y0 + sz, x0:x0 + sz]
        out[:, y0:y0 + sz, x0:x0 + sz] = ((1 - alpha) * patch
                                           + alpha * fill).clamp(0.0, 1.0)
        return out

    def _phase_jitter(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp = random.uniform(0.1, 0.4)
        gray = img_t.mean(0)
        F_c  = torch.fft.rfft2(gray)
        phase_noise = (torch.rand_like(F_c.real) - 0.5) * math.pi * 2
        mag    = F_c.abs()
        phase  = torch.angle(F_c) + phase_noise
        F_out  = mag * torch.exp(1j * phase)
        jittered = torch.fft.irfft2(F_out, s=(H, W)).clamp(0, 1)
        out = img_t.clone()
        for c in range(C):
            out[c] = (1 - amp) * img_t[c] + amp * jittered
        return out.clamp(0.0, 1.0)

    def _color_distort(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        out = img_t.clone()
        for c in range(C):
            gamma = random.uniform(0.4, 2.5)
            scale = random.uniform(0.7, 1.3)
            out[c] = (out[c].clamp(1e-6, 1.0).pow(gamma) * scale).clamp(0.0, 1.0)
        return out

    def _smooth_noise(self, img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        grid_n = random.randint(4, 8)
        amp    = random.uniform(0.10, 0.30)
        coarse = torch.randn(1, C, grid_n, grid_n)
        field  = torch.nn.functional.interpolate(
            coarse, size=(H, W), mode="bicubic", align_corners=True
        ).squeeze(0)
        field  = (field - field.mean()) / (field.std() + 1e-6)
        return (img_t + amp * field).clamp(0.0, 1.0)

class HardNegAug:

    def __call__(self, img_t: torch.Tensor) -> torch.Tensor:
        choice = random.randint(0, 3)
        if choice == 0:
            return self._jpeg_artifact(img_t)
        elif choice == 1:
            return self._resize_upsample(img_t)
        elif choice == 2:
            return self._blur_sharpen(img_t)
        else:
            return self._mild_freq_noise(img_t)

    @staticmethod
    def _jpeg_artifact(img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        block = 8
        quality = random.uniform(0.3, 0.7)
        out = img_t.clone()
        for y in range(0, H - block + 1, block):
            for x in range(0, W - block + 1, block):
                patch = out[:, y:y + block, x:x + block]
                steps = max(2, int(8 * (1.0 - quality)))
                quant = (patch * steps).round() / steps
                out[:, y:y + block, x:x + block] = quant
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _resize_upsample(img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        scale = random.uniform(0.40, 0.70)
        small_h = max(4, int(H * scale))
        small_w = max(4, int(W * scale))
        down = F.interpolate(img_t.unsqueeze(0), size=(small_h, small_w),
                             mode="bilinear", align_corners=False)
        up   = F.interpolate(down, size=(H, W),
                             mode="bilinear", align_corners=False)
        return up.squeeze(0).clamp(0.0, 1.0)

    @staticmethod
    def _blur_sharpen(img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        k = random.choice([3, 5])
        blurred = F.avg_pool2d(img_t.unsqueeze(0),
                               kernel_size=k, stride=1, padding=k // 2
                               ).squeeze(0)
        if random.random() < 0.5:
            return blurred.clamp(0.0, 1.0)
        strength = random.uniform(0.8, 2.0)
        sharpened = (img_t + strength * (img_t - blurred)).clamp(0.0, 1.0)
        return sharpened

    @staticmethod
    def _mild_freq_noise(img_t: torch.Tensor) -> torch.Tensor:
        C, H, W = img_t.shape
        amp = random.uniform(0.03, 0.10)
        noise = torch.randn(C, H, W) * amp
        return (img_t + noise).clamp(0.0, 1.0)

_WANET_GRIDS: Dict[tuple, torch.Tensor] = {}

def _get_wanet_grid(H: int, W: int) -> torch.Tensor:
    key = (H, W)
    if key not in _WANET_GRIDS:
        prev = torch.get_rng_state()
        torch.manual_seed(42)
        gy = torch.linspace(-1, 1, H)
        gx = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        disp = 0.4 * 2.0 / min(H, W)
        dx   = disp * torch.sin(2 * math.pi * 3 * grid_y)
        dy   = disp * torch.sin(2 * math.pi * 3 * grid_x)
        _WANET_GRIDS[key] = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
        torch.set_rng_state(prev)
    return _WANET_GRIDS[key]

def _ev_badnets_fixed(img_t: torch.Tensor, sz: int) -> torch.Tensor:
    out = img_t.clone(); out[:, -sz:, -sz:] = 1.0; return out

def _ev_checker_fixed(img_t: torch.Tensor, sz: int) -> torch.Tensor:
    out   = img_t.clone()
    patch = torch.zeros(img_t.shape[0], sz, sz)
    for i in range(sz):
        for j in range(sz):
            patch[:, i, j] = 0.9 if (i + j) % 2 == 0 else 0.1
    out[:, -sz:, -sz:] = patch
    return out

def _ev_blended_fixed(img_t: torch.Tensor, sz: int, alpha: float = 0.3) -> torch.Tensor:
    out  = img_t.clone()
    ones = torch.ones(img_t.shape[0], sz, sz)
    out[:, -sz:, -sz:] = (1.0 - alpha) * out[:, -sz:, -sz:] + alpha * ones
    return out

def _ev_badnets_rand(img_t: torch.Tensor, sz: int) -> torch.Tensor:
    C, H, W = img_t.shape
    y0 = random.randint(0, H - sz); x0 = random.randint(0, W - sz)
    out = img_t.clone(); out[:, y0:y0 + sz, x0:x0 + sz] = 1.0; return out

def _ev_checker_rand(img_t: torch.Tensor, sz: int) -> torch.Tensor:
    C, H, W = img_t.shape
    y0 = random.randint(0, H - sz); x0 = random.randint(0, W - sz)
    out = img_t.clone()
    for i in range(sz):
        for j in range(sz):
            out[:, y0 + i, x0 + j] = 0.9 if (i + j) % 2 == 0 else 0.1
    return out

def _ev_blended_rand(img_t: torch.Tensor, sz: int, alpha: float = 0.3) -> torch.Tensor:
    C, H, W = img_t.shape
    y0 = random.randint(0, H - sz); x0 = random.randint(0, W - sz)
    out  = img_t.clone()
    ones = torch.ones(C, sz, sz)
    out[:, y0:y0 + sz, x0:x0 + sz] = (
        (1.0 - alpha) * out[:, y0:y0 + sz, x0:x0 + sz] + alpha * ones)
    return out

def _ev_wanet(img_t: torch.Tensor) -> torch.Tensor:
    C, H, W = img_t.shape
    grid = _get_wanet_grid(H, W)
    out  = F.grid_sample(img_t.unsqueeze(0), grid,
                         mode="bilinear", padding_mode="border",
                         align_corners=True)
    return out.squeeze(0)

def _ev_noise_fixed(img_t: torch.Tensor, amp: float = 0.20) -> torch.Tensor:
    C, H, W = img_t.shape
    freq = TriggerSampler._NOISE_FREQ
    ys   = torch.arange(H, dtype=torch.float32)
    wave = amp * torch.sin(2 * math.pi * freq / H * ys)
    wave = wave.view(1, H, 1).expand(C, H, W)
    return (img_t + wave).clamp(0.0, 1.0)

def _ev_soda_fixed(img_t: torch.Tensor, sz: int) -> torch.Tensor:
    C = img_t.shape[0]
    rng = np.random.RandomState(TriggerSampler._SODA_SEED)
    ood_raw = rng.randn(C, sz, sz).astype(np.float32)
    ood = torch.from_numpy(ood_raw)
    ood = (ood - ood.min()) / (ood.max() - ood.min() + 1e-8)
    alpha = 0.45
    out = img_t.clone()
    out[:, -sz:, -sz:] = (1.0 - alpha) * out[:, -sz:, -sz:] + alpha * ood
    return out.clamp(0.0, 1.0)

def get_diagnostic_triggers(dcfg: DatasetConfig) -> Dict[str, callable]:
    sz = dcfg.trigger_sz_max
    return {
        "BN-f": lambda img: _ev_badnets_fixed(img, sz),
        "CB-f": lambda img: _ev_checker_fixed(img, sz),
        "BL-f": lambda img: _ev_blended_fixed(img, sz),
        "BN-r": lambda img: _ev_badnets_rand(img, sz),
        "CB-r": lambda img: _ev_checker_rand(img, sz),
        "BL-r": lambda img: _ev_blended_rand(img, sz),
        "WN-f": _ev_wanet,
        "NS-f": _ev_noise_fixed,
        "SD-f": lambda img: _ev_soda_fixed(img, sz),
    }


def get_eval_triggers(dcfg: DatasetConfig) -> Dict[str, callable]:
    """Evaluate the six named families with the supplied family implementations.

    Parameters are drawn from the same family distributions as training. These
    family-level instantiations are not the original authors' full attack code.
    """
    return {name: TriggerSampler(dcfg, families=[name]) for name in CORE_FAMILIES}
