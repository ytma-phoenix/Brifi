"""The BRIFi filter in Algorithm 1 of the WSDM 2027 paper.

The detector is pretrained separately. Calibration is client-local and uses
only retained scores from earlier batches. No labels enter the keep decision.
"""
from collections import deque
import math
from statistics import NormalDist

import torch
from torch import nn
from torch.nn import functional as F

from config import MethodConfig


def canonicalize_detector(images, image_size=32):
    """Shared RGB/[0,1]/tensor-bilinear preprocessing for auxiliary and FL data."""
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("Expected NCHW input with one or three channels")
    images = images.float()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if image_size and images.shape[-2:] != (image_size,) * 2:
        images = F.interpolate(images, size=(image_size,) * 2,
                               mode="bilinear", align_corners=False, antialias=True)
    return images.clamp(0., 1.)


class BudgetedFilter:
    """Frozen scorer, retained-only median/MAD, and a hard per-batch cap.

    Inputs are NCHW float tensors in [0, 1]. Returned masks are on the CPU,
    matching the DataLoader batches consumed by ``trainer.train_epoch``.
    """

    def __init__(self, detector: nn.Module, mcfg: MethodConfig,
                 device: str = "cpu", det_img_size: int = 32):
        if not 0 <= mcfg.rho_max <= 1:
            raise ValueError("rho_max must lie in [0, 1]")
        if not 0 < mcfg.lrt_alpha < 1 or mcfg.eps <= 0:
            raise ValueError("alpha must lie in (0, 1) and eps must be positive")
        if not 1 <= mcfg.min_history_size <= mcfg.mad_buffer_size:
            raise ValueError("Require 1 <= min_history_size <= mad_buffer_size")
        self.detector = detector.to(device).eval().requires_grad_(False)
        self.mcfg = mcfg
        self.device = device
        self._det_img_size = det_img_size
        self.buf = deque(maxlen=mcfg.mad_buffer_size)
        self._z_crit = NormalDist().inv_cdf(1 - mcfg.lrt_alpha)
        self._seen = self._dropped = 0
        self._epoch = 0
        self._active = not mcfg.use_warmup

    @property
    def is_active(self):
        return self._active

    def set_active(self, active=True):
        self._active = bool(active)

    def set_epoch(self, epoch):
        self._epoch = epoch

    @staticmethod
    def _median_mad(values):
        center = torch.quantile(values, 0.5)
        scale = 1.4826 * torch.quantile((values - center).abs(), 0.5)
        return center, scale

    @torch.no_grad()
    def keep_with_weights(self, imgs, classifier_loss=None):
        del classifier_loss  # The paper's filter does not use classifier labels/loss.
        if imgs.ndim != 4 or imgs.shape[1] not in (1, 3):
            raise ValueError("Expected NCHW input with one or three channels")
        batch_size = len(imgs)
        keep = torch.ones(batch_size, dtype=torch.bool)
        if batch_size == 0:
            return keep, None
        x = canonicalize_detector(imgs.to(self.device), self._det_img_size)
        self.detector.eval()  # Freeze BatchNorm and dropout as well as parameters.
        scores = self.detector(x).reshape(-1).detach().cpu().double()
        if scores.numel() != batch_size or not torch.isfinite(scores).all():
            raise ValueError("The detector must return one finite score per sample")
        cap = math.floor(self.mcfg.rho_max * batch_size)
        if self._active and cap:
            if len(self.buf) < self.mcfg.min_history_size:
                # Rank raw logits before a calibration history exists. Ranking
                # all-zero standardized values would drop arbitrary examples.
                candidates = torch.arange(batch_size)
                ranking = scores
            else:
                center, scale = self._median_mad(torch.tensor(list(self.buf), dtype=torch.float64))
                ranking = (scores - center) / (scale + self.mcfg.eps)
                candidates = torch.where(ranking > self._z_crit)[0]
            # Do not clip large standardized scores before selecting the top c.
            order = torch.argsort(ranking[candidates], descending=True, stable=True)
            keep[candidates[order[:cap]]] = False
        self.buf.extend(scores[keep].tolist())
        self._seen += batch_size
        self._dropped += int((~keep).sum())
        return keep, None

    def keep(self, imgs):
        return self.keep_with_weights(imgs)[0]

    @property
    def drop_rate(self):
        return self._dropped / max(self._seen, 1)

    def get_calibration_stats(self):
        if not self.buf:
            return {"buf_len": 0, "z_crit": self._z_crit}
        center, scale = self._median_mad(torch.tensor(list(self.buf), dtype=torch.float64))
        return {"buf_len": len(self.buf), "buf_median": float(center),
                "buf_mad": float(scale), "z_crit": self._z_crit}
