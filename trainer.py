from __future__ import annotations
import math, random, time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path

from config import ExperimentConfig, MethodConfig
from attacks import HardNegAug, TriggerSampler, get_eval_triggers
from datasets import (CleanTestSet, DetectorDataset, MixedBaseDataset,
                       PoisonedDataset, TriggeredTestSet, get_raw_dataset,
                       _DETECTOR_REF_DCFG)
from models import (DetectorCNN, RandomScorer, build_classifier,
                     build_detector_model, build_frequency_scorer,
                     build_highpass_scorer, build_consistency_scorer)

def _ndtri(p: float) -> float:
    assert 0 < p < 1, f"p must be in (0,1), got {p}"
    if p >= 0.5:
        sign, q = 1.0, 1.0 - p
    else:
        sign, q = -1.0, p
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return sign * (t - num / den)

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

class Prefilter:

    def __init__(self, detector: nn.Module, mcfg: MethodConfig,
                 device: str, det_img_size: int = 0):
        self.detector      = detector
        self.mcfg          = mcfg
        self.device        = device
        self.buf           = deque(maxlen=mcfg.mad_buffer_size)
        self._fill_only    = True
        self._dropped      = 0
        self._seen         = 0
        self._epoch        = 0
        self._det_img_size = det_img_size

        if mcfg.use_lrt:
            alpha = mcfg.lrt_alpha
            self._z_crit = (_ndtri(1.0 - alpha / 2.0) if mcfg.use_bilateral
                            else _ndtri(1.0 - alpha))
        else:
            self._z_crit = mcfg.tau

        self.buf_fused = deque(maxlen=mcfg.mad_buffer_size)

        if mcfg.use_multiview:
            self._hp_scorer   = build_highpass_scorer()
            self._hp_scorer.eval()
            self.buf_b        = deque(maxlen=mcfg.mad_buffer_size)

            self._freq_scorer = build_frequency_scorer()
            self._freq_scorer.eval()
            n_freq_stats = getattr(self._freq_scorer, "K", 1)
            self.buf_c    = [deque(maxlen=mcfg.mad_buffer_size)
                             for _ in range(n_freq_stats)]
        else:
            self._hp_scorer   = None
            self._freq_scorer = None
            self.buf_b        = None
            self.buf_c        = None

        if mcfg.use_dynamics_view:
            self.buf_d = deque(maxlen=mcfg.mad_buffer_size)
        else:
            self.buf_d = None

        if getattr(mcfg, "use_consistency_view", False):
            self._cons_scorer = build_consistency_scorer()
            self._cons_scorer.eval()
            self.buf_e        = deque(maxlen=mcfg.mad_buffer_size)
        else:
            self._cons_scorer = None
            self.buf_e        = None

        self._dyn_ema_median: Optional[float] = None
        self._dyn_ema_alpha:  float = getattr(mcfg, "dynamics_ema_alpha", 0.9)

        if getattr(mcfg, "use_dual_threshold", False):
            self._tau_hard = self._z_crit
            self._tau_soft = self._z_crit
        else:
            self._tau_hard = self._z_crit
            self._tau_soft = self._z_crit

        self.buf_fq = (self.buf_c[0] if self.buf_c is not None else None)

        if mcfg.use_freq_bonus:
            self._freq_bonus_scorer = build_frequency_scorer()
            self._freq_bonus_scorer.eval()
            n_fstats = getattr(self._freq_bonus_scorer, "K", 1)
            self.buf_freq_bonus = [deque(maxlen=mcfg.mad_buffer_size)
                                   for _ in range(n_fstats)]
        else:
            self._freq_bonus_scorer = None
            self.buf_freq_bonus = None

        if mcfg.use_warp_bonus:
            self.buf_warp_bonus = deque(maxlen=mcfg.mad_buffer_size)
            if getattr(mcfg, "warp_bonus_adapt", False):
                self.buf_warp_z      = deque(maxlen=mcfg.mad_buffer_size)
                self._warp_tau_ema:  Optional[float] = None
                self._warp_adapt_min = getattr(mcfg, "warp_bonus_min_history", 64)
                self._warp_log: Optional[dict] = {
                    "n": 0, "sum_med": 0.0, "sum_mad": 0.0,
                    "sum_tau": 0.0, "sum_bonus": 0.0}
            else:
                self.buf_warp_z      = None
                self._warp_tau_ema   = None
                self._warp_log       = None
            self._last_warp_z: Optional[torch.Tensor] = None
        else:
            self.buf_warp_bonus  = None
            self.buf_warp_z      = None
            self._warp_tau_ema   = None
            self._warp_log       = None
            self._last_warp_z    = None

        self._unified = getattr(mcfg, "unified_channels", False)
        if self._unified:
            bsz = mcfg.mad_buffer_size
            self._buf_geo  = deque(maxlen=bsz) if mcfg.unified_geometry else None
            self._buf_sig  = deque(maxlen=bsz) if mcfg.unified_sig      else None
            self._buf_cons = deque(maxlen=bsz) if mcfg.unified_sig_cons else None
            self._tau_geo_ema:  Optional[float] = None
            self._tau_sig_ema:  Optional[float] = None
            self._tau_cons_ema: Optional[float] = None
            self._last_geo_raw:  Optional[torch.Tensor] = None
            self._last_sig_raw:  Optional[torch.Tensor] = None
            self._last_cons_raw: Optional[torch.Tensor] = None
            self._unified_log: dict = self._new_unified_log()
        else:
            self._buf_geo = self._buf_sig = self._buf_cons = None
            self._tau_geo_ema = self._tau_sig_ema = self._tau_cons_ema = None
            self._unified_log = None

    def set_active(self) -> None:
        self._fill_only = False

    def set_epoch(self, epoch: int) -> None:
        if epoch != self._epoch:
            self._flush_warp_log()
            self._flush_unified_log()
        self._epoch = epoch

    def _flush_warp_log(self) -> None:
        if self._warp_log is None or self._warp_log["n"] == 0:
            return
        log = self._warp_log
        n = log["n"]
        buf_z_len = len(self.buf_warp_z) if self.buf_warp_z is not None else 0
        print(f"  [WarpAdapt] ep={self._epoch}  "
              f"warp_raw: med={log['sum_med']/n:.4f} "
              f"mad={log['sum_mad']/n:.4f}  "
              f"tau_w={log['sum_tau']/n:.3f}  "
              f"mean_bonus={log['sum_bonus']/n:.4f}  "
              f"buf_z={buf_z_len}")
        self._warp_log = {"n": 0, "sum_med": 0.0, "sum_mad": 0.0,
                          "sum_tau": 0.0, "sum_bonus": 0.0}

    @property
    def is_active(self) -> bool:
        return not self._fill_only

    def _compute_view_zscore(self, raw_scores: torch.Tensor,
                             buf: deque) -> torch.Tensor:
        if len(buf) < 64:
            return torch.zeros_like(raw_scores)

        buf_t = torch.tensor(list(buf))
        if self.mcfg.use_lrt:
            mu0  = buf_t.median()
            mad0 = ((buf_t - mu0).abs().median() * 1.4826).clamp(min=1e-3)
            z    = (raw_scores - mu0) / (mad0 + self.mcfg.eps)
        else:
            med_all = buf_t.median()
            upper   = buf_t[buf_t >= med_all]
            med     = upper.median()
            mad     = ((upper - med).abs().median() * 1.4826).clamp(min=1e-3)
            z       = (raw_scores - med) / (mad + self.mcfg.eps)

        return z.abs() if self.mcfg.use_bilateral else z

    def _onesided_high_zscore(self, raw_scores: torch.Tensor,
                              buf: deque) -> torch.Tensor:
        if len(buf) < 64:
            return torch.zeros_like(raw_scores)
        buf_t = torch.tensor(list(buf))
        mu    = buf_t.median()
        sigma = ((buf_t - mu).abs().median() * 1.4826).clamp(min=1e-3)
        z     = (raw_scores - mu) / (sigma + self.mcfg.eps)
        return z.clamp(min=0.0)

    @staticmethod
    def _new_unified_log() -> dict:
        return {"n": 0,
                "geo": 0.0, "sig": 0.0, "cons": 0.0,
                "tau_geo": 0.0, "tau_sig": 0.0, "tau_cons": 0.0}

    def _adaptive_tau_from_raw(self, raw_buf: deque, q: float,
                               beta: float,
                               prev_ema: Optional[float],
                               ) -> Tuple[Optional[float], float, float]:
        min_h = self.mcfg.unified_min_history
        if len(raw_buf) < min_h:
            return None, 0.0, 0.0
        buf_t = torch.tensor(list(raw_buf))
        mu    = float(buf_t.median())
        sigma = float(((buf_t - mu).abs().median() * 1.4826).clamp(min=1e-3))
        sorted_vals = sorted(raw_buf)
        q_idx   = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
        theta   = float(sorted_vals[q_idx])
        tau_raw = max(0.0, (theta - mu) / (sigma + self.mcfg.eps))
        if prev_ema is None or beta == 0.0:
            tau_ema = tau_raw
        else:
            tau_ema = beta * prev_ema + (1.0 - beta) * tau_raw
        return tau_ema, mu, sigma

    def _geometry_raw(self, imgs: torch.Tensor,
                      imgs_det: torch.Tensor,
                      pix_logits: torch.Tensor) -> torch.Tensor:
        H, W = imgs.shape[-2:]
        r   = self.mcfg.unified_warp_ratio
        Hs  = max(2, int(H * r))
        Ws  = max(2, int(W * r))
        imgs_r = F.interpolate(
            F.interpolate(imgs, size=(Hs, Ws),
                          mode="bilinear", align_corners=False),
            size=(H, W), mode="bilinear", align_corners=False)
        if self._det_img_size > 0 and imgs_r.shape[-1] != self._det_img_size:
            imgs_r = F.interpolate(imgs_r, size=self._det_img_size,
                                   mode="bilinear", align_corners=False)
        logits_r = self.detector(imgs_r.to(self.device)).cpu()
        return (pix_logits - logits_r).abs()

    def _sig_projection_raw(self, imgs: torch.Tensor) -> torch.Tensor:
        gray = imgs.mean(dim=1)
        r    = gray.mean(dim=2)
        H    = r.shape[1]
        f    = self.mcfg.unified_sig_target_cycles
        ys   = torch.arange(H, dtype=torch.float32, device=imgs.device)
        sin_f = torch.sin(2 * math.pi * f * ys / H)
        cos_f = torch.cos(2 * math.pi * f * ys / H)
        proj_sin = (r * sin_f).sum(dim=1)
        proj_cos = (r * cos_f).sum(dim=1)
        return (proj_sin.pow(2) + proj_cos.pow(2)).sqrt()

    def _sig_consistency_raw(self, imgs: torch.Tensor,
                             imgs_det: torch.Tensor,
                             pix_logits: torch.Tensor) -> torch.Tensor:
        H = imgs.shape[-2]
        k = max(3, round(H / self.mcfg.unified_sig_target_cycles))
        if k % 2 == 0:
            k += 1
        pad = k // 2
        imgs_s = F.avg_pool2d(imgs, kernel_size=(k, 1), stride=1,
                              padding=(pad, 0))
        if self._det_img_size > 0 and imgs_s.shape[-1] != self._det_img_size:
            imgs_s = F.interpolate(imgs_s, size=self._det_img_size,
                                   mode="bilinear", align_corners=False)
        logits_s = self.detector(imgs_s.to(self.device)).cpu()
        return (pix_logits - logits_s).abs()

    def _compute_unified_channels(
        self,
        imgs: torch.Tensor,
        imgs_det: torch.Tensor,
        pix_logits: torch.Tensor,
    ) -> torch.Tensor:
        bonus = torch.zeros_like(pix_logits)
        q    = self.mcfg.unified_q
        beta = self.mcfg.unified_tau_beta
        log  = self._unified_log

        if self._buf_geo is not None:
            geo_raw = self._geometry_raw(imgs, imgs_det, pix_logits)
            z_geo   = self._onesided_high_zscore(geo_raw, self._buf_geo)
            tau_result = self._adaptive_tau_from_raw(
                self._buf_geo, q, beta, self._tau_geo_ema)
            if tau_result[0] is not None:
                self._tau_geo_ema = tau_result[0]
                geo_c = self.mcfg.unified_lambda_w * (
                    z_geo - self._tau_geo_ema).clamp(min=0.0)
            else:
                geo_c = torch.zeros_like(pix_logits)
            bonus = bonus + geo_c
            self._last_geo_raw = geo_raw
            if log is not None and tau_result[0] is not None:
                log["n"] += 1
                log["geo"]     += float(geo_c.mean())
                log["tau_geo"] += self._tau_geo_ema

        if self._buf_sig is not None:
            sig_raw = self._sig_projection_raw(imgs)
            z_sig   = self._onesided_high_zscore(sig_raw, self._buf_sig)
            tau_result = self._adaptive_tau_from_raw(
                self._buf_sig, q, beta, self._tau_sig_ema)
            if tau_result[0] is not None:
                self._tau_sig_ema = tau_result[0]
                sig_c = self.mcfg.unified_lambda_s * (
                    z_sig - self._tau_sig_ema).clamp(min=0.0)
            else:
                sig_c = torch.zeros_like(pix_logits)
            bonus = bonus + sig_c
            self._last_sig_raw = sig_raw
            if log is not None and tau_result[0] is not None:
                if log["n"] == 0:
                    log["n"] = 1
                log["sig"]     += float(sig_c.mean())
                log["tau_sig"] += self._tau_sig_ema

        if self._buf_cons is not None:
            cons_raw = self._sig_consistency_raw(imgs, imgs_det, pix_logits)
            z_cons   = self._onesided_high_zscore(cons_raw, self._buf_cons)
            tau_result = self._adaptive_tau_from_raw(
                self._buf_cons, q, beta, self._tau_cons_ema)
            if tau_result[0] is not None:
                self._tau_cons_ema = tau_result[0]
                cons_c = self.mcfg.unified_lambda_c * (
                    z_cons - self._tau_cons_ema).clamp(min=0.0)
            else:
                cons_c = torch.zeros_like(pix_logits)
            bonus = bonus + cons_c
            self._last_cons_raw = cons_raw
            if log is not None and tau_result[0] is not None:
                if log["n"] == 0:
                    log["n"] = 1
                log["cons"]     += float(cons_c.mean())
                log["tau_cons"] += self._tau_cons_ema

        return bonus

    def _flush_unified_log(self) -> None:
        if self._unified_log is None or self._unified_log["n"] == 0:
            return
        log = self._unified_log
        n   = log["n"]
        parts = [f"ep={self._epoch}"]
        if self._buf_geo is not None:
            parts.append(f"geo={log['geo']/n:.4f}(τ={log['tau_geo']/n:.3f})")
        if self._buf_sig is not None:
            parts.append(f"sig={log['sig']/n:.4f}(τ={log['tau_sig']/n:.3f})")
        if self._buf_cons is not None:
            parts.append(f"cons={log['cons']/n:.4f}(τ={log['tau_cons']/n:.3f})")
        print(f"  [Unified] {' '.join(parts)}")
        self._unified_log = self._new_unified_log()

    def _compute_bonus(
        self,
        imgs: torch.Tensor,
        imgs_det: torch.Tensor,
        pix_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        bonus    = torch.zeros_like(pix_logits)
        freq_raw = None
        warp_raw = None

        if self._freq_bonus_scorer is not None and self.buf_freq_bonus is not None:
            freq_raw = self._freq_bonus_scorer(imgs)
            if freq_raw.dim() == 2:
                zf_list = [self._onesided_high_zscore(freq_raw[:, k], buf_k)
                           for k, buf_k in enumerate(self.buf_freq_bonus)]
                z_freq = torch.stack(zf_list, dim=1).max(dim=1).values
            else:
                z_freq = self._onesided_high_zscore(freq_raw, self.buf_freq_bonus[0])
            bonus = bonus + self.mcfg.freq_bonus_lambda * (
                z_freq - self.mcfg.freq_bonus_tau).clamp(min=0.0)

        if self.mcfg.use_warp_bonus and self.buf_warp_bonus is not None:
            H_orig, W_orig = imgs.shape[-2:]
            r   = self.mcfg.warp_resample_ratio
            H_s = max(2, int(H_orig * r))
            W_s = max(2, int(W_orig * r))
            imgs_resamp = F.interpolate(
                F.interpolate(imgs, size=(H_s, W_s),
                              mode='bilinear', align_corners=False),
                size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            if self._det_img_size > 0 and imgs_resamp.shape[-1] != self._det_img_size:
                imgs_resamp_det = F.interpolate(
                    imgs_resamp, size=self._det_img_size,
                    mode='bilinear', align_corners=False)
            else:
                imgs_resamp_det = imgs_resamp
            logits_resamp = self.detector(imgs_resamp_det.to(self.device)).cpu()
            warp_raw = (pix_logits - logits_resamp).abs()
            z_warp = self._onesided_high_zscore(warp_raw, self.buf_warp_bonus)

            adapt = getattr(self.mcfg, "warp_bonus_adapt", False)
            if adapt and self.buf_warp_z is not None:
                min_hist = getattr(self, "_warp_adapt_min", 64)
                if len(self.buf_warp_z) >= min_hist:
                    q       = self.mcfg.warp_bonus_q
                    sorted_z = sorted(self.buf_warp_z)
                    q_idx   = min(int(len(sorted_z) * q), len(sorted_z) - 1)
                    tau_curr = float(sorted_z[q_idx])
                    beta     = self.mcfg.warp_bonus_tau_beta
                    if self._warp_tau_ema is None or beta == 0.0:
                        self._warp_tau_ema = tau_curr
                    else:
                        self._warp_tau_ema = (beta * self._warp_tau_ema
                                              + (1.0 - beta) * tau_curr)
                    tau_w = self._warp_tau_ema
                else:
                    tau_w = float("inf")
                self._last_warp_z = z_warp
            else:
                tau_w = self.mcfg.warp_bonus_tau
                self._last_warp_z = None

            bonus_w = self.mcfg.warp_bonus_lambda * (
                z_warp - tau_w).clamp(min=0.0)
            bonus = bonus + bonus_w

            if self._warp_log is not None and len(self.buf_warp_bonus) >= 64:
                buf_t = torch.tensor(list(self.buf_warp_bonus))
                med   = float(buf_t.median())
                mad   = float((buf_t - med).abs().median() * 1.4826)
                self._warp_log["n"]          += 1
                self._warp_log["sum_med"]    += med
                self._warp_log["sum_mad"]    += mad
                eff_tau = tau_w if tau_w != float("inf") else 0.0
                self._warp_log["sum_tau"]    += eff_tau
                self._warp_log["sum_bonus"]  += float(bonus_w.mean())

        return bonus, freq_raw, warp_raw

    @staticmethod
    def _fuse_view_scores(zv_list: list, fusion_mode: str,
                          a_cap: float) -> torch.Tensor:
        clipped = [z.clamp(min=0.0, max=a_cap) for z in zv_list]
        stacked = torch.stack(clipped, dim=1)
        if fusion_mode == "top2mean":
            k = min(2, stacked.size(1))
            top_vals, _ = stacked.topk(k, dim=1)
            return top_vals.mean(dim=1)
        if fusion_mode == "logsumexp":
            return torch.logsumexp(stacked, dim=1)
        return stacked.max(dim=1).values

    def _compute_rolling_tau_soft(self) -> float:
        if len(self.buf_fused) < 64:
            return self._z_crit

        alpha   = getattr(self.mcfg, "rolling_alpha",   0.05)
        win_top = getattr(self.mcfg, "rolling_win_top", 0.01)

        arr  = sorted(self.buf_fused)
        n    = len(arr)
        cut  = max(1, int(n * (1.0 - win_top)))
        arr  = arr[:cut]
        q_idx = min(int(len(arr) * (1.0 - alpha / 2.0)), len(arr) - 1)
        return float(arr[q_idx])

    def _get_fused_scores(
        self,
        imgs: torch.Tensor,
        classifier_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list,
               torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        imgs_det = imgs
        if self._det_img_size > 0 and imgs.shape[-1] != self._det_img_size:
            imgs_det = F.interpolate(imgs, size=self._det_img_size,
                                     mode="bilinear", align_corners=False)

        pix_logits = self.detector(imgs_det.to(self.device)).cpu()
        z_a        = self._compute_view_zscore(pix_logits, self.buf)

        if self._unified:
            z_a = z_a + self._compute_unified_channels(imgs, imgs_det, pix_logits)

        bonus_freq_raw = None
        bonus_warp_raw = None
        if (not self._unified) and (self.mcfg.use_freq_bonus or self.mcfg.use_warp_bonus):
            bonus, bonus_freq_raw, bonus_warp_raw = self._compute_bonus(
                imgs, imgs_det, pix_logits)
            z_a = z_a + bonus

        a_cap        = getattr(self.mcfg, "a_cap", 6.0)
        fusion_mode  = getattr(self.mcfg, "fusion_mode", "max")

        zv_list    = [z_a]
        hp_scores  = None
        fq_scores  = None

        if self.mcfg.use_multiview:
            hp_scores = self._hp_scorer(imgs)
            fq_scores = self._freq_scorer(imgs)
            z_b       = self._compute_view_zscore(hp_scores, self.buf_b)
            zv_list.append(z_b)
            if fq_scores.dim() == 2:
                for k, buf_ck in enumerate(self.buf_c):
                    z_ck = self._compute_view_zscore(fq_scores[:, k], buf_ck)
                    zv_list.append(z_ck)
            else:
                z_c = self._compute_view_zscore(fq_scores, self.buf_c[0])
                zv_list.append(z_c)

        if self.mcfg.use_dynamics_view and classifier_loss is not None:
            dyn_signal = classifier_loss
            if getattr(self.mcfg, "use_dynamics_delta", False):
                if self._dyn_ema_median is not None:
                    dyn_signal = classifier_loss - self._dyn_ema_median
                cur_med = float(classifier_loss.median().item())
                a       = self._dyn_ema_alpha
                self._dyn_ema_median = (
                    cur_med if self._dyn_ema_median is None
                    else a * self._dyn_ema_median + (1.0 - a) * cur_med
                )
            z_d = self._compute_view_zscore(dyn_signal, self.buf_d)
            zv_list.append(z_d)

        if self._cons_scorer is not None and self.buf_e is not None:
            cons_scores = self._cons_scorer(imgs)
            z_e = self._compute_view_zscore(cons_scores, self.buf_e)
            zv_list.append(z_e)
        else:
            cons_scores = None

        fused = self._fuse_view_scores(zv_list, fusion_mode, a_cap)
        return (fused, zv_list, pix_logits, hp_scores, fq_scores,
                cons_scores, bonus_freq_raw, bonus_warp_raw)

    def _update_buffers(
        self,
        keep_mask: torch.Tensor,
        pix_logits: torch.Tensor,
        hp_scores: Optional[torch.Tensor],
        fq_scores: Optional[torch.Tensor],
        classifier_loss: Optional[torch.Tensor],
        cons_scores: Optional[torch.Tensor] = None,
        bonus_freq_raw: Optional[torch.Tensor] = None,
        bonus_warp_raw: Optional[torch.Tensor] = None,
    ) -> None:
        self.buf.extend(pix_logits[keep_mask].tolist())

        if self.mcfg.use_multiview and self.buf_b is not None:
            if hp_scores is not None:
                self.buf_b.extend(hp_scores[keep_mask].tolist())
            if fq_scores is not None and self.buf_c is not None:
                if fq_scores.dim() == 2:
                    for k, buf_ck in enumerate(self.buf_c):
                        buf_ck.extend(fq_scores[keep_mask, k].tolist())
                else:
                    self.buf_c[0].extend(fq_scores[keep_mask].tolist())

        if (self.mcfg.use_dynamics_view and self.buf_d is not None
                and classifier_loss is not None):
            self.buf_d.extend(classifier_loss[keep_mask].tolist())

        if self.buf_e is not None and cons_scores is not None:
            self.buf_e.extend(cons_scores[keep_mask].tolist())

        if self.buf_freq_bonus is not None and bonus_freq_raw is not None:
            if bonus_freq_raw.dim() == 2:
                for k, buf_k in enumerate(self.buf_freq_bonus):
                    buf_k.extend(bonus_freq_raw[keep_mask, k].tolist())
            else:
                self.buf_freq_bonus[0].extend(bonus_freq_raw[keep_mask].tolist())
        if self.buf_warp_bonus is not None and bonus_warp_raw is not None:
            self.buf_warp_bonus.extend(bonus_warp_raw[keep_mask].tolist())
            if (self.buf_warp_z is not None
                    and self._last_warp_z is not None):
                self.buf_warp_z.extend(
                    self._last_warp_z[keep_mask].tolist())
                self._last_warp_z = None

        if self._unified:
            if self._buf_geo is not None and self._last_geo_raw is not None:
                self._buf_geo.extend(self._last_geo_raw[keep_mask].tolist())
                self._last_geo_raw = None
            if self._buf_sig is not None and self._last_sig_raw is not None:
                self._buf_sig.extend(self._last_sig_raw[keep_mask].tolist())
                self._last_sig_raw = None
            if self._buf_cons is not None and self._last_cons_raw is not None:
                self._buf_cons.extend(self._last_cons_raw[keep_mask].tolist())
                self._last_cons_raw = None

    @torch.no_grad()
    def keep(self, imgs: torch.Tensor) -> torch.Tensor:
        keep_mask, _ = self.keep_with_weights(imgs)
        return keep_mask

    @torch.no_grad()
    def keep_with_weights(
        self,
        imgs: torch.Tensor,
        classifier_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = imgs.size(0)
        self._seen += B

        (fused, zv_list, pix_logits, hp_scores, fq_scores,
         cons_scores, bonus_freq_raw, bonus_warp_raw) = (
            self._get_fused_scores(imgs, classifier_loss))

        def _do_update(mask: torch.Tensor) -> None:
            self._update_buffers(mask, pix_logits, hp_scores, fq_scores,
                                 classifier_loss, cons_scores,
                                 bonus_freq_raw, bonus_warp_raw)
            self.buf_fused.extend(fused[mask].tolist())

        if self._fill_only:
            all_mask = torch.ones(B, dtype=torch.bool)
            _do_update(all_mask)
            return all_mask, (torch.ones(B) if self.mcfg.use_soft_weight else None)

        cap = int(math.floor(self.mcfg.rho_max * B))
        if cap == 0:
            all_mask = torch.ones(B, dtype=torch.bool)
            _do_update(all_mask)
            return all_mask, (torch.ones(B) if self.mcfg.use_soft_weight else None)

        buf_ready = self.mcfg.use_mad and len(self.buf) >= 64
        use_dual  = getattr(self.mcfg, "use_dual_threshold", False)

        use_topk = self.mcfg.always_topk and (
            self.mcfg.always_topk_epochs == 0
            or self._epoch < self.mcfg.always_topk_epochs)

        if use_topk:
            flag   = torch.zeros(B, dtype=torch.bool)
            _, top = fused.topk(min(cap, B))
            flag[top] = True

        elif buf_ready:
            if use_dual:
                tau = float(self._tau_hard)
            elif self.mcfg.use_lrt:
                tau = self._z_crit
            elif self.mcfg.auto_tau:
                buf_t   = torch.tensor(list(self.buf))
                med_all = buf_t.median()
                upper   = buf_t[buf_t >= med_all]
                med     = upper.median()
                mad     = ((upper - med).abs().median() * 1.4826).clamp(min=1e-3)
                T   = torch.quantile(buf_t, 1.0 - self.mcfg.alpha)
                tau = float(((T - med) / (mad + self.mcfg.eps)).clamp(
                    self.mcfg.tau_min, self.mcfg.tau_max))
            else:
                tau = self.mcfg.tau

            min_votes = min(getattr(self.mcfg, "min_votes", 1), len(zv_list))
            a_cap     = getattr(self.mcfg, "a_cap", 6.0)
            votes = sum(
                (z.clamp(max=a_cap) > tau).long() for z in zv_list
            )
            flag = votes >= min_votes
            if flag.sum().item() > cap:
                tmp        = fused.clone()
                tmp[~flag] = float("-inf")
                _, top     = tmp.topk(cap)
                flag       = torch.zeros(B, dtype=torch.bool)
                flag[top]  = True

            if use_dual:
                rho_hat    = flag.float().mean().item()
                rho_target = getattr(self.mcfg, "rho_target",  0.05)
                tau_lr     = getattr(self.mcfg, "tau_lr",       0.01)
                new_tau = tau + tau_lr * (rho_hat - rho_target)
                self._tau_hard = float(
                    max(self._z_crit * 0.3,
                        min(self._z_crit * 5.0, new_tau)))

        elif self.mcfg.use_mad:
            flag   = torch.zeros(B, dtype=torch.bool)
            _, top = fused.topk(min(cap, B))
            flag[top] = True

        else:
            flag = fused > 0.0
            if flag.sum().item() > cap:
                tmp        = fused.clone()
                tmp[~flag] = float("-inf")
                _, top     = tmp.topk(cap)
                flag       = torch.zeros(B, dtype=torch.bool)
                flag[top]  = True

        keep_mask = ~flag
        _do_update(keep_mask)
        self._dropped += flag.sum().item()

        if self.mcfg.use_soft_weight:
            eta   = getattr(self.mcfg, "soft_weight_eta", self.mcfg.soft_weight_k)
            w_min = getattr(self.mcfg, "w_min", 1e-3)

            if use_dual and len(self.buf_fused) >= 64:
                tau_soft = self._compute_rolling_tau_soft()
                self._tau_soft = tau_soft
            else:
                tau_soft = self._z_crit if self.mcfg.use_lrt else self.mcfg.tau

            soft_w = torch.exp(-eta * (fused - tau_soft).clamp(min=0.0))
            soft_w = soft_w.clamp(min=w_min, max=1.0)
            return keep_mask, soft_w
        else:
            return keep_mask, None

    @property
    def drop_rate(self) -> float:
        return self._dropped / max(self._seen, 1)

    def get_calibration_stats(self) -> dict:
        stats: dict = {}
        if len(self.buf_fused) < 4:
            return stats
        buf_t = torch.tensor(list(self.buf_fused))
        mu  = float(buf_t.median())
        mad = float(((buf_t - mu).abs().median() * 1.4826).clamp(min=1e-3))
        stats["buf_median"]  = mu
        stats["buf_mad"]     = mad
        stats["buf_len"]     = len(self.buf_fused)
        stats["tau_hard"]    = float(self._tau_hard)
        stats["tau_soft"]    = float(self._tau_soft)
        stats["z_crit"]      = float(self._z_crit)
        return stats

def build_and_train_detector(ecfg: ExperimentConfig) -> nn.Module:
    """Load a frozen scorer or train it on an explicitly supplied auxiliary split.

    There is deliberately no fallback to the downstream training dataset.
    """
    from dataclasses import replace
    from datasets import AuxiliaryImageDataset
    mcfg, tcfg, dev = ecfg.method, ecfg.train, ecfg.train.device
    det_cfg = replace(_DETECTOR_REF_DCFG, image_size=tcfg.detector_image_size)
    if not mcfg.train_detector:
        return build_detector_model(det_cfg, trained=False).to(dev).eval()
    detector = build_detector_model(det_cfg, trained=True,
                                    use_fft=mcfg.detector_use_fft).to(dev)
    ckpt_path = Path(tcfg.detector_ckpt_path) if tcfg.detector_ckpt_path else None
    if ckpt_path is not None and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=dev, weights_only=True)
        if isinstance(payload, dict) and 'state_dict' in payload:
            expected = 'DetectorCNNFFT' if mcfg.detector_use_fft else 'DetectorCNN'
            metadata = payload.get('metadata', {})
            if metadata.get('architecture', expected) != expected:
                raise ValueError('Checkpoint detector architecture does not match configuration')
            if metadata.get('image_size', tcfg.detector_image_size) != tcfg.detector_image_size:
                raise ValueError('Checkpoint detector resolution does not match configuration')
            payload = payload['state_dict']
        detector.load_state_dict(payload)
        print(f"  [Detector] Loaded frozen checkpoint: {ckpt_path}")
        return detector.eval().requires_grad_(False)
    if not tcfg.detector_aux_root:
        raise ValueError(
            'BRIFi requires an existing --detector-checkpoint. Train a scorer '
            'separately with the train-detector command and an independent '
            '--aux-root; downstream data are never used as a fallback.')
    base_ds = AuxiliaryImageDataset(tcfg.detector_aux_root, tcfg.detector_image_size)
    families = ecfg.attack.detector_families or ['badnets', 'blended', 'noise', 'wanet']
    # The paper samples the four surrogate families uniformly.
    ts = TriggerSampler(det_cfg, families=families, weights=[1.] * len(families))
    det_ds = DetectorDataset(base_ds, det_cfg, ts, tcfg.detector_base_n, seed=tcfg.seed)
    det_dl = DataLoader(det_ds, 256, shuffle=True, num_workers=tcfg.num_workers,
                        pin_memory=dev.startswith('cuda'))
    optimizer = optim.Adam(detector.parameters(), lr=tcfg.detector_lr, weight_decay=1e-4)
    for epoch in range(tcfg.detector_epochs):
        detector.train()
        loss_sum = correct = count = 0
        for images, targets in det_dl:
            images, targets = images.to(dev), targets.to(dev)
            logits = detector(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += len(images)
            loss_sum += loss.item() * len(images)
            correct += int(((logits > 0).float() == targets).sum())
        print(f"  [Detector] epoch={epoch+1} loss={loss_sum/count:.4f} acc={correct/count:.4f}")
    detector.eval().requires_grad_(False)
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': detector.state_dict(), 'metadata': {
            'architecture': type(detector).__name__, 'image_size': tcfg.detector_image_size,
            'auxiliary_base_images': len(det_ds.pil_imgs), 'families': families,
            'seed': tcfg.seed, 'detector_epochs': tcfg.detector_epochs,
            'source': 'user-provided auxiliary training split'}}, ckpt_path)
    return detector

def detector_separation_stats(detector: nn.Module, ecfg: ExperimentConfig) -> dict:
    dcfg  = ecfg.dataset
    tcfg  = ecfg.train
    dev   = tcfg.device
    ts    = TriggerSampler(dcfg)

    raw_te    = get_raw_dataset(dcfg, train=False)
    rng       = np.random.RandomState(789)
    sel       = rng.choice(len(raw_te), min(500, len(raw_te)), replace=False)

    from datasets import _make_to_tensor, _make_aug
    aug_tf    = _make_aug(dcfg, True)
    to_tensor = _make_to_tensor(dcfg)

    clean_list, poison_list = [], []
    for i in sel:
        pil = raw_te[i][0]
        aug = aug_tf(pil) if aug_tf else pil
        t   = to_tensor(aug)
        clean_list.append(t)
        poison_list.append(ts(t))

    @torch.no_grad()
    def _get_logits(lst):
        out = []
        detector.eval()
        for s in range(0, len(lst), 256):
            b = torch.stack(lst[s:s+256]).to(dev)
            from brifi import canonicalize_detector
            b = canonicalize_detector(b, tcfg.detector_image_size)
            out.append(detector(b).cpu())
        return torch.cat(out)

    cl = _get_logits(clean_list)
    po = _get_logits(poison_list)

    stats = {}
    for tag, vals in [("clean", cl), ("poison", po)]:
        q = {f"q{int(p*100)}": vals.quantile(p).item() for p in (0.1, 0.5, 0.9)}
        stats[tag] = q
    if _HAS_SK:
        labels = np.array([0]*len(cl) + [1]*len(po))
        scores = torch.cat([cl, po]).numpy()
        stats["auc"] = roc_auc_score(labels, scores)
        print(f"  [DetStats] AUC={stats['auc']:.4f}  "
              f"clean_q50={stats['clean']['q50']:+.3f}  "
              f"poison_q50={stats['poison']['q50']:+.3f}")
    return stats

def train_epoch(model: nn.Module, loader: DataLoader,
                opt: optim.Optimizer, dev: str,
                pf: Optional[Prefilter] = None,
                poison_flags: Optional[np.ndarray] = None
                ) -> Tuple[float, float, int, int]:
    model.train()
    ls = ok = n = ep_drop = ep_drop_p = 0

    for imgs, labs, idx in loader:
        if pf is not None:
            dyn_signal_cpu: Optional[torch.Tensor] = None
            if pf.mcfg.use_dynamics_view:
                with torch.no_grad():
                    _out_all = model(imgs.to(dev))
                    labs_d   = labs.to(dev)
                    target_logits = _out_all.gather(
                        1, labs_d.view(-1, 1)).squeeze(1)
                    _masked = _out_all.clone()
                    _masked.scatter_(1, labs_d.view(-1, 1), float("-inf"))
                    max_other = _masked.max(dim=1).values
                    dyn_signal_cpu = (target_logits - max_other).cpu()

            keep_mask, soft_w = pf.keep_with_weights(
                imgs, classifier_loss=dyn_signal_cpu)
            n_drop = (~keep_mask).sum().item()
            ep_drop += n_drop
            if poison_flags is not None and n_drop > 0:
                ep_drop_p += int(poison_flags[idx[~keep_mask].numpy()].sum())
            if keep_mask.sum() == 0:
                continue
            imgs_k = imgs[keep_mask].to(dev)
            labs_k = labs[keep_mask].to(dev)
        else:
            imgs_k, labs_k = imgs.to(dev), labs.to(dev)
            soft_w = None
            keep_mask = None

        out = model(imgs_k)
        if (soft_w is not None and pf is not None
                and pf.mcfg.use_soft_weight and keep_mask is not None):
            w_k = soft_w[keep_mask].to(dev)
            per_sample = F.cross_entropy(out, labs_k, reduction='none')
            l = (per_sample * w_k).sum() / w_k.sum().clamp(min=1e-6)
        else:
            l = F.cross_entropy(out, labs_k)

        opt.zero_grad(); l.backward(); opt.step()

        ls += l.item() * labs_k.size(0)
        ok += (out.argmax(1) == labs_k).sum().item()
        n  += labs_k.size(0)

    return ls / max(n, 1), ok / max(n, 1), ep_drop, ep_drop_p

@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, dev: str) -> float:
    model.eval()
    ok = n = 0
    for x, y in loader:
        x = x.to(dev)
        ok += (model(x).argmax(1) == y.to(dev)).sum().item()
        n  += x.size(0)
    return ok / max(n, 1)

@torch.no_grad()
def eval_asr(model: nn.Module, loader: DataLoader, tgt: int, dev: str) -> float:
    model.eval()
    hit = n = 0
    for x, y in loader:
        x = x.to(dev)
        p = model(x).argmax(1)
        m = (y.to(dev) != tgt)
        hit += (p[m] == tgt).sum().item()
        n   += m.sum().item()
    return hit / max(n, 1)

@torch.no_grad()
def eval_asr_suite(model: nn.Module, ecfg: ExperimentConfig) -> Dict[str, float]:
    dcfg, dev = ecfg.dataset, ecfg.train.device
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        raw_te = get_raw_dataset(dcfg, train=False)
        out: Dict[str, float] = {}
        for index, (name, trigger) in enumerate(get_eval_triggers(dcfg).items()):
            # The samplers use Python, NumPy, and torch CPU random sources.
            random.seed(42 + index)
            np.random.seed(42 + index)
            torch.random.default_generator.manual_seed(42 + index)
            dataset = TriggeredTestSet(raw_te, dcfg, trigger)
            loader = DataLoader(dataset, 512, num_workers=0)
            out[name] = eval_asr(model, loader, dcfg.target_label, dev)
        return out
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

def run_one_experiment(ecfg: ExperimentConfig) -> dict:

    dcfg  = ecfg.dataset
    acfg  = ecfg.attack
    mcfg  = ecfg.method
    tcfg  = ecfg.train
    dev   = tcfg.device

    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)

    print(f"\n{'='*62}\n  {ecfg.exp_id}\n{'='*62}")

    detector = build_and_train_detector(ecfg)
    if mcfg.train_detector:
        det_stats = detector_separation_stats(detector, ecfg)
    else:
        det_stats = {}

    attack_ts = TriggerSampler(dcfg, families=[acfg.name])
    raw_tr    = get_raw_dataset(dcfg, train=True)
    raw_te    = get_raw_dataset(dcfg, train=False)

    tr_ds = PoisonedDataset(raw_tr, dcfg, acfg.poison_rate, attack_ts, aug=True,
                            seed=tcfg.seed)
    cl_ds = CleanTestSet(raw_te, dcfg)

    tr_dl = DataLoader(tr_ds, tcfg.batch_size, shuffle=True,
                       num_workers=tcfg.num_workers, pin_memory=True)
    cl_dl = DataLoader(cl_ds, 512, num_workers=tcfg.num_workers)

    pf = None
    if mcfg.name != "baseline":
        from brifi import BudgetedFilter
        filter_type = BudgetedFilter if mcfg.name == "brifi" else Prefilter
        pf = filter_type(detector, mcfg, dev, det_img_size=tcfg.detector_image_size)

    poison_flags = tr_ds.is_poisoned if pf is not None else None
    total_poison = int(tr_ds.is_poisoned.sum())

    np.random.seed(tcfg.seed)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)
    model = build_classifier(dcfg).to(dev)
    opt   = optim.SGD(model.parameters(), lr=tcfg.lr,
                      momentum=tcfg.momentum, weight_decay=tcfg.weight_decay)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, tcfg.epochs)

    active_drops: List[int] = []
    active_drop_p: List[int] = []
    epoch_history: List[dict] = []

    _cum_drop_p = 0
    _cum_drop   = 0

    print(f"\n  {'Ep':>4}  {'loss':>6}  {'tacc':>6}  {'CleanACC':>9}"
          f"  {'DropRate':>9}  {'Prec':>6}  {'Rec':>6}  {'s':>5}")
    print(f"  {'-'*70}")

    for ep in range(tcfg.epochs):
        if pf is not None:
            pf.set_epoch(ep)
            if not pf.is_active:
                if not mcfg.use_warmup or ep >= mcfg.warmup_epochs:
                    pf.set_active()
                    label = "ep 0+" if ep == 0 else f"ep {ep+1}+"
                    print(f"  {'·'*35}  [Prefilter active {label}]")

        t0 = time.time()
        l, ta, ep_drop, ep_drop_p = train_epoch(
            model, tr_dl, opt, dev, pf, poison_flags)
        sch.step()
        dt = time.time() - t0

        if pf is not None and pf.is_active:
            active_drops.append(ep_drop)
            active_drop_p.append(ep_drop_p)

        ca   = eval_acc(model, cl_dl, dev)
        dr   = pf.drop_rate if pf else 0.0
        prec = ep_drop_p / max(ep_drop, 1)    if pf else 0.0
        rec  = ep_drop_p / max(total_poison, 1) if pf else 0.0
        print(f"  {ep+1:4d}  {l:6.3f}  {ta:6.3f}  {ca:9.4f}"
              f"  {dr:9.4f}  {prec:6.3f}  {rec:6.3f}  {dt:4.1f}s")

        if pf is not None:
            _cum_drop_p += ep_drop_p
            _cum_drop   += ep_drop
            # Samples are revisited each epoch, not permanently deleted.
            remaining_poison = max(total_poison - ep_drop_p, 0)
            remaining_total  = max(len(tr_ds) - ep_drop, 1)
            eff_pr = remaining_poison / remaining_total
            ep_dr  = ep_drop / max(len(tr_dl.dataset), 1) if hasattr(tr_dl, 'dataset') else 0.0
            rec_ep = ep_drop_p / max(total_poison, 1)
            fdrop  = (ep_drop - ep_drop_p) / max(len(tr_dl.dataset) - total_poison, 1)
            epoch_history.append({
                "epoch":               ep + 1,
                "loss":                l,
                "train_acc":           ta,
                "clean_acc":           ca,
                "ep_drop":             ep_drop,
                "ep_drop_p":           ep_drop_p,
                "ep_prec":             prec,
                "ep_rec":              rec_ep,
                "ep_false_drop_rate":  fdrop,
                "cumulative_drop_rate": dr,
                "effective_poison_rate": eff_pr,
                **pf.get_calibration_stats(),
            })

    if pf is not None:
        if hasattr(pf, "_flush_warp_log"):
            pf._flush_warp_log()
        if hasattr(pf, "_flush_unified_log"):
            pf._flush_unified_log()

    ca        = eval_acc(model, cl_dl, dev)
    asr_suite = eval_asr_suite(model, ecfg)
    dr        = pf.drop_rate if pf else 0.0

    tot_drop   = sum(active_drops)
    tot_drop_p = sum(active_drop_p)
    n_active   = max(len(active_drops), 1)
    cum_prec   = tot_drop_p / max(tot_drop, 1)
    avg_rec    = sum(d / max(total_poison, 1) for d in active_drop_p) / n_active

    warp_stats: dict = {}
    if pf is not None and getattr(pf, "_warp_tau_ema", None) is not None:
        warp_stats["tau_w_final"] = pf._warp_tau_ema
        warp_stats["warp_resample_ratio"] = mcfg.warp_resample_ratio

    unified_stats: dict = {}
    if pf is not None and getattr(pf, "_unified", False):
        if pf._tau_geo_ema is not None:
            unified_stats["tau_geo_final"] = pf._tau_geo_ema
        if pf._tau_sig_ema is not None:
            unified_stats["tau_sig_final"] = pf._tau_sig_ema
        if pf._tau_cons_ema is not None:
            unified_stats["tau_cons_final"] = pf._tau_cons_ema

    asr_str = "  ".join(f"{k}={v:.4f}" for k, v in asr_suite.items())
    print(f"\n  FINAL  CA={ca:.4f}  ASR: {asr_str}"
          f"\n         DR={dr:.4f}  Prec={cum_prec:.3f}  Rec={avg_rec:.3f}")
    if warp_stats:
        print(f"         tau_w_final={warp_stats['tau_w_final']:.4f}  "
              f"ratio={warp_stats['warp_resample_ratio']}")
    if unified_stats:
        parts = [f"{k}={v:.4f}" for k, v in unified_stats.items()]
        print(f"         {' '.join(parts)}")

    return {
        "exp_id":     ecfg.exp_id,
        "tag":        ecfg.tag,
        "clean_acc":  ca,
        "asr":        asr_suite,
        "drop_rate":  dr,
        "precision":  cum_prec,
        "recall":     avg_rec,
        "det_stats":  det_stats,
        "warp_stats": warp_stats,
        "unified_stats": unified_stats,
        "epoch_history": epoch_history,
        "config": {
            "dataset":    dcfg.name,
            "attack":     acfg.name,
            "method":     mcfg.name,
            "fl":         False,
            "epochs":     tcfg.epochs,
            "poison_rate": acfg.poison_rate,
            "agnostic_det": bool(tcfg.detector_base_datasets),
        },
    }
