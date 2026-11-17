from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from config import DatasetConfig

class DetectorCNN(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1)).squeeze(-1)

class DetectorCNNFFT(nn.Module):

    def __init__(self, in_channels: int = 3, fft_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32,          64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,          128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fft_conv = nn.Sequential(
            nn.Conv2d(1, fft_dim, 3, padding=1), nn.BatchNorm2d(fft_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128 + fft_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_cnn = self.conv(x).flatten(1)

        g   = x.mean(dim=1, keepdim=True)
        F_c = torch.fft.rfft2(g, norm="ortho")
        A   = torch.log1p(torch.abs(F_c))
        feat_fft = self.fft_conv(A).flatten(1)

        feat = torch.cat([feat_cnn, feat_fft], dim=1)
        return self.fc(feat).squeeze(-1)

class FrequencyStatsScorer(nn.Module):

    K: int = 4

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1)
        F_c  = torch.fft.rfft2(gray, norm="ortho")
        mag2 = F_c.real.pow(2) + F_c.imag.pow(2)

        B, H, W2 = mag2.shape
        W  = (W2 - 1) * 2

        freq_row = torch.arange(H, dtype=torch.float32, device=x.device)
        freq_row = torch.minimum(freq_row, H - freq_row) / (H / 2.0)
        freq_col = torch.arange(W2, dtype=torch.float32, device=x.device) / (W / 2.0)
        freq_r   = (freq_row.unsqueeze(1).pow(2) + freq_col.unsqueeze(0).pow(2)
                    ).sqrt().clamp(max=1.0)

        dc_mask  = (freq_r < 1e-6)
        mag2_ndc = mag2 * (~dc_mask).float()

        flat_ndc = mag2_ndc.flatten(start_dim=1)
        eps      = 1e-10

        peak   = flat_ndc.max(dim=1).values
        med    = flat_ndc.median(dim=1).values
        s0     = torch.log1p(peak / (med + eps))

        log_p   = torch.log(flat_ndc + eps)
        lgeom   = log_p.mean(dim=1)
        larith  = torch.log(flat_ndc.mean(dim=1) + eps)
        s1      = (larith - lgeom).clamp(min=0.0)

        lo_mask = (freq_r < 0.33).float()
        mi_mask = ((freq_r >= 0.33) & (freq_r < 0.67)).float()
        hi_mask = (freq_r >= 0.67).float()
        lo_e    = (mag2_ndc * lo_mask).sum(dim=(-2, -1))
        mi_e    = (mag2_ndc * mi_mask).sum(dim=(-2, -1))
        hi_e    = (mag2_ndc * hi_mask).sum(dim=(-2, -1))
        total_e = lo_e + mi_e + hi_e + eps
        max_share = torch.stack([lo_e, mi_e, hi_e], dim=1).max(dim=1).values
        s2      = torch.log1p(max_share / total_e)

        mid_band_mask = ((freq_r >= 0.2) & (freq_r < 0.7)).float()
        mag2_mid  = mag2_ndc * mid_band_mask
        flat_mid  = mag2_mid.flatten(start_dim=1)
        n_mid     = mid_band_mask.sum().long().item()
        if n_mid > 0:
            pk_mid  = flat_mid.max(dim=1).values
            med_mid = flat_mid.median(dim=1).values
            s3      = torch.log1p(pk_mid / (med_mid + eps))
        else:
            s3 = torch.zeros(B, device=x.device)

        return torch.stack([s0, s1, s2, s3], dim=1)

FrequencyAnomalyScorer = FrequencyStatsScorer

class SpectralStabilityScorer(nn.Module):

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad    = self.kernel_size // 2
        x_blur = F.avg_pool2d(x, kernel_size=self.kernel_size,
                              stride=1, padding=pad)
        s_orig = self._peak_ratio(x)
        s_blur = self._peak_ratio(x_blur)
        score  = (s_orig - s_blur).abs() / (s_orig + s_blur + 1e-10)
        return score

    @staticmethod
    def _peak_ratio(y: torch.Tensor) -> torch.Tensor:
        gray = y.mean(dim=1)
        F_c  = torch.fft.rfft2(gray, norm="ortho")
        mag2 = F_c.real.pow(2) + F_c.imag.pow(2)

        B, H, W2 = mag2.shape
        W = (W2 - 1) * 2
        freq_row = torch.arange(H,  dtype=torch.float32, device=y.device)
        freq_row = torch.minimum(freq_row, H - freq_row) / (H / 2.0)
        freq_col = torch.arange(W2, dtype=torch.float32, device=y.device) / (W / 2.0)
        freq_r   = (freq_row.unsqueeze(1).pow(2)
                    + freq_col.unsqueeze(0).pow(2)).sqrt()
        dc_mask  = freq_r < 1e-6
        flat     = (mag2 * (~dc_mask).float()).flatten(start_dim=1)
        peak     = flat.max(dim=1).values
        med      = flat.median(dim=1).values
        return torch.log1p(peak / (med + 1e-10))

class RandomScorer(nn.Module):

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.size(0), device=x.device)

    def train(self, mode: bool = True) -> "RandomScorer":
        return self

    def eval(self) -> "RandomScorer":
        return self

def _cifar_resnet18(nc: int, in_channels: int = 3) -> nn.Module:
    m         = tvm.resnet18(weights=None)
    m.conv1   = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, nc)
    return m

def build_classifier(dcfg: DatasetConfig) -> nn.Module:
    in_ch = 3
    return _cifar_resnet18(nc=dcfg.num_classes, in_channels=in_ch)

class HighpassEnergyScorer(nn.Module):

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad     = self.kernel_size // 2
        blurred = F.avg_pool2d(x, kernel_size=self.kernel_size,
                               stride=1, padding=pad)
        residual = x - blurred
        return residual.pow(2).mean(dim=(1, 2, 3))

def build_frequency_scorer() -> "FrequencyStatsScorer":
    return FrequencyStatsScorer()

def build_highpass_scorer() -> "HighpassEnergyScorer":
    return HighpassEnergyScorer(kernel_size=5)

def build_consistency_scorer() -> "SpectralStabilityScorer":
    return SpectralStabilityScorer(kernel_size=5)

def build_detector_model(dcfg: DatasetConfig, trained: bool,
                         use_fft: bool = False) -> nn.Module:
    if not trained:
        return RandomScorer()
    in_ch = 3
    if use_fft:
        return DetectorCNNFFT(in_channels=in_ch)
    return DetectorCNN(in_channels=in_ch)
