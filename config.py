from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

_ROOT = "./data/"

@dataclass
class DatasetConfig:
    name: str
    root: str         = _ROOT
    num_classes: int  = 10
    image_size: int   = 32
    channels: int     = 3
    to_rgb: bool      = False
    target_label: int = 0
    trigger_sz_min: int = 2
    trigger_sz_max: int = 6
    aug_padding: int  = 4
    use_affine: bool  = False

    @classmethod
    def cifar10(cls, root: str = _ROOT) -> "DatasetConfig":
        return cls(name="cifar10", root=root, num_classes=10)

    @classmethod
    def cifar100(cls, root: str = _ROOT) -> "DatasetConfig":
        return cls(name="cifar100", root=root, num_classes=100)

    @classmethod
    def mnist(cls, root: str = _ROOT) -> "DatasetConfig":
        return cls(name="mnist", root=root, num_classes=10,
                   image_size=28, channels=1, to_rgb=True,
                   trigger_sz_min=1, trigger_sz_max=4,
                   aug_padding=2, use_affine=False)

    @classmethod
    def tinyimagenet(cls, root: str = _ROOT) -> "DatasetConfig":
        return cls(name="tinyimagenet", root=root, num_classes=200,
                   image_size=64, channels=3,
                   trigger_sz_min=3, trigger_sz_max=10, aug_padding=8)

ALL_DATASETS = {
    "cifar10":      DatasetConfig.cifar10,
    "cifar100":     DatasetConfig.cifar100,
    "mnist":        DatasetConfig.mnist,
    "tinyimagenet": DatasetConfig.tinyimagenet,
}

CORE_FAMILIES = ("badnets", "blended", "dba", "pattern", "noise", "wanet")
SYNTH_FAMILIES = ("freq_band", "edge_patch", "grid_warp")
AGNOSTIC_FAMILIES = (
    "fourier_shuffle", "narrowband", "cutout_blend", "grid_warp",
    "phase_jitter",
    "color_distort",
    "smooth_noise",
)
ALL_FAMILIES = CORE_FAMILIES + SYNTH_FAMILIES

@dataclass
class AttackConfig:
    name: str
    poison_rate: float = 0.10
    detector_families: Optional[List[str]] = None
    lofo: bool = False

    def get_detector_families(self) -> List[str]:
        if self.detector_families is not None:
            return list(self.detector_families)
        if self.lofo:
            return [f for f in CORE_FAMILIES if f != self.name]
        return list(CORE_FAMILIES)

ALL_ATTACKS = {n: AttackConfig(name=n) for n in ALL_FAMILIES}

@dataclass
class MethodConfig:
    name: str
    train_detector: bool = True
    use_mad: bool        = True
    use_warmup: bool     = True
    tau: float           = 2.0
    rho_max: float       = 0.2
    mad_buffer_size: int = 2048
    min_history_size: int = 64
    eps: float           = 1e-6
    warmup_epochs: int   = 0
    always_topk: bool       = False
    always_topk_epochs: int = 0
    detector_use_fft: bool  = False
    use_lrt: bool    = False
    lrt_alpha: float = 0.05
    auto_tau: bool   = False
    alpha: float     = 0.02
    tau_min: float   = 1.5
    tau_max: float   = 3.5
    hardneg_ratio: float = 0.0
    lambda_consistency: float = 0.0
    use_synth_families: bool = False
    use_multiview: bool      = False
    use_bilateral: bool      = False
    use_soft_weight: bool    = False
    soft_weight_k: float     = 1.0
    soft_weight_eta: float   = 1.0
    w_min: float             = 0.2
    freq_rand_prob: float    = 0.0
    use_dynamics_view: bool  = False
    a_cap: float             = 6.0
    fusion_mode: str         = "max"
    min_votes: int           = 1
    use_agnostic_det: bool   = False
    drop_mode: str = "guard"
    use_dual_threshold: bool = False
    rho_target:  float       = 0.05
    tau_lr:      float       = 0.01
    rolling_alpha: float     = 0.05
    rolling_win_top: float   = 0.01
    use_consistency_view: bool = False
    use_dynamics_delta:  bool  = False
    dynamics_ema_alpha:  float = 0.9
    use_freq_bonus: bool   = False
    use_warp_bonus: bool   = False
    freq_bonus_lambda: float = 0.5
    warp_bonus_lambda: float = 0.5
    freq_bonus_tau: float    = 1.5
    warp_bonus_tau: float    = 1.5
    warp_resample_ratio: float = 0.75
    warp_bonus_adapt: bool       = False
    warp_bonus_q: float          = 0.95
    warp_bonus_tau_beta: float   = 0.9
    warp_bonus_min_history: int  = 64
    unified_channels: bool        = False
    unified_geometry: bool        = False
    unified_sig: bool             = False
    unified_sig_cons: bool        = False
    unified_sig_target_cycles: float = 6.0
    unified_q: float              = 0.975
    unified_lambda_w: float       = 0.25
    unified_lambda_s: float       = 0.25
    unified_lambda_c: float       = 0.20
    unified_tau_beta: float       = 0.9
    unified_min_history: int      = 64
    unified_warp_ratio: float     = 0.75

    @classmethod
    def baseline(cls) -> "MethodConfig":
        return cls(name="baseline", train_detector=False,
                   use_mad=False, use_warmup=False)

    @classmethod
    def brifi(cls, lrt_alpha: float = 0.05, **kw) -> "MethodConfig":
        return cls(name="brifi", train_detector=True,
                   use_mad=True, use_warmup=False, warmup_epochs=0,
                   use_lrt=True, lrt_alpha=lrt_alpha, **kw)

    @classmethod
    def brifi_freq(cls, lrt_alpha: float = 0.05,
                   freq_lambda: float = 0.5,
                   freq_tau: float = 1.5, **kw) -> "MethodConfig":
        return cls(name="brifi_freq", train_detector=True,
                   use_mad=True, use_warmup=False, warmup_epochs=0,
                   use_lrt=True, lrt_alpha=lrt_alpha,
                   use_freq_bonus=True,
                   freq_bonus_lambda=freq_lambda,
                   freq_bonus_tau=freq_tau, **kw)

    @classmethod
    def brifi_warp(cls, lrt_alpha: float = 0.05,
                   warp_lambda: float = 0.5,
                   warp_tau: float = 1.5,
                   warp_ratio: float = 0.75, **kw) -> "MethodConfig":
        return cls(name="brifi_warp", train_detector=True,
                   use_mad=True, use_warmup=False, warmup_epochs=0,
                   use_lrt=True, lrt_alpha=lrt_alpha,
                   use_warp_bonus=True,
                   warp_bonus_lambda=warp_lambda,
                   warp_bonus_tau=warp_tau,
                   warp_resample_ratio=warp_ratio, **kw)

    @classmethod
    def brifi_u(
        cls,
        lrt_alpha: float = 0.05,
        q: float = 0.975,
        lambda_w: float = 0.25,
        lambda_s: float = 0.25,
        lambda_c: float = 0.20,
        tau_beta: float = 0.9,
        warp_ratio: float = 0.75,
        sig_cycles: float = 6.0,
        **kw,
    ) -> "MethodConfig":
        return cls(
            name="brifi_u", train_detector=True,
            use_mad=True, use_warmup=False, warmup_epochs=0,
            use_lrt=True, lrt_alpha=lrt_alpha,
            unified_channels=True,
            unified_geometry=True, unified_sig=True, unified_sig_cons=True,
            unified_q=q, unified_lambda_w=lambda_w,
            unified_lambda_s=lambda_s, unified_lambda_c=lambda_c,
            unified_tau_beta=tau_beta, unified_warp_ratio=warp_ratio,
            unified_sig_target_cycles=sig_cycles, **kw)

ALL_METHODS = {
    "baseline":    MethodConfig.baseline(),
    "brifi":       MethodConfig.brifi(),
    "brifi_freq":  MethodConfig.brifi_freq(),
    "brifi_warp":  MethodConfig.brifi_warp(),
    "brifi_u":     MethodConfig.brifi_u(),
}

@dataclass
class TrainConfig:
    epochs: int          = 20
    batch_size: int      = 128
    lr: float            = 0.01
    momentum: float      = 0.9
    weight_decay: float  = 5e-4
    seed: int            = 0
    num_workers: int     = 2
    detector_base_n: int = 10_000
    detector_epochs: int = 10
    detector_lr: float   = 1e-3
    detector_base_datasets: Optional[List[str]] = None
    detector_n_per_base: int = 5_000
    detector_ckpt_path: Optional[str] = None
    detector_aux_root: Optional[str] = None
    detector_image_size: int = 32
    fl: bool                  = False
    fl_n_clients: int         = 100
    fl_rounds: int            = 20
    fl_local_epochs: int      = 1
    fl_alpha: float           = 0.5
    fl_warmup_rounds: int     = 0
    fl_malicious_frac: float  = 0.2
    fl_participation_rate: float = 0.3
    sl: bool                  = False
    sfl: bool                 = False
    sl_cut: str               = "middle"
    sfl_n_clients: int        = 10
    sfl_rounds: int           = 20
    sfl_local_epochs: int     = 1
    sfl_alpha: float          = 0.5
    sfl_malicious_frac: float = 0.0
    device: str               = "cpu"

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    attack:  AttackConfig
    method:  MethodConfig
    train:   TrainConfig = field(default_factory=TrainConfig)
    tag:     str = ""

    @property
    def exp_id(self) -> str:
        fl_tag   = "_fl" if self.train.fl else ""
        tag_part = f"_{self.tag}" if self.tag else ""
        return (f"{self.dataset.name}/{self.attack.name}/"
                f"{self.method.name}{tag_part}{fl_tag}")

    def __str__(self) -> str:
        return self.exp_id
