"""Single source of truth for architecture, objective, and planning hyper-parameters.

Everything in SokuBot reads its dimensions from one :class:`Config`, so the smoke
test, the training loop, and the planner cannot drift apart.

The defaults reproduce LeWorldModel (arXiv:2603.19312) as specified, with the
parameter budget split 5M encoder / 10M predictor.

WHERE THE NUMBERS COME FROM
---------------------------
LeWM Sec. 3.1 and App. D specify:

  encoder    ViT-Tiny, patch 14, 12 layers, 3 heads, width 192  (~5M)
             [CLS] -> 1-layer MLP + BatchNorm projection
  predictor  6 layers, 16 heads, 10% dropout                    (~10M)
             causal mask over the observation history,
             AdaLN(-Zero) action conditioning at every layer,
             followed by the same MLP+BatchNorm projector
  objective  L_pred + lambda * SIGReg(Z),  lambda = 0.1, M = 1024
  data       frame-skip 5, sub-trajectories of 4 frames, batch 128, 224x224

Two hyper-parameters the paper leaves unstated, and how they are set here:

* ``pred_mlp_ratio = 3.0``. A 6-layer, width-384 transformer costs 12*d^2 per
  layer at the usual ratio 4, i.e. 10.6M for the backbone alone -- before any
  AdaLN modulation, which adds 6*d^2 per layer (another 5.3M) and would put the
  predictor at 16M. Since the paper states depth, width and head count but not
  the MLP ratio, the ratio is what gets adjusted: at 3.0 the backbone is 8.9M
  and the predictor lands at 9.9M *including* modulation. See
  ``pred_shared_adaln`` for the other half of that fix.

* ``latent_dim = 192``. Unstated; matched to the ViT-Tiny [CLS] width so the
  projector is square.

Presets: :meth:`Config.pusht` (paper repro / smoke target),
:meth:`Config.soku` (the real target -- 20 binary buttons at frame-skip 4),
:meth:`Config.tiny` (CPU-sized, for tests).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional


ActionSpace = Literal["continuous", "binary"]


@dataclass
class Config:
    # ---------------- observation ----------------
    image_size: int = 224
    patch_size: int = 14
    in_chans: int = 3

    # ---------------- encoder (ViT-Tiny, ~5.5M) ----------------
    enc_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3
    enc_mlp_ratio: float = 4.0
    latent_dim: int = 192

    # ---------------- predictor (~9.9M) ----------------
    pred_dim: int = 384
    pred_depth: int = 6
    pred_heads: int = 16
    pred_mlp_ratio: float = 3.0
    pred_dropout: float = 0.1
    # One modulation MLP shared by every block, plus a per-block learned offset
    # (PixArt-alpha's "adaLN-single"). A private modulation MLP per block would
    # cost 6*pred_dim^2 each -- 5.3M of the 10M budget spent on conditioning.
    # The shared form costs 0.9M once and 2.3k per block, and is what makes the
    # paper's stated 6 layers / 16 heads fit the stated ~10M.
    pred_shared_adaln: bool = True

    # ---------------- action ----------------
    # `action_dim` is per environment tick; `action_ticks` is how many ticks one
    # model step covers, i.e. the frame-skip. One action *chunk* is therefore
    # action_ticks x action_dim numbers.
    action_dim: int = 2
    action_ticks: int = 5
    action_space: ActionSpace = "continuous"
    act_hidden: int = 256
    # Per-tick bounds, used by the planner to clamp/initialise candidates.
    action_low: float = 0.0
    action_high: float = 1.0

    # ---------------- temporal ----------------
    # Sub-trajectory length fed to the model. The predictor is causal, so a
    # length-`seq_len` window yields `seq_len - 1` next-step predictions.
    seq_len: int = 4
    history: int = 3          # context length at planning time
    frame_skip: int = 5       # kept in sync with action_ticks; see __post_init__

    # ---------------- objective ----------------
    sigreg_dirs: int = 1024       # M random projections
    sigreg_points: int = 65       # quadrature points for the Epps-Pulley integral
    sigreg_range: float = 5.0     # integrate over [-range, range]
    sigreg_lambda_w: float = 1.0  # w(t) = exp(-t^2 / (2*lambda_w))
    # The classical Epps-Pulley statistic carries a factor of the sample size:
    # T_n = n * integral |phi_n - phi_0|^2 w dt. Eq. (EP) in the LeWM appendix
    # writes the population form without it, but the n-scaled estimator is the
    # one that must be used, and this is not cosmetic -- it decides whether the
    # objective has collapse as its global minimum.
    #
    # Measured on this codebase: SIGReg is ~0.004 on a true N(0, I) sample and
    # ~0.409 on a collapsed one. Unscaled, lambda * SIGReg tops out at 0.041,
    # while a *healthy* L_pred is ~1.0 -- so collapsing costs 0.04 and saves
    # 1.0, and gradient descent correctly takes that trade. The first PushT run
    # collapsed by step 10 for exactly this reason. Scaled by n, the collapse
    # penalty becomes n * 0.041, which dominates.
    #
    # The escape is also one-way: at an exactly-collapsed latent every
    # projection is 0, the ECF is identically 1, and d/dz cos(t*z)|_0 = 0, so
    # SIGReg's gradient vanishes. It can only prevent collapse, never undo it --
    # which is why the scaling has to be right up front.
    sigreg_scale_n: bool = True
    lambda_sigreg: float = 0.1

    # ---------------- optimisation ----------------
    lr: float = 5e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 500
    total_steps: int = 100_000
    grad_clip: float = 1.0
    batch_size: int = 128
    num_workers: int = 0
    # Ship frames to the GPU as uint8 and do the /255 there. Pure throughput --
    # the arithmetic is identical, see data/window.frames_to_chw.
    loader_uint8: bool = True
    # Batches each worker keeps queued. The default of 2 leaves the GPU waiting
    # whenever a worker hits a slow capture (a new ffmpeg process, a seek); more
    # queued batches absorb that jitter at the cost of host RAM.
    prefetch_factor: int = 6

    # ---------------- throughput (all change numerics, none change the method) ----
    # How often to compute training diagnostics. Every one of them ends in
    # .item(), which synchronises the GPU and drains the pipeline, and
    # effective_rank additionally runs a 192x192 eigendecomposition on the CPU.
    # Paying that per step costs more than the diagnostics are worth; 0 means
    # every step, which is only useful when debugging a specific run.
    metrics_every: int = 50
    # TF32 matmuls/convs on Ampere+. Under bf16 autocast most matmuls are
    # already reduced precision, so this mainly affects the fp32 residue.
    tf32: bool = True
    # torch.compile the model. Costs ~1-2 min of graph capture on the first
    # steps and recompiles when a shape changes (e.g. the eval batch size).
    compile: bool = False
    compile_mode: str = "default"      # "default" | "max-autotune"
    # Fused AdamW: one kernel for the whole parameter update instead of a
    # foreach loop. Pure win on CUDA.
    fused_optimizer: bool = True
    # Compute SIGReg for all timesteps in one batched call rather than looping.
    # Same statistic, T times fewer kernel launches, T times the peak memory.
    sigreg_batched: bool = True
    amp_dtype: str = "bf16"       # "bf16" | "fp16" | "fp32"

    # ---------------- planning (LeWM App. D / AdaJEPA Sec. 4.1) ----------------
    plan_horizon: int = 5
    cem_samples: int = 300
    cem_elites: int = 30
    cem_iters: int = 30
    cem_init_var: float = 1.0

    # ---------------- AdaJEPA test-time adaptation ----------------
    tta_buffer: int = 5           # recent-N transitions
    tta_steps: int = 1            # U gradient updates per replan
    tta_lr_pred: float = 5e-4     # eta_pred
    tta_lr_enc: float = 1e-5      # eta_enc
    tta_stop_grad: bool = True    # sg(.) on the adaptation target

    # ---------------- misc ----------------
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        # The frame-skip *is* the number of environment ticks bundled into one
        # model action. Keeping two names for it invites them to disagree, so
        # the constructor ties them together and `frame_skip` stays the one that
        # datasets read.
        if self.action_ticks != self.frame_skip:
            object.__setattr__(self, "action_ticks", self.frame_skip)
        if self.history > self.seq_len:
            raise ValueError(
                f"history ({self.history}) cannot exceed seq_len ({self.seq_len})"
            )

    # ---------------- derived ----------------
    @property
    def num_patches(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} not divisible by patch {self.patch_size}"
            )
        return (self.image_size // self.patch_size) ** 2

    @property
    def action_chunk_dim(self) -> int:
        """Flattened width of one action chunk: ticks x per-tick dim."""
        return self.action_ticks * self.action_dim

    # ---------------- presets ----------------
    @classmethod
    def pusht(cls, **overrides) -> "Config":
        """LeWM's PushT setting: 2-D continuous actions in [0, 512], frame-skip 5."""
        return replace(
            cls(
                action_dim=2,
                action_space="continuous",
                action_low=0.0,
                action_high=512.0,
                frame_skip=5,
                history=3,
                seq_len=4,
            ),
            **overrides,
        )

    @classmethod
    def soku(cls, **overrides) -> "Config":
        """Hisoutensoku: both players' 10 buttons each, 60 fps decimated to 15 Hz.

        Action layout per tick (see data/soku.py):
            [p1 up down left right a b c d change spell |
             p2 up down left right a b c d change spell]
        """
        return replace(
            cls(
                action_dim=20,
                action_space="binary",
                action_low=0.0,
                action_high=1.0,
                frame_skip=4,
                history=3,
                seq_len=4,
            ),
            **overrides,
        )

    @classmethod
    def tiny(cls, base: Optional["Config"] = None, **overrides) -> "Config":
        """CPU-sized variant for the smoke test. Same code paths, ~40x fewer FLOPs."""
        base = base or cls.pusht()
        shrunk = dict(
            image_size=112,      # 8x8 = 64 patches at patch 14
            enc_dim=96,
            enc_depth=4,
            enc_heads=3,
            latent_dim=96,
            pred_dim=128,
            pred_depth=2,
            pred_heads=4,
            act_hidden=64,
            sigreg_dirs=128,
            sigreg_points=33,
            # Dense diagnostics. This preset exists for the smoke test and for
            # debugging, where the training history is the output; the per-step
            # sync it costs is irrelevant at this size and skipping it leaves
            # a 150-step run with four recorded points.
            metrics_every=0,
            # SIGReg is a *distributional* test applied per timestep, so its
            # sample size is the batch, not batch x time. 8 embeddings in 96
            # dimensions carry almost no distributional signal; 32 is the
            # smallest that behaves like the paper's 128 while staying quick on
            # a laptop CPU.
            batch_size=32,
            warmup_steps=5,
            total_steps=100,
            cem_samples=64,
            cem_elites=8,
            cem_iters=4,
        )
        # Caller overrides win; passing both here would make replace() raise
        # "got multiple values for keyword argument".
        shrunk.update(overrides)
        return replace(base, **shrunk)
