"""Warm-starting a higher-resolution encoder, and the two auxiliary objectives.

    python -m pytest tests/test_highres.py -q
"""

from __future__ import annotations

import pytest
import torch

from sokubot.config import Config
from sokubot.model.encoder import resize_pos_embed
from sokubot.model.world_model import LeWorldModel, N_HUD
from sokubot.train import compute_losses


def test_only_pos_embed_is_tied_to_the_grid():
    """The claim that makes warm-starting cheap. If a second tensor ever becomes
    resolution-dependent, this fails rather than silently reinitialising it."""
    lo, hi = Config.soku(), Config.soku448()
    a, b = LeWorldModel(lo).state_dict(), LeWorldModel(hi).state_dict()
    differ = [k for k in a if a[k].shape != b[k].shape]
    # The property, not a hard count: a count assertion went stale the moment the
    # HUD head added two tensors, which is the kind of brittleness that makes a
    # test noise rather than a guard.
    assert differ == ["encoder.pos_embed"], differ
    assert len(a) - len(differ) > 200, "almost everything should transfer"


def test_resize_pos_embed_preserves_cls_and_shape():
    lo, hi = Config.soku(), Config.soku448()
    pe = LeWorldModel(lo).state_dict()["encoder.pos_embed"]
    out = resize_pos_embed(pe, 16, 32)
    assert out.shape == (1, 32 * 32 + 1, lo.enc_dim)
    # The CLS position is not part of the grid and must pass through untouched.
    assert torch.equal(out[:, :1], pe[:, :1])


def test_resize_is_identity_at_the_same_grid():
    pe = torch.randn(1, 257, 192)
    assert torch.equal(resize_pos_embed(pe, 16, 16), pe)


def test_resize_rejects_a_mismatched_grid():
    with pytest.raises(ValueError, match=r"expected \[1, 1 \+ 16\^2"):
        resize_pos_embed(torch.randn(1, 100, 192), 16, 32)


def test_a_224_state_dict_loads_into_a_448_model():
    lo, hi = Config.soku(), Config.soku448()
    sd = LeWorldModel(lo).state_dict()
    sd["encoder.pos_embed"] = resize_pos_embed(sd["encoder.pos_embed"], 16, 32)
    r = LeWorldModel(hi).load_state_dict(sd, strict=False)
    assert r.unexpected_keys == []
    # Both were built with hud_coef > 0, so nothing should be missing either.
    assert r.missing_keys == []


def test_auxiliary_losses_are_off_by_default_when_weights_are_zero():
    """Setting both to zero must reproduce the original objective exactly, so the
    320k-step run on record stays reproducible."""
    cfg = Config.tiny(base=Config.soku(), cf_coef=0.0, hud_coef=0.0,
                      idm_coef=0.0)
    m = LeWorldModel(cfg)
    assert m.hud_head is None
    assert m.idm_head is None
    B, T = 2, cfg.seq_len
    batch = {"obs": torch.randint(0, 255, (B, T, 3, cfg.image_size, cfg.image_size),
                                 dtype=torch.uint8),
             "actions": torch.rand(B, T, cfg.action_ticks, cfg.action_dim).round()}
    _, mets = compute_losses(m, batch, cfg)
    assert mets["l_cf"] == 0.0 and mets["l_hud"] == 0.0 and mets["l_idm"] == 0.0
    assert mets["loss"] == pytest.approx(
        mets["l_pred"] + cfg.lambda_sigreg * mets["l_sigreg"], rel=1e-5)


def test_counterfactual_term_starts_at_chance():
    """ln(1 + n_negatives) is chance. Starting anywhere else would mean the term
    is measuring something other than action discrimination."""
    import math
    cfg = Config.tiny(base=Config.soku(), hud_coef=0.0)
    m = LeWorldModel(cfg)
    B, T = 4, cfg.seq_len
    batch = {"obs": torch.randint(0, 255, (B, T, 3, cfg.image_size, cfg.image_size),
                                 dtype=torch.uint8),
             "actions": torch.rand(B, T, cfg.action_ticks, cfg.action_dim).round()}
    _, mets = compute_losses(m, batch, cfg)
    assert mets["l_cf"] == pytest.approx(math.log(1 + cfg.cf_negatives), rel=0.05)


def test_hud_supervision_reaches_the_encoder():
    """The head exists to put a gradient on the *encoder*; a head that trained
    alone would fit the readout and leave the latent exactly as uninformative."""
    cfg = Config.tiny(base=Config.soku(), cf_coef=0.0)
    m = LeWorldModel(cfg)
    B, T = 2, cfg.seq_len
    batch = {"obs": torch.randint(0, 255, (B, T, 3, cfg.image_size, cfg.image_size),
                                 dtype=torch.uint8),
             "actions": torch.rand(B, T, cfg.action_ticks, cfg.action_dim).round(),
             "hud": torch.rand(B, T, N_HUD)}
    m.zero_grad(set_to_none=True)
    compute_losses(m, batch, cfg)[0].backward()
    assert sum(float(p.grad.abs().sum()) for p in m.encoder.parameters()
               if p.grad is not None) > 0
    assert float(m.hud_head.weight.grad.abs().sum()) > 0


def test_hud_head_output_is_a_gauge_fraction():
    cfg = Config.tiny(base=Config.soku())
    m = LeWorldModel(cfg)
    h = m.predict_hud(torch.randn(8, cfg.latent_dim) * 10)
    assert h.shape == (8, N_HUD)
    assert ((h >= 0) & (h <= 1)).all()
