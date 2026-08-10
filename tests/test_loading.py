"""A checkpoint's architecture must come from its weights, not today's defaults."""

from __future__ import annotations

import torch

from sokubot.config import Config
from sokubot.model.loading import load_world_model, reconcile_config
from sokubot.model.world_model import LeWorldModel


def _tiny(**kw) -> Config:
    cfg = Config.tiny()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_old_checkpoint_does_not_gain_a_phantom_hud_head(tmp_path):
    """The exact failure that broke every pre-HUD checkpoint.

    `hud_coef` was added to `Config` with a non-zero default. Configs pickled
    before it existed have no such entry, so reading the attribute falls through
    to the class default, `LeWorldModel` builds a `hud_head`, and the load fails
    on missing keys -- or, with `strict=False`, succeeds with a random head.
    """
    cfg = _tiny(hud_coef=0.0)
    model = LeWorldModel(cfg)
    assert model.hud_head is None
    path = tmp_path / "old.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)

    # Simulate the class default moving after the checkpoint was written.
    saved = dict(torch.load(path, map_location="cpu", weights_only=False))
    del saved["cfg"].__dict__["hud_coef"]
    torch.save(saved, path)
    assert Config.hud_coef > 0, "this test is meaningless if the default is 0"

    wm, cfg2, _ = load_world_model(path, "cpu", verbose=False)
    assert wm.hud_head is None
    assert cfg2.hud_coef == 0.0


def test_checkpoint_with_a_head_keeps_it(tmp_path):
    cfg = _tiny(hud_coef=0.25)
    model = LeWorldModel(cfg)
    assert model.hud_head is not None
    path = tmp_path / "new.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)

    wm, cfg2, _ = load_world_model(path, "cpu", verbose=False)
    assert wm.hud_head is not None
    assert cfg2.hud_coef > 0
    # Strict load means the head carries the *saved* weights, not fresh ones.
    assert torch.allclose(wm.hud_head.weight, model.hud_head.weight)


def test_image_size_is_read_off_the_positional_grid(tmp_path):
    """The same class of bug, one field over: `image_size` moved 224 -> 448."""
    cfg = _tiny()
    model = LeWorldModel(cfg)
    state = model.state_dict()
    wrong = _tiny()
    wrong.image_size = cfg.image_size * 2          # as if the default had moved
    notes = reconcile_config(wrong, state)
    assert wrong.image_size == cfg.image_size
    assert any("positional grid" in n for n in notes)


def test_a_matching_config_is_left_alone(tmp_path):
    cfg = _tiny(hud_coef=0.25)
    state = LeWorldModel(cfg).state_dict()
    assert reconcile_config(_tiny(hud_coef=0.25), state) == []


def test_freeze_is_explicit(tmp_path):
    cfg = _tiny(hud_coef=0.0)
    path = tmp_path / "m.pt"
    torch.save({"model": LeWorldModel(cfg).state_dict(), "cfg": cfg}, path)
    frozen, _, _ = load_world_model(path, "cpu", verbose=False)
    assert not any(p.requires_grad for p in frozen.parameters())
    live, _, _ = load_world_model(path, "cpu", freeze=False, verbose=False)
    assert all(p.requires_grad for p in live.parameters())


def test_idm_head_is_reconciled_too(tmp_path):
    """The same bug, reintroduced. `idm_coef` was added with a default of 1.0,
    which gave every checkpoint written before it an `idm_head` with no weights
    -- breaking artifacts that had loaded fine minutes earlier. The reconciler
    handles heads from a table now, so this test is really asking whether the
    table was updated."""
    cfg = _tiny(hud_coef=0.0, idm_coef=0.0)
    model = LeWorldModel(cfg)
    assert model.idm_head is None
    path = tmp_path / "pre_idm.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)

    saved = dict(torch.load(path, map_location="cpu", weights_only=False))
    del saved["cfg"].__dict__["idm_coef"]
    torch.save(saved, path)
    assert Config.idm_coef > 0, "meaningless unless the default is non-zero"

    wm, cfg2, _ = load_world_model(path, "cpu", verbose=False)
    assert wm.idm_head is None
    assert cfg2.idm_coef == 0.0


def test_a_checkpoint_with_an_idm_head_keeps_it(tmp_path):
    cfg = _tiny(idm_coef=1.0)
    model = LeWorldModel(cfg)
    assert model.idm_head is not None
    path = tmp_path / "with_idm.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    wm, cfg2, _ = load_world_model(path, "cpu", verbose=False)
    assert wm.idm_head is not None
    assert torch.allclose(wm.idm_head.net[0].weight, model.idm_head.net[0].weight)


# Coefficients that weight a loss term without gating a module. These leave no
# trace in the weights, so there is nothing for reconcile_config to restore.
# Listing them explicitly is the point: a new *_coef is assumed to gate a
# module until someone says otherwise, so forgetting fails the test.
NON_GATING_COEFS = {
    "cf_coef", "sigreg_coef", "pred_coef", "var_coef", "cov_coef",
    "entropy_coef", "kl_coef", "value_coef", "aux_coef",
}


def test_every_config_gated_module_is_in_the_reconcile_table():
    """A guard against the next one, and this time not one that must be
    remembered.

    The previous version listed ("hud_coef", "idm_coef") by hand, so adding a
    third gating field and forgetting it passed cleanly -- which is exactly the
    failure it was written to stop. `hud_coef` broke every earlier checkpoint
    once, `idm_coef` did it again a day later, and both times the symptom was a
    confusing "Missing key(s) in state_dict" on artifacts that loaded fine the
    day before.

    So the fields are discovered rather than listed, and anything ending in
    _coef must either be reconciled or be declared non-gating above.
    """
    import inspect
    from dataclasses import fields
    from sokubot.config import Config
    from sokubot.model import loading

    src = inspect.getsource(loading.reconcile_config)
    coefs = {f.name for f in fields(Config) if f.name.endswith("_coef")}
    assert coefs, "no *_coef fields found; has Config been renamed?"

    missing = [c for c in sorted(coefs)
               if c not in src and c not in NON_GATING_COEFS]
    assert not missing, (
        f"{missing} end in _coef but are neither reconciled in "
        f"reconcile_config nor declared in NON_GATING_COEFS. If one gates a "
        f"module, add it to the table; if it only weights a loss, add it to "
        f"NON_GATING_COEFS.")
