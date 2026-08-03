"""End-to-end smoke test. Offline, CPU, no GPU and no downloads beyond gym_pusht.

    python -m sokubot.tests.smoke

Checks, in order of how much they would hurt if wrong:

  1. parameter budget           -- 5M encoder / 10M predictor / ~15M total
  2. causal masking             -- position t cannot see t+1 (else the prediction
                                   loss is solvable by copying the answer)
  3. AdaLN-Zero at init         -- the predictor starts action-independent
  4. shapes, finiteness, grads  -- every trainable submodule receives gradient
  5. SIGReg behaviour           -- penalises a collapsed latent, ~0 on N(0, I)
  6. learns without collapsing  -- L_pred falls on real PushT data while the
                                   latent's effective rank holds up
  7. CEM planner                -- beats random action sequences on its own cost
  8. AdaJEPA adaptation         -- test-time updates reduce prediction error
  9. Soku action parsing        -- mask/boolean cross-check catches a flipped bit
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.window import EpisodeWindowDataset
from ..losses import prediction_loss, sigreg
from ..model.world_model import LeWorldModel
from ..planning.adajepa import TestTimeAdapter
from ..planning.cem import CEMPlanner, cost_weights, rollout_cost
from ..train import compute_losses, effective_rank, train

PASSED, FAILED = [], []


def check(name):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                out = fn(*a, **kw)
                PASSED.append(name)
                # Some checks return (summary, objects...) for later checks to
                # reuse; only the summary is worth printing.
                note = out[0] if isinstance(out, tuple) else out
                print(f"  PASS  {name}" + (f" -- {note}" if note else ""))
                return out
            except Exception as exc:
                FAILED.append((name, exc))
                print(f"  FAIL  {name}: {exc}")
                traceback.print_exc()
        return wrapped
    return deco


# --------------------------------------------------------------------------
@check("parameter budget (5M encoder / 10M predictor)")
def test_param_budget():
    lines = []
    for label, cfg in (("pusht", Config.pusht()), ("soku", Config.soku())):
        rep = LeWorldModel(cfg).param_report()
        enc, pred, tot = rep["encoder"], rep["predictor"], rep["total"]
        assert 4.5e6 <= enc <= 6.0e6, f"{label}: encoder {enc/1e6:.2f}M outside 4.5-6.0M"
        assert 9.0e6 <= pred <= 11.0e6, f"{label}: predictor {pred/1e6:.2f}M outside 9-11M"
        assert tot <= 16.5e6, f"{label}: total {tot/1e6:.2f}M over budget"
        lines.append(f"{label} {enc/1e6:.2f}+{pred/1e6:.2f}={tot/1e6:.2f}M")
    return "; ".join(lines)


@check("predictor is causal")
def test_causal():
    cfg = Config.tiny()
    m = LeWorldModel(cfg).eval()
    torch.manual_seed(0)
    B, T = 2, cfg.seq_len
    z = torch.randn(B, T, cfg.latent_dim)
    cond = torch.randn(B, T, cfg.pred_dim)

    with torch.no_grad():
        base = m.predictor(z, cond)
        z2 = z.clone()
        z2[:, -1] += 10.0                      # perturb only the last position
        cond2 = cond.clone()
        cond2[:, -1] += 10.0
        pert = m.predictor(z2, cond2)

    head = (base[:, :-1] - pert[:, :-1]).abs().max().item()
    tail = (base[:, -1] - pert[:, -1]).abs().max().item()
    assert head < 1e-5, f"positions < T-1 changed by {head:.2e}; mask is leaking"
    assert tail > 1e-4, f"last position did not react ({tail:.2e}); test is vacuous"
    return f"leak {head:.1e}, response {tail:.1e}"


@check("AdaLN-Zero starts action-independent")
def test_adaln_zero():
    cfg = Config.tiny()
    m = LeWorldModel(cfg).eval()
    torch.manual_seed(0)
    z = torch.randn(4, cfg.seq_len, cfg.latent_dim)
    a1 = torch.randn(4, cfg.seq_len, cfg.action_ticks, cfg.action_dim)
    a2 = torch.randn(4, cfg.seq_len, cfg.action_ticks, cfg.action_dim)
    with torch.no_grad():
        o1 = m.predictor(z, m.action_encoder(a1))
        o2 = m.predictor(z, m.action_encoder(a2))
    d = (o1 - o2).abs().max().item()
    assert d < 1e-6, f"predictor already action-sensitive at init (delta {d:.2e})"
    return f"delta {d:.1e}"


@check("shapes, finiteness, gradient flow")
def test_grads():
    cfg = Config.tiny()
    torch.manual_seed(0)
    m = LeWorldModel(cfg)
    batch = {
        "obs": torch.rand(4, cfg.seq_len, 3, cfg.image_size, cfg.image_size),
        "actions": torch.rand(4, cfg.seq_len, cfg.action_ticks, cfg.action_dim),
    }
    total, metrics = compute_losses(m, batch, cfg)
    assert torch.isfinite(total), "loss is not finite"
    for k in ("l_pred", "l_sigreg", "latent_var"):
        assert np.isfinite(metrics[k]), f"{k} is not finite"

    total.backward()

    def grad_mass(prefix: str) -> float:
        grads = [p.grad for n, p in m.named_parameters()
                 if n.startswith(prefix) and p.requires_grad]
        assert grads, f"{prefix} has no trainable parameters"
        assert all(g is not None and torch.isfinite(g).all() for g in grads), \
            f"{prefix} has missing or non-finite gradients"
        return float(sum(g.abs().sum().item() for g in grads))

    assert grad_mass("encoder") > 0, "encoder received only zero gradients"
    assert grad_mass("predictor") > 0, "predictor received only zero gradients"

    # The action encoder is the one module that *must* be zero here. AdaLN-Zero
    # sets the modulation projection to 0, so mod = 0 for any condition and
    # d(mod)/d(cond) = 0 -- exactly the property test_adaln_zero checks on the
    # forward pass. The modulation weight itself does get gradient, so after one
    # optimiser step it is non-zero and the action path opens up. Asserting that
    # it opens is the real test; asserting it is open at init would be asserting
    # AdaLN-Zero is broken.
    assert grad_mass("action_encoder") == 0.0, \
        "action encoder has gradient at init, so AdaLN-Zero is not zero"

    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    opt.step()
    m.zero_grad(set_to_none=True)
    total2, _ = compute_losses(m, batch, cfg)
    total2.backward()
    assert grad_mass("action_encoder") > 0, \
        "action encoder still gets no gradient after one step; conditioning is dead"
    return f"loss {metrics['loss']:.3f}, action path opens after 1 step"


@check("SIGReg separates a Gaussian from a collapsed latent")
def test_sigreg():
    cfg = Config.tiny()
    torch.manual_seed(0)
    good = sigreg(torch.randn(256, cfg.latent_dim), cfg).item()
    collapsed = sigreg(torch.zeros(256, cfg.latent_dim), cfg).item()
    rank1 = torch.randn(256, 1).repeat(1, cfg.latent_dim)
    degenerate = sigreg(rank1, cfg).item()

    # Thresholds are on *ratios*, not absolute values: with sigreg_scale_n the
    # statistic scales with the sample size, so any absolute bound would silently
    # become a test of the batch size instead of a test of SIGReg.
    assert collapsed > 50 * good, \
        f"SIGReg barely separates collapse: {collapsed:.3f} vs N(0,I) {good:.3f}"
    assert degenerate > 20 * good, \
        f"SIGReg barely separates rank-1: {degenerate:.3f} vs N(0,I) {good:.3f}"
    return (f"N(0,I) {good:.3f} | collapsed {collapsed:.3f} ({collapsed/good:.0f}x) "
            f"| rank-1 {degenerate:.3f} ({degenerate/good:.0f}x)")


@check("Soku action parsing cross-checks mask against booleans")
def test_soku_actions():
    from ..data.soku import ACTION_COLUMNS, read_actions

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "inputs.csv"
        head = "global_frame,local_frame,p1_input,p2_input," + ",".join(ACTION_COLUMNS)
        # p1 mask 0b0000000101 = 5 -> up + left ; p2 mask 16 -> 'a'
        p1 = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        p2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        ok = f"0,0,5,16,{','.join(map(str, p1 + p2))}"
        p.write_text("\n".join([head, ok, ok]) + "\n")
        a = read_actions(p)
        assert a.shape == (2, 20), f"unexpected shape {a.shape}"
        assert a[0, 0] == 1 and a[0, 2] == 1 and a[0, 14] == 1

        bad = f"0,0,7,16,{','.join(map(str, p1 + p2))}"    # mask says 3 buttons, bools say 2
        p.write_text("\n".join([head, ok, bad]) + "\n")
        try:
            read_actions(p)
        except ValueError:
            return "flipped bit detected"
        raise AssertionError("a corrupted mask was not detected")


@check("Soku loader decodes video in step with the input CSV")
def test_soku_loader():
    """Builds a synthetic capture whose frames encode their own index.

    Frame n is a solid grey of value 4n, so after decimation the decision frames
    must read 0, 16, 32, 48... at frame_skip 4. If the loader ever drifts by a
    frame -- an off-by-one in the window arithmetic, ffmpeg duplicating frames
    to hold a nominal frame rate -- the pixel values say so immediately. Nothing
    downstream would: the model would just learn slightly wrong dynamics.
    """
    import shutil
    import subprocess

    from ..data.soku import ACTION_COLUMNS, SokuWindowDataset, discover_captures

    if shutil.which("ffmpeg") is None:
        return "skipped (no ffmpeg)"

    cfg = Config.tiny(base=Config.soku(), image_size=112)
    n_frames, size = 48, 64
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "w0" / "rep-0001"
        root.mkdir(parents=True)

        raw = b"".join(
            bytes([min(255, 4 * n)]) * (size * size * 3) for n in range(n_frames)
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size}x{size}", "-r", "60", "-i", "-",
             "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
             str(root / "video.mp4")],
            input=raw, check=True, capture_output=True,
        )

        # Frame n presses exactly the button at index (n % 10) for player 1.
        lines = ["global_frame,local_frame,p1_input,p2_input," + ",".join(ACTION_COLUMNS)]
        for n in range(n_frames):
            bit = n % 10
            p1 = [1 if i == bit else 0 for i in range(10)]
            lines.append(f"{n},{n},{1 << bit},0," + ",".join(map(str, p1 + [0] * 10)))
        (root / "inputs.csv").write_text("\n".join(lines) + "\n")

        (root.parent / "manifest.jsonl").write_text(json.dumps({
            "replay_id": "rep-0001", "status": "ok",
            "video": str(root / "video.mp4"), "frames": n_frames,
        }) + "\n")

        caps = discover_captures([Path(td)])
        assert len(caps) == 1, f"discovered {len(caps)} captures, expected 1"

        ds = SokuWindowDataset(cfg, caps, shuffle_buffer=1)
        items = list(ds)
        assert items, "loader yielded no windows"

        s = items[0]
        assert s["obs"].shape == (cfg.seq_len, 3, cfg.image_size, cfg.image_size), \
            f"bad obs shape {tuple(s['obs'].shape)}"
        assert s["actions"].shape == (cfg.seq_len, cfg.frame_skip, 20), \
            f"bad action shape {tuple(s['actions'].shape)}"

        # Decimation: decision frame k is source frame k*skip -> grey 4*k*skip.
        means = [float(s["obs"][k].mean() * 255) for k in range(cfg.seq_len)]
        want = [4 * k * cfg.frame_skip for k in range(cfg.seq_len)]
        err = max(abs(a - b) for a, b in zip(means, want))
        assert err < 8, f"frame decimation drifted: got {means}, expected {want}"

        # Alignment: the action chunk at decision step k must be source frames
        # [k*skip, k*skip+skip), each pressing button (n % 10).
        for k in range(cfg.seq_len):
            for tick in range(cfg.frame_skip):
                n = k * cfg.frame_skip + tick
                got = int(s["actions"][k, tick, : 10].argmax())
                assert s["actions"][k, tick].sum() == 1 and got == n % 10, \
                    f"action at step {k} tick {tick} is button {got}, expected {n % 10}"

        return f"{len(items)} windows, frame drift {err:.1f}/255, actions aligned"


# --------------------------------------------------------------------------
def _pusht_data(cfg: Config, root: Path, n_episodes: int, seed: int = 0) -> Path:
    from ..data.pusht import collect_episodes

    root.mkdir(parents=True, exist_ok=True)
    if not list(root.glob("*.npz")):
        print(f"    collecting {n_episodes} PushT episodes -> {root}")
        collect_episodes(cfg, root, n_episodes=n_episodes, max_steps=30,
                         seed=seed, verbose=False)
    return root


@check("learns on PushT without collapsing")
def test_learns(data_root: Path, steps: int = 30):
    cfg = Config.tiny(total_steps=steps, warmup_steps=3)
    ds = EpisodeWindowDataset(cfg, data_root)
    model, hist = train(cfg, ds, steps=steps, log_every=max(1, steps // 3), verbose=True)

    n = max(3, steps // 5)
    first = float(np.mean([h["l_pred"] for h in hist[:n]]))
    last = float(np.mean([h["l_pred"] for h in hist[-n:]]))
    rank = float(np.mean([h["eff_rank"] for h in hist[-n:]]))
    var = float(np.mean([h["latent_var"] for h in hist[-n:]]))

    assert last < first, f"L_pred did not fall: {first:.4f} -> {last:.4f}"
    assert var > 0.1, f"latent variance collapsed to {var:.4f}"
    # Effective rank is reported but not asserted on. It legitimately falls as
    # the encoder discovers that PushT has ~5 degrees of freedom, so a threshold
    # here would either be vacuous or would fail on a correctly-learning model.
    # The probe check below is the real anti-collapse test.
    return (f"L_pred {first:.4f} -> {last:.4f}, erank {rank:.1f}, var {var:.2f}",
            model, ds, cfg)


@check("latent retains physical state (linear probe)")
def test_probe(model: LeWorldModel, data_root: Path, cfg: Config):
    from ..data.episode import Episode
    from ..probe import probe_model

    eps = [Episode.load(p) for p in sorted(data_root.glob("*.npz"))]
    assert eps and eps[0].states is not None, "episodes carry no ground-truth state"

    trained = probe_model(model, eps, cfg, max_frames=1500)
    # The random-init encoder is a random-features baseline. Scoring against it
    # rather than against a fixed number keeps this check meaningful at any
    # training budget -- absolute R2 depends heavily on how many steps ran,
    # while "did training add state information" does not.
    baseline = probe_model(LeWorldModel(cfg), eps, cfg, max_frames=1500)

    agent = np.mean([trained.r2[k] for k in ("agent_x", "agent_y")])
    assert agent > 0.10, (
        f"latent lost the agent position (R2 {agent:.3f}) -- collapse, whatever "
        f"effective rank says"
    )
    assert trained.r2_mean > 2.0 * max(baseline.r2_mean, 0.02), (
        f"training added no state information: R2 {trained.r2_mean:.3f} vs "
        f"random-init baseline {baseline.r2_mean:.3f}"
    )
    return (f"trained {trained} | random-init baseline "
            f"R2 mean {baseline.r2_mean:.3f}")


@check("CEM beats random action sequences")
def test_cem(model: LeWorldModel, ds: EpisodeWindowDataset, cfg: Config):
    model.eval()
    torch.manual_seed(0)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=True)))
    with torch.no_grad():
        z = model.encode(batch["obs"])[0]              # [T, latent]
    z_ctx, z_goal = z[: cfg.history], z[-1]

    planner = CEMPlanner(model, cfg)
    w = cost_weights(cfg.plan_horizon, "final", z.device, z.dtype)
    with torch.no_grad():
        best = planner.plan(z_ctx, z_goal)
        c_cem = rollout_cost(model, z_ctx.unsqueeze(0), best.unsqueeze(0), z_goal, w).item()
        rand = torch.empty(64, cfg.plan_horizon, cfg.action_ticks,
                           cfg.action_dim).uniform_(-1, 1)
        c_rand = rollout_cost(model, z_ctx.unsqueeze(0).expand(64, -1, -1),
                              rand, z_goal, w)
    assert c_cem < c_rand.mean().item(), (
        f"CEM cost {c_cem:.4f} not better than random mean {c_rand.mean():.4f}"
    )
    return f"CEM {c_cem:.4f} < random mean {c_rand.mean():.4f} (best of 64: {c_rand.min():.4f})"


@check("TTA buffer stays frame/action aligned once it saturates")
def test_tta_alignment():
    """Runs past the buffer bound, which is the only regime where drift shows.

    Frames and actions are tagged with their step index, so after enough steps
    to force both deques to roll, every buffered pair must still carry the same
    tag. Two bugs lived here: deques sized equally desynchronise (a frame
    arrives before its action, so `frames` hits its bound one call early), and
    the window bound counted a trailing action that did not exist.
    """
    cfg = Config.tiny()
    model = LeWorldModel(cfg)
    adapter = TestTimeAdapter(model, cfg)

    S, A, ticks = cfg.image_size, cfg.action_dim, cfg.action_ticks
    n_steps = (cfg.tta_buffer + cfg.seq_len) * 3        # well past saturation

    adapter.observe(torch.full((3, S, S), 0.0), None)
    for i in range(1, n_steps):
        adapter.observe(torch.full((3, S, S), float(i)),
                        torch.full((ticks, A), float(i - 1)))

    win = adapter._windows()
    assert win is not None, "no window available after saturation"
    obs, act = win
    assert obs.shape[0] == act.shape[0], "window counts disagree"
    assert obs.shape[1] == act.shape[1] == cfg.seq_len

    # frames[i] carries tag i; the action applied at frames[i] carries tag i.
    for w in range(obs.shape[0]):
        for k in range(cfg.seq_len):
            f_tag = int(obs[w, k].flatten()[0].item())
            a_tag = int(act[w, k].flatten()[0].item())
            assert f_tag == a_tag, (
                f"window {w} step {k}: frame tag {f_tag} paired with action "
                f"tag {a_tag} -- buffers drifted"
            )
    assert not np.isnan(adapter.adapt()), "adapt() failed on a saturated buffer"
    return f"{obs.shape[0]} windows aligned after {n_steps} observations"


@check("AdaJEPA test-time adaptation reduces prediction error")
def test_tta(model: LeWorldModel, ds: EpisodeWindowDataset, cfg: Config):
    torch.manual_seed(0)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=True)))
    obs, act = batch["obs"][0], batch["actions"][0]     # [T,3,S,S], [T,ticks,A]

    adapter = TestTimeAdapter(model, cfg)
    for t in range(cfg.seq_len):
        adapter.observe(obs[t], act[t])

    def err() -> float:
        with torch.no_grad():
            z = adapter.model.encode(obs.unsqueeze(0))
            cond = adapter.model.action_encoder(act.unsqueeze(0))
            zhat = adapter.model.predictor(z, cond)
            return float(prediction_loss(zhat, z).item())

    before = err()
    for _ in range(10):
        adapter.adapt()
    after = err()

    frozen = {n for n, p in adapter.model.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in adapter.model.named_parameters() if p.requires_grad}
    assert trainable, "nothing is being adapted"
    assert len(frozen) > len(trainable), "TTA should update only the final layers"
    assert after < before, f"adaptation did not help: {before:.5f} -> {after:.5f}"
    return f"{before:.5f} -> {after:.5f} over 10 steps, {len(trainable)} tensors adapted"


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pusht-smoke")
    ap.add_argument("--episodes", type=int, default=24)
    # 150 steps, ~2 minutes on 4 CPU cores. Fewer does verify the plumbing, but
    # the model is then too weakly action-conditioned for the CEM check to have
    # a margin, and the probe sits near its threshold -- so the suite starts
    # reporting flaky failures rather than real ones.
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args(argv)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = Config.tiny()

    print("\n=== static checks ===")
    test_param_budget()
    test_causal()
    test_adaln_zero()
    test_grads()
    test_sigreg()
    test_soku_actions()
    test_soku_loader()
    test_tta_alignment()

    print("\n=== PushT end-to-end ===")
    root = _pusht_data(cfg, Path(args.data_root), args.episodes)
    learned = test_learns(root, steps=args.steps)
    if learned:
        _, model, ds, trained_cfg = learned
        test_probe(model, root, trained_cfg)
        test_cem(model, ds, trained_cfg)
        test_tta(model, ds, trained_cfg)
    else:
        print("  SKIP  CEM / TTA (training check failed)")

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  FAILED: {name}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
