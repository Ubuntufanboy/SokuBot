"""PPO self-play in imagination: the estimator, the league, that it LEARNS, and that it resumes.

The learning test uses a simulator with one planted rule -- pressing A takes health off the other
player -- so the correct policy change is known in advance and PPO has to find it. The resume test
checks the property a Slurm job depends on: stopping at step k and resuming gives bit-identical
weights to never having stopped.

    python -m pytest tests/test_ppo.py -q
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.make_smoke_assets import make_bank, write_bank, write_sim
from sokubot.data.state import CH, PROJ_FEATURES, STATE_CHANNELS
from sokubot.model.state_head import BINARY
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.ppo import (League, PPOConfig, advantages, assemble, bank_fingerprint,
                            evaluate_vs, init_policy_from_corpus, ppo_update)
from sokubot.rl.state_arena import StateArena, StateObs, StatePolicyOpponent, corpus_stats
from sokubot.rl.state_critic import StateCritic

BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
A_BTN = BUTTONS.index("a")
H, TICKS, SLOTS = 3, 5, 2


class PlantedSim(nn.Module):
    """Everything stays put except health: each player loses `k` x (the other's A-press rate)."""

    proj_feedback = "sigmoid"
    n_moves = 0

    def __init__(self, k: float = 0.05):
        super().__init__()
        self.k = k
        self.dummy = nn.Parameter(torch.zeros(1))      # arena freezes parameters; give it one

    def forward(self, s, p, a):
        ns = s.clone()
        binr = list(BINARY)
        ns[..., binr] = (2 * s[..., binr] - 1) * 20.0   # the arena sigmoids these back to 0/1
        a_p1 = a[..., 0 * 10 + A_BTN].mean(-1)          # [B, T]
        a_p2 = a[..., 1 * 10 + A_BTN].mean(-1)
        hp = CH["hp"]
        ns[..., 0, hp] = s[..., 0, hp] - self.k * a_p2
        ns[..., 1, hp] = s[..., 1, hp] - self.k * a_p1
        return ns, torch.full_like(p, -20.0)


@pytest.fixture(scope="module")
def world():
    S, P, A, E, V, _ = make_bank(replays=8, steps=120, slots=SLOTS, ticks=TICKS)
    obs = StateObs(*corpus_stats(S, P), SLOTS)
    return S, P, A, E, V, obs


def ctx_for(S, P, A, idx):
    off = np.arange(H) - (H - 1)
    w = idx[:, None] + off[None, :]
    return (torch.from_numpy(S[w]), torch.from_numpy(P[w]), torch.from_numpy(A[w[:, :-1]]).float())


def starts(n_total, n, rng):
    return rng.integers(H, n_total - 10, size=n)


# --- the estimator ------------------------------------------------------------------------------
class FeatureCritic:
    """v(obs) = the first feature of the newest frame, so WHICH observation is valued is visible."""

    def value(self, obs):
        return obs[..., -1, 0]


def test_gae_matches_a_hand_computation_including_the_true_bootstrap():
    g, lam, v, v_boot = 0.9, 0.5, 2.0, 7.0
    r = torch.tensor([[1.0, 0.0, 3.0]])
    alive = torch.ones(1, 3)
    term = torch.zeros(1, 3)
    obs, obs_last = torch.full((1, 3, H, 4), v), torch.full((1, H, 4), v_boot)
    ret, adv, _ = advantages(FeatureCritic(), obs, obs_last, r, alive, term, g, lam)
    # V_2 = r2 + g*((1-l)*v_boot + l*v_boot);  V_t = r_t + g*((1-l)*v + l*V_{t+1})
    V2 = 3 + g * v_boot
    V1 = 0 + g * ((1 - lam) * v + lam * V2)
    V0 = 1 + g * ((1 - lam) * v + lam * V1)
    assert torch.allclose(ret, torch.tensor([[V0, V1, V2]]))
    assert torch.allclose(adv, ret - v)


def test_a_ko_stops_the_bootstrap():
    r = torch.tensor([[0.0, 1.0]])
    ret, _, _ = advantages(FeatureCritic(), torch.full((1, 2, H, 4), 5.0), torch.full((1, H, 4), 5.0), r,
                           torch.ones(1, 2), torch.tensor([[0.0, 1.0]]), 0.9, 0.5)
    assert float(ret[0, 1]) == pytest.approx(1.0)          # nothing carried past the KO


# --- batch assembly -----------------------------------------------------------------------------
def fake_traj(B=4, T=3, dim=5, two=True):
    t = {"obs": torch.randn(B, T, H, dim), "obs_last": torch.randn(B, H, dim),
         "mine": torch.zeros(B, T, TICKS, 10), "reward": torch.randn(B, T),
         "alive": torch.ones(B, T), "terminal": torch.zeros(B, T),
         "side": torch.zeros(B, dtype=torch.long)}
    t["alive"][0, 2] = 0                                     # one dead step
    if two:
        t.update({"obs_opp": torch.randn(B, T, H, dim), "obs_last_opp": torch.randn(B, H, dim),
                  "mine_opp": torch.zeros(B, T, TICKS, 10), "reward_opp": torch.randn(B, T),
                  "alive_opp": torch.ones(B, T), "terminal_opp": torch.zeros(B, T),
                  "side_opp": torch.ones(B, dtype=torch.long)})
    return t


def test_two_sided_doubles_the_samples_and_dead_steps_are_dropped():
    critic = StateCritic(5, H)
    cfg = PPOConfig()
    one = assemble(fake_traj(), critic, cfg, both_chairs=False)
    both = assemble(fake_traj(), critic, cfg, both_chairs=True)
    assert len(one["adv"]) == 4 * 3 - 1
    assert len(both["adv"]) == (4 * 3 - 1) + 4 * 3
    assert set(both["side"].tolist()) == {0, 1}              # each chair keeps its own side
    assert abs(float(both["adv"].mean())) < 1e-5 and float(both["adv"].std()) == pytest.approx(1, abs=1e-4)


def test_both_chairs_without_the_opponents_view_is_an_error():
    with pytest.raises(ValueError, match="two-sided"):
        assemble(fake_traj(two=False), StateCritic(5, H), PPOConfig(), both_chairs=True)


# --- the league ---------------------------------------------------------------------------------
def test_league_adds_on_cadence_evicts_the_oldest_and_round_trips():
    cfg = PPOConfig(snapshot_every=10, max_snapshots=2)
    pol = SokuPolicy(8, H, TICKS)
    lg = League(cfg, pol)
    assert not lg.maybe_add(pol, 5) and lg.maybe_add(pol, 10)
    lg.maybe_add(pol, 20); lg.maybe_add(pol, 30)
    assert [e["step"] for e in lg.entries] == [20, 30]
    lg2 = League(cfg, pol)
    lg2.load_state_dict(lg.state_dict())
    assert [e["step"] for e in lg2.entries] == [20, 30]


def test_the_league_plays_the_weights_it_was_given():
    cfg = PPOConfig(snapshot_every=1)
    pol = SokuPolicy(8, H, TICKS)
    lg = League(cfg, pol)
    lg.maybe_add(pol, 1)
    with torch.no_grad():
        for p in pol.parameters():
            p.add_(1.0)                                      # the live policy moves on
    lg.maybe_add(pol, 2)
    w0 = next(iter(lg.policy(0).state_dict().values())).clone()   # the module is SHARED:
    w1 = next(iter(lg.policy(1).state_dict().values())).clone()   # clone before loading the next
    assert not torch.equal(w0, w1)
    assert torch.equal(lg.policy(0).state_dict()["head.weight"], lg.entries[0]["state"]["head.weight"])


def test_pfsp_favours_the_opponent_we_do_worst_against_and_tries_unplayed_ones():
    cfg = PPOConfig(snapshot_every=1)
    pol = SokuPolicy(8, H, TICKS)
    lg = League(cfg, pol)
    for s in (1, 2, 3, 4):
        lg.maybe_add(pol, s)
    for i, net in enumerate((0.3, -0.2, 0.1)):
        lg.record(i, net)
    w = lg.weights()
    assert w.argmax() in (1, 3)                              # hardest played, or unplayed
    assert w[1] > w[2] > w[0]                                # lower score -> sampled more
    assert w[3] == pytest.approx(w.max())                    # never played: top weight
    assert w.sum() == pytest.approx(1.0)


# --- the arena additions ------------------------------------------------------------------------
def test_the_arena_refuses_a_simulator_it_cannot_roll(world):
    S, P, A, E, V, obs = world
    sim = PlantedSim()
    sim.n_moves = 12
    with pytest.raises(ValueError, match="move identity"):
        StateArena(sim, obs, PPOConfig(horizon=2), H, TICKS)


def test_a_two_sided_self_play_rollout_carries_both_views_and_the_final_one(world):
    S, P, A, E, V, obs = world
    arena = StateArena(PlantedSim(), obs, PPOConfig(horizon=3), H, TICKS)
    pol = SokuPolicy(obs.dim, H, TICKS)
    ctx = ctx_for(S, P, A, starts(len(S), 6, np.random.default_rng(0)))
    side = torch.tensor([0, 1, 0, 1, 0, 1])
    tr = arena.rollout(*ctx, side, pol, StatePolicyOpponent(pol), two_sided=True)
    assert tr["obs_opp"].shape == tr["obs"].shape == (6, 3, H, obs.dim)
    assert tr["obs_last"].shape == tr["obs_last_opp"].shape == (6, H, obs.dim)
    # The final observation is the ego view of the final window, which ends in states[:, T].
    last_state = tr["states"][:, -1]
    me = last_state.gather(1, side.view(-1, 1, 1).expand(-1, 1, last_state.shape[-1]))[:, 0]
    s_mu, s_sd = obs.s_mu, obs.s_sd
    assert torch.allclose(tr["obs_last"][:, -1, :len(STATE_CHANNELS)],
                          ((me - s_mu) / s_sd).clamp(-obs.clip, obs.clip), atol=1e-5)


# --- it learns ----------------------------------------------------------------------------------
def a_rate(pol, obs_batch, side):
    with torch.no_grad():
        return float(pol(obs_batch, side, sample=True).actions[..., A_BTN].mean())


def test_ppo_learns_the_planted_rule(world):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    S, P, A, E, V, obs = world
    cfg = PPOConfig(horizon=3, starts_per_batch=96, epochs=4, minibatches=4, lr=3e-3,
                    kl_ref_coef=0.0, entropy_floor_frac=0.0, target_kl=1.0, snapshot_every=10**9)
    arena = StateArena(PlantedSim(k=0.05), obs, cfg, H, TICKS)
    pol = SokuPolicy(obs.dim, H, TICKS)
    init_policy_from_corpus(pol, A, BUTTONS[4:10])
    ref = SokuPolicy(obs.dim, H, TICKS)
    ref.load_state_dict(pol.state_dict())
    critic = StateCritic(obs.dim, H)
    opt = torch.optim.AdamW(pol.parameters(), lr=cfg.lr)
    copt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    probe = ctx_for(S, P, A, starts(len(S), 256, np.random.default_rng(1)))
    probe_obs = obs(probe[0], probe[1], torch.zeros(256, dtype=torch.long))
    probe_side = torch.zeros(256, dtype=torch.long)
    before = a_rate(pol, probe_obs, probe_side)
    for _ in range(25):
        ctx = ctx_for(S, P, A, starts(len(S), cfg.starts_per_batch, rng))
        side = torch.from_numpy(rng.integers(0, 2, cfg.starts_per_batch)).long()
        tr = arena.rollout(*ctx, side, pol, StatePolicyOpponent(pol), two_sided=True)
        batch = assemble(tr, critic, cfg, both_chairs=True)
        ppo_update(pol, critic, opt, copt, batch, cfg, ref, rng)
    after = a_rate(pol, probe_obs, probe_side)
    assert after > before + 0.15, f"A-press rate {before:.3f} -> {after:.3f}"
    # And against the frozen start it now wins the exchange, in both chairs.
    assert evaluate_vs(arena, pol, ref, probe)["net"] > 0


# --- the driver: files, resume, refusal, signals -------------------------------------------------
@pytest.fixture
def assets(tmp_path):
    return (write_sim(tmp_path / "sim.pt", slots=SLOTS, history=H, ticks=TICKS, width=32, depth=1),
            write_bank(tmp_path / "bank.npz", replays=6, steps=80, slots=SLOTS, ticks=TICKS))


def run(sim, bank, out, *extra):
    from scripts.train_state_ppo import main
    return main(["--sim", str(sim), "--bank", str(bank), "--out", str(out), "--device", "cpu",
                 "--starts", "16", "--horizon", "2", "--epochs", "2", "--minibatches", "2",
                 "--log-every", "1", "--eval-every", "2", "--eval-starts", "16",
                 "--snapshot-every", "2", "--ckpt-every", "100", "--seed", "3", *extra])


def weights(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_stopping_and_resuming_is_bit_identical_to_never_stopping(assets, tmp_path):
    sim, bank = assets
    assert run(sim, bank, tmp_path / "straight", "--steps", "6") == 0
    assert run(sim, bank, tmp_path / "split", "--steps", "6", "--stop-at-step", "3") == 99
    assert weights(tmp_path / "split" / "latest.pt")["step"] == 3
    # A job killed hard (no clean stop) can have LOGGED past its last checkpoint. The resume must
    # drop those lines, or the log carries two different step-4s.
    with open(tmp_path / "split" / "log.jsonl", "a") as f:
        f.write('{"step": 4, "from": "a run that died before checkpointing"}\n')
    # Re-run with the SAME arguments, as a requeued Slurm job would: it must not stop again.
    assert run(sim, bank, tmp_path / "split", "--steps", "6", "--stop-at-step", "3") == 0
    a, b = weights(tmp_path / "straight" / "latest.pt"), weights(tmp_path / "split" / "latest.pt")
    assert a["step"] == b["step"] == 6
    for k in a["policy"]:
        assert torch.equal(a["policy"][k], b["policy"][k]), k
    for k in a["critic"]:
        assert torch.equal(a["critic"][k], b["critic"][k]), k
    assert [e["step"] for e in a["league"]["entries"]] == [e["step"] for e in b["league"]["entries"]]
    # The log is continuous: one line per step, none duplicated by the resume.
    steps = [int(__import__("json").loads(l)["step"])
             for l in (tmp_path / "split" / "log.jsonl").read_text().splitlines()]
    assert steps == list(range(1, 7))


def test_resuming_against_a_different_simulator_is_refused(assets, tmp_path):
    sim, bank = assets
    assert run(sim, bank, tmp_path / "r", "--steps", "4", "--stop-at-step", "2") == 99
    other = write_sim(tmp_path / "other.pt", slots=SLOTS, history=H, ticks=TICKS, width=32,
                      depth=1, seed=7)
    with pytest.raises(SystemExit, match="refusing to resume"):
        run(other, bank, tmp_path / "r", "--steps", "4")


def test_a_bank_built_for_another_simulator_is_refused(assets, tmp_path):
    sim, _ = assets
    wrong = write_bank(tmp_path / "wrong.npz", replays=4, steps=60, slots=SLOTS + 1, ticks=TICKS)
    with pytest.raises(SystemExit, match="Rebuild the bank"):
        run(sim, wrong, tmp_path / "w", "--steps", "2")


def test_sigusr1_checkpoints_and_asks_to_be_requeued(assets, tmp_path, capsys):
    sim, bank = assets
    prev = signal.signal(signal.SIGUSR1, lambda *a: None)    # never let a stray signal kill pytest
    try:
        t = threading.Timer(3.0, os.kill, (os.getpid(), signal.SIGUSR1))
        t.start()
        rc = run(sim, bank, tmp_path / "s", "--steps", "100000", "--stop-after-hours", "0.05")
        t.cancel()
    finally:
        signal.signal(signal.SIGUSR1, prev)
    assert rc == 99
    assert "(SIGUSR1)" in capsys.readouterr().out
    assert weights(tmp_path / "s" / "latest.pt")["step"] >= 1


def test_the_bank_fingerprint_sees_a_change_anywhere_it_samples():
    S, P, A, E, V, _ = make_bank(replays=2, steps=50)
    base = bank_fingerprint(S, P, A, E, V)
    S2 = S.copy(); S2[0, 0, 0] += 1.0
    assert bank_fingerprint(S2, P, A, E, V) != base
    assert bank_fingerprint(S[:-1], P[:-1], A[:-1], E[:-1], V[:-1]) != base    # shape


def test_a_resume_loads_the_checkpoint_to_cpu_whatever_the_training_device(assets, tmp_path,
                                                                            monkeypatch):
    """A GPU run resuming with map_location="cuda" moved the RNG state onto the GPU and died in
    torch.set_rng_state. This pins the call, but on a CPU-only machine the buggy form
    (map_location=device) is indistinguishable, since the device IS "cpu": the GPU smoke job
    (ops/amarel/smoke.slurm, which stops, requeues and resumes on a GPU) is the real guard."""
    sim, bank = assets
    assert run(sim, bank, tmp_path / "c", "--steps", "4", "--stop-at-step", "2") == 99
    seen = []
    real = torch.load

    def spy(path, *a, **kw):
        if str(path).endswith("latest.pt"):
            seen.append(kw.get("map_location"))
        return real(path, *a, **kw)
    monkeypatch.setattr(torch, "load", spy)
    assert run(sim, bank, tmp_path / "c", "--steps", "4") == 0
    assert seen == ["cpu"]
