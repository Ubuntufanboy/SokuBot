"""Hand-built scenarios for the reward landscape.

Every mistake in here is silent: the policy trains, the loss falls, and the
agent learns the wrong game. These check the cases that would be invisible from
a training curve -- side symmetry, the end-of-match heal, double-paid KOs.

    python -m pytest tests/test_reward.py -q
"""

from __future__ import annotations

import torch

from sokubot.rl.reward import RewardConfig, compute_rewards, SPELL, UP

TICKS = 4


def make(states, actions=None, side=0):
    """states: [T, 6] as a list of rows -> batched tensors for one trajectory."""
    s = torch.tensor([states], dtype=torch.float32)
    T = s.shape[1]
    a = torch.zeros(1, T - 1, TICKS, 20) if actions is None else actions
    return s, a, torch.tensor([side])


def full(hp1=1.0, hp2=1.0, sp1=1.0, sp2=1.0, cb1=0.0, cb2=0.0):
    return [hp1, hp2, sp1, sp2, cb1, cb2]


def test_damage_dealt_is_positive_and_scaled():
    # P2 loses a fifth of a bar; agent is P1.
    s, a, side = make([full(hp2=1.0), full(hp2=0.8)])
    r, alive, terms = compute_rewards(s, a, side)
    assert torch.allclose(terms["dealt"][0, 0], torch.tensor(0.2), atol=1e-5)
    assert r[0, 0] > 0


def test_oscillating_health_is_not_damage():
    """A flat-but-noisy health trace must not read as damage.

    The probe's per-step residual is around 0.14 of a bar, so an imagined health
    trajectory wobbles even when nothing is happening. Summing per-step clamped
    decreases rectifies that wobble into a large fake damage total which no
    action can influence -- the reward the policy could actually move ends up
    buried under it. Net mode differences the endpoints instead, so a trace that
    returns to where it started scores nothing.
    """
    wobble = [full(hp2=1.0), full(hp2=0.9), full(hp2=1.0), full(hp2=0.9),
              full(hp2=1.0)]
    net = compute_rewards(*make(wobble), RewardConfig(damage_mode="net"))[2]
    step = compute_rewards(*make(wobble), RewardConfig(damage_mode="step"))[2]
    assert net["dealt"].sum().item() == 0.0
    assert step["dealt"].sum().item() > 0.19        # two rectified 0.1 dips

    # A real drop must still be scored, and by the same amount either way.
    fall = [full(hp2=1.0), full(hp2=0.9), full(hp2=0.8), full(hp2=0.7)]
    net_d = compute_rewards(*make(fall), RewardConfig(damage_mode="net"))[2]
    step_d = compute_rewards(*make(fall), RewardConfig(damage_mode="step"))[2]
    assert abs(net_d["dealt"].sum().item() - 0.3) < 1e-5
    assert abs(step_d["dealt"].sum().item() - 0.3) < 1e-5


def test_side_symmetry():
    """The same events from P2's chair must pay P2 exactly what they paid P1."""
    cfg = RewardConfig()
    as_p1 = compute_rewards(*make([full(hp2=1.0), full(hp2=0.7)], side=0), cfg)[0]
    # Mirror every channel: the agent is now P2 and P1 is the one losing health.
    as_p2 = compute_rewards(*make([full(hp1=1.0), full(hp1=0.7)], side=1), cfg)[0]
    assert torch.allclose(as_p1, as_p2, atol=1e-6)


def test_combo_reads_the_opponents_bar():
    """Red on my own bar is damage being done to me and must not pay me."""
    mine_red = compute_rewards(*make([full(), full(cb1=0.5)], side=0))[2]
    their_red = compute_rewards(*make([full(), full(cb2=0.5)], side=0))[2]
    assert mine_red["combo"][0, 0].item() == 0.0
    assert their_red["combo"][0, 0].item() > 0.0


def test_end_of_match_heal_pays_nothing():
    """Health snapping back to full is a round reset, not the agent's doing."""
    s, a, side = make([full(hp1=0.1, hp2=0.1), full(hp1=1.0, hp2=1.0)])
    r, alive, terms = compute_rewards(s, a, side)
    assert terms["dealt"][0, 0].item() == 0.0
    assert terms["taken"][0, 0].item() == 0.0


def test_ko_pays_once_and_masks_the_rest():
    # Opponent drops and *stays* down, which is what separates a KO from noise.
    s, a, side = make([full(hp2=0.3), full(hp2=0.0), full(hp2=0.0),
                       full(hp2=0.0), full(hp2=0.5)])
    r, alive, terms = compute_rewards(s, a, side)
    assert terms["outcome"][0, 0].item() > 0        # win paid at the KO step
    assert alive[0, 0].item() == 1.0
    assert alive[0, 1].item() == 0.0                # everything after is masked
    assert r[0, 1].item() == 0.0 and r[0, 2].item() == 0.0
    assert (terms["outcome"][0, 1:] == 0).all()     # and never paid twice


def test_a_single_frame_dip_is_not_a_ko():
    """The probe's health residual is ~0.13 of a bar; one low read means nothing.

    This is the property that keeps a +-5 term from firing on probe noise many
    times per rollout and drowning out damage, which lives around 0.1.
    """
    s, a, side = make([full(hp2=0.6), full(hp2=0.0), full(hp2=0.6),
                       full(hp2=0.6), full(hp2=0.6)])
    r, alive, terms = compute_rewards(s, a, side)
    assert terms["outcome"].abs().max().item() == 0.0
    assert alive.min().item() == 1.0                # nothing was terminated


def test_a_side_that_starts_low_does_not_pay_out():
    """Starts are sampled from real gameplay, so some begin near death."""
    s, a, side = make([full(hp2=0.02), full(hp2=0.0), full(hp2=0.0),
                       full(hp2=0.0), full(hp2=0.0)])
    terms = compute_rewards(s, a, side)[2]
    assert terms["outcome"].abs().max().item() == 0.0


def test_simultaneous_ko_is_a_draw_not_a_loss():
    """An ambiguous double reading must not be scored as a loss.

    `ko_them & ~ko_me` paid `lose` for every simultaneous read, and with a noisy
    probe those are common -- it biased the whole outcome term negative.
    """
    s, a, side = make([full(hp1=0.5, hp2=0.5), full(hp1=0.0, hp2=0.0),
                       full(hp1=0.0, hp2=0.0), full(hp1=0.0, hp2=0.0)])
    terms = compute_rewards(s, a, side)[2]
    assert terms["outcome"].sum().item() == 0.0


def test_losing_is_penalised_not_rewarded():
    s, a, side = make([full(hp1=0.3), full(hp1=0.0), full(hp1=0.0),
                       full(hp1=0.0), full(hp1=0.0)])
    r, alive, terms = compute_rewards(s, a, side)
    assert terms["outcome"][0, 0].item() < 0


def test_spellcard_damage_is_worth_double():
    cfg = RewardConfig()
    # Press the card button at step 0 and spend spirit, then land a hit.
    a = torch.zeros(1, 2, TICKS, 20)
    a[0, 0, :, SPELL] = 1.0                          # agent is P1 -> block 0
    s = torch.tensor([[full(hp2=1.0, sp1=1.0),
                       full(hp2=1.0, sp1=0.6),       # 0.4 spirit spent = cast
                       full(hp2=0.8, sp1=0.6)]], dtype=torch.float32)
    with_card = compute_rewards(s, a, torch.tensor([0]), cfg)[2]["dealt"][0, 1]

    # Identical damage with no cast.
    s2 = torch.tensor([[full(hp2=1.0), full(hp2=1.0), full(hp2=0.8)]],
                      dtype=torch.float32)
    plain = compute_rewards(s2, torch.zeros(1, 2, TICKS, 20),
                            torch.tensor([0]), cfg)[2]["dealt"][0, 1]
    assert torch.allclose(with_card, plain * cfg.spell_multiplier, atol=1e-5)


def test_whiffed_card_costs_more_when_the_card_costs_more():
    cfg = RewardConfig()

    def whiff_for(spirit_spent):
        a = torch.zeros(1, 2, TICKS, 20)
        a[0, 0, :, SPELL] = 1.0
        s = torch.tensor([[full(sp1=1.0),
                           full(sp1=1.0 - spirit_spent),
                           full(sp1=1.0 - spirit_spent)]], dtype=torch.float32)
        return compute_rewards(s, a, torch.tensor([0]), cfg)[2]["whiff"][0, 0].item()

    cheap, dear = whiff_for(0.2), whiff_for(0.8)
    assert cheap < 0 and dear < cheap        # more expensive whiff hurts more


def test_pressing_the_button_without_paying_spirit_is_not_a_cast():
    """The button fires for skill cards and for presses that buy nothing."""
    a = torch.zeros(1, 2, TICKS, 20)
    a[0, 0, :, SPELL] = 1.0
    s = torch.tensor([[full(sp1=1.0), full(sp1=1.0), full(sp1=1.0)]],
                     dtype=torch.float32)          # no spirit spent
    terms = compute_rewards(s, a, torch.tensor([0]))[2]
    assert terms["whiff"][0, 0].item() == 0.0


def test_guard_crush_when_spirit_hits_zero():
    s, a, side = make([full(sp1=0.3), full(sp1=0.0)])
    terms = compute_rewards(s, a, side)[2]
    assert terms["crush"][0, 0].item() < 0
    # Already-empty spirit must not be charged again on the next step.
    s2, a2, side2 = make([full(sp1=0.0), full(sp1=0.0)])
    assert compute_rewards(s2, a2, side2)[2]["crush"][0, 0].item() == 0.0


def test_idle_is_penalised_and_holding_up_is_not():
    idle = compute_rewards(*make([full(), full()]))[2]
    a = torch.zeros(1, 1, TICKS, 20)
    a[0, :, :, UP] = 1.0
    moving = compute_rewards(torch.tensor([[full(), full()]], dtype=torch.float32),
                             a, torch.tensor([0]))[2]
    assert idle["idle"][0, 0].item() < 0
    assert moving["idle"][0, 0].item() == 0.0
    assert moving["flying"][0, 0].item() > 0


def test_button_block_follows_the_side():
    """The agent must read its own ten buttons, never the opponent's block."""
    from sokubot.rl.reward import _my_buttons
    a = torch.zeros(2, 1, TICKS, 20)
    a[0, 0, :, 3] = 1.0            # P1 right
    a[1, 0, :, 13] = 1.0           # P2 right
    seen = _my_buttons(a, torch.tensor([0, 1]))
    assert seen[0, 0, 0, 3].item() == 1.0
    assert seen[1, 0, 0, 3].item() == 1.0
    # Given only P1's input, an agent playing P2 must see nothing at all.
    only_p1 = torch.zeros(1, 1, TICKS, 20)
    only_p1[0, 0, :, 3] = 1.0
    assert _my_buttons(only_p1, torch.tensor([1])).max().item() == 0.0


def test_batch_of_mixed_sides_matches_one_at_a_time():
    """Batching must not leak state between trajectories on different sides."""
    cfg = RewardConfig()
    s = torch.tensor([[full(hp2=1.0), full(hp2=0.6)],
                      [full(hp1=1.0), full(hp1=0.6)]], dtype=torch.float32)
    a = torch.zeros(2, 1, TICKS, 20)
    side = torch.tensor([0, 1])
    both = compute_rewards(s, a, side, cfg)[0]
    one = compute_rewards(s[:1], a[:1], side[:1], cfg)[0]
    two = compute_rewards(s[1:], a[1:], side[1:], cfg)[0]
    assert torch.allclose(both[0], one[0], atol=1e-6)
    assert torch.allclose(both[1], two[0], atol=1e-6)
