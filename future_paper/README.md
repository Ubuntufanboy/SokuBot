# Making world-model RL competitive with real-environment RL

## The thesis

The agent must learn to play Touhou 12.3 *Hisoutensoku* almost entirely inside
a learned world model, and at inference must consume only pixels and its own
inputs. Training against the real engine would be easier and is explicitly NOT
the goal: the research question is what has to be true of a world model for
model-based learning to match real-environment RL. A later phase conditions the
policy on natural-language instruction, which is a second reason the model
matters -- an instruction has to be interpretable as a goal *inside* the model,
not just as a reward on the real environment.

So every result here is recorded as evidence about world-model viability, not
merely as progress on a game.

## Where the project stands, empirically

The agent has played a human three times under real match conditions.

| match | state source | result |
|---|---|---|
| 2026-08-16 | game memory (cheating diagnostic) | agent won |
| 2026-08-17 a | pixels, encoder R2 0.359 | human won 2-1 |
| 2026-08-17 b | pixels, encoder R2 0.557, LAN inference | human won 2-1 |

The human reports the last two were at comparable skill and that they played
near their best. A reproducible human exploit exists (zone from across the
screen); its cause is documented in `findings/04`.

## The findings, in order of how much they bound the result

1. `findings/01-missing-move-identity.md` -- the state the model conditions on
   omits *which move is executing*, so its dynamics are non-Markov. Suspected
   root cause of the blocking failure.
2. `findings/02-fidelity-decay.md` -- per-channel error against real actions,
   and the anomaly that one channel is broken at step 2 rather than degrading.
3. `findings/03-credit-beyond-the-horizon.md` -- which behaviours are
   structurally unlearnable at a 0.67 s horizon, with the measurements.
4. `findings/04-perception-transfer.md` -- corpus-to-live generalisation gap in
   the pixel encoder, and two failure modes found only by logging ground truth
   during a real match.
5. `findings/05-reward-shaping.md` -- what shaping bought, what it did not, and
   why.
6. `findings/07-action-information.md` -- I(A;F_{t+k}|F_t) measured on the
   corpus, the peak at k=4, the 2x context cost, and the normalisation that
   turns "the model cannot represent blocking" into "the objective prices
   blocking at half a percent". The central diagnostic result.

7. `findings/09-counterfactual-sufficiency.md` -- the payoff. On a proxy
   validated against Soku's own pathologies, observational training recovers
   29% of the true causal action effect and interventional training recovers
   all of it. Also the fourth instance of a units error deciding a conclusion.

8. `findings/08-observational-ceiling.md` -- five redesign arms, two seeds,
   two instruments, all null on action sensitivity; why a variational bound
   certified 98.9% and bought nothing; and the argument that the remaining
   move is interventional data.

9. `findings/06-metric-pathologies.md` -- measurements that actively misled us,
   kept because a paper about world models should say how its own numbers lie.

## House rule for these notes

Every claim carries the measurement that produced it and the caveat that
weakens it. Several entries record things that were WRONG on first analysis and
how the error was found; those are as useful as the positive results and they
are not to be quietly edited out.
