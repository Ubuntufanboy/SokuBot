# Open questions, ranked by how much they bound the result

1. **Does conditioning on move identity fix the blocking channel?**
   Falsifiable prediction in `findings/01`. *Running now* -- two arms, same
   corpus, same seed, same 10k steps, differing only in whether the simulator
   is given an embedding of the nominal action id and asked to predict the next
   one. Next-move accuracy reached 0.91 within 4500 steps, so the id is very
   predictable and the input is nearly free; what remains to be seen is whether
   any of that reaches `guarding`.

   The instrument was rebuilt for it. The obvious per-channel sigma error
   cannot answer the question for a rare flag: `guarding` has a 3-4% base rate,
   the simulator emits ~0.45, and 2.5 of its 2.5 sigma of "error" is that one
   constant offset. **AUC is the number that separates knowledge from
   calibration** -- it reads only the ordering, so 0.5 means the output carries
   no information about when the flag is set, whatever the offset. Measured on
   the current simulators, guarding AUC is 1.00 at one step and decays to 0.55
   (fix5) and 0.46 (rawproj) by 32, so the existing models *do* rank blocking
   correctly in the short term and lose it entirely by half a second. That is a
   different and more precise claim than "the model cannot see blocking", and
   it is the baseline the move arm has to beat.

2. **Is teacher-forced next-step training the reason rollouts blur?**
   The predictor has never consumed its own output during training, while RL
   uses it autoregressively for 8-24 steps. Unrolled or scheduled-sampling
   training is the standard remedy and has not been tried here.

3. **Does the reward have to be dense because the model is bad, or because the
   horizon is short?** The critic separates these in principle; the first run
   confounded it with an entropy-coefficient mismatch (see `findings/03`).

4. **Can a world model support a mechanic it cannot predict one step ahead?**
   Blocking is the test case. If the answer is no in general, then per-channel
   one-step fidelity becomes a *precondition* for any mechanic you intend the
   policy to learn -- a stronger and more useful claim than a horizon budget.

5. **Why does depth per replay beat breadth for perception?** Answered
   empirically in `findings/04` -- 149 replays at 150 frames beat 2003 at 7 --
   but only at one sample budget, and the mechanism (consecutive frames sample
   the local geometry of position, distinct replays sample the pose manifold)
   is an explanation rather than a measurement. The clean test is a fixed
   sample budget swept across the frames-per-replay axis.

6. **Behaviour cloning is untouched.** The policy is initialised from a
   *marginal* button prior, never from conditional human behaviour, despite
   2003 replays being on disk. This is the largest unexplored lever and is also
   the initialisation any league-style self-play would need.

7. **For the language phase:** an instruction has to be expressible as a goal
   inside the model. That argues for a latent world model with a structured or
   compositional state, rather than 33 hand-picked channels -- which is a
   different architecture from the current one and worth deciding before more
   is built on top of the channel representation.
