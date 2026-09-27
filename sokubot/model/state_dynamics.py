"""The simulator: (state, both players' buttons) -> next state. No pixels.

WHY THE SIMULATOR NO LONGER SHARES A LATENT WITH PERCEPTION
-----------------------------------------------------------
Seven configurations were measured, and every one of them destroys the spatial
information the policy needs. A randomly-initialised encoder reads the
play-area mirror at probe_play 0.90 on real frames; training takes it to 0.51,
which is chance:

    full objective        0.83 -> 0.77 over 300 steps
    SIGReg off            0.83 -> 0.76      identical
    HUD + counterfactual off  0.83 -> 0.78  identical
    prediction only       0.83 -> 0.77      identical
    stop-grad on target   0.82 -> 0.76      identical
    state supervision x30 0.82 -> 0.75      identical, marginally worse
    3000 steps            0.77 -> 0.51      terminal, not asymptotic

Not a term that can be deleted, not a gradient path that can be cut, not a
coefficient that can be outweighed. A 192-dimensional latent shared between
"what is happening" and "where everything is" does not hold both, and the
compression keeps the first.

So the simulator stops using one. It consumes the game's own state -- the 33
channels and 24 projectile slots per player that the extractor reads out of
memory and pipeline/verify_extended.py checks against the running game -- and
predicts the next one. Position cannot be eroded here because position is an
input column, not something a representation has to choose to keep.

WHAT THIS BUYS RL
-----------------
  * The rollout is in state space, so a reward reads `hp` exactly instead of
    through a probe with a 0.116 residual.
  * Start states are real corpus states, not encodings of them.
  * Training needs no video at all: 20.9M frames of state are 1.8 GB, and the
    whole simulator fits on the LAN box.

The encoder becomes a separate, ordinary supervised problem -- pixels to state
-- and its quality bounds only PLAY, never the simulator. That separation is
the point: the two jobs stopped competing for the same 192 numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..data.state import PROJ_FEATURES, STATE_CHANNELS
from .layers import Block
from .state_head import BINARY, CONTINUOUS, N_PLAYERS

N_STATE = len(STATE_CHANNELS)
N_PROJF = len(PROJ_FEATURES)


# Index 0 of every vocabulary is RESERVED for "a move this run never saw".
# Without it a rare id in the evaluation split would index past the table, and
# with a silent clamp instead it would masquerade as whichever move happens to
# sit at that index.
MOVE_OOV = 0


def build_move_vocab(action, min_count: int = 32):
    """Nominal action ids -> the sorted vocabulary to embed.

    Rare ids are dropped into MOVE_OOV rather than given their own row: an id
    seen thirty times in twenty million frames cannot train an embedding, and
    each one spent is a row of the classifier's output that can only ever be
    wrong.
    """
    import numpy as np
    ids, counts = np.unique(np.asarray(action).reshape(-1), return_counts=True)
    keep = ids[counts >= min_count]
    return np.concatenate([np.array([-1], keep.dtype), keep])


def index_moves(action, vocab):
    """Ids -> embedding indices, anything unknown to MOVE_OOV."""
    import numpy as np
    a = np.asarray(action)
    pos = np.searchsorted(vocab, a)
    pos = np.clip(pos, 0, len(vocab) - 1)
    hit = vocab[pos] == a
    return np.where(hit, pos, MOVE_OOV).astype(np.int64)


class StateDynamics(nn.Module):
    """A causal transformer over state history. [B,T,...] -> next state.

    PREDICTS DELTAS FOR THE CONTINUOUS CHANNELS
    -------------------------------------------
    At 60 Hz with frame-skip 5, most channels barely move between steps, so
    predicting the absolute value makes "copy the input" a near-optimal
    solution that scores well and has learned nothing. A delta head makes the
    identity prediction exactly zero, so any skill the model reports is skill
    over standing still -- which is the baseline every rollout must beat to be
    worth simulating in.

    Binary channels are predicted as absolute logits, because a flag's delta is
    not a meaningful quantity.
    """

    def __init__(self, cfg: Config, slots: int = 8, width: int = 384,
                 depth: int = 6, heads: int = 8, history: int = 8,
                 ticks: int | None = None, proj_feedback: str = "sigmoid",
                 n_moves: int = 0, move_dim: int = 0,
                 idm: bool = False, state_history: int = 0,
                 act_skip: bool = False):
        super().__init__()
        self.slots = slots
        self.history = history
        # STATE HISTORY SHORTER THAN ACTION HISTORY.
        #
        # Every sequence position carries a state and an action, so `history`
        # is both. Measured, that is the problem: conditioning on 12 frames of
        # state destroys 2.03x of the action information at k=8, because "has
        # been holding away for 44 frames" is legible in the state drift and
        # the model reads intent there instead of from the stick. Masking the
        # state on older positions keeps the action history long and the state
        # history short, so intent has only one place to live.
        self.state_history = int(state_history or history)
        if not 1 <= self.state_history <= history:
            raise ValueError(f"state_history {self.state_history} outside "
                             f"1..{history}")
        # How the projectile head's output is turned back into an input during
        # an autoregressive rollout. A PROPERTY OF THE MODEL, not of whoever is
        # rolling it, because the model is trained through this path and a
        # consumer that picks the other one is feeding it inputs it has never
        # seen. Measured on `fix5/sim.pt`: reading its rollout with "raw" costs
        # -0.048 of kinematic skill at eight steps.
        #
        #   "sigmoid"  squash the WHOLE tensor. This is a defect -- `dx`, `dy`,
        #              `vx`, `vy` are regression outputs that can be negative,
        #              and squashing them into (0,1) destroys the sign, which
        #              is the entire content of "is that bullet coming at me".
        #              Kept as the default because every checkpoint to date was
        #              trained through it.
        #   "raw"      squash only `present` and `hb`, which really are logits.
        #              Correct, and only usable on a model retrained with it.
        if proj_feedback not in ("sigmoid", "raw"):
            raise ValueError(f"proj_feedback {proj_feedback!r}; want sigmoid or raw")
        self.proj_feedback = proj_feedback
        # Ticks per decision step, i.e. the frame skip. NOT cfg.action_ticks:
        # the whole point of a frame_skip=1 arm is that one step is one frame,
        # and taking the width from the config would silently keep a 5-tick
        # action block while the data supplies one.
        self.ticks = int(ticks or cfg.action_ticks)
        self.n_act = self.ticks * 2 * 10             # both players' buttons

        # MOVE IDENTITY.
        #
        # The 33 state channels carry `action_frame` -- how far into a move a
        # character is -- and never say WHICH move. Two situations with the
        # same frame count, positions and buttons can end in a hit, a block or
        # a whiff depending on the attack's geometry, so a model given only the
        # summary must predict the conditional mean over every move consistent
        # with it. Measured, that is exactly what `guarding` looks like: 0.51
        # sigma of error by step TWO and flat thereafter, while every other
        # channel degrades smoothly with horizon. Flat-from-the-start is not
        # compounding rollout error; it is a variable the model cannot see.
        #
        # An embedding, not a number, because ids are nominal: 801 is not one
        # more than 800. `n_moves` is the size of the corpus vocabulary and 0
        # means the channel is off, which is what every checkpoint before this
        # was trained as.
        self.n_moves = int(n_moves)
        self.move_dim = int(move_dim) if n_moves else 0
        self.move_embed = (nn.Embedding(self.n_moves, self.move_dim)
                           if self.move_dim else None)

        self.in_dim = (N_PLAYERS * N_STATE
                       + N_PLAYERS * slots * N_PROJF
                       + self.n_act
                       + N_PLAYERS * self.move_dim)
        self.embed = nn.Linear(self.in_dim, width)
        self.pos = nn.Parameter(torch.zeros(1, history, width))
        self.blocks = nn.ModuleList(
            Block(width, heads, 4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head_state = nn.Linear(width, N_PLAYERS * N_STATE)
        self.head_proj = nn.Linear(width, N_PLAYERS * slots * N_PROJF)
        # The move must be PREDICTED as well as consumed, or an autoregressive
        # rollout has nothing to feed itself at step two. It is also the output
        # a policy wants most directly -- "the opponent has started move X" is
        # the whole content of a reaction.
        self.head_move = (nn.Linear(width, N_PLAYERS * self.n_moves)
                          if self.move_dim else None)
        # INVERSE DYNAMICS, on the model's OWN PREDICTION.
        #
        # Reads the action back out of (current state, predicted next state).
        # It can only succeed if the prediction actually depends on the action,
        # so its loss is a variational lower bound on I(A_t ; F_{t+1} | F_t)
        # *as expressed by this model* -- the exact quantity measured on the
        # corpus at 3.03 nats and largely discarded by next-state regression.
        # Every other proposal works around the sparse action Jacobian; this
        # one pays for it directly.
        # ACTION SKIP: a direct path from the buttons to the output.
        #
        # The action reaches the prediction only by being concatenated into a
        # 278-dim input, embedded, and carried through six transformer blocks
        # and their LayerNorms. Flipping the defender's stick perturbs 10 of
        # those dims at one of twelve positions, and measured against the game
        # the model recovers 47% of the true effect of that flip -- it responds,
        # but weakly.
        #
        # Reweighting the loss at the frames where it matters was tried first
        # and is a null (+1.5%), which points at the path rather than the
        # objective. This adds a short one. Zero-initialised, so the model still
        # starts as an exact identity predictor and this can only be learned
        # into, never inherited as noise.
        self.act_skip = nn.Linear(self.n_act, width) if act_skip else None
        if self.act_skip is not None:
            nn.init.zeros_(self.act_skip.weight)
            nn.init.zeros_(self.act_skip.bias)
        self.head_idm = nn.Sequential(
            nn.Linear(2 * N_PLAYERS * N_STATE, width), nn.LayerNorm(width),
            nn.SiLU(), nn.Linear(width, self.n_act)) if idm else None
        nn.init.trunc_normal_(self.pos, std=0.02)
        # Zero-init the heads so the model starts as an exact identity
        # predictor: deltas of 0 and unchanged flags. Training then only has to
        # learn the departure from standing still, and the reported skill is
        # honest from step one rather than climbing out of a random hole.
        nn.init.zeros_(self.head_state.weight); nn.init.zeros_(self.head_state.bias)
        nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)
        if self.head_move is not None:
            # Uniform over the vocabulary rather than a random opinion, for the
            # same reason the other heads start at identity.
            nn.init.zeros_(self.head_move.weight)
            nn.init.zeros_(self.head_move.bias)

    def idm_logits(self, state: torch.Tensor, nxt: torch.Tensor):
        """(current state, predicted next state) -> the action that did it.

        Deliberately given the PREDICTION rather than the truth: the gradient
        then flows back into the dynamics and pays it for making the prediction
        action-dependent. Fed the true next state instead, this would be an
        ordinary inverse-dynamics model and would teach the simulator nothing.
        """
        if self.head_idm is None:
            return None
        x = torch.cat([state.flatten(-2), nxt.flatten(-2)], -1)
        return self.head_idm(x)

    def forward(self, state: torch.Tensor, proj: torch.Tensor,
                actions: torch.Tensor, moves: torch.Tensor | None = None,
                want_moves: bool = False):
        """state [B,T,2,C], proj [B,T,2,K,F], actions [B,T,ticks,20].

        `moves` is [B,T,2] int64 vocabulary indices, required exactly when the
        model was built with a move embedding. Returns (next_state [B,T,2,C],
        next_proj [B,T,2,K,F]), or additionally the next-move logits
        [B,T,2,n_moves] when `want_moves`; index t is the prediction of step
        t+1, so the last one is the model's next frame.
        """
        B, T = state.shape[:2]
        st = state
        if self.state_history < T:
            # Zeroed, not truncated: the action at those positions must still
            # be seen, and the transformer keeps its positional identity.
            keep = st.new_zeros(1, T, 1, 1)
            keep[:, T - self.state_history:] = 1.0
            st = st * keep
            proj = proj * keep.unsqueeze(-1)
        parts = [st.reshape(B, T, -1), proj.reshape(B, T, -1),
                 actions.reshape(B, T, -1)]
        if self.move_embed is not None:
            if moves is None:
                raise ValueError(
                    "this simulator was trained with move identity and cannot "
                    "be rolled without it; pass moves [B,T,2]")
            parts.append(self.move_embed(moves.long()).reshape(B, T, -1))
        elif moves is not None:
            raise ValueError("moves passed to a simulator with no move head")
        x = torch.cat(parts, dim=-1)
        h = self.embed(x) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h, causal=True)
        h = self.norm(h)
        if self.act_skip is not None:
            # After the norm, so the skip is not rescaled away by it.
            h = h + self.act_skip(actions.reshape(B, T, -1))

        d = self.head_state(h).reshape(B, T, N_PLAYERS, N_STATE)
        # The residual is on the TRUE current state even when the input was
        # masked; masking is about what the model may look at, not about
        # changing what a delta is relative to.
        nxt = state.clone()
        cont = torch.tensor(CONTINUOUS, device=state.device)
        binr = torch.tensor(BINARY, device=state.device)
        # Continuous: a residual on the current value. Binary: absolute logits.
        nxt.index_copy_(-1, cont,
                        state.index_select(-1, cont) + d.index_select(-1, cont))
        nxt.index_copy_(-1, binr, d.index_select(-1, binr))
        p = self.head_proj(h).reshape(B, T, N_PLAYERS, self.slots, N_PROJF)
        if not want_moves:
            return nxt, p
        mlog = (self.head_move(h).reshape(B, T, N_PLAYERS, self.n_moves)
                if self.head_move is not None else None)
        return nxt, p, mlog


def load_sim(path, device="cpu") -> tuple[StateDynamics, dict]:
    """Rebuild a saved simulator from its own recorded shape.

    Every consumer needs the same six numbers -- slots, width, depth, history,
    ticks and the config -- and constructing the module by hand at each call
    site is the failure `docs/BUGS.md` §8 records: `horizon_ablation` built a
    world model from defaults, got `Missing key(s): hud_head...`, and the fix
    was to route every load through one function. Same fix, applied before it
    happens again.

    `ticks` is defaulted rather than required because `train_state_dynamics`
    writes it into `best_h1.pt` and not into `sim.pt`. The fallback is the
    config's `frame_skip`, which is what a run that never passed --frame-skip
    used; a run that DID pass one and saved only `sim.pt` would be
    misreconstructed, so the number is checked against the action width the
    weights actually have.
    """
    import torch as _torch
    d = _torch.load(path, map_location=device, weights_only=False)
    cfg = d["cfg"]
    ticks = int(d.get("ticks") or cfg.frame_skip)
    # Checkpoints written before the flag existed were all trained through the
    # squashing path, so that is what "absent" means -- not "unknown".
    m = StateDynamics(cfg, slots=int(d["slots"]), width=int(d["width"]),
                      depth=int(d["depth"]), history=int(d["history"]),
                      ticks=ticks,
                      proj_feedback=str(d.get("proj_feedback") or "sigmoid"),
                      n_moves=int(d.get("n_moves") or 0),
                      move_dim=int(d.get("move_dim") or 0),
                      idm=bool(d.get("idm")),
                      state_history=int(d.get("state_history") or 0),
                      # Absent means the checkpoint predates the skip, which is
                      # the same thing as having been trained without it.
                      act_skip=bool(d.get("act_skip")))
    # The action block is `ticks * 2 * 10` wide, so the saved embedding states
    # the stride outright. Believing a stale `ticks` here would pair state at
    # one rate with buttons at another and report nothing wrong.
    want = m.embed.weight.shape[1]
    got = d["model"]["embed.weight"].shape[1]
    if want != got:
        raise ValueError(
            f"{path}: rebuilt input width {want} but the checkpoint's is {got}. "
            f"ticks={ticks} slots={d['slots']} is not the shape this was "
            f"trained at; the checkpoint predates the `ticks` stamp and its "
            f"frame skip cannot be recovered from the config alone.")
    m.load_state_dict(d["model"])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    meta = {k: v for k, v in d.items() if k != "model"}
    meta["ticks"] = ticks
    meta["proj_feedback"] = m.proj_feedback
    # The vocabulary travels with the weights. An embedding index means nothing
    # without the table that produced it, and a consumer that rebuilt the
    # mapping from its own corpus would silently permute every move.
    meta["move_vocab"] = d.get("move_vocab")
    return m, meta


def unroll(model: StateDynamics, state: torch.Tensor, proj: torch.Tensor,
           actions: torch.Tensor, steps: int,
           moves: torch.Tensor | None = None,
           flag_feedback: str = "sigmoid") -> torch.Tensor:
    """Differentiable autoregressive rollout, for the TRAINING loss.

    The teacher-forced objective never asks the model to consume its own
    output, and the first run showed exactly what that costs: skill +0.248 at
    sixteen steps and -1.02 at ONE, i.e. twice as wrong as predicting that
    nothing changes. A model can learn average dynamics that look good over a
    long horizon while being worse than a no-op per step, and RL steps.

    Shares its body with `rollout` below via `_unroll_impl`, so the thing
    trained and the thing measured cannot drift apart.
    """
    return _unroll_impl(model, state, proj, actions, steps, moves,
                        flag_feedback)


@torch.no_grad()
def rollout(model: StateDynamics, state: torch.Tensor, proj: torch.Tensor,
            actions: torch.Tensor, steps: int,
            moves: torch.Tensor | None = None,
            flag_feedback: str = "sigmoid") -> torch.Tensor:
    """Autoregressive rollout, which is how RL will actually use this.

    `state`/`proj` are the history [B,H,...]; `actions` is [B,H+steps,...] --
    the buttons the agent intends to press. Returns the predicted states
    [B,steps,2,C].

    THE BINARY CHANNELS COME BACK AS THEY WERE FED BACK, NOT AS LOGITS: a
    probability under "sigmoid" feedback, 0/1 under "hard"/"sample". Applying
    sigmoid again maps every "no" to 0.5. Three instruments did, which hid what
    the flags really do (2026-09-27, tests/test_flag_measurements.py): read
    correctly, `guarding` on the full-corpus sim is calibrated at step 1 (0.070
    vs 0.054) and then jumps to ~0.85 against 0.05 from step 2 on, once the
    model consumes its own output. That part is REAL, and the unroll loss is a
    suspect: it supervises continuous channels only, so a fed-back flag at step
    2+ has no loss of its own while gradient still flows through it.

    Teacher-forced error is not the number that matters: the policy consumes
    the model's own output, so what has to be measured is what happens when the
    model eats its own predictions. This is that path, used by both the
    evaluation and the trainer's rollout loss so they cannot diverge.
    """
    return _unroll_impl(model, state, proj, actions, steps, moves,
                        flag_feedback)


def _unroll_impl(model: StateDynamics, state: torch.Tensor, proj: torch.Tensor,
                 actions: torch.Tensor, steps: int,
                 moves: torch.Tensor | None = None,
                 flag_feedback: str = "sigmoid") -> torch.Tensor:
    """`flag_feedback` decides what the binary channels look like when the
    rollout eats its own output. MEASURED, and it is the dominant defect:

      "sigmoid"  squash logits into (0, 1). The original fix, and it does not
                 go far enough. Every binary channel in the corpus takes
                 EXACTLY two values -- `guarding` is 0.0 or 1.0, never
                 anything else -- so a fed-back 0.52 is a state that occurs
                 zero times in training. Measured on `mv2/base`: at rollout
                 step 1, fed real history, 90.4% of predicted `guarding`
                 values are below 0.1 and 3.7% are in the middle. At step 2,
                 fed its own output, **83.2% land in the middle** and the mean
                 jumps 0.075 -> 0.740. From there it never recovers a bimodal
                 shape. The rollout is conditioned on an impossible state from
                 its second step onward.
      "hard"     threshold at 0.5, so the model receives the two-valued input
                 it was trained on.
      "sample"   Bernoulli draw, which keeps the marginal rate honest when a
                 flag is genuinely uncertain rather than forcing a decision.

    Straight-through on "hard": the forward value is the threshold and the
    gradient is the sigmoid's, so `unroll` stays differentiable and the
    training path is not silently cut.
    """
    if flag_feedback not in ("sigmoid", "hard", "sample"):
        raise ValueError(f"flag_feedback {flag_feedback!r}")
    H = state.shape[1]
    s, p, mv = state, proj, moves
    out = []
    for k in range(steps):
        a = actions[:, k:k + H]
        ns, np_, mlog = model(s, p, a, mv, want_moves=True)
        # Built out of place -- index_copy_ is in-place and would break autograd
        # for `unroll`, which is the same code path.
        binr = torch.tensor(BINARY, device=ns.device)
        soft = torch.sigmoid(ns[:, -1:].index_select(-1, binr))
        if flag_feedback == "hard":
            soft = soft + ((soft > 0.5).to(soft.dtype) - soft).detach()
        elif flag_feedback == "sample":
            soft = soft + (torch.bernoulli(soft) - soft).detach()
        nxt = ns[:, -1:].index_copy(-1, binr, soft)
        s = torch.cat([s[:, 1:], nxt], dim=1)
        p = torch.cat([p[:, 1:], feed_proj(np_[:, -1:], model.proj_feedback)],
                      dim=1)
        if mv is not None:
            # Hard argmax, and detached: this is the input RL will actually
            # feed back, and a soft mixture over the vocabulary would train the
            # model on an embedding no real frame ever produces. The state and
            # projectile gradient paths through the rollout are unchanged.
            mv = torch.cat([mv[:, 1:], mlog[:, -1:].argmax(-1).detach()], dim=1)
        out.append(s[:, -1])
    return torch.stack(out, dim=1)


def feed_proj(raw: torch.Tensor, mode: str) -> torch.Tensor:
    """Projectile head output -> the input the next step consumes.

    One function, used by the training unroll, the evaluation rollout and
    `rl/state_arena.py`, so the three cannot drift apart. They did once: the
    arena reimplemented this and would have had to be kept in step by hand,
    which is the same shape of bug as a probe fit on one checkpoint and read on
    another.
    """
    if mode == "sigmoid":
        return torch.sigmoid(raw)
    if mode == "raw":
        pb = torch.tensor([PROJ_FEATURES.index(n) for n in ("present", "hb")],
                          device=raw.device)
        return raw.index_copy(-1, pb, torch.sigmoid(raw.index_select(-1, pb)))
    raise ValueError(f"proj_feedback {mode!r}; want sigmoid or raw")
