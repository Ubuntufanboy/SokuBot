"""Pixels -> the policy's observation, legitimately. No game memory.

    vs = VisionState.load("~/rl/encoder.pt", device="cpu")
    vs.calibrate(pad, capture)          # once per round: who am I?
    state, proj = vs.read(frame)        # [2,33], [2,8,7] in ME-first order

WHAT THE ENCODER GIVES AND WHAT IT DOES NOT
--------------------------------------------
It reports the scene in LEFT-TO-RIGHT screen order, because that is what a
camera can see. It does not say which character is the agent, and that is not a
shortcoming to be trained away: seven arms across two resolutions, three feature
grids and four data scales left that bit at 0.518-0.585 against a 0.506 base
rate. There is no fact in the play area that names Player 1. A human knows
because their health bar is top-left.

So identity is recovered the way a person would recover it if the HUD were
covered: PRESS SOMETHING AND SEE WHO MOVES. That uses only the agent's own
inputs and the pixels, which is exactly what the inference constraint permits,
and it is exact rather than probabilistic -- `whoami` did it during the
memory-fed match and reported 135.0 units of movement against 0.0 for the other
character.

TRACKING, AND WHY IT IS THE HARD PART
--------------------------------------
Calibration answers "am I the left character" once. The two then swap sides
about 11 times a minute, so the answer has to be carried forward. That is
nearest-neighbour association: the agent's position moves a little between
decisions, so of the two characters reported this frame, the agent is whichever
is closer to where the agent was.

The failure mode is honest and worth stating: association is least reliable
exactly when the characters are close, which is also when they cross. With the
encoder's positional residual at roughly 164 game units against a typical
separation of 400, a crossup at close range can flip the assignment. `confidence`
reports the margin so a caller can see it happening, and `recalibrate_if_unsure`
exists because pressing a button and re-checking costs one decision step and
fixes it outright.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..data.state import CH, PROJ_FEATURES, STAGE_SPAN, STATE_CHANNELS

N_STATE = len(STATE_CHANNELS)
N_PROJF = len(PROJ_FEATURES)


class VisionState:
    """Encoder plus the identity bookkeeping the encoder deliberately omits."""

    def __init__(self, net, mu, sd, size: int, slots: int, device: str = "cpu"):
        self.net, self.size, self.slots, self.device = net, size, slots, device
        self.mu, self.sd = mu.to(device), sd.to(device)
        self.i_am_left: bool | None = None
        self.last_my_x: float | None = None
        # The character head, when the checkpoint has one. `my_char` is which
        # of the twenty the agent picked -- configuration, known before the
        # match, not read from the game.
        self.n_char = 0
        self.char_head: str | None = None
        self.chead = None
        self.my_char: int | None = None
        self.opp_char: int | None = None
        self.mirror_match = False
        self.char_flips = 0
        self.char_margin = 0.0
        self.used_char_head = False
        self._last_char_logits = None
        # Frames where the two rows' logit for the agent's own character are
        # nearly equal. Reported, not acted on: hysteresis was measured to be
        # WORSE than nothing for the position tracker (50% against 64%)
        # because it also suppressed the ~25 genuine side changes a match, and
        # that lesson is not re-learned here on a hunch.
        self.char_min_margin = 1.0
        self._char_low = 0
        # Running agreement between what the agent commanded and what each row
        # actually did. None until the pilot starts reporting its commands.
        self._id_score = None
        self._prev_x = None
        self._id_decay = 0.999
        self.id_disagreements = 0
        self.id_samples = 0
        # Below this the evidence has not separated the rows yet, so fall
        # through to the old rule rather than act on noise.
        self._id_floor = 1e-4
        self.last_my_hp: float | None = None
        self.last_conf: float = 0.0

    # CHANNELS THAT PASS THE CORPUS GATE AND ARE NOISE ON THE REAL GAME.
    #
    # `r2_floor` asks "does the network beat its own corpus mean". vx scores
    # 0.052 there and vy 0.084 -- over the line, but barely, and a fit that
    # thin does not survive the trip to a live window. Measured against the
    # game's own memory over two matches, correctly paired:
    #
    #             corpus R2      live R2
    #     vx        0.052         -0.38
    #     vy        0.084         -0.47
    #     timestop  0.529         -0.15
    #
    # Worse than a constant on the thing they are for. The docstring below
    # already made this argument for an earlier encoder whose vx and vy came
    # out negative on the corpus too; this one squeaks over zero and is let
    # through, which is the whole failure -- the gate is evaluated on the
    # distribution the encoder was fitted to.
    #
    # Motion is not lost. The policy reads a 12-step history at 83 ms spacing
    # and x is live at R2 0.62, so velocity is recoverable from the positions.
    # What it loses is a fabricated velocity presented as a measurement.
    LIVE_FAILED = ("vx", "vy", "timestop")

    @classmethod
    def load(cls, path, device: str = "cpu",
             r2_floor: float = 0.0) -> "VisionState":
        """Load, and DROP any channel that does not beat its own corpus mean.

        `r2_floor` is not a tuning knob, it is the definition of the metric.
        R^2 <= 0 says, in the units the encoder was scored in, that the
        constant mean predictor is at least as good as the network -- so
        forwarding that channel hands the policy noise where it could have been
        handed a clean "unknown". Measured on encoder2: vx -0.027 and
        vy -0.077, both negative, because two frames 33 ms apart move a
        character a few pixels and that did not survive the 480->224 downscale.

        Dropped channels take the corpus-mean fill like the 23 channels that
        were never supervised at all, which the policy's normaliser maps to
        exactly zero. The policy is not blind to motion as a result: it reads a
        12-step history at 83 ms spacing, so velocity remains recoverable from
        the position sequence -- just not from a single pair.
        """
        from scripts.encoder_gate import Net
        d = torch.load(Path(path).expanduser(), map_location=device,
                       weights_only=False)
        # Layout is [state | p1_left | char_left | char_right]; a checkpoint
        # without the character head simply has n_char 0 and loads unchanged.
        n_char = int(d.get("n_char", 0))
        # The attention head keeps its logits OUT of the shared output layer,
        # so the main net's width depends on which head was trained.
        char_head = d.get("char_head") or ("linear" if n_char else None)
        n_tail = 0 if char_head == "attn" else 2 * n_char
        n_out = d["n_state"] + d["n_proj"] + 1 + n_tail
        # EVERY architectural choice comes from the checkpoint, with the old
        # defaults as fallbacks. This used to hardcode head="spatial", in_ch=6
        # and the default width and keypoints, which was fine while exactly one
        # encoder existed and becomes a silent wrong-weights load the moment an
        # ablation sweep produces a winner with a different width. A shape
        # mismatch would at least raise; `head` and `input` would not.
        mode = str(d.get("input", "pair"))
        in_ch = 3 if mode == "single" else 6
        net = Net(n_out, width=int(d.get("width", 32)),
                  head=str(d.get("head", "spatial")),
                  keypoints=int(d.get("keypoints", 32)),
                  downs=d["downs"], in_ch=in_ch).to(device)
        net.load_state_dict(d["net"])          # strict: a mismatch must not pass
        net.eval()
        obj = cls(net, d["mu"], d["sd"], d["size"], d["slots"], device)
        obj.mode = mode
        obj.r2 = np.array(d["r2"])
        obj.n_state, obj.n_proj = d["n_state"], d["n_proj"]
        obj.n_char = n_char
        obj.char_head = char_head
        obj.chead = None
        if char_head == "attn" and d.get("char_state"):
            from ..model.char_head import CharHead
            obj.chead = CharHead(int(d["feat_ch"]), n_char).to(device)
            obj.chead.load_state_dict(d["char_state"])
            obj.chead.eval()
        obj.char_acc = float(d.get("char_acc", float("nan")))
        obj.decide_acc = float(d.get("decide_acc", float("nan")))
        obj.supervised = tuple(d["supervised"])
        obj.fill = np.array(d["fill"], dtype=np.float32)
        obj.delta = int(d.get("delta", 2))
        obj.proj_features = tuple(d.get("proj_features", ()))
        obj.r2_proj = np.array(d.get("r2_proj", []), dtype=np.float32)
        # Same rule as the state channels: a projectile feature that loses to
        # its own corpus mean is written absent rather than guessed.
        obj.proj_trusted = bool(len(obj.r2_proj)
                                and float(obj.r2_proj.mean()) > r2_floor)
        # r2 is per OUTPUT, i.e. per (character, channel). Average the two
        # characters and keep or drop a channel for BOTH: the observation is
        # ego-ordered against statistics pooled over both chairs, so a channel
        # that is live for one player and dead for the other would arrive on
        # two different footings depending on which side the agent drew.
        per_ch = obj.r2[:obj.n_state].reshape(2, len(obj.supervised)).mean(0)
        obj.r2_per_channel = per_ch
        obj.trusted = tuple(n for n, r in zip(obj.supervised, per_ch)
                            if r > r2_floor and n not in cls.LIVE_FAILED)
        obj.dropped = tuple(n for n, r in zip(obj.supervised, per_ch)
                            if r <= r2_floor or n in cls.LIVE_FAILED)
        # hp and spirit stay in `trusted` deliberately, as a FALLBACK: on the
        # vision path serve_vision overwrites both with the client's HUD
        # reading right after `read()`, and a client that sends none still
        # plays on the encoder's guess rather than on the corpus mean. Which
        # of the two the policy actually got is in the `hud N` counter on the
        # server's status line -- check it rather than assuming.
        return obj

    # ------------------------------------------------------------------
    @staticmethod
    def _resize(frame: np.ndarray, size: int) -> np.ndarray:
        """Area-averaging downscale, matching how the training frames were made.

        Training used cv2's INTER_AREA. PIL's BOX filter is the same
        area-average, and PIL is already present on the game host where cv2 is
        not -- system Python on Arch is externally managed and installing into
        it to get one resize would be a poor trade. The filter has to match:
        a different downscale is a different input distribution, and the
        encoder has no idea it moved.
        """
        # Size check BEFORE the import: when the caller has already resized --
        # which the LAN client does, to send 301 KB instead of 1.38 MB -- there
        # is nothing to do, and importing an optional dependency to do nothing
        # is how the inference server died on a box without Pillow.
        if frame.shape[0] == size and frame.shape[1] == size:
            return frame
        from PIL import Image
        return np.asarray(Image.fromarray(frame).resize((size, size),
                                                        Image.BOX))

    @torch.no_grad()
    def _raw(self, frame: np.ndarray):
        """RGB frame(s) -> (state [2,33], proj [2,slots,7]) in LEFT/RIGHT order.

        `frame` is HxWx6 -- the older image stacked on the current one, because
        velocity is a derivative and one image cannot carry it. How those two
        become the network's input depends on how it was TRAINED, and getting
        that wrong is invisible: a `diff`-trained encoder handed a raw pair
        sees plausible images and returns confident nonsense.
        """
        if frame.shape[2] == 6:
            old = self._resize(frame[:, :, :3], self.size).astype(np.float32)
            cur = self._resize(frame[:, :, 3:], self.size).astype(np.float32)
        else:
            old = cur = self._resize(frame, self.size).astype(np.float32)
        mode = getattr(self, "mode", "pair")
        if mode == "single":
            stack = cur / 255.0
        elif mode == "diff":
            # Must match encoder_ablate.train's `batch()` exactly, including the
            # 0.5 offset and the 510 divisor.
            stack = np.concatenate([cur / 255.0, (cur - old) / 510.0 + 0.5], 2)
        else:
            stack = np.concatenate([old / 255.0, cur / 255.0], 2)
        x = torch.from_numpy(np.ascontiguousarray(stack)).permute(2, 0, 1)[None].to(self.device)
        out = self.net(x.float(), return_feat=True)
        o, feat = (out[0][0], out[1]) if isinstance(out, tuple) else (out[0], None)
        # Slice by POSITION, not by "everything but the last". The character
        # head appends 2 x 20 logits after the p1_left bit, and `o[:-1]` would
        # have quietly scaled forty logits by the state normaliser.
        n_reg = self.n_state + self.n_proj
        y = o[:n_reg] * self.sd + self.mu
        if not self.n_char:
            self._last_char_logits = None
        elif getattr(self, "chead", None) is not None:
            with torch.no_grad():
                self._last_char_logits = self.chead(feat)[0][0]
        else:
            self._last_char_logits = o[n_reg + 1:n_reg + 1 + 2 * self.n_char] \
                .view(2, self.n_char)
        # The encoder predicts only the channels that are in the pixels; the
        # rest are filled with the corpus mean, which the policy's normaliser
        # maps to exactly zero -- "I do not know", stated rather than guessed.
        st = np.tile(self.fill, (2, 1)).astype(np.float32)
        pred = y[:self.n_state].reshape(2, len(self.supervised)).cpu().numpy()
        # Only the channels that beat their own corpus mean are written; the
        # rest keep the fill, which is the honest statement of what a single
        # frame pair does not carry.
        keep = [i for i, n in enumerate(self.supervised) if n in self.trusted]
        st[:, [CH[self.supervised[i]] for i in keep]] = pred[:, keep]
        pr = np.zeros((2, self.slots, N_PROJF), dtype=np.float32)
        # THE CHANNEL THE POLICY WAS BLIND TO.
        #
        # Every match so far wrote all eight slots absent, so an opponent
        # throwing projectiles from across the screen was attacking something
        # the agent could not perceive at all -- measured in a real match,
        # projectiles were present on 75% of steps while the encoder reported a
        # constant. Slot 0 is the extractor's danger-first slot: the single most
        # urgent object in the air, which is what a human reacts to.
        if self.n_proj and self.proj_trusted:
            q = y[self.n_state:self.n_state + self.n_proj]
            pr[:, 0, :] = q.reshape(2, N_PROJF).cpu().numpy()
        # Flags are trained as regression here, so they arrive as arbitrary
        # reals; the policy was trained on 0/1 and its normalisation assumes
        # that range. Clamping is not cosmetic.
        for name in ("guarding", "wrongblock", "crushed", "knockdown", "airborne"):
            st[:, CH[name]] = np.clip(st[:, CH[name]], 0.0, 1.0)
        st[:, CH["hp"]] = np.clip(st[:, CH["hp"]], 0.0, 1.0)
        st[:, CH["spirit"]] = np.clip(st[:, CH["spirit"]], 0.0, 1.0)
        pr[..., PROJ_FEATURES.index("present")] = np.clip(
            pr[..., PROJ_FEATURES.index("present")], 0.0, 1.0)
        pr[..., PROJ_FEATURES.index("hb")] = np.clip(
            pr[..., PROJ_FEATURES.index("hb")], 0.0, 1.0)
        return st, pr

    def calibrate(self, press_and_sample, hold: str = "left",
                  settle_s: float = 0.35) -> str:
        """Press a direction, see which character moves, and remember it.

        `press_and_sample(hold, settle_s)` must drive the pad and return
        (frame_before, frame_after). Everything it uses is the agent's own
        input and the screen, so this stays inside the inference constraint.
        """
        before, after = press_and_sample(hold, settle_s)
        s0, _ = self._raw(before)
        s1, _ = self._raw(after)
        move = np.abs(s1[:, CH["x"]] - s0[:, CH["x"]])
        if float(move.max()) < 1e-4:
            return "no movement detected -- the pad is not reaching the game"
        moved_left = bool(move[0] > move[1])
        margin = float(abs(move[0] - move[1]) / max(move.max(), 1e-9))
        self.i_am_left = moved_left
        self.last_my_x = float(s1[0 if moved_left else 1, CH["x"]])
        self.last_my_hp = float(s1[0 if moved_left else 1, CH["hp"]])
        return (f"agent is the {'LEFT' if moved_left else 'RIGHT'} character "
                f"(moved {move[0]*STAGE_SPAN:.0f} vs "
                f"{move[1]*STAGE_SPAN:.0f} units, margin {margin:.2f})")

    def set_character(self, my_char: int, opp_char: int | None = None) -> str:
        """Tell the agent which character it picked. Replaces `calibrate`.

        This is configuration, not game state: the agent chose it at the
        select screen. Nothing is read from the process, and the pixels still
        do all the work of finding it on screen.

        A MIRROR MATCH IS THE ONE CASE THIS CANNOT ANSWER. With both rows the
        same class the character logits are symmetric by construction, so the
        head is not merely unreliable there, it is meaningless. When the
        opponent's character is known to match, identity falls back to the
        probe and says so, rather than reporting a confident coin toss. The
        real fix is palette -- Soku forces the two sides onto different colours
        -- and the labels for it are already carried in the checkpoint's
        sibling fields; it is not built yet.
        """
        if not self.n_char:
            return ("this encoder has no character head; use calibrate()")
        self.my_char = int(my_char)
        self.opp_char = None if opp_char is None else int(opp_char)
        self.mirror_match = (self.opp_char is not None
                             and self.opp_char == self.my_char)
        if self.mirror_match:
            return ("MIRROR MATCH: both players are character "
                    f"{self.my_char}, so the character head cannot separate "
                    "the rows. Falling back to calibrate().")
        return (f"agent is character {self.my_char}; identity now decided per "
                f"frame from the character head "
                f"(val decide-accuracy {getattr(self, 'decide_acc', float('nan')):.3f})")

    def identity_report(self) -> str:
        """One line for the match log, so a bad session is legible afterwards."""
        if not self.used_char_head:
            return (f"identity: pinned probe, {self.id_disagreements}/"
                    f"{self.id_samples} monitor disagreements")
        return (f"identity: character head, {self.char_flips} side changes, "
                f"{self._char_low} low-margin frames, last margin "
                f"{self.char_margin:+.2f}, monitor disagreed "
                f"{self.id_disagreements}/{self.id_samples}")

    def note_command(self, st_raw, cmd: float) -> None:
        """Feed one decision's commanded horizontal direction (-1, 0, +1).

        Called by the pilot BEFORE `read`, with the raw (screen-ordered) state
        so the accumulator sees motion in a frame that has not been reordered
        by the very assignment it is trying to establish.
        """
        x = np.asarray(st_raw)[:, CH["x"]].astype(float)
        if self._id_score is None:
            self._id_score = np.zeros(2)
        if self._prev_x is not None and cmd != 0.0:
            dx = x - self._prev_x
            # Weight by how far it moved: a decisive stride is worth more than
            # a pixel of encoder jitter, and jitter has no consistent sign.
            self._id_score = (self._id_decay * self._id_score
                              + np.sign(dx) * cmd * np.abs(dx))
        self._prev_x = x

    def read(self, frame: np.ndarray):
        """-> (state [2,33], proj [2,slots,7]) with the AGENT first.

        The policy's observation is ego-ordered, so this is where left/right
        becomes me/them. Needs either `set_character` (character head) or
        `calibrate` (the old probe) to have established who the agent is.
        """
        st, pr = self._raw(frame)
        # IDENTITY: FROM THE CHARACTER HEAD IF THERE IS ONE, PINNED IF NOT.
        #
        # Three ways of INFERRING it from motion were measured on a recorded
        # match against the game's own state, and all three failed:
        #
        #   continuity of position   51%   81 flips, one wrong run of 42 s
        #   commanded dx agreement   67%   at the encoder's live x error of 55
        #                                  units against a 50-unit character
        #   commanded vx agreement   0.4%  systematically INVERTED
        #
        # They fail for the same reason: they all ask "which row moved the way
        # I told it to", and at the encoder's live positional error that
        # question is close to a coin toss. Worse, the continuity rule's error
        # is ABSORBING -- one bad frame at a crossup locks the policy onto the
        # opponent's row until the next crossing.
        #
        # So the old path pinned the answer from a 3 s probe and never revised
        # it, which is wrong the other way: the characters swap sides about
        # eleven times a minute, and a pinned answer is stale after the first
        # crossup.
        #
        # The character head asks a DIFFERENT question -- "which of the twenty
        # characters is standing in this row" -- and that one is genuinely in
        # the pixels. It needs no probe, costs no playing time, is recomputed
        # every frame so it cannot get stuck, and is decided by appearance
        # rather than by motion. `my_char` is configuration: the agent knows
        # what it picked.
        used_head = False
        if self.n_char and self.my_char is not None and not self.mirror_match:
            cl = self._last_char_logits
            margin = float(cl[0, self.my_char] - cl[1, self.my_char])
            now_left = margin > 0.0
            self.char_margin = margin
            if abs(margin) < self.char_min_margin:
                self._char_low += 1
            if self.i_am_left is not None and now_left != self.i_am_left:
                self.char_flips += 1
            self.i_am_left = now_left
            used_head = True
        if self.i_am_left is None:
            raise RuntimeError(
                "identity unknown: set_character() with a character head, or "
                "calibrate() without one")
        # The motion tracker stays on as a MONITOR either way, never as a
        # steering input -- it is the instrument that says how often the live
        # answer disagrees with the one that failed before.
        if self._id_score is not None:
            guess = bool(self._id_score[0] > self._id_score[1])
            if guess != self.i_am_left:
                self.id_disagreements += 1
            self.id_samples += 1
        self.used_char_head = used_head
        me = 0 if self.i_am_left else 1
        self.last_my_x = float(st[me, CH["x"]])
        self.last_my_hp = float(st[me, CH["hp"]])
        order = [me, 1 - me]
        # `proj[:, p]` is what player p OWNS, so it reorders with its owner.
        return st[order], pr[order]

    @property
    def confidence(self) -> float:
        """Separation between the two association candidates, in game units.

        Small means the two characters are close and the assignment could flip.
        A character is about 50 units wide, so anything under that is a coin
        toss dressed as a measurement.
        """
        return self.last_conf
