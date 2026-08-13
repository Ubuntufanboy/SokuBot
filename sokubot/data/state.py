"""Ground-truth game state from the extractor's sidecar.

WHY THESE LABELS EXIST
----------------------
The encoder did not represent where the characters are. `spatial_probe.py`
measured a linear probe at AUC 0.540 for "did the characters swap sides"
against a 0.956 ceiling for "did the HUD swap sides", and `block_effect.py`
measured the predictor forecasting *less* damage when the defender holds no
direction at all -- the sign of guarding, inverted. Guarding in Hisoutensoku is
holding away from the opponent, so a model that cannot see which side they are
on cannot represent the mechanic the whole matchup runs through.

Five objectives failed to recover it from pixels: JEPA (0.540), plus inverse
dynamics (0.651), plus class-balanced IDM (0.688), direct dx supervision over
2003 replays (0.620), and a play-area-mirror augmentation built specifically to
forbid the HUD shortcut (0.6047). The reward stayed flat in the one dimension
the mechanic lives in: forcing the defender to hold AWAY rather than TOWARD was
worth -0.00025 +- 0.00761.

So the model no longer predicts latents of pixels. The state below IS the
representation, the rollout happens in it, and the encoder's whole job is to
read it off the screen.

THE CONSTRAINT IS UNCHANGED
---------------------------
It was always about *inference*: the policy at play time consumes pixels and
its own inputs, nothing else. These labels never reach it. They shape a world
model, which is the same asymmetric arrangement `hud_coef` already uses -- the
only difference is that the HUD is legible in pixels and position is not.

WHAT IS DERIVED HERE AND WHY
----------------------------
Three things are computed rather than copied, and each one is a correction the
raw column cannot supply:

  * `dx`/`dy` -- blocking is holding away from the opponent, so what matters is
    the *sign of their separation*, not two absolute coordinates. Absolute x is
    kept too, because corner pressure is a fact about the stage.
  * `vx` -- the game stores speed FACING-RELATIVE (measured: world dx tracks
    vx*direction at err 1.59 where the world frame errs 6.20). A velocity whose
    sign flips with facing makes "moving away" unlearnable, which is the exact
    failure this redesign exists to escape. It is multiplied out here, once, so
    no consumer can forget.
  * `spirit` -- as a fraction of `max_spirit`, because the denominator is not
    constant across characters.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

# Order is the contract with everything trained on these. Appending is safe;
# inserting silently permutes every previously trained head. The first seven
# are the original layout and are deliberately left where they were.
STATE_CHANNELS: tuple[str, ...] = (
    # --- the original seven, unmoved -------------------------------------
    "dx",            # x_opponent - x_me, normalised; sign is "which way is away"
    "facing",        # +1 if I face right, -1 if left, from the game's own flag
    "guarding",      # correct guard, ground or air
    # Blocked in the right direction at the wrong HEIGHT, not in the wrong
    # direction: SokuLib names the range ACTION_WRONGBLOCK_{HIGH,LOW}_*_
    # BLOCKSTUN, beside ACTION_RIGHTBLOCK_{HIGH,LOW}_*. Measured on a real
    # capture, 96.4% of these frames hold away, the same as a right block.
    "wrongblock",
    "crushed",       # the guard broke
    "knockdown",     # knocked down, or grabbed
    "airborne",      # y above the floor
    # --- appended with the full state capture -----------------------------
    "dy",            # y_opponent - y_me, normalised
    "x",             # my absolute position; corner pressure is about the stage
    "y",
    "vx",            # WORLD frame: the raw column is facing-relative
    "vy",
    "ax",            # the gravity/acceleration term
    "ay",
    "hitboxes",      # >0 means my attack is live RIGHT NOW
    "hurtboxes",
    "hitstop",
    "untech",        # frames before I can act again
    "action_frame",  # frames into the current action
    "hit_count",
    "hp",            # fraction of a full bar
    "spirit",        # fraction of max_spirit, which varies by character
    "spirit_delay",
    "timestop",
    "ground_dashes",
    "air_dashes",
    "correction",
    "combo_rate",
    "combo_hits",
    "combo_damage",
    "combo_limit",
    "proj_n",        # objects I own, live -- mostly visual effects
    "proj_hb",       # of those, how many can hurt the opponent RIGHT NOW
)
CH = {name: i for i, name in enumerate(STATE_CHANNELS)}

# Per-projectile features. Expressed relative to the player the projectile is
# flying AT, because that is the frame in which "is this about to hit me" is a
# subtraction rather than an inference.
PROJ_FEATURES: tuple[str, ...] = (
    "present",   # 1 if this slot holds an object at all
    "dx",        # projectile - target, normalised
    "dy",
    "vx",        # WORLD frame, as above
    "vy",
    "hb",        # carries a live hitbox: it can connect
    "closing",   # +1 if its velocity points at the target, -1 away, 0 still
)
PF = {name: i for i, name in enumerate(PROJ_FEATURES)}

# Maximum separation in game units, measured rather than assumed. A capture of
# a real match (10 073 frames) puts x in [40, 1240] and the signed separation in
# [-1200, +839], so the stage is about 1280 units wide with the origin at one
# edge -- NOT centred, which is what an earlier value of 600 here assumed.
STAGE_SPAN = 1200.0
FLOOR_EPS = 1.0            # y above this counts as airborne; the floor is 0.0
FULL_HP = 10000.0          # the exact int16 the game keeps, not the HUD's bar

# Scales that turn raw counters into something a network can consume without a
# normalisation layer having to discover the range. All generous: clipping is
# for pathology, not for shaping.
VEL_SCALE = 30.0           # |vx| over 30 a tick is a dash or a launch
FRAME_SCALE = 60.0         # a second of animation
COUNT_SCALE = 10.0

# Projectile coordinates are CLAMPED, and this is not defensive tidiness.
# Measured over 1.23M live projectile records: 98.9% sit on the stage, 1.07%
# legitimately just off it (a bullet that left the screen and has not despawned
# yet), 0.035% are far out -- and a rare few carry x = +-100 * 2^30, a specific
# repeated constant belonging to two object types that appear to use an extreme
# coordinate to mean "spans everything". Those are real game values, faithfully
# recorded, and a single one of them in a batch destroys the normalisation of
# every other feature in it. Four stage widths keeps every honest off-screen
# bullet and truncates only the sentinels.
POS_CLAMP = 4.0

# Columns the extractor writes per player, from dll/src/video_encoder.cpp.
_PER_PLAYER = (
    "x", "y", "vx", "vy", "ax", "ay", "dir", "action", "action_frame",
    "hitstop", "untech", "hitboxes", "hurtboxes", "hit_count", "hp", "spirit",
    "max_spirit", "spirit_delay", "timestop", "ground_dashes", "air_dashes",
    "correction", "combo_rate", "combo_hits", "combo_damage", "combo_limit",
    "guarding", "wrongblock", "crushed", "knockdown",
)
# `proj_n` and `proj_hb` are read; `proj_raw` is a capture-side diagnostic (the
# list length the game reported, so a refused walk is visible) that nothing
# here consumes. It is therefore NOT required -- a sidecar staged over a slow
# link may legitimately drop it, and failing on a column no code reads would
# reject the whole corpus for nothing.
_PROJ_COUNTS = ("proj_n", "proj_hb")
_PROJ_FIELDS = ("x", "y", "vx", "vy", "dir", "act", "hb")

LABEL_VALID = "label_valid"


def _open(path: Path):
    """Text handle for a sidecar, gzipped or not.

    Both `align_sidecar.py` and the collector write `.gz`: the full sidecar is
    5.5 MB a replay and about 0.9 MB gzipped, which over 2003 replays is the
    difference between 11 GB and 1.9 GB on a box that had 3.2 GB free.
    Deciding by suffix rather than by a flag means a caller cannot pass the
    wrong one.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def n_proj_slots(cols) -> int:
    """How many projectile slots this sidecar carries.

    Read from the header rather than assumed. The slot count is a capture-time
    constant that has already changed once -- 8, then 24, sized from an exact
    count once the first sizing turned out to have been measured through its
    own cap -- and a reader with the number baked in would either crash or, far
    worse, silently read 8 of 24.
    """
    k = 0
    while f"p1_pr{k}_x" in cols:
        k += 1
    return k


def has_state_columns(path: Path) -> bool:
    """True if this sidecar carries the full state, projectiles included."""
    with _open(path) as fh:
        cols = set(csv.DictReader(fh).fieldnames or [])
    return (all(f"p{i}_{c}" in cols for i in (1, 2) for c in _PER_PLAYER)
            and all(f"p{i}_{c}" in cols for i in (1, 2) for c in _PROJ_COUNTS)
            and n_proj_slots(cols) > 0)


def read_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
    """state csv -> (state, proj, action, valid).

        state   float32 [N, 2, len(STATE_CHANNELS)]
        proj    float32 [N, 2, slots, len(PROJ_FEATURES)]
        action  int32   [N, 2]        -- nominal ids, for an embedding
        valid   bool    [N]

    PLAYER-MAJOR, AND WHAT INDEX 1 MEANS FOR `proj`
    -----------------------------------------------
    `state[:, me]` is that player's own state. `proj[:, p]` holds the objects
    player p OWNS, expressed relative to the player they are flying at -- so
    the ones threatening me are `proj[:, 1 - me]`. That indexing is the
    ownership the game records, and the extractor has already sorted each
    player's slots danger-first (live hitbox, then nearest to the target), so
    slot 0 is the most urgent thing in the air.

    `action` is returned separately and as an integer because action ids are
    nominal: 801 is not "one more than" 800, and feeding them as a float asks a
    network to invent an ordering that does not exist.

    THE LAST RETURN IS NOT OPTIONAL, ON PURPOSE
    -------------------------------------------
    Labels produced by `pipeline/align_sidecar.py` are a *re-capture* of the
    replay attached to the video already on disk, and the two captures do not
    begin on exactly the same engine tick. Frames at the very start or end that
    the re-capture did not cover are filled by repeating the nearest real row
    and marked `label_valid = 0`, so that row i stays video frame i. Measured
    over 941 real pairs that is 420 frames in total, worst case 8 -- 0.004% --
    but they are invented and must not be trained on as if measured.

    Captures with no `label_valid` column -- anything taken directly rather
    than aligned -- report every frame valid, which is correct: nothing was
    filled in.
    """
    with _open(path) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        for i in (1, 2):
            missing = [c for c in (*_PER_PLAYER, *_PROJ_COUNTS)
                       if f"p{i}_{c}" not in cols]
            if missing:
                raise ValueError(
                    f"{path}: missing state columns for p{i}: {missing[:4]}. "
                    f"This capture predates the full state sidecar; use "
                    f"has_state_columns() to skip it, or re-run the capture.")
        slots = n_proj_slots(cols)
        has_valid = LABEL_VALID in cols
        raw = list(reader)

    if not raw:
        raise ValueError(f"{path}: no rows")
    n = len(raw)

    # A truncated final row is a real thing: the extractor is killed at
    # MAX_FRAMES and on scene changes, so the last line can be a partial write.
    # DictReader fills the missing fields with None, which otherwise surfaces
    # as `float() argument must be ... not NoneType` far from the cause.
    for probe in ("p1_x", "p2_x", "p1_proj_n", "p2_combo_limit"):
        if raw[-1].get(probe) is None:
            raw.pop()
            n -= 1
            break
    if n == 0:
        raise ValueError(f"{path}: no complete rows")

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in raw], dtype=np.float32)

    state = np.zeros((n, 2, len(STATE_CHANNELS)), dtype=np.float32)
    proj = np.zeros((n, 2, slots, len(PROJ_FEATURES)), dtype=np.float32)
    action = np.zeros((n, 2), dtype=np.int32)

    x = [col("p1_x"), col("p2_x")]
    y = [col("p1_y"), col("p2_y")]

    for me in (0, 1):
        them = 1 - me
        p = f"p{me + 1}_"
        s = state[:, me]

        s[:, CH["dx"]] = (x[them] - x[me]) / STAGE_SPAN
        s[:, CH["dy"]] = (y[them] - y[me]) / STAGE_SPAN
        s[:, CH["x"]] = x[me] / STAGE_SPAN
        s[:, CH["y"]] = y[me] / STAGE_SPAN

        # The game stores direction as a signed flag; normalise to +-1 so a
        # model never has to learn the encoding.
        d = col(p + "dir")
        facing = np.where(d > 0, 1.0, np.where(d < 0, -1.0, 0.0))
        s[:, CH["facing"]] = facing

        # FACING-RELATIVE -> world. Done once, here. See the module docstring.
        s[:, CH["vx"]] = col(p + "vx") * facing / VEL_SCALE
        s[:, CH["vy"]] = col(p + "vy") / VEL_SCALE
        s[:, CH["ax"]] = col(p + "ax") * facing / VEL_SCALE
        s[:, CH["ay"]] = col(p + "ay") / VEL_SCALE

        s[:, CH["hitboxes"]] = col(p + "hitboxes") / COUNT_SCALE
        s[:, CH["hurtboxes"]] = col(p + "hurtboxes") / COUNT_SCALE
        s[:, CH["hitstop"]] = col(p + "hitstop") / FRAME_SCALE
        s[:, CH["untech"]] = col(p + "untech") / FRAME_SCALE
        s[:, CH["action_frame"]] = col(p + "action_frame") / FRAME_SCALE
        s[:, CH["hit_count"]] = col(p + "hit_count") / COUNT_SCALE

        s[:, CH["hp"]] = col(p + "hp") / FULL_HP
        # max_spirit is per character, so the fraction is the comparable thing.
        # Guarded because a zero denominator between rounds is a real row.
        msp = col(p + "max_spirit")
        s[:, CH["spirit"]] = np.where(msp > 0, col(p + "spirit") / np.maximum(msp, 1.0), 0.0)
        s[:, CH["spirit_delay"]] = col(p + "spirit_delay") / FRAME_SCALE
        s[:, CH["timestop"]] = col(p + "timestop") / FRAME_SCALE
        s[:, CH["ground_dashes"]] = col(p + "ground_dashes")
        s[:, CH["air_dashes"]] = col(p + "air_dashes")
        s[:, CH["correction"]] = col(p + "correction") / 100.0

        s[:, CH["combo_rate"]] = col(p + "combo_rate")
        s[:, CH["combo_hits"]] = col(p + "combo_hits") / COUNT_SCALE
        s[:, CH["combo_damage"]] = col(p + "combo_damage") / FULL_HP
        s[:, CH["combo_limit"]] = col(p + "combo_limit") / COUNT_SCALE

        s[:, CH["guarding"]] = col(p + "guarding")
        s[:, CH["wrongblock"]] = col(p + "wrongblock")
        s[:, CH["crushed"]] = col(p + "crushed")
        s[:, CH["knockdown"]] = col(p + "knockdown")
        s[:, CH["airborne"]] = (y[me] > FLOOR_EPS).astype(np.float32)

        s[:, CH["proj_n"]] = col(p + "proj_n") / COUNT_SCALE
        s[:, CH["proj_hb"]] = col(p + "proj_hb") / COUNT_SCALE

        action[:, me] = col(p + "action").astype(np.int32)

        # --- this player's objects, relative to who they are flying at -----
        n_live = col(p + "proj_n")
        tx, ty = x[them], y[them]
        for k in range(slots):
            q = f"{p}pr{k}_"
            present = (n_live > k).astype(np.float32)
            pdir = col(q + "dir")
            pdir = np.where(pdir > 0, 1.0, np.where(pdir < 0, -1.0, 0.0))
            pdx = np.clip((col(q + "x") - tx) / STAGE_SPAN, -POS_CLAMP, POS_CLAMP)
            pdy = np.clip((col(q + "y") - ty) / STAGE_SPAN, -POS_CLAMP, POS_CLAMP)
            pvx = np.clip(col(q + "vx") * pdir / VEL_SCALE, -POS_CLAMP, POS_CLAMP)
            pvy = np.clip(col(q + "vy") / VEL_SCALE, -POS_CLAMP, POS_CLAMP)
            f = proj[:, me, k]
            f[:, PF["present"]] = present
            f[:, PF["dx"]] = pdx * present
            f[:, PF["dy"]] = pdy * present
            f[:, PF["vx"]] = pvx * present
            f[:, PF["vy"]] = pvy * present
            f[:, PF["hb"]] = (col(q + "hb") > 0).astype(np.float32) * present
            # Moving toward the target is -sign(dx)*sign(vx): if the target is
            # to the projectile's left (dx < 0) then closing means vx < 0.
            f[:, PF["closing"]] = (-np.sign(pdx) * np.sign(pvx)) * present

    valid = (np.array([int(float(r[LABEL_VALID])) for r in raw], dtype=bool)
             if has_valid else np.ones(n, dtype=bool))

    # `dx` is the channel that would expose a wrong position offset, and it is
    # checked rather than trusted: a stage is about 1200 units across, so a
    # separation far outside that is not a position at all. Projectiles are
    # clamped above and so cannot trip this; characters are not clamped,
    # precisely so that they can.
    lo, hi = float(state[..., CH["dx"]].min()), float(state[..., CH["dx"]].max())
    if not (-4.0 < lo and hi < 4.0):
        raise ValueError(
            f"{path}: dx spans [{lo:.2f}, {hi:.2f}] in stage widths, which is "
            f"not a position. Check CHAR_POSITION_X_OFFSET against the game "
            f"build before trusting any of these labels.")
    return state, proj, action, valid
