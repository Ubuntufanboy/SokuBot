"""The corpus as four arrays, cached, at decision rate.

    S  float32 [N, 2, C]          per-player state
    P  float32 [N, 2, K, F]       per-player projectile slots
    A  uint8   [N, ticks, 20]     both players' buttons for that decision step
    E  int32   [N]                which replay each step came from
    V  bool    [N]                every underlying frame carried a real label

WHY THIS EXISTS RATHER THAN `build_gyms --bank`
-----------------------------------------------
A gym start index only means something against the array the trainer samples
from. `scripts/build_gyms.py` can survey a sidecar tree or index a HUD bank,
and neither is the array a state-space RL run reads: that one is strided to the
decision rate, carries its actions pre-chunked, and holds only the projectile
slots the simulator was built for. Building gyms from one array and sampling
from another is the single easiest way to train on situations that are not the
situations -- so the trainer builds its gyms in-process, from these arrays,
and this module is what guarantees they are the same object.

READING IS THE EXPENSIVE PART, SO IT HAPPENS ONCE
-------------------------------------------------
`data/state.py:read_state` walks a 427-column CSV a row at a time. Over 500
gzipped sidecars that is tens of minutes; the resulting arrays are under a
gigabyte. The cache records the settings that shaped it, because a bank strided
at 5 and loaded by a run expecting 1 is wrong in a way that trains happily.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state import has_state_columns, read_state

BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
MIN_STEPS = 64          # a replay shorter than this is a menu, not a match


def read_actions(path: Path) -> np.ndarray:
    """inputs.csv[.gz] -> [N, 20] uint8, P1's ten buttons then P2's."""
    import csv
    import gzip
    op = gzip.open if path.suffix == ".gz" else open
    cols = [f"p{p}_{b}" for p in (1, 2) for b in BUTTONS]
    with op(path, "rt", newline="") as fh:
        return np.array([[int(float(row[c])) for c in cols]
                         for row in csv.DictReader(fh)], dtype=np.uint8)


def find_replays(roots: list[Path], limit: int = 0) -> list[Path]:
    """Replay directories under the given corpus roots, in a stable order.

    Stable because the frame offsets a gym index into are assigned in this
    order, so a bank rebuilt with the directories enumerated differently would
    make every previously saved index point somewhere else.
    """
    out: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        waves = sorted(w for w in root.glob("w*") if w.is_dir())
        for parent in (waves or [root]):
            out += [d for d in sorted(parent.iterdir()) if d.is_dir()]
    if limit:
        out = out[:limit]
    return out


def build(roots: list[Path], skip: int, slots: int, replays: int = 0,
          verbose: bool = True):
    """Read the corpus into the five arrays. See the module docstring."""
    S, P, A, E, V, names = [], [], [], [], [], []
    skipped = 0
    for d in find_replays(roots, replays):
        # A capture aligned to a corpus video carries its state in `state.csv*`; a FRESH capture
        # (runner.collect --no-video, as the Amarel regeneration makes them) carries it inside
        # `inputs.csv[.gz]`, which is what train_state_dynamics already reads. Without the last two
        # the PPO bank found no sidecars at all in a capture the simulator trained on happily.
        # `has_state_columns` below still rejects an old inputs.csv that holds only buttons.
        sc = next((c for c in (d / "state.csv.gz", d / "state.csv",
                               d / "state_s5.csv", d / "inputs.csv.gz",
                               d / "inputs.csv") if c.exists()), None)
        inp = next((c for c in (d / "inputs.csv", d / "inputs.csv.gz")
                    if c.exists()), None)
        if sc is None or inp is None:
            skipped += 1
            continue
        try:
            if not has_state_columns(sc):
                skipped += 1
                continue
            st, pr, _act, valid = read_state(sc)
            btn = read_actions(inp)
        except (ValueError, OSError, KeyError) as exc:
            # One killed capture must not end a build over two thousand of
            # them; the extractor is stopped at MAX_FRAMES and on scene
            # changes, so a partial final write is normal.
            if verbose:
                print(f"  skip {d.name}: {exc}")
            skipped += 1
            continue

        # A pre-strided sidecar is already at decision rate; a full-rate one is
        # not. Getting this backwards pairs state at frame t with buttons from
        # frame 5t, which is a lie the run cannot detect.
        strided = sc.name.startswith("state_s")
        if strided and skip != 5:
            skipped += 1
            continue
        n = len(st) if strided else len(st) // skip
        n = min(n, (len(btn) - (0 if strided else 0)) // (1 if strided else skip))
        if n < MIN_STEPS:
            skipped += 1
            continue
        rows = np.arange(n) if strided else np.arange(n) * skip
        step = 1 if strided else skip
        chunks = np.stack([btn[i * step:i * step + skip] for i in range(n)
                           if i * step + skip <= len(btn)])
        n = min(n, len(chunks))
        if n < MIN_STEPS:
            skipped += 1
            continue
        rows = rows[:n]
        # A decision step is only real if EVERY frame it covers was a real
        # label. `align_sidecar` fills frames the re-capture missed by
        # repeating a neighbour, and a gym selected on an invented frame drills
        # a situation that never happened.
        if strided:
            ok = valid[rows]
        else:
            ok = np.stack([valid[i * skip:i * skip + skip] for i in range(n)]).all(1)

        S.append(st[rows]); P.append(pr[rows][:, :, :slots])
        A.append(chunks[:n].astype(np.uint8))
        E.append(np.full(n, len(names), np.int32))
        V.append(ok)
        names.append(d.name)
        if verbose and len(names) % 100 == 0:
            print(f"  {len(names)} replays, {sum(len(s) for s in S)} steps",
                  flush=True)
    if not S:
        raise SystemExit(f"no usable sidecars under {[str(r) for r in roots]}")
    if verbose:
        print(f"  {len(names)} replays kept, {skipped} skipped, "
              f"{sum(len(s) for s in S)} decision steps", flush=True)
    return (np.concatenate(S), np.concatenate(P), np.concatenate(A),
            np.concatenate(E), np.concatenate(V), names)


def load(roots: list[Path], skip: int, slots: int, cache: Path,
         replays: int = 0, verbose: bool = True):
    """Cached `build`. The cache carries the settings that shaped it.

    A bank is only interchangeable with another built at the same stride, the
    same slot count and over the same replays. Recording those and refusing a
    mismatch costs one comparison; the alternative is a run that reads a
    5-frame step as a 1-frame one and reports nothing wrong.
    """
    cache = Path(cache).expanduser()
    key = {"skip": int(skip), "slots": int(slots), "replays": int(replays),
           "roots": sorted(str(Path(r).expanduser()) for r in roots)}
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        got = json.loads(str(d["key"])) if "key" in d.files else None
        if got == key:
            if verbose:
                print(f"bank: {len(d['S'])} decision steps from cache {cache}",
                      flush=True)
            return (d["S"], d["P"], d["A"], d["E"], d["V"],
                    [str(x) for x in d["names"]])
        print(f"bank: rebuilding, cache was built with {got}", flush=True)
    S, P, A, E, V, names = build([Path(r) for r in roots], skip, slots,
                                 replays, verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, S=S, P=P, A=A, E=E, V=V, names=np.array(names),
             key=json.dumps(key))
    if verbose:
        print(f"bank: {len(S)} decision steps -> {cache} "
              f"({cache.stat().st_size/1e9:.2f} GB)", flush=True)
    return S, P, A, E, V, names
