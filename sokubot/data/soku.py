"""Bridge from SokuFrameExtractor captures to training windows.

Each capture produced by milestone 2 is a directory:

    <replay_id>/
      video.mp4    480x480 h264, exactly one frame per CSV row, 60 fps
      inputs.csv   global_frame,local_frame,p1_input,p2_input,<20 boolean columns>
      meta.json

and every worker keeps a ``manifest.jsonl`` recording which of those succeeded.

WHY THIS DECODES ON THE FLY INSTEAD OF PRE-BUILDING SHARDS
----------------------------------------------------------
200 hours at 60 fps is 43.2M frames. Decimated to the 15 Hz decision rate that
is still 10.8M frames, and stored as raw 224x224x3 uint8 that is ~1.6 TB -- an
order of magnitude more than the 130 GB of mp4 it came from. Video compression
is the only reason the corpus fits on a disk at all, so the training pipeline
has to consume it compressed.

The access pattern is chosen to make that cheap: one ffmpeg process per capture,
decoding **sequentially** and emitting only every ``frame_skip``-th frame, with
windows drawn from a shuffle buffer. Random seeks into h264 would cost a keyframe
re-decode per sample; sequential decode of a 480x480 stream runs at thousands of
frames per second and the pipe carries a quarter of them.

Action layout per tick (20 channels):
    p1: up down left right a b c d change spell
    p2: up down left right a b c d change spell
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..config import Config
from .window import frames_to_chw

BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
PLAYERS = ("p1", "p2")
ACTION_COLUMNS: tuple[str, ...] = tuple(f"{p}_{b}" for p in PLAYERS for b in BUTTONS)
ACTION_DIM = len(ACTION_COLUMNS)      # 20
INPUT_MASK_BITS = len(BUTTONS)        # 10


@dataclass
class Capture:
    replay_id: str
    video: Path
    inputs: Path
    frames: int


def discover_captures(roots: Sequence[Path | str]) -> List[Capture]:
    """Collect every ``status == "ok"`` capture from the manifests under `roots`.

    Only "ok" entries are used. Milestone 2 records failed and truncated
    captures in the same manifest with a reason, and those are exactly the
    directories where the CSV and the video may disagree.
    """
    out: List[Capture] = []
    seen: set[str] = set()
    for root in roots:
        for mf in sorted(Path(root).rglob("manifest.jsonl")):
            for line in mf.read_text(errors="ignore").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("status") != "ok":
                    continue
                rid = e.get("replay_id")
                if not rid or rid in seen:
                    continue
                video = Path(e.get("video", ""))
                if not video.is_absolute():
                    video = mf.parent / video
                inputs = video.parent / "inputs.csv"
                if not (video.exists() and inputs.exists()):
                    continue
                seen.add(rid)
                out.append(
                    Capture(
                        replay_id=rid,
                        video=video,
                        inputs=inputs,
                        frames=int(e.get("frames") or 0),
                    )
                )
    return out


def read_actions(path: Path) -> np.ndarray:
    """inputs.csv -> float32 [N, 20], one row per video frame.

    The extractor writes the button state twice: packed into ``p1_input`` /
    ``p2_input`` bitmasks and expanded into 20 boolean columns. Both are read and
    cross-checked here, which is a free bit-flip detector on data that has been
    sitting on disk -- a single flipped bit in either representation makes them
    disagree, and a silently corrupted action would be invisible in training but
    would teach the model a transition that never happened.
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        has_bools = all(c in cols for c in ACTION_COLUMNS)
        has_masks = "p1_input" in cols and "p2_input" in cols
        if not (has_bools or has_masks):
            raise ValueError(f"{path}: neither boolean columns nor input masks present")

        rows: List[np.ndarray] = []
        for n, row in enumerate(reader):
            if has_bools:
                vec = np.fromiter(
                    (float(row[c]) for c in ACTION_COLUMNS), dtype=np.float32, count=ACTION_DIM
                )
                if has_masks:
                    for pi, p in enumerate(PLAYERS):
                        mask = int(row[f"{p}_input"])
                        bits = np.fromiter(
                            ((mask >> i) & 1 for i in range(INPUT_MASK_BITS)),
                            dtype=np.float32, count=INPUT_MASK_BITS,
                        )
                        off = pi * INPUT_MASK_BITS
                        if not np.array_equal(bits, vec[off : off + INPUT_MASK_BITS]):
                            raise ValueError(
                                f"{path}:{n + 2}: {p} mask {mask} disagrees with its "
                                "boolean columns -- corrupted row"
                            )
            else:
                vec = np.zeros(ACTION_DIM, dtype=np.float32)
                for pi, p in enumerate(PLAYERS):
                    mask = int(row[f"{p}_input"])
                    off = pi * INPUT_MASK_BITS
                    for i in range(INPUT_MASK_BITS):
                        vec[off + i] = (mask >> i) & 1
            rows.append(vec)

    if not rows:
        raise ValueError(f"{path}: no rows")
    return np.stack(rows)


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; it is required to read Soku captures")
    return exe


def decode_frames(
    video: Path, size: int, frame_skip: int, chunk_frames: int = 64
) -> Iterator[np.ndarray]:
    """Yield every ``frame_skip``-th frame of `video`, resized to `size`, as uint8 HWC.

    Decimation happens inside ffmpeg (``select``), so three quarters of the
    frames never cross the pipe. ``-vsync 0`` is required: without it ffmpeg
    duplicates frames to keep the nominal frame rate and the stream stops being
    index-aligned with the CSV.
    """
    cmd = [
        _ffmpeg_bin(), "-v", "error", "-nostdin",
        "-i", str(video),
        "-vf", f"select='not(mod(n\\,{frame_skip}))',scale={size}:{size}:flags=bilinear",
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    nbytes = size * size * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * chunk_frames)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
        if proc.stderr:
            proc.stderr.close()


NATIVE = 480          # the capture resolution data/hud.py's geometry assumes


def decode_frames_hud(
    video: Path, size: int, frame_skip: int, chunk_frames: int = 64
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Like `decode_frames`, but also reads the HUD -- before the resize.

    Yields ``(frame[size,size,3] uint8, hud[8] float32)``.

    The HUD has to be read at the native 480 px, not at whatever the model is
    trained on, and that is the whole point. The spirit gauge is five six-pixel
    hexagons; at the 224 px this project trained at they are about three pixels
    and probe out of the latent at R^2 0.05. Reading them here, then handing the
    model a resized frame, means the *labels* keep the resolution even where the
    input does not.

    Decoding at 480 and resizing in-process costs 1.15x the pipe bytes against
    decoding at 448 directly, which is nothing next to what it buys.
    """
    from .hud import read_frame_hud
    cmd = [
        _ffmpeg_bin(), "-v", "error", "-nostdin",
        "-i", str(video),
        "-vf", f"select='not(mod(n\\,{frame_skip}))'",
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    nbytes = NATIVE * NATIVE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * 4)
    try:
        assert proc.stdout is not None
        import cv2
        while True:
            # A chunk at a time: read_frame_hud is vectorised over frames, and
            # per-frame calls would pay its setup cost 43 million times.
            raw = proc.stdout.read(nbytes * chunk_frames)
            if not raw or len(raw) < nbytes:
                break
            n = len(raw) // nbytes
            block = np.frombuffer(memoryview(raw)[: n * nbytes], dtype=np.uint8)
            block = block.reshape(n, NATIVE, NATIVE, 3)
            hud = read_frame_hud(block)
            for i in range(n):
                f = block[i]
                if size != NATIVE:
                    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
                yield np.ascontiguousarray(f), hud[i]
            if n < chunk_frames:
                break
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
        if proc.stderr:
            proc.stderr.close()


class SokuWindowDataset(IterableDataset):
    """Streams training windows out of SokuFrameExtractor captures.

    Captures are sharded across DataLoader workers, shuffled per epoch, and each
    is decoded once, sequentially. Windows are emitted through a shuffle buffer
    so consecutive samples in a batch are not consecutive in time -- without it a
    batch would be 128 near-identical frames and the batch statistics that both
    SIGReg and the projector's BatchNorm depend on would be meaningless.
    """

    def __init__(
        self,
        cfg: Config,
        captures: Sequence[Capture],
        shuffle_buffer: int = 2048,
        seed: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        if cfg.action_dim != ACTION_DIM:
            raise ValueError(
                f"config action_dim {cfg.action_dim} != Soku's {ACTION_DIM}; "
                "use Config.soku()"
            )
        self.cfg = cfg
        self.captures = list(captures)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.stride = stride
        if not self.captures:
            raise ValueError("no captures given")

    def _shard(self) -> List[Capture]:
        info = get_worker_info()
        if info is None:
            return self.captures
        return self.captures[info.id :: info.num_workers]

    def _windows(self, cap: Capture) -> Iterator[dict]:
        cfg = self.cfg
        skip, T = cfg.frame_skip, cfg.seq_len
        try:
            actions = read_actions(cap.inputs)
        except (ValueError, OSError) as exc:
            print(f"[soku] skipping {cap.replay_id}: {exc}")
            return

        frame_buf: List[np.ndarray] = []
        hud_buf: List[np.ndarray] = []
        idx = 0          # index of the next decision frame
        emitted = 0
        want_hud = cfg.hud_coef > 0
        stream = (decode_frames_hud(cap.video, cfg.image_size, skip) if want_hud
                  else ((f, None) for f in
                        decode_frames(cap.video, cfg.image_size, skip)))
        for frame, hud in stream:
            frame_buf.append(frame)
            if want_hud:
                hud_buf.append(hud)
            if len(frame_buf) > T:
                frame_buf.pop(0)
                if want_hud:
                    hud_buf.pop(0)
            idx += 1
            if len(frame_buf) < T:
                continue
            start = idx - T                       # decision index of frame_buf[0]
            # Action chunk for decision step d covers source frames
            # [d*skip, d*skip + skip). Require the last chunk to be complete.
            hi = (start + T - 1) * skip + skip
            if hi > len(actions):
                break
            emitted += 1
            if (emitted - 1) % self.stride:
                continue
            chunks = np.stack([
                actions[(start + k) * skip : (start + k) * skip + skip]
                for k in range(T)
            ])                                     # [T, skip, 20]
            sample = {
                "obs": frames_to_chw(np.stack(frame_buf), as_uint8=cfg.loader_uint8),
                "actions": torch.from_numpy(chunks).float(),
            }
            if want_hud:
                sample["hud"] = torch.from_numpy(np.stack(hud_buf)).float()
            yield sample

    def __iter__(self) -> Iterator[dict]:
        caps = self._shard()
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        rng.shuffle(caps)

        buf: List[dict] = []
        for cap in caps:
            for sample in self._windows(cap):
                if len(buf) < self.shuffle_buffer:
                    buf.append(sample)
                    continue
                j = rng.randrange(len(buf))
                buf[j], sample = sample, buf[j]
                yield sample
        rng.shuffle(buf)
        yield from buf


def build_soku_dataset(
    cfg: Config,
    roots: Sequence[Path | str],
    limit: Optional[int] = None,
    **kwargs,
) -> SokuWindowDataset:
    caps = discover_captures(roots)
    if not caps:
        raise FileNotFoundError(f"no ok captures found under {list(roots)}")
    if limit:
        caps = caps[:limit]
    return SokuWindowDataset(cfg, caps, **kwargs)
