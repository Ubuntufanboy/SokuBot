"""Pull the capture corpus from the Hugging Face staging repos and unpack it.

    python -m scripts.fetch_corpus --out /root/corpus \
        --repos Smashlytics/soku-frames-a Smashlytics/soku-frames-b Smashlytics/soku-frames-c

Downloads each shard, extracts it, deletes the tar, and writes one combined
`manifest.jsonl` that `sokubot.data.soku` can discover.

WHY ONE SHARD AT A TIME
-----------------------
`snapshot_download` would fetch every tar first and then extract, which needs
room for the corpus twice over -- ~260 GB for a 130 GB dataset, on a box with
237 GB. Downloading, extracting and deleting each shard in turn keeps the peak
at the corpus plus one shard.

Download and extraction overlap across a small thread pool, because a single
stream leaves the link idle during untar and the CPU idle during transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def shard_list(api, repo: str) -> List[str]:
    return sorted(f for f in api.list_repo_files(repo, repo_type="dataset")
                  if f.startswith("shards/") and f.endswith(".tar"))


def fetch_one(api, repo: str, shard: str, out: Path, cache: Path) -> tuple[str, int, float]:
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=shard,
                            local_dir=str(cache))
    size = os.path.getsize(local)
    # tar is faster than tarfile here and releases the GIL, which matters when
    # several of these run at once.
    proc = subprocess.run(["tar", "xf", local, "-C", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"extract {shard}: {proc.stderr[:200]}")
    os.unlink(local)
    return shard, size, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=Path("/root/hf_cache"))
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="fetch at most N shards; 0 = all. For rehearsing the "
                         "fetch on a machine that cannot hold the corpus.")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    work = []
    for repo in args.repos:
        for s in shard_list(api, repo):
            work.append((repo, s))
    if not work:
        print("no shards found", file=sys.stderr)
        return 1
    if args.limit:
        work = work[: args.limit]
    print(f"{len(work)} shards across {len(args.repos)} repos", flush=True)

    total, done, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch_one, api, r, s, args.out, args.cache): (r, s)
                for r, s in work}
        for fut in as_completed(futs):
            repo, shard = futs[fut]
            try:
                name, size, dt = fut.result()
            except Exception as exc:
                print(f"FAILED {repo}:{shard}: {exc}", file=sys.stderr, flush=True)
                continue
            total += size
            done += 1
            el = time.time() - t0
            print(f"[{done}/{len(work)}] {repo.split('/')[-1]}:{name} "
                  f"{size/1e9:.2f} GB | {total/1e9:.1f} GB total, "
                  f"{total/1e6/el:.0f} MB/s avg", flush=True)

            st = os.statvfs(args.out)
            free = st.f_bavail * st.f_frsize / (1 << 30)
            if free < args.min_free_gb:
                print(f"free space {free:.1f} GB below floor; stopping",
                      file=sys.stderr)
                break

    # One manifest over everything that landed, so the dataset can find it.
    rows = 0
    with (args.out / "manifest.jsonl").open("w") as fh:
        for csv in sorted(args.out.rglob("inputs.csv")):
            d = csv.parent
            video = d / "video.mp4"
            if not video.exists():
                continue
            frames = sum(1 for _ in csv.open(errors="ignore")) - 1
            fh.write(json.dumps({"replay_id": d.name, "status": "ok",
                                 "video": str(video), "frames": frames}) + "\n")
            rows += 1

    el = time.time() - t0
    print(f"\n{rows} captures, {total/1e9:.1f} GB in {el/60:.1f} min "
          f"({total/1e6/el:.0f} MB/s) -> {args.out}/manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
