"""A vision server that runs no model, for testing everything that is not inference.

    python -m scripts.serve_null                       # neutral: the agent never presses
    python -m scripts.serve_null --mode wiggle         # left, right, left...: see the pad move
    python -m scripts.serve_null --latency-ms 200      # a slow server
    python -m scripts.serve_null --drop-after 50       # dies after 50 decisions, keeps listening

then, on the game host:

    python -m scripts.play_cheat_match --server 127.0.0.1 --port 5599

Needs no torch, no encoder and no policy. See `sokubot/live/nullbrain.py`.
"""

from __future__ import annotations

import argparse
import time

from sokubot.live.nullbrain import MODES, NullBrain, NullServer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--mode", choices=MODES, default="neutral")
    ap.add_argument("--ticks", type=int, default=5)
    ap.add_argument("--history", type=int, default=12)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--drop-after", type=int, default=None)
    a = ap.parse_args()

    brain = NullBrain(a.mode, a.ticks, a.history, a.latency_ms, a.drop_after)
    srv = NullServer(brain, a.host, a.port)
    srv.start()
    srv.ready.wait(5)
    print(f"null server ({a.mode}) listening on {a.host}:{srv.port} -- "
          f"period {1000 * a.ticks / 60:.1f} ms, history {a.history}", flush=True)
    try:
        while True:
            time.sleep(5)
            print(f"  connections {srv.connections} | decisions answered "
                  f"{brain.decides} | dropped {brain.dropped}", flush=True)
    except KeyboardInterrupt:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
