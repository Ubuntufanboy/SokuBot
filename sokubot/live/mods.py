"""Which SWRSToys modules are on, as far as the harness needs to know.

Only one question so far: is giuroll (the rollback netcode mod) enabled? If it is, this is a
netplay session, and a few things that are harmless locally become harmful.

WHY `pause` MUST REFUSE UNDER NETPLAY
-------------------------------------
`pause` SIGSTOPs the game so the human can type without the game reading their keystrokes.
Against a rollback peer that is a disaster: the peer keeps simulating, the stopped side falls
behind and its queued input replays on SIGCONT, and the two states diverge -- a desync that
looks like a network fault. `resume` is always allowed, so a game that got stopped can be
started again.

The settings file is read, not the process: `ModLoaderSettings.json` is what the loader uses
at startup, and it is the only thing that can be checked before the game exists.
"""

from __future__ import annotations

import json
from pathlib import Path


def giuroll_enabled(game_dir: Path) -> bool:
    """True if giuroll is enabled in ModLoaderSettings.json.

    A missing file means no loader config, so no giuroll: False. A file that exists but
    cannot be read or parsed returns True: when we cannot tell, the safe answer for a
    question whose wrong answer desyncs a match is "assume netplay".
    """
    path = Path(game_dir) / "ModLoaderSettings.json"
    if not path.is_file():
        return False
    try:
        modules = json.loads(path.read_text())["modules"]
    except (OSError, ValueError, KeyError, TypeError):
        return True
    for key, val in modules.items():
        stem = key.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if stem == "giuroll.dll" and isinstance(val, dict) and val.get("enabled"):
            return True
    return False


def pause_refusal(cmd: str, game_dir: Path) -> str | None:
    """Why `cmd` must not run right now, or None if it may. Only `pause` can be refused."""
    if cmd == "pause" and giuroll_enabled(game_dir):
        return ("REFUSED: giuroll is enabled, so this is a netplay session and stopping the "
                "game would desync the rollback peer. Use `disarm`/`hands-off` instead; "
                "`resume` still works if it is already stopped.")
    return None
