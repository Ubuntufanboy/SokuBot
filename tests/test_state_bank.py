"""The PPO bank must read the sidecars the Amarel capture actually writes.

A fresh `runner.collect --no-video --gzip-csv` capture keeps its state columns inside
`inputs.csv.gz`; there is no separate `state.csv`. The simulator trainer read those; the bank did not,
so PPO would have found "no usable sidecars" in the same directory the simulator trained on.

    python -m pytest tests/test_state_bank.py -q
"""
from __future__ import annotations

import gzip

from sokubot.data import state_bank
from sokubot.data.state import STATE_CHANNELS

from test_state_loader import HEADER, SLOTS, _row

SKIP = 5


def _capture(dirpath, n_rows, *, gz=True, name="inputs.csv", header=HEADER):
    dirpath.mkdir(parents=True)
    rows = [_row(frame=i, p1_a=i % 2) for i in range(n_rows)]
    text = ",".join(header) + "\n" + "\n".join(",".join(r[c] for c in header) for r in rows) + "\n"
    if gz:
        with gzip.open(dirpath / (name + ".gz"), "wt") as f:
            f.write(text)
    else:
        (dirpath / name).write_text(text)


def test_a_fresh_gzipped_capture_is_read(tmp_path):
    _capture(tmp_path / "corpus" / "w0" / "5263003-abcdef12", 400)
    S, P, A, E, V, names = state_bank.build([tmp_path / "corpus"], SKIP, SLOTS, verbose=False)
    assert names == ["5263003-abcdef12"]
    assert len(S) == 400 // SKIP and S.shape[1:] == (2, len(STATE_CHANNELS))
    assert P.shape[1:3] == (2, SLOTS)
    assert A.shape == (len(S), SKIP, 20)
    assert A[:, :, 4].sum() > 0                      # the buttons came through too


def test_an_uncompressed_capture_is_read(tmp_path):
    _capture(tmp_path / "corpus" / "w3" / "5262777-12345678", 400, gz=False)
    *_, names = state_bank.build([tmp_path / "corpus"], SKIP, SLOTS, verbose=False)
    assert names == ["5262777-12345678"]


def test_a_buttons_only_inputs_csv_is_still_rejected(tmp_path):
    """Old corpus captures hold only buttons in inputs.csv. They must be skipped, not misread."""
    buttons_only = HEADER[:24]
    _capture(tmp_path / "corpus" / "old", 400, header=buttons_only)
    _capture(tmp_path / "corpus" / "new", 400)
    *_, names = state_bank.build([tmp_path / "corpus"], SKIP, SLOTS, verbose=False)
    assert names == ["new"]
