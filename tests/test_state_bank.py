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


def test_a_plain_inputs_csv_is_not_read_it_is_a_failed_or_killed_capture(tmp_path):
    """The runner gzips a capture only after it passes validation. A plain inputs.csv is what a
    capture killed mid-replay leaves behind, and must not enter the bank as a (truncated) match."""
    _capture(tmp_path / "corpus" / "w3" / "5262777-12345678", 400, gz=False)
    _capture(tmp_path / "corpus" / "w3" / "5263014-12345678", 400)
    *_, names = state_bank.build([tmp_path / "corpus"], SKIP, SLOTS, verbose=False)
    assert names == ["5263014-12345678"]


def test_a_buttons_only_inputs_csv_is_still_rejected(tmp_path):
    """Old corpus captures hold only buttons in inputs.csv. They must be skipped, not misread."""
    buttons_only = HEADER[:24]
    _capture(tmp_path / "corpus" / "old", 400, header=buttons_only)          # gzipped, no state
    _capture(tmp_path / "corpus" / "new", 400)
    *_, names = state_bank.build([tmp_path / "corpus"], SKIP, SLOTS, verbose=False)
    assert names == ["new"]


def test_the_simulator_trainer_reads_the_same_captures_and_zero_means_all(tmp_path):
    """--replays 0 used to read nothing at all (`kept >= 0` stops before the first replay)."""
    from scripts.train_state_dynamics import load_sequences
    for i in range(3):
        _capture(tmp_path / "corpus" / "w0" / f"52630{i:02d}-abcdef12", 400)
    for limit, want in ((0, 3), (2, 2)):
        out = load_sequences(tmp_path / "corpus", limit, SLOTS, SKIP)
        assert len(set(out[3].tolist())) == want, (limit, want)


def test_the_runners_scratch_directory_is_not_counted_as_a_replay(tmp_path):
    """Each capture shard holds a `.work` directory; with a limit it used to eat one slot."""
    _capture(tmp_path / "corpus" / "w0" / "5262777-11111111", 400)
    _capture(tmp_path / "corpus" / "w0" / "5263014-22222222", 400)
    (tmp_path / "corpus" / "w0" / ".work" / "replay").mkdir(parents=True)
    found = state_bank.find_replays([tmp_path / "corpus"], limit=2)
    assert [d.name for d in found] == ["5262777-11111111", "5263014-22222222"]


def test_the_simulators_corpus_cache_is_not_reused_once_the_corpus_grows(tmp_path, capsys):
    """A cache written while the capture was partway done must not serve the finished corpus."""
    from scripts.train_state_dynamics import cached_sequences
    corpus, cache = tmp_path / "corpus", tmp_path / "cache.npz"
    for i in range(2):
        _capture(corpus / "w0" / f"52630{i:02d}-abcdef12", 400)
    first = cached_sequences(corpus, 0, SLOTS, SKIP, cache)
    assert len(set(first[3].tolist())) == 2
    again = cached_sequences(corpus, 0, SLOTS, SKIP, cache)       # same corpus: served from cache
    assert "from cache" in capsys.readouterr().out and len(again[0]) == len(first[0])
    _capture(corpus / "w1" / "5263099-abcdef12", 400)              # the capture finished more
    grown = cached_sequences(corpus, 0, SLOTS, SKIP, cache)
    assert "rebuilding" in capsys.readouterr().out
    assert len(set(grown[3].tolist())) == 3


def test_a_required_cache_that_does_not_match_stops_instead_of_rebuilding(tmp_path):
    import pytest
    from scripts.train_state_dynamics import cached_sequences
    corpus, cache = tmp_path / "corpus", tmp_path / "cache.npz"
    for i in range(2):
        _capture(corpus / "w0" / f"52630{i:02d}-abcdef12", 400)
    with pytest.raises(SystemExit, match="no cache"):
        cached_sequences(corpus, 0, SLOTS, SKIP, cache, require=True)
    first = cached_sequences(corpus, 0, SLOTS, SKIP, cache)
    assert len(cached_sequences(corpus, 0, SLOTS, SKIP, cache, require=True)[0]) == len(first[0])
    _capture(corpus / "w1" / "5263099-abcdef12", 400)
    before = cache.read_bytes()
    with pytest.raises(SystemExit, match="refusing to re-parse"):
        cached_sequences(corpus, 0, SLOTS, SKIP, cache, require=True)
    assert cache.read_bytes() == before


def test_parse_state_on_streamed_lines_is_read_state_on_the_file(tmp_path):
    """The live env parses rows as they arrive; it must be the corpus parser, row for row."""
    import numpy as np
    from sokubot.data.state import parse_state, read_state
    _capture(tmp_path / "cap", 120, gz=False)
    path = tmp_path / "cap" / "inputs.csv"
    a = read_state(path)
    b = parse_state(iter(path.read_text().splitlines(keepends=True)), "live")
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_the_streaming_row_parser_is_parse_state_row_for_row(tmp_path):
    """The vs-COM actors' fast parser shares parse_state's channel code; check it on many rows."""
    import numpy as np
    from sokubot.data.state import RowParser, parse_state
    _capture(tmp_path / "cap", 200, gz=False)
    lines = (tmp_path / "cap" / "inputs.csv").read_text().splitlines()
    s, p, _, _ = parse_state([l + "\n" for l in lines])
    rp = RowParser(lines[0])
    for i, row in enumerate(lines[1:]):
        a, b = rp.parse(row)
        assert np.array_equal(a, s[i]) and np.array_equal(b, p[i]), i
