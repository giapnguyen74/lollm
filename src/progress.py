"""
progress.py — load-phase progress display, kept OUT of model code.

The decoupling: loading code (a family's `weights.load`, the shard reader) stays UI-free —
it accepts an optional `progress` callback and just calls `progress(done, total)` as it
goes, knowing nothing about how (or whether) that's shown. The *display* lives here.

`bar(desc)` returns such a callback driving a single progress bar — `tqdm` if importable
(animated, rate, ETA; it ships with transformers, so it's almost always there), otherwise
periodic one-line stderr updates. All output is on stderr, so it never pollutes stdout.

    report = bar("streaming → cpu")
    for i, x in enumerate(items):
        ...
        report(i + 1, len(items))         # model code only emits this signal
"""

from __future__ import annotations

import sys
import time


def bar(desc: str = ""):
    """Return a callback `cb(done, total)` that drives one progress bar (lazily opened on the
    first call, closed when done >= total). Pass it as a loader's `progress=` argument."""
    return _Display(desc)


class _Display:
    def __init__(self, desc):
        self.desc, self._impl = desc, None

    def __call__(self, done, total):
        if self._impl is None:
            self._impl = _open(self.desc, total)
        self._impl.to(done, total)
        if total and done >= total:
            self._impl.close()


def _open(desc, total):
    try:
        from tqdm import tqdm
        return _Tqdm(tqdm(total=total, desc=desc, unit="t",
                          file=sys.stderr, dynamic_ncols=True, leave=False))
    except Exception:
        return _Line(desc)


class _Tqdm:
    def __init__(self, pb):
        self.pb = pb

    def to(self, done, total):
        self.pb.n = done
        self.pb.refresh()

    def close(self):
        self.pb.close()


class _Line:
    def __init__(self, desc):
        self.desc, self.t0 = desc, time.time()

    def to(self, done, total):
        pct = (100 * done / total) if total else 0
        print(f"\r{self.desc}: {done}/{total} ({pct:3.0f}%)  {time.time()-self.t0:5.1f}s",
              end="", file=sys.stderr, flush=True)

    def close(self):
        print("", file=sys.stderr, flush=True)
