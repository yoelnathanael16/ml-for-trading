"""
Training progress reporter with self-correcting ETA.

Framework-agnostic: no Streamlit import. Consumers pass an on_update sink.
"""
import json
import time
import os
from typing import Callable, Optional

# (key, default_weight) — ARIMA dominates real wall time
STAGES = [
    ("preprocess", 0.05),
    ("gmm",        0.05),
    ("svm",        0.07),
    ("randomforest", 0.08),
    ("xgboost",    0.07),
    ("lightgbm",   0.07),
    ("arima",      0.61),
]
_STAGE_KEYS = [k for k, _ in STAGES]
_TOTAL_DEFAULT = sum(w for _, w in STAGES)


def fmt_eta(seconds: float) -> str:
    """Format seconds as 'm:ss'."""
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


class ProgressReporter:
    """
    Tracks multi-stage training progress and emits (label, fraction, eta_seconds)
    to an on_update callback after each advance/sub call.

    ETA is self-correcting: after each completed run, per-stage durations are
    persisted to timings_path and reloaded as weights on the next run.
    """

    def __init__(
        self,
        ticker: str,
        on_update: Optional[Callable[[str, float, float], None]] = None,
        timings_path: str = "models/.train_timings.json",
    ):
        self.ticker = ticker
        self.on_update = on_update
        self.timings_path = timings_path

        # Load saved per-stage durations → weights; fall back to defaults
        weights = self._load_weights()
        total = sum(weights[k] for k in _STAGE_KEYS)
        self._weights = {k: weights[k] / total for k in _STAGE_KEYS}

        self._t0: float = 0.0
        self._stage_t0: float = 0.0
        self._current_stage: Optional[str] = None
        self._completed: dict[str, float] = {}   # key → measured seconds
        self._done_fraction: float = 0.0          # cumulative weight of finished stages

    # ------------------------------------------------------------------
    def _load_weights(self) -> dict:
        defaults = dict(STAGES)
        try:
            with open(self.timings_path) as f:
                saved = json.load(f)
            # Merge: use saved values where available, else defaults
            return {k: float(saved.get(k, defaults[k])) for k in _STAGE_KEYS}
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return dict(defaults)

    def _save_timings(self):
        os.makedirs(os.path.dirname(self.timings_path) or ".", exist_ok=True)
        try:
            with open(self.timings_path, "w") as f:
                json.dump(self._completed, f)
        except OSError:
            pass  # non-fatal

    # ------------------------------------------------------------------
    def start(self):
        """Call once before the first stage."""
        self._t0 = time.monotonic()
        self._done_fraction = 0.0
        self._completed = {}

    def advance(self, label: str, stage_key: str):
        """Mark stage_key as starting; emit update at sub=0."""
        # Record duration of previous stage
        if self._current_stage is not None:
            self._completed[self._current_stage] = time.monotonic() - self._stage_t0
            self._done_fraction += self._weights.get(self._current_stage, 0.0)

        self._current_stage = stage_key
        self._stage_t0 = time.monotonic()
        self._emit(label, 0.0)

    def sub(self, label: str, stage_key: str, fraction: float):
        """Update intra-stage sub-progress (0–1). Used by ARIMA loop."""
        self._emit(label, min(max(fraction, 0.0), 1.0))

    def finish(self):
        """Call after last stage completes. Saves timings for next run."""
        if self._current_stage is not None:
            self._completed[self._current_stage] = time.monotonic() - self._stage_t0
        self._save_timings()

    # ------------------------------------------------------------------
    def _emit(self, label: str, sub: float):
        cur_weight = self._weights.get(self._current_stage or "", 0.0)
        fraction = min(self._done_fraction + sub * cur_weight, 1.0)
        elapsed = time.monotonic() - self._t0
        eta = elapsed * (1.0 - fraction) / max(fraction, 1e-6)
        if self.on_update:
            self.on_update(label, fraction, eta)


# ------------------------------------------------------------------
if __name__ == "__main__":
    # ponytail: minimal self-check — asserts fraction→1 and ETA monotonically decreases
    updates: list[tuple[float, float]] = []

    def _sink(label, frac, eta):
        updates.append((frac, eta))

    r = ProgressReporter("TEST", on_update=_sink, timings_path="/tmp/test_timings.json")
    r.start()

    for key, label in zip(_STAGE_KEYS, ["Preprocess", "GMM", "SVM", "RF", "XGB", "LGBM", "ARIMA"]):
        r.advance(label, key)
        time.sleep(0.01)  # simulate work
        if key == "arima":
            for i in range(1, 11):
                r.sub("ARIMA", "arima", i / 10)
                time.sleep(0.005)

    r.finish()

    fracs = [f for f, _ in updates]
    etas  = [e for _, e in updates]

    assert fracs[-1] >= 0.99, f"Final fraction {fracs[-1]} < 0.99"
    assert fracs == sorted(fracs), "Fractions not monotonically increasing"
    # ETA should broadly decrease (allow small noise from timing jitter)
    assert etas[0] > etas[-1], f"ETA didn't decrease: {etas[0]:.1f} → {etas[-1]:.1f}"
    print(f"OK — {len(updates)} updates, final fraction={fracs[-1]:.3f}, "
          f"ETA {etas[0]:.1f}s → {etas[-1]:.1f}s")
    print(f"fmt_eta(75) = {fmt_eta(75)}  (expect 1:15)")
    assert fmt_eta(75) == "1:15"
    print("All assertions passed.")
