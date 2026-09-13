"""A detector that replays a black box's cached output for the current turn.

Flux and Nova-3 consumed the eval audio once, paid for, and their responses
are cached under results/asr/. To put them on the same chart as everything
else they must go through the same harness, so this detector answers
update() from the cache. It needs to know which turn it is on, which the
harness tells it through begin(turn).

Three uses, one class:
  Flux confidence      value = end_of_turn_confidence on every update; the
                       harness sweeps the fire threshold across it
  Flux as shipped      value = 1.0 from Flux's own EndOfTurn event on
  Nova-3 endpointing   value = 1.0 from Nova-3's speech_final on

Timing: the harness polls once per 32ms VAD frame. update() returns the
maximum of every cached value that arrived since the previous poll, so a
crossing that lasted one message is not lost between two frames, and holds
the last value otherwise. Event series carry only their 1.0s: the harness
scores the first fire, and a fire is not undone by the next message.
"""

from __future__ import annotations

from src.eot.base import Update

Series = list[tuple[float, float]]  # (available_ms, value), sorted


class ReplayedEOT:
    def __init__(
        self, name: str, series_by_turn: dict[str, Series], meta: dict | None = None
    ):
        self.name = name
        self._by_turn = series_by_turn
        self.meta = meta or {}
        self._series: Series = []
        self._i = 0
        self._value = 0.0

    def begin(self, turn) -> None:
        if turn.turn_id not in self._by_turn:
            raise KeyError(
                f"{self.name}: no cached output for {turn.turn_id}; the cache does "
                f"not cover the eval set. Not guessing (rule 3)."
            )
        self._series = self._by_turn[turn.turn_id]
        self._i, self._value = 0, 0.0

    def reset(self) -> None:
        self._i, self._value = 0, 0.0

    def update(self, u: Update) -> float:
        peak = None
        while self._i < len(self._series) and self._series[self._i][0] <= u.t_ms:
            self._value = self._series[self._i][1]
            peak = self._value if peak is None else max(peak, self._value)
            self._i += 1
        return float(self._value if peak is None else peak)


def flux_confidence_series(
    cache: dict, use_wall: bool = True, in_turn_only: bool = True
) -> dict[str, Series]:
    """end_of_turn_confidence at each message's arrival (wall clock on the
    audio timeline: network and model latency included), or at the server's
    own audio position with use_wall=False.

    Flux keeps reporting a confidence when no turn is open -- before its
    StartOfTurn, and after its EndOfTurn -- and on one-word backchannels it
    never opens a turn at all while that idle value drifts past 0.7. The
    number is defined as the confidence that the CURRENT turn has ended, its
    own EndOfTurn cannot fire without one, so by default only messages from
    StartOfTurn through EndOfTurn count. in_turn_only=False is the raw feed.
    """
    key = "wall_ms" if use_wall else "audio_ms"
    out = {}
    for tid, events in cache["turns"].items():
        series, active = [], False
        for e in sorted(events, key=lambda e: float(e[key])):
            if e["event"] == "StartOfTurn":
                active = True
            if active or not in_turn_only:
                series.append((float(e[key]), float(e["eot_confidence"])))
            if e["event"] == "EndOfTurn":
                active = False
        out[tid] = series
    return out


def flux_shipped_series(cache: dict) -> dict[str, Series]:
    """Flux's own EndOfTurn events, at the eot_threshold the pass was run with."""
    return {
        tid: sorted(
            (float(e["wall_ms"]), 1.0) for e in events if e["event"] == "EndOfTurn"
        )
        for tid, events in cache["turns"].items()
    }


def nova3_endpoint_series(cache: dict) -> dict[str, Series]:
    """Nova-3's speech_final results: Deepgram's endpointing on its own VAD."""
    return {
        tid: sorted(
            (float(p["audio_ms"]) + float(p["compute_ms"]), 1.0)
            for p in parts
            if p.get("speech_final")
        )
        for tid, parts in cache["turns"].items()
    }
