"""A strategy's DNA: entry block + filters + exit plan + sides, for one market and timeframe."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .blocks import ENTRIES, EXIT_PARAMS, FILTERS, SIDES, describe_exit
from .indicators import Data

MAX_FILTERS = 2


def _sample_params(spec, rng):
    return {k: p.sample(rng) for k, p in spec.items()}


@dataclass
class Genome:
    symbol: str
    tf: str
    entry: dict                      # {"type": name, **params}
    filters: list[dict] = field(default_factory=list)
    exit: dict = field(default_factory=dict)
    side: str = "both"

    # --- identity ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "tf": self.tf, "entry": self.entry, "filters": self.filters,
                "exit": self.exit, "side": self.side}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(d["symbol"], d["tf"], dict(d["entry"]), [dict(f) for f in d.get("filters", [])], dict(d["exit"]),
                   d.get("side", "both"))

    @property
    def id(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return "g" + hashlib.sha1(blob.encode()).hexdigest()[:9]

    @property
    def family(self) -> str:
        return ENTRIES[self.entry["type"]].family

    def complexity(self) -> int:
        return len(self.filters) + (1 if self.exit.get("trail") != "none" else 0)

    def describe(self) -> str:
        e = ENTRIES[self.entry["type"]]
        parts = [e.describe(self.entry)]
        parts += [FILTERS[f["type"]].describe(f) for f in self.filters]
        side = {"long": "longs only", "short": "shorts only", "both": "long & short"}[self.side]
        return f"{'; '.join(parts)} | {describe_exit(self.exit)} | {side}"

    def short_name(self) -> str:
        e = self.entry
        main = ",".join(str(v) for k, v in e.items() if k != "type")
        flt = "+".join(f["type"] for f in self.filters)
        return f"{self.symbol} {self.tf} {e['type']}({main}){'+' + flt if flt else ''} {self.side}"

    # --- behaviour ---------------------------------------------------------------------
    def signals(self, d: Data) -> tuple[np.ndarray, np.ndarray]:
        long_, short = ENTRIES[self.entry["type"]].fn(d, self.entry)
        long_, short = long_.copy(), short.copy()
        for f in self.filters:
            fl, fs = FILTERS[f["type"]].fn(d, f)
            long_ &= np.asarray(fl, dtype=bool)
            short &= np.asarray(fs, dtype=bool)
        if self.side == "long":
            short[:] = False
        elif self.side == "short":
            long_[:] = False
        if self.entry["type"] == "time_of_day" and self.side == "both":
            short[:] = False   # can't be both long and short at the same moment
        return long_, short

    def overlays(self, d: Data) -> dict[str, np.ndarray]:
        out = {}
        for blk, p in [(ENTRIES[self.entry["type"]], self.entry)] + [(FILTERS[f["type"]], f) for f in self.filters]:
            if blk.overlay:
                try:
                    out.update(blk.overlay(d, p))
                except Exception:
                    pass
        return out

    # --- evolution ---------------------------------------------------------------------
    @classmethod
    def random(cls, symbol: str, tf: str, rng: np.random.Generator, entry_type: str | None = None,
               hints: dict | None = None) -> "Genome":
        et = entry_type or list(ENTRIES)[int(rng.integers(len(ENTRIES)))]
        entry = {"type": et, **_sample_params(ENTRIES[et].params, rng)}
        filters = []
        for _ in range(int(rng.choice([0, 1, 1, 2]))):
            ft = list(FILTERS)[int(rng.integers(len(FILTERS)))]
            if all(f["type"] != ft for f in filters):
                filters.append({"type": ft, **_sample_params(FILTERS[ft].params, rng)})
        ex = _sample_params(EXIT_PARAMS, rng)
        if ex["rr"] == 0.0 and ex["trail"] == "none":
            ex["trail"] = "chandelier"
        side = SIDES[int(rng.integers(3))]
        g = cls(symbol, tf, entry, filters, ex, side)
        return g.apply_hints(hints)

    def apply_hints(self, hints: dict | None) -> "Genome":
        """Lessons from live post-mortems, e.g. 'stops in this family were too tight'."""
        h = (hints or {}).get(self.family, {})
        if "sl_atr_min" in h:
            self.exit["sl_atr"] = max(self.exit["sl_atr"], min(4.0, float(h["sl_atr_min"])))
        return self

    def mutate(self, rng: np.random.Generator, rate: float = 0.35) -> "Genome":
        g = copy.deepcopy(self)
        r = rng.random()
        if r < 0.45:   # tune entry params
            spec = ENTRIES[g.entry["type"]].params
            for k, p in spec.items():
                if rng.random() < max(rate, 1 / len(spec)):
                    g.entry[k] = p.mutate(g.entry[k], rng)
        elif r < 0.75:  # tune exit
            for k, p in EXIT_PARAMS.items():
                if rng.random() < rate / 2:
                    g.exit[k] = p.mutate(g.exit[k], rng)
            if g.exit["rr"] == 0.0 and g.exit["trail"] == "none":
                g.exit["trail"] = "chandelier"
        elif r < 0.9:   # add / remove / tweak a filter
            if g.filters and rng.random() < 0.4:
                g.filters.pop(int(rng.integers(len(g.filters))))
            elif len(g.filters) < MAX_FILTERS:
                ft = list(FILTERS)[int(rng.integers(len(FILTERS)))]
                if all(f["type"] != ft for f in g.filters):
                    g.filters.append({"type": ft, **_sample_params(FILTERS[ft].params, rng)})
            elif g.filters:
                f = g.filters[int(rng.integers(len(g.filters)))]
                for k, p in FILTERS[f["type"]].params.items():
                    f[k] = p.mutate(f[k], rng)
        else:           # flip sides
            g.side = SIDES[int(rng.integers(3))]
        return g

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        g = copy.deepcopy(self)
        if rng.random() < 0.5:
            g.filters = copy.deepcopy(other.filters)
        if rng.random() < 0.5:
            g.exit = copy.deepcopy(other.exit)
        if rng.random() < 0.3:
            g.side = other.side
        return g

    def neighbors(self, rng: np.random.Generator, k: int = 16, scale: float = 0.1) -> list["Genome"]:
        """Nearby parameter sets (same structure) for the plateau / robustness test."""
        out = []
        for _ in range(k):
            g = copy.deepcopy(self)
            for k2, p in ENTRIES[g.entry["type"]].params.items():
                if p.kind != "choice":
                    g.entry[k2] = p.mutate(g.entry[k2], rng, scale)
            for k2 in ("sl_atr", "trail_atr", "be_r", "max_bars"):
                g.exit[k2] = EXIT_PARAMS[k2].mutate(g.exit[k2], rng, scale)
            for f in g.filters:
                for k2, p in FILTERS[f["type"]].params.items():
                    if p.kind != "choice":
                        f[k2] = p.mutate(f[k2], rng, scale)
            out.append(g)
        return out

    def for_symbol(self, symbol: str) -> "Genome":
        g = copy.deepcopy(self)
        g.symbol = symbol
        return g
