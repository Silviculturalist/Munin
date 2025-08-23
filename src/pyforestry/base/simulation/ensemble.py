# pyforestry/base/simulation/ensemble.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .core import SimulationContext


class BatchEngine:
    """Protocol for batch engines."""

    def grow(
        self, model: Any, vec: Dict[str, np.ndarray], dt: float, extra=None
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class PythonEngine(BatchEngine):
    """NumPy baseline."""

    def grow(
        self, model: Any, vec: Dict[str, np.ndarray], dt: float, extra=None
    ) -> Dict[str, np.ndarray]:
        ba = vec["ba"]
        n = vec["n"]
        fert_mask = (extra or {}).get("fert_mask", np.zeros_like(ba))
        if hasattr(model, "batch_grow_step"):
            new_ba, new_n = model.batch_grow_step(ba, n, dt, fert_mask)
        else:
            new_ba, new_n = ba, n
        return {"ba": new_ba, "n": new_n}


def _optional_numba_engine() -> Optional[BatchEngine]:
    try:
        import numba  # noqa: F401
    except Exception:
        return None
    return PythonEngine()


def _optional_jax_engine() -> Optional[BatchEngine]:
    try:
        import jax  # noqa: F401
    except Exception:
        return None
    return PythonEngine()


@dataclass
class ContextEnsemble:
    contexts: List[SimulationContext]
    model: Any
    engine: Optional[BatchEngine] = None

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = _optional_jax_engine() or _optional_numba_engine() or PythonEngine()

    def grow(self, dt: float) -> None:
        assert self.engine is not None
        agg_ctxs = [
            c
            for c in self.contexts
            if c.mode == "aggregate" and getattr(self.model, "has_batch_engine", lambda: False)()
        ]
        other_ctxs = [c for c in self.contexts if c not in agg_ctxs]
        for c in other_ctxs:
            c.grow(dt)
        if agg_ctxs:
            ba = np.array([float(c.metrics["BasalArea"]["TOTAL"]) for c in agg_ctxs], dtype=float)
            n = np.array([float(c.metrics["Stems"]["TOTAL"]) for c in agg_ctxs], dtype=float)
            fert_mask = np.array(
                [
                    1.0 if c.attrs.get("fertilized_remaining_years", 0.0) > 0.0 else 0.0
                    for c in agg_ctxs
                ],
                dtype=float,
            )
            out = self.engine.grow(
                self.model, {"ba": ba, "n": n}, dt, extra={"fert_mask": fert_mask}
            )
            for i, c in enumerate(agg_ctxs):
                c.set_aggregate_metrics(
                    ba_total=float(out["ba"][i]), stems_total=float(out["n"][i])
                )
                c.state["t"] = c.state.get("t", 0.0) + dt
                c.state["last_dt"] = dt
                c._log_external_update("grow", {"dt": dt})

    def do(self, name: str, **kwargs: Any) -> None:
        for c in self.contexts:
            c.do(name, **kwargs)

    def to_pandas(self):
        import pandas as pd

        return pd.concat(
            [c.to_pandas().assign(context_id=i) for i, c in enumerate(self.contexts)],
            ignore_index=True,
        )
