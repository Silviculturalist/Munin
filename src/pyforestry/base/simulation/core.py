# pyforestry/base/simulation/core.py
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import pi, sqrt

# ----------------------------- Actions & History ------------------------------
from typing import (
    Any,
    Callable,
    Concatenate,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    ParamSpec,
    TypedDict,
    Union,
    cast,
)

import pandas as pd

from pyforestry.base.helpers import CircularPlot, Tree, TreeName, parse_tree_species
from pyforestry.base.helpers.primitives import QuadraticMeanDiameter, StandBasalArea, Stems

# ---- Type aliases for metric containers ----
MetricKey = Union[TreeName, str]
MetricValue = Union[StandBasalArea, Stems, QuadraticMeanDiameter]
MetricView = Mapping[str, Mapping[MetricKey, MetricValue]]  # read-only facade


# Mutable store
class MetricMap(TypedDict):
    Stems: Dict[MetricKey, Stems]
    BasalArea: Dict[MetricKey, StandBasalArea]
    QMD: Dict[MetricKey, QuadraticMeanDiameter]


P = ParamSpec("P")
ActionFn = Callable[Concatenate["SimulationContext", P], None]


@dataclass(frozen=True)
class ActionSpec:
    """Declarative action descriptor with mode gating."""

    name: str
    fn: Callable[..., None]
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    # Legacy toggle (mapped to {"tree_list","spatial"}); prefer requires_modes.
    requires_tree_list: bool = False
    # New: explicit allowed modes, e.g. ["tree_list","spatial"] or []
    requires_modes: Optional[List[str]] = None


@dataclass
class HistoryEntry:
    t: float
    op: str
    details: Dict[str, Any]
    model_state: Dict[str, Any]
    metrics: Mapping[
        str, Mapping[Union[TreeName, str], Union[StandBasalArea, Stems, QuadraticMeanDiameter]]
    ]
    pre_snapshot: Dict[str, Any]
    post_snapshot: Dict[str, Any]


# --------------------------------- Context -----------------------------------


class SimulationContext:
    """
    Sandboxed, auditable working copy for a single run.

    mode ∈ {"spatial","tree_list","diameter_class","aggregate"}
    """

    def __init__(
        self,
        *,
        mode: str,
        area_ha: Optional[float],
        site: Optional[Any],
        origin_ref: Any,
        inventory: Dict[str, Any],
        initial_state: Dict[str, Any],
        model: Any,
        initial_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode not in ("spatial", "tree_list", "diameter_class", "aggregate"):
            raise ValueError("mode must be 'spatial','tree_list','diameter_class', or 'aggregate'")
        self.mode = mode
        self.area_ha = area_ha
        self.site = site
        self.origin_ref = origin_ref  # provenance only
        self.model = model
        # Internal inventory & metrics containers (dicts for mutability)
        self._metrics: MetricMap = {"Stems": {}, "BasalArea": {}, "QMD": {}}
        self._dclass: Dict[Any, Dict[str, List[float]]] = {}

        if self.mode in ("tree_list", "spatial"):
            self.plots: List[CircularPlot] = self._deepcopy_plots(inventory["plots"])
            self._metrics = self._recompute_metrics_tree_list(self.plots)
        elif self.mode == "aggregate":
            self._metrics = self._normalize_aggregate_metrics(inventory["metrics"])
        else:  # "diameter_class"
            self._dclass = self._normalize_dclass_inventory(inventory["dclass"])
            self._metrics = self._recompute_metrics_dclass(self._dclass)

        self.state: Dict[str, Any] = dict(initial_state)
        self.attrs: Dict[str, Any] = dict(initial_attrs or {})
        self.history: List[HistoryEntry] = []
        self.state.setdefault("t", 0.0)
        self.state.setdefault("last_dt", 0.0)

    # ------------------------------ Public API --------------------------------

    def grow(self, years: float) -> None:
        pre = self.snapshot()
        t0 = self.state.get("t", 0.0)

        self.model.grow(self, years)
        self.state["t"] = t0 + years
        self.state["last_dt"] = years

        self._refresh_metrics()
        post = self.snapshot()
        self._append_history("grow", {"dt": years}, pre, post)

    def do(self, action: str, **kwargs: Any) -> None:
        actions = self.model.available_actions()
        if action not in actions:
            raise KeyError(f"Action '{action}' not available for this model.")
        spec: ActionSpec = actions[action]

        # New gating
        requires_modes = set(spec.requires_modes or [])
        if spec.requires_tree_list:
            requires_modes.update({"tree_list", "spatial"})
        if requires_modes and self.mode not in requires_modes:
            req_str = ", ".join(sorted(requires_modes))
            raise RuntimeError(
                f"Action '{action}' requires mode in {{{req_str}}}, got '{self.mode}'."
            )

        pre = self.snapshot()
        spec.fn(self, **kwargs)
        self._refresh_metrics()
        post = self.snapshot()
        self._append_history(f"action:{action}", {"params": kwargs}, pre, post)

    def snapshot(self) -> Dict[str, Any]:
        if self.mode in ("tree_list", "spatial"):
            n_plots = len(self.plots)
            n_trees = sum(len(p.trees) for p in self.plots)
            tree_stats = {"n_plots": n_plots, "n_trees": n_trees}
        else:
            tree_stats = {"n_plots": 0, "n_trees": 0}

        ba = float(self._metrics["BasalArea"]["TOTAL"]) if "BasalArea" in self._metrics else 0.0
        stems = float(self._metrics["Stems"]["TOTAL"]) if "Stems" in self._metrics else 0.0
        qmd = float(self._metrics["QMD"]["TOTAL"]) if "QMD" in self._metrics else 0.0

        return {
            "mode": self.mode,
            "t": self.state.get("t", 0.0),
            "metrics_total": {"BasalArea": ba, "Stems": stems, "QMD": qmd},
            "tree_stats": tree_stats,
            "state": copy.deepcopy(self.state),
        }

    @property
    def metrics(self) -> MetricView:
        m = self._metrics
        return cast(
            MetricView,
            {
                "Stems": dict(m["Stems"]),
                "BasalArea": dict(m["BasalArea"]),
                "QMD": dict(m["QMD"]),
            },
        )

    def to_pandas(self) -> pd.DataFrame:
        rows = []
        for h in self.history:
            rows.append(
                {
                    "t": h.t,
                    "op": h.op,
                    "details": h.details,
                    "ba_total": float(h.metrics.get("BasalArea", {}).get("TOTAL", 0.0)),
                    "n_total": float(h.metrics.get("Stems", {}).get("TOTAL", 0.0)),
                    "qmd_total_cm": float(h.metrics.get("QMD", {}).get("TOTAL", 0.0)),
                    "model_state": h.model_state,
                }
            )
        return pd.DataFrame(rows)

    # Aggregate helpers
    def set_aggregate_metrics(self, *, ba_total: float, stems_total: float) -> None:
        self._metrics.setdefault("BasalArea", {})
        self._metrics.setdefault("Stems", {})
        self._metrics.setdefault("QMD", {})
        self._metrics["BasalArea"]["TOTAL"] = StandBasalArea(ba_total, species=None, precision=0.0)
        self._metrics["Stems"]["TOTAL"] = Stems(stems_total, species=None, precision=0.0)
        self._recompute_qmd()

    def scale_stems(self, factor: float) -> None:
        total_n = float(self._metrics["Stems"]["TOTAL"])
        total_ba = float(self._metrics["BasalArea"]["TOTAL"])
        new_n = max(0.0, total_n * factor)
        new_ba = max(0.0, total_ba * factor)
        self.set_aggregate_metrics(ba_total=new_ba, stems_total=new_n)

    # Diameter-class helper
    def set_diameter_class(self, dclass: Dict[Any, Dict[str, List[float]]]) -> None:
        self._dclass = self._normalize_dclass_inventory(dclass)
        self._metrics = self._recompute_metrics_dclass(self._dclass)

    # ----------------------------- Internal utils -----------------------------

    def _refresh_metrics(self) -> None:
        if self.mode in ("tree_list", "spatial"):
            computed = self._recompute_metrics_tree_list(self.plots)
            self._metrics = cast(
                MetricMap,
                {
                    "Stems": dict(computed.get("Stems", {})),
                    "BasalArea": dict(computed.get("BasalArea", {})),
                    "QMD": dict(computed.get("QMD", {})),
                },
            )

        elif self.mode == "aggregate":
            self._recompute_qmd()
        else:
            computed = self._recompute_metrics_dclass(self._dclass)
            self._metrics = cast(
                MetricMap,
                {
                    "Stems": dict(computed.get("Stems", {})),
                    "BasalArea": dict(computed.get("BasalArea", {})),
                    "QMD": dict(computed.get("QMD", {})),
                },
            )

    def _recompute_qmd(self) -> None:
        try:
            ba = float(self._metrics["BasalArea"]["TOTAL"])
            n = float(self._metrics["Stems"]["TOTAL"])
            qmd_val = sqrt((40000.0 * ba) / (pi * n)) if (ba > 0 and n > 0) else 0.0
        except KeyError:
            qmd_val = 0.0
        self._metrics.setdefault("QMD", {})
        self._metrics["QMD"]["TOTAL"] = QuadraticMeanDiameter(qmd_val, precision=0.0)

    def _append_history(
        self, op: str, details: Dict[str, Any], pre: Dict[str, Any], post: Dict[str, Any]
    ) -> None:
        model_state = {k: v for k, v in self.state.items()}
        entry = HistoryEntry(
            t=self.state.get("t", 0.0),
            op=op,
            details=copy.deepcopy(details),
            model_state=model_state,
            metrics=self.metrics,
            pre_snapshot=pre,
            post_snapshot=post,
        )
        self.history.append(entry)

    def _deepcopy_plots(self, plots: Iterable[CircularPlot]) -> List[CircularPlot]:
        out: List[CircularPlot] = []
        for p in plots:
            new_trees = [
                Tree(
                    position=getattr(t, "position", None),
                    species=getattr(t, "species", None),
                    age=getattr(t, "age", None),
                    diameter_cm=getattr(t, "diameter_cm", None),
                    height_m=getattr(t, "height_m", None),
                    weight_n=getattr(t, "weight_n", 1.0),
                    uid=getattr(t, "uid", None),
                )
                for t in p.trees
            ]
            out.append(
                CircularPlot(
                    id=p.id,
                    occlusion=p.occlusion,
                    position=p.position,
                    area_m2=p.area_m2,
                    site=p.site,
                    AngleCount=[],  # never copy AngleCount tallies into sim inventory
                    trees=new_trees,
                )
            )
        return out

    def _normalize_aggregate_metrics(self, metrics_in: Dict[str, Dict[Any, Any]]) -> MetricMap:
        stems_dict = cast(Dict[MetricKey, Stems], dict(metrics_in.get("Stems", {})))
        ba_dict = cast(Dict[MetricKey, StandBasalArea], dict(metrics_in.get("BasalArea", {})))
        qmd_dict: Dict[MetricKey, QuadraticMeanDiameter] = {}
        stems_dict.setdefault("TOTAL", Stems(0.0))
        ba_dict.setdefault("TOTAL", StandBasalArea(0.0))
        out_map = cast(
            MetricMap,
            {"Stems": stems_dict, "BasalArea": ba_dict, "QMD": qmd_dict},
        )
        self._metrics = out_map
        self._recompute_qmd()
        return out_map

    def _recompute_metrics_tree_list(self, plots: Iterable[CircularPlot]) -> MetricMap:
        species_data: Dict[TreeName, Dict[str, List[float]]] = {}

        def _eff_area_ha(p: CircularPlot) -> float:
            area_ha = p.area_ha or 1.0
            return area_ha * (1 - p.occlusion) if (1 - p.occlusion) > 0 else area_ha

        for plot in plots:
            eff = _eff_area_ha(plot)
            by_sp: Dict[TreeName, List[Tree]] = {}
            for t in plot.trees:
                sp = getattr(t, "species", None)
                if sp is None:
                    continue
                if isinstance(sp, str):
                    sp = parse_tree_species(sp)
                by_sp.setdefault(sp, []).append(t)

            for sp, trs in by_sp.items():
                stems = sum(getattr(t, "weight_n", 1.0) for t in trs)
                stems_ha = stems / eff
                ba_sum = 0.0
                for t in trs:
                    d_cm = float(getattr(t, "diameter_cm", 0.0) or 0.0)
                    r_m = (d_cm / 100.0) / 2.0
                    ba_sum += pi * (r_m**2) * getattr(t, "weight_n", 1.0)
                ba_ha = ba_sum / eff
                species_data.setdefault(sp, {"stems_per_ha": [], "basal_area_per_ha": []})
                species_data[sp]["stems_per_ha"].append(stems_ha)
                species_data[sp]["basal_area_per_ha"].append(ba_ha)

        stems_dict: Dict[Union[TreeName, str], Stems] = {}
        ba_dict: Dict[Union[TreeName, str], StandBasalArea] = {}
        total_stems_val = 0.0
        total_ba_val = 0.0
        for sp, vals in species_data.items():
            s_vals = vals["stems_per_ha"]
            b_vals = vals["basal_area_per_ha"]
            stems_mean = sum(s_vals) / len(s_vals) if s_vals else 0.0
            ba_mean = sum(b_vals) / len(b_vals) if b_vals else 0.0
            stems_dict[sp] = Stems(stems_mean, species=sp, precision=0.0)
            ba_dict[sp] = StandBasalArea(ba_mean, species=sp, precision=0.0)
            total_stems_val += stems_mean
            total_ba_val += ba_mean

        stems_dict["TOTAL"] = Stems(total_stems_val, species=None, precision=0.0)
        ba_dict["TOTAL"] = StandBasalArea(total_ba_val, species=None, precision=0.0)
        qmd_dict: Dict[Union[TreeName, str], QuadraticMeanDiameter] = {}
        if total_stems_val > 0 and total_ba_val > 0:
            total_qmd = sqrt((40000.0 * total_ba_val) / (pi * total_stems_val))
        else:
            total_qmd = 0.0
        qmd_dict["TOTAL"] = QuadraticMeanDiameter(total_qmd, precision=0.0)
        return cast(
            MetricMap,
            {"Stems": stems_dict, "BasalArea": ba_dict, "QMD": qmd_dict},
        )

    def _normalize_dclass_inventory(
        self, dclass_in: Dict[Any, Dict[str, List[float]]]
    ) -> Dict[Any, Dict[str, List[float]]]:
        out: Dict[Any, Dict[str, List[float]]] = {}
        for key, rec in dclass_in.items():
            mids = list(rec.get("bin_mids_cm", []))
            nph = list(rec.get("n_per_ha", []))
            if len(mids) != len(nph):
                raise ValueError(f"Diameter-class arrays length mismatch for {key}.")
            out[key] = {"bin_mids_cm": mids, "n_per_ha": nph}
        return out

    def _recompute_metrics_dclass(self, dclass: Dict[Any, Dict[str, List[float]]]) -> MetricMap:
        stems_dict: Dict[Union[TreeName, str], Stems] = {}
        ba_dict: Dict[Union[TreeName, str], StandBasalArea] = {}
        total_n = 0.0
        total_ba = 0.0
        for key, rec in dclass.items():
            mids = rec["bin_mids_cm"]
            nph = rec["n_per_ha"]
            n_sp = sum(nph)
            ba_sp = 0.0
            for D_cm, n_i in zip(mids, nph, strict=False):
                r_m = (float(D_cm) / 100.0) / 2.0
                ba_sp += float(n_i) * (pi * r_m * r_m)
            stems_dict[key] = Stems(n_sp, species=key if key != "TOTAL" else None, precision=0.0)
            ba_dict[key] = StandBasalArea(
                ba_sp, species=key if key != "TOTAL" else None, precision=0.0
            )
            if key != "TOTAL":
                total_n += n_sp
                total_ba += ba_sp
        stems_dict["TOTAL"] = Stems(total_n, species=None, precision=0.0)
        ba_dict["TOTAL"] = StandBasalArea(total_ba, species=None, precision=0.0)
        qmd_dict: Dict[Union[TreeName, str], QuadraticMeanDiameter] = {}
        if total_ba > 0.0 and total_n > 0.0:
            qmd_dict["TOTAL"] = QuadraticMeanDiameter(
                sqrt((40000.0 * total_ba) / (pi * total_n)), precision=0.0
            )
        else:
            qmd_dict["TOTAL"] = QuadraticMeanDiameter(0.0, precision=0.0)
        return cast(
            MetricMap,
            {"Stems": stems_dict, "BasalArea": ba_dict, "QMD": qmd_dict},
        )

    # Used by ensemble to log vector updates
    def _log_external_update(self, op: str, details: Dict[str, Any]) -> None:
        pre = self.snapshot()
        self._refresh_metrics()
        post = self.snapshot()
        self._append_history(op, details, pre, post)
