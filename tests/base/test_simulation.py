# test_simulation_expanded.py
import math

import pytest

from pyforestry.base.helpers import PICEA_ABIES, AngleCount, CircularPlot, Stand, Tree
from pyforestry.base.simulation import (
    ContextEnsemble,
    ExampleStandGeneralModel,
)

# ----------------------------- Test fixtures ---------------------------------


def _tree_list_stand(with_positions=False):
    plots = [
        CircularPlot(
            id=1,
            area_m2=200.0,
            trees=[
                Tree(
                    species="Picea abies",
                    diameter_cm=20.0,
                    height_m=16.0,
                    weight_n=5,
                    position=(1.0, 2.0) if with_positions else None,
                ),
                Tree(
                    species="Picea abies",
                    diameter_cm=18.0,
                    height_m=14.0,
                    weight_n=4,
                    position=(2.0, 1.0) if with_positions else None,
                ),
            ],
        ),
    ]
    return Stand(area_ha=1.0, plots=plots)


def _ac_stand():
    ac1 = AngleCount(ba_factor=2.0, value=[10], species=[PICEA_ABIES], point_id="p1")
    ac2 = AngleCount(ba_factor=2.0, value=[12], species=[PICEA_ABIES], point_id="p2")
    p1 = CircularPlot(id=1, area_m2=200.0, AngleCount=[ac1])
    p2 = CircularPlot(id=2, area_m2=200.0, AngleCount=[ac2])
    return Stand(area_ha=1.0, plots=[p1, p2])


# --------------------------- Your existing tests -----------------------------


def test_build_modes_and_adapters():
    model = ExampleStandGeneralModel()
    ac = _ac_stand()
    # 1) diameter_class from Angle-Count
    ok, missing = model.can_build(ac, allow_adapters=True, mode_hint="diameter_class")
    assert ok, missing
    ctx_dc = model.build_context(ac, mode_hint="diameter_class")
    assert ctx_dc.mode == "diameter_class"
    assert pytest.approx(float(ac.BasalArea)) == float(ctx_dc.metrics["BasalArea"]["TOTAL"])
    assert pytest.approx(float(ac.Stems)) == float(ctx_dc.metrics["Stems"]["TOTAL"])
    ctx_dc.do("thin_smallest_classes", fraction=0.1)

    # 2) spatial pseudo from Angle-Count (opt-in)
    ctx_sp = model.build_context(ac, mode_hint="spatial", adapter_kwargs={"seed": 42})
    assert ctx_sp.mode in ("spatial", "aggregate")
    if ctx_sp.mode == "spatial":
        ctx_sp.do("thin_fraction", fraction=0.05)


def test_spatial_from_tree_list():
    model = ExampleStandGeneralModel()
    st = _tree_list_stand(with_positions=False)
    ok, missing = model.can_build(st, allow_adapters=True, mode_hint="spatial")
    assert ok, missing
    ctx = model.build_context(st, mode_hint="spatial")
    assert ctx.mode == "spatial"
    assert all(getattr(t, "position", None) is not None for p in ctx.plots for t in p.trees)


def test_ensemble_batch_writeback():
    model = ExampleStandGeneralModel()
    st = _tree_list_stand()
    # Build aggregate contexts with known totals
    ctxs = []
    for _ in range(8):
        ctx = model.build_context(st, mode_hint="aggregate")
        ctx.set_aggregate_metrics(ba_total=20.0, stems_total=1200.0)
        ctxs.append(ctx)
    ens = ContextEnsemble(ctxs, model=model)
    for _ in range(3):
        ens.grow(dt=1.0)
    for c in ctxs:
        assert float(c.metrics["BasalArea"]["TOTAL"]) > 20.0
        assert float(c.metrics["Stems"]["TOTAL"]) < 1200.0


# -------------------------- New/expanded coverage ----------------------------


def test_metrics_is_copy_not_view():
    """Mutating the returned mapping must not affect the internal store."""
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(), mode_hint="aggregate")
    ctx.set_aggregate_metrics(ba_total=30.0, stems_total=900.0)
    m1 = ctx.metrics  # snapshot 1
    m2 = ctx.metrics  # snapshot 2
    # Same values, different objects — proves it's a copy each time
    assert m1 is not m2
    assert m1.keys() == m2.keys()
    assert m1["Stems"] is not m2["Stems"]
    # Now change the *internal* store; a new snapshot should change, old should not
    ctx.set_aggregate_metrics(ba_total=31.0, stems_total=901.0)
    assert float(m1["Stems"]["TOTAL"]) == pytest.approx(900.0)
    assert float(ctx.metrics["Stems"]["TOTAL"]) == pytest.approx(901.0)


def test_qmd_recomputed_from_ba_and_n():
    """QMD should match sqrt((40000 * BA) / (pi * N))."""
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(), mode_hint="aggregate")
    BA, N = 25.0, 1000.0
    ctx.set_aggregate_metrics(ba_total=BA, stems_total=N)
    qmd = float(ctx.metrics["QMD"]["TOTAL"])
    expected = math.sqrt((40000.0 * BA) / (math.pi * N))
    assert qmd == pytest.approx(expected, rel=1e-9)


def test_action_mode_gating_thin_smallest_requires_dclass():
    """thin_smallest_classes must be rejected outside diameter_class mode."""
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(), mode_hint="tree_list")
    with pytest.raises(RuntimeError):
        ctx.do("thin_smallest_classes", fraction=0.1)


def test_dclass_normalize_mismatch_raises():
    """_normalize_dclass_inventory should reject mismatched arrays."""
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(), mode_hint="diameter_class")
    bad = {"X": {"bin_mids_cm": [10.0, 12.0], "n_per_ha": [100.0]}}
    with pytest.raises(ValueError):
        ctx.set_diameter_class(bad)


def test_deepcopy_plots_strips_anglecount_and_copies_trees():
    """Simulation inventory should not carry AngleCount tallies in plots."""
    model = ExampleStandGeneralModel()
    # Build a tree-list plot that *also* has AngleCount tallies; the sim copy should drop them.
    ac = AngleCount(ba_factor=2.0, value=[10], species=[PICEA_ABIES], point_id="px")
    p = CircularPlot(
        id=9,
        area_m2=200.0,
        AngleCount=[ac],
        trees=[Tree(species="Picea abies", diameter_cm=20.0, weight_n=1.0)],
    )
    st = Stand(area_ha=1.0, plots=[p])
    ctx = model.build_context(st, mode_hint="tree_list")
    assert all(len(pp.AngleCount) == 0 for pp in ctx.plots)


def test_to_pandas_history_shape_and_values():
    """History rows -> DataFrame columns and content."""
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(), mode_hint="aggregate")
    ctx.set_aggregate_metrics(ba_total=10.0, stems_total=100.0)
    ctx.do("fertilize", years=1.0)
    ctx.grow(0.5)
    df = ctx.to_pandas()
    assert {"t", "op", "ba_total", "n_total", "qmd_total_cm"}.issubset(set(df.columns))
    # last op is grow; totals should be updated
    assert df.iloc[-1]["op"] == "grow"
    assert df.iloc[-1]["ba_total"] > 10.0
    assert df.iloc[-1]["n_total"] < 100.0


def test_apply_mortality_rate_in_dclass_scales_totals():
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_ac_stand(), mode_hint="diameter_class")
    n0 = float(ctx.metrics["Stems"]["TOTAL"])
    ctx.do("apply_mortality_rate", rate=0.25)
    n1 = float(ctx.metrics["Stems"]["TOTAL"])
    assert n1 == pytest.approx(n0 * (1.0 - 0.25))


def test_thin_fraction_tree_list_reduces_stems():
    model = ExampleStandGeneralModel()
    ctx = model.build_context(_tree_list_stand(with_positions=True), mode_hint="tree_list")
    n0 = float(ctx.metrics["Stems"]["TOTAL"])
    ctx.do("thin_fraction", fraction=0.5)
    n1 = float(ctx.metrics["Stems"]["TOTAL"])
    assert n1 == pytest.approx(n0 * 0.5, rel=1e-12)


def test_spatial_adapter_seed_makes_positions_deterministic():
    """Adapter should be deterministic for a given seed."""
    model = ExampleStandGeneralModel()
    st1 = _tree_list_stand(with_positions=False)
    st2 = _tree_list_stand(with_positions=False)
    ctx1 = model.build_context(st1, mode_hint="spatial", adapter_kwargs={"seed": 1234})
    ctx2 = model.build_context(st2, mode_hint="spatial", adapter_kwargs={"seed": 1234})

    def _xy_list(ctx):
        coords = []
        for p in ctx.plots:
            for t in p.trees:
                pos = getattr(t, "position", None)
                # Prove to the type checker and the test that position exists
                assert pos is not None, "expected a position on every tree in spatial mode"
                # Support both tuple-like (x, y) and attribute (.x, .y) forms
                try:
                    x, y = pos  # tuple-like
                except Exception:
                    x, y = pos.x, pos.y
                coords.append((float(x), float(y)))
        return coords

    pos1 = _xy_list(ctx1)
    pos2 = _xy_list(ctx2)
    assert pos1 == pos2


def test_ensemble_batch_engine_path_and_logging():
    """Exercise the ContextEnsemble batch path by advertising a batch engine."""

    class BatchyModel(ExampleStandGeneralModel):
        def has_batch_engine(self):  # <- makes agg_ctxs non-empty
            return True

        # vectorized step used by PythonEngine
        def batch_grow_step(self, ba, n, dt, fert_mask):
            growth = self.ba_rel + self.fert_boost * fert_mask  # fert_mask ∈ {0,1}
            new_ba = ba * (1.0 + growth * dt)
            new_n = n * (1.0 - self.mort * dt)
            return new_ba, new_n

    model = BatchyModel()
    ctxs = []
    for i in range(4):
        ctx = model.build_context(_tree_list_stand(), mode_hint="aggregate")
        ctx.set_aggregate_metrics(ba_total=20.0, stems_total=1000.0)
        # mark half as fertilized so fert_mask=1.0
        ctx.attrs["fertilized_remaining_years"] = 1.0 if i % 2 == 0 else 0.0
        ctxs.append(ctx)

    ens = ContextEnsemble(ctxs, model=model)
    ens.grow(dt=1.0)

    # writeback happened and history was logged via _log_external_update("grow", ...)
    for i, c in enumerate(ctxs):
        hist_ops = [h.op for h in c.history]
        assert "grow" in hist_ops
        ba = float(c.metrics["BasalArea"]["TOTAL"])
        # fertilized contexts grew faster
        if i % 2 == 0:
            assert ba > 20.0 * (1.0 + model.ba_rel)  # got the +fert_boost
        else:
            assert ba == pytest.approx(20.0 * (1.0 + model.ba_rel))
