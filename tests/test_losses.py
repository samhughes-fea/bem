"""Prandtl tip and hub loss and the Buhl high-induction correction."""

from __future__ import annotations

import math

import pytest

from bem import (
    LOSS_MODELS,
    LOSS_NONE,
    LOSS_PRANDTL,
    LOSS_PRANDTL_BUHL,
    Inflow,
    OperatingPoint,
    OuterShape,
    aerodynamic_load_case,
)
from bem.losses import axial_induction, prandtl_factor


def _outer(*, hub: float = 2.0, tip: float = 20.0) -> OuterShape:
    return OuterShape(
        z_m=(10.0, 18.0),
        chord_m=(1.5, 0.8),
        twist_deg=(0.0, 0.0),
        aerofoil=("plate", "plate"),
        hub_radius_m=hub,
        tip_radius_m=tip,
    )


def _point() -> OperatingPoint:
    return OperatingPoint(
        inflow=Inflow(speed_m_s=8.0, density_kg_m3=1.25),
        rotor_speed_rad_s=1.2,
        pitch_deg=0.0,
        n_blades=3,
    )


def _polar(cl: float, cd: float, cm: float):
    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        assert aerofoil == "plate"
        assert math.isfinite(alpha_rad)
        return cl, cd, cm

    return polar


def test_default_annular_solve_keeps_a_loss_factor_of_one() -> None:
    case = aerodynamic_load_case(_outer(), _point(), _polar(0.0, 0.04, 0.02))
    assert case.loss_model == LOSS_NONE
    assert case.loss_factor.tolist() == pytest.approx([1.0, 1.0])
    assert case.pitch_moment_nm_per_m[0] != pytest.approx(0.0)


def test_prandtl_factor_matches_the_closed_form() -> None:
    radius, phi, blades, hub, tip = 10.0, 0.3, 3, 2.0, 20.0
    sine = abs(math.sin(phi))
    f_tip = 2.0 / math.pi * math.acos(
        math.exp(-blades / 2.0 * (tip - radius) / (radius * sine)))
    f_hub = 2.0 / math.pi * math.acos(
        math.exp(-blades / 2.0 * (radius - hub) / (hub * sine)))
    assert prandtl_factor(radius, phi, blades, hub, tip) == pytest.approx(f_tip * f_hub)
    assert prandtl_factor(radius, 0.0, blades, hub, tip) == pytest.approx(1.0)
    assert prandtl_factor(tip, phi, blades, hub, tip, hub=False) == pytest.approx(0.0)
    assert prandtl_factor(hub, phi, blades, hub, tip, tip=False) == pytest.approx(0.0)


def test_buhl_meets_momentum_at_the_switch_and_rises_above_it() -> None:
    assert axial_induction(2.0 / 3.0, 1.0, buhl=True) == pytest.approx(0.4)
    assert axial_induction(2.0 / 3.0 + 1e-9, 1.0, buhl=True) == pytest.approx(0.4, abs=1e-6)
    below = axial_induction(0.5, 1.0, buhl=True)
    assert below == pytest.approx(0.5 / 1.5)
    high = axial_induction(2.0, 1.0, buhl=True)
    classic = axial_induction(2.0, 1.0, buhl=False)
    assert high != pytest.approx(classic)
    assert high > 0.4
    ks = [0.7, 1.0, 2.0, 5.0]
    values = [axial_induction(k, 1.0, buhl=True) for k in ks]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))


def test_prandtl_reduces_the_tip_loss_factor_and_the_induction() -> None:
    polar = _polar(0.0, 0.04, 0.02)
    none = aerodynamic_load_case(_outer(), _point(), polar)
    prandtl = aerodynamic_load_case(
        _outer(), _point(), polar, loss_model=LOSS_PRANDTL)
    assert prandtl.loss_model == LOSS_PRANDTL
    assert prandtl.loss_factor[1] < prandtl.loss_factor[0] < 1.0
    assert abs(prandtl.drag_n_per_m[1]) < abs(none.drag_n_per_m[1])
    assert prandtl.pitch_moment_nm_per_m[0] != pytest.approx(0.0)
    assert prandtl.pitch_moment_nm_per_m[0] == pytest.approx(
        0.5 * 1.25 * (
            (8.0 * (1.0 - prandtl.axial_induction[0])) ** 2
            + (1.2 * 10.0 * (1.0 + prandtl.tangential_induction[0])) ** 2
        ) * 1.5 * 1.5 * 0.02,
        rel=1e-9,
    )


def test_buhl_changes_a_high_induction_station() -> None:
    """A heavily loaded inboard station sits above k = 2/3; Buhl changes a."""

    outer = OuterShape(
        z_m=(4.0,),
        chord_m=(3.0,),
        twist_deg=(0.0,),
        aerofoil=("plate",),
        hub_radius_m=1.0,
        tip_radius_m=20.0,
    )
    point = OperatingPoint(
        inflow=Inflow(speed_m_s=2.5, density_kg_m3=1.25),
        rotor_speed_rad_s=0.8,
        pitch_deg=0.0,
        n_blades=3,
    )
    polar = _polar(1.2, 0.02, -0.05)
    prandtl = aerodynamic_load_case(outer, point, polar, loss_model=LOSS_PRANDTL)
    buhl = aerodynamic_load_case(outer, point, polar, loss_model=LOSS_PRANDTL_BUHL)
    assert buhl.loss_model == LOSS_PRANDTL_BUHL
    assert buhl.loss_factor[0] == pytest.approx(prandtl.loss_factor[0])
    assert prandtl.axial_induction[0] > 0.4
    assert buhl.axial_induction[0] != pytest.approx(prandtl.axial_induction[0])
    assert buhl.pitch_moment_nm_per_m[0] != pytest.approx(0.0)


def test_pitching_moment_is_taken_from_the_polar() -> None:
    case = aerodynamic_load_case(
        _outer(), _point(), _polar(0.0, 0.04, 0.15), loss_model=LOSS_PRANDTL)
    assert case.pitch_moment_nm_per_m[0] != pytest.approx(0.0)
    ratio = case.pitch_moment_nm_per_m[0] / case.drag_n_per_m[0]
    assert ratio == pytest.approx(1.5 * 0.15 / 0.04, rel=1e-12)


def test_prandtl_without_hub_and_tip_is_refused() -> None:
    outer = OuterShape(
        z_m=(10.0,), chord_m=(1.5,), twist_deg=(0.0,), aerofoil=("plate",))
    with pytest.raises(ValueError, match="hub_radius_m and tip_radius_m"):
        aerodynamic_load_case(outer, _point(), _polar(0.0, 0.04, 0.0), loss_model=LOSS_PRANDTL)


def test_unknown_loss_model_is_refused() -> None:
    with pytest.raises(ValueError, match="loss_model"):
        aerodynamic_load_case(_outer(), _point(), _polar(0.0, 0.04, 0.0), loss_model="glauert")
    assert LOSS_MODELS == (LOSS_NONE, LOSS_PRANDTL, LOSS_PRANDTL_BUHL)


def test_hub_and_tip_must_cover_the_stations() -> None:
    with pytest.raises(ValueError, match="between the hub radius and the tip radius"):
        OuterShape(
            z_m=(10.0, 18.0),
            chord_m=(1.5, 0.8),
            twist_deg=(0.0, 0.0),
            aerofoil=("plate", "plate"),
            hub_radius_m=12.0,
            tip_radius_m=20.0,
        )
