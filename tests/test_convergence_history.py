"""Loss factor and induction iteration history on the load case."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bem import (
    LOSS_NONE,
    LOSS_PRANDTL,
    Inflow,
    OperatingPoint,
    OuterShape,
    aerodynamic_load_case,
)


def _outer() -> OuterShape:
    return OuterShape(
        z_m=(10.0, 18.0),
        chord_m=(1.5, 0.8),
        twist_deg=(0.0, 0.0),
        aerofoil=("plate", "plate"),
        hub_radius_m=2.0,
        tip_radius_m=20.0,
    )


def _point() -> OperatingPoint:
    return OperatingPoint(
        inflow=Inflow(speed_m_s=8.0, density_kg_m3=1.25),
        rotor_speed_rad_s=1.2,
        pitch_deg=0.0,
        n_blades=3,
    )


def _polar(cl: float = 0.0, cd: float = 0.04, cm: float = 0.02):
    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        assert aerofoil == "plate"
        assert math.isfinite(alpha_rad)
        return cl, cd, cm

    return polar


def test_default_solve_returns_unit_loss_and_a_history_per_station() -> None:
    case = aerodynamic_load_case(_outer(), _point(), _polar())

    assert case.loss_model == LOSS_NONE
    np.testing.assert_allclose(case.loss_factor, [1.0, 1.0], rtol=0.0, atol=0.0)
    assert len(case.convergence_history) == 2
    for index, history in enumerate(case.convergence_history):
        assert history.ndim == 2
        assert history.shape[1] == 3
        assert history.shape[0] >= 1
        np.testing.assert_allclose(history[-1, 0], case.axial_induction[index], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            history[-1, 1], case.tangential_induction[index], rtol=1e-12, atol=0.0
        )
        assert history[-1, 2] < 1e-12
        assert np.all(np.diff(history[:, 2]) <= 1e-15)


def test_prandtl_history_last_row_matches_the_solved_induction() -> None:
    case = aerodynamic_load_case(_outer(), _point(), _polar(), loss_model=LOSS_PRANDTL)

    assert case.loss_model == LOSS_PRANDTL
    assert case.loss_factor[1] < case.loss_factor[0] < 1.0
    assert len(case.convergence_history) == case.z_m.shape[0]
    for index, history in enumerate(case.convergence_history):
        np.testing.assert_allclose(history[-1, 0], case.axial_induction[index], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            history[-1, 1], case.tangential_induction[index], rtol=1e-12, atol=0.0
        )
        assert history[-1, 2] < 1e-12
