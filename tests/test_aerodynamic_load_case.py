"""Hand check for a uniform-chord, zero-twist blade."""

import math

import pytest

from bem import Inflow, OperatingPoint, OuterShape, aerodynamic_load_case


def test_uniform_chord_zero_twist_matches_hand_calculation():
    """Constant Cl = 0, Cd and Cm on a uniform unswept blade.

    With zero lift, zero twist and zero pitch the annular balance closes:

        φ = atan2(V, Ω r)
        σ = B c / (2 π r)
        a = σ Cd / (4 sin φ + σ Cd)
        a' = -a
        W = (1 - a) sqrt(V² + (Ω r)²)

    because both relative-velocity components scale by (1 - a), so φ is the
    zero-induction inflow angle at every radius. Lift is zero. Drag and the
    nose-up pitch moment are the usual section resultants, and the blade-frame
    force is drag along the relative wind.
    """

    chord = 1.5
    radii = (10.0, 20.0)
    density = 1.25
    speed = 8.0
    omega = 1.2
    cd = 0.04
    cm = 0.02
    n_blades = 3
    outer = OuterShape(
        z_m=radii,
        chord_m=(chord, chord),
        twist_deg=(0.0, 0.0),
        aerofoil=("plate", "plate"),
    )
    operating_point = OperatingPoint(
        inflow=Inflow(speed_m_s=speed, density_kg_m3=density),
        rotor_speed_rad_s=omega,
        pitch_deg=0.0,
        n_blades=n_blades,
    )

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        assert aerofoil == "plate"
        assert math.isfinite(alpha_rad)
        return 0.0, cd, cm

    case = aerodynamic_load_case(outer, operating_point, polar)

    assert case.operating_point == operating_point
    assert case.z_m.tolist() == [10.0, 20.0]
    assert case.drag_n_per_m[0] != pytest.approx(case.drag_n_per_m[1])

    for index, radius in enumerate(radii):
        solidity = n_blades * chord / (2.0 * math.pi * radius)
        tangential = omega * radius
        wind = math.hypot(tangential, speed)
        sin_phi = speed / wind
        cos_phi = tangential / wind
        induction = solidity * cd / (4.0 * sin_phi + solidity * cd)
        relative = (1.0 - induction) * wind
        dynamic_pressure = 0.5 * density * relative * relative
        drag = dynamic_pressure * chord * cd
        moment = dynamic_pressure * chord * chord * cm

        assert case.lift_n_per_m[index] == pytest.approx(0.0, abs=1e-12)
        assert case.drag_n_per_m[index] == pytest.approx(drag, rel=1e-9)
        assert case.pitch_moment_nm_per_m[index] == pytest.approx(moment, rel=1e-9)
        assert case.force_x_n_per_m[index] == pytest.approx(drag * cos_phi, rel=1e-9)
        assert case.force_y_n_per_m[index] == pytest.approx(drag * sin_phi, rel=1e-9)
        assert case.axial_induction[index] == pytest.approx(induction, rel=1e-9)
        assert case.tangential_induction[index] == pytest.approx(-induction, rel=1e-9)
        assert case.angle_of_attack_rad[index] == pytest.approx(case.inflow_angle_rad[index])
        assert case.inflow_angle_rad[index] == pytest.approx(math.atan2(speed, tangential))
