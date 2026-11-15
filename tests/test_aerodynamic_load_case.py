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


def test_rm1_radius_2_35_m_reaches_the_annular_balance():
    """The station that stopped cases 2.1@U_in and 4.1@U_in.

    Radius 2.35 m is the RM1 hub radius, 1 m, plus the AeroDyn station at
    1.35 m (chord 1.610 m, twist 12.86 deg). Both cases are rated rotor speed
    at U_in: 0.5 m/s, 11.5 rpm, pitch 0, two blades, density 1025 kg/m3, and
    the constant polar the reference run uses (Cl 0.1, Cd 0.02, Cm 0). After
    80 half-steps the residual is still 1.04e-12, one step short of 1e-12.
    The fixed point is an ordinary annular balance, not a cancelled wind.
    """

    radius = 2.35
    omega = 11.5 * math.pi / 30.0
    outer = OuterShape(
        z_m=(radius,),
        chord_m=(1.610,),
        twist_deg=(12.86,),
        aerofoil=("af5",),
    )
    operating_point = OperatingPoint(
        inflow=Inflow(speed_m_s=0.5, density_kg_m3=1025.0),
        rotor_speed_rad_s=omega,
        pitch_deg=0.0,
        n_blades=2,
    )

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        assert aerofoil == "af5"
        assert math.isfinite(alpha_rad)
        return 0.1, 0.02, 0.0

    case = aerodynamic_load_case(outer, operating_point, polar)

    assert case.axial_induction[0] == pytest.approx(0.235580190404, abs=1e-12)
    assert case.tangential_induction[0] == pytest.approx(-0.002617489547, abs=1e-12)
    assert case.inflow_angle_rad[0] == pytest.approx(0.134589876552, abs=1e-12)
    assert case.lift_n_per_m[0] == pytest.approx(669.457, rel=1e-4)
    assert case.drag_n_per_m[0] > 0.0


def test_a_cancelled_relative_wind_is_not_accepted():
    """The same operating point further out settles on induction (1, -1).

    That fixed point has no relative wind. It is the vanished-velocity error,
    not a converged load and not a dropped station.
    """

    omega = 11.5 * math.pi / 30.0
    outer = OuterShape(
        z_m=(10.0,),
        chord_m=(0.626,),
        twist_deg=(2.18,),
        aerofoil=("af9",),
    )
    operating_point = OperatingPoint(
        inflow=Inflow(speed_m_s=0.5, density_kg_m3=1025.0),
        rotor_speed_rad_s=omega,
        pitch_deg=0.0,
        n_blades=2,
    )

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        del aerofoil, alpha_rad
        return 0.1, 0.02, 0.0

    with pytest.raises(RuntimeError, match="relative velocity vanished at z = 10.0") as caught:
        aerodynamic_load_case(outer, operating_point, polar)
    error = caught.value
    assert error.axial_induction == pytest.approx(1.0, abs=1e-12)
    assert error.tangential_induction == pytest.approx(-1.0, abs=1e-12)
    assert error.radius == 10.0


def test_cut_in_at_3_55_m_keeps_the_induction_that_cancelled_the_wind():
    """Cases 2.1@U_in and 4.1@U_in stop here, and the error keeps the cause.

    Radius 3.55 m is the RM1 hub radius, 1 m, plus the AeroDyn station at
    2.55 m (chord 1.577 m, twist 9.50 deg). The operating point is the same
    cut-in used at 2.35 m. From this radius outward the iteration settles on
    induction (1, -1), which cancels the relative wind. The error still
    names the station, and it carries that axial induction, that tangential
    induction and the radius. It is not replaced by a load.
    """

    radius = 3.55
    omega = 11.5 * math.pi / 30.0
    outer = OuterShape(
        z_m=(radius,),
        chord_m=(1.577,),
        twist_deg=(9.50,),
        aerofoil=("af9",),
    )
    operating_point = OperatingPoint(
        inflow=Inflow(speed_m_s=0.5, density_kg_m3=1025.0),
        rotor_speed_rad_s=omega,
        pitch_deg=0.0,
        n_blades=2,
    )

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        del aerofoil, alpha_rad
        return 0.1, 0.02, 0.0

    with pytest.raises(RuntimeError, match="relative velocity vanished at z = 3.55") as caught:
        aerodynamic_load_case(outer, operating_point, polar)
    error = caught.value
    assert error.axial_induction == pytest.approx(1.0, abs=1e-12)
    assert error.tangential_induction == pytest.approx(-1.0, abs=1e-12)
    assert error.radius == radius
    assert type(error) is RuntimeError
