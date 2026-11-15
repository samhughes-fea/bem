"""Blade-element momentum loads in the blade frame.

The momentum balance is the steady annular solution: no tip loss, no hub
loss and no turbulent-wake correction. One blade is solved. ``n_blades``
enters only through the rotor solidity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bem.stage_log import stage

SectionPolar = Callable[[str, float], tuple[float, float, float]]

# Half-step iteration contracts linearly. At the RM1 station z = 2.35 m the
# residual is still just above 1e-12 after 80 steps, and the fixed point is
# an ordinary annular balance. 400 steps let that contraction finish. A
# residual that never settles still raises the same non-convergence error.
_MAX_ITERATIONS = 400
_TOLERANCE = 1e-12
_VANISHED_SPEED = 1e-14


@dataclass(frozen=True)
class Inflow:
    """Axial freestream.

    ``speed_m_s`` is the axial speed. In the zero-pitch blade frame it points
    along ``+y``, from the pressure side toward the suction side.
    """

    speed_m_s: float
    density_kg_m3: float = 1.225

    def __post_init__(self) -> None:
        speed = _finite("inflow speed", self.speed_m_s)
        density = _finite("density", self.density_kg_m3)
        if speed < 0.0:
            raise ValueError("inflow speed must be >= 0")
        if density <= 0.0:
            raise ValueError("density must be positive")
        object.__setattr__(self, "speed_m_s", speed)
        object.__setattr__(self, "density_kg_m3", density)


@dataclass(frozen=True)
class OperatingPoint:
    """The rotor condition asked for, recorded on the load case."""

    inflow: Inflow
    rotor_speed_rad_s: float
    pitch_deg: float
    n_blades: int

    def __post_init__(self) -> None:
        if not isinstance(self.inflow, Inflow):
            raise TypeError("inflow must be an Inflow")
        speed = _finite("rotor speed", self.rotor_speed_rad_s)
        pitch = _finite("pitch", self.pitch_deg)
        if speed < 0.0:
            raise ValueError("rotor speed must be >= 0")
        if isinstance(self.n_blades, bool) or not isinstance(self.n_blades, int):
            raise TypeError("n_blades must be an int")
        if self.n_blades < 1:
            raise ValueError("n_blades must be >= 1")
        if speed == 0.0 and self.inflow.speed_m_s == 0.0:
            raise ValueError("inflow speed and rotor speed cannot both be zero")
        object.__setattr__(self, "rotor_speed_rad_s", speed)
        object.__setattr__(self, "pitch_deg", pitch)


@dataclass(frozen=True)
class OuterShape:
    """Chord, twist and aerofoil at radial stations.

    ``z_m`` is the distance from the rotor axis, in metres, strictly
    increasing. ``twist_deg`` is positive when it rotates the nose toward the
    axial inflow (it is subtracted from the inflow angle). ``aerofoil`` is the
    label passed to the section polar at that station.
    """

    z_m: Sequence[float]
    chord_m: Sequence[float]
    twist_deg: Sequence[float]
    aerofoil: Sequence[str]

    def __post_init__(self) -> None:
        z = tuple(_finite("z", value) for value in self.z_m)
        chord = tuple(_finite("chord", value) for value in self.chord_m)
        twist = tuple(_finite("twist", value) for value in self.twist_deg)
        aerofoil = tuple(self.aerofoil)
        n = len(z)
        if n < 1:
            raise ValueError("outer shape needs at least one station")
        if not (len(chord) == len(twist) == len(aerofoil) == n):
            raise ValueError("z, chord, twist and aerofoil must have the same length")
        if any(value <= 0.0 for value in z):
            raise ValueError("z must be the positive distance from the rotor axis")
        if n > 1 and any(b <= a for a, b in zip(z, z[1:])):
            raise ValueError("z must be strictly increasing")
        if any(value <= 0.0 for value in chord):
            raise ValueError("chord must be positive")
        if any(not isinstance(name, str) or not name.strip() for name in aerofoil):
            raise ValueError("each aerofoil must be a non-empty string")
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "chord_m", chord)
        object.__setattr__(self, "twist_deg", twist)
        object.__setattr__(self, "aerofoil", aerofoil)


@dataclass(frozen=True)
class AerodynamicLoadCase:
    """Distributed aerodynamic load at the outer-shape stations.

    Lift and drag are section forces per unit span, perpendicular to and along
    the relative wind. ``force_x_n_per_m`` and ``force_y_n_per_m`` are that
    resultant in the blade frame. Pitch moment is per unit span, positive
    nose-up (the same sense as positive pitch).

    Arrays have shape ``(n,)`` in station order. ``operating_point`` is the
    condition that was solved. Inflow angle, angle of attack and induction are
    the solved state at each station.
    """

    operating_point: OperatingPoint
    z_m: np.ndarray
    lift_n_per_m: np.ndarray
    drag_n_per_m: np.ndarray
    pitch_moment_nm_per_m: np.ndarray
    force_x_n_per_m: np.ndarray
    force_y_n_per_m: np.ndarray
    inflow_angle_rad: np.ndarray
    angle_of_attack_rad: np.ndarray
    axial_induction: np.ndarray
    tangential_induction: np.ndarray


def aerodynamic_load_case(
    outer_shape: OuterShape,
    operating_point: OperatingPoint,
    polar: SectionPolar,
) -> AerodynamicLoadCase:
    """Solve steady blade-element momentum at each outer-shape station.

    Parameters
    ----------
    outer_shape
        Chord, twist and aerofoil along the span. ``z`` is the radius.
    operating_point
        Axial inflow, rotor speed, collective pitch and blade count.
    polar
        ``(aerofoil, alpha_rad) -> (Cl, Cd, Cm)``. ``Cm`` is positive nose-up.
        The pitch moment is ``0.5 ρ W² c² Cm`` about the polar reference axis.

    Returns
    -------
    AerodynamicLoadCase
        Lift, drag and pitch moment in the blade frame, with the operating
        point and the solved induction.

    Raises
    ------
    RuntimeError
        ``relative velocity vanished at z = ...`` when the induction at a
        station cancels the relative wind. The error carries that station's
        ``axial_induction``, ``tangential_induction`` and ``radius``. The
        station is not returned as a load and is not dropped.
    """

    with stage("bem", "bem") as rec:
        if isinstance(outer_shape, OuterShape):
            rec.set("stations", len(outer_shape.z_m))
        return _aerodynamic_load_case(outer_shape, operating_point, polar)


def _aerodynamic_load_case(
    outer_shape: OuterShape,
    operating_point: OperatingPoint,
    polar: SectionPolar,
) -> AerodynamicLoadCase:
    if not isinstance(outer_shape, OuterShape):
        raise TypeError("outer_shape must be an OuterShape")
    if not isinstance(operating_point, OperatingPoint):
        raise TypeError("operating_point must be an OperatingPoint")
    if not callable(polar):
        raise TypeError("polar must be callable")

    rows = []
    for z, chord, twist, name in zip(
        outer_shape.z_m,
        outer_shape.chord_m,
        outer_shape.twist_deg,
        outer_shape.aerofoil,
        strict=True,
    ):
        with stage("bem", "station") as rec:
            try:
                row, iterations = _solve_station(
                    radius=z,
                    chord=chord,
                    twist_deg=twist,
                    aerofoil=name,
                    operating_point=operating_point,
                    polar=polar,
                )
            except RuntimeError as exc:
                taken = getattr(exc, "iterations", None)
                if isinstance(taken, int) and not isinstance(taken, bool):
                    rec.set("iterations", taken)
                raise
            rec.set("iterations", iterations)
        rows.append(row)
    columns = zip(*rows, strict=True)
    arrays = [np.asarray(column, dtype=np.float64) for column in columns]
    return AerodynamicLoadCase(operating_point, *arrays)


def _solve_station(
    *,
    radius: float,
    chord: float,
    twist_deg: float,
    aerofoil: str,
    operating_point: OperatingPoint,
    polar: SectionPolar,
) -> tuple[tuple[float, ...], int]:
    """Return one station, starting with ``radius``, and the iteration count."""

    inflow = operating_point.inflow
    a = 0.0
    a_prime = 0.0
    solved: _Station | None = None
    iterations = 0
    for iterations in range(1, _MAX_ITERATIONS + 1):
        try:
            step = _momentum_step(
                radius=radius,
                chord=chord,
                twist_deg=twist_deg,
                aerofoil=aerofoil,
                speed=inflow.speed_m_s,
                density=inflow.density_kg_m3,
                omega=operating_point.rotor_speed_rad_s,
                pitch_deg=operating_point.pitch_deg,
                n_blades=operating_point.n_blades,
                a=a,
                a_prime=a_prime,
                polar=polar,
            )
        except RuntimeError as exc:
            exc.iterations = iterations  # type: ignore[attr-defined]
            raise
        if max(abs(step.a_target - a), abs(step.a_prime_target - a_prime)) < _TOLERANCE:
            # (1, -1) is a fixed point with no relative wind. The step that
            # proposes it is evaluated just off that point, so the guard
            # inside the step does not see the wind the targets would return.
            axial = inflow.speed_m_s * (1.0 - step.a_target)
            tangential = operating_point.rotor_speed_rad_s * radius * (1.0 + step.a_prime_target)
            if math.hypot(axial, tangential) < _VANISHED_SPEED:
                failed = _vanished_wind(
                    radius, step.a_target, step.a_prime_target
                )
                failed.iterations = iterations  # type: ignore[attr-defined]
                raise failed
            solved = step
            break
        a = a + 0.5 * (step.a_target - a)
        a_prime = a_prime + 0.5 * (step.a_prime_target - a_prime)
    if solved is None:
        failed = RuntimeError(f"blade-element momentum did not converge at z = {radius}")
        failed.iterations = iterations  # type: ignore[attr-defined]
        raise failed
    row = (
        radius,
        solved.lift,
        solved.drag,
        solved.pitch_moment,
        solved.force_x,
        solved.force_y,
        solved.inflow_angle,
        solved.angle_of_attack,
        solved.a_target,
        solved.a_prime_target,
    )
    return row, iterations


@dataclass(frozen=True)
class _Station:
    lift: float
    drag: float
    pitch_moment: float
    force_x: float
    force_y: float
    inflow_angle: float
    angle_of_attack: float
    a_target: float
    a_prime_target: float


def _vanished_wind(
    radius: float, axial_induction: float, tangential_induction: float
) -> RuntimeError:
    """The induction at ``radius`` cancelled the relative wind.

    The message stays ``relative velocity vanished at z = ...``. Axial and
    tangential induction, and the radius, stay on the error so the failed
    station is still identifiable. Callers do not receive a load.
    """

    failed = RuntimeError(f"relative velocity vanished at z = {radius}")
    failed.radius = radius  # type: ignore[attr-defined]
    failed.axial_induction = axial_induction  # type: ignore[attr-defined]
    failed.tangential_induction = tangential_induction  # type: ignore[attr-defined]
    return failed


def _momentum_step(
    *,
    radius: float,
    chord: float,
    twist_deg: float,
    aerofoil: str,
    speed: float,
    density: float,
    omega: float,
    pitch_deg: float,
    n_blades: int,
    a: float,
    a_prime: float,
    polar: SectionPolar,
) -> _Station:
    axial = speed * (1.0 - a)
    tangential = omega * radius * (1.0 + a_prime)
    if math.hypot(axial, tangential) < _VANISHED_SPEED:
        raise _vanished_wind(radius, a, a_prime)
    inflow_angle = math.atan2(axial, tangential)
    angle_of_attack = inflow_angle - math.radians(twist_deg + pitch_deg)
    cl, cd, cm = _coefficients(polar, aerofoil, angle_of_attack)

    sin_phi = math.sin(inflow_angle)
    cos_phi = math.cos(inflow_angle)
    solidity = n_blades * chord / (2.0 * math.pi * radius)
    normal = cl * cos_phi + cd * sin_phi
    tangent = cl * sin_phi - cd * cos_phi
    axial_denominator = 4.0 * sin_phi * sin_phi + solidity * normal
    swirl_denominator = 4.0 * sin_phi * cos_phi - solidity * tangent
    if abs(axial_denominator) < 1e-14 or abs(swirl_denominator) < 1e-14:
        raise RuntimeError(f"momentum balance is singular at z = {radius}")
    a_target = solidity * normal / axial_denominator
    a_prime_target = solidity * tangent / swirl_denominator
    if not math.isfinite(a_target) or not math.isfinite(a_prime_target):
        raise RuntimeError(f"momentum balance is singular at z = {radius}")

    dynamic_pressure = 0.5 * density * (axial * axial + tangential * tangential)
    lift = dynamic_pressure * chord * cl
    drag = dynamic_pressure * chord * cd
    force_x_rotor = drag * cos_phi - lift * sin_phi
    force_y_rotor = drag * sin_phi + lift * cos_phi
    # Positive pitch rotates blade +x (leading edge to trailing edge) toward +y.
    cosine = math.cos(math.radians(pitch_deg))
    sine = math.sin(math.radians(pitch_deg))
    force_x = cosine * force_x_rotor + sine * force_y_rotor
    force_y = -sine * force_x_rotor + cosine * force_y_rotor
    return _Station(
        lift=lift,
        drag=drag,
        pitch_moment=dynamic_pressure * chord * chord * cm,
        force_x=force_x,
        force_y=force_y,
        inflow_angle=inflow_angle,
        angle_of_attack=angle_of_attack,
        a_target=a_target,
        a_prime_target=a_prime_target,
    )


def _coefficients(polar: SectionPolar, aerofoil: str, alpha: float) -> tuple[float, float, float]:
    coeffs = polar(aerofoil, alpha)
    try:
        cl, cd, cm = coeffs
    except (TypeError, ValueError) as exc:
        raise ValueError("polar must return Cl, Cd, Cm") from exc
    return _finite("Cl", cl), _finite("Cd", cd), _finite("Cm", cm)


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
