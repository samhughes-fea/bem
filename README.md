# bem

Steady blade-element momentum for one blade.

## Contract

The input is the outer shape along the span — chord, twist and aerofoil — together with the inflow, the rotor speed and the pitch.

The output is an aerodynamic load case: distributed lift, drag and pitch moment in the blade frame, with the operating point recorded on the case.

This is blade-element momentum only. There is no mesh, no structural stiffness and no rotor assembly. Shell meshes, the anisotropic section model and 1d-fem are not inputs and are not imported. Chord, twist and aerofoil are passed in as spanwise tables, so implicit-geometry-kernel is not a dependency.

## Blade frame

`z` is the distance from the rotor axis, in metres. At zero pitch and zero twist, `x` runs along the chord from leading edge to trailing edge and `y` points to the suction side.

Axial inflow lies along `+y` in that frame: the freestream meets the pressure side. Positive rotor speed moves the section nose-first, so the relative wind from rotation lies along `+x`.

Positive twist and positive pitch rotate the nose toward the axial inflow. Both are subtracted from the inflow angle. Pitch also rotates the blade frame about the span. Twist changes the angle of attack and does not rotate the frame the forces are reported in.

Lift and drag are the section forces perpendicular to and along the relative wind. The load case reports those magnitudes and their resultant, `force_x_n_per_m` and `force_y_n_per_m`, in the blade frame. Pitch moment is per unit span, positive nose-up, and is `0.5 ρ W² c² Cm` from the polar.

The balance is the steady annular momentum solution, with no tip loss, hub loss or turbulent-wake correction. `operating_point` stores the inflow, rotor speed, pitch and blade count. Inflow angle, angle of attack and induction are stored at each station.

## Public API

```python
from bem import Inflow, OperatingPoint, OuterShape, aerodynamic_load_case

outer = OuterShape(
    z_m=(10.0, 20.0),
    chord_m=(1.5, 1.5),
    twist_deg=(0.0, 0.0),
    aerofoil=("plate", "plate"),
)
point = OperatingPoint(
    inflow=Inflow(speed_m_s=8.0, density_kg_m3=1.25),
    rotor_speed_rad_s=1.2,
    pitch_deg=0.0,
    n_blades=3,
)

def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
    return 0.0, 0.04, 0.02  # Cl, Cd, Cm; Cm is positive nose-up

case = aerodynamic_load_case(outer, point, polar)
```

A polar may use `alpha_rad`. `Cm` is about the polar reference axis.

## Licence

MIT, the same licence as [samhughes-fea/1d-fem](https://github.com/samhughes-fea/1d-fem).
