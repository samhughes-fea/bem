"""Prandtl tip and hub loss and the Buhl high-induction correction.

The default annular solve does not use these. They are a labelled variant of
the momentum balance: Prandtl multiplies the annulus by ``F = F_tip F_hub``,
and Buhl replaces the axial induction when Ning's loading parameter
``k > 2/3``.
"""

from __future__ import annotations

import math

LOSS_NONE = "none"
LOSS_PRANDTL = "prandtl"
LOSS_PRANDTL_BUHL = "prandtl_buhl"
LOSS_MODELS = (LOSS_NONE, LOSS_PRANDTL, LOSS_PRANDTL_BUHL)

# Ning 2014: classic momentum for k <= 2/3 (a <= 0.4). Above that, Buhl.
_BUHL_SWITCH = 2.0 / 3.0
_BUHL_G3_LIMIT = 1e-6


def prandtl_factor(
    radius: float,
    inflow_angle: float,
    n_blades: int,
    hub_radius: float,
    tip_radius: float,
    *,
    tip: bool = True,
    hub: bool = True,
) -> float:
    """Prandtl ``F = F_tip F_hub``.

    ``sin φ → 0`` sends the exponent to infinity, so ``F → 1`` (the exact
    limit). At the tip or hub the corresponding factor is zero.
    """

    sine = abs(math.sin(inflow_angle))
    factor = 1.0
    if tip:
        factor *= _prandtl_term(n_blades * (tip_radius - radius) / (2.0 * radius * sine)
                                if sine > 0.0 else math.inf)
    if hub:
        factor *= _prandtl_term(n_blades * (radius - hub_radius) / (2.0 * hub_radius * sine)
                                if sine > 0.0 else math.inf)
    return factor


def _prandtl_term(exponent: float) -> float:
    """``(2/π) arccos(exp(-f))`` with ``exp`` clipped to ``[0, 1]``."""

    if exponent >= 0.0:
        decay = 0.0 if exponent == math.inf else math.exp(-exponent)
    else:
        decay = 1.0
    return 2.0 / math.pi * math.acos(min(1.0, max(0.0, decay)))


def axial_induction(k: float, loss_factor: float, *, buhl: bool) -> float:
    """Axial induction from Ning's ``k = σ Cn / (4 F sin² φ)``.

    Classic momentum is ``a = k / (1 + k)``. Buhl's empirical branch (Ning
    2014) replaces it when ``k > 2/3``: ``g1 = 2 F k − (10/9 − F)``,
    ``g2 = 2 F k − F (4/3 − F)``, ``g3 = 2 F k − (25/9 − 2 F)``,
    ``a = (g1 − √g2) / g3``, or ``1 − 1 / (2 √g2)`` as ``g3 → 0``.
    """

    if not buhl or k <= _BUHL_SWITCH:
        return k / (1.0 + k)
    return _buhl_axial(k, loss_factor)


def _buhl_axial(k: float, loss_factor: float) -> float:
    two_fk = 2.0 * loss_factor * k
    g1 = two_fk - (10.0 / 9.0 - loss_factor)
    g2 = two_fk - loss_factor * (4.0 / 3.0 - loss_factor)
    g3 = two_fk - (25.0 / 9.0 - 2.0 * loss_factor)
    if abs(g3) < _BUHL_G3_LIMIT:
        return 1.0 - 1.0 / (2.0 * math.sqrt(g2))
    return (g1 - math.sqrt(g2)) / g3
