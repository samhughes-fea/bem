"""Blade-element momentum reports one stage on the shared logger."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from bem import Inflow, OperatingPoint, OuterShape, aerodynamic_load_case
from bem.stage_log import LOGGER_NAME, StageCollector, StageReport, collecting, fold_reports, stage


def test_public_contract() -> None:
    assert LOGGER_NAME == "blade.stage"
    assert [item.name for item in dataclasses.fields(StageReport)] == [
        "package", "stage", "seconds", "error", "depth", "started",
        "cpu_seconds", "measures"]
    assert callable(stage) and callable(fold_reports)
    assert issubclass(StageCollector, logging.Handler)


def test_a_stage_records_seconds_and_an_error_keeps_its_time() -> None:
    with collecting() as sink:
        with stage("bem", "kept"):
            pass
        with pytest.raises(RuntimeError):
            with stage("bem", "failed"):
                raise RuntimeError("stage failed")
    folded = fold_reports(sink.take())
    assert folded["kept"] >= 0.0
    failed = folded["failed"]
    assert isinstance(failed, dict)
    assert failed["seconds"] >= 0.0
    assert failed["error"] == "stage failed"


def test_aerodynamic_load_case_reports_bem() -> None:
    outer = OuterShape(
        z_m=(10.0,), chord_m=(1.5,), twist_deg=(0.0,), aerofoil=("plate",))
    point = OperatingPoint(
        inflow=Inflow(speed_m_s=8.0, density_kg_m3=1.25),
        rotor_speed_rad_s=1.2, pitch_deg=0.0, n_blades=3)

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        del aerofoil, alpha_rad
        return 0.0, 0.04, 0.02

    with collecting() as sink:
        aerodynamic_load_case(outer, point, polar)
    reports = sink.take()
    stations = [item for item in reports if item.stage == "station"]
    assert len(stations) == 1
    assert stations[0].depth == 1
    assert stations[0].error is None
    assert stations[0].measures["iterations"] >= 1
    parent = [item for item in reports if item.stage == "bem"]
    assert parent[0].depth == 0
    assert parent[0].measures["stations"] == 1
    folded = fold_reports(reports)
    assert set(folded) == {"bem"}
    assert folded["bem"] >= 0.0


def test_a_station_that_does_not_converge_keeps_its_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import bem.load as load

    monkeypatch.setattr(load, "_MAX_ITERATIONS", 1)
    outer = OuterShape(
        z_m=(10.0,), chord_m=(1.5,), twist_deg=(0.0,), aerofoil=("plate",))
    point = OperatingPoint(
        inflow=Inflow(speed_m_s=8.0, density_kg_m3=1.25),
        rotor_speed_rad_s=1.2, pitch_deg=0.0, n_blades=3)

    def polar(aerofoil: str, alpha_rad: float) -> tuple[float, float, float]:
        del aerofoil, alpha_rad
        return 1.2, 0.2, 0.0

    with collecting() as sink:
        with pytest.raises(RuntimeError, match="did not converge"):
            aerodynamic_load_case(outer, point, polar)
    reports = sink.take()
    station = next(item for item in reports if item.stage == "station")
    assert station.error is not None
    assert "did not converge" in station.error
    assert station.measures["iterations"] == 1
