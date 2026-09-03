import json
import pytest

from src.monitoring.subsidence_monitor import MeshReading, MeshSimulator, SubsidenceMonitor


def reading(elevation, timestamp=0.0, **kwargs):
    return MeshReading("N-01-01", timestamp, 0.0, 0.0, elevation, **kwargs)


def test_baseline_and_downward_displacement():
    monitor = SubsidenceMonitor()
    monitor.ingest([reading(100.0)])
    result = monitor.ingest([reading(99.985, timestamp=10.0)])[0]
    assert result.displacement_mm == pytest.approx(15.0)
    assert result.velocity_mm_per_s == pytest.approx(1.5)
    assert result.risk_level in {"WATCH", "HIGH", "CRITICAL"}


def test_risk_escalates_for_deformation_signature():
    monitor = SubsidenceMonitor()
    monitor.ingest([reading(100.0)])
    result = monitor.ingest([reading(
        99.96, timestamp=1.0, tilt_deg=1.8, vibration_rms=1.2, crack_mm=4.0,
    )])[0]
    assert result.risk_level == "CRITICAL"
    assert monitor.summary()["critical_nodes"] == 1
    assert monitor.alerts


def test_mesh_stream_and_geojson_export(tmp_path):
    monitor = SubsidenceMonitor()
    simulator = MeshSimulator()
    for frame in range(10):
        monitor.ingest(simulator.readings_for_frame(frame))
    summary = monitor.summary()
    assert summary["nodes"] == 25
    assert summary["online_nodes"] == 25
    assert summary["max_displacement_mm"] > 0
    output = tmp_path / "risk.geojson"
    monitor.export_geojson(output)
    payload = json.loads(output.read_text())
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 25
    assert "risk_level" in payload["features"][0]["properties"]
