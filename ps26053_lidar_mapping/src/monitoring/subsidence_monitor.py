"""Sensor-fusion primitives for a wireless surface subsidence prototype."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MeshReading:
    node_id: str
    timestamp_s: float
    x_m: float
    y_m: float
    elevation_m: float
    tilt_deg: float = 0.0
    vibration_rms: float = 0.0
    crack_mm: float = 0.0
    battery_pct: float = 100.0
    rssi_dbm: float = -60.0
    sequence: int = 0

    @property
    def link_quality(self) -> float:
        """Approximate 0..1 radio quality from RSSI and battery health."""
        rssi_quality = max(0.0, min(1.0, (self.rssi_dbm + 110.0) / 60.0))
        return 0.7 * rssi_quality + 0.3 * max(0.0, min(1.0, self.battery_pct / 100.0))


@dataclass(frozen=True)
class NodeAssessment:
    node_id: str
    x_m: float
    y_m: float
    displacement_mm: float
    velocity_mm_per_s: float
    forecast_mm_60s: float
    tilt_deg: float
    vibration_rms: float
    crack_mm: float
    quality: float
    risk_score: float
    risk_level: str


@dataclass(frozen=True)
class Alert:
    node_id: str
    timestamp_s: float
    risk_level: str
    risk_score: float
    message: str


class SubsidenceMonitor:
    """Baseline-relative deformation monitor with explainable risk scoring."""

    def __init__(self, displacement_warn_mm: float = 8.0, displacement_critical_mm: float = 20.0):
        self.displacement_warn_mm = displacement_warn_mm
        self.displacement_critical_mm = displacement_critical_mm
        self.baseline: dict[str, float] = {}
        self.previous: dict[str, tuple[float, float]] = {}
        self.latest: dict[str, NodeAssessment] = {}
        self.alerts: list[Alert] = []

    def ingest(self, readings: Iterable[MeshReading]) -> list[NodeAssessment]:
        assessments = []
        for reading in readings:
            if reading.node_id not in self.baseline:
                self.baseline[reading.node_id] = reading.elevation_m
            displacement_mm = (self.baseline[reading.node_id] - reading.elevation_m) * 1000.0
            previous = self.previous.get(reading.node_id)
            velocity = 0.0
            if previous is not None and reading.timestamp_s > previous[0]:
                velocity = (displacement_mm - previous[1]) / (reading.timestamp_s - previous[0])
            self.previous[reading.node_id] = (reading.timestamp_s, displacement_mm)
            forecast = displacement_mm + velocity * 60.0
            risk_score = self._risk_score(displacement_mm, velocity, reading)
            level = self._risk_level(risk_score)
            assessment = NodeAssessment(
                node_id=reading.node_id, x_m=reading.x_m, y_m=reading.y_m,
                displacement_mm=displacement_mm, velocity_mm_per_s=velocity,
                forecast_mm_60s=forecast, tilt_deg=reading.tilt_deg,
                vibration_rms=reading.vibration_rms, crack_mm=reading.crack_mm,
                quality=reading.link_quality, risk_score=risk_score, risk_level=level,
            )
            self.latest[reading.node_id] = assessment
            if level in {"HIGH", "CRITICAL"}:
                self.alerts.append(Alert(
                    reading.node_id, reading.timestamp_s, level, risk_score,
                    f"{level} subsidence signature at {reading.node_id}: "
                    f"{displacement_mm:.1f} mm cumulative movement",
                ))
            assessments.append(assessment)
        return assessments

    def summary(self) -> dict[str, float | int | str]:
        values = list(self.latest.values())
        high = sum(a.risk_level in {"HIGH", "CRITICAL"} for a in values)
        critical = sum(a.risk_level == "CRITICAL" for a in values)
        return {
            "nodes": len(values),
            "online_nodes": sum(a.quality >= 0.45 for a in values),
            "high_risk_nodes": high,
            "critical_nodes": critical,
            "max_displacement_mm": max((a.displacement_mm for a in values), default=0.0),
            "max_risk_score": max((a.risk_score for a in values), default=0.0),
            "overall_status": "CRITICAL" if critical else "HIGH" if high else "NORMAL",
        }

    def to_geojson(self) -> dict:
        features = []
        for assessment in self.latest.values():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [assessment.x_m, assessment.y_m]},
                "properties": {
                    "node_id": assessment.node_id,
                    "displacement_mm": round(assessment.displacement_mm, 3),
                    "velocity_mm_per_s": round(assessment.velocity_mm_per_s, 4),
                    "forecast_mm_60s": round(assessment.forecast_mm_60s, 3),
                    "risk_score": round(assessment.risk_score, 4),
                    "risk_level": assessment.risk_level,
                    "quality": round(assessment.quality, 3),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def export_geojson(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_geojson(), indent=2), encoding="utf-8")

    def _risk_score(self, displacement_mm: float, velocity: float, reading: MeshReading) -> float:
        displacement = min(1.0, max(0.0, displacement_mm / self.displacement_critical_mm))
        velocity_component = min(1.0, abs(velocity) / 0.5)
        tilt_component = min(1.0, abs(reading.tilt_deg) / 2.0)
        crack_component = min(1.0, max(0.0, reading.crack_mm) / 5.0)
        vibration_component = min(1.0, max(0.0, reading.vibration_rms) / 1.5)
        quality_penalty = 0.85 + 0.15 * reading.link_quality
        return min(1.0, quality_penalty * (
            0.40 * displacement + 0.20 * velocity_component +
            0.15 * tilt_component + 0.15 * crack_component + 0.10 * vibration_component
        ))

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75:
            return "CRITICAL"
        if score >= 0.45:
            return "HIGH"
        if score >= 0.20:
            return "WATCH"
        return "NORMAL"


class MeshSimulator:
    """Deterministic 25-node mesh stream with a progressive subsidence bowl."""

    def __init__(self, rows: int = 5, cols: int = 5, spacing_m: float = 20.0):
        self.nodes = [
            (f"N-{row:02d}-{col:02d}", (col - (cols - 1) / 2) * spacing_m,
             (row - (rows - 1) / 2) * spacing_m)
            for row in range(rows) for col in range(cols)
        ]

    def readings_for_frame(self, frame: int, interval_s: float = 1.0) -> list[MeshReading]:
        timestamp = frame * interval_s
        readings = []
        for index, (node_id, x_m, y_m) in enumerate(self.nodes):
            distance = math.hypot(x_m, y_m)
            bowl = max(0.0, 1.0 - distance / 70.0)
            movement_m = 0.004 * max(0, frame - 2) * bowl ** 2
            readings.append(MeshReading(
                node_id=node_id, timestamp_s=timestamp, x_m=x_m, y_m=y_m,
                elevation_m=-movement_m, tilt_deg=0.12 * max(0, frame - 2) * bowl,
                vibration_rms=0.08 + 0.15 * bowl * max(0, frame - 2),
                crack_mm=max(0.0, frame - 5) * 0.5 * bowl,
                battery_pct=98.0 - frame * 0.08, rssi_dbm=-54.0 - (index % 4) * 7,
                sequence=frame,
            ))
        return readings
