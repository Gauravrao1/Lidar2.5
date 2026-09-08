# SIH Technical Document

## Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

**Problem context:** DRDO PS-26053  
**Repository analyzed:** `ps26053_lidar_mapping`  
**Document purpose:** Technical presentation, feasibility justification, production transition plan, and honest prototype assessment for Smart India Hackathon (SIH).

---

## 1. Executive Summary

The proposed system converts raw LiDAR scans into a semantic, memory-efficient 2.5D map for autonomous navigation and monitoring in dynamic environments.

The central innovation is **adaptive spatial resolution**:

- Use fine cells near the sensor, where collision avoidance needs high precision.
- Increase cell size with range, where angular and sensor uncertainty already reduce useful detail.
- Preserve terrain, static-obstacle, and dynamic-object semantics in each occupied cell.
- Remove stale dynamic cells so that moving objects do not become permanent “ghost” obstacles.

The current repository is a functional research demonstrator. It proves the grid mathematics, ingestion format, baseline PointNet inference path, metrics generation, visualization, temporal fusion, elevation analysis, object tracking, and export utilities. It is not yet a field-deployable autonomy product. Production deployment requires a real sensor driver, calibrated timestamps and coordinate frames, a validated segmentation model, deterministic inference, ROS2 packaging, monitoring, security, and hardware-in-the-loop testing.

The correct SIH claim is therefore:

> “We have implemented and validated the core adaptive mapping approach on reproducible sample data, and we have a defined engineering path to deploy it on real LiDAR and ROS2 hardware.”

Do not claim that the current sample metrics represent real-world accuracy or production safety.

---

## 2. Problem and Motivation

Conventional mapping choices have a difficult trade-off:

1. A uniformly fine grid gives good near-field accuracy but consumes excessive memory.
2. A uniformly coarse grid saves memory but loses the detail required for nearby obstacles, terrain edges, and narrow passages.
3. A static map fails in dynamic scenes because previously observed vehicles or pedestrians remain visible after they move.
4. A dense 3D voxel map is often unnecessary for navigation tasks that mainly require horizontal occupancy plus compact elevation information.

This project addresses the trade-off using a polar, variable-resolution, 2.5D representation. The map is organized by radial ring, angular sector, and bounded vertical bin. Each occupied cell stores semantic state and compact statistics rather than allocating every possible location in a dense Cartesian volume.

### Target use cases

- Autonomous ground-vehicle navigation in warehouses, roads, mines, and defense environments.
- Near-field collision avoidance and traversability estimation.
- Dynamic-obstacle awareness for vehicles and robots.
- Resource-constrained edge deployments where RAM, CPU, and power are limited.
- Replay and analysis of recorded LiDAR missions.
- Future fusion with subsidence or infrastructure monitoring sensors.

---

## 3. Technical Structure of the Current Repository

```text
ps26053_lidar_mapping/
|-- configs/
|   |-- grid_config.yaml          Grid geometry, resolution, z limits, decay
|   `-- class_remap.yaml          SemanticKITTI labels -> 4 project classes
|-- data/
|   |-- sample/                   Synthetic .bin/.label demonstration data
|   `-- download_semantickitti.sh Dataset acquisition helper
|-- checkpoints/
|   `-- pointnet_3class.pth       PointNet checkpoint artifact
|-- src/
|   |-- ingestion/kitti_loader.py Binary point-cloud and label loader
|   |-- segmentation/
|   |   |-- model_interface.py    Backend abstraction
|   |   |-- pointnet_infer.py     Pure PyTorch PointNet inference
|   |   `-- randlanet_infer.py    Documented alternative backend
|   |-- grid/
|   |   |-- resolution.py         Cell-size function and key generation
|   |   |-- adaptive_grid.py      Sparse cell store and dynamic decay
|   |   |-- temporal_fusion.py    Bayesian-style confidence accumulation
|   |   `-- elevation_map.py      Slope, roughness, traversability
|   |-- tracking/object_tracker.py Dynamic point clustering and tracks
|   |-- metrics/                  IoU, latency, memory, report generation
|   |-- export/                   PLY and CSV output
|   |-- pipeline/standalone_runner.py Sequential end-to-end runner
|   |-- pipeline/ros2_nodes/      ROS2 architecture placeholder/source
|   |-- monitoring/subsidence_monitor.py Sensor-fusion risk prototype
|   `-- viz/standalone_dashboard.py Plotly Dash replay dashboard
|-- tests/                        Pytest unit and smoke tests
|-- reports/                      Sample metrics, logs, and write-up
`-- requirements.txt              Python dependencies
```

### Layered architecture

```text
LiDAR driver or recorded scan
            |
            v
[Ingestion and validation]
            |
            v
[Semantic segmentation backend]
            |
            +------> [Point labels and confidence]
            v
[Adaptive polar 2.5D grid]
            |
            +------> [Temporal confidence fusion]
            +------> [Dynamic object tracking]
            +------> [Elevation and traversability]
            +------> [Memory, latency, IoU metrics]
            +------> [ROS2 topics / dashboard / PLY / CSV]
```

### Current module responsibilities

| Layer | Current implementation | Technical responsibility |
|---|---|---|
| Ingestion | `kitti_loader.py` | Reads `(x, y, z, intensity)` float32 scans and optional labels; validates remapping. |
| Semantics | `SegmentationBackend`, `PointNetBackend` | Stable interface for interchangeable models; outputs one class per point. |
| Spatial map | `resolution.py`, `AdaptiveGrid` | Converts points to `(ring, sector, z_bin)` and stores occupied cells sparsely. |
| Temporal state | `TemporalFusionGrid` | Accumulates class evidence and confidence over frames. |
| Terrain | `ElevationMap` | Estimates elevation, slope, roughness, and traversability. |
| Dynamics | `DynamicObjectTracker` | Clusters dynamic points, associates detections, estimates velocity. |
| Evaluation | `metrics/*` | Computes IoU, distance-bucketed IoU, latency, FPS, and baseline estimates. |
| Visualization | Plotly Dash, RViz marker source | Shows map state, tracks, and performance indicators. |
| Operations prototype | `SubsidenceMonitor` | Demonstrates baseline-relative sensor risk scoring and GeoJSON export. |

---

## 4. End-to-End Technical Approach

### 4.1 Input representation

Each LiDAR point is represented as:

```text
[x, y, z, intensity] : float32[4]
```

SemanticKITTI labels contain semantic and instance information in a 32-bit value. The loader extracts the lower 16-bit semantic ID and maps it to the project classes:

| Project class | Meaning | Map behavior |
|---|---|---|
| `0` | Terrain | Candidate drivable or walkable surface |
| `1` | Static | Persistent obstacle or environmental structure |
| `2` | Dynamic | Moving or potentially moving object |
| `3` | Ignore | Unlabeled, outlier, or sensor artifact; excluded from IoU |

The complete remap is configuration-driven in `configs/class_remap.yaml`. Unknown labels fail loudly rather than silently entering the map.

### 4.2 Segmentation

The implemented baseline is a vanilla PointNet segmentation model:

```text
Conv1d 4 -> 64 -> 128 -> 512
Global max pooling: 512 features
Concatenate local 64 features + global 512 features = 576
Conv1d 576 -> 256 -> 128 -> 4 classes
Argmax per point
```

PointNet is suitable for the prototype because it is pure PyTorch, has no custom CUDA extension, and can run in a constrained Windows/CPU environment. The backend subsamples large scans to 16,384 points and maps predictions back using a nearest-neighbor KD-tree.

**Production requirement:** replace the sample-trained or randomly initialized model with a validated outdoor LiDAR model, preferably a sparse convolution or RandLA-Net-style backend exported to ONNX/TensorRT where available. The model must be trained and evaluated on representative Indian roads, terrain, lighting/weather conditions, sensor mounting positions, and dynamic-object classes.

### 4.3 Variable-resolution grid

For radial distance $r$, the current cell size is:

$$
s(r)=
\begin{cases}
S_{near}, & r \le R_{near} \\
S_{near}\left(\frac{S_{far}}{S_{near}}\right)^{\frac{\ln(r/R_{near})}{\ln(R_{far}/R_{near})}}, & R_{near}<r<R_{far} \\
S_{far}, & r \ge R_{far}
\end{cases}
$$

Current defaults:

- $R_{near}=10$ m
- $R_{far}=100$ m
- $S_{near}=0.05$ m
- $S_{far}=0.50$ m
- 360 angular sectors, approximately 1 degree per sector
- vertical range from -3 m to +3 m with 0.25 m bins

The mapping key is:

```text
(floor(r / s(r)), floor((theta + pi) / sector_angle), z_bin)
```

The cell store is sparse: only cells containing observations are allocated. Points outside the configured vertical range are clamped to the nearest valid bin, and points beyond the far radius use the configured far-resolution policy.

### 4.4 Cell state and semantic aggregation

For each occupied cell, the current implementation stores:

- Semantic class label.
- Point count.
- Running center `(x, y, z)`.
- Average cell size.
- Last update frame.

Within a frame, labels are aggregated by majority vote. Across frames, existing cells update their center and point count and dynamic cells can be deleted after `DECAY_MAX_AGE` frames without refresh.

### 4.5 Temporal confidence fusion

`TemporalFusionGrid` maintains class pseudo-counts per cell. Old evidence is multiplied by a decay rate and new observations receive a configurable weight. The resulting normalized distribution provides:

- Predicted class.
- Confidence as maximum class probability.
- Entropy as an uncertainty indicator.
- Confident and uncertain cell queries.

This is useful for suppressing one-frame classification noise. In a production system, confidence must be calibrated against validation data and should influence planning conservatively; confidence is not a safety guarantee.

### 4.6 Dynamic-object tracking

The tracker currently:

1. Filters points classified as dynamic.
2. Quantizes them into voxels.
3. Finds connected components using 26-neighbor connectivity.
4. Computes cluster centroid and bounding box.
5. Performs greedy nearest-centroid association.
6. Estimates velocity from centroid displacement and an assumed frame rate.
7. Removes tracks after a configurable missed-frame limit.

For production, use sensor timestamps rather than assumed FPS, a gated Hungarian or probabilistic association method, track confidence, class-aware dimensions, and explicit handling of occlusion and split/merge events.

### 4.7 Elevation and traversability

For a cell with at least three points, the elevation module fits:

$$z=ax+by+c$$

It derives slope from $\sqrt{a^2+b^2}$ and roughness from plane-fit residuals. Traversability is the product of slope and roughness factors, bounded between zero and one.

This is a useful navigation feature, but it needs calibration for vehicle type, wheelbase, clearance, tire model, soil, wet surfaces, and mission rules before it can drive a planner.

### 4.8 Outputs

- Sparse grid cells for navigation or analysis.
- Semantic class and confidence layers.
- Dynamic-object tracks and trajectories.
- Elevation/traversability statistics.
- CSV and PLY exports.
- Plotly Dash replay dashboard.
- Intended ROS2 label and marker topics.
- IoU, latency, memory, and distance-bucketed evaluation reports.

---

## 5. Configuration and Reproducibility

The main tunable parameters are externalized in YAML:

| Parameter group | Current values | Why it matters |
|---|---|---|
| Radial bounds | 10 m / 100 m | Defines near and far resolution zones. |
| Cell sizes | 5 cm / 50 cm | Accuracy versus memory trade-off. |
| Angular sectors | 360 | Angular detail and cell count. |
| Vertical bins | 24 bins over 6 m | Compact 2.5D height representation. |
| Dynamic decay | 10 frames | Ghost-object removal versus persistence. |
| Memory model | 48 bytes/cell | Approximate comparison, not allocator RSS. |

The intended reproducibility command is:

```powershell
cd "d:\Lidar 2.5\ps26053_lidar_mapping"
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\generate_sample_data.py
python -m src.pipeline.standalone_runner --data-dir data\sample --max-frames 10
python -m src.viz.standalone_dashboard --data-dir data\sample
```

For a real benchmark, record commit ID, dataset version, sensor model, CPU/GPU, driver versions, configuration hash, random seed, warm-up policy, and whether labels are predicted or ground truth.

---

## 6. Evidence: What Is Proven and What Is Not

### Implemented in the repository

- SemanticKITTI binary and label loading.
- Configuration-driven 28-class to 4-class remapping.
- Log-linear resolution function and edge-case handling.
- Sparse adaptive grid insertion and dynamic-cell decay.
- PointNet backend interface and CPU inference path.
- IoU and distance-bucketed IoU calculation.
- Latency and memory comparison report generation.
- Temporal fusion, elevation analysis, tracking, and file export utilities.
- Dash replay dashboard and ROS2 source skeleton.
- Pytest test files covering grid, loader, inference shape/range, pipeline smoke behavior, tracking, fusion, elevation, and exports.

### Evidence limitations that must be stated in the SIH presentation

1. The available sample data is synthetic: 10 scans of approximately 26,000 points each.
2. The PointNet model is not evidence of production segmentation accuracy. The backend can use random initialization when no checkpoint is supplied, which produces valid-shaped output but poor semantics.
3. The dashboard deliberately uses ground-truth labels during preprocessing for fast visualization. It must not be presented as a prediction-only evaluation.
4. The standalone runner is the correct path for measured inference latency; dashboard latency is not equivalent.
5. The memory comparison uses a 48-byte-per-cell estimate and theoretical dense baselines. It is not a full process-memory or allocator measurement.
6. Existing report files contain different sample latency and cell-count snapshots. Treat them as run artifacts, not one canonical benchmark.
7. ROS2 source is documented but not built or tested in the current Windows environment.
8. The project-specific test runner reported no discoverable tests, and the direct shell test command was blocked because `python` resolves to the Microsoft Store alias. Run the suite with the activated virtual-environment interpreter before final submission.

### Minimum evidence package before final SIH judging

- One real SemanticKITTI sequence or locally recorded LiDAR sequence.
- Fixed train/validation/test split with no frame leakage.
- Predicted-label-only dashboard mode.
- Confusion matrix and IoU for each class and distance bucket.
- p50/p95/p99 latency at the target frame rate.
- Peak RSS, GPU memory, and cell-memory measurements.
- Ablation: uniform grid versus adaptive grid; no decay versus decay; no fusion versus fusion.
- ROS2 replay demo with timestamps and dropped-frame counters.
- Repeatable setup script and a clean test report.

---

## 7. Feasibility Analysis

### 7.1 Technical feasibility

**Feasible with staged scope.** The core representation uses familiar numerical operations, sparse dictionaries, and configuration-driven geometry. The current prototype already demonstrates the core transformation from points to semantic cells. The largest technical risks are segmentation quality, real-time throughput, ROS2 integration, and sensor calibration rather than the resolution formula itself.

### 7.2 Compute feasibility

The current environment is an 8 GB RAM, CPU-only Windows laptop. The sample report records approximately 4 FPS in one run, which is useful for replay but below a typical 10 Hz autonomous-navigation target. A production architecture should:

- Use GPU inference where available.
- Export and benchmark the model with ONNX Runtime, TensorRT, or an edge accelerator.
- Avoid unnecessary point copies and Python per-point loops.
- Replace dictionary-heavy hot paths with vectorized arrays or compiled kernels.
- Process bounded queues with backpressure and explicit frame-drop policy.

The grid concept remains feasible on CPU because it stores occupied cells rather than a dense 3D volume. Actual capacity must be measured on the target vehicle computer.

### 7.3 Data feasibility

SemanticKITTI is suitable for initial benchmarking, but it is not sufficient for final deployment. The real dataset must contain:

- The target LiDAR model and mounting geometry.
- Local road, terrain, vegetation, infrastructure, and weather conditions.
- Rare but important dynamic classes.
- Ground-truth labels or an auditable labeling process.
- Sequences separated by scene and time to prevent leakage.

### 7.4 Integration feasibility

ROS2 is the appropriate integration boundary for sensor, perception, planning, visualization, and logging nodes. The current code shows intended topics, but a real integration still needs a package manifest, setup/build files, message synchronization, QoS profiles, TF transforms, diagnostics, launch files, and replay tests.

### 7.5 Operational feasibility

The dashboard and CSV/PLY reports make the prototype easy to demonstrate and audit. Field operation additionally requires service supervision, health checks, log rotation, time synchronization, configuration versioning, secure deployment, and recovery after sensor or model failure.

---

## 8. Viability and Deployment Model

### 8.1 Technical viability

The strongest product value is not merely “a smaller map.” It is a map that spends detail where the vehicle has the most immediate risk and avoids retaining stale dynamic observations. This can lower memory pressure and improve the planning input for edge systems.

### 8.2 Economic viability

The approach can reduce the need for high-memory compute hardware and can support lower-cost edge computers. This claim must be quantified using total system cost, power draw, thermal limits, sensor cost, and maintenance cost, not only estimated grid bytes.

### 8.3 Adoption viability

The backend interface allows the model to change without rewriting the grid engine. ROS2 integration can expose standard outputs to existing autonomy stacks. CSV, PLY, and dashboard outputs support engineering review and offline diagnosis.

### 8.4 Scale viability

The architecture can scale from replay to live operation if state ownership is made explicit. For multiple sensors or vehicles, use a timestamped message bus, frame IDs, map tiles or bounded local windows, and a persistent telemetry store. Do not share mutable global dashboard state across missions in a production service.

### 8.5 Success metrics

| Area | Pilot target | Measurement method |
|---|---:|---|
| End-to-end latency | At or below sensor period | p50/p95/p99 from timestamped pipeline stages |
| Semantic quality | Defined per class and range | Held-out real-data IoU and confusion matrix |
| Dynamic freshness | No stale obstacle beyond policy | Replay motion/occlusion scenarios |
| Memory | Bounded under mission limit | Peak RSS plus grid allocator statistics |
| Availability | Defined mission uptime | Fault-injection and soak testing |
| Navigation value | Fewer unsafe or blocked decisions | Planner-level scenario benchmark |

---

## 9. Expected Impact and Benefits

### Direct technical benefits

- Lower map memory than a uniformly fine 2D or dense 3D representation.
- Higher near-field spatial detail for obstacle avoidance.
- Less far-field over-allocation where fine resolution has lower value.
- Dynamic-cell decay reduces stale obstacle persistence.
- Semantic layers distinguish terrain, static obstacles, and dynamic objects.
- Temporal confidence exposes uncertainty instead of hiding it.
- Elevation statistics support traversability-aware planning.
- Backend abstraction enables model upgrades without replacing the map engine.

### Operational and societal impact

- More affordable autonomy for resource-constrained platforms.
- Better safety margins in crowded or changing environments.
- Reduced compute and energy demand at the edge.
- Faster inspection and replay during incident analysis.
- Reusable architecture for defense, logistics, mining, infrastructure, and disaster response.

### Impact caveat

The current repository does not prove reduced collision rate, energy savings, or field safety. Those benefits require scenario-based navigation tests and field trials with a safety driver or supervisory controller.

---

## 10. What to Keep and What Not to Present as Production

### Keep and strengthen

- The adaptive radial resolution formula and its configuration.
- Sparse occupied-cell representation.
- Explicit terrain/static/dynamic class contract.
- Dynamic-cell aging and ghost-object tests.
- Temporal confidence and entropy outputs.
- Elevation and traversability features.
- Segmentation backend abstraction.
- Distance-bucketed IoU, p95 latency, and memory reporting.
- Reproducible sample-data generator and replay dashboard.
- PLY/CSV exports for debugging and audit.

### Keep as prototype/demo-only until repaired

- Synthetic sample dataset.
- PointNet trained only on the sample data.
- Dashboard mode that substitutes ground truth labels.
- Theoretical 48-byte memory estimate as the only memory metric.
- ROS2 node source without package/build/runtime validation.
- Tracker’s assumed 10 FPS and greedy association.
- Subsidence simulator and risk score unless the SIH scope explicitly includes sensor fusion.

### Do not claim

- “Real-time autonomous navigation” based only on the current CPU sample run.
- “60% mIoU” unless reproduced with the stated model, checkpoint, dataset, split, and evaluation script.
- Field robustness from synthetic scans.
- Safety certification or collision avoidance guarantees.
- Production ROS2 readiness from source files alone.
- Exact memory savings as physical RAM savings without allocator-level measurement.

---

## 11. Production Architecture to Build Next

```text
[Velodyne/Ouster driver]
          |
          v
[Timestamp + calibration + TF validation]
          |
          v
[Bounded ingest queue / frame policy]
          |
          v
[GPU or optimized semantic inference]
          |
          v
[Vectorized adaptive grid service]
       /       |        \
      v        v         v
[Fusion] [Tracker] [Traversability]
       \       |        /
          v
[Planner-facing local map]
          |
          +--> ROS2 topics and diagnostics
          +--> Mission recorder and metrics
          `--> Operator dashboard
```

### Required production components

1. ROS2 package metadata, launch files, parameters, QoS, and message definitions.
2. Sensor timestamp synchronization and TF validation.
3. Prediction confidence, invalid-input handling, and model version metadata.
4. Deterministic sampling or a documented seeded sampling policy.
5. Vectorized or compiled grid insertion.
6. Proper track association with timestamp-derived velocity.
7. Persistent mission logs and replayable event records.
8. Health, diagnostics, watchdog, and graceful degradation behavior.
9. Authentication and network isolation for dashboard and telemetry interfaces.
10. Automated unit, integration, performance, fault-injection, and hardware-in-the-loop tests.

---

## 12. Risk Register and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Weak segmentation model | Incorrect obstacle or terrain semantics | Train on representative real data; report class/range metrics; use conservative unknown handling. |
| Point subsampling misses small objects | Unsafe near-field map | Range-aware sampling, small-object tests, and minimum point coverage checks. |
| Python hot loops | Missed frame deadlines | Vectorize, profile, and move critical kernels to compiled/GPU implementations. |
| Incorrect timestamps or TF | Misaligned tracks and maps | Hardware timestamps, TF buffer validation, synchronization diagnostics. |
| Dynamic decay too aggressive | Real object disappears | Tune using motion/occlusion replay; retain uncertainty and track state separately. |
| Dynamic decay too slow | Ghost obstacles | Measure stale-cell lifetime and expose it as a monitored metric. |
| Sensor dropout or corruption | Invalid map | Input validation, heartbeat, quarantine invalid frames, and degraded mode. |
| Memory estimate differs from RSS | Capacity planning failure | Track allocator/RSS/GPU memory in target deployment. |
| ROS2 queue mismatch | Dropped or stale data | Explicit QoS, queue-depth metrics, and synchronized integration tests. |
| Dataset leakage | Inflated accuracy | Split by sequence and scene, not random frames. |

---

## 13. SIH Demonstration Script

1. Show one raw LiDAR scan and explain the four project classes.
2. Show the resolution curve: 5 cm near the vehicle, smoothly increasing to 50 cm far away.
3. Run the standalone pipeline on a fixed sample or real replay sequence.
4. Display the sparse grid and compare it with uniform 2D and dense 3D baselines.
5. Move a dynamic-object sequence through frames and show decay of stale cells.
6. Show temporal confidence and uncertainty for stable versus flickering observations.
7. Show elevation slope and traversability for flat and steep surfaces.
8. Show latency p50/p95, peak memory, FPS, and per-stage timing.
9. State clearly which views use predictions and which are visualization-only.
10. Finish with the ROS2 production architecture and the validation roadmap.

### Suggested 60-second pitch

“Our system builds a semantic 2.5D LiDAR map that is precise near the vehicle and memory-efficient at long range. Instead of allocating a uniformly fine 3D volume, it uses a log-linear polar grid and stores only occupied cells. Terrain, static obstacles, and dynamic objects are handled differently; dynamic cells decay when they are no longer observed, preventing ghost obstacles. The prototype includes segmentation, temporal confidence, tracking, traversability, metrics, and replay visualization. We have validated the core method on reproducible sample data and have separated the remaining production work: real-data training, optimized inference, ROS2 integration, timestamped tracking, and hardware-in-the-loop safety validation.”

---

## 14. Implementation Roadmap

### Phase 1: Evidence hardening

- Repair the local Python environment and run every pytest test.
- Make all benchmark runs deterministic with a recorded seed.
- Consolidate report generation so one run produces one canonical metrics table.
- Add predicted-label-only dashboard mode.
- Add true RSS and peak allocation measurements.

### Phase 2: Real-data validation

- Download and validate a complete SemanticKITTI sequence.
- Train or obtain a documented pretrained segmentation model.
- Evaluate by class and distance bucket.
- Add small-object, occlusion, weather, and long-range test scenarios.

### Phase 3: Performance engineering

- Profile point loading, inference, KD-tree remapping, and grid insertion.
- Replace per-point Python loops in the grid and tracker hot paths.
- Benchmark CPU, CUDA, and edge-accelerator configurations.
- Define frame-drop and backpressure behavior.

### Phase 4: ROS2 and field integration

- Create a real ROS2 package with launch, parameters, QoS, TF, and diagnostics.
- Integrate a real LiDAR driver and timestamped replay.
- Connect the map to a planner or collision-checking consumer.
- Run soak, fault-injection, and hardware-in-the-loop tests.

### Phase 5: Pilot deployment

- Deploy under supervision on a representative platform.
- Collect mission logs and compare predictions with reviewed ground truth.
- Tune resolution, decay, confidence, and traversability thresholds.
- Define acceptance criteria for availability, safety, latency, and map quality.

---

## 15. Final Assessment

**Technical feasibility:** High for the adaptive grid and replay pipeline; medium for production perception because model quality and real-time performance are not yet demonstrated on representative hardware.

**Deployment viability:** Medium today, high after ROS2 packaging, optimized inference, timestamp/TF correctness, and field validation are completed.

**Prototype maturity:** Strong demonstrator with multiple implemented research features; not a certified or safety-critical autonomy component.

**Primary SIH strength:** A clear, explainable memory-versus-resolution design with dynamic-scene handling and a realistic path to edge deployment.

**Primary SIH weakness to address:** The current evidence is synthetic and partially demo-oriented. The team should present this limitation openly and use the judging time to show the validation plan and a prediction-only real-data benchmark rather than overstate current numbers.
