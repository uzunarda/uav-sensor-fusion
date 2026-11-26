# 6-DOF Flight Physics & Three.js 3D Tactical Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a literature-standard 6-DOF RK4 flight & maneuvering target physics engine coupled with a Three.js 3D Airspace & Onboard EO/IR Gimbal Camera simulator launched directly from the PyQt6 Tactical Command Center.

**Architecture:** Pure NumPy 64-bit RK4 kinematic/aerodynamic trajectory generators (`CV`, `CT`, `EVASIVE`) and 3-2-1 Euler DCM coordinate transformations in `src/dynamics.py` feed both the PyQt6 `SimulationWorker` and a background thread-safe HTTP/JSON bridge (`src/sim3d/server.py`). A Three.js WebGL application (`src/sim3d/index.html`) renders the 3D tactical airspace, 5-mode EKF tracks, 3D covariance ellipsoid, and a Picture-in-Picture onboard EO/IR gimbal camera view.

**Tech Stack:** Python 3.9+, NumPy, SciPy, PyQt6, Three.js (WebGL), standard library `http.server` / `threading`.

**Spec:** `docs/superpowers/specs/2026-11-26-threejs-flight-simulator-design.md`

## Global Constraints
- Strictly NO external Kalman libraries (`filterpy`, `pykalman`, etc.) and NO `PyQt6-WebEngine`.
- All state vectors must maintain `shape=(6, 1)` or 6-element `dtype=np.float64` in 3D NED frame (Altitude = `-z`).
- All GUI tests must remain 100% compatible with `os.environ["QT_QPA_PLATFORM"] = "offscreen"`.
- No emojis in tactical GUI headers; preserve monospace `Consolas` dark slate theme.

## Review Focus
- RK4 integration with `t == t0` or `dt == 0` must return exact initial state without NaN or division by zero.
- 3-2-1 Euler DCM transformation must be orthogonal ($R^T R = I$, $\det(R) = +1$).
- Switching scenarios (`CV`, `CT`, `EVASIVE`) in `SimulationWorker` must recompute all 5 modes (`MODE_A`..`MODE_E`) cleanly.
- `Sim3DBridgeServer` must bind to an ephemeral localhost port without blocking or leaking threads and serve valid JSON & CSV exports.

---

### Task 1: 6-DOF RK4 Flight & Maneuvering Target Dynamics (`src/dynamics.py`)

**Files:**
- Modify: `src/dynamics.py`
- Test: `tests/test_ekf.py`

**Interfaces:**
- Produces:
  - `euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray`
  - `rk4_step(deriv_fn: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, dt: float) -> np.ndarray`
  - `CoordinatedTurnTrajectory(initial_state: TargetState, turn_rate_rad_s: float = 0.08, t0: float = 0.0)`
  - `EvasiveWeaveTrajectory(initial_state: TargetState, weave_amp_y: float = 45.0, weave_freq_hz: float = 0.12, climb_amp_z: float = 12.0, t0: float = 0.0)`
  - `UAVPlatformState` and `UAVFlightTrajectory(initial_pos: np.ndarray, cruise_speed: float = 28.0, orbit_radius: float = 450.0, dynamic_flight: bool = False)`
  - `create_trajectory(scenario: str, initial_state: TargetState, t0: float = 0.0)`

- [ ] **Step 1: Add unit tests for RK4, Euler DCM orthogonality, Coordinated Turn, and Evasive Weave trajectories in `tests/test_ekf.py`**
- [ ] **Step 2: Implement 6-DOF DCM, RK4 solver, `CoordinatedTurnTrajectory`, `EvasiveWeaveTrajectory`, `UAVFlightTrajectory`, and `create_trajectory` in `src/dynamics.py`**
- [ ] **Step 3: Run `python -m unittest tests/test_ekf.py -v` and verify all tests pass**

---

### Task 2: Three.js 3D Tactical Airspace & EO/IR Gimbal Camera Simulator (`src/sim3d/`)

**Files:**
- Create: `src/sim3d/__init__.py`
- Create: `src/sim3d/server.py`
- Create: `src/sim3d/index.html`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `create_trajectory`, `UAVFlightTrajectory` from `src/dynamics.py`, `ExtendedKalmanFilter` from `src/ekf.py`, `evaluate_Comprehensive_metrics` from `src/metrics.py`
- Produces:
  - `Sim3DBridgeServer(worker: Optional[SimulationWorker] = None)` with `.start() -> str`, `.stop() -> None`, `.build_simulation_payload(scenario: str, duration: float) -> dict`, `.build_csv_export(scenario: str, mode: str) -> str`

- [ ] **Step 1: Implement `src/sim3d/server.py` providing JSON simulation payload, live worker sync, and CSV dataset generator**
- [ ] **Step 2: Implement `src/sim3d/index.html` with Three.js 3D NED Airspace, 3D UAV & Target meshes, 3D FOV frustum, 5-mode EKF trajectories, 3D covariance ellipsoid, and Picture-in-Picture Onboard EO/IR Gimbal Camera HUD**

---

### Task 3: PyQt6 GUI Scenario Selector & One-Click 3D Simulator Launcher

**Files:**
- Modify: `src/gui/worker.py`
- Modify: `src/gui/widgets/control_bar.py`
- Modify: `src/gui/views/config_view.py`
- Modify: `src/gui/app.py`
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: `Sim3DBridgeServer` from `src/sim3d/server.py`, `create_trajectory` and `UAVFlightTrajectory` from `src/dynamics.py`
- Produces:
  - `SimulationWorker.set_scenario(scenario: str)` (`"CV"`, `"CT"`, `"EVASIVE"`)
  - `TacticalRadarApp.launch_3d_simulator(open_browser: bool = True) -> str`

- [ ] **Step 1: Update `src/gui/worker.py` to support `scenario_type` (`"CV"`, `"CT"`, `"EVASIVE"`) and 6-DOF UAV orientation telemetry in `SimulationFrame`**
- [ ] **Step 2: Add scenario selector dropdown (`SABİT HIZ [CV]`, `KOORDİNELİ DÖNÜŞ [CT]`, `KAÇINMA MANEVRASI [S-WEAVE]`) to `src/gui/widgets/control_bar.py` and `src/gui/views/config_view.py`**
- [ ] **Step 3: Add `[05] 3B UÇUŞ & EO/IR SİMÜLATÖRÜ [THREE.JS]` button and `Sim3DBridgeServer` integration in `src/gui/app.py`**
- [ ] **Step 4: Add headless verification tests in `tests/test_gui.py` for scenario switching, 3D server startup, JSON payload, and CSV export**
- [ ] **Step 5: Run full test suite `python -m unittest discover tests -v` and commit on `feat/flight-simulation-engine`**
