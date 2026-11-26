# 6-DOF Flight & Sensor Physics Engine + Three.js 3D Tactical Arena Design Spec

**Date:** 2026-11-26  
**Branch:** `feat/flight-simulation-engine`

## 1. Intent and Why
Build an open, mathematically verifiable alternative to MATLAB + FlightGear for UAV multi-sensor target tracking and OOSM compensation research. It serves two purposes simultaneously:
1. **Academic & Mathematical Rigor:** Standard 64-bit IEEE-754 flight kinematics (4th-Order Runge-Kutta RK4 solver, Coordinated Turn, Evasive S-Weave, and 6-DOF moving UAV with 3-2-1 Euler DCM transformations) + CSV dataset export and 5-mode OOSM benchmarking.
2. **Interactive 3D Visual Simulation (Three.js):** A FlightGear-style 60 FPS WebGL 3D Airspace & Synthetic Onboard EO/IR Gimbal Camera (Picture-in-Picture) simulator, launched with a single button click directly from the PyQt6 Tactical Command Center (`src/gui/app.py`).

## 2. Goals, Non-Goals, and Anti-Goals
- **Goals:**
  - Extend `src/dynamics.py` with RK4-integrated maneuvering target trajectories (`CONSTANT_VELOCITY`, `COORDINATED_TURN`, `EVASIVE_WEAVE`) and a moving 6-DOF UAV trajectory generator (`UAVPlatformState`, `UAVFlightTrajectory`, 3-2-1 Euler rotation matrix `euler_to_dcm`).
  - Support dynamic scenario selection in `SimulationWorker` (`src/gui/worker.py`), `PlaybackControlBar` (`src/gui/widgets/control_bar.py`), and `ConfigView` (`src/gui/views/config_view.py`).
  - Create `src/sim3d/` containing:
    - `server.py`: Thread-safe background HTTP/JSON simulation bridge & CSV dataset exporter that serves the live 5-mode EKF trajectories, 6-DOF UAV attitude, sensor FOV geometry, and literature metrics.
    - `index.html`: Self-contained Three.js 3D Tactical Airspace & Synthetic EO/IR Gimbal Camera (Picture-in-Picture) simulator with live scenario switching, mode comparison (Modes A–E), interactive camera views, and CSV dataset download.
  - Add a one-click launcher button (`[05] 3B UÇUŞ & EO/IR SİMÜLATÖRÜ [THREE.JS]`) in `TacticalRadarApp` (`src/gui/app.py`).
- **Non-Goals / Anti-Goals:**
  - No external Kalman libraries (`filterpy`, `pykalman`) and no heavy `PyQt6-WebEngine` dependency.
  - No breaking changes to existing 5-mode OOSM architecture or existing 12 unit/GUI tests.

## 3. Approaches Considered
1. **Approach 1 (Chosen): Shared Python 6-DOF RK4 Physics/EKF Core + Background HTTP Bridge + Three.js 3D Tactical Simulator launched via PyQt6 button.** Zero heavy dependencies, 60 FPS hardware WebGL rendering, full dual-camera (Main Orbit + UAV Onboard EO/IR Gimbal PIP) support, 100% headless test compatibility.
2. **Approach 2: Embedding Chromium via `PyQt6-WebEngine`.** Rejected due to heavy external dependency and OpenGL context conflicts during offscreen testing.
3. **Approach 3: Extending only `PyQtGraph` `GLViewWidget`.** Rejected because it cannot render a secondary UAV-mounted EO/IR Gimbal camera feed or rich 3D flight visuals.

## 4. Architecture & Component Design
- **`src/dynamics.py`:**
  - `euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray`: Standard aerospace 3-2-1 body-to-NED rotation matrix $R_b^n$.
  - `rk4_step(deriv_fn, t: float, state_vec: np.ndarray, dt: float) -> np.ndarray`: Classic 4th-order Runge-Kutta integrator.
  - `CoordinatedTurnTrajectory`: 3D helical / constant turn-rate ($\omega$) maneuvering target trajectory integrated with RK4.
  - `EvasiveWeaveTrajectory`: Sinusoidal S-turn + altitude weave maneuvering target trajectory integrated with RK4.
  - `UAVPlatformTrajectory`: Moving UAV ownship trajectory (supports stationary hover or 3D patrol/pursuit with bank angle $\phi$, pitch $\theta$, heading $\psi$).
  - `create_trajectory(scenario: str, initial_state: TargetState, t0: float = 0.0)`: Factory for `"CV"`, `"CT"`, `"EVASIVE"`.
- **`src/sim3d/server.py`:**
  - `Sim3DBridgeServer`: Background daemon `ThreadingHTTPServer` bound to `127.0.0.1` (auto-selecting a free port) serving:
    - `GET /`: The Three.js 3D Tactical Flight & EO/IR Gimbal Simulator (`index.html`).
    - `GET /api/simulation?scenario=...`: Full 5-mode (Modes A–E) 6-DOF trajectory frames, sensor packets, OOSM events, and 16 literature metrics as JSON.
    - `GET /api/export_csv?scenario=...&mode=...`: Academic benchmark dataset in CSV format (timestamps, ground truth, UAV 6-DOF pose, raw measurements, EKF estimates, NEES, OSPA).
- **`src/gui/app.py`, `src/gui/worker.py`, `src/gui/widgets/control_bar.py`, `src/gui/views/config_view.py`:**
  - `worker.scenario_type` (`"CV"`, `"CT"`, `"EVASIVE"`) and `worker.uav_motion_enabled` (`bool`).
  - Scenario selector combo box in `control_bar.py` & `config_view.py`.
  - `[05] 3B SİMÜLATÖR [THREE.JS]` launch button in `app.py` sidebar and top header strip.
