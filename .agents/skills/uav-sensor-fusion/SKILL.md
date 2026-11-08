---
name: uav-sensor-fusion
description: >-
  Operational workflows, testing protocols, benchmark executions, and architectural guidelines
  for the UAV Asynchronous Multi-Sensor EKF Target Tracking and Tactical Radar Console.
  Use when developing, testing, benchmarking, or debugging filters, sensors, or GUI widgets in this repository.
---

# UAV Multi-Sensor EKF Fusion & Tactical Radar Engineering Workflows

This workspace skill provides the exact commands, architectural rules, and validation steps needed to work on the UAV Asynchronous EKF target tracker and PyQt6 tactical radar GUI console.

## 1. Quick Verification Runbook

Always verify the codebase after any algorithmic, sensor, or GUI modification:

```bash
# Run all test suites (EKF analytical Jacobians, OOSM rollback, and GUI offscreen dispatch)
python -m unittest discover tests

# Run analytical benchmark and regenerate results plots
python src/main.py

# Launch Tactical Radar GUI console
python -m src.gui.app
```

## 2. Core Architectural Invariants

- **No Blackbox Filter Libraries:** Do NOT use `filterpy` or `pykalman`. All equations (CWNA, analytic Jacobians, Joseph form, OOSM buffer replay) must be written from scratch using pure NumPy and SciPy.
- **Coordinate System:** 3D North-East-Down (NED). State vector is $\mathbf{x} = [x, \dot{x}, y, \dot{y}, z, \dot{z}]^T \in \mathbb{R}^{6 \times 1}$. Down is positive; Altitude is $-z$.
- **Precision:** Use `np.float64` for all matrix and vector representations.
- **Joseph Form Covariance Update:**
  $$P_k = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$
  Guarantees symmetry and positive semi-definiteness.
- **OOSM Rollback / Buffer-Replay:**
  When a delayed measurement arrives ($t_{capture} < t_{filter}$), the EKF rewinds its state to the snapshot immediately preceding $t_{capture}$, inserts the measurement, and replays all subsequent predictions and updates up to the current wall-clock epoch.

## 3. Playbooks & Reference Links

- Complete System Architecture & Rules: [AGENTS.md](../../../AGENTS.md)
- Step-by-Step Engineering Workflows: [SKILLS.md](../../../SKILLS.md)
- Test Suite: [test_ekf.py](../../../tests/test_ekf.py) and [test_gui.py](../../../tests/test_gui.py)
- EKF Engine: [ekf.py](../../../src/ekf.py)
- Synthetic Sensors: [sensors.py](../../../src/sensors.py)
- Event Queue Simulator: [simulator.py](../../../src/simulator.py)
- Tactical Radar GUI: [app.py](../../../src/gui/app.py)
