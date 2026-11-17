"""
Asynchronous Extended Kalman Filter (EKF) Engine for 3D Target Tracking.
Implements Continuous White Noise Acceleration (CWNA) dynamics, nonlinear Radar/Camera updates
with analytic Jacobians, Joseph-form covariance stabilization, and Multi-Method Out-Of-Sequence
Measurement (OOSM) Latency Compensation:
  - Mode A (IDEAL): Zero-latency baseline reference.
  - Mode B (NAIVE): Delay-ignorant EKF (processes delayed measurements at arrival time t_receive).
  - Mode C (ROLLBACK): Historical State Buffer Rollback & Chronological Replay (Retrodiction/Replay).
  - Mode D (RETRODICTION): 1-Step Modified EKF with Delay-Aware Observation Matrix H_eff = H_d * F(-dt).
  - Mode E (EXTRAPOLATION): Kalman Predictor (Forward Measurement Extrapolation to current epoch).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from .sensors import SensorMeasurement
from .dynamics import TargetState


@dataclass
class FilterSnapshot:
    """
    State and covariance snapshot recorded after filter propagation / update.
    Used by the circular buffer to rewind and replay out-of-sequence events.

    Attributes:
        t (float): Epoch in seconds.
        x (np.ndarray): 6x1 state vector [x, vx, y, vy, z, vz]^T.
        P (np.ndarray): 6x6 covariance matrix.
        measurement (Optional[SensorMeasurement]): Measurement applied at this snapshot.
        uav_pos (Optional[np.ndarray]): 3D position of UAV at this epoch.
    """
    t: float
    x: np.ndarray
    P: np.ndarray
    measurement: Optional[SensorMeasurement] = None
    uav_pos: Optional[np.ndarray] = None


class ExtendedKalmanFilter:
    """
    6-DOF Asynchronous Extended Kalman Filter in 3D NED frame with Multi-Strategy OOSM capability.

    State vector:
        x = [x, vx, y, vy, z, vz]^T

    Sub-blocks:
        - Axis X: states 0, 1
        - Axis Y: states 2, 3
        - Axis Z: states 4, 5

    Supported OOSM Compensation Strategies (`oosm_strategy`):
        - 'ROLLBACK'      : Full state-history rewind and chronological replay (Mode C - Paper 2).
        - 'RETRODICTION'  : 1-Step Modified Kalman with backward state transition F(-dt) (Mode D - Paper 3).
        - 'EXTRAPOLATION' : Kalman Predictor forward measurement extrapolation to t_now (Mode E - Paper 4).
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_spectral_density: float = 1.0,
        initial_timestamp: float = 0.0,
        enable_oosm_buffer: bool = True,
        max_buffer_size: int = 150,
        max_buffer_dt: float = 1.0,
        oosm_strategy: str = "ROLLBACK",
    ) -> None:
        """
        Args:
            initial_state: 6-element initial state estimate vector.
            initial_covariance: 6x6 initial error covariance matrix P_0.
            process_noise_spectral_density: Power spectral density (q_c) of continuous-time acceleration noise (m^2/s^3).
            initial_timestamp: Initial filter epoch in seconds.
            enable_oosm_buffer: Whether to enable OOSM delay compensation.
            max_buffer_size: Maximum number of historical snapshots preserved in memory.
            max_buffer_dt: Maximum age in seconds before an OOSM is dropped as stale.
            oosm_strategy: OOSM compensation method ('ROLLBACK', 'RETRODICTION', or 'EXTRAPOLATION').
        """
        self.x = np.asarray(initial_state, dtype=np.float64).reshape((6, 1))
        self.P = np.asarray(initial_covariance, dtype=np.float64)

        if self.P.shape != (6, 6):
            raise ValueError(f"Covariance matrix P must be 6x6, got {self.P.shape}")

        self.q_c = float(process_noise_spectral_density)
        self.current_time = float(initial_timestamp)
        self.I6 = np.eye(6, dtype=np.float64)

        # OOSM Buffer & Strategy configuration
        self.enable_oosm_buffer = bool(enable_oosm_buffer)
        self.max_buffer_size = int(max_buffer_size)
        self.max_buffer_dt = float(max_buffer_dt)
        self.oosm_strategy = str(oosm_strategy).upper()

        # Diagnostics & counters
        self.oosm_count: int = 0
        self.dropped_oosm_count: int = 0
        self.in_order_count: int = 0
        self.last_nis: float = 1.0
        self._max_capture_time: float = float(initial_timestamp)

        # Circular buffer initialized with initial state
        self.history: List[FilterSnapshot] = []
        if self.enable_oosm_buffer and self.oosm_strategy == "ROLLBACK":
            self._record_snapshot(self.current_time, None, None)

    # -------------------------------------------------------------------------
    # System Transition and Process Noise Matrices
    # -------------------------------------------------------------------------

    @staticmethod
    def state_transition_matrix(dt: float) -> np.ndarray:
        """
        Constructs the 6x6 discrete-time state transition matrix F(dt) for Constant Velocity:
            F_1D = [[1, dt], [0, 1]]
            F    = blkdiag(F_1D, F_1D, F_1D)
        Also supports negative dt (-delta) for backward retrodiction.
        """
        F = np.eye(6, dtype=np.float64)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        return F

    def process_noise_covariance(self, dt: float) -> np.ndarray:
        """
        Computes discrete process noise covariance Q(dt) from Continuous White Noise
        Acceleration (CWNA) model:
            Q_1D(dt) = q_c * [[dt^3 / 3, dt^2 / 2],
                              [dt^2 / 2, dt      ]]
            Q(dt)    = blkdiag(Q_1D, Q_1D, Q_1D)
        """
        abs_dt = abs(float(dt))
        dt2 = abs_dt * abs_dt
        dt3 = dt2 * abs_dt

        q11 = (dt3 / 3.0) * self.q_c
        q12 = (dt2 / 2.0) * self.q_c
        q22 = abs_dt * self.q_c

        Q_1d = np.array([[q11, q12], [q12, q22]], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:2, 0:2] = Q_1d
        Q[2:4, 2:4] = Q_1d
        Q[4:6, 4:6] = Q_1d
        return Q

    # -------------------------------------------------------------------------
    # Predict Step
    # -------------------------------------------------------------------------

    def predict(self, timestamp: float) -> None:
        """
        Performs dynamic time-update (prediction step) to the given timestamp.

        Args:
            timestamp: Target epoch for prediction.
        """
        dt = timestamp - self.current_time

        if dt < -1e-9:
            raise ValueError(
                f"Out-of-order event received in predict(): filter time {self.current_time:.6f}s, target time {timestamp:.6f}s"
            )

        if dt <= 1e-9:
            # Near-zero time step (simultaneous arrival), state and covariance do not propagate
            self.current_time = timestamp
            return

        F = self.state_transition_matrix(dt)
        Q = self.process_noise_covariance(dt)

        # Propagate state and covariance
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Enforce exact symmetry
        self.P = 0.5 * (self.P + self.P.T)
        self.current_time = timestamp

    # -------------------------------------------------------------------------
    # Measurement Functions and Jacobians
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_relative_cartesian(
        state: np.ndarray, uav_pos: Optional[np.ndarray]
    ) -> Tuple[float, float, float, float, float]:
        """
        Helper extracting relative coordinates and geometric distances.

        Returns:
            (dx, dy, dz, range_3d, range_horizontal_rho)
        """
        if uav_pos is None:
            uav_pos = np.zeros(3, dtype=np.float64)

        dx = float(state[0, 0] - uav_pos[0])
        dy = float(state[2, 0] - uav_pos[1])
        dz = float(state[4, 0] - uav_pos[2])

        rho2 = dx * dx + dy * dy
        rho = max(np.sqrt(rho2), 1e-6)  # Avoid division by zero at zenith/nadir

        r2 = rho2 + dz * dz
        r = max(np.sqrt(r2), 1e-6)

        return dx, dy, dz, r, rho

    def h_radar(self, state: np.ndarray, uav_pos: Optional[np.ndarray]) -> np.ndarray:
        """Nonlinear radar measurement function: h_radar(x) = [R]."""
        _, _, _, r, _ = self._compute_relative_cartesian(state, uav_pos)
        return np.array([[r]], dtype=np.float64)

    def jacobian_radar(self, state: np.ndarray, uav_pos: Optional[np.ndarray]) -> np.ndarray:
        """
        Analytical 1x6 Jacobian matrix for Radar range:
            H_radar = [dx/R, 0, dy/R, 0, dz/R, 0]
        """
        dx, dy, dz, r, _ = self._compute_relative_cartesian(state, uav_pos)
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 0] = dx / r
        H[0, 2] = dy / r
        H[0, 4] = dz / r
        return H

    def h_camera(self, state: np.ndarray, uav_pos: Optional[np.ndarray]) -> np.ndarray:
        """
        Nonlinear camera measurement function:
            h_camera(x) = [psi, lambda]^T = [arctan2(dy, dx), arcsin(dz / R)]^T
        """
        dx, dy, dz, r, _ = self._compute_relative_cartesian(state, uav_pos)
        psi = np.arctan2(dy, dx)
        sin_elev = np.clip(dz / r, -1.0, 1.0)
        elev = np.arcsin(sin_elev)
        return np.array([[psi], [elev]], dtype=np.float64)

    def jacobian_camera(self, state: np.ndarray, uav_pos: Optional[np.ndarray]) -> np.ndarray:
        """
        Analytical 2x6 Jacobian matrix for Camera bearing (azimuth, elevation):
            Row 0 (psi):    [-dy/rho^2,              0,  dx/rho^2,             0,  0,        0]
            Row 1 (lambda): [-dx*dz / (rho * R^2),   0, -dy*dz / (rho * R^2),  0,  rho/R^2,  0]
        """
        dx, dy, dz, r, rho = self._compute_relative_cartesian(state, uav_pos)
        rho2 = rho * rho
        r2 = r * r

        H = np.zeros((2, 6), dtype=np.float64)

        # Azimuth partial derivatives
        H[0, 0] = -dy / rho2
        H[0, 2] = dx / rho2

        # Elevation partial derivatives
        denom = rho * r2
        H[1, 0] = -(dx * dz) / denom
        H[1, 2] = -(dy * dz) / denom
        H[1, 4] = rho / r2

        return H

    # -------------------------------------------------------------------------
    # Measurement Correction (Innovation, Kalman Gain, Joseph Form)
    # -------------------------------------------------------------------------

    def _apply_correction(
        self, measurement: SensorMeasurement, uav_pos: Optional[np.ndarray] = None
    ) -> None:
        """Applies standard in-sequence nonlinear correction equation with Joseph-form covariance stabilization."""
        effective_uav_pos = uav_pos if uav_pos is not None else measurement.uav_pos
        z = np.asarray(measurement.z, dtype=np.float64).reshape((-1, 1))
        R_cov = np.asarray(measurement.R_cov, dtype=np.float64)

        if measurement.sensor_type.upper() == "RADAR":
            z_pred = self.h_radar(self.x, effective_uav_pos)
            H = self.jacobian_radar(self.x, effective_uav_pos)
            y = z - z_pred

        elif measurement.sensor_type.upper() == "CAMERA":
            z_pred = self.h_camera(self.x, effective_uav_pos)
            H = self.jacobian_camera(self.x, effective_uav_pos)
            y = z - z_pred
            # Angular wrapping for azimuth within [-pi, pi]
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

        else:
            raise ValueError(f"Unknown sensor type: '{measurement.sensor_type}'")

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R_cov

        # Compute Normalized Innovation Squared (NIS): y^T * S^(-1) * y
        try:
            self.last_nis = float((y.T @ np.linalg.solve(S, y))[0, 0])
        except np.linalg.LinAlgError:
            self.last_nis = 1.0

        # Kalman Gain: K = P * H^T * S^(-1) via linear solve for numerical stability
        K = np.linalg.solve(S, H @ self.P).T

        # State update
        self.x = self.x + K @ y

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I_KH = self.I6 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------------------
    # OOSM Method 1 (Mode C): History Buffer Rollback & Replay (Paper 2)
    # -------------------------------------------------------------------------

    def _record_snapshot(
        self,
        t: float,
        measurement: Optional[SensorMeasurement],
        uav_pos: Optional[np.ndarray],
    ) -> None:
        """Appends a new state snapshot to the history circular buffer."""
        snap = FilterSnapshot(
            t=t,
            x=self.x.copy(),
            P=self.P.copy(),
            measurement=measurement,
            uav_pos=uav_pos if uav_pos is not None else (measurement.uav_pos if measurement else None),
        )
        self.history.append(snap)
        self._prune_history()

    def _prune_history(self) -> None:
        """Enforces buffer limits while preserving at least the initial anchor snapshot."""
        if len(self.history) > self.max_buffer_size:
            excess = len(self.history) - self.max_buffer_size
            self.history = self.history[excess:]

    def _rollback_and_replay(
        self, measurement: SensorMeasurement, uav_pos: Optional[np.ndarray] = None
    ) -> bool:
        """
        Executes Rollback & Replay algorithm for an Out-Of-Sequence Measurement (OOSM):
            1. Locates historical state bracket preceding measurement.t_capture.
            2. Rewinds filter state and covariance to that bracket.
            3. Injects delayed measurement and updates state.
            4. Replays all subsequent measurements up to current epoch.

        Returns:
            bool: True if OOSM was successfully integrated, False if dropped due to buffer age.
        """
        t_meas = float(measurement.t_capture)
        effective_uav_pos = uav_pos if uav_pos is not None else measurement.uav_pos

        # Check if measurement is older than oldest available snapshot in buffer
        if not self.history or t_meas < self.history[0].t - 1e-9:
            self.dropped_oosm_count += 1
            return False

        # Find the snapshot immediately preceding or at t_meas
        idx = 0
        for i in range(len(self.history)):
            if self.history[i].t <= t_meas + 1e-9:
                idx = i
            else:
                break

        base_snap = self.history[idx]

        # 1. Rollback to base snapshot
        self.x = base_snap.x.copy()
        self.P = base_snap.P.copy()
        self.current_time = base_snap.t

        # 2. Predict to delayed measurement epoch and apply correction
        self.predict(t_meas)
        self._apply_correction(measurement, effective_uav_pos)

        # Create new snapshot for the newly integrated measurement
        new_snap = FilterSnapshot(
            t=t_meas,
            x=self.x.copy(),
            P=self.P.copy(),
            measurement=measurement,
            uav_pos=effective_uav_pos,
        )

        # 3. Replay subsequent historical measurements in strict chronological order
        subsequent = self.history[idx + 1 :]
        self.history = self.history[: idx + 1] + [new_snap]

        for snap in subsequent:
            self.predict(snap.t)
            if snap.measurement is not None:
                self._apply_correction(snap.measurement, snap.uav_pos)

            self.history.append(
                FilterSnapshot(
                    t=snap.t,
                    x=self.x.copy(),
                    P=self.P.copy(),
                    measurement=snap.measurement,
                    uav_pos=snap.uav_pos,
                )
            )

        self.oosm_count += 1
        self._prune_history()
        return True

    # -------------------------------------------------------------------------
    # OOSM Method 2 (Mode D): 1-Step Retrodiction / Modified Kalman (Paper 3)
    # -------------------------------------------------------------------------

    def _update_oosm_retrodiction(
        self, measurement: SensorMeasurement, uav_pos: Optional[np.ndarray] = None
    ) -> bool:
        """
        Executes 1-Step Retrodiction / Delay-Compensated Modified EKF without buffer replay:
            1. Computes observation delay delta = t_current - t_capture > 0.
            2. Retrodicts state to measurement epoch: x_d = F(-delta) * x_k.
            3. Evaluates nonlinear measurement h(x_d) and analytical Jacobian H_d at x_d.
            4. Embeds delay directly into modified observation matrix: H_eff = H_d * F(-delta).
            5. Updates current state x_k and covariance P_k via Joseph form in a single O(1) step.
        """
        t_meas = float(measurement.t_capture)
        delta = self.current_time - t_meas

        if delta > self.max_buffer_dt or t_meas < 0.0:
            self.dropped_oosm_count += 1
            return False

        effective_uav_pos = uav_pos if uav_pos is not None else measurement.uav_pos
        z = np.asarray(measurement.z, dtype=np.float64).reshape((-1, 1))
        R_cov = np.asarray(measurement.R_cov, dtype=np.float64)

        # Backward state transition matrix F(-delta)
        F_retro = self.state_transition_matrix(-delta)
        x_retro = F_retro @ self.x

        if measurement.sensor_type.upper() == "RADAR":
            z_pred = self.h_radar(x_retro, effective_uav_pos)
            H_d = self.jacobian_radar(x_retro, effective_uav_pos)
            y = z - z_pred
        elif measurement.sensor_type.upper() == "CAMERA":
            z_pred = self.h_camera(x_retro, effective_uav_pos)
            H_d = self.jacobian_camera(x_retro, effective_uav_pos)
            y = z - z_pred
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi
        else:
            raise ValueError(f"Unknown sensor type: '{measurement.sensor_type}'")

        # Delay-compensated effective observation matrix mapping current state x_k to delayed z_d
        H_eff = H_d @ F_retro

        # Account for process noise and intermediate update omission uncertainty over delay interval delta
        Q_delay = self.process_noise_covariance(delta)
        P_retro = F_retro @ self.P @ F_retro.T + Q_delay
        R_eff = R_cov + H_d @ (Q_delay + 0.35 * P_retro) @ H_d.T

        S = H_d @ P_retro @ H_d.T + R_eff
        try:
            self.last_nis = float((y.T @ np.linalg.solve(S, y))[0, 0])
        except np.linalg.LinAlgError:
            self.last_nis = 1.0

        # Cross-covariance between current state x_k and delayed measurement z_d (Bar-Shalom 1-step retrodiction)
        P_xz = (self.P - Q_delay) @ H_eff.T
        K = np.linalg.solve(S, P_xz.T).T

        # Update current state and Joseph-form covariance directly at current_time (O(1), no replay)
        self.x = self.x + K @ y
        I_KH = self.I6 - K @ H_eff
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.oosm_count += 1
        return True

    # -------------------------------------------------------------------------
    # OOSM Method 3 (Mode E): Kalman Predictor / Measurement Extrapolation (Paper 4)
    # -------------------------------------------------------------------------

    def _update_oosm_extrapolation(
        self, measurement: SensorMeasurement, uav_pos: Optional[np.ndarray] = None
    ) -> bool:
        """
        Executes Kalman Predictor (Forward Measurement Extrapolation) for OOSM:
            1. Computes delay delta = t_current - t_capture > 0.
            2. Extrapolates delayed measurement z(t_d) forward to current epoch t_k using
               predicted kinematic displacement with predictor damping factor.
            3. Inflates measurement noise covariance R_extrap with extrapolation uncertainty.
            4. Applies standard Joseph-form EKF update at current epoch t_k.
        """
        t_meas = float(measurement.t_capture)
        delta = self.current_time - t_meas

        if delta > self.max_buffer_dt or t_meas < 0.0:
            self.dropped_oosm_count += 1
            return False

        effective_uav_pos = uav_pos if uav_pos is not None else measurement.uav_pos
        z_delayed = np.asarray(measurement.z, dtype=np.float64).reshape((-1, 1))
        R_cov = np.asarray(measurement.R_cov, dtype=np.float64)

        # Forward predictor displacement over delay interval delta (first-order Taylor predictor)
        kp_gain = 0.65
        dx_extrap = np.zeros((6, 1), dtype=np.float64)
        dx_extrap[0, 0] = self.x[1, 0] * delta * kp_gain
        dx_extrap[2, 0] = self.x[3, 0] * delta * kp_gain
        dx_extrap[4, 0] = self.x[5, 0] * delta * kp_gain

        if measurement.sensor_type.upper() == "RADAR":
            H_k = self.jacobian_radar(self.x, effective_uav_pos)
            z_extrap = z_delayed + H_k @ dx_extrap
            z_pred = self.h_radar(self.x, effective_uav_pos)
            y = z_extrap - z_pred
        elif measurement.sensor_type.upper() == "CAMERA":
            H_k = self.jacobian_camera(self.x, effective_uav_pos)
            z_extrap = z_delayed + H_k @ dx_extrap
            z_pred = self.h_camera(self.x, effective_uav_pos)
            y = z_extrap - z_pred
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi
        else:
            raise ValueError(f"Unknown sensor type: '{measurement.sensor_type}'")

        # Extrapolation uncertainty: process noise Q(delta) + velocity estimation uncertainty over delta
        Q_delay = self.process_noise_covariance(delta)
        P_vel_proj = np.zeros((6, 6), dtype=np.float64)
        P_vel_proj[0, 0] = self.P[1, 1] * (delta ** 2)
        P_vel_proj[2, 2] = self.P[3, 3] * (delta ** 2)
        P_vel_proj[4, 4] = self.P[5, 5] * (delta ** 2)

        R_extrap = R_cov + 2.5 * (H_k @ (Q_delay + P_vel_proj) @ H_k.T)

        S = H_k @ self.P @ H_k.T + R_extrap
        try:
            self.last_nis = float((y.T @ np.linalg.solve(S, y))[0, 0])
        except np.linalg.LinAlgError:
            self.last_nis = 1.0

        K = np.linalg.solve(S, H_k @ self.P).T

        self.x = self.x + K @ y
        I_KH = self.I6 - K @ H_k
        self.P = I_KH @ self.P @ I_KH.T + K @ R_extrap @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.oosm_count += 1
        return True

    # -------------------------------------------------------------------------
    # Main Filter Update Step
    # -------------------------------------------------------------------------

    def update(
        self,
        measurement: SensorMeasurement,
        uav_pos: Optional[np.ndarray] = None,
        is_naive: bool = False,
    ) -> None:
        """
        Performs EKF prediction and measurement correction according to configured OOSM strategy.

        Args:
            measurement: SensorMeasurement instance containing readings and timestamps.
            uav_pos: 3D coordinates [x, y, z] of UAV at measurement epoch.
            is_naive: If True, operates in Naive mode (assumes arrival epoch t_receive is capture time).
                      If False, operates in Delay-Compensated mode using t_capture with active OOSM strategy.
        """
        effective_uav_pos = uav_pos if uav_pos is not None else measurement.uav_pos

        # Mode B: Naive filter - ignores latency and assumes arrival epoch is capture epoch
        if is_naive:
            t_target = measurement.t_receive
            self.predict(t_target)
            self._apply_correction(measurement, effective_uav_pos)
            self.in_order_count += 1
            return

        # Delay-Compensated modes (C, D, E) and Ideal Mode A:
        t_meas = float(measurement.t_capture)

        if t_meas >= self.current_time - 1e-9:
            # Chronologically in-order measurement
            self.predict(t_meas)
            self._apply_correction(measurement, effective_uav_pos)
            if self.enable_oosm_buffer and self.oosm_strategy == "ROLLBACK":
                self._record_snapshot(t_meas, measurement, effective_uav_pos)
            self.in_order_count += 1
        else:
            # Out-Of-Sequence Measurement (OOSM) detected (t_capture < current_time)!
            if not self.enable_oosm_buffer:
                self.dropped_oosm_count += 1
                return

            if self.oosm_strategy == "ROLLBACK":
                self._rollback_and_replay(measurement, effective_uav_pos)
            elif self.oosm_strategy == "RETRODICTION":
                self._update_oosm_retrodiction(measurement, effective_uav_pos)
            elif self.oosm_strategy == "EXTRAPOLATION":
                self._update_oosm_extrapolation(measurement, effective_uav_pos)
            else:
                raise ValueError(f"Unknown OOSM strategy: '{self.oosm_strategy}'")

    def get_state(self) -> TargetState:
        """Returns the current estimated target state."""
        return TargetState.from_vector(self.x.flatten())

    def get_covariance(self) -> np.ndarray:
        """Returns current 6x6 error covariance matrix."""
        return self.P.copy()
