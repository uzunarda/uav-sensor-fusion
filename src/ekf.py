"""
Asynchronous Extended Kalman Filter (EKF) Engine for 3D Target Tracking.
Implements Continuous White Noise Acceleration (CWNA) dynamics and nonlinear Radar/Camera updates
with analytic Jacobians and Joseph-form covariance stabilization.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .sensors import SensorMeasurement
from .dynamics import TargetState


class ExtendedKalmanFilter:
    """
    6-DOF Asynchronous Extended Kalman Filter in 3D NED frame.

    State vector:
        x = [x, vx, y, vy, z, vz]^T

    Sub-blocks:
        - Axis X: states 0, 1
        - Axis Y: states 2, 3
        - Axis Z: states 4, 5
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise_spectral_density: float = 1.0,
        initial_timestamp: float = 0.0,
    ) -> None:
        """
        Args:
            initial_state: 6-element initial state estimate vector.
            initial_covariance: 6x6 initial error covariance matrix P_0.
            process_noise_spectral_density: Power spectral density (q_c) of continuous-time acceleration noise (m^2/s^3).
            initial_timestamp: Initial filter epoch in seconds.
        """
        self.x = np.asarray(initial_state, dtype=np.float64).reshape((6, 1))
        self.P = np.asarray(initial_covariance, dtype=np.float64)

        if self.P.shape != (6, 6):
            raise ValueError(f"Covariance matrix P must be 6x6, got {self.P.shape}")

        self.q_c = float(process_noise_spectral_density)
        self.current_time = float(initial_timestamp)
        self.I6 = np.eye(6, dtype=np.float64)

    # -------------------------------------------------------------------------
    # System Transition and Process Noise Matrices
    # -------------------------------------------------------------------------

    @staticmethod
    def state_transition_matrix(dt: float) -> np.ndarray:
        """
        Constructs the 6x6 discrete-time state transition matrix F(dt) for Constant Velocity:
            F_1D = [[1, dt], [0, 1]]
            F    = blkdiag(F_1D, F_1D, F_1D)
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
        dt2 = dt * dt
        dt3 = dt2 * dt

        q11 = (dt3 / 3.0) * self.q_c
        q12 = (dt2 / 2.0) * self.q_c
        q22 = dt * self.q_c

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
                f"Out-of-order event received: filter time {self.current_time:.6f}s, event time {timestamp:.6f}s"
            )

        if dt <= 1e-9:
            # Zero or near-zero time step (simultaneous arrival), state and covariance do not propagate
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
    # Update Step
    # -------------------------------------------------------------------------

    def update(
        self, measurement: SensorMeasurement, uav_pos: Optional[np.ndarray] = None
    ) -> None:
        """
        Performs EKF measurement correction with Joseph-form covariance stabilization.

        Args:
            measurement: SensorMeasurement instance containing timestamp, type, z, R_cov.
            uav_pos: 3D coordinates [x, y, z] of UAV at measurement epoch.
        """
        # Ensure state is predicted up to measurement epoch
        self.predict(measurement.timestamp)

        z = np.asarray(measurement.z, dtype=np.float64).reshape((-1, 1))
        R_cov = np.asarray(measurement.R_cov, dtype=np.float64)

        if measurement.sensor_type.upper() == "RADAR":
            z_pred = self.h_radar(self.x, uav_pos)
            H = self.jacobian_radar(self.x, uav_pos)
            # Innovation
            y = z - z_pred

        elif measurement.sensor_type.upper() == "CAMERA":
            z_pred = self.h_camera(self.x, uav_pos)
            H = self.jacobian_camera(self.x, uav_pos)
            # Innovation with angular wrapping for azimuth in [-pi, pi]
            y = z - z_pred
            y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

        else:
            raise ValueError(f"Unknown sensor type: '{measurement.sensor_type}'")

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R_cov

        # Kalman Gain: K = P * H^T * S^(-1)
        # Solve S * K^T = H * P via linear system for numerical stability
        K = np.linalg.solve(S, H @ self.P).T

        # State update
        self.x = self.x + K @ y

        # Joseph form covariance update for positive semi-definiteness & symmetry:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = self.I6 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def get_state(self) -> TargetState:
        """Returns the current estimated target state."""
        return TargetState.from_vector(self.x.flatten())

    def get_covariance(self) -> np.ndarray:
        """Returns current 6x6 error covariance matrix."""
        return self.P.copy()
