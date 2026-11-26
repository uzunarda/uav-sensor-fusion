"""
Target & UAV 6-DOF Flight Dynamics & Ground Truth Trajectory Generators in 3D NED Frame.
Implements aerospace-standard kinematic and maneuvering trajectory models:
  1. Constant Velocity (CV) - 3D linear flight
  2. Coordinated Turn (CT) - 3D constant turn-rate helical maneuver solved via 4th-order Runge-Kutta (RK4)
  3. Evasive S-Weave (EVASIVE) - High-agility sinusoidal lateral/vertical evasive maneuver solved via RK4
  4. 6-DOF Moving UAV Ownship Trajectory with 3-2-1 Euler Direction Cosine Matrix (DCM) body-to-NED attitude
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Computes the standard aerospace 3-2-1 (Yaw-Pitch-Roll) Direction Cosine Matrix (DCM)
    mapping vectors from the UAV Body Frame to the North-East-Down (NED) Frame:
        R_b^n = R_z(yaw) * R_y(pitch) * R_x(roll)

    Args:
        roll: Bank angle phi in radians (rotation about body X-axis).
        pitch: Elevation/pitch angle theta in radians (rotation about body Y-axis).
        yaw: Heading angle psi in radians (rotation about NED Z-axis, 0 = North, +pi/2 = East).

    Returns:
        np.ndarray: 3x3 orthogonal rotation matrix R_b^n (dtype=np.float64).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rk4_step(
    deriv_fn: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Performs a single 4th-Order Runge-Kutta (RK4) numerical integration step for dy/dt = f(t, y).

    Args:
        deriv_fn: Continuous-time state derivative function f(t, y).
        t: Current epoch (s).
        y: Current state vector (np.float64).
        dt: Integration time step (s).

    Returns:
        np.ndarray: Integrated state vector y(t + dt).
    """
    if abs(dt) < 1e-15:
        return y.copy()

    k1 = deriv_fn(t, y)
    k2 = deriv_fn(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = deriv_fn(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = deriv_fn(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class TargetState:
    """
    Kinematic state of a target in 3D North-East-Down (NED) coordinate frame.

    State vector representation:
    x = [x, v_x, y, v_y, z, v_z]^T

    Attributes:
        x (float): North position (m)
        vx (float): North velocity (m/s)
        y (float): East position (m)
        vy (float): East velocity (m/s)
        z (float): Down position (m, positive downwards in NED)
        vz (float): Down velocity (m/s)
    """
    x: float
    vx: float
    y: float
    vy: float
    z: float
    vz: float

    def to_vector(self) -> np.ndarray:
        """Converts the kinematic state to a 6-element numpy array."""
        return np.array([self.x, self.vx, self.y, self.vy, self.z, self.vz], dtype=np.float64)

    @classmethod
    def from_vector(cls, state_vector: np.ndarray) -> TargetState:
        """Creates a TargetState instance from a 6-element numpy array or vector."""
        flattened = np.asarray(state_vector, dtype=np.float64).flatten()
        if flattened.shape[0] != 6:
            raise ValueError(f"State vector must have 6 elements, received {flattened.shape[0]}")
        return cls(
            x=float(flattened[0]),
            vx=float(flattened[1]),
            y=float(flattened[2]),
            vy=float(flattened[3]),
            z=float(flattened[4]),
            vz=float(flattened[5]),
        )

    @property
    def position(self) -> np.ndarray:
        """Returns 3D Cartesian position vector [x, y, z]^T in meters."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        """Returns 3D Cartesian velocity vector [vx, vy, vz]^T in m/s."""
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @property
    def attitude_euler(self) -> Tuple[float, float, float]:
        """Estimates target (roll, pitch, yaw) in radians from velocity vector."""
        horiz_speed = max(float(np.hypot(self.vx, self.vy)), 1e-6)
        yaw = float(np.arctan2(self.vy, self.vx))
        pitch = float(np.arctan2(-self.vz, horiz_speed))
        return 0.0, pitch, yaw


class ConstantVelocityTrajectory:
    """
    Analytical Ground Truth trajectory generator under 3D Constant Velocity (CV) dynamics.

    Kinematic propagation equations:
        x(t)  = x_0  + v_x * t
        vx(t) = v_x
        y(t)  = y_0  + v_y * t
        vy(t) = v_y
        z(t)  = z_0  + v_z * t
        vz(t) = v_z
    """
    scenario_code = "CV"

    def __init__(self, initial_state: TargetState, t0: float = 0.0) -> None:
        self.initial_state = initial_state
        self.t0 = float(t0)

    def get_state(self, t: float) -> TargetState:
        """Compute analytical ground truth target state at time t."""
        dt = t - self.t0
        x = self.initial_state.x + self.initial_state.vx * dt
        vx = self.initial_state.vx
        y = self.initial_state.y + self.initial_state.vy * dt
        vy = self.initial_state.vy
        z = self.initial_state.z + self.initial_state.vz * dt
        vz = self.initial_state.vz
        return TargetState(x=x, vx=vx, y=y, vy=vy, z=z, vz=vz)

    def generate_ground_truth_samples(
        self, t_start: float, t_end: float, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sampled ground truth trajectory vectors for offline benchmarking and plotting."""
        timestamps = np.arange(t_start, t_end + 1e-9, dt)
        states = np.zeros((len(timestamps), 6), dtype=np.float64)
        for i, t in enumerate(timestamps):
            states[i, :] = self.get_state(t).to_vector()
        return timestamps, states


class CoordinatedTurnTrajectory(ConstantVelocityTrajectory):
    """
    3D Helical Coordinated Turn (CT) Maneuvering Target Trajectory integrated with RK4.

    Differential equations in 3D NED:
        dx/dt  = vx
        dvx/dt = -omega * vy
        dy/dt  = vy
        dvy/dt = +omega * vx
        dz/dt  = vz
        dvz/dt = 0
    where omega (rad/s) is the constant horizontal turn rate.
    """
    scenario_code = "CT"

    def __init__(
        self,
        initial_state: TargetState,
        turn_rate_rad_s: float = 0.065,
        t0: float = 0.0,
        rk4_dt: float = 0.01,
    ) -> None:
        super().__init__(initial_state, t0=t0)
        self.omega = float(turn_rate_rad_s)
        self.rk4_dt = float(rk4_dt)
        self._y_cache: list[np.ndarray] = [self.initial_state.to_vector()]

    def _deriv(self, _t: float, state_vec: np.ndarray) -> np.ndarray:
        vx = state_vec[1]
        vy = state_vec[3]
        vz = state_vec[5]
        return np.array(
            [
                vx,
                -self.omega * vy,
                vy,
                self.omega * vx,
                vz,
                0.0,
            ],
            dtype=np.float64,
        )

    def _ensure_cache_up_to(self, target_idx: int) -> None:
        curr_len = len(self._y_cache)
        if target_idx < curr_len:
            return
        y = self._y_cache[-1].copy()
        h = self.rk4_dt
        for i in range(curr_len - 1, target_idx):
            curr_t = self.t0 + i * h
            y = rk4_step(self._deriv, curr_t, y, h)
            self._y_cache.append(y.copy())

    def get_state(self, t: float) -> TargetState:
        """Evaluates CT differential equations at time t in O(1) using cached RK4 grid + remainder RK4 step."""
        total_dt = float(t - self.t0)
        if abs(total_dt) < 1e-12:
            return TargetState.from_vector(self.initial_state.to_vector())
        if total_dt < 0.0:
            y = rk4_step(self._deriv, self.t0, self.initial_state.to_vector(), total_dt)
            return TargetState.from_vector(y)

        idx = int(total_dt / self.rk4_dt)
        self._ensure_cache_up_to(idx)
        y = self._y_cache[idx]
        rem = total_dt - idx * self.rk4_dt
        if rem > 1e-12:
            y = rk4_step(self._deriv, self.t0 + idx * self.rk4_dt, y, rem)
        return TargetState.from_vector(y)


class EvasiveWeaveTrajectory(ConstantVelocityTrajectory):
    """
    High-Agility Evasive S-Weave Maneuvering Target Trajectory integrated via RK4.
    Simulates an aircraft executing sinusoidal lateral S-turns coupled with vertical altitude weaving:
        ax(t) = -A_lat * omega_w * sin(omega_w * t) * sin(psi_0)
        ay(t) = +A_lat * omega_w * cos(omega_w * t) * cos(psi_0)
        az(t) = -A_vert * (2 * omega_w) * cos(2 * omega_w * t)
    """
    scenario_code = "EVASIVE"

    def __init__(
        self,
        initial_state: TargetState,
        lateral_accel_amp: float = 22.0,
        weave_freq_hz: float = 0.11,
        vertical_accel_amp: float = 6.5,
        t0: float = 0.0,
        rk4_dt: float = 0.01,
    ) -> None:
        super().__init__(initial_state, t0=t0)
        self.lateral_accel_amp = float(lateral_accel_amp)
        self.omega_w = 2.0 * np.pi * float(weave_freq_hz)
        self.vertical_accel_amp = float(vertical_accel_amp)
        self.rk4_dt = float(rk4_dt)
        self.psi_0 = float(np.arctan2(initial_state.vy, initial_state.vx))
        self._y_cache: list[np.ndarray] = [self.initial_state.to_vector()]

    def _deriv(self, t: float, state_vec: np.ndarray) -> np.ndarray:
        tau = t - self.t0
        lat_acc = self.lateral_accel_amp * np.sin(self.omega_w * tau)
        # Apply lateral acceleration perpendicular to nominal heading psi_0
        ax = -lat_acc * np.sin(self.psi_0)
        ay = lat_acc * np.cos(self.psi_0)
        az = self.vertical_accel_amp * np.cos(1.5 * self.omega_w * tau)
        return np.array(
            [
                state_vec[1],
                ax,
                state_vec[3],
                ay,
                state_vec[5],
                az,
            ],
            dtype=np.float64,
        )

    def _ensure_cache_up_to(self, target_idx: int) -> None:
        curr_len = len(self._y_cache)
        if target_idx < curr_len:
            return
        y = self._y_cache[-1].copy()
        h = self.rk4_dt
        for i in range(curr_len - 1, target_idx):
            curr_t = self.t0 + i * h
            y = rk4_step(self._deriv, curr_t, y, h)
            self._y_cache.append(y.copy())

    def get_state(self, t: float) -> TargetState:
        """Evaluates Evasive Weave dynamics at time t in O(1) using cached RK4 grid + remainder RK4 step."""
        total_dt = float(t - self.t0)
        if abs(total_dt) < 1e-12:
            return TargetState.from_vector(self.initial_state.to_vector())
        if total_dt < 0.0:
            y = rk4_step(self._deriv, self.t0, self.initial_state.to_vector(), total_dt)
            return TargetState.from_vector(y)

        idx = int(total_dt / self.rk4_dt)
        self._ensure_cache_up_to(idx)
        y = self._y_cache[idx]
        rem = total_dt - idx * self.rk4_dt
        if rem > 1e-12:
            y = rk4_step(self._deriv, self.t0 + idx * self.rk4_dt, y, rem)
        return TargetState.from_vector(y)


@dataclass
class UAVPlatformState:
    """
    6-DOF Pose and Kinematic State of the Ownship UAV in 3D NED frame.

    Attributes:
        position: 3D position [x, y, z] in NED (m).
        velocity: 3D velocity [vx, vy, vz] in NED (m/s).
        roll: Bank angle phi (rad).
        pitch: Pitch angle theta (rad).
        yaw: Heading angle psi (rad).
        dcm_body_to_ned: 3x3 orthogonal rotation matrix R_b^n.
    """
    position: np.ndarray
    velocity: np.ndarray
    roll: float
    pitch: float
    yaw: float
    dcm_body_to_ned: np.ndarray


class UAVFlightTrajectory:
    """
    6-DOF Ownship UAV Flight Path Generator.
    Supports both stationary hover reference and dynamic 6-DOF pursuit/interception flight
    with coordinated turn banking (phi = atan(v * omega / g)), pitch climb, and heading changes.
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        cruise_speed: float = 42.0,
        orbit_radius: float = 420.0,
        dynamic_flight: bool = True,
    ) -> None:
        self.initial_pos = np.asarray(initial_pos, dtype=np.float64).flatten()
        self.cruise_speed = float(cruise_speed)
        self.orbit_radius = max(50.0, float(orbit_radius))
        self.dynamic_flight = bool(dynamic_flight)
        self.omega = self.cruise_speed / self.orbit_radius if self.dynamic_flight else 0.0

    def get_pose(self, t: float) -> UAVPlatformState:
        """Returns 6-DOF UAV position, velocity, Euler angles, and DCM at epoch t."""
        if not self.dynamic_flight:
            pos = self.initial_pos.copy()
            vel = np.zeros(3, dtype=np.float64)
            dcm = euler_to_dcm(0.0, 0.0, 0.0)
            return UAVPlatformState(
                position=pos,
                velocity=vel,
                roll=0.0,
                pitch=0.0,
                yaw=0.0,
                dcm_body_to_ned=dcm,
            )

        # Dynamic 3D tactical pursuit & banking S-curve flight path
        tau = float(t)
        w = 0.18  # rad/s turn frequency
        # Forward pursuit along North-East with sinusoidal banking weave & climb
        x = self.initial_pos[0] + 34.0 * tau + 95.0 * np.sin(w * tau)
        y = self.initial_pos[1] - 18.0 * tau + 130.0 * (1.0 - np.cos(w * tau))
        z = self.initial_pos[2] - 4.5 * tau - 25.0 * np.sin(0.8 * w * tau)

        vx = 34.0 + 95.0 * w * np.cos(w * tau)
        vy = -18.0 + 130.0 * w * np.sin(w * tau)
        vz = -4.5 - 25.0 * (0.8 * w) * np.cos(0.8 * w * tau)

        ax = -95.0 * (w ** 2) * np.sin(w * tau)
        ay = 130.0 * (w ** 2) * np.cos(w * tau)

        yaw = float(np.arctan2(vy, vx))
        horiz_v = max(float(np.hypot(vx, vy)), 1e-6)
        pitch = float(np.arctan2(-vz, horiz_v))
        # Lateral centripetal acceleration -> coordinated bank angle roll
        a_lat = (-ax * np.sin(yaw) + ay * np.cos(yaw))
        roll = float(np.clip(np.arctan(a_lat / 9.80665), -0.85, 0.85))

        dcm = euler_to_dcm(roll, pitch, yaw)
        return UAVPlatformState(
            position=np.array([x, y, z], dtype=np.float64),
            velocity=np.array([vx, vy, vz], dtype=np.float64),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            dcm_body_to_ned=dcm,
        )

    def get_position(self, t: float) -> np.ndarray:
        """Convenience method returning 3D NED position vector at time t."""
        return self.get_pose(t).position


def create_trajectory(
    scenario: str,
    initial_state: TargetState,
    t0: float = 0.0,
) -> Union[ConstantVelocityTrajectory, CoordinatedTurnTrajectory, EvasiveWeaveTrajectory]:
    """
    Factory creating the requested target maneuver trajectory generator.

    Supported scenarios:
        - 'CV'      : 3D Constant Velocity linear flight
        - 'CT'      : 3D Coordinated Turn helical maneuver (RK4)
        - 'EVASIVE' : High-agility Evasive S-Weave maneuver (RK4)
    """
    key = str(scenario).upper().strip()
    if key in ("CT", "COORDINATED_TURN"):
        return CoordinatedTurnTrajectory(initial_state, turn_rate_rad_s=0.095, t0=t0)
    if key in ("EVASIVE", "S_WEAVE", "WEAVE"):
        return EvasiveWeaveTrajectory(
            initial_state,
            lateral_accel_amp=22.0,
            weave_freq_hz=0.11,
            vertical_accel_amp=6.5,
            t0=t0,
        )
    return ConstantVelocityTrajectory(initial_state, t0=t0)
