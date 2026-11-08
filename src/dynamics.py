"""
Target Dynamics & Ground Truth Trajectory Generator in 3D NED Frame.
Implements the 3D Constant Velocity (CV) kinematic state model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


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
        """Converts the kinematic state to a 6x1 numpy array."""
        return np.array([self.x, self.vx, self.y, self.vy, self.z, self.vz], dtype=np.float64)

    @classmethod
    def from_vector(cls, state_vector: np.ndarray) -> TargetState:
        """
        Creates a TargetState instance from a 6-element numpy array or vector.
        """
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

    def __init__(self, initial_state: TargetState, t0: float = 0.0) -> None:
        """
        Initialize the trajectory with an initial target state at time t0.

        Args:
            initial_state: TargetState at reference time t0.
            t0: Reference starting time (seconds), default 0.0.
        """
        self.initial_state = initial_state
        self.t0 = float(t0)

    def get_state(self, t: float) -> TargetState:
        """
        Compute analytical ground truth target state at time t.

        Args:
            t: Query time in seconds.

        Returns:
            TargetState: Ground truth kinematic state at time t.
        """
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
        """
        Generates sampled ground truth trajectory vectors for offline benchmarking and plotting.

        Args:
            t_start: Start time (s)
            t_end: End time (s)
            dt: Time step resolution (s)

        Returns:
            timestamps (np.ndarray): 1D array of sampled timestamps of length N.
            states (np.ndarray): 2D array of shape (N, 6) containing states at each timestamp.
        """
        timestamps = np.arange(t_start, t_end + 1e-9, dt)
        states = np.zeros((len(timestamps), 6), dtype=np.float64)
        for i, t in enumerate(timestamps):
            states[i, :] = self.get_state(t).to_vector()
        return timestamps, states
