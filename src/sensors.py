"""
Synthetic Sensor Models for UAV Air Target Tracking.
Includes Radar (Range only, 10 Hz) and Camera / Electro-Optical (Azimuth & Elevation, 30 Hz)
with transmission/processing latency, Gaussian timing jitter, and hardware clock bias modeling.
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


@dataclass
class SensorMeasurement:
    """
    Standard data container for asynchronous sensor readings with latency and OOSM support.

    Attributes:
        t_capture (float): Physical ground truth epoch when measurement was taken (s).
        sensor_type (str): Identifier ('RADAR' or 'CAMERA').
        z (np.ndarray): Measurement vector.
            - RADAR: 1D array of shape (1,) -> [R]
            - CAMERA: 2D array of shape (2,) -> [azimuth, elevation]
        R_cov (np.ndarray): Measurement noise covariance matrix.
            - RADAR: 2D array of shape (1, 1) -> [[sigma_r^2]]
            - CAMERA: 2D array of shape (2, 2) -> diag([sigma_psi^2, sigma_lambda^2])
        t_receive (float): Epoch when measurement arrives at EKF processor (s).
            t_receive = t_true + latency + jitter
        uav_pos (Optional[np.ndarray]): UAV platform 3D Cartesian position at capture time.
    """
    t_capture: float
    sensor_type: str
    z: np.ndarray
    R_cov: np.ndarray
    t_receive: float = 0.0
    uav_pos: Optional[np.ndarray] = None

    def __init__(
        self,
        t_capture: Optional[float] = None,
        sensor_type: str = "RADAR",
        z: Optional[np.ndarray] = None,
        R_cov: Optional[np.ndarray] = None,
        t_receive: Optional[float] = None,
        uav_pos: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,  # Backward compatibility for Phase 1
    ) -> None:
        if t_capture is None:
            if timestamp is not None:
                t_capture = float(timestamp)
            else:
                raise ValueError("Either 't_capture' or 'timestamp' must be specified.")
        self.t_capture = float(t_capture)
        self.sensor_type = str(sensor_type)
        self.z = np.asarray(z, dtype=np.float64) if z is not None else np.zeros(1, dtype=np.float64)
        self.R_cov = np.asarray(R_cov, dtype=np.float64) if R_cov is not None else np.eye(len(self.z), dtype=np.float64)
        self.t_receive = float(t_receive) if t_receive is not None else self.t_capture
        self.uav_pos = np.asarray(uav_pos, dtype=np.float64) if uav_pos is not None else None

    @property
    def timestamp(self) -> float:
        """Backward compatibility: returns t_capture."""
        return self.t_capture

    def __lt__(self, other: SensorMeasurement) -> bool:
        """
        Priority queue ordering is strictly based on processor arrival epoch (t_receive).
        Tie-breaking falls back to t_capture.
        """
        if abs(self.t_receive - other.t_receive) < 1e-12:
            return self.t_capture < other.t_capture
        return self.t_receive < other.t_receive


class BaseSensor(ABC):
    """Abstract base class for airborne sensors with latency, jitter, and clock bias modeling."""

    def __init__(
        self,
        frequency_hz: float,
        latency_mean: float = 0.0,
        latency_jitter_std: float = 0.0,
        clock_bias: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            frequency_hz: Sensor measurement frequency in Hz.
            latency_mean: Nominal latency / delay in seconds (default: 0.0s).
            latency_jitter_std: 1-sigma standard deviation of Gaussian latency jitter (s).
            clock_bias: Sensor hardware clock bias / offset relative to system time (s).
            rng: NumPy random number generator for reproducible noise & jitter.
        """
        if frequency_hz <= 0:
            raise ValueError("Frequency must be positive.")
        self.frequency_hz = float(frequency_hz)
        self.period = 1.0 / self.frequency_hz
        self.latency_mean = float(latency_mean)
        self.latency_jitter_std = float(latency_jitter_std)
        self.clock_bias = float(clock_bias)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_latency(self) -> float:
        """Computes transmission / processing latency with Gaussian jitter."""
        if self.latency_jitter_std > 0.0:
            jitter = float(self.rng.normal(0.0, self.latency_jitter_std))
        else:
            jitter = 0.0
        return max(0.0, self.latency_mean + jitter)

    @property
    @abstractmethod
    def sensor_type(self) -> str:
        """Returns sensor type name string."""
        pass

    @abstractmethod
    def measure(
        self, timestamp: float, target_pos: np.ndarray, uav_pos: Optional[np.ndarray] = None
    ) -> SensorMeasurement:
        """Generates a synthetic measurement given true target and UAV positions."""
        pass


class RadarSensor(BaseSensor):
    """
    Airborne 1D Radar Range Sensor operating at 10 Hz by default.
    Nominal transmission latency: 8 ms + N(0, 2ms) Gaussian jitter.

    Measurement equation:
        dx = x_tgt - x_uav
        dy = y_tgt - y_uav
        dz = z_tgt - z_uav
        R  = sqrt(dx^2 + dy^2 + dz^2)
        z  = R + N(0, sigma_r^2)
    """

    def __init__(
        self,
        frequency_hz: float = 10.0,
        sigma_r: float = 5.0,
        latency_mean: float = 0.008,          # 8 ms default transmission latency
        latency_jitter_std: float = 0.002,    # 2 ms Gaussian jitter
        clock_bias: float = 0.0,              # Radar synchronized with reference clock
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            frequency_hz: Radar update rate in Hz (default: 10.0 Hz).
            sigma_r: 1-sigma standard deviation of range measurement error in meters (default: 5.0 m).
            latency_mean: Nominal transmission latency in seconds (default: 0.008 s = 8 ms).
            latency_jitter_std: Gaussian jitter standard deviation in seconds (default: 0.002 s = 2 ms).
            clock_bias: Hardware clock offset in seconds (default: 0.0 s).
            rng: NumPy random generator instance for reproducibility.
        """
        super().__init__(
            frequency_hz=frequency_hz,
            latency_mean=latency_mean,
            latency_jitter_std=latency_jitter_std,
            clock_bias=clock_bias,
            rng=rng,
        )
        self.sigma_r = float(sigma_r)
        self.R_cov = np.array([[self.sigma_r**2]], dtype=np.float64)

    @property
    def sensor_type(self) -> str:
        return "RADAR"

    def measure(
        self, timestamp: float, target_pos: np.ndarray, uav_pos: Optional[np.ndarray] = None
    ) -> SensorMeasurement:
        """
        Generate synthetic range reading with Gaussian noise, latency, and timing jitter.

        Args:
            timestamp: True physical epoch of observation (s).
            target_pos: True 3D Cartesian position [x, y, z] of target (m).
            uav_pos: 3D position [x, y, z] of the UAV platform (m), defaults to [0, 0, 0].

        Returns:
            SensorMeasurement with 1D range observation and arrival timestamp.
        """
        if uav_pos is None:
            uav_pos = np.zeros(3, dtype=np.float64)

        rel_pos = np.asarray(target_pos, dtype=np.float64) - np.asarray(uav_pos, dtype=np.float64)
        true_range = float(np.linalg.norm(rel_pos))

        noise = self.rng.normal(0.0, self.sigma_r)
        meas_range = max(0.1, true_range + noise)  # Avoid non-physical negative distance

        z = np.array([meas_range], dtype=np.float64)
        t_capture = float(timestamp + self.clock_bias)
        t_receive = float(t_capture + self.sample_latency())

        return SensorMeasurement(
            t_capture=t_capture,
            sensor_type=self.sensor_type,
            z=z,
            R_cov=self.R_cov.copy(),
            t_receive=t_receive,
            uav_pos=np.asarray(uav_pos, dtype=np.float64),
        )


class CameraSensor(BaseSensor):
    """
    Airborne Electro-Optical / Infrared (EO/IR) Camera Sensor operating at 30 Hz by default.
    Nominal processing latency: 40 ms + N(0, 5ms) Gaussian jitter.
    Hardware clock bias: Independent clock with unknown constant bias (e.g., +25 ms).

    Measurement equations:
        dx = x_tgt - x_uav
        dy = y_tgt - y_uav
        dz = z_tgt - z_uav
        R  = sqrt(dx^2 + dy^2 + dz^2)
        Azimuth   (psi):    arctan2(dy, dx)
        Elevation (lambda): arcsin(dz / R)

    Measurement vector:
        z = [psi_meas, lambda_meas]^T
    Noise model:
        psi_meas    = psi    + N(0, sigma_psi^2)
        lambda_meas = lambda + N(0, sigma_lambda^2)
    """

    def __init__(
        self,
        frequency_hz: float = 30.0,
        sigma_psi: float = 0.005,             # ~0.286 degrees
        sigma_lambda: float = 0.005,          # ~0.286 degrees
        latency_mean: float = 0.040,          # 40 ms heavy image processing delay
        latency_jitter_std: float = 0.005,    # 5 ms Gaussian jitter
        clock_bias: float = 0.025,            # +25 ms hardware clock offset
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Args:
            frequency_hz: Camera update rate in Hz (default: 30.0 Hz).
            sigma_psi: Azimuth noise standard deviation in radians.
            sigma_lambda: Elevation noise standard deviation in radians.
            latency_mean: Processing latency in seconds (default: 0.040 s = 40 ms).
            latency_jitter_std: Jitter standard deviation in seconds (default: 0.005 s = 5 ms).
            clock_bias: Hardware clock bias / offset in seconds (default: 0.025 s = +25 ms).
            rng: NumPy random generator instance.
        """
        super().__init__(
            frequency_hz=frequency_hz,
            latency_mean=latency_mean,
            latency_jitter_std=latency_jitter_std,
            clock_bias=clock_bias,
            rng=rng,
        )
        self.sigma_psi = float(sigma_psi)
        self.sigma_lambda = float(sigma_lambda)
        self.R_cov = np.diag([self.sigma_psi**2, self.sigma_lambda**2]).astype(np.float64)

    @property
    def sensor_type(self) -> str:
        return "CAMERA"

    def measure(
        self, timestamp: float, target_pos: np.ndarray, uav_pos: Optional[np.ndarray] = None
    ) -> SensorMeasurement:
        """
        Generate synthetic bearing (azimuth, elevation) reading with Gaussian noise, latency, and jitter.

        Args:
            timestamp: True physical epoch of observation (s).
            target_pos: True 3D Cartesian position [x, y, z] of target (m).
            uav_pos: 3D position [x, y, z] of the UAV platform (m), defaults to [0, 0, 0].

        Returns:
            SensorMeasurement with 2D angle observation and arrival timestamp.
        """
        if uav_pos is None:
            uav_pos = np.zeros(3, dtype=np.float64)

        rel_pos = np.asarray(target_pos, dtype=np.float64) - np.asarray(uav_pos, dtype=np.float64)
        dx, dy, dz = rel_pos[0], rel_pos[1], rel_pos[2]
        true_range = float(np.linalg.norm(rel_pos))

        if true_range < 1e-6:
            raise ValueError("Target and UAV positions are co-located; angles undefined.")

        true_psi = np.arctan2(dy, dx)
        true_lambda = np.arcsin(np.clip(dz / true_range, -1.0, 1.0))

        psi_noise = self.rng.normal(0.0, self.sigma_psi)
        lambda_noise = self.rng.normal(0.0, self.sigma_lambda)

        meas_psi = true_psi + psi_noise
        meas_lambda = np.clip(true_lambda + lambda_noise, -np.pi / 2, np.pi / 2)

        # Normalize azimuth angle within [-pi, pi]
        meas_psi = (meas_psi + np.pi) % (2.0 * np.pi) - np.pi

        z = np.array([meas_psi, meas_lambda], dtype=np.float64)
        t_capture = float(timestamp + self.clock_bias)
        t_receive = float(t_capture + self.sample_latency())

        return SensorMeasurement(
            t_capture=t_capture,
            sensor_type=self.sensor_type,
            z=z,
            R_cov=self.R_cov.copy(),
            t_receive=t_receive,
            uav_pos=np.asarray(uav_pos, dtype=np.float64),
        )


def reconstruct_pseudo_cartesian(
    range_val: float,
    azimuth_val: float,
    elevation_val: float,
    uav_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstructs rough 3D Cartesian target position from spherical measurements:
        x = R * cos(lambda) * cos(psi) + x_uav
        y = R * cos(lambda) * sin(psi) + y_uav
        z = R * sin(lambda) + z_uav

    Used for raw sensor data visualization and baseline comparisons.
    """
    if uav_pos is None:
        uav_pos = np.zeros(3, dtype=np.float64)

    dx = range_val * np.cos(elevation_val) * np.cos(azimuth_val)
    dy = range_val * np.cos(elevation_val) * np.sin(azimuth_val)
    dz = range_val * np.sin(elevation_val)

    return np.asarray(uav_pos, dtype=np.float64) + np.array([dx, dy, dz], dtype=np.float64)
