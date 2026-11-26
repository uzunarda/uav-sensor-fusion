"""
Synthetic Sensor Models for UAV Air Target Tracking.
Includes Radar (Range only, 10 Hz) and Camera / Electro-Optical (Azimuth & Elevation, 30 Hz)
with transmission/processing latency, Gaussian timing jitter, and hardware clock bias modeling.
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
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
    cv_telemetry: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        t_capture: Optional[float] = None,
        sensor_type: str = "RADAR",
        z: Optional[np.ndarray] = None,
        R_cov: Optional[np.ndarray] = None,
        t_receive: Optional[float] = None,
        uav_pos: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,  # Backward compatibility for Phase 1
        cv_telemetry: Optional[Dict[str, Any]] = None,
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
        self.cv_telemetry = cv_telemetry

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
        enable_image_processing: bool = True,
        res_width: int = 320,
        res_height: int = 240,
        fov_deg: float = 22.0,
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
            enable_image_processing: When True, synthesizes a 2D Focal Plane Array (FPA) infrared image,
                applies adaptive CFAR thresholding, extracts sub-pixel intensity centroid (u_c, v_c) and
                bounding box, and back-projects through the inverse Pinhole camera matrix.
            res_width: FPA horizontal pixel resolution (default: 320 px).
            res_height: FPA vertical pixel resolution (default: 240 px).
            fov_deg: Horizontal field-of-view in degrees (default: 22.0 deg).
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

        # Pinhole Camera Intrinsic Parameters (Focal Plane Array)
        self.enable_image_processing = bool(enable_image_processing)
        self.res_width = int(res_width)
        self.res_height = int(res_height)
        self.fov_rad = float(np.radians(fov_deg))
        self.fx = (self.res_width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.fy = self.fx
        self.cx = self.res_width / 2.0
        self.cy = self.res_height / 2.0
        # Pre-allocate local 33x33 FPA ROI grid for fast vectorized 2D PSF rendering & moment centroiding
        self._roi_half = 16
        grid_axis = np.arange(-self._roi_half, self._roi_half + 1, dtype=np.float64)
        self._du_grid, self._dv_grid = np.meshgrid(grid_axis, grid_axis)
        self._fpa_rng = np.random.default_rng(104729)

    @property
    def sensor_type(self) -> str:
        return "CAMERA"

    def _run_fpa_vision_pipeline(
        self,
        true_psi: float,
        true_lambda: float,
        true_range: float,
        psi_turbulence: float,
        lambda_turbulence: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Executes a genuine 2D Focal Plane Array (FPA) Image Processing Pipeline:
          1. Gimbal Servo Pointing & Forward Pinhole Projection -> (u_true, v_true) on 320x240 sensor.
          2. 2D Thermal/EO Point Spread Function (Anisotropic Gaussian signature + Beer-Lambert attenuation)
             synthesized onto a 2D pixel intensity matrix I(u, v) with thermal background & readout noise.
          3. Adaptive CFAR Statistical Thresholding: T = mu_bg + 3.8 * sigma_bg.
          4. Connected Blob Segmentation, Bounding Box [u_min, v_min, u_max, v_max], & SNR (dB) calculation.
          5. Intensity-Weighted First-Order Spatial Moment Sub-Pixel Centroiding -> (u_c, v_c).
          6. Inverse Pinhole Back-Projection: (u_c, v_c) -> measured (psi_meas, lambda_meas).
        """
        # 1. Gimbal servo tracking with slight mechanical pointing lag (~0.35 deg around target LOS)
        gimbal_psi = true_psi + float(self._fpa_rng.normal(0.0, 0.0055))
        gimbal_lam = true_lambda + float(self._fpa_rng.normal(0.0, 0.0055))

        d_psi_true = (true_psi - gimbal_psi + np.pi) % (2.0 * np.pi) - np.pi
        d_lam_true = true_lambda - gimbal_lam

        # Forward Pinhole Projection onto 320x240 Focal Plane Array
        u_true = self.cx + self.fx * np.tan(d_psi_true)
        v_true = self.cy + self.fy * np.tan(d_lam_true)

        # Anchor integer pixel center of the 33x33 ROI window on the FPA
        u0 = int(np.round(u_true))
        v0 = int(np.round(v_true))
        u_grid = u0 + self._du_grid
        v_grid = v0 + self._dv_grid

        # Sub-pixel optical turbulence / focal plane displacement on the FPA
        u_spot = self.cx + self.fx * np.tan(d_psi_true + psi_turbulence)
        v_spot = self.cy + self.fy * np.tan(d_lam_true + lambda_turbulence)

        # Apparent target thermal spread radius in pixels (wingspan projected by focal length fx / R)
        sigma_px_x = float(np.clip((self.fx * 6.5) / max(true_range, 150.0), 1.35, 5.5))
        sigma_px_y = float(np.clip((self.fy * 3.8) / max(true_range, 150.0), 1.15, 4.2))

        # Beer-Lambert atmospheric transmittance over slant range R
        atm_trans = float(np.exp(-0.00012 * true_range))
        peak_signal = 220.0 * atm_trans

        # 2. Synthesize 2D FPA Pixel Matrix I(u, v) = Background + Target PSF + Sensor Readout Noise
        bg_mean = 22.0
        bg_sigma = 3.2
        psf = peak_signal * np.exp(
            -0.5 * (((u_grid - u_spot) / sigma_px_x) ** 2 + ((v_grid - v_spot) / sigma_px_y) ** 2)
        )
        noise_frame = self._fpa_rng.normal(bg_mean, bg_sigma, size=psf.shape)
        img_roi = np.clip(psf + noise_frame, 0.0, 255.0)

        # 3. Adaptive CFAR Background Estimation (from outer border pixels of ROI) & Thresholding
        border_pixels = np.concatenate([
            img_roi[0, :], img_roi[-1, :], img_roi[1:-1, 0], img_roi[1:-1, -1]
        ])
        est_mu_bg = float(np.mean(border_pixels))
        est_sigma_bg = max(float(np.std(border_pixels)), 1.0)
        threshold = est_mu_bg + 3.8 * est_sigma_bg

        # Binary Segmentation Mask
        mask = img_roi > threshold
        if not np.any(mask):
            # Fallback to peak pixel if target SNR is extremely low
            mask = img_roi >= float(np.max(img_roi))

        # 4. Extract Bounding Box, Blob Pixel Area & Peak SNR (dB)
        weights = np.where(mask, img_roi - threshold, 0.0)
        m00 = float(np.sum(weights))
        if m00 <= 1e-9:
            u_c = float(u_spot)
            v_c = float(v_spot)
        else:
            # 5. First-Order Spatial Moment Sub-Pixel Centroid (Center of Gravity)
            u_c = float(np.sum(weights * u_grid) / m00)
            v_c = float(np.sum(weights * v_grid) / m00)

        active_u = u_grid[mask]
        active_v = v_grid[mask]
        u_min, u_max = int(np.min(active_u)), int(np.max(active_u))
        v_min, v_max = int(np.min(active_v)), int(np.max(active_v))
        blob_area = int(np.sum(mask))
        peak_val = float(np.max(img_roi))
        snr_db = float(20.0 * np.log10(max(peak_val - est_mu_bg, 1.0) / est_sigma_bg))

        # 6. Inverse Pinhole Back-Projection: (u_c, v_c) -> (psi_meas, lambda_meas)
        d_psi_meas = float(np.arctan2(u_c - self.cx, self.fx))
        d_lam_meas = float(np.arctan2(v_c - self.cy, self.fy))

        meas_psi = (gimbal_psi + d_psi_meas + np.pi) % (2.0 * np.pi) - np.pi
        meas_lambda = float(np.clip(gimbal_lam + d_lam_meas, -np.pi / 2.0, np.pi / 2.0))

        # Downsample 11x11 center patch of the FPA intensity matrix for live HUD thermal visualization
        patch_11 = img_roi[self._roi_half - 5 : self._roi_half + 6, self._roi_half - 5 : self._roi_half + 6]
        cv_info: Dict[str, Any] = {
            "centroid_u": round(u_c, 2),
            "centroid_v": round(v_c, 2),
            "true_u": round(float(u_true), 2),
            "true_v": round(float(v_true), 2),
            "bbox": [u_min, v_min, max(4, u_max - u_min + 1), max(4, v_max - v_min + 1)],
            "blob_area_px": blob_area,
            "snr_db": round(snr_db, 1),
            "threshold": round(threshold, 1),
            "peak_intensity": round(peak_val, 1),
            "fpa_patch": [[int(val) for val in row] for row in patch_11],
        }
        return meas_psi, meas_lambda, cv_info

    def measure(
        self, timestamp: float, target_pos: np.ndarray, uav_pos: Optional[np.ndarray] = None
    ) -> SensorMeasurement:
        """
        Generate synthetic bearing (azimuth, elevation) reading via 2D FPA Image Processing
        (Pinhole projection -> 2D IR PSF & noise synthesis -> CFAR thresholding -> sub-pixel moment centroiding).
        """
        if uav_pos is None:
            uav_pos = np.zeros(3, dtype=np.float64)

        rel_pos = np.asarray(target_pos, dtype=np.float64) - np.asarray(uav_pos, dtype=np.float64)
        dx, dy, dz = rel_pos[0], rel_pos[1], rel_pos[2]
        true_range = float(np.linalg.norm(rel_pos))

        if true_range < 1e-6:
            raise ValueError("Target and UAV positions are co-located; angles undefined.")

        true_psi = float(np.arctan2(dy, dx))
        true_lambda = float(np.arcsin(np.clip(dz / true_range, -1.0, 1.0)))

        psi_noise = float(self.rng.normal(0.0, self.sigma_psi))
        lambda_noise = float(self.rng.normal(0.0, self.sigma_lambda))

        cv_telemetry: Optional[Dict[str, Any]] = None
        if self.enable_image_processing:
            meas_psi, meas_lambda, cv_telemetry = self._run_fpa_vision_pipeline(
                true_psi=true_psi,
                true_lambda=true_lambda,
                true_range=true_range,
                psi_turbulence=psi_noise,
                lambda_turbulence=lambda_noise,
            )
        else:
            meas_psi = (true_psi + psi_noise + np.pi) % (2.0 * np.pi) - np.pi
            meas_lambda = float(np.clip(true_lambda + lambda_noise, -np.pi / 2, np.pi / 2))

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
            cv_telemetry=cv_telemetry,
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
