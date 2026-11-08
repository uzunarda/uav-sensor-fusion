"""
Simulation Engine and Background Worker for UAV Multi-Sensor EKF Tactical GUI.
Supports Real-Time playback, multi-speed multipliers (0.5x, 1x, 2x, Max),
dynamic Mode switching (Mode A: Ideal, Mode B: Naive, Mode C: Rollback/Replay),
and instantaneous timeline scrubbing (seeking).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from src.dynamics import TargetState, ConstantVelocityTrajectory
from src.sensors import RadarSensor, CameraSensor, reconstruct_pseudo_cartesian, SensorMeasurement
from src.simulator import AsyncSensorEventSimulator
from src.ekf import ExtendedKalmanFilter


@dataclass
class SimulationFrame:
    """Telemetry and display state for a single simulation time slice."""
    timestamp: float
    sim_mode: str  # "MODE_A", "MODE_B", "MODE_C"
    target_gt_pos: np.ndarray  # [x, y, z] NED
    target_gt_vel: np.ndarray  # [vx, vy, vz]
    uav_pos: np.ndarray        # [x, y, z] NED
    ekf_pos: np.ndarray        # [x, y, z]
    ekf_vel: np.ndarray        # [vx, vy, vz]
    cov_2d: np.ndarray         # 2x2 position covariance matrix in North-East (X-Y)
    latest_raw_pos: Optional[np.ndarray]  # [x, y, z] pseudo-cartesian or None
    latest_raw_type: Optional[str]        # 'RADAR' or 'CAMERA' or None
    is_oosm_triggered: bool
    oosm_total_count: int
    pos_error_3d: float
    vel_error_3d: float
    range_to_target: float
    altitude: float            # Height above ground (-z in NED)
    progress_pct: float        # 0.0 to 100.0


class SimulationWorker(QObject):
    """
    Simulation Controller and State Generator for the Tactical GUI.

    Emits:
        frame_ready(SimulationFrame, list[tuple], list[tuple]): Emitted at ~60 FPS with current frame and track histories.
        playback_state_changed(bool): Emitted with is_playing.
        mode_changed(str): Emitted when active mode is switched.
        simulation_finished(): Emitted when timeline reaches the end.
    """
    frame_ready = pyqtSignal(object, object, object, object)  # frame, gt_history, ekf_history, raw_history
    playback_state_changed = pyqtSignal(bool)
    mode_changed = pyqtSignal(str)
    simulation_finished = pyqtSignal()

    def __init__(
        self,
        sim_duration: float = 20.0,
        seed: int = 42,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.sim_duration = float(sim_duration)
        self.seed = int(seed)

        # Simulation parameters
        self.uav_pos = np.array([0.0, 0.0, -150.0], dtype=np.float64)  # 150m altitude
        self.true_initial_state = TargetState(
            x=1200.0,
            vx=35.0,
            y=-800.0,
            vy=25.0,
            z=-500.0,
            vz=-4.0,
        )
        self.initial_offset = np.array([80.0, -15.0, -90.0, 10.0, 40.0, 6.0])
        self.x_init_guess = self.true_initial_state.to_vector() + self.initial_offset
        self.P_init = np.diag([
            150.0**2, 20.0**2,
            150.0**2, 20.0**2,
            100.0**2, 15.0**2,
        ])
        self.q_continuous = 0.8

        # Active state
        self.current_mode = "MODE_C"
        self.speed_multiplier = 1.0  # 0.5, 1.0, 2.0, or 0.0 (Max speed)
        self.is_playing = False
        self.current_time = 0.0

        # Frame cache per mode for instant scrubbing
        self._mode_frames: Dict[str, List[SimulationFrame]] = {}
        self._mode_gt_tracks: Dict[str, np.ndarray] = {}
        self._mode_ekf_tracks: Dict[str, np.ndarray] = {}
        self._mode_raw_points: Dict[str, List[tuple]] = {}

        # Precompute all three modes for instantaneous mode switching and scrubbing
        self.recompute_all_modes()

        # Timer for 60 FPS playback (16 ms interval)
        self._timer_interval_ms = 16
        self._timer = QTimer(self)
        self._timer.setInterval(self._timer_interval_ms)
        self._timer.timeout.connect(self._on_timer_tick)

    def recompute_all_modes(self) -> None:
        """Precomputes simulation trajectories for Modes A, B, and C."""
        for mode in ["MODE_A", "MODE_B", "MODE_C"]:
            self._compute_mode_data(mode)

    def _compute_mode_data(self, mode: str) -> None:
        """Runs the EKF engine for a specific mode and saves timeline frames."""
        rng = np.random.default_rng(self.seed)
        trajectory = ConstantVelocityTrajectory(self.true_initial_state, t0=0.0)

        if mode == "MODE_A":
            radar = RadarSensor(10.0, 6.0, latency_mean=0.0, latency_jitter_std=0.0, clock_bias=0.0, rng=rng)
            camera = CameraSensor(30.0, 0.004, 0.004, latency_mean=0.0, latency_jitter_std=0.0, clock_bias=0.0, rng=rng)
            is_naive = False
            enable_oosm = False
        elif mode == "MODE_B":
            radar = RadarSensor(10.0, 6.0, latency_mean=0.008, latency_jitter_std=0.002, clock_bias=0.0, rng=rng)
            camera = CameraSensor(30.0, 0.004, 0.004, latency_mean=0.040, latency_jitter_std=0.005, clock_bias=0.025, rng=rng)
            is_naive = True
            enable_oosm = False
        else:  # MODE_C
            radar = RadarSensor(10.0, 6.0, latency_mean=0.008, latency_jitter_std=0.002, clock_bias=0.0, rng=rng)
            camera = CameraSensor(30.0, 0.004, 0.004, latency_mean=0.040, latency_jitter_std=0.005, clock_bias=0.025, rng=rng)
            is_naive = False
            enable_oosm = True

        sim = AsyncSensorEventSimulator(trajectory)
        sim.register_sensor(radar, phase_offset=0.0)
        sim.register_sensor(camera, phase_offset=0.0125)
        sim.populate_events(t_start=0.0, t_end=self.sim_duration, uav_position_fn=lambda t: self.uav_pos)

        ekf = ExtendedKalmanFilter(
            initial_state=self.x_init_guess.copy(),
            initial_covariance=self.P_init.copy(),
            process_noise_spectral_density=self.q_continuous,
            enable_oosm_buffer=enable_oosm,
            max_buffer_size=150,
        )

        frames: List[SimulationFrame] = []
        gt_track_list = []
        ekf_track_list = []
        raw_points_list = []

        latest_radar_range: Optional[float] = None
        latest_cam_angles: Optional[tuple[float, float]] = None

        prev_oosm_count = 0

        for event in sim:
            if event.sensor_type == "RADAR":
                latest_radar_range = float(event.z[0])
            elif event.sensor_type == "CAMERA":
                latest_cam_angles = (float(event.z[0]), float(event.z[1]))

            raw_pos: Optional[np.ndarray] = None
            if latest_radar_range is not None and latest_cam_angles is not None:
                raw_pos = reconstruct_pseudo_cartesian(
                    range_val=latest_radar_range,
                    azimuth_val=latest_cam_angles[0],
                    elevation_val=latest_cam_angles[1],
                    uav_pos=self.uav_pos,
                )

            # Check if this incoming event triggers OOSM in Mode C
            is_oosm_event = False
            if mode == "MODE_C" and event.t_capture < ekf.current_time - 1e-9:
                is_oosm_event = True

            ekf.update(event, uav_pos=self.uav_pos, is_naive=is_naive)

            t_eval = event.t_receive if is_naive else ekf.current_time
            gt_state = trajectory.get_state(t_eval)
            target_gt_pos = gt_state.position.copy()
            target_gt_vel = gt_state.velocity.copy()

            est_state = ekf.x.flatten().copy()
            ekf_pos = est_state[[0, 2, 4]]
            ekf_vel = est_state[[1, 3, 5]]

            # 2D horizontal covariance for ellipse: rows/cols for x (North) and y (East)
            # Covariance matrix P order: 0:x, 1:vx, 2:y, 3:vy, 4:z, 5:vz
            cov_2d = ekf.P[[0, 2], :][:, [0, 2]].copy()

            err_pos = float(np.linalg.norm(ekf_pos - target_gt_pos))
            err_vel = float(np.linalg.norm(ekf_vel - target_gt_vel))
            range_to_target = float(np.linalg.norm(ekf_pos - self.uav_pos))
            altitude = float(-ekf_pos[2])  # -z is height in NED

            is_oosm_now = is_oosm_event or (ekf.oosm_count > prev_oosm_count)
            prev_oosm_count = ekf.oosm_count

            frame = SimulationFrame(
                timestamp=t_eval,
                sim_mode=mode,
                target_gt_pos=target_gt_pos,
                target_gt_vel=target_gt_vel,
                uav_pos=self.uav_pos.copy(),
                ekf_pos=ekf_pos,
                ekf_vel=ekf_vel,
                cov_2d=cov_2d,
                latest_raw_pos=raw_pos,
                latest_raw_type=event.sensor_type,
                is_oosm_triggered=is_oosm_now,
                oosm_total_count=ekf.oosm_count,
                pos_error_3d=err_pos,
                vel_error_3d=err_vel,
                range_to_target=range_to_target,
                altitude=altitude,
                progress_pct=(t_eval / self.sim_duration) * 100.0,
            )
            frames.append(frame)
            gt_track_list.append(target_gt_pos)
            ekf_track_list.append(ekf_pos)
            if raw_pos is not None:
                raw_points_list.append((t_eval, raw_pos))

        self._mode_frames[mode] = frames
        self._mode_gt_tracks[mode] = np.array(gt_track_list)
        self._mode_ekf_tracks[mode] = np.array(ekf_track_list)
        self._mode_raw_points[mode] = raw_points_list

    # -------------------------------------------------------------------------
    # Playback Control Methods
    # -------------------------------------------------------------------------

    def play(self) -> None:
        """Starts playback."""
        if not self.is_playing:
            self.is_playing = True
            self._timer.start()
            self.playback_state_changed.emit(True)

    def pause(self) -> None:
        """Pauses playback."""
        if self.is_playing:
            self.is_playing = False
            self._timer.stop()
            self.playback_state_changed.emit(False)

    def toggle_play_pause(self) -> None:
        """Toggles between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def reset(self) -> None:
        """Resets playback timeline to epoch 0.0s."""
        self.pause()
        self.current_time = 0.0
        self._emit_current_frame()

    def seek_time(self, target_time: float) -> None:
        """Instantly jumps to specified simulation epoch in seconds."""
        self.current_time = max(0.0, min(self.sim_duration, float(target_time)))
        self._emit_current_frame()

    def set_speed(self, multiplier: float) -> None:
        """Sets playback speed multiplier (0.5, 1.0, 2.0, or 0.0 for max speed)."""
        self.speed_multiplier = float(multiplier)

    def set_mode(self, mode: str) -> None:
        """Switches active simulation mode ('MODE_A', 'MODE_B', 'MODE_C')."""
        if mode in ["MODE_A", "MODE_B", "MODE_C"] and mode != self.current_mode:
            self.current_mode = mode
            self.mode_changed.emit(mode)
            self._emit_current_frame()

    # -------------------------------------------------------------------------
    # Internal Timer Tick and Frame Dispatch
    # -------------------------------------------------------------------------

    def _on_timer_tick(self) -> None:
        """Advances simulation clock and dispatches the corresponding frame."""
        dt_real = self._timer_interval_ms / 1000.0

        if self.speed_multiplier > 0.0:
            sim_dt = dt_real * self.speed_multiplier
        else:
            # Real-Time OFF: advance very rapidly (e.g. 5x per tick)
            sim_dt = dt_real * 8.0

        self.current_time += sim_dt

        if self.current_time >= self.sim_duration:
            self.current_time = self.sim_duration
            self._emit_current_frame()
            self.pause()
            self.simulation_finished.emit()
            return

        self._emit_current_frame()

    def _emit_current_frame(self) -> None:
        """Finds closest frame at self.current_time and emits Qt signal."""
        frames = self._mode_frames.get(self.current_mode, [])
        if not frames:
            return

        # Binary search for closest frame timestamp
        times = [f.timestamp for f in frames]
        idx = int(np.searchsorted(times, self.current_time))
        idx = min(idx, len(frames) - 1)

        current_frame = frames[idx]

        # Extract track histories up to this index
        gt_history = self._mode_gt_tracks[self.current_mode][: idx + 1]
        ekf_history = self._mode_ekf_tracks[self.current_mode][: idx + 1]

        # Raw points up to current time (last 30 points for clarity)
        raw_all = self._mode_raw_points[self.current_mode]
        raw_filtered = [p[1] for p in raw_all if p[0] <= self.current_time]
        raw_tail = raw_filtered[-40:] if len(raw_filtered) > 40 else raw_filtered

        self.frame_ready.emit(current_frame, gt_history, ekf_history, raw_tail)
