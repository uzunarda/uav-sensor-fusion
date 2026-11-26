from __future__ import annotations
import os
import sys
import unittest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.dynamics import TargetState, ConstantVelocityTrajectory
from src.sensors import RadarSensor, CameraSensor, SensorMeasurement
from src.simulator import AsyncSensorEventSimulator
from src.ekf import ExtendedKalmanFilter
from src.metrics import (
    compute_nees_series,
    compute_ospa_series,
    compute_track_continuity_and_idf1,
    evaluate_Comprehensive_metrics,
)


class TestEKFSystem(unittest.TestCase):

    def test_cv_dynamics(self):
        init_state = TargetState(x=100.0, vx=10.0, y=-50.0, vy=5.0, z=-20.0, vz=-1.0)
        traj = ConstantVelocityTrajectory(init_state, t0=0.0)

        state_5s = traj.get_state(5.0)
        self.assertTrue(np.isclose(state_5s.x, 150.0))
        self.assertTrue(np.isclose(state_5s.vx, 10.0))
        self.assertTrue(np.isclose(state_5s.y, -25.0))
        self.assertTrue(np.isclose(state_5s.vy, 5.0))
        self.assertTrue(np.isclose(state_5s.z, -25.0))
        self.assertTrue(np.isclose(state_5s.vz, -1.0))

    def test_radar_jacobian_numerical(self):
        """Verify analytical Radar Jacobian against central finite differences."""
        ekf = ExtendedKalmanFilter(
            initial_state=np.array([500.0, 20.0, -300.0, -10.0, -150.0, 2.0]),
            initial_covariance=np.eye(6),
        )
        uav_pos = np.array([50.0, -30.0, -100.0])
        eps = 1e-6

        H_analytic = ekf.jacobian_radar(ekf.x, uav_pos)
        H_numeric = np.zeros_like(H_analytic)

        for i in range(6):
            x_plus = ekf.x.copy()
            x_minus = ekf.x.copy()
            x_plus[i, 0] += eps
            x_minus[i, 0] -= eps

            r_plus = ekf.h_radar(x_plus, uav_pos)[0, 0]
            r_minus = ekf.h_radar(x_minus, uav_pos)[0, 0]

            H_numeric[0, i] = (r_plus - r_minus) / (2.0 * eps)

        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_camera_jacobian_numerical(self):
        """Verify analytical Camera Jacobian against central finite differences."""
        ekf = ExtendedKalmanFilter(
            initial_state=np.array([450.0, 15.0, 320.0, -8.0, -120.0, 1.5]),
            initial_covariance=np.eye(6),
        )
        uav_pos = np.array([0.0, 0.0, 0.0])
        eps = 1e-6

        H_analytic = ekf.jacobian_camera(ekf.x, uav_pos)
        H_numeric = np.zeros_like(H_analytic)

        for i in range(6):
            x_plus = ekf.x.copy()
            x_minus = ekf.x.copy()
            x_plus[i, 0] += eps
            x_minus[i, 0] -= eps

            z_plus = ekf.h_camera(x_plus, uav_pos)
            z_minus = ekf.h_camera(x_minus, uav_pos)

            diff = z_plus - z_minus
            diff[0, 0] = (diff[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

            H_numeric[:, i] = (diff / (2.0 * eps)).flatten()

        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_sensor_latency_and_timestamps(self):
        """Phase 2: Verify sensor latency modeling, jitter, and clock bias."""
        rng = np.random.default_rng(123)
        radar = RadarSensor(frequency_hz=10.0, latency_mean=0.008, latency_jitter_std=0.002, clock_bias=0.0, rng=rng)
        camera = CameraSensor(frequency_hz=30.0, latency_mean=0.040, latency_jitter_std=0.005, clock_bias=0.025, rng=rng)

        tgt = np.array([1000.0, 200.0, -300.0])
        uav = np.array([0.0, 0.0, -100.0])

        m_radar = radar.measure(timestamp=1.0, target_pos=tgt, uav_pos=uav)
        m_cam = camera.measure(timestamp=1.0, target_pos=tgt, uav_pos=uav)

        # Radar: t_capture == 1.0, t_receive ~= 1.008s
        self.assertAlmostEqual(m_radar.t_capture, 1.0, places=5)
        self.assertGreater(m_radar.t_receive, m_radar.t_capture)
        self.assertAlmostEqual(m_radar.t_receive - m_radar.t_capture, 0.008, delta=0.008)

        # Camera: t_capture == 1.025 (due to +25ms clock bias), t_receive ~= 1.065s
        self.assertAlmostEqual(m_cam.t_capture, 1.025, places=5)
        self.assertGreater(m_cam.t_receive, m_cam.t_capture)
        self.assertAlmostEqual(m_cam.t_receive - m_cam.t_capture, 0.040, delta=0.020)

    def test_priority_queue_arrival_ordering_and_oosm_generation(self):
        """Phase 2: Verify priority queue orders events by t_receive and generates OOSMs."""
        traj = ConstantVelocityTrajectory(TargetState(1000.0, 20.0, 500.0, -15.0, -300.0, 0.0))
        sim = AsyncSensorEventSimulator(traj)

        radar = RadarSensor(10.0, 5.0, latency_mean=0.008, latency_jitter_std=0.002, clock_bias=0.0)
        camera = CameraSensor(30.0, 0.005, 0.005, latency_mean=0.040, latency_jitter_std=0.005, clock_bias=0.025)

        sim.register_sensor(radar, phase_offset=0.0)
        sim.register_sensor(camera, phase_offset=0.0123)
        sim.populate_events(t_start=0.0, t_end=2.0)

        events = sim.clone_event_stream()
        self.assertGreater(len(events), 50)

        # Ensure strict arrival ordering (t_receive)
        prev_recv = -1.0
        oosm_detected = False
        curr_capture_max = -1.0

        for e in events:
            self.assertGreaterEqual(e.t_receive, prev_recv)
            prev_recv = e.t_receive

            if e.t_capture < curr_capture_max:
                oosm_detected = True
            else:
                curr_capture_max = e.t_capture

        self.assertTrue(oosm_detected, "Out-Of-Sequence Measurements (OOSM) must occur due to differential latency.")

    def test_ekf_oosm_rollback_replay(self):
        """Phase 2: Verify EKF buffer rewinds, updates with delayed measurement, and replays."""
        x0 = np.array([1000.0, 20.0, 500.0, -10.0, -300.0, 0.0])
        P0 = np.eye(6) * 100.0

        ekf = ExtendedKalmanFilter(
            initial_state=x0,
            initial_covariance=P0,
            process_noise_spectral_density=0.5,
            initial_timestamp=0.0,
            enable_oosm_buffer=True,
            max_buffer_size=50,
        )

        uav_pos = np.zeros(3)

        # Step 1: In-order Radar measurement at t=1.00s
        m1 = SensorMeasurement(t_capture=1.00, sensor_type="RADAR", z=np.array([1100.0]), R_cov=np.array([[25.0]]), t_receive=1.008, uav_pos=uav_pos)
        ekf.update(m1, uav_pos=uav_pos)
        self.assertAlmostEqual(ekf.current_time, 1.00)
        self.assertEqual(ekf.oosm_count, 0)
        self.assertEqual(len(ekf.history), 2)  # Initial anchor + m1

        # Step 2: In-order Radar measurement at t=1.10s
        m2 = SensorMeasurement(t_capture=1.10, sensor_type="RADAR", z=np.array([1120.0]), R_cov=np.array([[25.0]]), t_receive=1.108, uav_pos=uav_pos)
        ekf.update(m2, uav_pos=uav_pos)
        self.assertAlmostEqual(ekf.current_time, 1.10)
        self.assertEqual(ekf.oosm_count, 0)
        self.assertEqual(len(ekf.history), 3)

        # Step 3: Delayed Camera measurement arriving at t_receive=1.12s, but t_capture=1.05s (OOSM!)
        m3_oosm = SensorMeasurement(t_capture=1.05, sensor_type="CAMERA", z=np.array([0.45, -0.25]), R_cov=np.diag([1e-4, 1e-4]), t_receive=1.12, uav_pos=uav_pos)
        ekf.update(m3_oosm, uav_pos=uav_pos)

        # EKF should have rewound to t=1.00s, applied m3 at t=1.05s, replayed m2 at t=1.10s, and ended back at t=1.10s
        self.assertAlmostEqual(ekf.current_time, 1.10)
        self.assertEqual(ekf.oosm_count, 1)
        self.assertEqual(len(ekf.history), 4)  # Initial + m1 (1.00) + m3 (1.05) + m2 (1.10)

        # Check covariance sanity (symmetric and positive-definite)
        P = ekf.get_covariance()
        self.assertTrue(np.allclose(P, P.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(P) > 0.0))

    def test_ekf_stale_oosm_dropped(self):
        """Phase 2: Verify that measurements older than the buffer's oldest anchor are dropped safely."""
        x0 = np.array([1000.0, 20.0, 500.0, -10.0, -300.0, 0.0])
        P0 = np.eye(6) * 100.0

        ekf = ExtendedKalmanFilter(
            initial_state=x0,
            initial_covariance=P0,
            process_noise_spectral_density=0.5,
            initial_timestamp=5.0,  # Filter starts at t=5.0
            enable_oosm_buffer=True,
            max_buffer_size=10,
        )

        # Measurement at t=2.0s (before filter start epoch t=5.0s)
        stale_meas = SensorMeasurement(t_capture=2.0, sensor_type="RADAR", z=np.array([1000.0]), R_cov=np.array([[25.0]]), t_receive=5.5)
        ekf.update(stale_meas)

        self.assertEqual(ekf.dropped_oosm_count, 1)
        self.assertEqual(ekf.oosm_count, 0)
        self.assertAlmostEqual(ekf.current_time, 5.0)

    def test_ekf_oosm_retrodiction_mode_d(self):
        """Paper 3 (Mode D): Verify 1-Step Retrodiction / Modified EKF updates state without replay."""
        x0 = np.array([1000.0, 20.0, 500.0, -10.0, -300.0, 0.0])
        P0 = np.eye(6) * 100.0

        ekf = ExtendedKalmanFilter(
            initial_state=x0,
            initial_covariance=P0,
            process_noise_spectral_density=0.5,
            initial_timestamp=0.0,
            enable_oosm_buffer=True,
            oosm_strategy="RETRODICTION",
        )
        uav_pos = np.zeros(3)

        m1 = SensorMeasurement(t_capture=1.00, sensor_type="RADAR", z=np.array([1150.0]), R_cov=np.array([[25.0]]), t_receive=1.008, uav_pos=uav_pos)
        ekf.update(m1, uav_pos=uav_pos)
        self.assertAlmostEqual(ekf.current_time, 1.00)

        # Delayed camera measurement captured at t=0.96s (OOSM relative to t=1.00s)
        m2_oosm = SensorMeasurement(t_capture=0.96, sensor_type="CAMERA", z=np.array([0.45, -0.25]), R_cov=np.diag([1e-4, 1e-4]), t_receive=1.02, uav_pos=uav_pos)
        ekf.update(m2_oosm, uav_pos=uav_pos)

        self.assertEqual(ekf.oosm_count, 1)
        self.assertAlmostEqual(ekf.current_time, 1.00)
        self.assertEqual(len(ekf.history), 0)  # Mode D does not require snapshot replay buffer
        P = ekf.get_covariance()
        self.assertTrue(np.allclose(P, P.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(P) > 0.0))

    def test_ekf_oosm_extrapolation_mode_e(self):
        """Paper 4 (Mode E): Verify Kalman Predictor forward measurement extrapolation."""
        x0 = np.array([1000.0, 20.0, 500.0, -10.0, -300.0, 0.0])
        P0 = np.eye(6) * 100.0

        ekf = ExtendedKalmanFilter(
            initial_state=x0,
            initial_covariance=P0,
            process_noise_spectral_density=0.5,
            initial_timestamp=0.0,
            enable_oosm_buffer=True,
            oosm_strategy="EXTRAPOLATION",
        )
        uav_pos = np.zeros(3)

        m1 = SensorMeasurement(t_capture=1.00, sensor_type="RADAR", z=np.array([1150.0]), R_cov=np.array([[25.0]]), t_receive=1.008, uav_pos=uav_pos)
        ekf.update(m1, uav_pos=uav_pos)

        m2_oosm = SensorMeasurement(t_capture=0.95, sensor_type="CAMERA", z=np.array([0.45, -0.25]), R_cov=np.diag([1e-4, 1e-4]), t_receive=1.03, uav_pos=uav_pos)
        ekf.update(m2_oosm, uav_pos=uav_pos)

        self.assertEqual(ekf.oosm_count, 1)
        self.assertAlmostEqual(ekf.current_time, 1.00)
        P = ekf.get_covariance()
        self.assertTrue(np.allclose(P, P.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(P) > 0.0))

    def test_literature_metrics_ospa_nees_idf1(self):
        """Verify mathematical properties of NEES, OSPA, MAE, Track Continuity, and IDF1 metrics."""
        states_gt = np.zeros((10, 6), dtype=np.float64)
        states_est = np.ones((10, 6), dtype=np.float64)
        covs = [np.eye(6, dtype=np.float64) for _ in range(10)]

        nees = compute_nees_series(states_est, states_gt, covs)
        # Error is 1 in each of 6 dimensions with P = I_6 -> NEES must be exactly 6.0
        np.testing.assert_allclose(nees, np.full(10, 6.0), atol=1e-6)

        pos_err = np.array([2.0, 2.5, 3.0, 20.0])
        ospa = compute_ospa_series(pos_err, cutoff_c=15.0, order_p=2.0, gate_threshold_m=6.0)
        self.assertAlmostEqual(ospa[0], 2.0)
        self.assertLessEqual(ospa[3], 15.0)
        self.assertGreater(ospa[3], 10.0)

        tc = compute_track_continuity_and_idf1(np.array([1.5, 2.0, 2.5, 8.0]), gate_threshold_m=4.5)
        self.assertAlmostEqual(tc["track_continuity_pct"], 75.0)
        self.assertGreater(tc["idf1_score"], 0.5)
        self.assertLess(tc["idf1_score"], 1.0)

    def test_rk4_and_6dof_maneuver_trajectories(self):
        """Verify 3-2-1 Euler DCM orthogonality, RK4 integration, Coordinated Turn, and Evasive Weave."""
        from src.dynamics import (
            euler_to_dcm,
            rk4_step,
            CoordinatedTurnTrajectory,
            EvasiveWeaveTrajectory,
            UAVFlightTrajectory,
            create_trajectory,
        )

        # 1. Euler DCM orthogonality: R^T * R == I_3 and det(R) == +1
        R = euler_to_dcm(roll=0.25, pitch=-0.15, yaw=1.10)
        self.assertEqual(R.shape, (3, 3))
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

        # 2. RK4 on simple harmonic oscillator dx/dt = v, dv/dt = -x
        def sho_deriv(_t: float, y: np.ndarray) -> np.ndarray:
            return np.array([y[1], -y[0]], dtype=np.float64)

        y = np.array([1.0, 0.0], dtype=np.float64)
        for i in range(100):
            y = rk4_step(sho_deriv, i * 0.01, y, 0.01)
        np.testing.assert_allclose(y, np.array([np.cos(1.0), -np.sin(1.0)]), atol=1e-6)

        # 3. Coordinated Turn (CT) preserves horizontal speed magnitude under constant turn rate
        init_state = TargetState(x=1000.0, vx=40.0, y=-500.0, vy=0.0, z=-400.0, vz=-2.0)
        ct_traj = CoordinatedTurnTrajectory(init_state, turn_rate_rad_s=0.08, t0=0.0)
        st_0 = ct_traj.get_state(0.0)
        st_10 = ct_traj.get_state(10.0)
        self.assertAlmostEqual(st_0.x, 1000.0, places=6)
        speed_horiz_0 = np.hypot(st_0.vx, st_0.vy)
        speed_horiz_10 = np.hypot(st_10.vx, st_10.vy)
        self.assertAlmostEqual(speed_horiz_0, speed_horiz_10, places=4)
        self.assertNotAlmostEqual(st_10.vy, 0.0)  # Velocity vector rotated

        # 4. Evasive S-Weave trajectory and factory
        ev_traj = create_trajectory("EVASIVE", init_state, t0=0.0)
        self.assertIsInstance(ev_traj, EvasiveWeaveTrajectory)
        st_ev = ev_traj.get_state(6.5)
        self.assertEqual(st_ev.to_vector().shape, (6,))

        # 5. Moving 6-DOF UAV flight trajectory
        uav_traj = UAVFlightTrajectory(
            initial_pos=np.array([0.0, 0.0, -150.0]),
            cruise_speed=25.0,
            orbit_radius=400.0,
            dynamic_flight=True,
        )
        uav_pose = uav_traj.get_pose(5.0)
        self.assertEqual(uav_pose.position.shape, (3,))
        self.assertEqual(uav_pose.dcm_body_to_ned.shape, (3, 3))
        np.testing.assert_allclose(uav_pose.dcm_body_to_ned.T @ uav_pose.dcm_body_to_ned, np.eye(3), atol=1e-12)

    def test_camera_fpa_image_processing_pipeline(self):
        """Verify 2D FPA synthetic image generation, CFAR thresholding, sub-pixel centroiding, and inverse pinhole projection."""
        rng = np.random.default_rng(42)
        cam = CameraSensor(
            frequency_hz=30.0,
            sigma_psi=0.004,
            sigma_lambda=0.004,
            latency_mean=0.040,
            latency_jitter_std=0.005,
            clock_bias=0.025,
            rng=rng,
            enable_image_processing=True,
        )
        uav_pos = np.array([0.0, 0.0, -150.0])
        target_pos = np.array([900.0, -450.0, -380.0])

        meas = cam.measure(timestamp=2.0, target_pos=target_pos, uav_pos=uav_pos)
        self.assertIsNotNone(meas.cv_telemetry)
        cv = meas.cv_telemetry
        self.assertIn("centroid_u", cv)
        self.assertIn("centroid_v", cv)
        self.assertIn("bbox", cv)
        self.assertIn("blob_area_px", cv)
        self.assertIn("snr_db", cv)
        self.assertGreater(cv["blob_area_px"], 0)
        self.assertGreater(cv["snr_db"], 8.0)

        # Verify that inverse-pinhole angle from centroid matches true bearing within 3-sigma
        rel = target_pos - uav_pos
        true_psi = np.arctan2(rel[1], rel[0])
        true_lam = np.arcsin(rel[2] / np.linalg.norm(rel))
        self.assertAlmostEqual(float(meas.z[0]), float(true_psi), delta=0.03)
        self.assertAlmostEqual(float(meas.z[1]), float(true_lam), delta=0.03)


if __name__ == "__main__":
    unittest.main()
