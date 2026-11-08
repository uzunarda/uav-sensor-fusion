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


if __name__ == "__main__":
    unittest.main()
