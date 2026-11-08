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
            # Wrap azimuth difference if needed
            diff[0, 0] = (diff[0, 0] + np.pi) % (2.0 * np.pi) - np.pi

            H_numeric[:, i] = (diff / (2.0 * eps)).flatten()

        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_event_simulator_ordering(self):
        """Ensure priority queue delivers strictly non-decreasing timestamps."""
        init_state = TargetState(x=1000.0, vx=20.0, y=500.0, vy=-15.0, z=-300.0, vz=0.0)
        traj = ConstantVelocityTrajectory(init_state)

        sim = AsyncSensorEventSimulator(traj)
        sim.register_sensor(RadarSensor(frequency_hz=10.0), phase_offset=0.0)
        sim.register_sensor(CameraSensor(frequency_hz=30.0), phase_offset=0.0123)

        count = sim.populate_events(t_start=0.0, t_end=1.0)
        self.assertGreater(count, 0)

        prev_t = -1.0
        radar_count = 0
        camera_count = 0

        for event in sim:
            self.assertGreaterEqual(event.timestamp, prev_t)
            prev_t = event.timestamp
            if event.sensor_type == "RADAR":
                radar_count += 1
            elif event.sensor_type == "CAMERA":
                camera_count += 1

        self.assertGreaterEqual(radar_count, 10)
        self.assertGreaterEqual(camera_count, 30)

    def test_ekf_filtering_convergence(self):
        """Ensure EKF converges and maintains symmetric positive-definite covariance."""
        init_true = TargetState(x=800.0, vx=25.0, y=-400.0, vy=10.0, z=-200.0, vz=-2.0)
        traj = ConstantVelocityTrajectory(init_true)

        # Initial guess with offset
        init_guess = np.array([850.0, 10.0, -350.0, 0.0, -150.0, 0.0])
        init_cov = np.diag([200.0**2, 20.0**2, 200.0**2, 20.0**2, 100.0**2, 10.0**2])

        ekf = ExtendedKalmanFilter(
            initial_state=init_guess,
            initial_covariance=init_cov,
            process_noise_spectral_density=0.1,
        )

        sim = AsyncSensorEventSimulator(traj)
        sim.register_sensor(RadarSensor(frequency_hz=10.0, sigma_r=2.0), phase_offset=0.0)
        sim.register_sensor(CameraSensor(frequency_hz=30.0, sigma_psi=0.002, sigma_lambda=0.002), phase_offset=0.015)
        sim.populate_events(t_start=0.0, t_end=10.0)

        for event in sim:
            ekf.update(event)

        # Check symmetry and positive definiteness of P
        P = ekf.get_covariance()
        self.assertTrue(np.allclose(P, P.T))
        eigvals = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigvals > 0.0))

        # Check error reduction
        final_true = traj.get_state(10.0)
        final_est = ekf.get_state()
        pos_err = np.linalg.norm(final_true.position - final_est.position)
        vel_err = np.linalg.norm(final_true.velocity - final_est.velocity)

        self.assertLess(pos_err, 15.0)  # High precision convergence
        self.assertLess(vel_err, 5.0)


if __name__ == "__main__":
    unittest.main()
