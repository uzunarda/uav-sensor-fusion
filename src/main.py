"""
Main execution script for Asynchronous Multi-Sensor EKF Fusion in UAV target tracking.
Simulates a 20-second flight scenario, fuses Radar (10 Hz) and Camera (30 Hz) measurements,
computes trajectory RMSE metrics, and visualizes tracking and velocity convergence with Matplotlib.
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import TargetState, ConstantVelocityTrajectory
from src.sensors import RadarSensor, CameraSensor, reconstruct_pseudo_cartesian
from src.simulator import AsyncSensorEventSimulator
from src.ekf import ExtendedKalmanFilter


def run_simulation(
    sim_duration: float = 20.0,
    seed: int = 42,
    save_fig_path: str = "results/ekf_fusion_results.png",
    show_plot: bool = True,
) -> dict:
    """
    Runs asynchronous sensor fusion simulation and benchmarks EKF against Ground Truth.

    Args:
        sim_duration: Total simulation time in seconds (default: 20.0s).
        seed: Random number generator seed for deterministic reproducibility.
        save_fig_path: Filepath where the output plot will be saved.
        show_plot: Whether to display interactive Matplotlib window.

    Returns:
        dict: Summary metrics containing RMSE and final estimation errors.
    """
    rng = np.random.default_rng(seed)

    print("=" * 78)
    print("  ASYNCHRONOUS MULTI-SENSOR DATA FUSION & EKF TARGET TRACKING ENGINE")
    print("=" * 78)

    # 1. Define True Target Kinematics (NED Frame)
    # Target starts at [1200m North, -800m East, -500m Down] with CV dynamics
    true_initial_state = TargetState(
        x=1200.0,
        vx=35.0,     # 35 m/s North (~126 km/h)
        y=-800.0,
        vy=25.0,     # 25 m/s East (~90 km/h)
        z=-500.0,    # 500m Altitude in NED (z is negative downwards)
        vz=-4.0,     # Climbing at 4 m/s (NED dz/dt is negative)
    )
    trajectory = ConstantVelocityTrajectory(true_initial_state, t0=0.0)

    # UAV assumed cruising at [0, 0, -150] m (150m altitude)
    uav_pos = np.array([0.0, 0.0, -150.0], dtype=np.float64)

    print(f"[Dynamics] True Initial Target State: x0={true_initial_state.x:.1f}m, "
          f"y0={true_initial_state.y:.1f}m, z0={true_initial_state.z:.1f}m")
    print(f"           True Velocities: vx={true_initial_state.vx:.1f}m/s, "
          f"vy={true_initial_state.vy:.1f}m/s, vz={true_initial_state.vz:.1f}m/s")
    print(f"[Platform] UAV Reference Position: North={uav_pos[0]:.1f}m, "
          f"East={uav_pos[1]:.1f}m, Alt={-uav_pos[2]:.1f}m")

    # 2. Configure Asynchronous Sensors
    # Radar: 10 Hz, sigma_r = 6.0 m
    # Camera: 30 Hz, sigma_angle = 0.004 rad (~0.23 deg)
    radar = RadarSensor(frequency_hz=10.0, sigma_r=6.0, rng=rng)
    camera = CameraSensor(frequency_hz=30.0, sigma_psi=0.004, sigma_lambda=0.004, rng=rng)

    simulator = AsyncSensorEventSimulator(trajectory)
    # Introduce initial clock phase offsets to model real-world asynchronous arrivals
    simulator.register_sensor(radar, phase_offset=0.0)
    simulator.register_sensor(camera, phase_offset=0.0125)

    num_events = simulator.populate_events(
        t_start=0.0,
        t_end=sim_duration,
        uav_position_fn=lambda t: uav_pos,
        jitter_std=0.0002,  # 0.2 ms clock jitter
    )
    print(f"[Simulator] Enqueued {num_events} asynchronous sensor events over {sim_duration:.1f}s.")

    # 3. Initialize Extended Kalman Filter
    # Add initial estimation offset to verify convergence capability
    initial_offset = np.array([80.0, -15.0, -90.0, 10.0, 40.0, 6.0])
    x_init_guess = true_initial_state.to_vector() + initial_offset

    # Initial covariance: significant uncertainty in position & velocity
    P_init = np.diag([
        150.0**2,   # x pos variance
        20.0**2,    # vx vel variance
        150.0**2,   # y pos variance
        20.0**2,    # vy vel variance
        100.0**2,   # z pos variance
        15.0**2,    # vz vel variance
    ])
    q_continuous = 0.8  # Continuous process noise acceleration spectral density (m^2/s^3)

    ekf = ExtendedKalmanFilter(
        initial_state=x_init_guess,
        initial_covariance=P_init,
        process_noise_spectral_density=q_continuous,
        initial_timestamp=0.0,
    )
    print("[EKF] Filter initialized with intentional state offset and large covariance.")

    # 4. Asynchronous Simulation Loop
    log_time = []
    log_true_state = []
    log_ekf_state = []
    log_ekf_cov = []

    # Raw Cartesian pseudo-measurements
    log_raw_time = []
    log_raw_pos = []

    latest_radar_range: float | None = None
    latest_cam_angles: tuple[float, float] | None = None

    for event in simulator:
        # Update raw measurements tracking
        if event.sensor_type == "RADAR":
            latest_radar_range = float(event.z[0])
        elif event.sensor_type == "CAMERA":
            latest_cam_angles = (float(event.z[0]), float(event.z[1]))

        # Reconstruct pseudo-Cartesian if both sensor readings are available
        if latest_radar_range is not None and latest_cam_angles is not None:
            raw_pos = reconstruct_pseudo_cartesian(
                range_val=latest_radar_range,
                azimuth_val=latest_cam_angles[0],
                elevation_val=latest_cam_angles[1],
                uav_pos=uav_pos,
            )
            log_raw_time.append(event.timestamp)
            log_raw_pos.append(raw_pos)

        # Execute EKF predict and update steps
        ekf.update(event, uav_pos=uav_pos)

        # Log filter estimates and ground truth
        t_now = event.timestamp
        gt_state = trajectory.get_state(t_now).to_vector()
        est_state = ekf.x.flatten().copy()
        est_cov = np.diag(ekf.P).copy()

        log_time.append(t_now)
        log_true_state.append(gt_state)
        log_ekf_state.append(est_state)
        log_ekf_cov.append(est_cov)

    # Convert logs to NumPy arrays
    t_arr = np.array(log_time)
    gt_arr = np.array(log_true_state)    # (N, 6)
    ekf_arr = np.array(log_ekf_state)    # (N, 6)
    cov_arr = np.array(log_ekf_cov)      # (N, 6) - diagonal elements

    t_raw = np.array(log_raw_time)
    raw_pos_arr = np.array(log_raw_pos)  # (M, 3)

    # 5. Performance Metrics (Errors & RMSE)
    error_state = ekf_arr - gt_arr  # [ex, evx, ey, evy, ez, evz]
    pos_err_3d = np.linalg.norm(error_state[:, [0, 2, 4]], axis=1)
    vel_err_3d = np.linalg.norm(error_state[:, [1, 3, 5]], axis=1)

    # Calculate overall RMSE
    rmse_x = float(np.sqrt(np.mean(error_state[:, 0] ** 2)))
    rmse_y = float(np.sqrt(np.mean(error_state[:, 2] ** 2)))
    rmse_z = float(np.sqrt(np.mean(error_state[:, 4] ** 2)))
    rmse_pos_total = float(np.sqrt(np.mean(pos_err_3d**2)))

    rmse_vx = float(np.sqrt(np.mean(error_state[:, 1] ** 2)))
    rmse_vy = float(np.sqrt(np.mean(error_state[:, 3] ** 2)))
    rmse_vz = float(np.sqrt(np.mean(error_state[:, 5] ** 2)))
    rmse_vel_total = float(np.sqrt(np.mean(vel_err_3d**2)))

    # Steady-state metrics (after t >= 3.0s convergence period)
    ss_mask = t_arr >= 3.0
    ss_rmse_pos = float(np.sqrt(np.mean(pos_err_3d[ss_mask] ** 2)))
    ss_rmse_vel = float(np.sqrt(np.mean(vel_err_3d[ss_mask] ** 2)))

    # Raw measurement RMSE for comparison
    gt_interp_x = np.interp(t_raw, t_arr, gt_arr[:, 0])
    gt_interp_y = np.interp(t_raw, t_arr, gt_arr[:, 2])
    gt_interp_z = np.interp(t_raw, t_arr, gt_arr[:, 4])
    raw_err_x = raw_pos_arr[:, 0] - gt_interp_x
    raw_err_y = raw_pos_arr[:, 1] - gt_interp_y
    raw_err_z = raw_pos_arr[:, 2] - gt_interp_z
    raw_pos_err_3d = np.sqrt(raw_err_x**2 + raw_err_y**2 + raw_err_z**2)
    raw_rmse_total = float(np.sqrt(np.mean(raw_pos_err_3d**2)))

    print("\n" + "=" * 78)
    print("                         FUSION PERFORMANCE BENCHMARK")
    print("=" * 78)
    print(f" Total Simulation Duration : {sim_duration:.2f} s")
    print(f" Total Fused Events        : {len(t_arr)} (Radar: 10 Hz, Camera: 30 Hz)")
    print("-" * 78)
    print(" POSITIONAL RMSE (m):")
    print(f"   Axis X (North)          : {rmse_x:8.3f} m  (Steady-State: {np.sqrt(np.mean(error_state[ss_mask, 0]**2)):.3f} m)")
    print(f"   Axis Y (East)           : {rmse_y:8.3f} m  (Steady-State: {np.sqrt(np.mean(error_state[ss_mask, 2]**2)):.3f} m)")
    print(f"   Axis Z (Down)           : {rmse_z:8.3f} m  (Steady-State: {np.sqrt(np.mean(error_state[ss_mask, 4]**2)):.3f} m)")
    print(f"   Total 3D Position RMSE  : {rmse_pos_total:8.3f} m  (Steady-State: {ss_rmse_pos:.3f} m)")
    print(f"   Raw Sensor Unfused RMSE : {raw_rmse_total:8.3f} m  (EKF Improvement: {(1.0 - ss_rmse_pos / raw_rmse_total) * 100:.1f}%)")
    print("-" * 78)
    print(" VELOCITY RMSE (m/s):")
    print(f"   Axis Vx (North)         : {rmse_vx:8.3f} m/s")
    print(f"   Axis Vy (East)          : {rmse_vy:8.3f} m/s")
    print(f"   Axis Vz (Down)          : {rmse_vz:8.3f} m/s")
    print(f"   Total 3D Velocity RMSE  : {rmse_vel_total:8.3f} m/s (Steady-State: {ss_rmse_vel:.3f} m/s)")
    print("=" * 78)

    # 6. Visualization (3 Subplots: Positions, Velocities, Errors & 3-Sigma)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")

    # ---------------- Subplot 1: 3D Positions (x, y, z) ----------------
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    # Ground Truth
    ax1.plot(t_arr, gt_arr[:, 0], "k-", linewidth=2.2, label=r"Ground Truth $X$ (North)", alpha=0.9)
    ax1.plot(t_arr, gt_arr[:, 2], "k--", linewidth=2.0, label=r"Ground Truth $Y$ (East)", alpha=0.9)
    ax1.plot(t_arr, gt_arr[:, 4], "k:", linewidth=2.0, label=r"Ground Truth $Z$ (Down)", alpha=0.9)

    # Raw Measurements (subsampled for clarity)
    step = 4
    ax1.scatter(t_raw[::step], raw_pos_arr[::step, 0], s=8, color="#E74C3C", alpha=0.35, label="Raw $X$")
    ax1.scatter(t_raw[::step], raw_pos_arr[::step, 1], s=8, color="#2ECC71", alpha=0.35, label="Raw $Y$")
    ax1.scatter(t_raw[::step], raw_pos_arr[::step, 2], s=8, color="#3498DB", alpha=0.35, label="Raw $Z$")

    # EKF Estimates
    ax1.plot(t_arr, ekf_arr[:, 0], color="#C0392B", linewidth=1.8, label=r"EKF $\hat{X}$")
    ax1.plot(t_arr, ekf_arr[:, 2], color="#27AE60", linewidth=1.8, label=r"EKF $\hat{Y}$")
    ax1.plot(t_arr, ekf_arr[:, 4], color="#2980B9", linewidth=1.8, label=r"EKF $\hat{Z}$")

    ax1.set_title("1. Hedef Konum Takibi: Yer Gerçeği, Ham Ölçüm ve Asenkron EKF Kestirimi (NED)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Konum (m)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)

    # ---------------- Subplot 2: 3D Velocities (vx, vy, vz) ----------------
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    # True Velocities
    ax2.plot(t_arr, gt_arr[:, 1], "r--", linewidth=2.0, label=r"True $V_x$ (North)", alpha=0.8)
    ax2.plot(t_arr, gt_arr[:, 3], "g--", linewidth=2.0, label=r"True $V_y$ (East)", alpha=0.8)
    ax2.plot(t_arr, gt_arr[:, 5], "b--", linewidth=2.0, label=r"True $V_z$ (Down)", alpha=0.8)

    # EKF Velocity Estimates
    ax2.plot(t_arr, ekf_arr[:, 1], color="#922B21", linewidth=1.8, label=r"EKF $\hat{V}_x$")
    ax2.plot(t_arr, ekf_arr[:, 3], color="#1E8449", linewidth=1.8, label=r"EKF $\hat{V}_y$")
    ax2.plot(t_arr, ekf_arr[:, 5], color="#1F618D", linewidth=1.8, label=r"EKF $\hat{V}_z$")

    ax2.set_title("2. Hız Vektörü Yakınsaması (Doppler Olmayan Ölçümlerden Türetilen Hız)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Hız (m/s)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.9)

    # ---------------- Subplot 3: Position Errors & 3-Sigma Bounds ----------------
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    sigma_x = np.sqrt(np.maximum(cov_arr[:, 0], 0.0))
    sigma_y = np.sqrt(np.maximum(cov_arr[:, 2], 0.0))
    sigma_z = np.sqrt(np.maximum(cov_arr[:, 4], 0.0))

    # Plot errors
    ax3.plot(t_arr, error_state[:, 0], color="#C0392B", label=r"Hata $e_x$ ($X - \hat{X}$)")
    ax3.plot(t_arr, error_state[:, 2], color="#27AE60", label=r"Hata $e_y$ ($Y - \hat{Y}$)")
    ax3.plot(t_arr, error_state[:, 4], color="#2980B9", label=r"Hata $e_z$ ($Z - \hat{Z}$)")

    # 3-sigma bounds for X (representative)
    ax3.fill_between(t_arr, -3 * sigma_x, 3 * sigma_x, color="#C0392B", alpha=0.12, label=r"$\pm 3\sigma_x$ Kovaryans Zarfı")
    ax3.fill_between(t_arr, -3 * sigma_y, 3 * sigma_y, color="#27AE60", alpha=0.10, label=r"$\pm 3\sigma_y$ Kovaryans Zarfı")

    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.set_title(f"3. Konum Hataları ve $3\\sigma$ Filtre Tutarlılık Sınırları (Kalıcı Durum RMSE: {ss_rmse_pos:.2f} m)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Zaman (s)", fontsize=11)
    ax3.set_ylabel("Hata (m)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n[Visualization] Yüksek çözünürlüklü grafik kaydedildi -> '{save_fig_path}'")

    if show_plot:
        plt.show()

    return {
        "rmse_pos_total": rmse_pos_total,
        "ss_rmse_pos": ss_rmse_pos,
        "rmse_vel_total": rmse_vel_total,
        "ss_rmse_vel": ss_rmse_vel,
        "raw_rmse_total": raw_rmse_total,
    }


if __name__ == "__main__":
    run_simulation(sim_duration=20.0, show_plot=False)
