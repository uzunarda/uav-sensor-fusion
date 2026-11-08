"""
Phase 2 Benchmark and Validation Script for UAV Asynchronous Sensor Fusion.
Compares 3 Operational Modes:
  - Mode A (Ideal - Phase 1): Zero-delay asynchronous fusion (t_receive == t_capture).
  - Mode B (Naive EKF): Latencies, jitter & clock bias present, but filter assumes t_receive is capture epoch.
  - Mode C (Delay-Compensated EKF): Latencies, jitter & clock bias present, Rollback / Buffer-Replay active.

Computes 3D Position & Velocity RMSE, analyzes OOSM events, and generates comprehensive plots.
"""

from __future__ import annotations
import os
import sys

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import TargetState, ConstantVelocityTrajectory
from src.sensors import RadarSensor, CameraSensor
from src.simulator import AsyncSensorEventSimulator
from src.ekf import ExtendedKalmanFilter


def run_benchmark(
    sim_duration: float = 20.0,
    seed: int = 42,
    save_fig_path: str = "results/ekf_oosm_comparison.png",
    show_plot: bool = False,
) -> dict:
    """
    Executes identical simulation scenario under Modes A, B, and C.

    Args:
        sim_duration: Simulation duration in seconds (default: 20.0s).
        seed: Random generator seed for deterministic reproducibility.
        save_fig_path: Path to save the multi-panel comparison figure.
        show_plot: Whether to display interactive plot window.

    Returns:
        dict: Detailed performance metrics for Modes A, B, and C.
    """
    print("=" * 84)
    print("      UAV ASYNCHRONOUS SENSOR FUSION - PHASE 2: OOSM & LATENCY BENCHMARK")
    print("=" * 84)

    # 1. Common Scenario & Target Kinematics (NED Frame)
    true_initial_state = TargetState(
        x=1200.0,
        vx=35.0,     # 35 m/s North (~126 km/h)
        y=-800.0,
        vy=25.0,     # 25 m/s East (~90 km/h)
        z=-500.0,    # 500m Altitude in NED (z is negative downwards)
        vz=-4.0,     # Climbing at 4 m/s
    )
    trajectory = ConstantVelocityTrajectory(true_initial_state, t0=0.0)
    uav_pos = np.array([0.0, 0.0, -150.0], dtype=np.float64)

    # Common filter initialization parameters
    initial_offset = np.array([80.0, -15.0, -90.0, 10.0, 40.0, 6.0])
    x_init_guess = true_initial_state.to_vector() + initial_offset
    P_init = np.diag([
        150.0**2, 20.0**2,
        150.0**2, 20.0**2,
        100.0**2, 15.0**2,
    ])
    q_continuous = 0.8  # Process noise acceleration PSD (m^2/s^3)

    print(f"[Dynamics] Hedef Başlangıç Konumu: [{true_initial_state.x:.0f}, {true_initial_state.y:.0f}, {true_initial_state.z:.0f}] m")
    print(f"           Hedef Hız Vektörü     : [{true_initial_state.vx:.1f}, {true_initial_state.vy:.1f}, {true_initial_state.vz:.1f}] m/s")
    print(f"[Platform] İHA Referans Konumu   : Kuzey={uav_pos[0]:.0f}m, Doğu={uav_pos[1]:.0f}m, İrtifa={-uav_pos[2]:.0f}m")
    print("-" * 84)

    # -------------------------------------------------------------------------
    # MODE A: IDEAL (Zero Latency, Zero Jitter, Zero Clock Bias)
    # -------------------------------------------------------------------------
    rng_a = np.random.default_rng(seed)
    radar_a = RadarSensor(frequency_hz=10.0, sigma_r=6.0, latency_mean=0.0, latency_jitter_std=0.0, clock_bias=0.0, rng=rng_a)
    camera_a = CameraSensor(frequency_hz=30.0, sigma_psi=0.004, sigma_lambda=0.004, latency_mean=0.0, latency_jitter_std=0.0, clock_bias=0.0, rng=rng_a)

    sim_a = AsyncSensorEventSimulator(trajectory)
    sim_a.register_sensor(radar_a, phase_offset=0.0)
    sim_a.register_sensor(camera_a, phase_offset=0.0125)
    sim_a.populate_events(t_start=0.0, t_end=sim_duration, uav_position_fn=lambda t: uav_pos)

    ekf_a = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=False,
    )

    log_a_time, log_a_err_pos, log_a_err_vel, log_a_state, log_a_gt = [], [], [], [], []
    for event in sim_a:
        ekf_a.update(event, uav_pos=uav_pos, is_naive=False)
        t_now = ekf_a.current_time
        gt = trajectory.get_state(t_now).to_vector()
        est = ekf_a.x.flatten().copy()
        err = est - gt
        pos_err = float(np.linalg.norm(err[[0, 2, 4]]))
        vel_err = float(np.linalg.norm(err[[1, 3, 5]]))

        log_a_time.append(t_now)
        log_a_err_pos.append(pos_err)
        log_a_err_vel.append(vel_err)
        log_a_state.append(est)
        log_a_gt.append(gt)

    print(f"[Mod A - İdeal]           {len(log_a_time)} olay işlendi (Sıfır gecikmeli asenkron füzyon tamamlandı).")

    # -------------------------------------------------------------------------
    # MODES B & C: SHARED REALISTIC SENSOR STREAM (Latencies, Jitter & Clock Bias)
    #   - Radar: 8 ms nominal transmission latency + N(0, 2ms) Gaussian jitter
    #   - Camera: 40 ms nominal image processing latency + N(0, 5ms) Gaussian jitter
    #   - Camera Clock Bias: +25 ms constant hardware clock offset
    # -------------------------------------------------------------------------
    rng_bc = np.random.default_rng(seed)
    radar_delayed = RadarSensor(
        frequency_hz=10.0,
        sigma_r=6.0,
        latency_mean=0.008,          # 8 ms
        latency_jitter_std=0.002,    # 2 ms
        clock_bias=0.0,
        rng=rng_bc,
    )
    camera_delayed = CameraSensor(
        frequency_hz=30.0,
        sigma_psi=0.004,
        sigma_lambda=0.004,
        latency_mean=0.040,          # 40 ms
        latency_jitter_std=0.005,    # 5 ms
        clock_bias=0.025,            # +25 ms hardware clock bias
        rng=rng_bc,
    )

    sim_delayed = AsyncSensorEventSimulator(trajectory)
    sim_delayed.register_sensor(radar_delayed, phase_offset=0.0)
    sim_delayed.register_sensor(camera_delayed, phase_offset=0.0125)
    num_events_delayed = sim_delayed.populate_events(
        t_start=0.0,
        t_end=sim_duration,
        uav_position_fn=lambda t: uav_pos,
    )

    # Clone exact stream so Modes B and C process the identical real-world sequence
    events_mode_b = sim_delayed.clone_event_stream()
    events_mode_c = sim_delayed.clone_event_stream()

    print(f"[Mod B & C - Gecikmeli]   {num_events_delayed} olay üretildi:")
    print("                          * Radar: 8ms gecikme ± 2ms jitter")
    print("                          * Kamera: 40ms gecikme ± 5ms jitter, +25ms saat kayması (clock bias)")

    # -------------------------------------------------------------------------
    # MODE B: NAIVE EKF (Assumes processor arrival t_receive is capture epoch)
    # -------------------------------------------------------------------------
    ekf_b = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=False,
    )

    log_b_time, log_b_err_pos, log_b_err_vel, log_b_state, log_b_gt = [], [], [], [], []
    for event in events_mode_b:
        ekf_b.update(event, uav_pos=uav_pos, is_naive=True)
        t_eval = event.t_receive  # In naive mode, filter moves along arrival timeline
        gt = trajectory.get_state(t_eval).to_vector()
        est = ekf_b.x.flatten().copy()
        err = est - gt
        pos_err = float(np.linalg.norm(err[[0, 2, 4]]))
        vel_err = float(np.linalg.norm(err[[1, 3, 5]]))

        log_b_time.append(t_eval)
        log_b_err_pos.append(pos_err)
        log_b_err_vel.append(vel_err)
        log_b_state.append(est)
        log_b_gt.append(gt)

    print(f"[Mod B - Naive EKF]       {len(log_b_time)} olay işlendi (Gecikmeler ihmal edildi / varış anı capture sanıldı).")

    # -------------------------------------------------------------------------
    # MODE C: DELAY-COMPENSATED EKF (Rollback / Buffer-Replay Active)
    # -------------------------------------------------------------------------
    ekf_c = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=True,
        max_buffer_size=150,
        max_buffer_dt=1.0,
    )

    log_c_time, log_c_err_pos, log_c_err_vel, log_c_state, log_c_gt = [], [], [], [], []
    log_delays = []
    log_oosm_events = []

    for event in events_mode_c:
        is_oosm = event.t_capture < ekf_c.current_time - 1e-9
        if is_oosm:
            log_oosm_events.append((event.t_receive, event.t_capture, event.sensor_type))

        log_delays.append((event.t_receive, event.t_receive - event.t_capture, event.sensor_type))

        ekf_c.update(event, uav_pos=uav_pos, is_naive=False)

        # Evaluate filter state against ground truth at current filter epoch
        t_eval = ekf_c.current_time
        gt = trajectory.get_state(t_eval).to_vector()
        est = ekf_c.x.flatten().copy()
        err = est - gt
        pos_err = float(np.linalg.norm(err[[0, 2, 4]]))
        vel_err = float(np.linalg.norm(err[[1, 3, 5]]))

        log_c_time.append(t_eval)
        log_c_err_pos.append(pos_err)
        log_c_err_vel.append(vel_err)
        log_c_state.append(est)
        log_c_gt.append(gt)

    print(f"[Mod C - Telafili EKF]    {len(log_c_time)} olay işlendi:")
    print(f"                          * Başarıyla yönetilen OOSM sayısı : {ekf_c.oosm_count}")
    print(f"                          * Tampon dışı atılan (dropped)    : {ekf_c.dropped_oosm_count}")

    # -------------------------------------------------------------------------
    # METRICS EVALUATION & STATISTICAL SUMMARY
    # -------------------------------------------------------------------------
    def calc_metrics(times, err_pos_list, err_vel_list, states, gts):
        t_arr = np.array(times)
        p_err = np.array(err_pos_list)
        v_err = np.array(err_vel_list)
        st_arr = np.array(states)
        gt_arr = np.array(gts)
        diff = st_arr - gt_arr

        # Steady-state window (t >= 3.0s)
        ss_mask = t_arr >= 3.0
        ss_p_err = p_err[ss_mask] if np.any(ss_mask) else p_err
        ss_v_err = v_err[ss_mask] if np.any(ss_mask) else v_err
        ss_diff = diff[ss_mask] if np.any(ss_mask) else diff

        return {
            "total_pos_rmse": float(np.sqrt(np.mean(p_err**2))),
            "ss_pos_rmse": float(np.sqrt(np.mean(ss_p_err**2))),
            "ss_rmse_x": float(np.sqrt(np.mean(ss_diff[:, 0] ** 2))),
            "ss_rmse_y": float(np.sqrt(np.mean(ss_diff[:, 2] ** 2))),
            "ss_rmse_z": float(np.sqrt(np.mean(ss_diff[:, 4] ** 2))),
            "total_vel_rmse": float(np.sqrt(np.mean(v_err**2))),
            "ss_vel_rmse": float(np.sqrt(np.mean(ss_v_err**2))),
            "diff": diff,
            "t_arr": t_arr,
        }

    met_a = calc_metrics(log_a_time, log_a_err_pos, log_a_err_vel, log_a_state, log_a_gt)
    met_b = calc_metrics(log_b_time, log_b_err_pos, log_b_err_vel, log_b_state, log_b_gt)
    met_c = calc_metrics(log_c_time, log_c_err_pos, log_c_err_vel, log_c_state, log_c_gt)

    pos_improvement_vs_naive = (1.0 - met_c["ss_pos_rmse"] / met_b["ss_pos_rmse"]) * 100.0
    vel_improvement_vs_naive = (1.0 - met_c["ss_vel_rmse"] / met_b["ss_vel_rmse"]) * 100.0

    print("\n" + "=" * 84)
    print("            PROJE 2. AŞAMA: PERFORMANS VE KIYASLAMA TABLOSU")
    print("=" * 84)
    col_w = [34, 15, 15, 16]
    print(f"{'Metrik / Parametre':<{col_w[0]}} | {'Mod A (İdeal)':<{col_w[1]}} | {'Mod B (Naive)':<{col_w[2]}} | {'Mod C (Rollback)':<{col_w[3]}}")
    print("-" * 84)
    print(f"{'Kalıcı Durum 3B Konum RMSE (m)':<{col_w[0]}} | {met_a['ss_pos_rmse']:<{col_w[1]}.3f} | {met_b['ss_pos_rmse']:<{col_w[2]}.3f} | {met_c['ss_pos_rmse']:<{col_w[3]}.3f}")
    print(f"{'  - Eksen X (Kuzey) RMSE (m)':<{col_w[0]}} | {met_a['ss_rmse_x']:<{col_w[1]}.3f} | {met_b['ss_rmse_x']:<{col_w[2]}.3f} | {met_c['ss_rmse_x']:<{col_w[3]}.3f}")
    print(f"{'  - Eksen Y (Doğu) RMSE (m)':<{col_w[0]}} | {met_a['ss_rmse_y']:<{col_w[1]}.3f} | {met_b['ss_rmse_y']:<{col_w[2]}.3f} | {met_c['ss_rmse_y']:<{col_w[3]}.3f}")
    print(f"{'  - Eksen Z (İrtifa) RMSE (m)':<{col_w[0]}} | {met_a['ss_rmse_z']:<{col_w[1]}.3f} | {met_b['ss_rmse_z']:<{col_w[2]}.3f} | {met_c['ss_rmse_z']:<{col_w[3]}.3f}")
    print(f"{'Toplam 3B Konum RMSE (m)':<{col_w[0]}} | {met_a['total_pos_rmse']:<{col_w[1]}.3f} | {met_b['total_pos_rmse']:<{col_w[2]}.3f} | {met_c['total_pos_rmse']:<{col_w[3]}.3f}")
    print("-" * 84)
    print(f"{'Kalıcı Durum 3B Hız RMSE (m/s)':<{col_w[0]}} | {met_a['ss_vel_rmse']:<{col_w[1]}.3f} | {met_b['ss_vel_rmse']:<{col_w[2]}.3f} | {met_c['ss_vel_rmse']:<{col_w[3]}.3f}")
    print(f"{'Toplam 3B Hız RMSE (m/s)':<{col_w[0]}} | {met_a['total_vel_rmse']:<{col_w[1]}.3f} | {met_b['total_vel_rmse']:<{col_w[2]}.3f} | {met_c['total_vel_rmse']:<{col_w[3]}.3f}")
    print("-" * 84)
    print(f"{'İşlenen OOSM Olay Sayısı':<{col_w[0]}} | {'0 (Gecikmesiz)':<{col_w[1]}} | {'0 (Yönetilemez)':<{col_w[2]}} | {ekf_c.oosm_count:<{col_w[3]}}")
    print(f"{'Atılan (Dropped) OOSM Sayısı':<{col_w[0]}} | {'0':<{col_w[1]}} | {'0':<{col_w[2]}} | {ekf_c.dropped_oosm_count:<{col_w[3]}}")
    lbl_imp = "Mod B (Naive) Moduna Göre İyileşme"
    print(f"{lbl_imp:<{col_w[0]}} | {'-':<{col_w[1]}} | {'Referans':<{col_w[2]}} | %{pos_improvement_vs_naive:<{col_w[3]-1}.1f}")
    print("=" * 84)

    # -------------------------------------------------------------------------
    # MULTI-PANEL VISUALIZATION (4 SUBPLOTS)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.patch.set_facecolor("#F8F9FA")

    c_ideal = "#1F77B4"       # Deep Blue
    c_naive = "#D62728"       # Crimson Red
    c_replay = "#2CA02C"      # Forest Green
    c_radar = "#FF7F0E"       # Amber
    c_cam = "#9467BD"         # Purple

    # Subplot 1: 3D Position Tracking Error
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(log_b_time, log_b_err_pos, color=c_naive, linewidth=1.8, alpha=0.85,
             label=f"Mod B (Naive EKF) - Kalıcı RMSE: {met_b['ss_pos_rmse']:.2f} m")
    ax1.plot(log_c_time, log_c_err_pos, color=c_replay, linewidth=2.0, alpha=0.9,
             label=f"Mod C (Rollback/Replay EKF) - Kalıcı RMSE: {met_c['ss_pos_rmse']:.2f} m")
    ax1.plot(log_a_time, log_a_err_pos, color=c_ideal, linewidth=1.5, linestyle="--", alpha=0.8,
             label=f"Mod A (İdeal Sıfır Gecikme) - Kalıcı RMSE: {met_a['ss_pos_rmse']:.2f} m")

    ax1.set_title("1. 3B Konum Kestirim Hatası: İdeal vs. Naive vs. Rollback/Replay Karşılaştırması",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("3B Hata (m)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax1.set_ylim(0, max(20.0, min(60.0, float(np.percentile(log_b_err_pos, 98)))))

    # Subplot 2: Axial Position Errors (X, Y, Z for Mode B vs Mode C)
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    diff_b = met_b["diff"]
    diff_c = met_c["diff"]
    ax2.plot(met_b["t_arr"], diff_b[:, 0], color="#E74C3C", linestyle=":", linewidth=1.4, label=r"Naive $e_x$ (Kuzey)")
    ax2.plot(met_b["t_arr"], diff_b[:, 2], color="#E67E22", linestyle=":", linewidth=1.4, label=r"Naive $e_y$ (Doğu)")
    ax2.plot(met_c["t_arr"], diff_c[:, 0], color="#27AE60", linestyle="-", linewidth=1.6, label=r"Rollback $e_x$ (Kuzey)")
    ax2.plot(met_c["t_arr"], diff_c[:, 2], color="#2980B9", linestyle="-", linewidth=1.6, label=r"Rollback $e_y$ (Doğu)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    ax2.set_title("2. Eksenel Konum Hataları (X ve Y Eksenlerinde Gecikme Kaynaklı Kaymanın Telafisi)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Eksenel Hata (m)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.95)

    # Subplot 3: 3D Velocity Error
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(log_b_time, log_b_err_vel, color=c_naive, linewidth=1.8, alpha=0.85,
             label=f"Mod B (Naive EKF) - Kalıcı RMSE: {met_b['ss_vel_rmse']:.2f} m/s")
    ax3.plot(log_c_time, log_c_err_vel, color=c_replay, linewidth=2.0, alpha=0.9,
             label=f"Mod C (Rollback/Replay EKF) - Kalıcı RMSE: {met_c['ss_vel_rmse']:.2f} m/s")
    ax3.plot(log_a_time, log_a_err_vel, color=c_ideal, linewidth=1.5, linestyle="--", alpha=0.8,
             label=f"Mod A (İdeal Sıfır Gecikme) - Kalıcı RMSE: {met_a['ss_vel_rmse']:.2f} m/s")

    ax3.set_title("3. 3B Hız Kestirim Hatası ve Yakınsama Performansı", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Hız Hatası (m/s)", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax3.set_ylim(0, max(8.0, min(20.0, float(np.percentile(log_b_err_vel, 98)))))

    # Subplot 4: Latency Distribution and OOSM Arrival Events
    ax4 = axes[3]
    ax4.set_facecolor("#FFFFFF")
    radar_delays = [(t, d * 1000.0) for t, d, s in log_delays if s == "RADAR"]
    camera_delays = [(t, d * 1000.0) for t, d, s in log_delays if s == "CAMERA"]

    if radar_delays:
        rt, rd = zip(*radar_delays)
        ax4.scatter(rt, rd, color=c_radar, s=16, alpha=0.65, label="Radar İletim Gecikmesi (~8 ms + 2ms Jitter)", marker="o")
    if camera_delays:
        ct, cd = zip(*camera_delays)
        ax4.scatter(ct, cd, color=c_cam, s=14, alpha=0.65, label="Kamera İşleme Gecikmesi (~40 ms + 5ms Jitter)", marker="^")

    if log_oosm_events:
        oosm_t, oosm_c, _ = zip(*log_oosm_events)
        ax4.scatter(oosm_t, [(ot - oc) * 1000.0 for ot, oc in zip(oosm_t, oosm_c)],
                    color="#D62728", s=35, marker="x", linewidth=1.5,
                    label=f"Algılanan OOSM Olayları (Toplam: {len(log_oosm_events)} adet)")

    ax4.set_title("4. Sensör İletim/İşleme Gecikmeleri ve Algılanan Sıra Dışı Ölçüm (OOSM) Dağılımı",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Simülasyon Zamanı (s)", fontsize=11)
    ax4.set_ylabel("Gecikme (ms)", fontsize=11)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=10, framealpha=0.95)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n[Görselleştirme] Kıyaslama grafiği kaydedildi -> '{save_fig_path}'")

    alt_save_path = "results/ekf_fusion_results.png"
    plt.savefig(alt_save_path, dpi=300, bbox_inches="tight")
    print(f"[Görselleştirme] Ana sonuç grafiği güncellendi -> '{alt_save_path}'")

    if show_plot:
        plt.show()

    return {
        "mode_a": met_a,
        "mode_b": met_b,
        "mode_c": met_c,
        "pos_improvement_vs_naive": pos_improvement_vs_naive,
        "vel_improvement_vs_naive": vel_improvement_vs_naive,
        "oosm_count": ekf_c.oosm_count,
        "dropped_oosm_count": ekf_c.dropped_oosm_count,
    }


if __name__ == "__main__":
    run_benchmark(sim_duration=20.0, show_plot=False)
