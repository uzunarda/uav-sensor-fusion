"""
Phase 2 & Literature OOSM Methods Benchmark and Validation Script for UAV Asynchronous Sensor Fusion.
Compares 5 Operational Modes across 7 Literature Performance Metrics:
  - Mode A (Ideal - Reference): Zero-delay asynchronous fusion (t_receive == t_capture).
  - Mode B (Naive EKF): Latencies, jitter & clock bias present, filter assumes t_receive is capture epoch.
  - Mode C (OOSM Rollback/Replay - Paper 2): Circular state buffer rewind & chronological replay.
  - Mode D (OOSM 1-Step Retrodiction / Modified EKF - Paper 3): Direct delay-embedded observation matrix H_eff = H_d * F(-dt).
  - Mode E (OOSM Kalman Predictor / Extrapolation - Paper 4): Forward measurement extrapolation to current epoch.

Evaluates:
  1. 3D Position & Velocity RMSE (m, m/s) + Axial RMSE
  2. MAE (Mean Absolute Error, m)
  3. OSPA (Optimal SubPattern Assignment Metric, m)
  4. NEES & NIS (Normalized Estimation / Innovation Error Squared - Filter Consistency)
  5. Track Continuity (%) & IDF1 Score (Identity Preservation)
  6. Execution Time (Total ms & Per-Packet us)
"""

from __future__ import annotations
import os
import sys
import time

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
from src.metrics import evaluate_Comprehensive_metrics


def run_benchmark(
    sim_duration: float = 20.0,
    seed: int = 42,
    save_fig_path: str = "results/ekf_oosm_comparison.png",
    show_plot: bool = False,
) -> dict:
    """
    Executes identical simulation scenario under Modes A, B, C, D, and E.

    Args:
        sim_duration: Simulation duration in seconds (default: 20.0s).
        seed: Random generator seed for deterministic reproducibility.
        save_fig_path: Path to save the multi-panel comparison figure.
        show_plot: Whether to display interactive plot window.

    Returns:
        dict: Detailed performance metrics for Modes A, B, C, D, and E.
    """
    print("=" * 108)
    print("   UAV ASYNCHRONOUS SENSOR FUSION - MULTI-METHOD OOSM & LITERATURE METRICS BENCHMARK (MODES A-E)")
    print("=" * 108)

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
    print("-" * 108)

    def _execute_filter_run(events_stream, ekf_instance: ExtendedKalmanFilter, is_naive: bool = False):
        log_time, log_err_pos, log_err_vel, log_state, log_gt, log_cov, log_nis = [], [], [], [], [], [], []
        pure_filter_ns = 0
        for event in events_stream:
            t0_ns = time.perf_counter_ns()
            ekf_instance.update(event, uav_pos=uav_pos, is_naive=is_naive)
            pure_filter_ns += time.perf_counter_ns() - t0_ns

            t_eval = event.t_receive if is_naive else ekf_instance.current_time
            gt = trajectory.get_state(t_eval).to_vector()
            est = ekf_instance.x.flatten().copy()
            cov = ekf_instance.P.copy()
            err = est - gt
            pos_err = float(np.linalg.norm(err[[0, 2, 4]]))
            vel_err = float(np.linalg.norm(err[[1, 3, 5]]))

            log_time.append(t_eval)
            log_err_pos.append(pos_err)
            log_err_vel.append(vel_err)
            log_state.append(est)
            log_gt.append(gt)
            log_cov.append(cov)
            log_nis.append(ekf_instance.last_nis)

        exec_ms = pure_filter_ns / 1e6
        return evaluate_Comprehensive_metrics(
            times=log_time,
            err_pos_list=log_err_pos,
            err_vel_list=log_err_vel,
            states_est=log_state,
            states_gt=log_gt,
            covs=log_cov,
            nis_list=log_nis,
            exec_time_ms=exec_ms,
        )

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
    events_mode_a = sim_a.clone_event_stream()

    ekf_a = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=False,
    )
    met_a = _execute_filter_run(events_mode_a, ekf_a, is_naive=False)
    print(f"[Mod A - İdeal (0ms)]             {len(events_mode_a)} olay işlendi | Süre: {met_a['exec_time_ms']:.2f} ms")

    # -------------------------------------------------------------------------
    # SHARED REALISTIC SENSOR STREAM FOR MODES B, C, D, E
    # -------------------------------------------------------------------------
    rng_delayed = np.random.default_rng(seed)
    radar_delayed = RadarSensor(
        frequency_hz=10.0,
        sigma_r=6.0,
        latency_mean=0.008,          # 8 ms
        latency_jitter_std=0.002,    # 2 ms
        clock_bias=0.0,
        rng=rng_delayed,
    )
    camera_delayed = CameraSensor(
        frequency_hz=30.0,
        sigma_psi=0.004,
        sigma_lambda=0.004,
        latency_mean=0.040,          # 40 ms
        latency_jitter_std=0.005,    # 5 ms
        clock_bias=0.025,            # +25 ms hardware clock bias
        rng=rng_delayed,
    )

    sim_delayed = AsyncSensorEventSimulator(trajectory)
    sim_delayed.register_sensor(radar_delayed, phase_offset=0.0)
    sim_delayed.register_sensor(camera_delayed, phase_offset=0.0125)
    num_events_delayed = sim_delayed.populate_events(
        t_start=0.0,
        t_end=sim_duration,
        uav_position_fn=lambda t: uav_pos,
    )

    events_mode_b = sim_delayed.clone_event_stream()
    events_mode_c = sim_delayed.clone_event_stream()
    events_mode_d = sim_delayed.clone_event_stream()
    events_mode_e = sim_delayed.clone_event_stream()

    # Record latency distribution and OOSM arrival epochs
    log_delays = []
    log_oosm_events = []
    max_cap_seen = 0.0
    for ev in events_mode_c:
        if ev.t_capture < max_cap_seen - 1e-9:
            log_oosm_events.append((ev.t_receive, ev.t_capture, ev.sensor_type))
        else:
            max_cap_seen = ev.t_capture
        log_delays.append((ev.t_receive, ev.t_receive - ev.t_capture, ev.sensor_type))

    # -------------------------------------------------------------------------
    # MODE B: NAIVE EKF (Delay-Ignorant)
    # -------------------------------------------------------------------------
    ekf_b = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=False,
    )
    met_b = _execute_filter_run(events_mode_b, ekf_b, is_naive=True)
    print(f"[Mod B - Naive EKF (Telafisiz)]   {len(events_mode_b)} olay işlendi | Süre: {met_b['exec_time_ms']:.2f} ms")

    # -------------------------------------------------------------------------
    # MODE C: OOSM ROLLBACK / BUFFER-REPLAY EKF (Paper 2)
    # -------------------------------------------------------------------------
    ekf_c = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=True,
        max_buffer_size=150,
        max_buffer_dt=1.0,
        oosm_strategy="ROLLBACK",
    )
    met_c = _execute_filter_run(events_mode_c, ekf_c, is_naive=False)
    print(f"[Mod C - OOSM Rollback/Replay]    {len(events_mode_c)} olay işlendi | OOSM: {ekf_c.oosm_count} | Süre: {met_c['exec_time_ms']:.2f} ms")

    # -------------------------------------------------------------------------
    # MODE D: OOSM 1-STEP RETRODICTION / MODIFIED EKF (Paper 3)
    # -------------------------------------------------------------------------
    ekf_d = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=True,
        max_buffer_size=150,
        max_buffer_dt=1.0,
        oosm_strategy="RETRODICTION",
    )
    met_d = _execute_filter_run(events_mode_d, ekf_d, is_naive=False)
    print(f"[Mod D - OOSM Modifiye EKF]       {len(events_mode_d)} olay işlendi | OOSM: {ekf_d.oosm_count} | Süre: {met_d['exec_time_ms']:.2f} ms")

    # -------------------------------------------------------------------------
    # MODE E: OOSM KALMAN PREDICTOR / MEASUREMENT EXTRAPOLATION (Paper 4)
    # -------------------------------------------------------------------------
    ekf_e = ExtendedKalmanFilter(
        initial_state=x_init_guess.copy(),
        initial_covariance=P_init.copy(),
        process_noise_spectral_density=q_continuous,
        enable_oosm_buffer=True,
        max_buffer_size=150,
        max_buffer_dt=1.0,
        oosm_strategy="EXTRAPOLATION",
    )
    met_e = _execute_filter_run(events_mode_e, ekf_e, is_naive=False)
    print(f"[Mod E - OOSM Kalman Predictor]   {len(events_mode_e)} olay işlendi | OOSM: {ekf_e.oosm_count} | Süre: {met_e['exec_time_ms']:.2f} ms")

    pos_improvement_c = (1.0 - met_c["ss_pos_rmse"] / met_b["ss_pos_rmse"]) * 100.0
    pos_improvement_d = (1.0 - met_d["ss_pos_rmse"] / met_b["ss_pos_rmse"]) * 100.0
    pos_improvement_e = (1.0 - met_e["ss_pos_rmse"] / met_b["ss_pos_rmse"]) * 100.0
    vel_improvement_vs_naive = (1.0 - met_c["ss_vel_rmse"] / met_b["ss_vel_rmse"]) * 100.0

    # -------------------------------------------------------------------------
    # CONSOLE BENCHMARK TABLE (5 MODES x ALL METRICS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 116)
    print("                 PROJE KAPSAMLI OOSM YÖNTEMLERİ VE LİTERATÜR BAŞARI METRİKLERİ TABLOSU")
    print("=" * 116)
    cw = [34, 13, 13, 15, 15, 14]
    hdr = (
        f"{'Metrik / Parametre':<{cw[0]}} | "
        f"{'Mod A (İdeal)':<{cw[1]}} | "
        f"{'Mod B (Naive)':<{cw[2]}} | "
        f"{'Mod C (Rollback)':<{cw[3]}} | "
        f"{'Mod D (Mod-EKF)':<{cw[4]}} | "
        f"{'Mod E (Predict)':<{cw[5]}}"
    )
    print(hdr)
    print("-" * 116)

    def _print_row(label: str, va: str, vb: str, vc: str, vd: str, ve: str) -> None:
        print(f"{label:<{cw[0]}} | {va:<{cw[1]}} | {vb:<{cw[2]}} | {vc:<{cw[3]}} | {vd:<{cw[4]}} | {ve:<{cw[5]}}")

    _print_row(
        "Kalıcı Durum 3B Konum RMSE (m)",
        f"{met_a['ss_pos_rmse']:.3f}", f"{met_b['ss_pos_rmse']:.3f}",
        f"{met_c['ss_pos_rmse']:.3f}", f"{met_d['ss_pos_rmse']:.3f}", f"{met_e['ss_pos_rmse']:.3f}"
    )
    _print_row(
        "  - Eksen X (Kuzey) RMSE (m)",
        f"{met_a['ss_rmse_x']:.3f}", f"{met_b['ss_rmse_x']:.3f}",
        f"{met_c['ss_rmse_x']:.3f}", f"{met_d['ss_rmse_x']:.3f}", f"{met_e['ss_rmse_x']:.3f}"
    )
    _print_row(
        "  - Eksen Y (Doğu) RMSE (m)",
        f"{met_a['ss_rmse_y']:.3f}", f"{met_b['ss_rmse_y']:.3f}",
        f"{met_c['ss_rmse_y']:.3f}", f"{met_d['ss_rmse_y']:.3f}", f"{met_e['ss_rmse_y']:.3f}"
    )
    _print_row(
        "  - Eksen Z (İrtifa) RMSE (m)",
        f"{met_a['ss_rmse_z']:.3f}", f"{met_b['ss_rmse_z']:.3f}",
        f"{met_c['ss_rmse_z']:.3f}", f"{met_d['ss_rmse_z']:.3f}", f"{met_e['ss_rmse_z']:.3f}"
    )
    _print_row(
        "Toplam 3B Konum RMSE (m)",
        f"{met_a['total_pos_rmse']:.3f}", f"{met_b['total_pos_rmse']:.3f}",
        f"{met_c['total_pos_rmse']:.3f}", f"{met_d['total_pos_rmse']:.3f}", f"{met_e['total_pos_rmse']:.3f}"
    )
    print("-" * 116)
    _print_row(
        "Kalıcı Durum 3B Konum MAE (m)",
        f"{met_a['ss_pos_mae']:.3f}", f"{met_b['ss_pos_mae']:.3f}",
        f"{met_c['ss_pos_mae']:.3f}", f"{met_d['ss_pos_mae']:.3f}", f"{met_e['ss_pos_mae']:.3f}"
    )
    _print_row(
        "Kalıcı Durum OSPA Skoru (m)",
        f"{met_a['ss_ospa']:.3f}", f"{met_b['ss_ospa']:.3f}",
        f"{met_c['ss_ospa']:.3f}", f"{met_d['ss_ospa']:.3f}", f"{met_e['ss_ospa']:.3f}"
    )
    _print_row(
        "Kalıcı Durum 3B Hız RMSE (m/s)",
        f"{met_a['ss_vel_rmse']:.3f}", f"{met_b['ss_vel_rmse']:.3f}",
        f"{met_c['ss_vel_rmse']:.3f}", f"{met_d['ss_vel_rmse']:.3f}", f"{met_e['ss_vel_rmse']:.3f}"
    )
    print("-" * 116)
    _print_row(
        "Filtre Tutarlılığı NEES (İdeal~6)",
        f"{met_a['ss_nees']:.2f}", f"{met_b['ss_nees']:.2f}",
        f"{met_c['ss_nees']:.2f}", f"{met_d['ss_nees']:.2f}", f"{met_e['ss_nees']:.2f}"
    )
    _print_row(
        "NEES %95 Güven Bandı Uyumu (%)",
        f"%{met_a['nees_consistency_pct']:.1f}", f"%{met_b['nees_consistency_pct']:.1f}",
        f"%{met_c['nees_consistency_pct']:.1f}", f"%{met_d['nees_consistency_pct']:.1f}", f"%{met_e['nees_consistency_pct']:.1f}"
    )
    _print_row(
        "Yenilik Tutarlılığı NIS",
        f"{met_a['ss_nis']:.2f}", f"{met_b['ss_nis']:.2f}",
        f"{met_c['ss_nis']:.2f}", f"{met_d['ss_nis']:.2f}", f"{met_e['ss_nis']:.2f}"
    )
    print("-" * 116)
    _print_row(
        "İz Sürekliliği (Track Cont. %)",
        f"%{met_a['track_continuity_pct']:.1f}", f"%{met_b['track_continuity_pct']:.1f}",
        f"%{met_c['track_continuity_pct']:.1f}", f"%{met_d['track_continuity_pct']:.1f}", f"%{met_e['track_continuity_pct']:.1f}"
    )
    _print_row(
        "Kimlik Koruma IDF1 Skoru",
        f"{met_a['idf1_score']:.3f}", f"{met_b['idf1_score']:.3f}",
        f"{met_c['idf1_score']:.3f}", f"{met_d['idf1_score']:.3f}", f"{met_e['idf1_score']:.3f}"
    )
    _print_row(
        "İşlenen OOSM Olay Sayısı",
        "0 (0ms)", "0 (Telafisiz)",
        f"{ekf_c.oosm_count}", f"{ekf_d.oosm_count}", f"{ekf_e.oosm_count}"
    )
    _print_row(
        "Döngü Süresi (us/pkt | Toplam ms)",
        f"{met_a['exec_us_per_pkt']:.1f}us ({met_a['exec_time_ms']:.1f}ms)",
        f"{met_b['exec_us_per_pkt']:.1f}us ({met_b['exec_time_ms']:.1f}ms)",
        f"{met_c['exec_us_per_pkt']:.1f}us ({met_c['exec_time_ms']:.1f}ms)",
        f"{met_d['exec_us_per_pkt']:.1f}us ({met_d['exec_time_ms']:.1f}ms)",
        f"{met_e['exec_us_per_pkt']:.1f}us ({met_e['exec_time_ms']:.1f}ms)",
    )
    _print_row(
        "Mod B (Naive)'ye Göre İyileşme",
        "-", "Referans",
        f"%{pos_improvement_c:.1f}", f"%{pos_improvement_d:.1f}", f"%{pos_improvement_e:.1f}"
    )
    print("=" * 116)

    # -------------------------------------------------------------------------
    # MULTI-PANEL VISUALIZATION (4 SUBPLOTS FOR ALL 5 MODES & NEW METRICS)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
    fig.patch.set_facecolor("#F8F9FA")

    c_ideal = "#1F77B4"       # Deep Blue (Mode A)
    c_naive = "#D62728"       # Crimson Red (Mode B)
    c_replay = "#2CA02C"      # Forest Green (Mode C - Rollback)
    c_retro = "#9467BD"       # Royal Purple (Mode D - 1-Step Retrodiction)
    c_extrap = "#FF7F0E"      # Vibrant Amber (Mode E - Kalman Predictor)

    # Subplot 1: 3D Position Tracking Error (RMSE comparison across Modes A-E)
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    ax1.plot(met_b["t_arr"], met_b["p_err"], color=c_naive, linewidth=1.7, alpha=0.80,
             label=f"Mod B (Naive EKF) - RMSE: {met_b['ss_pos_rmse']:.2f}m | MAE: {met_b['ss_pos_mae']:.2f}m")
    ax1.plot(met_e["t_arr"], met_e["p_err"], color=c_extrap, linewidth=1.6, linestyle="-.", alpha=0.85,
             label=f"Mod E (Kalman Predictor / İleri Taşıma) - RMSE: {met_e['ss_pos_rmse']:.2f}m")
    ax1.plot(met_d["t_arr"], met_d["p_err"], color=c_retro, linewidth=1.8, linestyle="--", alpha=0.90,
             label=f"Mod D (Modifiye EKF / 1-Adım Retrodiction) - RMSE: {met_d['ss_pos_rmse']:.2f}m")
    ax1.plot(met_c["t_arr"], met_c["p_err"], color=c_replay, linewidth=2.1, alpha=0.95,
             label=f"Mod C (Rollback/Replay EKF) - RMSE: {met_c['ss_pos_rmse']:.2f}m")
    ax1.plot(met_a["t_arr"], met_a["p_err"], color=c_ideal, linewidth=1.4, linestyle=":", alpha=0.80,
             label=f"Mod A (İdeal Sıfır Gecikme) - RMSE: {met_a['ss_pos_rmse']:.2f}m")

    ax1.set_title("1. 3B Konum Kestirim Hatası (RMSE & MAE): 5 Farklı OOSM / Füzyon Mimarisinin Karşılaştırması",
                  fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("3B Hata (m)", fontsize=10.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    ax1.set_ylim(0, max(20.0, min(60.0, float(np.percentile(met_b["p_err"], 98)))))

    # Subplot 2: OSPA Metric & Track Continuity
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    ax2.plot(met_b["t_arr"], met_b["ospa_arr"], color=c_naive, linewidth=1.6, alpha=0.80,
             label=f"Mod B (Naive) - OSPA: {met_b['ss_ospa']:.2f}m | IDF1: {met_b['idf1_score']:.2f}")
    ax2.plot(met_e["t_arr"], met_e["ospa_arr"], color=c_extrap, linewidth=1.5, linestyle="-.", alpha=0.85,
             label=f"Mod E (Kalman Predictor) - OSPA: {met_e['ss_ospa']:.2f}m | IDF1: {met_e['idf1_score']:.2f}")
    ax2.plot(met_d["t_arr"], met_d["ospa_arr"], color=c_retro, linewidth=1.7, linestyle="--", alpha=0.90,
             label=f"Mod D (Modifiye EKF) - OSPA: {met_d['ss_ospa']:.2f}m | IDF1: {met_d['idf1_score']:.2f}")
    ax2.plot(met_c["t_arr"], met_c["ospa_arr"], color=c_replay, linewidth=2.0, alpha=0.95,
             label=f"Mod C (Rollback/Replay) - OSPA: {met_c['ss_ospa']:.2f}m | IDF1: {met_c['idf1_score']:.2f}")
    ax2.plot(met_a["t_arr"], met_a["ospa_arr"], color=c_ideal, linewidth=1.3, linestyle=":", alpha=0.80,
             label=f"Mod A (İdeal) - OSPA: {met_a['ss_ospa']:.2f}m | IDF1: {met_a['idf1_score']:.2f}")

    ax2.set_title("2. OSPA (Optimal SubPattern Assignment) Skoru ve İz Sürekliliği (IDF1) Performansı",
                  fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("OSPA Mesafesi (m)", fontsize=10.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)
    ax2.set_ylim(0, 16.5)

    # Subplot 3: Filter Consistency NEES (Normalized Estimation Error Squared)
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    ax3.plot(met_b["t_arr"], met_b["nees_arr"], color=c_naive, linewidth=1.5, alpha=0.75,
             label=f"Mod B (Naive) Ort. NEES: {met_b['ss_nees']:.2f} (Tutarsız / Aşırı Özgüven)")
    ax3.plot(met_e["t_arr"], met_e["nees_arr"], color=c_extrap, linewidth=1.5, linestyle="-.", alpha=0.85,
             label=f"Mod E (Predictor) Ort. NEES: {met_e['ss_nees']:.2f}")
    ax3.plot(met_d["t_arr"], met_d["nees_arr"], color=c_retro, linewidth=1.6, linestyle="--", alpha=0.90,
             label=f"Mod D (Modifiye EKF) Ort. NEES: {met_d['ss_nees']:.2f}")
    ax3.plot(met_c["t_arr"], met_c["nees_arr"], color=c_replay, linewidth=1.9, alpha=0.95,
             label=f"Mod C (Rollback) Ort. NEES: {met_c['ss_nees']:.2f}")
    ax3.axhline(6.0, color="#1F77B4", linestyle="--", linewidth=1.2, label="İdeal 6-DOF Beklenen NEES (E[NEES] = 6.0)")
    ax3.axhline(14.45, color="#7F8C8D", linestyle=":", linewidth=1.1, label=r"%95 $\chi^2_6$ Üst Güven Sınırı (14.45)")

    ax3.set_title("3. Filtre Kovaryans Tutarlılığı Analizi: NEES (Normalized Estimation Error Squared)",
                  fontsize=11.5, fontweight="bold")
    ax3.set_ylabel("NEES Değeri", fontsize=10.5)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", ncol=2, fontsize=8.5, framealpha=0.95)
    ax3.set_ylim(0, max(25.0, min(50.0, float(np.percentile(met_b["nees_arr"], 95)))))

    # Subplot 4: Latency Distribution and OOSM Arrival Events
    ax4 = axes[3]
    ax4.set_facecolor("#FFFFFF")
    radar_delays = [(t, d * 1000.0) for t, d, s in log_delays if s == "RADAR"]
    camera_delays = [(t, d * 1000.0) for t, d, s in log_delays if s == "CAMERA"]

    if radar_delays:
        rt, rd = zip(*radar_delays)
        ax4.scatter(rt, rd, color="#FF7F0E", s=16, alpha=0.65, label="Radar İletim Gecikmesi (~8 ms + 2ms Jitter)", marker="o")
    if camera_delays:
        ct, cd = zip(*camera_delays)
        ax4.scatter(ct, cd, color="#9467BD", s=14, alpha=0.65, label="Kamera İşleme Gecikmesi (~40 ms + 5ms Jitter)", marker="^")

    if log_oosm_events:
        oosm_t, oosm_c, _ = zip(*log_oosm_events)
        ax4.scatter(oosm_t, [(ot - oc) * 1000.0 for ot, oc in zip(oosm_t, oosm_c)],
                    color="#D62728", s=35, marker="x", linewidth=1.5,
                    label=f"Algılanan OOSM Olayları (Toplam: {len(log_oosm_events)} adet)")

    ax4.set_title(
        f"4. Sensör Gecikmeleri ve OOSM Dağılımı | Çalışma Süreleri: Mod C={met_c['exec_time_ms']:.1f}ms, Mod D={met_d['exec_time_ms']:.1f}ms, Mod E={met_e['exec_time_ms']:.1f}ms",
        fontsize=11.5, fontweight="bold",
    )
    ax4.set_xlabel("Simülasyon Zamanı (s)", fontsize=11)
    ax4.set_ylabel("Gecikme (ms)", fontsize=10.5)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    print(f"\n[Görselleştirme] Kıyaslama grafiği kaydedildi -> '{save_fig_path}'")

    alt_save_path = "results/ekf_fusion_results.png"
    plt.savefig(alt_save_path, dpi=300, bbox_inches="tight")
    print(f"[Görselleştirme] Ana sonuç grafiği güncellendi -> '{alt_save_path}'")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mode_a": met_a,
        "mode_b": met_b,
        "mode_c": met_c,
        "mode_d": met_d,
        "mode_e": met_e,
        "pos_improvement_vs_naive": pos_improvement_c,
        "pos_improvement_d_vs_naive": pos_improvement_d,
        "pos_improvement_e_vs_naive": pos_improvement_e,
        "vel_improvement_vs_naive": vel_improvement_vs_naive,
        "oosm_count": ekf_c.oosm_count,
        "oosm_count_d": ekf_d.oosm_count,
        "oosm_count_e": ekf_e.oosm_count,
        "dropped_oosm_count": ekf_c.dropped_oosm_count,
    }


if __name__ == "__main__":
    run_benchmark(sim_duration=20.0, show_plot=False)
