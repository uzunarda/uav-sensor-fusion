"""
Tactical Target Tracking & Kalman Filter Performance Metrics Module.
Implements literature-standard evaluation metrics from UAV OOSM tracking papers:
  1. 3D Position & Velocity RMSE (Root Mean Square Error) & Axial RMSE
  2. MAE (Mean Absolute Error) - Outlier-resistant tracking accuracy
  3. OSPA (Optimal SubPattern Assignment) - Combined localization and track-break/cardinality penalty
  4. NEES (Normalized Estimation Error Squared) & NIS (Normalized Innovation Squared) - Filter consistency
  5. Track Continuity (%) & IDF1 Score - Track life and identity preservation under sensor latency
  6. Execution Time (ms total & us/packet) - Computational complexity benchmarking
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np


def compute_nees_series(
    states_est: np.ndarray,
    states_gt: np.ndarray,
    covs: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Computes Normalized Estimation Error Squared (NEES) at each epoch k:
        NEES_k = (x_est_k - x_gt_k)^T * P_k^(-1) * (x_est_k - x_gt_k)

    For a consistent 6-DOF state estimator, E[NEES_k] = 6.0, with 95% chi-square
    confidence bounds [1.237, 14.449].

    Args:
        states_est: Array of shape (N, 6) containing estimated state vectors.
        states_gt: Array of shape (N, 6) containing ground truth state vectors.
        covs: Sequence of N covariance matrices, each of shape (6, 6).

    Returns:
        np.ndarray: 1D array of length N with NEES values.
    """
    n_samples = len(states_est)
    nees = np.zeros(n_samples, dtype=np.float64)
    reg_eye = np.eye(6, dtype=np.float64) * 1e-9

    for i in range(n_samples):
        err = (states_est[i] - states_gt[i]).reshape((6, 1))
        P = np.asarray(covs[i], dtype=np.float64) + reg_eye
        try:
            sol = np.linalg.solve(P, err)
            val = float((err.T @ sol)[0, 0])
            nees[i] = max(0.0, val)
        except np.linalg.LinAlgError:
            nees[i] = float("nan")

    return nees


def compute_ospa_series(
    pos_errors: np.ndarray,
    nees_series: Optional[np.ndarray] = None,
    cutoff_c: float = 15.0,
    order_p: float = 2.0,
    gate_threshold_m: float = 6.0,
    nees_gate_max: float = 25.0,
) -> np.ndarray:
    """
    Computes the Optimal SubPattern Assignment (OSPA) metric distance series.

    Combines 3D localization error with cardinality / false-track penalty when delayed
    out-of-sequence measurements cause track gating excursions or ghost track splits:
        d_OSPA(X, X_hat) = ( (1/n) * [ sum min(d, c)^p + c^p * |n - m| ] )^(1/p)

    Args:
        pos_errors: 1D array of 3D Euclidean position errors (m).
        nees_series: Optional 1D array of NEES values to detect covariance inconsistency / ghosting.
        cutoff_c: OSPA cutoff threshold c in meters (default: 15.0m).
        order_p: OSPA metric order p (default: 2.0).
        gate_threshold_m: Tactical validation gate radius (m) for track continuity.
        nees_gate_max: Chi-square upper consistency threshold for false-track anomaly penalty.

    Returns:
        np.ndarray: 1D array of instant OSPA distances in meters.
    """
    n_samples = len(pos_errors)
    ospa = np.zeros(n_samples, dtype=np.float64)
    c_p = cutoff_c ** order_p

    for i in range(n_samples):
        d = float(pos_errors[i])
        base_cost_p = min(d, cutoff_c) ** order_p

        # Check if delayed measurement caused a track validation gate excursion (cardinality penalty)
        is_gate_broken = d > gate_threshold_m
        is_nees_broken = (
            nees_series is not None
            and not np.isnan(nees_series[i])
            and float(nees_series[i]) > nees_gate_max
        )

        if is_gate_broken and is_nees_broken:
            # Both spatial and statistical gates broken -> full cutoff penalty
            ospa[i] = cutoff_c
        elif is_gate_broken or is_nees_broken:
            # Partial cardinality / association degradation (n=2 effective cardinality)
            ospa[i] = float(((0.5 * base_cost_p + 0.5 * c_p)) ** (1.0 / order_p))
        else:
            ospa[i] = min(d, cutoff_c)

    return ospa


def compute_track_continuity_and_idf1(
    pos_errors: np.ndarray,
    nees_series: Optional[np.ndarray] = None,
    gate_threshold_m: float = 4.5,
    nees_gate_max: float = 18.0,
) -> Dict[str, float]:
    """
    Computes Track Continuity (%) and IDF1 (Identity F1-Score) over a trajectory window.

    - Track Continuity (%): Percentage of epochs where the target remains locked within
      the tactical validation gate (position error <= gate_threshold_m).
    - IDF1 Score: Harmonic mean of identity precision and recall:
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
      where IDTP counts consistent in-gate track epochs, IDFN counts missed/lost track epochs,
      and IDFP counts false/diverged track epochs triggered by uncompensated latency spikes.

    Args:
        pos_errors: 1D array of 3D position errors (m).
        nees_series: Optional 1D array of NEES consistency values.
        gate_threshold_m: Spatial validation gate threshold in meters (default: 4.5m).
        nees_gate_max: Statistical consistency threshold for ghost/false track detection.

    Returns:
        Dict[str, float]: Dictionary containing 'track_continuity_pct', 'idf1_score', 'id_switches'.
    """
    n_samples = len(pos_errors)
    if n_samples == 0:
        return {"track_continuity_pct": 100.0, "idf1_score": 1.0, "id_switches": 0.0}

    in_spatial_gate = pos_errors <= gate_threshold_m
    if nees_series is not None:
        in_stat_gate = np.isnan(nees_series) | (nees_series <= nees_gate_max)
    else:
        in_stat_gate = np.ones(n_samples, dtype=bool)

    # Track continuity measures spatial lock persistence
    continuity_pct = float(np.mean(in_spatial_gate) * 100.0)

    # Identity True Positives (both spatial and statistical consistency maintained)
    idtp = int(np.sum(in_spatial_gate & in_stat_gate))
    # Identity False Negatives (target lost outside spatial gate)
    idfn = int(np.sum(~in_spatial_gate))
    # Identity False Positives (false / ghost track update due to latency spike or inconsistency)
    idfp = int(np.sum((~in_spatial_gate) | (~in_stat_gate)))

    denom = 2 * idtp + idfp + idfn
    idf1 = float((2 * idtp) / denom) if denom > 0 else 1.0

    # Count track fragmentation / ID switch events (transitions from in-gate to out-of-gate)
    locked = in_spatial_gate & in_stat_gate
    id_switches = 0
    for i in range(1, n_samples):
        if locked[i - 1] and not locked[i]:
            id_switches += 1

    return {
        "track_continuity_pct": continuity_pct,
        "idf1_score": idf1,
        "id_switches": float(id_switches),
    }


def evaluate_Comprehensive_metrics(
    times: Sequence[float],
    err_pos_list: Sequence[float],
    err_vel_list: Sequence[float],
    states_est: Sequence[np.ndarray],
    states_gt: Sequence[np.ndarray],
    covs: Optional[Sequence[np.ndarray]] = None,
    nis_list: Optional[Sequence[float]] = None,
    exec_time_ms: float = 0.0,
    steady_state_start: float = 3.0,
) -> Dict[str, object]:
    """
    Evaluates the full academic benchmark metric suite for a single simulation mode.

    Includes:
      - Steady-State & Total 3D Position RMSE (m) and Axial RMSE (X, Y, Z)
      - Steady-State & Total 3D Velocity RMSE (m/s)
      - Steady-State & Total 3D Position MAE (Mean Absolute Error, m)
      - Steady-State & Total OSPA (Optimal SubPattern Assignment, m)
      - Steady-State Mean NEES & NIS (Filter Consistency)
      - Track Continuity (%) & IDF1 Score
      - Execution Time (Total ms and Per-Packet microseconds)
    """
    t_arr = np.asarray(times, dtype=np.float64)
    p_err = np.asarray(err_pos_list, dtype=np.float64)
    v_err = np.asarray(err_vel_list, dtype=np.float64)
    st_arr = np.asarray(states_est, dtype=np.float64)
    gt_arr = np.asarray(states_gt, dtype=np.float64)
    diff = st_arr - gt_arr

    ss_mask = t_arr >= steady_state_start
    if not np.any(ss_mask):
        ss_mask = np.ones_like(t_arr, dtype=bool)

    ss_p_err = p_err[ss_mask]
    ss_v_err = v_err[ss_mask]
    ss_diff = diff[ss_mask]

    # 1. NEES series
    if covs is not None and len(covs) == len(st_arr):
        nees_arr = compute_nees_series(st_arr, gt_arr, covs)
    else:
        nees_arr = np.full(len(t_arr), 6.0, dtype=np.float64)
    ss_nees = nees_arr[ss_mask]
    mean_nees = float(np.nanmean(ss_nees)) if len(ss_nees) > 0 else 6.0

    # Consistency percentage within 95% chi-square bounds for 6-DOF [1.237, 14.449]
    nees_in_bounds = (ss_nees >= 1.237) & (ss_nees <= 14.449)
    nees_consistency_pct = float(np.mean(nees_in_bounds) * 100.0) if len(ss_nees) > 0 else 100.0

    # 2. NIS series
    if nis_list is not None and len(nis_list) > 0:
        nis_arr = np.asarray(nis_list, dtype=np.float64)
        ss_nis = nis_arr[ss_mask] if len(nis_arr) == len(t_arr) else nis_arr
        mean_nis = float(np.nanmean(ss_nis))
    else:
        nis_arr = np.ones(len(t_arr), dtype=np.float64)
        mean_nis = 1.0

    # 3. OSPA series
    ospa_arr = compute_ospa_series(p_err, nees_series=nees_arr, cutoff_c=15.0, order_p=2.0)
    ss_ospa = ospa_arr[ss_mask]

    # 4. Track Continuity & IDF1 Score (evaluated in steady-state tracking window)
    tc_idf1 = compute_track_continuity_and_idf1(ss_p_err, nees_series=ss_nees)

    # 5. Execution timing
    n_events = max(1, len(t_arr))
    exec_us_per_pkt = (float(exec_time_ms) * 1000.0) / n_events

    return {
        # RMSE Metrics
        "total_pos_rmse": float(np.sqrt(np.mean(p_err ** 2))),
        "ss_pos_rmse": float(np.sqrt(np.mean(ss_p_err ** 2))),
        "ss_rmse_x": float(np.sqrt(np.mean(ss_diff[:, 0] ** 2))),
        "ss_rmse_y": float(np.sqrt(np.mean(ss_diff[:, 2] ** 2))),
        "ss_rmse_z": float(np.sqrt(np.mean(ss_diff[:, 4] ** 2))),
        "total_vel_rmse": float(np.sqrt(np.mean(v_err ** 2))),
        "ss_vel_rmse": float(np.sqrt(np.mean(ss_v_err ** 2))),
        # MAE Metrics (Paper 4)
        "total_pos_mae": float(np.mean(np.abs(p_err))),
        "ss_pos_mae": float(np.mean(np.abs(ss_p_err))),
        "ss_vel_mae": float(np.mean(np.abs(ss_v_err))),
        # OSPA Metrics (Paper 2 & Core Recommendation)
        "total_ospa": float(np.mean(ospa_arr)),
        "ss_ospa": float(np.mean(ss_ospa)),
        # NEES & NIS Consistency Metrics (Paper 3)
        "ss_nees": mean_nees,
        "nees_consistency_pct": nees_consistency_pct,
        "ss_nis": mean_nis,
        # Track Continuity & IDF1 Metrics (Paper 2 & Core Recommendation)
        "track_continuity_pct": tc_idf1["track_continuity_pct"],
        "idf1_score": tc_idf1["idf1_score"],
        "id_switches": tc_idf1["id_switches"],
        # Computational Cost Metrics (Paper 3 & Core Recommendation)
        "exec_time_ms": float(exec_time_ms),
        "exec_us_per_pkt": float(exec_us_per_pkt),
        # Time Series Arrays for Plotting
        "diff": diff,
        "t_arr": t_arr,
        "p_err": p_err,
        "v_err": v_err,
        "ospa_arr": ospa_arr,
        "nees_arr": nees_arr,
        "nis_arr": nis_arr,
    }
