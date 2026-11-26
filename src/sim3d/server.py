"""
Thread-safe Local HTTP & JSON Telemetry Bridge for the Three.js 3D Tactical Flight & Sensor Simulator.
Serves:
  - GET /                        -> Three.js 3D Tactical Flight & EO/IR Gimbal Simulator UI (index.html)
  - GET /api/simulation          -> Complete 5-Mode (Modes A-E) 6-DOF trajectory, sensor stream & 16 literature metrics
  - GET /api/export_csv          -> Downloadable academic CSV dataset for MATLAB/Python paper verification
"""

from __future__ import annotations
import csv
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import numpy as np

from src.dynamics import TargetState, UAVFlightTrajectory, create_trajectory
from src.sensors import RadarSensor, CameraSensor, reconstruct_pseudo_cartesian
from src.simulator import AsyncSensorEventSimulator
from src.ekf import ExtendedKalmanFilter
from src.metrics import evaluate_Comprehensive_metrics, compute_nees_series, compute_ospa_series


class Sim3DBridgeServer:
    """
    Background HTTP server connecting the Python 6-DOF RK4 & 5-Mode EKF engine
    to the Three.js 3D Flight & EO/IR Gimbal Camera Simulator.
    """

    def __init__(self, worker: Optional[Any] = None, host: str = "127.0.0.1", port: int = 0) -> None:
        self.worker = worker
        self.host = host
        self.port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    @property
    def url(self) -> str:
        """Returns active HTTP URL of the 3D simulator."""
        return f"http://{self.host}:{self.port}/"

    def invalidate_cache(self) -> None:
        """Clears cached simulation runs when user updates sensor/filter parameters in PyQt6."""
        self._cache.clear()

    def _get_params(self) -> Dict[str, Any]:
        """Extracts current sensor and EKF parameters from attached PyQt6 worker or defaults."""
        if self.worker is not None:
            return {
                "sim_duration": float(getattr(self.worker, "sim_duration", 20.0)),
                "seed": int(getattr(self.worker, "seed", 42)),
                "scenario_type": str(getattr(self.worker, "scenario_type", "CV")),
                "uav_dynamic": bool(getattr(self.worker, "uav_motion_enabled", False)),
                "radar_freq_hz": float(getattr(self.worker, "radar_freq_hz", 10.0)),
                "radar_sigma_m": float(getattr(self.worker, "radar_sigma_m", 6.0)),
                "radar_latency_s": float(getattr(self.worker, "radar_latency_s", 0.008)),
                "radar_jitter_s": float(getattr(self.worker, "radar_jitter_s", 0.002)),
                "cam_freq_hz": float(getattr(self.worker, "cam_freq_hz", 30.0)),
                "cam_sigma_rad": float(getattr(self.worker, "cam_sigma_rad", 0.004)),
                "cam_latency_s": float(getattr(self.worker, "cam_latency_s", 0.040)),
                "cam_jitter_s": float(getattr(self.worker, "cam_jitter_s", 0.005)),
                "cam_clock_bias_s": float(getattr(self.worker, "cam_clock_bias_s", 0.025)),
                "q_continuous": float(getattr(self.worker, "q_continuous", 0.8)),
                "max_buffer_size": int(getattr(self.worker, "max_buffer_size", 150)),
            }
        return {
            "sim_duration": 20.0,
            "seed": 42,
            "scenario_type": "EVASIVE",
            "uav_dynamic": True,
            "radar_freq_hz": 10.0,
            "radar_sigma_m": 6.0,
            "radar_latency_s": 0.008,
            "radar_jitter_s": 0.002,
            "cam_freq_hz": 30.0,
            "cam_sigma_rad": 0.004,
            "cam_latency_s": 0.040,
            "cam_jitter_s": 0.005,
            "cam_clock_bias_s": 0.025,
            "q_continuous": 0.8,
            "max_buffer_size": 150,
        }

    def build_simulation_payload(
        self,
        scenario: Optional[str] = None,
        uav_dynamic: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Runs all 5 EKF modes (Modes A-E) under the specified 6-DOF flight scenario
        and packages 3D trajectories, attitude angles, sensor measurements, and 16 literature metrics.
        """
        params = self._get_params()
        scen = (scenario or params["scenario_type"]).upper().strip()
        if scen not in ("CV", "CT", "EVASIVE"):
            scen = "CV"
        dyn_uav = params["uav_dynamic"] if uav_dynamic is None else bool(uav_dynamic)

        cache_key = (
            f"{scen}_{int(dyn_uav)}_{params['radar_freq_hz']}_{params['radar_sigma_m']}_"
            f"{params['radar_latency_s']}_{params['cam_freq_hz']}_{params['cam_sigma_rad']}_"
            f"{params['cam_latency_s']}_{params['cam_clock_bias_s']}_{params['q_continuous']}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        duration = params["sim_duration"]
        seed = params["seed"]

        true_init = TargetState(x=1200.0, vx=35.0, y=-800.0, vy=25.0, z=-500.0, vz=-4.0)
        init_offset = np.array([80.0, -15.0, -90.0, 10.0, 40.0, 6.0], dtype=np.float64)
        x_init_guess = true_init.to_vector() + init_offset
        P_init = np.diag([150.0**2, 20.0**2, 150.0**2, 20.0**2, 100.0**2, 15.0**2])

        # Use adapted process noise for maneuvering targets so EKF tracks CT/EVASIVE turns smoothly
        q_c = params["q_continuous"]
        if scen == "CT":
            q_c = max(q_c, 6.0)
        elif scen == "EVASIVE":
            q_c = max(q_c, 14.0)

        target_traj = create_trajectory(scen, true_init, t0=0.0)
        uav_traj = UAVFlightTrajectory(
            initial_pos=np.array([0.0, 0.0, -150.0], dtype=np.float64),
            cruise_speed=28.0,
            orbit_radius=380.0,
            dynamic_flight=dyn_uav,
        )

        # 1. Ideal Stream (Mode A)
        rng_a = np.random.default_rng(seed)
        radar_a = RadarSensor(params["radar_freq_hz"], params["radar_sigma_m"], 0.0, 0.0, 0.0, rng=rng_a)
        cam_a = CameraSensor(params["cam_freq_hz"], params["cam_sigma_rad"], params["cam_sigma_rad"], 0.0, 0.0, 0.0, rng=rng_a)
        sim_a = AsyncSensorEventSimulator(target_traj)
        sim_a.register_sensor(radar_a, phase_offset=0.0)
        sim_a.register_sensor(cam_a, phase_offset=0.0125)
        sim_a.populate_events(0.0, duration, uav_position_fn=uav_traj.get_position)
        events_a = sim_a.clone_event_stream()

        # 2. Realistic Delayed Stream (Modes B, C, D, E)
        rng_d = np.random.default_rng(seed)
        radar_d = RadarSensor(
            params["radar_freq_hz"],
            params["radar_sigma_m"],
            params["radar_latency_s"],
            params["radar_jitter_s"],
            0.0,
            rng=rng_d,
        )
        cam_d = CameraSensor(
            params["cam_freq_hz"],
            params["cam_sigma_rad"],
            params["cam_sigma_rad"],
            params["cam_latency_s"],
            params["cam_jitter_s"],
            params["cam_clock_bias_s"],
            rng=rng_d,
        )
        sim_d = AsyncSensorEventSimulator(target_traj)
        sim_d.register_sensor(radar_d, phase_offset=0.0)
        sim_d.register_sensor(cam_d, phase_offset=0.0125)
        sim_d.populate_events(0.0, duration, uav_position_fn=uav_traj.get_position)
        events_delayed = sim_d.clone_event_stream()

        mode_configs = {
            "MODE_A": (events_a, False, False, "ROLLBACK"),
            "MODE_B": (events_delayed, True, False, "ROLLBACK"),
            "MODE_C": (events_delayed, False, True, "ROLLBACK"),
            "MODE_D": (events_delayed, False, True, "RETRODICTION"),
            "MODE_E": (events_delayed, False, True, "EXTRAPOLATION"),
        }

        mode_summaries: Dict[str, Dict[str, Any]] = {}
        mode_series: Dict[str, Dict[str, List[Any]]] = {}

        for m_name, (ev_stream, is_naive, en_oosm, strat) in mode_configs.items():
            ekf = ExtendedKalmanFilter(
                initial_state=x_init_guess.copy(),
                initial_covariance=P_init.copy(),
                process_noise_spectral_density=q_c,
                enable_oosm_buffer=en_oosm,
                max_buffer_size=params["max_buffer_size"],
                oosm_strategy=strat,
            )
            times: List[float] = []
            p_errs: List[float] = []
            v_errs: List[float] = []
            ests: List[np.ndarray] = []
            gts: List[np.ndarray] = []
            covs: List[np.ndarray] = []
            nis_vals: List[float] = []
            oosm_flags: List[bool] = []

            prev_oosm = 0
            t0_ns = time.perf_counter_ns()
            for ev in ev_stream:
                uav_p = ev.uav_pos if ev.uav_pos is not None else uav_traj.get_position(ev.t_capture)
                ekf.update(ev, uav_pos=uav_p, is_naive=is_naive)
                t_eval = ev.t_receive if is_naive else ekf.current_time
                gt_st = target_traj.get_state(t_eval)
                gt_v = gt_st.to_vector()
                est_v = ekf.x.flatten().copy()
                P_curr = ekf.P.copy()

                err_p = float(np.linalg.norm(est_v[[0, 2, 4]] - gt_v[[0, 2, 4]]))
                err_v = float(np.linalg.norm(est_v[[1, 3, 5]] - gt_v[[1, 3, 5]]))

                times.append(float(t_eval))
                p_errs.append(err_p)
                v_errs.append(err_v)
                ests.append(est_v)
                gts.append(gt_v)
                covs.append(P_curr)
                nis_vals.append(float(ekf.last_nis))
                oosm_flags.append(ekf.oosm_count > prev_oosm)
                prev_oosm = ekf.oosm_count

            exec_ms = (time.perf_counter_ns() - t0_ns) / 1e6
            comp = evaluate_Comprehensive_metrics(
                times=times,
                err_pos_list=p_errs,
                err_vel_list=v_errs,
                states_est=ests,
                states_gt=gts,
                covs=covs,
                nis_list=nis_vals,
                exec_time_ms=exec_ms,
            )

            mode_summaries[m_name] = {
                "ss_pos_rmse": round(float(comp["ss_pos_rmse"]), 3),
                "total_pos_rmse": round(float(comp["total_pos_rmse"]), 3),
                "ss_pos_mae": round(float(comp["ss_pos_mae"]), 3),
                "ss_ospa": round(float(comp["ss_ospa"]), 3),
                "ss_vel_rmse": round(float(comp["ss_vel_rmse"]), 3),
                "ss_nees": round(float(comp["ss_nees"]), 2),
                "nees_consistency_pct": round(float(comp["nees_consistency_pct"]), 1),
                "ss_nis": round(float(comp["ss_nis"]), 2),
                "track_continuity_pct": round(float(comp["track_continuity_pct"]), 1),
                "idf1_score": round(float(comp["idf1_score"]), 3),
                "oosm_count": int(ekf.oosm_count),
                "exec_us_per_pkt": round(float(comp["exec_us_per_pkt"]), 1),
                "exec_time_ms": round(float(comp["exec_time_ms"]), 2),
            }
            mode_series[m_name] = {
                "ests": ests,
                "covs": covs,
                "p_errs": p_errs,
                "v_errs": v_errs,
                "ospa": [float(x) for x in comp["ospa_arr"]],
                "nees": [float(x) for x in comp["nees_arr"]],
                "nis": nis_vals,
                "oosm_flags": oosm_flags,
            }

        # Build unified timeline array synced to the delayed event stream (800 events over 20s)
        timeline: List[Dict[str, Any]] = []
        last_r: Optional[float] = None
        last_cam: Optional[tuple] = None
        last_cv: Optional[Dict[str, Any]] = None

        for idx, ev in enumerate(events_delayed):
            t_epoch = float(ev.t_receive)
            uav_pose = uav_traj.get_pose(t_epoch)
            gt_state = target_traj.get_state(t_epoch)
            _, tgt_pitch, tgt_yaw = gt_state.attitude_euler

            # Estimate target roll angle from turn rate / lateral velocity change
            gt_prev = target_traj.get_state(max(0.0, t_epoch - 0.05))
            _, _, prev_yaw = gt_prev.attitude_euler
            dyaw = (tgt_yaw - prev_yaw + np.pi) % (2.0 * np.pi) - np.pi
            omega_inst = dyaw / 0.05
            horiz_spd = max(float(np.hypot(gt_state.vx, gt_state.vy)), 1.0)
            tgt_roll = float(np.clip(np.arctan((horiz_spd * omega_inst) / 9.80665), -1.1, 1.1))

            if ev.sensor_type == "RADAR":
                last_r = float(ev.z[0])
            else:
                last_cam = (float(ev.z[0]), float(ev.z[1]))
                if getattr(ev, "cv_telemetry", None) is not None:
                    last_cv = ev.cv_telemetry

            raw_pos_list: Optional[List[float]] = None
            if last_r is not None and last_cam is not None:
                rp = reconstruct_pseudo_cartesian(last_r, last_cam[0], last_cam[1], uav_pose.position)
                raw_pos_list = [round(float(rp[0]), 2), round(float(rp[1]), 2), round(float(rp[2]), 2)]

            bias_off = params["cam_clock_bias_s"] if ev.sensor_type == "CAMERA" else 0.0
            t_phys = max(0.0, ev.t_capture - bias_off)
            latency_ms = round(max(0.0, (ev.t_receive - t_phys) * 1000.0), 2)

            modes_frame: Dict[str, Any] = {}
            for m_name in ("MODE_A", "MODE_B", "MODE_C", "MODE_D", "MODE_E"):
                m_idx = min(idx, len(mode_series[m_name]["ests"]) - 1)
                est_vec = mode_series[m_name]["ests"][m_idx]
                P_mat = mode_series[m_name]["covs"][m_idx]
                modes_frame[m_name] = {
                    "pos": [round(float(est_vec[0]), 2), round(float(est_vec[2]), 2), round(float(est_vec[4]), 2)],
                    "vel": [round(float(est_vec[1]), 2), round(float(est_vec[3]), 2), round(float(est_vec[5]), 2)],
                    "std": [
                        round(float(np.sqrt(max(1e-4, P_mat[0, 0]))), 2),
                        round(float(np.sqrt(max(1e-4, P_mat[2, 2]))), 2),
                        round(float(np.sqrt(max(1e-4, P_mat[4, 4]))), 2),
                    ],
                    "pos_err": round(float(mode_series[m_name]["p_errs"][m_idx]), 3),
                    "vel_err": round(float(mode_series[m_name]["v_errs"][m_idx]), 3),
                    "ospa": round(float(mode_series[m_name]["ospa"][m_idx]), 3),
                    "nees": round(float(mode_series[m_name]["nees"][m_idx]), 2),
                    "nis": round(float(mode_series[m_name]["nis"][m_idx]), 2),
                    "is_oosm": bool(mode_series[m_name]["oosm_flags"][m_idx]),
                }

            timeline.append({
                "seq": idx + 1,
                "t": round(t_epoch, 4),
                "t_capture": round(float(ev.t_capture), 4),
                "sensor": ev.sensor_type,
                "latency_ms": latency_ms,
                "raw_range": round(float(last_r), 2) if last_r is not None else 0.0,
                "raw_az_deg": round(float(np.degrees(last_cam[0])), 2) if last_cam is not None else 0.0,
                "raw_el_deg": round(float(np.degrees(last_cam[1])), 2) if last_cam is not None else 0.0,
                "raw_pos": raw_pos_list,
                "cv": last_cv,
                "uav": {
                    "pos": [round(float(uav_pose.position[0]), 2), round(float(uav_pose.position[1]), 2), round(float(uav_pose.position[2]), 2)],
                    "vel": [round(float(uav_pose.velocity[0]), 2), round(float(uav_pose.velocity[1]), 2), round(float(uav_pose.velocity[2]), 2)],
                    "euler": [round(float(uav_pose.roll), 4), round(float(uav_pose.pitch), 4), round(float(uav_pose.yaw), 4)],
                },
                "gt": {
                    "pos": [round(float(gt_state.x), 2), round(float(gt_state.y), 2), round(float(gt_state.z), 2)],
                    "vel": [round(float(gt_state.vx), 2), round(float(gt_state.vy), 2), round(float(gt_state.vz), 2)],
                    "euler": [round(tgt_roll, 4), round(tgt_pitch, 4), round(tgt_yaw, 4)],
                },
                "modes": modes_frame,
            })

        payload = {
            "scenario": scen,
            "uav_dynamic": dyn_uav,
            "duration": duration,
            "params": params,
            "metrics": mode_summaries,
            "timeline": timeline,
        }
        self._cache[cache_key] = payload
        return payload

    def build_csv_export(self, scenario: str = "CV", mode: str = "MODE_C") -> str:
        """Generates an academic paper-ready CSV dataset for the requested scenario and EKF mode."""
        payload = self.build_simulation_payload(scenario=scenario)
        mode_key = mode.upper() if mode.upper() in ("MODE_A", "MODE_B", "MODE_C", "MODE_D", "MODE_E") else "MODE_C"

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "seq_id", "epoch_s", "t_capture_s", "scenario", "mode", "sensor_type", "latency_ms", "is_oosm",
            "uav_x_m", "uav_y_m", "uav_z_m", "uav_roll_rad", "uav_pitch_rad", "uav_yaw_rad",
            "gt_x_m", "gt_y_m", "gt_z_m", "gt_vx_ms", "gt_vy_ms", "gt_vz_ms",
            "ekf_x_m", "ekf_y_m", "ekf_z_m", "ekf_vx_ms", "ekf_vy_ms", "ekf_vz_ms",
            "pos_error_3d_m", "vel_error_3d_ms", "ospa_m", "nees", "nis",
        ])

        for frame in payload["timeline"]:
            m = frame["modes"][mode_key]
            u = frame["uav"]
            g = frame["gt"]
            writer.writerow([
                frame["seq"], frame["t"], frame["t_capture"], payload["scenario"], mode_key,
                frame["sensor"], frame["latency_ms"], int(m["is_oosm"]),
                u["pos"][0], u["pos"][1], u["pos"][2], u["euler"][0], u["euler"][1], u["euler"][2],
                g["pos"][0], g["pos"][1], g["pos"][2], g["vel"][0], g["vel"][1], g["vel"][2],
                m["pos"][0], m["pos"][1], m["pos"][2], m["vel"][0], m["vel"][1], m["vel"][2],
                m["pos_err"], m["vel_err"], m["ospa"], m["nees"], m["nis"],
            ])
        return buf.getvalue()

    def start(self) -> str:
        """Starts the background HTTP server thread if not already running and returns its URL."""
        if self._httpd is not None:
            return self.url

        bridge = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress noisy console logs during tests and 60 FPS polling

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)

                if parsed.path in ("/", "/index.html"):
                    try:
                        with open(bridge._html_path, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                    except OSError as exc:
                        self.send_error(500, f"Failed to load 3D simulator HTML: {exc}")
                    return

                if parsed.path == "/api/simulation":
                    scen = qs.get("scenario", [None])[0]
                    uav_dyn_str = qs.get("uav_dynamic", [None])[0]
                    uav_dyn = None if uav_dyn_str is None else (uav_dyn_str in ("1", "true", "True"))
                    data = bridge.build_simulation_payload(scenario=scen, uav_dynamic=uav_dyn)
                    raw_json = json.dumps(data).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(raw_json)))
                    self.end_headers()
                    self.wfile.write(raw_json)
                    return

                if parsed.path == "/api/export_csv":
                    scen = qs.get("scenario", ["CV"])[0]
                    mode = qs.get("mode", ["MODE_C"])[0]
                    csv_data = bridge.build_csv_export(scenario=scen, mode=mode).encode("utf-8")
                    filename = f"uav_ekf_dataset_{scen}_{mode}.csv"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                    self.send_header("Content-Length", str(len(csv_data)))
                    self.end_headers()
                    self.wfile.write(csv_data)
                    return

                self.send_error(404, "Not Found")

        self._httpd = ThreadingHTTPServer((self.host, self.port), RequestHandler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="Sim3DBridgeThread", daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        """Stops the background HTTP server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            self._thread = None
