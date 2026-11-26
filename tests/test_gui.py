from __future__ import annotations
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import QApplication
from src.gui.worker import SimulationWorker, SimulationFrame
from src.gui.app import TacticalRadarApp


class TestTacticalRadarGUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create single QApplication instance for offscreen testing
        cls.app = QApplication.instance() or QApplication([])

    def test_simulation_worker_precomputation(self):
        """Verify that simulation worker precomputes all 5 modes (Modes A-E) and metrics properly."""
        worker = SimulationWorker(sim_duration=2.0, seed=42)
        for mode in ["MODE_A", "MODE_B", "MODE_C", "MODE_D", "MODE_E"]:
            frames = worker._mode_frames.get(mode, [])
            self.assertGreater(len(frames), 20)
            self.assertEqual(frames[0].sim_mode, mode)
            self.assertGreaterEqual(frames[-1].ospa_val, 0.0)
            self.assertGreaterEqual(frames[-1].nees_val, 0.0)
            self.assertGreaterEqual(frames[-1].idf1_score, 0.0)

    def test_gui_window_initialization_and_frame_dispatch(self):
        """Verify that TacticalRadarApp initializes and handles frame updates."""
        window = TacticalRadarApp(sim_duration=2.0, seed=42)
        window.show()
        self.app.processEvents()

        # Check initial widget state
        self.assertIsNotNone(window.view_live_radar.radar_display)
        self.assertIsNotNone(window.view_live_radar.gl_display)
        self.assertIsNotNone(window.view_live_radar.sensor_stream_panel)
        self.assertIsNotNone(window.view_live_radar.telemetry_panel)
        self.assertIsNotNone(window.view_live_radar.control_bar)
        self.assertIsNotNone(window.view_benchmark)
        self.assertIsNotNone(window.view_tests)
        self.assertIsNotNone(window.view_config)

        # Verify synthetic packet stream is populated on initial frame
        self.assertGreater(window.view_live_radar.sensor_stream_panel.table.rowCount(), 0)

        # Step forward a few ticks
        window.worker.play()
        for _ in range(10):
            window.worker._on_timer_tick()
            self.app.processEvents()

        self.assertGreater(window.worker.current_time, 0.0)

        # Test mode switching via worker across all 5 modes
        for m in ["MODE_A", "MODE_B", "MODE_C", "MODE_D", "MODE_E"]:
            window.worker.set_mode(m)
            self.app.processEvents()
            self.assertEqual(window.worker.current_mode, m)

        # Test scrubbing
        window.worker.pause()
        window.worker.seek_time(1.0)
        self.app.processEvents()
        self.assertAlmostEqual(window.worker.current_time, 1.0)

        # Test scenario switching (RK4 Coordinated Turn and Evasive Weave)
        window.worker.set_scenario("CT", uav_dynamic=True)
        self.assertEqual(window.worker.scenario_type, "CT")
        self.assertTrue(window.worker.uav_motion_enabled)
        self.assertGreater(len(window.worker._mode_frames["MODE_C"]), 20)

        # Test One-Click Three.js 3D Simulator Server launch & endpoints
        import json
        import urllib.request

        sim_url = window.launch_3d_simulator(open_browser=False)
        self.assertTrue(sim_url.startswith("http://127.0.0.1:"))

        with urllib.request.urlopen(sim_url) as resp_html:
            html_body = resp_html.read().decode("utf-8")
            self.assertEqual(resp_html.status, 200)
            self.assertIn("THREE.JS", html_body)

        with urllib.request.urlopen(f"{sim_url}api/simulation?scenario=CT&uav_dynamic=1") as resp_json:
            payload = json.loads(resp_json.read().decode("utf-8"))
            self.assertEqual(payload["scenario"], "CT")
            self.assertIn("MODE_A", payload["metrics"])
            self.assertIn("MODE_E", payload["metrics"])
            self.assertGreater(len(payload["timeline"]), 20)

        with urllib.request.urlopen(f"{sim_url}api/export_csv?scenario=EVASIVE&mode=MODE_C") as resp_csv:
            csv_text = resp_csv.read().decode("utf-8")
            self.assertIn("pos_error_3d_m,vel_error_3d_ms,ospa_m,nees,nis", csv_text)

        window.sim3d_server.stop()
        window.close()


if __name__ == "__main__":
    unittest.main()
