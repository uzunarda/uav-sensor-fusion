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
        """Verify that simulation worker precomputes all 3 modes properly."""
        worker = SimulationWorker(sim_duration=2.0, seed=42)
        for mode in ["MODE_A", "MODE_B", "MODE_C"]:
            frames = worker._mode_frames.get(mode, [])
            self.assertGreater(len(frames), 20)
            self.assertEqual(frames[0].sim_mode, mode)

    def test_gui_window_initialization_and_frame_dispatch(self):
        """Verify that TacticalRadarApp initializes and handles frame updates."""
        window = TacticalRadarApp(sim_duration=2.0, seed=42)
        window.show()
        self.app.processEvents()

        # Check initial widget state
        self.assertIsNotNone(window.view_live_radar.radar_display)
        self.assertIsNotNone(window.view_live_radar.gl_display)
        self.assertIsNotNone(window.view_live_radar.telemetry_panel)
        self.assertIsNotNone(window.view_live_radar.control_bar)
        self.assertIsNotNone(window.view_benchmark)
        self.assertIsNotNone(window.view_tests)
        self.assertIsNotNone(window.view_config)

        # Step forward a few ticks
        window.worker.play()
        for _ in range(10):
            window.worker._on_timer_tick()
            self.app.processEvents()

        self.assertGreater(window.worker.current_time, 0.0)

        # Test mode switching via worker
        window.worker.set_mode("MODE_A")
        self.app.processEvents()
        self.assertEqual(window.worker.current_mode, "MODE_A")

        window.worker.set_mode("MODE_C")
        self.app.processEvents()
        self.assertEqual(window.worker.current_mode, "MODE_C")

        # Test scrubbing
        window.worker.pause()
        window.worker.seek_time(1.0)
        self.app.processEvents()
        self.assertAlmostEqual(window.worker.current_time, 1.0)

        window.close()


if __name__ == "__main__":
    unittest.main()
