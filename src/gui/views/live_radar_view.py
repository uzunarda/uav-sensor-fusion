"""
Tab 1: Live Tactical Radar View (2x2 MFD Layout: PPI Scope, 3D Spatial Trajectory,
Live Synthetic Sensor Data Stream Inspector, HUD Telemetry & Playback Bar).
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSplitter,
)

from src.gui.worker import SimulationWorker, SimulationFrame
from src.gui.widgets.radar_display import TacticalRadarDisplay
from src.gui.widgets.gl_display import Trajectory3DDisplay
from src.gui.widgets.telemetry_panel import TelemetryPanel
from src.gui.widgets.sensor_stream_panel import SensorDataStreamPanel
from src.gui.widgets.control_bar import PlaybackControlBar


class LiveRadarView(QWidget):
    """
    Live Radar & Multi-Sensor Fusion MFD Console View.
    """

    def __init__(self, worker: SimulationWorker, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker = worker

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Main Horizontal Splitter: Left MFD Column vs Right MFD Column
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(4)

        # Left Column Vertical Splitter: Top (2D PPI Radar) / Bottom (Synthetic Sensor Data Stream)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setHandleWidth(4)

        self.radar_display = TacticalRadarDisplay(self)
        self.sensor_stream_panel = SensorDataStreamPanel(self)

        self.radar_display.setMinimumHeight(320)
        self.sensor_stream_panel.setMinimumHeight(230)

        left_splitter.addWidget(self.radar_display)
        left_splitter.addWidget(self.sensor_stream_panel)
        left_splitter.setStretchFactor(0, 6)
        left_splitter.setStretchFactor(1, 4)
        left_splitter.setSizes([470, 310])

        main_splitter.addWidget(left_splitter)

        # Right Column Vertical Splitter: Top (3D Trajectory) / Bottom (HUD Telemetry)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(4)

        self.gl_display = Trajectory3DDisplay(self)
        self.telemetry_panel = TelemetryPanel(self)

        self.gl_display.setMinimumWidth(480)
        self.gl_display.setMinimumHeight(320)
        self.telemetry_panel.setMinimumWidth(480)
        self.telemetry_panel.setMinimumHeight(230)

        right_splitter.addWidget(self.gl_display)
        right_splitter.addWidget(self.telemetry_panel)
        right_splitter.setStretchFactor(0, 6)
        right_splitter.setStretchFactor(1, 4)
        right_splitter.setSizes([470, 310])

        main_splitter.addWidget(right_splitter)

        # Balanced 52% / 48% split across Left and Right MFD columns
        main_splitter.setStretchFactor(0, 52)
        main_splitter.setStretchFactor(1, 48)
        main_splitter.setSizes([720, 660])

        layout.addWidget(main_splitter, stretch=1)

        # Bottom Section: Playback & Mode Control Bar
        self.control_bar = PlaybackControlBar(sim_duration=worker.sim_duration, parent=self)
        layout.addWidget(self.control_bar, stretch=0)

        # Connect Worker Signals
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.playback_state_changed.connect(self.control_bar.set_playing_state)

        # Connect Controls to Worker
        self.control_bar.play_requested.connect(self.worker.play)
        self.control_bar.pause_requested.connect(self.worker.pause)
        self.control_bar.reset_requested.connect(self.worker.reset)
        self.control_bar.seek_requested.connect(self.worker.seek_time)
        self.control_bar.speed_changed.connect(self.worker.set_speed)
        self.control_bar.mode_selected.connect(self.worker.set_mode)

    def _on_frame_ready(
        self,
        frame: SimulationFrame,
        gt_history: object,
        ekf_history: object,
        raw_tail: object,
    ) -> None:
        """Dispatches simulation frame to all 4 MFD quadrants and control bar."""
        self.radar_display.update_frame(frame, gt_history, ekf_history, raw_tail)
        self.gl_display.update_frame(frame, gt_history, ekf_history, raw_tail)
        self.sensor_stream_panel.update_frame(frame)
        self.telemetry_panel.update_frame(frame)
        self.control_bar.update_frame(frame)
