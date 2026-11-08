"""
2D Tactical Radar Display (PPI Scope) for UAV Multi-Sensor Fusion.
Renders Ownship UAV, radar range rings, camera FOV cone, ground truth track,
EKF fusion estimate, 2D error covariance uncertainty ellipse, and raw observations.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

import pyqtgraph as pg

from src.gui.worker import SimulationFrame


class TacticalRadarDisplay(QWidget):
    """
    2D Tactical Radar PPI Scope Widget using PyQtGraph for 60 FPS rendering.
    Coordinates: Horizontal = East (Y), Vertical = North (X) in NED frame.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header Title
        self.title_label = QLabel("2B TAKTİK RADAR (PPI SCOPE) - DOĞU-KUZEY DÜZLEMİ")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #00FF66;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                background-color: #0D1612;
                padding: 4px 8px;
                border-bottom: 1px solid #1A3324;
            }
        """)
        layout.addWidget(self.title_label)

        # PyQtGraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#050A07")  # Deep tactical radar black-green
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getAxis("bottom").setLabel("Doğu (East - Y)", units="m", color="#5C8A6F")
        self.plot_widget.getAxis("left").setLabel("Kuzey (North - X)", units="m", color="#5C8A6F")

        # Range bounds
        self.plot_widget.setXRange(-1600, 400)
        self.plot_widget.setYRange(-200, 1800)

        layout.addWidget(self.plot_widget)

        # Create static radar elements (rings, crosshairs)
        self._init_radar_grid()

        # Dynamic plot items
        self._init_dynamic_layers()

    def _init_radar_grid(self) -> None:
        """Draws concentric range rings (500m, 1000m, 1500m) and crosshairs centered at UAV."""
        ring_pen = pg.mkPen(color=QColor(0, 100, 45, 120), width=1, style=Qt.PenStyle.DashLine)

        phi = np.linspace(0, 2 * np.pi, 120)
        for r in [500.0, 1000.0, 1500.0]:
            x_ring = r * np.sin(phi)  # East
            y_ring = r * np.cos(phi)  # North
            ring_item = pg.PlotCurveItem(x_ring, y_ring, pen=ring_pen)
            self.plot_widget.addItem(ring_item)

            # Label for ring
            label = pg.TextItem(f"{int(r)}m", color=QColor(0, 180, 80, 160), anchor=(0.5, 1.2))
            label.setFont(QFont("Consolas", 8))
            label.setPos(0.0, r)
            self.plot_widget.addItem(label)

        # Crosshairs through origin (UAV)
        axis_pen = pg.mkPen(color=QColor(0, 80, 35, 90), width=1)
        self.plot_widget.addItem(pg.PlotCurveItem([-1800, 1800], [0, 0], pen=axis_pen))
        self.plot_widget.addItem(pg.PlotCurveItem([0, 0], [-1800, 1800], pen=axis_pen))

        # Ownship UAV Marker at (0, 0)
        self.ownship_marker = pg.ScatterPlotItem(
            [0.0], [0.0],
            symbol="t1",  # Upward pointing triangle
            size=14,
            pen=pg.mkPen("#00FFFF", width=1.5),
            brush=pg.mkBrush("#008888"),
        )
        self.plot_widget.addItem(self.ownship_marker)

        uav_label = pg.TextItem("OWNSHIP (İHA)", color=QColor(0, 255, 255, 200), anchor=(0.5, -0.6))
        uav_label.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        uav_label.setPos(0.0, 0.0)
        self.plot_widget.addItem(uav_label)

    def _init_dynamic_layers(self) -> None:
        """Initializes dynamic data curves and scatter layers."""
        # 1. Camera FOV Cone
        self.fov_cone_item = pg.PlotCurveItem(
            pen=pg.mkPen(color=QColor(0, 255, 128, 70), width=1, style=Qt.PenStyle.DotLine),
            brush=pg.mkBrush(0, 255, 128, 18),
            fillLevel=0,
        )
        self.plot_widget.addItem(self.fov_cone_item)

        # 2. Raw Measurements (Red Scatter Points)
        self.raw_scatter = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(color=QColor(255, 50, 50, 200), width=0.8),
            brush=pg.mkBrush(255, 60, 60, 140),
            symbol="x",
        )
        self.plot_widget.addItem(self.raw_scatter)

        # 3. Ground Truth Track (Blue Dashed Line)
        self.gt_track_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#00A2FF", width=1.8, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.gt_track_item)

        # 4. EKF Fusion Track (Green Solid Line)
        self.ekf_track_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#00FF66", width=2.2),
        )
        self.plot_widget.addItem(self.ekf_track_item)

        # 5. 2D Covariance Uncertainty Ellipse (Amber Solid Line)
        self.cov_ellipse_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#FFB300", width=1.8),
            brush=pg.mkBrush(255, 179, 0, 35),
            fillLevel=None,
        )
        self.plot_widget.addItem(self.cov_ellipse_item)

        # 6. Current Target Head Markers
        self.target_gt_head = pg.ScatterPlotItem(
            size=10,
            symbol="o",
            pen=pg.mkPen("#00A2FF", width=1.5),
            brush=pg.mkBrush(0, 162, 255, 120),
        )
        self.plot_widget.addItem(self.target_gt_head)

        self.target_ekf_head = pg.ScatterPlotItem(
            size=12,
            symbol="+",
            pen=pg.mkPen("#00FF66", width=2.0),
            brush=pg.mkBrush(0, 255, 102),
        )
        self.plot_widget.addItem(self.target_ekf_head)

    def update_frame(
        self,
        frame: SimulationFrame,
        gt_history: np.ndarray,
        ekf_history: np.ndarray,
        raw_tail: List[np.ndarray],
    ) -> None:
        """
        Updates radar layers with the current simulation frame.
        Note: In NED coordinates, x is North (plot Y), y is East (plot X).
        """
        # 1. Update Ground Truth Track
        if len(gt_history) > 0:
            east_gt = gt_history[:, 1]
            north_gt = gt_history[:, 0]
            self.gt_track_item.setData(east_gt, north_gt)
            self.target_gt_head.setData([east_gt[-1]], [north_gt[-1]])

        # 2. Update EKF Fusion Track
        if len(ekf_history) > 0:
            east_ekf = ekf_history[:, 1]
            north_ekf = ekf_history[:, 0]
            self.ekf_track_item.setData(east_ekf, north_ekf)
            self.target_ekf_head.setData([east_ekf[-1]], [north_ekf[-1]])

            # 3. Update Camera FOV Cone towards target
            self._update_fov_cone(uav_east=frame.uav_pos[1], uav_north=frame.uav_pos[0],
                                  tgt_east=east_ekf[-1], tgt_north=north_ekf[-1])

        # 4. Update Raw Measurements
        if raw_tail:
            raw_east = [p[1] for p in raw_tail]
            raw_north = [p[0] for p in raw_tail]
            self.raw_scatter.setData(raw_east, raw_north)
        else:
            self.raw_scatter.setData([], [])

        # 5. Update 2D Covariance Uncertainty Ellipse (2-Sigma)
        self._update_covariance_ellipse(
            center_east=frame.ekf_pos[1],
            center_north=frame.ekf_pos[0],
            cov_2d=frame.cov_2d,
            n_sigma=2.0,
        )

    def _update_fov_cone(self, uav_east: float, uav_north: float, tgt_east: float, tgt_north: float) -> None:
        """Draws a triangular FOV wedge from UAV pointing towards target."""
        dx = tgt_north - uav_north
        dy = tgt_east - uav_east
        dist = np.sqrt(dx * dx + dy * dy)
        bearing = np.arctan2(dy, dx)

        cone_range = min(dist * 1.25, 2000.0)
        half_angle = np.radians(18.0)  # ~36 degree FOV

        phi_left = bearing - half_angle
        phi_right = bearing + half_angle

        # Triangle vertices: UAV -> Left Cone Edge -> Right Cone Edge -> UAV
        x_cone = [uav_east, uav_east + cone_range * np.sin(phi_left), uav_east + cone_range * np.sin(phi_right), uav_east]
        y_cone = [uav_north, uav_north + cone_range * np.cos(phi_left), uav_north + cone_range * np.cos(phi_right), uav_north]

        self.fov_cone_item.setData(x_cone, y_cone)

    def _update_covariance_ellipse(
        self, center_east: float, center_north: float, cov_2d: np.ndarray, n_sigma: float = 2.0
    ) -> None:
        """
        Calculates and draws 2D covariance ellipse on East-North plane.
        cov_2d: [ [Pxx, Pxy], [Pyx, Pyy] ] where x=North, y=East.
        """
        # Reorder cov matrix to [East, North] for plotting: [[Pyy, Pyx], [Pxy, Pxx]]
        cov_en = np.array([
            [cov_2d[1, 1], cov_2d[1, 0]],
            [cov_2d[0, 1], cov_2d[0, 0]],
        ], dtype=np.float64)

        vals, vecs = np.linalg.eigh(cov_en)
        vals = np.maximum(vals, 1e-6)

        # Semi-major and semi-minor axes
        r_major = n_sigma * np.sqrt(vals[1])
        r_minor = n_sigma * np.sqrt(vals[0])
        angle = np.arctan2(vecs[1, 1], vecs[0, 1])

        # Generate ellipse boundary points
        phi = np.linspace(0, 2 * np.pi, 60)
        e_pts = r_major * np.cos(phi) * np.cos(angle) - r_minor * np.sin(phi) * np.sin(angle) + center_east
        n_pts = r_major * np.cos(phi) * np.sin(angle) + r_minor * np.sin(phi) * np.cos(angle) + center_north

        self.cov_ellipse_item.setData(e_pts, n_pts)
