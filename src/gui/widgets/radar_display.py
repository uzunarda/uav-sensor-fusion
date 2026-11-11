"""
2D Tactical Radar Display (PPI Scope) for UAV Multi-Sensor Fusion.
Renders Ownship UAV, radar range rings, azimuth bearing spokes, camera FOV cone,
ground truth track, EKF fusion estimate, 2D error covariance uncertainty ellipse,
and raw synthetic sensor observations.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

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
        layout.setSpacing(0)

        # Tactical MFD Header Bar
        header = QFrame()
        header.setFixedHeight(26)
        header.setStyleSheet("""
            QFrame {
                background-color: #0C131B;
                border-top: 1px solid #1C2C3B;
                border-bottom: 1px solid #1C2C3B;
                border-left: 3px solid #10B981;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(8)

        self.title_label = QLabel("[MOD-01A] 2B TAKTİK RADAR (PPI SCOPE // DOĞU-KUZEY DÜZLEMİ)")
        self.title_label.setStyleSheet("""
            color: #10B981;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0.8px;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(self.title_label)
        h_layout.addStretch(1)

        legend_label = QLabel("[- -] GERÇEK (GT)   [───] EKF FÜZYON   [ O ] 2σ KOVARYANS   [ x ] HAM ÖLÇÜM")
        legend_label.setStyleSheet("""
            color: #7FA0B8;
            font-family: 'Consolas', monospace;
            font-size: 9px;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(legend_label)
        layout.addWidget(header)

        # PyQtGraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#05080C")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.12)
        self.plot_widget.getAxis("bottom").setLabel("DOĞU EKSENİ (EAST / +Y)", units="m", color="#68899E")
        self.plot_widget.getAxis("left").setLabel("KUZEY EKSENİ (NORTH / +X)", units="m", color="#68899E")

        # Range bounds
        self.plot_widget.setXRange(-1600, 450)
        self.plot_widget.setYRange(-250, 1850)

        layout.addWidget(self.plot_widget, stretch=1)

        # Create static radar elements (rings, azimuth spokes, crosshairs)
        self._init_radar_grid()

        # Dynamic plot items
        self._init_dynamic_layers()

    def _init_radar_grid(self) -> None:
        """Draws concentric range rings (500m, 1000m, 1500m, 2000m) and azimuth spokes centered at UAV."""
        ring_pen = pg.mkPen(color=QColor(16, 185, 129, 75), width=1, style=Qt.PenStyle.DashLine)
        spoke_pen = pg.mkPen(color=QColor(16, 185, 129, 38), width=1, style=Qt.PenStyle.DotLine)

        phi = np.linspace(0, 2 * np.pi, 160)
        for r in [500.0, 1000.0, 1500.0, 2000.0]:
            x_ring = r * np.sin(phi)  # East
            y_ring = r * np.cos(phi)  # North
            ring_item = pg.PlotCurveItem(x_ring, y_ring, pen=ring_pen)
            self.plot_widget.addItem(ring_item)

            # Technical Range Ring Label
            label = pg.TextItem(f"RNG {int(r)}m", color=QColor(52, 211, 153, 145), anchor=(0.5, 1.15))
            label.setFont(QFont("Consolas", 8))
            label.setPos(0.0, r)
            self.plot_widget.addItem(label)

        # Azimuth Bearing Spokes every 45 degrees
        max_r = 2000.0
        for deg in [45, 135, 225, 315]:
            rad = np.radians(deg)
            x_end = max_r * np.sin(rad)
            y_end = max_r * np.cos(rad)
            self.plot_widget.addItem(pg.PlotCurveItem([0.0, x_end], [0.0, y_end], pen=spoke_pen))

        # Primary Cardinal Crosshairs through origin (UAV)
        axis_pen = pg.mkPen(color=QColor(16, 185, 129, 95), width=1)
        self.plot_widget.addItem(pg.PlotCurveItem([-2000, 2000], [0, 0], pen=axis_pen))
        self.plot_widget.addItem(pg.PlotCurveItem([0, 0], [-2000, 2000], pen=axis_pen))

        # Ownship UAV Marker at (0, 0)
        self.ownship_marker = pg.ScatterPlotItem(
            [0.0], [0.0],
            symbol="t1",
            size=13,
            pen=pg.mkPen("#00B4D8", width=1.6),
            brush=pg.mkBrush(0, 180, 216, 140),
        )
        self.plot_widget.addItem(self.ownship_marker)

        uav_label = pg.TextItem("OWNSHIP [İHA-01]", color=QColor(56, 189, 248, 210), anchor=(0.5, -0.5))
        uav_label.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        uav_label.setPos(0.0, 0.0)
        self.plot_widget.addItem(uav_label)

    def _init_dynamic_layers(self) -> None:
        """Initializes dynamic data curves and scatter layers."""
        # 1. Camera FOV Cone
        self.fov_cone_item = pg.PlotCurveItem(
            pen=pg.mkPen(color=QColor(16, 185, 129, 75), width=1, style=Qt.PenStyle.DotLine),
            brush=pg.mkBrush(16, 185, 129, 14),
            fillLevel=0,
        )
        self.plot_widget.addItem(self.fov_cone_item)

        # 2. Raw Synthetic Measurements (Amber/Red Crosshair Scatter Points)
        self.raw_scatter = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(color=QColor(239, 68, 68, 210), width=1.0),
            brush=pg.mkBrush(239, 68, 68, 140),
            symbol="x",
        )
        self.plot_widget.addItem(self.raw_scatter)

        # 3. Ground Truth Track (Cyan Dashed Line)
        self.gt_track_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#38BDF8", width=1.8, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.gt_track_item)

        # 4. EKF Fusion Track (Phosphor Green Solid Line)
        self.ekf_track_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#10B981", width=2.2),
        )
        self.plot_widget.addItem(self.ekf_track_item)

        # 5. 2D Covariance Uncertainty Ellipse (Amber Solid Line)
        self.cov_ellipse_item = pg.PlotCurveItem(
            pen=pg.mkPen(color="#F59E0B", width=1.6),
            brush=pg.mkBrush(245, 158, 11, 30),
            fillLevel=None,
        )
        self.plot_widget.addItem(self.cov_ellipse_item)

        # 6. Current Target Head Markers
        self.target_gt_head = pg.ScatterPlotItem(
            size=9,
            symbol="o",
            pen=pg.mkPen("#38BDF8", width=1.5),
            brush=pg.mkBrush(56, 189, 248, 110),
        )
        self.plot_widget.addItem(self.target_gt_head)

        self.target_ekf_head = pg.ScatterPlotItem(
            size=12,
            symbol="+",
            pen=pg.mkPen("#10B981", width=2.0),
            brush=pg.mkBrush(16, 185, 129),
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
            self._update_fov_cone(
                uav_east=frame.uav_pos[1],
                uav_north=frame.uav_pos[0],
                tgt_east=east_ekf[-1],
                tgt_north=north_ekf[-1],
            )

        # 4. Update Raw Synthetic Measurements
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
        cov_en = np.array([
            [cov_2d[1, 1], cov_2d[1, 0]],
            [cov_2d[0, 1], cov_2d[0, 0]],
        ], dtype=np.float64)

        vals, vecs = np.linalg.eigh(cov_en)
        vals = np.maximum(vals, 1e-6)

        r_major = n_sigma * np.sqrt(vals[1])
        r_minor = n_sigma * np.sqrt(vals[0])
        angle = np.arctan2(vecs[1, 1], vecs[0, 1])

        phi = np.linspace(0, 2 * np.pi, 60)
        e_pts = r_major * np.cos(phi) * np.cos(angle) - r_minor * np.sin(phi) * np.sin(angle) + center_east
        n_pts = r_major * np.cos(phi) * np.sin(angle) + r_minor * np.sin(phi) * np.cos(angle) + center_north

        self.cov_ellipse_item.setData(e_pts, n_pts)
