"""
3D Trajectory and Altitude View Widget using pyqtgraph.opengl (GLViewWidget).
Visualizes 3D Cartesian coordinates of Target Ground Truth, EKF Estimated Track,
Ownship UAV, 3D Coordinate Axes (North, East, Altitude), and the 3D LOS Approach Vector in real-time.
Includes automatic 2D Altitude Profile fallback if hardware OpenGL is unavailable.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QVector3D
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame

import pyqtgraph as pg

# Try importing PyOpenGL / GLViewWidget
OPENGL_AVAILABLE = False
try:
    import pyqtgraph.opengl as gl
    OPENGL_AVAILABLE = True
except Exception:
    OPENGL_AVAILABLE = False

from src.gui.worker import SimulationFrame


class Trajectory3DDisplay(QWidget):
    """
    3D Trajectory & Altitude Visualization Widget with Full Coordinate Axes.
    Renders 3D tracks via GLViewWidget, or falls back to 2D Altitude profile.
    Coordinates translated: X = East, Y = North, Z = Altitude (-Down).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Single Compact Tactical MFD Header (Title on left, Legend on right)
        header = QFrame()
        header.setFixedHeight(26)
        header.setStyleSheet("""
            QFrame {
                background-color: #0C131B;
                border-top: 1px solid #1C2C3B;
                border-bottom: 1px solid #1C2C3B;
                border-left: 3px solid #38BDF8;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(8)

        title_label = QLabel("[MOD-01B] 3B UZAYSAL YÖRÜNGE VE İRTİFA GÖRÜNÜMÜ (NED // OPENGL)")
        title_label.setStyleSheet("""
            color: #38BDF8;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0.8px;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(title_label)

        h_layout.addStretch(1)

        legend_label = QLabel("[+Y] KUZEY   [+X] DOĞU   [+Z] İRTİFA   [───] LOS / İZDÜŞÜM")
        legend_label.setStyleSheet("""
            color: #7FA0B8;
            font-family: 'Consolas', monospace;
            font-size: 9px;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(legend_label)

        self.layout.addWidget(header, stretch=0)

        self.use_gl = False
        if OPENGL_AVAILABLE:
            try:
                self._init_gl_view()
                self.use_gl = True
            except Exception as e:
                print(f"[GUI 3D] OpenGL initialization failed ({e}), falling back to 2D Altitude view.")
                self.use_gl = False

        if not self.use_gl:
            self._init_fallback_2d_view()

    # -------------------------------------------------------------------------
    # 3D OpenGL Implementation
    # -------------------------------------------------------------------------

    def _init_gl_view(self) -> None:
        """Initializes 3D scene with OpenGL grid, coordinate axes, and line plot items."""
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor("#04080B")
        self.gl_widget.setCameraPosition(distance=2500, elevation=28, azimuth=-60)
        self.layout.addWidget(self.gl_widget, stretch=1)

        # 1. 3D Ground Grid (meters)
        grid = gl.GLGridItem()
        grid.setSize(x=3600, y=3600, z=0)
        grid.setSpacing(x=300, y=300, z=0)
        grid.setColor(QColor(20, 60, 90, 80))
        self.gl_widget.addItem(grid)

        # 2. Build 3D Coordinate Axes with Ticks and Labels
        self._build_coordinate_axes()

        # 3. 3D Track: Ground Truth (Blue)
        # Coordinates in OpenGL display: X = East, Y = North, Z = Altitude (-z)
        self.gl_gt_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=(0.0, 0.65, 1.0, 0.8),
            width=2.0,
            antialias=True,
            mode="line_strip",
        )
        self.gl_widget.addItem(self.gl_gt_line)

        # 4. 3D Track: EKF Estimate (Green)
        self.gl_ekf_line = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=(0.0, 1.0, 0.4, 0.95),
            width=2.5,
            antialias=True,
            mode="line_strip",
        )
        self.gl_widget.addItem(self.gl_ekf_line)

        # 5. 3D Approach LOS Vector (Ownship -> Target)
        self.gl_los_vector = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(1.0, 0.7, 0.0, 0.6),
            width=1.5,
            antialias=True,
            mode="lines",
        )
        self.gl_widget.addItem(self.gl_los_vector)

        # 6. Target Altitude Drop Line (Vertical projection to Ground Z=0)
        self.gl_drop_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)),
            color=(1.0, 0.85, 0.0, 0.75),
            width=1.5,
            antialias=True,
            mode="lines",
        )
        self.gl_widget.addItem(self.gl_drop_line)

        # 7. Target Ground Footprint (Shadow point at Z=0)
        self.gl_ground_shadow = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1.0, 0.85, 0.0, 0.85),
            size=8,
            pxMode=True,
        )
        self.gl_widget.addItem(self.gl_ground_shadow)

        # 8. 3D Ownship UAV Symbol (Cyan diamond/box)
        self.gl_uav_point = gl.GLScatterPlotItem(
            pos=np.array([[0.0, 0.0, 150.0]]),
            color=(0.0, 1.0, 1.0, 1.0),
            size=14,
            pxMode=True,
        )
        self.gl_widget.addItem(self.gl_uav_point)

        # 9. 3D Target Symbol (Green circle)
        self.gl_tgt_point = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(0.0, 1.0, 0.4, 1.0),
            size=12,
            pxMode=True,
        )
        self.gl_widget.addItem(self.gl_tgt_point)

    def _build_coordinate_axes(self) -> None:
        """Constructs dedicated, high-visibility 3D coordinate axes with distance ticks and labels."""
        # Standard RGB Origin Triad
        try:
            axis_triad = gl.GLAxisItem()
            axis_triad.setSize(x=500, y=500, z=300)
            self.gl_widget.addItem(axis_triad)
        except Exception:
            pass

        # High-visibility Tactical Coordinate Axes lines
        # X: East (Doğu) -> Cyan
        # Y: North (Kuzey) -> Red / Amber
        # Z: Altitude (İrtifa) -> Green
        axis_pts: List[List[float]] = [
            # +X East
            [0.0, 0.0, 0.0], [1800.0, 0.0, 0.0],
            # +Y North
            [0.0, 0.0, 0.0], [0.0, 1800.0, 0.0],
            # +Z Altitude
            [0.0, 0.0, 0.0], [0.0, 0.0, 900.0],
            # -X West (subtle)
            [0.0, 0.0, 0.0], [-1200.0, 0.0, 0.0],
            # -Y South (subtle)
            [0.0, 0.0, 0.0], [0.0, -1200.0, 0.0],
        ]

        # Ticks along East (+X) at 500, 1000, 1500m
        for x_val in [500.0, 1000.0, 1500.0]:
            axis_pts.extend([[x_val, -40.0, 0.0], [x_val, 40.0, 0.0]])

        # Ticks along North (+Y) at 500, 1000, 1500m
        for y_val in [500.0, 1000.0, 1500.0]:
            axis_pts.extend([[-40.0, y_val, 0.0], [40.0, y_val, 0.0]])

        # Ticks along Altitude (+Z) at 300, 600m
        for z_val in [300.0, 600.0]:
            axis_pts.extend([[-30.0, 0.0, z_val], [30.0, 0.0, z_val]])

        axis_colors: List[tuple] = [
            # +X East (Cyan)
            (0.0, 0.85, 1.0, 0.95), (0.0, 0.85, 1.0, 0.95),
            # +Y North (Amber/Red)
            (1.0, 0.35, 0.2, 0.95), (1.0, 0.35, 0.2, 0.95),
            # +Z Altitude (Bright Green)
            (0.2, 1.0, 0.4, 0.95), (0.2, 1.0, 0.4, 0.95),
            # -X West (Faded Cyan)
            (0.0, 0.4, 0.6, 0.35), (0.0, 0.4, 0.6, 0.35),
            # -Y South (Faded Red)
            (0.6, 0.2, 0.2, 0.35), (0.6, 0.2, 0.2, 0.35),
        ]

        # X Ticks
        for _ in range(3):
            axis_colors.extend([(0.0, 0.85, 1.0, 0.7), (0.0, 0.85, 1.0, 0.7)])

        # Y Ticks
        for _ in range(3):
            axis_colors.extend([(1.0, 0.35, 0.2, 0.7), (1.0, 0.35, 0.2, 0.7)])

        # Z Ticks
        for _ in range(2):
            axis_colors.extend([(0.2, 1.0, 0.4, 0.7), (0.2, 1.0, 0.4, 0.7)])

        self.gl_axis_lines = gl.GLLinePlotItem(
            pos=np.array(axis_pts, dtype=np.float32),
            color=np.array(axis_colors, dtype=np.float32),
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self.gl_widget.addItem(self.gl_axis_lines)

        # 3D Text Labels at Axis Ends
        try:
            self.lbl_x = gl.GLTextItem(pos=np.array([1850.0, 0.0, 0.0]), text="+X DOGU (EAST)", color=QColor(0, 220, 255))
            self.lbl_y = gl.GLTextItem(pos=np.array([0.0, 1850.0, 0.0]), text="+Y KUZEY (NORTH)", color=QColor(255, 90, 60))
            self.lbl_z = gl.GLTextItem(pos=np.array([0.0, 0.0, 920.0]), text="+Z IRTIFA (ALT)", color=QColor(50, 255, 100))
            self.gl_widget.addItem(self.lbl_x)
            self.gl_widget.addItem(self.lbl_y)
            self.gl_widget.addItem(self.lbl_z)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 2D Altitude Profile Fallback Implementation
    # -------------------------------------------------------------------------

    def _init_fallback_2d_view(self) -> None:
        """Fallback 2D Altitude Profile (North vs Altitude & East vs Altitude)."""
        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setBackground("#050A0E")
        self.plot_2d.showGrid(x=True, y=True, alpha=0.2)
        self.plot_2d.getAxis("bottom").setLabel("Uzaklık (Horizontal Range)", units="m", color="#5C8A9F")
        self.plot_2d.getAxis("left").setLabel("İrtifa (Altitude)", units="m", color="#5C8A9F")

        self.fb_gt_curve = pg.PlotCurveItem(pen=pg.mkPen("#00A2FF", width=2, style=Qt.PenStyle.DashLine))
        self.fb_ekf_curve = pg.PlotCurveItem(pen=pg.mkPen("#00FF66", width=2))
        self.fb_uav_point = pg.ScatterPlotItem(size=12, symbol="t1", pen=pg.mkPen("#00FFFF"), brush=pg.mkBrush("#008888"))
        self.fb_tgt_point = pg.ScatterPlotItem(size=10, symbol="o", pen=pg.mkPen("#00FF66"), brush=pg.mkBrush("#00FF66"))

        self.plot_2d.addItem(self.fb_gt_curve)
        self.plot_2d.addItem(self.fb_ekf_curve)
        self.plot_2d.addItem(self.fb_uav_point)
        self.plot_2d.addItem(self.fb_tgt_point)

        self.layout.addWidget(self.plot_2d, stretch=1)

    # -------------------------------------------------------------------------
    # Update Loop
    # -------------------------------------------------------------------------

    def update_frame(
        self,
        frame: SimulationFrame,
        gt_history: np.ndarray,
        ekf_history: np.ndarray,
        raw_tail: List[np.ndarray],
    ) -> None:
        """
        Updates 3D or 2D display with latest track positions.
        Converts NED [North, East, Down] to Display [East(X), North(Y), Altitude -Down(Z)].
        """
        uav_disp = np.array([frame.uav_pos[1], frame.uav_pos[0], -frame.uav_pos[2]])
        tgt_disp = np.array([frame.ekf_pos[1], frame.ekf_pos[0], -frame.ekf_pos[2]])

        if self.use_gl:
            try:
                # Update GT Line
                if len(gt_history) > 1:
                    pts_gt = np.column_stack([gt_history[:, 1], gt_history[:, 0], -gt_history[:, 2]])
                    self.gl_gt_line.setData(pos=pts_gt)

                # Update EKF Line
                if len(ekf_history) > 1:
                    pts_ekf = np.column_stack([ekf_history[:, 1], ekf_history[:, 0], -ekf_history[:, 2]])
                    self.gl_ekf_line.setData(pos=pts_ekf)

                # Update UAV and Target points
                self.gl_uav_point.setData(pos=np.array([uav_disp]))
                self.gl_tgt_point.setData(pos=np.array([tgt_disp]))

                # Update Approach LOS Vector: [UAV -> Target]
                los_pts = np.vstack([uav_disp, tgt_disp])
                self.gl_los_vector.setData(pos=los_pts)

                # Update Drop Line: from target down to ground Z=0
                ground_pt = np.array([tgt_disp[0], tgt_disp[1], 0.0])
                drop_pts = np.vstack([tgt_disp, ground_pt])
                self.gl_drop_line.setData(pos=drop_pts)
                self.gl_ground_shadow.setData(pos=np.array([ground_pt]))
            except Exception:
                pass
        else:
            # Fallback 2D view: Horizontal Distance from UAV vs Altitude
            if len(gt_history) > 0:
                dist_gt = np.linalg.norm(gt_history[:, [0, 1]] - frame.uav_pos[[0, 1]], axis=1)
                alt_gt = -gt_history[:, 2]
                self.fb_gt_curve.setData(dist_gt, alt_gt)

            if len(ekf_history) > 0:
                dist_ekf = np.linalg.norm(ekf_history[:, [0, 1]] - frame.uav_pos[[0, 1]], axis=1)
                alt_ekf = -ekf_history[:, 2]
                self.fb_ekf_curve.setData(dist_ekf, alt_ekf)
                self.fb_tgt_point.setData([dist_ekf[-1]], [alt_ekf[-1]])

            self.fb_uav_point.setData([0.0], [-frame.uav_pos[2]])
