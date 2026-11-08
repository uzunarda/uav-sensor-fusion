"""
Tactical Telemetry, Tracking Error, and OOSM Threat Alert Panel.
Provides high-visibility HUD-style digital telemetry counters for 3D Velocity,
Altitude, Slant Range, Position Error, and dynamic flashing OOSM Rollback alert badge.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
)

from src.gui.worker import SimulationFrame


class TelemetryPanel(QWidget):
    """
    HUD-style Digital Telemetry and Sensor Status Panel.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Header Title
        title_label = QLabel("TELEMETRİ, HATA VE OOSM TEHDİT ANALİZİ (HUD TELEMETRY)")
        title_label.setStyleSheet("""
            QLabel {
                color: #FFB300;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
                background-color: #171208;
                padding: 5px 8px;
                border-bottom: 1px solid #362912;
            }
        """)
        main_layout.addWidget(title_label)

        # Content Frame
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #0A0F12;
                border: 1px solid #16242C;
                border-radius: 4px;
            }
        """)
        grid = QGridLayout(content_frame)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        # --- Row 1: Velocity & Speed ---
        self.val_speed = self._create_hud_card(grid, row=0, col=0, label="TOPLAM HIZ (|V|)", unit="m/s")
        self.val_vx = self._create_hud_card(grid, row=0, col=1, label="HIZ Vx (KUZEY)", unit="m/s")
        self.val_vy = self._create_hud_card(grid, row=0, col=2, label="HIZ Vy (DOĞU)", unit="m/s")

        # --- Row 2: Altitude, Range & Vz ---
        self.val_vz = self._create_hud_card(grid, row=1, col=0, label="DİKEY HIZ Vz", unit="m/s")
        self.val_alt = self._create_hud_card(grid, row=1, col=1, label="HEDEF İRTİFASI", unit="m")
        self.val_range = self._create_hud_card(grid, row=1, col=2, label="BAĞIL MESAFE (RANGE)", unit="m")

        # --- Row 3: Position & Velocity Errors ---
        self.val_pos_err = self._create_hud_card(grid, row=2, col=0, label="ANLIK 3B KONUM HATASI", unit="m", highlight=True)
        self.val_vel_err = self._create_hud_card(grid, row=2, col=1, label="ANLIK 3B HIZ HATASI", unit="m/s")
        self.val_last_sensor = self._create_hud_card(grid, row=2, col=2, label="SON SENSÖR", unit="")

        main_layout.addWidget(content_frame)

        # --- OOSM Threat Alert Badge Bar ---
        self.oosm_badge = QLabel("OOSM DURUMU: NOMİNAL (AKIŞ SIRALI)")
        self.oosm_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oosm_badge.setMinimumHeight(32)
        self._style_oosm_nominal()
        main_layout.addWidget(self.oosm_badge)

        # Timer to reset OOSM flash badge back to nominal after 150 ms
        self._oosm_flash_timer = QTimer(self)
        self._oosm_flash_timer.setSingleShot(True)
        self._oosm_flash_timer.setInterval(150)
        self._oosm_flash_timer.timeout.connect(self._style_oosm_nominal)

    def _create_hud_card(
        self, grid: QGridLayout, row: int, col: int, label: str, unit: str, highlight: bool = False
    ) -> QLabel:
        """Helper creating standardized tactical HUD telemetry cells."""
        box = QFrame()
        box.setStyleSheet("""
            QFrame {
                background-color: #0E161C;
                border: 1px solid #1C2D38;
                border-radius: 4px;
                padding: 4px 6px;
            }
        """)
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(2)

        lbl_caption = QLabel(label)
        lbl_caption.setStyleSheet("color: #7A9CAE; font-size: 9px; font-family: 'Consolas', monospace; font-weight: bold; letter-spacing: 0.5px;")
        vbox.addWidget(lbl_caption)

        val_color = "#00FF66" if not highlight else "#FFCC00"
        lbl_value = QLabel(f"-- {unit}")
        lbl_value.setStyleSheet(f"color: {val_color}; font-size: 14px; font-family: 'Consolas', monospace; font-weight: bold;")
        vbox.addWidget(lbl_value)

        grid.addWidget(box, row, col)
        return lbl_value

    def _style_oosm_nominal(self) -> None:
        """Styles OOSM badge for steady state nominal flow."""
        self.oosm_badge.setText("OOSM DURUMU: NOMİNAL (TAM DÜZENLİ AKIŞ)")
        self.oosm_badge.setStyleSheet("""
            QLabel {
                background-color: #101815;
                color: #4C8263;
                border: 1px solid #1E3325;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                letter-spacing: 1px;
            }
        """)

    def _style_oosm_alert(self, count: int) -> None:
        """Flashes OOSM badge in bright amber/orange when Rollback is triggered."""
        self.oosm_badge.setText(f"⚠️ OOSM ROLLBACK TRIGGERED! (SIRA DIŞI ÖLÇÜM TELAFİSİ #{count})")
        self.oosm_badge.setStyleSheet("""
            QLabel {
                background-color: #FF9900;
                color: #000000;
                border: 1px solid #FFCC00;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 6px 12px;
                letter-spacing: 1px;
            }
        """)

    def update_frame(self, frame: SimulationFrame) -> None:
        """Updates all telemetry digital meters."""
        # Velocities
        vx, vy, vz = frame.ekf_vel[0], frame.ekf_vel[1], frame.ekf_vel[2]
        speed_total = float(np.linalg.norm(frame.ekf_vel))

        self.val_speed.setText(f"{speed_total:5.1f} m/s ({speed_total * 3.6:.0f} km/h)")
        self.val_vx.setText(f"{vx:+5.1f} m/s")
        self.val_vy.setText(f"{vy:+5.1f} m/s")
        self.val_vz.setText(f"{vz:+5.1f} m/s")

        # Altitude and Distance
        self.val_alt.setText(f"{frame.altitude:6.1f} m")
        self.val_range.setText(f"{frame.range_to_target:6.1f} m")

        # Errors
        self.val_pos_err.setText(f"{frame.pos_error_3d:5.2f} m")
        self.val_vel_err.setText(f"{frame.vel_error_3d:5.2f} m/s")

        # Sensor info
        sensor_text = frame.latest_raw_type if frame.latest_raw_type else "SENKRON"
        self.val_last_sensor.setText(sensor_text)

        # Trigger flashing alert badge if an OOSM rollback was performed on this frame
        if frame.is_oosm_triggered:
            self._style_oosm_alert(frame.oosm_total_count)
            self._oosm_flash_timer.start()  # Active for 150 ms
