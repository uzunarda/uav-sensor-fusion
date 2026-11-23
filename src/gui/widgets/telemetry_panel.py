"""
Tactical Telemetry, Tracking Error, Literature Metrics (OSPA, NEES, IDF1, Exec Time) and OOSM Alert Panel.
Provides digital telemetry counters for 3D Velocity, Altitude, Slant Range,
Position/Velocity Error, OSPA/MAE, NEES/NIS consistency, IDF1 Track Continuity, and dynamic OOSM status bar.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
)

from src.gui.worker import SimulationFrame


class TelemetryPanel(QWidget):
    """
    Avionics MFD Digital Telemetry, Academic Metrics, and OOSM Status Panel.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Tactical MFD Header Bar
        header = QFrame()
        header.setFixedHeight(26)
        header.setStyleSheet("""
            QFrame {
                background-color: #0C131B;
                border-top: 1px solid #1C2C3B;
                border-bottom: 1px solid #1C2C3B;
                border-left: 3px solid #F59E0B;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(8)

        title_label = QLabel("[MOD-01D] TAKTİK TELEMETRİ, KİNEMATİK VE BAŞARI METRİKLERİ KONSOLU")
        title_label.setStyleSheet("""
            color: #F59E0B;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0.8px;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(title_label)
        h_layout.addStretch(1)

        self.lbl_mode_tag = QLabel("FİLTRE: JOSEPH-EKF // 6-DOF NED")
        self.lbl_mode_tag.setStyleSheet("""
            color: #7FA0B8;
            font-family: 'Consolas', monospace;
            font-size: 9px;
            font-weight: bold;
            background: transparent;
            border: none;
        """)
        h_layout.addWidget(self.lbl_mode_tag)
        main_layout.addWidget(header)

        # Content Frame (4x3 Tactical Grid)
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #070B10;
                border: 1px solid #16222E;
            }
        """)
        grid = QGridLayout(content_frame)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(5)

        # --- Row 1: Velocity & Speed ---
        self.val_speed = self._create_hud_card(grid, row=0, col=0, code="KIN-01", label="TOPLAM HIZ (|V|)", unit="m/s")
        self.val_vx = self._create_hud_card(grid, row=0, col=1, code="KIN-02", label="KUZEY HIZI (Vx)", unit="m/s")
        self.val_vy = self._create_hud_card(grid, row=0, col=2, code="KIN-03", label="DOĞU HIZI (Vy)", unit="m/s")

        # --- Row 2: Altitude, Range & Vz ---
        self.val_vz = self._create_hud_card(grid, row=1, col=0, code="KIN-04", label="DİKEY HIZ (Vz)", unit="m/s")
        self.val_alt = self._create_hud_card(grid, row=1, col=1, code="GEO-01", label="HEDEF İRTİFASI (-Z)", unit="m")
        self.val_range = self._create_hud_card(grid, row=1, col=2, code="GEO-02", label="EĞİK MENZİL (RANGE)", unit="m")

        # --- Row 3: Position Error, OSPA/MAE & Velocity Error ---
        self.val_pos_err = self._create_hud_card(grid, row=2, col=0, code="ERR-01", label="3B KONUM HATASI (|e_p|)", unit="m", highlight=True)
        self.val_ospa = self._create_hud_card(grid, row=2, col=1, code="MET-01", label="OSPA SKORU / MAE", unit="m", highlight=True)
        self.val_vel_err = self._create_hud_card(grid, row=2, col=2, code="ERR-02", label="3B HIZ HATASI (|e_v|)", unit="m/s")

        # --- Row 4: Literature Consistency & Continuity Metrics ---
        self.val_nees = self._create_hud_card(grid, row=3, col=0, code="MET-02", label="TUTARLILIK (NEES / NIS)", unit="")
        self.val_idf1 = self._create_hud_card(grid, row=3, col=1, code="MET-03", label="İZ SÜREKLİLİĞİ & IDF1", unit="")
        self.val_last_sensor = self._create_hud_card(grid, row=3, col=2, code="LNK-01", label="OOSM SAYAC / SÜRE", unit="")

        main_layout.addWidget(content_frame, stretch=1)

        # --- OOSM Threat Alert Status Bar ---
        self.oosm_badge = QLabel("OOSM TAMPON DURUMU: [NOMİNAL] // KRONOLOJİK ÖLÇÜM AKIŞI SENKRON")
        self.oosm_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oosm_badge.setFixedHeight(26)
        self._style_oosm_nominal()
        main_layout.addWidget(self.oosm_badge)

        # Timer to reset OOSM flash badge back to nominal after 180 ms
        self._oosm_flash_timer = QTimer(self)
        self._oosm_flash_timer.setSingleShot(True)
        self._oosm_flash_timer.setInterval(180)
        self._oosm_flash_timer.timeout.connect(self._style_oosm_nominal)

    def _create_hud_card(
        self, grid: QGridLayout, row: int, col: int, code: str, label: str, unit: str, highlight: bool = False
    ) -> QLabel:
        """Helper creating standardized sharp tactical telemetry cells."""
        accent = "#F59E0B" if highlight else "#1E3346"
        box = QFrame()
        box.setStyleSheet(f"""
            QFrame {{
                background-color: #0B1017;
                border: 1px solid #172533;
                border-left: 2px solid {accent};
            }}
        """)
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(6, 3, 6, 3)
        vbox.setSpacing(1)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        lbl_caption = QLabel(label)
        lbl_caption.setStyleSheet("color: #7FA0B8; font-size: 8.5px; font-family: 'Consolas', monospace; font-weight: bold; border: none; background: transparent;")
        lbl_code = QLabel(f"[{code}]")
        lbl_code.setStyleSheet("color: #466378; font-size: 8px; font-family: 'Consolas', monospace; font-weight: bold; border: none; background: transparent;")
        top_row.addWidget(lbl_caption)
        top_row.addStretch(1)
        top_row.addWidget(lbl_code)
        vbox.addLayout(top_row)

        val_color = "#F59E0B" if highlight else "#10B981"
        lbl_value = QLabel(f"-- {unit}")
        lbl_value.setStyleSheet(f"color: {val_color}; font-size: 11.5px; font-family: 'Consolas', monospace; font-weight: bold; border: none; background: transparent;")
        vbox.addWidget(lbl_value)

        grid.addWidget(box, row, col)
        return lbl_value

    def _style_oosm_nominal(self) -> None:
        """Styles OOSM badge for steady state nominal flow."""
        self.oosm_badge.setText("OOSM TAMPON DURUMU: [NOMİNAL] // KRONOLOJİK ÖLÇÜM AKIŞI SENKRON")
        self.oosm_badge.setStyleSheet("""
            QLabel {
                background-color: #091210;
                color: #34D399;
                border: 1px solid #143326;
                border-left: 3px solid #10B981;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 8px;
                letter-spacing: 0.6px;
            }
        """)

    def _style_oosm_alert(self, count: int, mode: str) -> None:
        """Flashes OOSM badge in high-contrast military amber when OOSM compensation is triggered."""
        method_tag = {
            "MODE_C": "ROLLBACK/REPLAY",
            "MODE_D": "1-ADIM RETRODICTION [MOD-EKF]",
            "MODE_E": "KALMAN PREDICTOR [İLERİ TAŞIMA]",
        }.get(mode, "OOSM TELAFİ")
        self.oosm_badge.setText(f"[UYARI: {method_tag} AKTİF] GECİKMELİ PAKET TELAFİ EDİLDİ // TOPLAM: #{count:03d}")
        self.oosm_badge.setStyleSheet("""
            QLabel {
                background-color: #D97706;
                color: #05080C;
                border: 1px solid #FBBF24;
                border-left: 4px solid #FFFFFF;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 8px;
                letter-spacing: 0.6px;
            }
        """)

    def update_frame(self, frame: SimulationFrame) -> None:
        """Updates all telemetry digital meters and academic metrics."""
        vx, vy, vz = frame.ekf_vel[0], frame.ekf_vel[1], frame.ekf_vel[2]
        speed_total = float(np.linalg.norm(frame.ekf_vel))

        self.val_speed.setText(f"{speed_total:5.1f} m/s ({speed_total * 3.6:3.0f} km/h)")
        self.val_vx.setText(f"{vx:+6.1f} m/s")
        self.val_vy.setText(f"{vy:+6.1f} m/s")
        self.val_vz.setText(f"{vz:+6.1f} m/s")

        # Altitude and Distance
        self.val_alt.setText(f"{frame.altitude:6.1f} m")
        self.val_range.setText(f"{frame.range_to_target:6.1f} m")

        # Errors & OSPA/MAE
        self.val_pos_err.setText(f"{frame.pos_error_3d:5.2f} m")
        self.val_ospa.setText(f"{frame.ospa_val:4.2f}m | MAE:{frame.mae_pos:4.2f}m")
        self.val_vel_err.setText(f"{frame.vel_error_3d:5.2f} m/s")

        # NEES / NIS & IDF1 / Track Continuity
        self.val_nees.setText(f"NEES:{frame.nees_val:4.1f} | NIS:{frame.nis_val:3.1f}")
        self.val_idf1.setText(f"%{frame.track_continuity_pct:4.1f} | IDF1:{frame.idf1_score:.2f}")

        # Sensor info, cumulative OOSM count & step execution time
        sensor_text = frame.latest_raw_type if frame.latest_raw_type else "SENKRON"
        self.val_last_sensor.setText(f"{sensor_text[:3]} OOSM:{frame.oosm_total_count:03d} | {frame.step_exec_us:3.0f}us")

        mode_header = {
            "MODE_A": "MOD A: İDEAL (0ms GECİKME)",
            "MODE_B": "MOD B: NAIVE EKF (TELAFİSİZ)",
            "MODE_C": "MOD C: OOSM ROLLBACK / REPLAY",
            "MODE_D": "MOD D: MODİFİYE EKF (RETRODICTION)",
            "MODE_E": "MOD E: KALMAN PREDICTOR (EXTRAP)",
        }.get(frame.sim_mode, "JOSEPH-EKF // 6-DOF NED")
        self.lbl_mode_tag.setText(mode_header)

        # Trigger flashing alert badge if an OOSM compensation was performed on this frame
        if frame.is_oosm_triggered:
            self._style_oosm_alert(frame.oosm_total_count, frame.sim_mode)
            self._oosm_flash_timer.start()
