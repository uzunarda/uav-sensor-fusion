"""
Live Synthetic Sensor Data Stream & Ground-Truth / Raw / EKF Comparison Panel.
Displays real-time incoming synthetic RADAR and EO/CAMERA measurement packets,
capture vs. receive timestamps, transmission/processing latencies, Gaussian noise deltas,
and side-by-side numeric comparison of Ground Truth, Raw Synthetic Sensor, and EKF Fusion states.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from src.gui.worker import SimulationFrame


class SensorDataStreamPanel(QWidget):
    """
    Live Synthetic Sensor Packet Inspector and
    Ground Truth vs. Raw Measurement vs. EKF State Comparison Matrix.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._last_seq_id: int = -1
        self._last_mode: str = ""

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(2)

        # =====================================================================
        # 1. Tactical Module Header Bar
        # =====================================================================
        header = QFrame()
        header.setFixedHeight(26)
        header.setStyleSheet("""
            QFrame {
                background-color: #0C131B;
                border-top: 1px solid #1C2C3B;
                border-bottom: 1px solid #1C2C3B;
                border-left: 3px solid #00B4D8;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(8)

        title_lbl = QLabel("[MOD-01C] SENTETİK SENSÖR VERİ AKIŞI VE GT / HAM / EKF KIYASLAMA")
        title_lbl.setStyleSheet("""
            color: #00B4D8;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0.8px;
            border: none;
            background: transparent;
        """)
        h_layout.addWidget(title_lbl)
        h_layout.addStretch(1)

        self.lbl_pkt_counter = QLabel("PKT: #0000  |  SON GECİKME: 0.0 ms")
        self.lbl_pkt_counter.setStyleSheet("""
            color: #8BA4B8;
            font-family: 'Consolas', monospace;
            font-size: 10px;
            font-weight: bold;
            border: none;
            background: transparent;
        """)
        h_layout.addWidget(self.lbl_pkt_counter)
        root_layout.addWidget(header)

        # =====================================================================
        # 2. 3-Column Comparison Strip: Ground Truth vs. Synthetic Raw vs. EKF
        # =====================================================================
        cmp_frame = QFrame()
        cmp_frame.setStyleSheet("""
            QFrame {
                background-color: #080C11;
                border: 1px solid #16222E;
            }
        """)
        cmp_grid = QGridLayout(cmp_frame)
        cmp_grid.setContentsMargins(6, 5, 6, 5)
        cmp_grid.setHorizontalSpacing(6)
        cmp_grid.setVerticalSpacing(3)

        # Column 0: Ground Truth (Gerçek Hedef)
        self.lbl_gt_pos, self.lbl_gt_polar = self._create_comparison_column(
            cmp_grid,
            col=0,
            title="GERÇEK HEDEF [GROUND TRUTH]",
            accent="#38BDF8",
            line1_default="K: +0000.0  D: +0000.0  İRT: 000.0m",
            line2_default="R_gt: 0000.0m | AZ: +00.0° EL: +00.0°",
        )

        # Column 1: Synthetic Raw Sensor (Üretilen Gürültülü Ölçüm)
        self.lbl_raw_meas, self.lbl_raw_cart = self._create_comparison_column(
            cmp_grid,
            col=1,
            title="SENTETİK HAM SENSÖR [RAW Z_k]",
            accent="#F59E0B",
            line1_default="R_m : 0000.0m | AZ: +00.0° EL: +00.0°",
            line2_default="HAM : K:+0000.0 D:+0000.0 İRT:000.0m",
        )

        # Column 2: EKF Fused Estimate (EKF Füzyon Kestirimi)
        self.lbl_ekf_pos, self.lbl_ekf_err = self._create_comparison_column(
            cmp_grid,
            col=2,
            title="EKF FÜZYON KESTİRİMİ [X_hat]",
            accent="#10B981",
            line1_default="K: +0000.0  D: +0000.0  İRT: 000.0m",
            line2_default="KONUM FARK: 0.00m | HIZ FARK: 0.00m/s",
        )

        root_layout.addWidget(cmp_frame)

        # =====================================================================
        # 3. Live Scrolling Synthetic Sensor Packet Table
        # =====================================================================
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "PKT#",
            "SENSÖR",
            "T_CAP (s)",
            "T_RX (s)",
            "GECİKME",
            "SENTETİK HAM ÖLÇÜM (z_k)",
            "GÜRÜLTÜ (ε) / DURUM",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setAlternatingRowColors(True)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self.table.setColumnWidth(0, 52)
        self.table.setColumnWidth(1, 64)
        self.table.setColumnWidth(2, 68)
        self.table.setColumnWidth(3, 68)
        self.table.setColumnWidth(4, 66)

        self.table.verticalHeader().setDefaultSectionSize(21)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #06090D;
                alternate-background-color: #090E14;
                color: #B8CCD8;
                gridline-color: #131E29;
                border: 1px solid #16222E;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
            }
            QHeaderView::section {
                background-color: #0D151E;
                color: #7FA0B8;
                padding: 3px 4px;
                border: none;
                border-right: 1px solid #16222E;
                border-bottom: 1px solid #1E3042;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 9px;
            }
            QTableWidget::item {
                padding: 1px 4px;
            }
            QTableWidget::item:selected {
                background-color: #132738;
                color: #FFFFFF;
            }
        """)
        root_layout.addWidget(self.table, stretch=1)

    def _create_comparison_column(
        self,
        grid: QGridLayout,
        col: int,
        title: str,
        accent: str,
        line1_default: str,
        line2_default: str,
    ) -> tuple[QLabel, QLabel]:
        """Creates a sharp, military-grade comparison cell."""
        cell = QFrame()
        cell.setStyleSheet(f"""
            QFrame {{
                background-color: #0B1017;
                border: 1px solid #182634;
                border-top: 2px solid {accent};
            }}
        """)
        vbox = QVBoxLayout(cell)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"""
            color: {accent};
            font-family: 'Consolas', monospace;
            font-size: 9px;
            font-weight: bold;
            letter-spacing: 0.5px;
            border: none;
            background: transparent;
        """)
        vbox.addWidget(lbl_title)

        lbl_line1 = QLabel(line1_default)
        lbl_line1.setStyleSheet("""
            color: #E2EEF8;
            font-family: 'Consolas', monospace;
            font-size: 10px;
            font-weight: bold;
            border: none;
            background: transparent;
        """)
        vbox.addWidget(lbl_line1)

        lbl_line2 = QLabel(line2_default)
        lbl_line2.setStyleSheet("""
            color: #8BA4B8;
            font-family: 'Consolas', monospace;
            font-size: 9px;
            border: none;
            background: transparent;
        """)
        vbox.addWidget(lbl_line2)

        grid.addWidget(cell, 0, col)
        return lbl_line1, lbl_line2

    def update_frame(self, frame: SimulationFrame) -> None:
        """Updates synthetic data comparison cards and live scrolling packet table."""
        # 1. Update Ground Truth Column
        gt_x, gt_y, gt_z = frame.target_gt_pos[0], frame.target_gt_pos[1], frame.target_gt_pos[2]
        gt_alt = -gt_z
        self.lbl_gt_pos.setText(f"K:{gt_x:+7.1f}  D:{gt_y:+7.1f}  İRT:{gt_alt:5.1f}m")

        r_gt = frame.latest_radar_gt if frame.latest_radar_gt is not None else float(np.linalg.norm(frame.target_gt_pos - frame.uav_pos))
        if frame.latest_cam_gt is not None:
            psi_gt, lam_gt = frame.latest_cam_gt
        else:
            rel = frame.target_gt_pos - frame.uav_pos
            psi_gt = float(np.degrees(np.arctan2(rel[1], rel[0])))
            lam_gt = float(np.degrees(np.arcsin(np.clip(rel[2] / max(r_gt, 1e-6), -1.0, 1.0))))
        self.lbl_gt_polar.setText(f"R_gt:{r_gt:6.1f}m | AZ:{psi_gt:+6.2f}° EL:{lam_gt:+6.2f}°")

        # 2. Update Synthetic Raw Sensor Column
        r_meas_str = f"{frame.latest_radar_meas:6.1f}m" if frame.latest_radar_meas is not None else "  --  "
        if frame.latest_cam_meas is not None:
            psi_m, lam_m = frame.latest_cam_meas
            cam_meas_str = f"AZ:{psi_m:+6.2f}° EL:{lam_m:+6.2f}°"
        else:
            cam_meas_str = "AZ:  --  ° EL:  --  °"
        self.lbl_raw_meas.setText(f"R_m :{r_meas_str} | {cam_meas_str}")

        if frame.latest_raw_pos is not None:
            rx, ry, rz = frame.latest_raw_pos[0], frame.latest_raw_pos[1], frame.latest_raw_pos[2]
            self.lbl_raw_cart.setText(f"HAM : K:{rx:+7.1f} D:{ry:+7.1f} İRT:{-rz:5.1f}m")
        else:
            self.lbl_raw_cart.setText("HAM : İLK RADAR+EO EŞLEŞMESİ BEKLENİYOR")

        # 3. Update EKF Fusion Column
        ex, ey, ez = frame.ekf_pos[0], frame.ekf_pos[1], frame.ekf_pos[2]
        self.lbl_ekf_pos.setText(f"K:{ex:+7.1f}  D:{ey:+7.1f}  İRT:{-ez:5.1f}m")
        self.lbl_ekf_err.setText(f"KONUM HATASI: {frame.pos_error_3d:5.2f}m | HIZ: {frame.vel_error_3d:4.2f}m/s")

        # Header packet counter
        self.lbl_pkt_counter.setText(
            f"PKT: #{frame.packet_seq_id:04d}  |  SON SENSÖR: {frame.latest_raw_type or '--'}  |  Δτ: {frame.latency_ms:4.1f} ms"
        )

        # 4. Update Packet Table only if sequence ID or mode changed
        if frame.packet_seq_id == self._last_seq_id and frame.sim_mode == self._last_mode:
            return
        self._last_seq_id = frame.packet_seq_id
        self._last_mode = frame.sim_mode

        packets = list(reversed(frame.recent_packets))  # Newest packet at row 0
        self.table.setRowCount(len(packets))

        font_mono = QFont("Consolas", 9)
        font_bold = QFont("Consolas", 9, QFont.Weight.Bold)

        for r_idx, pkt in enumerate(packets):
            sensor_label = "RADAR" if pkt.sensor_type == "RADAR" else "EO/IR"
            status_combined = f"{pkt.noise_delta_str} [{pkt.status}]"

            row_vals = [
                f"#{pkt.seq_id:04d}",
                sensor_label,
                f"{pkt.t_capture:6.3f}",
                f"{pkt.t_receive:6.3f}",
                f"{pkt.latency_ms:4.1f} ms",
                pkt.raw_meas_str,
                status_combined,
            ]

            for c_idx, text in enumerate(row_vals):
                item = QTableWidgetItem(text)
                item.setFont(font_bold if (r_idx == 0 or c_idx in (0, 1)) else font_mono)

                if c_idx in (0, 1, 2, 3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

                # Tactical row coloring based on sensor type and OOSM status
                if pkt.is_oosm:
                    item.setForeground(QColor("#F59E0B") if c_idx < 6 else QColor("#FF9F1C"))
                    if r_idx == 0:
                        item.setBackground(QColor(55, 32, 8))
                elif pkt.status == "NAIVE-DELAY" and c_idx == 6:
                    item.setForeground(QColor("#EF4444"))
                elif pkt.sensor_type == "RADAR":
                    item.setForeground(QColor("#10B981") if c_idx == 1 else QColor("#B8D4C8"))
                else:
                    item.setForeground(QColor("#38BDF8") if c_idx == 1 else QColor("#B8C8D8"))

                self.table.setItem(r_idx, c_idx, item)
