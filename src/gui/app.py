"""
Tactical Radar & Multi-Sensor Fusion Command Center Application.
Main Window integrating 4 Primary Tactical Modules:
  1. Live Tactical Radar, 3D Spatial Trajectory, Synthetic Sensor Data Stream & HUD Telemetry
  2. Phase 2 Analytical Benchmark & Comparative Metrics Report
  3. System Diagnostics & Unit Test Suite Runner
  4. Live Sensor Latency, Clock Bias & EKF Parameter Configurator
"""

from __future__ import annotations
import os
import sys

# Ensure UTF-8 on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QPushButton,
    QButtonGroup,
    QLabel,
    QFrame,
    QStatusBar,
)

import pyqtgraph as pg

# Configure global dark tactical theme for pyqtgraph
pg.setConfigOption("background", "#05080C")
pg.setConfigOption("foreground", "#68899E")
pg.setConfigOption("antialias", True)

from src.gui.worker import SimulationWorker, SimulationFrame
from src.gui.views.live_radar_view import LiveRadarView
from src.gui.views.benchmark_view import BenchmarkView
from src.gui.views.test_runner_view import TestRunnerView
from src.gui.views.config_view import ConfigView


class TacticalRadarApp(QMainWindow):
    """
    Tactical Command Center Desktop Application.
    """

    def __init__(self, sim_duration: float = 20.0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("İHA HAVA SAHASI FARKINDALIK VE ÇOK SENSÖRLÜ EKF FÜZYON KOMUTA MERKEZİ")
        self.resize(1600, 960)
        self.setMinimumSize(1220, 760)

        # Apply Global Dark Military QSS Theme
        self._apply_dark_theme()

        # Shared Simulation Worker Engine
        self.worker = SimulationWorker(sim_duration=sim_duration, seed=seed, parent=self)

        # Build Main View Hierarchy
        self._setup_ui()

        # Start on page 0 (Live Radar)
        self.nav_group.buttons()[0].setChecked(True)
        self.stacked_widget.setCurrentIndex(0)

        # Initial frame dispatch
        self.worker.seek_time(0.0)

    def _setup_ui(self) -> None:
        """Constructs top system telemetry strip, sidebar navigation, and stacked MFD workspace."""
        central = QWidget(self)
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        # =====================================================================
        # TOP: Full-Width Tactical Command & Telemetry Header Strip
        # =====================================================================
        top_strip = QFrame()
        top_strip.setObjectName("TopHeaderStrip")
        top_strip.setFixedHeight(34)
        top_strip.setStyleSheet("""
            #TopHeaderStrip {
                background-color: #080C12;
                border: 1px solid #182634;
                border-left: 4px solid #10B981;
            }
        """)
        ts_layout = QHBoxLayout(top_strip)
        ts_layout.setContentsMargins(10, 0, 10, 0)
        ts_layout.setSpacing(14)

        lbl_sys_id = QLabel("İHA HAVA SAHASI FARKINDALIK VE ÇOK SENSÖRLÜ HEDEF İZLEME KONSOLU")
        lbl_sys_id.setStyleSheet("""
            color: #E2EEF8;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            border: none;
        """)
        ts_layout.addWidget(lbl_sys_id)
        ts_layout.addStretch(1)

        # Live Top Strip Telemetry Counters
        self.lbl_top_epoch = self._create_top_badge(ts_layout, "EPOCH: T+00.00s", "#10B981")
        self.lbl_top_mode = self._create_top_badge(ts_layout, "MOD: MOD C [OOSM ROLLBACK]", "#38BDF8")
        self.lbl_top_sensors = self._create_top_badge(ts_layout, "SENSÖR HATTI: RADAR 10Hz | EO/IR 30Hz", "#9FB6C8")
        self.lbl_top_pkt = self._create_top_badge(ts_layout, "PKT: #0000 | OOSM: 000", "#F59E0B")

        outer_layout.addWidget(top_strip)

        # =====================================================================
        # BODY: Left Command Sidebar + Right MFD Workspace
        # =====================================================================
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(4)

        sidebar = QFrame()
        sidebar.setObjectName("SidebarFrame")
        sidebar.setFixedWidth(232)
        sidebar.setStyleSheet("""
            #SidebarFrame {
                background-color: #070B10;
                border: 1px solid #16222E;
            }
        """)
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(8, 10, 8, 10)
        s_layout.setSpacing(8)

        # Tactical Console Identifier Block
        id_box = QFrame()
        id_box.setStyleSheet("""
            QFrame {
                background-color: #0B1119;
                border: 1px solid #1A2A3A;
                border-top: 2px solid #10B981;
            }
        """)
        id_layout = QVBoxLayout(id_box)
        id_layout.setContentsMargins(8, 8, 8, 8)
        id_layout.setSpacing(2)

        lbl_logo = QLabel("TAKTİK RADAR &\nFÜZYON KONSOLU")
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        lbl_logo.setStyleSheet("""
            color: #10B981;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 1.5px;
            border: none;
        """)
        id_layout.addWidget(lbl_logo)

        lbl_sub = QLabel("6-DOF ASENKRON EKF FÜZYON\nSİSTEM KODU: UAV-TRK-2026")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignLeft)
        lbl_sub.setStyleSheet("""
            color: #68899E;
            font-family: 'Consolas', monospace;
            font-size: 9px;
            letter-spacing: 0.5px;
            border: none;
        """)
        id_layout.addWidget(lbl_sub)
        s_layout.addWidget(id_box)

        lbl_nav_hdr = QLabel("OPERASYONEL MODÜLLER")
        lbl_nav_hdr.setStyleSheet("color: #4D6B82; font-family: 'Consolas', monospace; font-size: 9px; font-weight: bold; letter-spacing: 1px; padding-top: 4px;")
        s_layout.addWidget(lbl_nav_hdr)

        # Navigation Buttons Group
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        nav_items = [
            ("[01] CANLI RADAR & VERİ AKIŞI", 0, "#10B981"),
            ("[02] ANALİTİK BENCHMARK", 1, "#38BDF8"),
            ("[03] SİSTEM TEŞHİS & TEST", 2, "#F59E0B"),
            ("[04] SENSÖR & FİLTRE AYARLARI", 3, "#00B4D8"),
        ]

        for text, page_idx, accent_color in nav_items:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #0B1017;
                    color: #9FB6C8;
                    border: 1px solid #172533;
                    border-left: 2px solid #24384D;
                    text-align: left;
                    padding-left: 10px;
                    font-family: 'Consolas', monospace;
                    font-weight: bold;
                    font-size: 10px;
                    letter-spacing: 0.4px;
                }}
                QPushButton:hover {{
                    background-color: #101A26;
                    color: #FFFFFF;
                    border: 1px solid {accent_color};
                    border-left: 3px solid {accent_color};
                }}
                QPushButton:checked {{
                    background-color: #11222E;
                    color: {accent_color};
                    border: 1px solid {accent_color};
                    border-left: 4px solid {accent_color};
                }}
            """)
            self.nav_group.addButton(btn, page_idx)
            s_layout.addWidget(btn)

        s_layout.addStretch()

        # Architecture Specification Matrix Box
        spec_box = QFrame()
        spec_box.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #162432;
                border-left: 2px solid #38BDF8;
            }
        """)
        spec_layout = QVBoxLayout(spec_box)
        spec_layout.setContentsMargins(8, 6, 8, 6)
        spec_layout.setSpacing(3)

        lbl_spec_title = QLabel("MİMARİ PARAMETRE ÖZETİ")
        lbl_spec_title.setStyleSheet("color: #38BDF8; font-size: 9px; font-family: 'Consolas', monospace; font-weight: bold; border: none;")
        spec_layout.addWidget(lbl_spec_title)

        for spec_line in [
            "ÇERÇEVE  : 3B NED [K-D-A]",
            "DURUM    : R^6 [x,vx,y,vy,z,vz]",
            "DİNAMİK  : CWNA BLOK-DİYAGONAL",
            "GÜNCELLE : JOSEPH FORMU P_k",
            "OOSM     : ROLLBACK / REPLAY",
        ]:
            lbl_s = QLabel(spec_line)
            lbl_s.setStyleSheet("color: #7FA0B8; font-size: 9px; font-family: 'Consolas', monospace; border: none;")
            spec_layout.addWidget(lbl_s)

        s_layout.addWidget(spec_box)

        # System Status Footer Box
        status_box = QFrame()
        status_box.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #162432;
                border-left: 2px solid #10B981;
            }
        """)
        sbox_layout = QVBoxLayout(status_box)
        sbox_layout.setContentsMargins(8, 6, 8, 6)
        sbox_layout.setSpacing(3)

        lbl_stat_title = QLabel("AKTİF ÇALIŞMA DURUMU")
        lbl_stat_title.setStyleSheet("color: #68899E; font-size: 9px; font-family: 'Consolas', monospace; font-weight: bold; border: none;")
        sbox_layout.addWidget(lbl_stat_title)

        self.lbl_active_mode = QLabel("FÜZYON: MOD C [ROLLBACK]")
        self.lbl_active_mode.setStyleSheet("color: #10B981; font-size: 10px; font-family: 'Consolas', monospace; font-weight: bold; border: none;")
        sbox_layout.addWidget(self.lbl_active_mode)

        lbl_fps = QLabel("ÖRNEKLEME: 60 FPS SENKRON")
        lbl_fps.setStyleSheet("color: #38BDF8; font-size: 9px; font-family: 'Consolas', monospace; border: none;")
        sbox_layout.addWidget(lbl_fps)

        s_layout.addWidget(status_box)
        root_layout.addWidget(sidebar)

        # =====================================================================
        # RIGHT: Stacked Workspace Area
        # =====================================================================
        self.stacked_widget = QStackedWidget()

        # Page 0: Live Tactical Radar View (2x2 MFD with Synthetic Data Stream)
        self.view_live_radar = LiveRadarView(self.worker, self)
        self.stacked_widget.addWidget(self.view_live_radar)

        # Page 1: Phase 2 Analytical Benchmark View
        self.view_benchmark = BenchmarkView(self)
        self.stacked_widget.addWidget(self.view_benchmark)

        # Page 2: System Diagnostics & Test Runner View
        self.view_tests = TestRunnerView(self)
        self.stacked_widget.addWidget(self.view_tests)

        # Page 3: Live Sensor Latency & EKF Configurator View
        self.view_config = ConfigView(self.worker, self)
        self.view_config.applied_parameters.connect(self._on_parameters_applied)
        self.stacked_widget.addWidget(self.view_config)

        root_layout.addWidget(self.stacked_widget, stretch=1)
        outer_layout.addLayout(root_layout, stretch=1)

        # Connect Navigation Click
        self.nav_group.idClicked.connect(self._switch_page)

        # Worker hooks
        self.worker.mode_changed.connect(self._on_worker_mode_changed)
        self.worker.frame_ready.connect(self._on_worker_frame_ready)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #06090D;
                color: #7FA0B8;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                border-top: 1px solid #16222E;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("SİSTEM DURUMU: [AKTİF] // 4 Operasyonel Modül ve Canlı Sentetik Veri Akışı Hazır. [Mod C - OOSM Rollback/Replay]")

    def _create_top_badge(self, layout: QHBoxLayout, text: str, color: str) -> QLabel:
        """Creates a boxed technical telemetry indicator in the top header strip."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            QLabel {{
                background-color: #0C131C;
                color: {color};
                border: 1px solid #1B2C3E;
                padding: 3px 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(lbl)
        return lbl

    def _switch_page(self, page_index: int) -> None:
        """Switches active workspace view."""
        self.stacked_widget.setCurrentIndex(page_index)
        page_titles = [
            "[01] CANLI TAKTİK RADAR, 3B YÖRÜNGE VE SENTETİK SENSÖR VERİ AKIŞI",
            "[02] AŞAMA 2 ANALİTİK BENCHMARK VE PERFORMANS RAPORU",
            "[03] SİSTEM TEŞHİS VE BİRİM TESTLERİ DOĞRULAMA KONSOLU",
            "[04] SENTETİK SENSÖR GECİKMELERİ VE EKF KONFİGÜRASYONU",
        ]
        if 0 <= page_index < len(page_titles):
            self.status_bar.showMessage(f"AKTİF MODÜL -> {page_titles[page_index]}")

    def _on_parameters_applied(self) -> None:
        """When user applies custom parameters, switch back to Tab 0 and play."""
        self.nav_group.buttons()[0].setChecked(True)
        self.stacked_widget.setCurrentIndex(0)
        self.lbl_top_sensors.setText(
            f"SENSÖR HATTI: RADAR {self.worker.radar_freq_hz:.0f}Hz | EO/IR {self.worker.cam_freq_hz:.0f}Hz"
        )
        self.status_bar.showMessage("DURUM: [GÜNCELLENDİ] // Yeni sensör ve EKF parametreleriyle canlı sentetik veri akışı başlatıldı.")
        self.worker.play()

    def _on_worker_mode_changed(self, mode: str) -> None:
        mode_text_map = {
            "MODE_A": "FÜZYON: MOD A [İDEAL]",
            "MODE_B": "FÜZYON: MOD B [NAIVE]",
            "MODE_C": "FÜZYON: MOD C [ROLLBACK]",
        }
        top_map = {
            "MODE_A": "MOD: MOD A [İDEAL / 0ms]",
            "MODE_B": "MOD: MOD B [NAIVE EKF]",
            "MODE_C": "MOD: MOD C [OOSM ROLLBACK]",
        }
        self.lbl_active_mode.setText(mode_text_map.get(mode, mode))
        self.lbl_top_mode.setText(top_map.get(mode, mode))

    def _on_worker_frame_ready(
        self,
        frame: SimulationFrame,
        gt_history: object,
        ekf_history: object,
        raw_tail: object,
    ) -> None:
        """Updates top command bar telemetry indicators."""
        self.lbl_top_epoch.setText(f"EPOCH: T+{frame.timestamp:05.2f}s")
        self.lbl_top_pkt.setText(f"PKT: #{frame.packet_seq_id:04d} | OOSM: {frame.oosm_total_count:03d}")

    def _apply_dark_theme(self) -> None:
        """Applies comprehensive dark slate QSS theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #04070A;
            }
            #CentralWidget {
                background-color: #04070A;
            }
            QSplitter::handle {
                background-color: #142230;
            }
            QSplitter::handle:hover {
                background-color: #00B4D8;
            }
            QLabel {
                color: #9FB6C8;
            }
        """)


def main() -> None:
    """Entry point for the All-in-One Tactical Command Center."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Consolas", 9)
    app.setFont(font)

    window = TacticalRadarApp(sim_duration=20.0, seed=42)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
