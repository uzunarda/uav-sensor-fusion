"""
Tactical Radar & Multi-Sensor Fusion Command Center Application.
Main Window integrating 4 Primary Tactical Modules:
  1. Live Tactical Radar & 3D Spatial Trajectory (60 FPS PPI Scope & GLView)
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
from PyQt6.QtGui import QFont, QIcon, QColor
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
pg.setConfigOption("background", "#050A08")
pg.setConfigOption("foreground", "#5C8A6F")
pg.setConfigOption("antialias", True)

from src.gui.worker import SimulationWorker
from src.gui.views.live_radar_view import LiveRadarView
from src.gui.views.benchmark_view import BenchmarkView
from src.gui.views.test_runner_view import TestRunnerView
from src.gui.views.config_view import ConfigView


class TacticalRadarApp(QMainWindow):
    """
    All-in-One Tactical Command Center Desktop Application.
    """

    def __init__(self, sim_duration: float = 20.0, seed: int = 42) -> None:
        super().__init__()
        self.setWindowTitle("İHA HAVA SAHASI FARKINDALIK VE ÇOK SENSÖRLÜ EKF FÜZYON KOMUTA MERKEZİ")
        self.resize(1560, 940)
        self.setMinimumSize(1180, 740)

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
        """Constructs sidebar navigation and stacked workspace."""
        central = QWidget(self)
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # =====================================================================
        # LEFT: Tactical Sidebar Navigation Drawer (~230px)
        # =====================================================================
        sidebar = QFrame()
        sidebar.setObjectName("SidebarFrame")
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("""
            #SidebarFrame {
                background-color: #080D11;
                border: 1px solid #16242D;
                border-radius: 6px;
            }
        """)
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(8, 12, 8, 12)
        s_layout.setSpacing(10)

        # Tactical Logo Header
        lbl_logo = QLabel("AIR DEFENSE\nTACTICAL CONSOLE")
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_logo.setStyleSheet("""
            color: #00FF66;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 2px;
            padding-bottom: 6px;
            border-bottom: 1px solid #162B38;
        """)
        s_layout.addWidget(lbl_logo)

        lbl_sub = QLabel("6-DOF ASYNC EKF FUSION")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("color: #4C7082; font-family: 'Consolas', monospace; font-size: 9px; letter-spacing: 1px;")
        s_layout.addWidget(lbl_sub)

        s_layout.addSpacing(10)

        # Navigation Buttons Group
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        nav_items = [
            ("🎯  1. CANLI RADAR (PPI)", 0, "#00FF66"),
            ("📊  2. BENCHMARK RAPORU", 1, "#00CCFF"),
            ("🧪  3. SİSTEM TEŞHİS & TEST", 2, "#FFB300"),
            ("⚙️  4. SENSÖR AYARLARI", 3, "#E056FD"),
        ]

        for text, page_idx, accent_color in nav_items:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(44)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #0D161C;
                    color: #A0C0B0;
                    border: 1px solid #16242C;
                    border-radius: 5px;
                    text-align: left;
                    padding-left: 14px;
                    font-family: 'Consolas', monospace;
                    font-weight: bold;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: #12212B;
                    color: #FFFFFF;
                    border: 1px solid {accent_color};
                }}
                QPushButton:checked {{
                    background-color: #142B28;
                    color: {accent_color};
                    border: 1px solid {accent_color};
                    border-left: 4px solid {accent_color};
                }}
            """)
            self.nav_group.addButton(btn, page_idx)
            s_layout.addWidget(btn)

        s_layout.addStretch()

        # System Status Footer Box
        status_box = QFrame()
        status_box.setStyleSheet("background-color: #0A1216; border: 1px solid #16252E; border-radius: 4px; padding: 6px;")
        sbox_layout = QVBoxLayout(status_box)
        sbox_layout.setContentsMargins(6, 6, 6, 6)
        sbox_layout.setSpacing(4)

        lbl_stat_title = QLabel("SİSTEM DURUMU")
        lbl_stat_title.setStyleSheet("color: #5C8A6F; font-size: 8px; font-family: 'Consolas', monospace; font-weight: bold;")
        sbox_layout.addWidget(lbl_stat_title)

        self.lbl_active_mode = QLabel("FÜZYON: MOD C (ROLLBACK)")
        self.lbl_active_mode.setStyleSheet("color: #00FF66; font-size: 10px; font-family: 'Consolas', monospace; font-weight: bold;")
        sbox_layout.addWidget(self.lbl_active_mode)

        lbl_fps = QLabel("GRAFİK MOTORU: 60 FPS GPU")
        lbl_fps.setStyleSheet("color: #00CCFF; font-size: 9px; font-family: 'Consolas', monospace;")
        sbox_layout.addWidget(lbl_fps)

        s_layout.addWidget(status_box)
        root_layout.addWidget(sidebar)

        # =====================================================================
        # RIGHT: Stacked Workspace Area
        # =====================================================================
        self.stacked_widget = QStackedWidget()

        # Page 0: Live Tactical Radar View
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

        # Connect Navigation Click
        self.nav_group.idClicked.connect(self._switch_page)

        # Worker mode change hook
        self.worker.mode_changed.connect(self._on_worker_mode_changed)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #060A0D;
                color: #5C8A6F;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                border-top: 1px solid #16242C;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("KOMUTA MERKEZİ AKTİF: 4 Taktik Modül Hazır. [Mod C - OOSM Destekli]")

    def _switch_page(self, page_index: int) -> None:
        """Switches active workspace view."""
        self.stacked_widget.setCurrentIndex(page_index)
        page_titles = [
            "1. CANLI TAKTİK RADAR (PPI SCOPE & 3B GÖRÜNÜM)",
            "2. AŞAMA 2 ANALİTİK BENCHMARK VE PERFORMANS RAPORU",
            "3. SİSTEM TEŞHİS VE BİRİM TESTLERİ KOŞTURUCU",
            "4. CANLI SENSÖR GECİKMELERİ VE EKF KONFİGÜRASYONU",
        ]
        if 0 <= page_index < len(page_titles):
            self.status_bar.showMessage(f"SEKME DEĞİŞTİRİLDİ -> {page_titles[page_index]}")

    def _on_parameters_applied(self) -> None:
        """When user applies custom parameters, switch back to Tab 0 and play."""
        self.nav_group.buttons()[0].setChecked(True)
        self.stacked_widget.setCurrentIndex(0)
        self.status_bar.showMessage("YENİ PARAMETRELERLE CANLI SİMÜLASYON BAŞLATILDI!")
        self.worker.play()

    def _on_worker_mode_changed(self, mode: str) -> None:
        mode_text_map = {
            "MODE_A": "FÜZYON: MOD A (İDEAL)",
            "MODE_B": "FÜZYON: MOD B (NAIVE)",
            "MODE_C": "FÜZYON: MOD C (ROLLBACK)",
        }
        self.lbl_active_mode.setText(mode_text_map.get(mode, mode))

    def _apply_dark_theme(self) -> None:
        """Applies comprehensive dark military QSS theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #05080A;
            }
            #CentralWidget {
                background-color: #05080A;
            }
            QSplitter::handle {
                background-color: #121F28;
            }
            QSplitter::handle:hover {
                background-color: #00FF66;
            }
            QLabel {
                color: #A0C0B0;
            }
        """)


def main() -> None:
    """Entry point for the All-in-One Tactical Command Center."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = TacticalRadarApp(sim_duration=20.0, seed=42)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
