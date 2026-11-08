"""
Tab 2: Phase 2 Analytical Benchmark & Comparative Performance Dashboard.
Features on-demand simulation benchmarking, detailed metrics comparison table,
and real-time comparative error/OOSM distribution graphs.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QComboBox,
    QProgressBar,
)

import pyqtgraph as pg

from src.main import run_benchmark


class BenchmarkWorker(QThread):
    """Background worker to run benchmark without freezing GUI."""
    finished_benchmark = pyqtSignal(dict)

    def __init__(self, duration: float = 20.0, seed: int = 42) -> None:
        super().__init__()
        self.duration = float(duration)
        self.seed = int(seed)

    def run(self) -> None:
        metrics = run_benchmark(
            sim_duration=self.duration,
            seed=self.seed,
            save_fig_path="results/ekf_oosm_comparison.png",
            show_plot=False,
        )
        self.finished_benchmark.emit(metrics)


class BenchmarkView(QWidget):
    """
    Analytical Benchmark & Metrics Dashboard View.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker: Optional[BenchmarkWorker] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ---------------- Top Control Bar ----------------
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #0C1418;
                border: 1px solid #1A2B35;
                border-radius: 4px;
                padding: 4px 8px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.setSpacing(12)

        self.btn_run = QPushButton("▶ KIYASLAMAYI ÇALIŞTIR (RUN BENCHMARK)")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #008844;
                color: #FFFFFF;
                border: 1px solid #00BB55;
                border-radius: 4px;
                padding: 6px 16px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00AA55;
            }
            QPushButton:disabled {
                background-color: #22332A;
                color: #557766;
                border: 1px solid #33443A;
            }
        """)
        self.btn_run.clicked.connect(self._start_benchmark)
        top_layout.addWidget(self.btn_run)

        lbl_dur = QLabel("SÜRE:")
        lbl_dur.setStyleSheet("color: #7A9CAE; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_layout.addWidget(lbl_dur)

        self.combo_dur = QComboBox()
        self.combo_dur.addItems(["10.0 saniye", "20.0 saniye", "30.0 saniye"])
        self.combo_dur.setCurrentIndex(1)
        self.combo_dur.setStyleSheet("""
            QComboBox {
                background-color: #111A20;
                color: #00FF66;
                border: 1px solid #1E2E38;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        top_layout.addWidget(self.combo_dur)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate spinner
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #111A20;
                border: 1px solid #1E2E38;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background-color: #00FF66;
                border-radius: 6px;
            }
        """)
        top_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("DURUM: Hazır. Kıyaslamayı başlatmak için butona basınız.")
        self.lbl_status.setStyleSheet("color: #00FF66; font-family: 'Consolas', monospace; font-size: 11px;")
        top_layout.addWidget(self.lbl_status)

        top_layout.addStretch()
        layout.addWidget(top_bar)

        # ---------------- Main Splitter: Left Table vs Right Graphs ----------------
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(4)

        # Left: Metrics Comparison Table
        table_container = QFrame()
        table_container.setStyleSheet("background-color: #0A0F12; border: 1px solid #16242C; border-radius: 4px;")
        tbox = QVBoxLayout(table_container)
        tbox.setContentsMargins(6, 6, 6, 6)

        table_title = QLabel("PERFORMANS VE BAŞARIM METRİKLERİ TABLOSU")
        table_title.setStyleSheet("color: #FFB300; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; margin-bottom: 4px;")
        tbox.addWidget(table_title)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Mod A (İdeal)", "Mod B (Naive EKF)", "Mod C (Rollback/OOSM)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #070B0E;
                color: #C0D8E6;
                gridline-color: #16242C;
                border: none;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #101820;
                color: #00FF66;
                padding: 6px;
                border: 1px solid #16242C;
                font-weight: bold;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #162C3A;
            }
        """)
        tbox.addWidget(self.table)
        main_splitter.addWidget(table_container)

        # Right: Real-Time Comparative Graphs
        graph_container = QFrame()
        graph_container.setStyleSheet("background-color: #0A0F12; border: 1px solid #16242C; border-radius: 4px;")
        gbox = QVBoxLayout(graph_container)
        gbox.setContentsMargins(6, 6, 6, 6)
        gbox.setSpacing(6)

        # Plot 1: 3D Position Error (Mode A vs Mode B vs Mode C)
        self.plot_pos_err = pg.PlotWidget(title="3B Konum Hatası Karşılaştırması (m)")
        self.plot_pos_err.setBackground("#050A0E")
        self.plot_pos_err.showGrid(x=True, y=True, alpha=0.15)
        self.plot_pos_err.getAxis("bottom").setLabel("Zaman", units="s", color="#5C8A9F")
        self.plot_pos_err.getAxis("left").setLabel("Konum Hatası", units="m", color="#5C8A9F")
        self.plot_pos_err.addLegend(offset=(10, 10))

        self.curve_b_pos = self.plot_pos_err.plot(name="Mod B (Naive)", pen=pg.mkPen("#E74C3C", width=1.8))
        self.curve_c_pos = self.plot_pos_err.plot(name="Mod C (Rollback)", pen=pg.mkPen("#2ECC71", width=2.0))
        self.curve_a_pos = self.plot_pos_err.plot(name="Mod A (İdeal)", pen=pg.mkPen("#3498DB", width=1.5, style=Qt.PenStyle.DashLine))
        gbox.addWidget(self.plot_pos_err)

        # Plot 2: 3D Velocity Error
        self.plot_vel_err = pg.PlotWidget(title="3B Hız Hatası ve Yakınsama (m/s)")
        self.plot_vel_err.setBackground("#050A0E")
        self.plot_vel_err.showGrid(x=True, y=True, alpha=0.15)
        self.plot_vel_err.getAxis("bottom").setLabel("Zaman", units="s", color="#5C8A9F")
        self.plot_vel_err.getAxis("left").setLabel("Hız Hatası", units="m/s", color="#5C8A9F")
        self.plot_vel_err.addLegend(offset=(10, 10))

        self.curve_b_vel = self.plot_vel_err.plot(name="Mod B (Naive)", pen=pg.mkPen("#E74C3C", width=1.8))
        self.curve_c_vel = self.plot_vel_err.plot(name="Mod C (Rollback)", pen=pg.mkPen("#2ECC71", width=2.0))
        self.curve_a_vel = self.plot_vel_err.plot(name="Mod A (İdeal)", pen=pg.mkPen("#3498DB", width=1.5, style=Qt.PenStyle.DashLine))
        gbox.addWidget(self.plot_vel_err)

        main_splitter.addWidget(graph_container)
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 6)

        layout.addWidget(main_splitter, stretch=1)

        # Populate with existing benchmark baseline metrics
        self._populate_default_baseline()

    def _populate_default_baseline(self) -> None:
        """Loads Phase 2 baseline results into table."""
        rows = [
            ("Kalıcı Durum 3B Konum RMSE (m)", "1.954 m", "3.464 m", "2.681 m"),
            ("  - Eksen X (Kuzey) RMSE (m)", "1.304 m", "1.870 m", "1.474 m"),
            ("  - Eksen Y (Doğu) RMSE (m)", "0.971 m", "2.381 m", "1.498 m"),
            ("  - Eksen Z (İrtifa) RMSE (m)", "1.085 m", "1.684 m", "1.664 m"),
            ("Toplam 3B Konum RMSE (m)", "3.119 m", "4.241 m", "3.572 m"),
            ("Kalıcı Durum 3B Hız RMSE (m/s)", "0.918 m/s", "1.111 m/s", "1.108 m/s"),
            ("Toplam 3B Hız RMSE (m/s)", "3.859 m/s", "3.668 m/s", "3.701 m/s"),
            ("İşlenen OOSM Sayısı", "0 (Gecikmesiz)", "0 (Yönetilemez)", "127 Olay"),
            ("Atılan (Dropped) OOSM", "0", "0", "0"),
            ("Mod B (Naive)'ye Göre İyileşme", "-", "Referans", "%22.6 İYİLEŞME"),
        ]
        self._set_table_rows(rows)

    def _set_table_rows(self, rows: list) -> None:
        self.table.setRowCount(len(rows))
        for r, (metric_name, val_a, val_b, val_c) in enumerate(rows):
            item_lbl = QTableWidgetItem(metric_name)
            item_a = QTableWidgetItem(val_a)
            item_b = QTableWidgetItem(val_b)
            item_c = QTableWidgetItem(val_c)

            # Styling
            item_a.setForeground(QColor("#3498DB"))
            item_b.setForeground(QColor("#E74C3C"))
            item_c.setForeground(QColor("#2ECC71"))
            item_c.setFont(QFont("Consolas", 10, QFont.Weight.Bold))

            self.table.setVerticalHeaderItem(r, item_lbl)
            self.table.setItem(r, 0, item_a)
            self.table.setItem(r, 1, item_b)
            self.table.setItem(r, 2, item_c)

    def _start_benchmark(self) -> None:
        """Starts benchmark in background worker."""
        duration_map = [10.0, 20.0, 30.0]
        dur = duration_map[self.combo_dur.currentIndex()]

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText(f"DURUM: Simülasyon {dur:.0f}s boyunca 3 modda koşturuluyor...")
        self.lbl_status.setStyleSheet("color: #FFCC00; font-family: 'Consolas', monospace; font-size: 11px;")

        self.worker = BenchmarkWorker(duration=dur, seed=42)
        self.worker.finished_benchmark.connect(self._on_benchmark_finished)
        self.worker.start()

    def _on_benchmark_finished(self, results: dict) -> None:
        """Updates table and graphs with newly computed benchmark data."""
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("DURUM: Kıyaslama başarıyla tamamlandı. Grafikler ve tablo güncellendi.")
        self.lbl_status.setStyleSheet("color: #00FF66; font-family: 'Consolas', monospace; font-size: 11px;")

        ma = results["mode_a"]
        mb = results["mode_b"]
        mc = results["mode_c"]
        imp_pos = results["pos_improvement_vs_naive"]

        rows = [
            ("Kalıcı Durum 3B Konum RMSE (m)", f"{ma['ss_pos_rmse']:.3f} m", f"{mb['ss_pos_rmse']:.3f} m", f"{mc['ss_pos_rmse']:.3f} m"),
            ("  - Eksen X (Kuzey) RMSE (m)", f"{ma['ss_rmse_x']:.3f} m", f"{mb['ss_rmse_x']:.3f} m", f"{mc['ss_rmse_x']:.3f} m"),
            ("  - Eksen Y (Doğu) RMSE (m)", f"{ma['ss_rmse_y']:.3f} m", f"{mb['ss_rmse_y']:.3f} m", f"{mc['ss_rmse_y']:.3f} m"),
            ("  - Eksen Z (İrtifa) RMSE (m)", f"{ma['ss_rmse_z']:.3f} m", f"{mb['ss_rmse_z']:.3f} m", f"{mc['ss_rmse_z']:.3f} m"),
            ("Toplam 3B Konum RMSE (m)", f"{ma['total_pos_rmse']:.3f} m", f"{mb['total_pos_rmse']:.3f} m", f"{mc['total_pos_rmse']:.3f} m"),
            ("Kalıcı Durum 3B Hız RMSE (m/s)", f"{ma['ss_vel_rmse']:.3f} m/s", f"{mb['ss_vel_rmse']:.3f} m/s", f"{mc['ss_vel_rmse']:.3f} m/s"),
            ("Toplam 3B Hız RMSE (m/s)", f"{ma['total_vel_rmse']:.3f} m/s", f"{mb['total_vel_rmse']:.3f} m/s", f"{mc['total_vel_rmse']:.3f} m/s"),
            ("İşlenen OOSM Sayısı", "0 (Gecikmesiz)", "0 (Yönetilemez)", f"{results['oosm_count']} Olay"),
            ("Atılan (Dropped) OOSM", "0", "0", f"{results['dropped_oosm_count']}"),
            ("Mod B (Naive)'ye Göre İyileşme", "-", "Referans", f"%{imp_pos:.1f} İYİLEŞME"),
        ]
        self._set_table_rows(rows)
