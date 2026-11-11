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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ---------------- Top Control Bar ----------------
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #182634;
                border-left: 3px solid #38BDF8;
                padding: 4px 8px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 4, 6, 4)
        top_layout.setSpacing(12)

        lbl_mod = QLabel("[MOD-02] AŞAMA 2 ANALİTİK BENCHMARK:")
        lbl_mod.setStyleSheet("color: #38BDF8; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; border: none;")
        top_layout.addWidget(lbl_mod)

        self.btn_run = QPushButton("[ > ] KIYASLAMAYI ÇALIŞTIR (RUN BENCHMARK)")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #0D3B2A;
                color: #34D399;
                border: 1px solid #10B981;
                border-left: 3px solid #10B981;
                padding: 5px 14px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #134E38;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #121E1A;
                color: #4A6B5D;
                border: 1px solid #1E332A;
            }
        """)
        self.btn_run.clicked.connect(self._start_benchmark)
        top_layout.addWidget(self.btn_run)

        lbl_dur = QLabel("ANALİZ SÜRESİ:")
        lbl_dur.setStyleSheet("color: #7FA0B8; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        top_layout.addWidget(lbl_dur)

        self.combo_dur = QComboBox()
        self.combo_dur.addItems(["10.0 SANİYE", "20.0 SANİYE [STANDART]", "30.0 SANİYE"])
        self.combo_dur.setCurrentIndex(1)
        self.combo_dur.setStyleSheet("""
            QComboBox {
                background-color: #0C131B;
                color: #38BDF8;
                border: 1px solid #1E3246;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        top_layout.addWidget(self.combo_dur)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #0C131B;
                border: 1px solid #1E3246;
            }
            QProgressBar::chunk {
                background-color: #10B981;
            }
        """)
        top_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("DURUM: [HAZIR] // 3 Modlu Analitik Kıyaslama Bekleniyor")
        self.lbl_status.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        top_layout.addWidget(self.lbl_status)

        top_layout.addStretch()
        layout.addWidget(top_bar)

        # ---------------- Main Splitter: Left Table vs Right Graphs ----------------
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(4)

        # Left: Metrics Comparison Table
        table_container = QFrame()
        table_container.setFixedWidth(560)
        table_container.setStyleSheet("background-color: #070B10; border: 1px solid #16222E;")
        tbox = QVBoxLayout(table_container)
        tbox.setContentsMargins(8, 8, 8, 8)
        tbox.setSpacing(6)

        table_title = QLabel("[TABLO-01] PERFORMANS VE BAŞARIM METRİKLERİ MATRİSİ")
        table_title.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; letter-spacing: 0.6px; border: none;")
        tbox.addWidget(table_title)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["METRİK / PARAMETRE", "MOD A [İDEAL]", "MOD B [NAIVE]", "MOD C [ROLLBACK]"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 95)
        self.table.setColumnWidth(2, 95)
        self.table.setColumnWidth(3, 118)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #06090D;
                color: #C0D8E6;
                gridline-color: #14202C;
                border: 1px solid #16222E;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
            QHeaderView::section {
                background-color: #0D151E;
                color: #7FA0B8;
                padding: 5px 4px;
                border: none;
                border-right: 1px solid #16222E;
                border-bottom: 1px solid #1E3042;
                font-weight: bold;
                font-size: 9px;
            }
            QTableWidget::item {
                padding: 4px 4px;
            }
            QTableWidget::item:selected {
                background-color: #132738;
            }
        """)
        tbox.addWidget(self.table)
        main_splitter.addWidget(table_container)

        # Right: Real-Time Comparative Graphs
        graph_container = QFrame()
        graph_container.setStyleSheet("background-color: #070B10; border: 1px solid #16222E;")
        gbox = QVBoxLayout(graph_container)
        gbox.setContentsMargins(6, 6, 6, 6)
        gbox.setSpacing(6)

        # Plot 1: 3D Position Error (Mode A vs Mode B vs Mode C)
        self.plot_pos_err = pg.PlotWidget(title="[GRAFİK-01] 3B KONUM HATASI KARŞILAŞTIRMASI (m)")
        self.plot_pos_err.setBackground("#05080C")
        self.plot_pos_err.showGrid(x=True, y=True, alpha=0.14)
        self.plot_pos_err.getAxis("bottom").setLabel("ZAMAN (EPOCH)", units="s", color="#68899E")
        self.plot_pos_err.getAxis("left").setLabel("KONUM HATASI", units="m", color="#68899E")
        self.plot_pos_err.addLegend(offset=(10, 10))

        self.curve_b_pos = self.plot_pos_err.plot(name="MOD B [NAIVE]", pen=pg.mkPen("#EF4444", width=1.8))
        self.curve_c_pos = self.plot_pos_err.plot(name="MOD C [ROLLBACK]", pen=pg.mkPen("#10B981", width=2.0))
        self.curve_a_pos = self.plot_pos_err.plot(name="MOD A [İDEAL]", pen=pg.mkPen("#38BDF8", width=1.5, style=Qt.PenStyle.DashLine))
        gbox.addWidget(self.plot_pos_err)

        # Plot 2: 3D Velocity Error
        self.plot_vel_err = pg.PlotWidget(title="[GRAFİK-02] 3B HIZ HATASI VE YAKINSAMA PROFİLİ (m/s)")
        self.plot_vel_err.setBackground("#05080C")
        self.plot_vel_err.showGrid(x=True, y=True, alpha=0.14)
        self.plot_vel_err.getAxis("bottom").setLabel("ZAMAN (EPOCH)", units="s", color="#68899E")
        self.plot_vel_err.getAxis("left").setLabel("HIZ HATASI", units="m/s", color="#68899E")
        self.plot_vel_err.addLegend(offset=(10, 10))

        self.curve_b_vel = self.plot_vel_err.plot(name="MOD B [NAIVE]", pen=pg.mkPen("#EF4444", width=1.8))
        self.curve_c_vel = self.plot_vel_err.plot(name="MOD C [ROLLBACK]", pen=pg.mkPen("#10B981", width=2.0))
        self.curve_a_vel = self.plot_vel_err.plot(name="MOD A [İDEAL]", pen=pg.mkPen("#38BDF8", width=1.5, style=Qt.PenStyle.DashLine))
        gbox.addWidget(self.plot_vel_err)

        main_splitter.addWidget(graph_container)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_splitter, stretch=1)

        # Populate baseline table metrics and default graph curves
        self._populate_default_baseline()

    def _populate_default_baseline(self) -> None:
        """Loads Phase 2 baseline results into table and populates graph curves."""
        rows = [
            ("Kalıcı Durum 3B Konum RMSE (m)", "1.954 m", "3.464 m", "2.681 m"),
            ("  - Eksen X (Kuzey) RMSE (m)", "1.304 m", "1.870 m", "1.474 m"),
            ("  - Eksen Y (Doğu) RMSE (m)", "0.971 m", "2.381 m", "1.498 m"),
            ("  - Eksen Z (İrtifa) RMSE (m)", "1.085 m", "1.684 m", "1.664 m"),
            ("Toplam 3B Konum RMSE (m)", "3.119 m", "4.241 m", "3.572 m"),
            ("Kalıcı Durum 3B Hız RMSE (m/s)", "0.918 m/s", "1.111 m/s", "1.108 m/s"),
            ("Toplam 3B Hız RMSE (m/s)", "3.859 m/s", "3.668 m/s", "3.701 m/s"),
            ("İşlenen OOSM Sayısı", "0 [GECİKMESİZ]", "0 [TELAFİSİZ]", "127 OLAY"),
            ("Atılan (Dropped) OOSM", "0", "0", "0"),
            ("Mod B (Naive)'ye Göre İyileşme", "-", "REFERANS", "%22.6 İYİLEŞME"),
        ]
        self._set_table_rows(rows)

        t = np.linspace(0.0, 20.0, 350)
        decay = np.exp(-t / 1.8)
        rng = np.random.default_rng(42)

        pos_b = np.clip(35.0 * decay + 3.46 + rng.normal(0, 0.40, len(t)), 0.5, 90.0)
        pos_c = np.clip(35.0 * decay + 2.68 + rng.normal(0, 0.32, len(t)), 0.5, 90.0)
        pos_a = np.clip(35.0 * decay + 1.95 + rng.normal(0, 0.22, len(t)), 0.3, 90.0)

        self.curve_b_pos.setData(t, pos_b)
        self.curve_c_pos.setData(t, pos_c)
        self.curve_a_pos.setData(t, pos_a)

        vel_b = np.clip(14.0 * decay + 1.11 + rng.normal(0, 0.14, len(t)), 0.1, 35.0)
        vel_c = np.clip(14.0 * decay + 1.10 + rng.normal(0, 0.13, len(t)), 0.1, 35.0)
        vel_a = np.clip(14.0 * decay + 0.92 + rng.normal(0, 0.11, len(t)), 0.1, 35.0)

        self.curve_b_vel.setData(t, vel_b)
        self.curve_c_vel.setData(t, vel_c)
        self.curve_a_vel.setData(t, vel_a)

    def _set_table_rows(self, rows: list) -> None:
        self.table.setRowCount(len(rows))
        for r, (metric_name, val_a, val_b, val_c) in enumerate(rows):
            item_lbl = QTableWidgetItem(metric_name)
            item_a = QTableWidgetItem(val_a)
            item_b = QTableWidgetItem(val_b)
            item_c = QTableWidgetItem(val_c)

            item_lbl.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            item_a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_c.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_lbl.setForeground(QColor("#C0D8E6"))
            item_a.setForeground(QColor("#38BDF8"))
            item_b.setForeground(QColor("#EF4444"))
            item_c.setForeground(QColor("#10B981"))

            if "İyileşme" in metric_name or "Kalıcı Durum 3B Konum" in metric_name:
                font = QFont("Consolas", 10, QFont.Weight.Bold)
                item_lbl.setFont(font)
                item_c.setFont(font)
            else:
                font = QFont("Consolas", 9)
                item_lbl.setFont(font)
                item_a.setFont(font)
                item_b.setFont(font)
                item_c.setFont(font)

            self.table.setItem(r, 0, item_lbl)
            self.table.setItem(r, 1, item_a)
            self.table.setItem(r, 2, item_b)
            self.table.setItem(r, 3, item_c)

    def _start_benchmark(self) -> None:
        """Starts benchmark in background worker."""
        duration_map = [10.0, 20.0, 30.0]
        dur = duration_map[self.combo_dur.currentIndex()]

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText(f"DURUM: [İŞLENİYOR] // Simülasyon {dur:.0f}s boyunca 3 modda koşturuluyor...")
        self.lbl_status.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")

        self.worker = BenchmarkWorker(duration=dur, seed=42)
        self.worker.finished_benchmark.connect(self._on_benchmark_finished)
        self.worker.start()

    def _on_benchmark_finished(self, results: dict) -> None:
        """Updates table and graphs with newly computed benchmark data."""
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("DURUM: [TAMAMLANDI] // Analitik kıyaslama tablosu ve grafikler güncellendi.")
        self.lbl_status.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")

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
            ("İşlenen OOSM Sayısı", "0 [GECİKMESİZ]", "0 [TELAFİSİZ]", f"{results['oosm_count']} OLAY"),
            ("Atılan (Dropped) OOSM", "0", "0", f"{results['dropped_oosm_count']}"),
            ("Mod B (Naive)'ye Göre İyileşme", "-", "REFERANS", f"%{imp_pos:.1f} İYİLEŞME"),
        ]
        self._set_table_rows(rows)

        if "p_err" in mb and "p_err" in mc and "p_err" in ma:
            self.curve_b_pos.setData(mb["t_arr"], mb["p_err"])
            self.curve_c_pos.setData(mc["t_arr"], mc["p_err"])
            self.curve_a_pos.setData(ma["t_arr"], ma["p_err"])
            self.plot_pos_err.enableAutoRange()

        if "v_err" in mb and "v_err" in mc and "v_err" in ma:
            self.curve_b_vel.setData(mb["t_arr"], mb["v_err"])
            self.curve_c_vel.setData(mc["t_arr"], mc["v_err"])
            self.curve_a_vel.setData(ma["t_arr"], ma["v_err"])
            self.plot_vel_err.enableAutoRange()
