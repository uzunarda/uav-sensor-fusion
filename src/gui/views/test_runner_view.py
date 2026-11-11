"""
Tab 3: System Diagnostics & Unit Test Runner Dashboard.
Allows executing all test suites with a single click, showing pass/fail status badges,
execution times, and full unittest output in a diagnostic console layout.
"""

from __future__ import annotations
import io
import unittest
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QFrame,
    QProgressBar,
)


class TestRunnerWorker(QThread):
    """Executes unittest suite in background thread."""
    test_suite_finished = pyqtSignal(bool, str, int, int)  # success, output, passed, total

    def run(self) -> None:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName("tests.test_ekf")

        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)

        output = stream.getvalue()
        passed = result.testsRun - len(result.failures) - len(result.errors)
        total = result.testsRun
        success = result.wasSuccessful()

        self.test_suite_finished.emit(success, output, passed, total)


class TestRunnerView(QWidget):
    """
    Unit Test Runner & System Diagnostics View.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker: Optional[TestRunnerWorker] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ---------------- Top Control Bar ----------------
        top_bar = QFrame()
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #182634;
                border-left: 3px solid #F59E0B;
                padding: 4px 8px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 4, 6, 4)
        top_layout.setSpacing(12)

        lbl_mod = QLabel("[MOD-03] SİSTEM TEŞHİS VE DOĞRULAMA:")
        lbl_mod.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; border: none;")
        top_layout.addWidget(lbl_mod)

        self.btn_run_tests = QPushButton("[ > ] TÜM DOĞRULAMA TESTLERİNİ KOŞTUR (RUN SUITE)")
        self.btn_run_tests.setStyleSheet("""
            QPushButton {
                background-color: #0C2D42;
                color: #38BDF8;
                border: 1px solid #00B4D8;
                border-left: 3px solid #00B4D8;
                padding: 5px 14px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #123E59;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #111D26;
                color: #4B6A7E;
                border: 1px solid #1E3242;
            }
        """)
        self.btn_run_tests.clicked.connect(self._start_tests)
        top_layout.addWidget(self.btn_run_tests)

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
                background-color: #38BDF8;
            }
        """)
        top_layout.addWidget(self.progress_bar)

        self.lbl_summary = QLabel("DURUM: [HAZIR] // Matematiksel Doğrulama Paketi Bekleniyor")
        self.lbl_summary.setStyleSheet("color: #38BDF8; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        top_layout.addWidget(self.lbl_summary)

        top_layout.addStretch()
        layout.addWidget(top_bar)

        # ---------------- Main Splitter: Left Test Cases vs Right Output Console ----------------
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(4)

        # Left: Test Items List
        left_frame = QFrame()
        left_frame.setStyleSheet("background-color: #070B10; border: 1px solid #16222E;")
        lbox = QVBoxLayout(left_frame)
        lbox.setContentsMargins(8, 8, 8, 8)
        lbox.setSpacing(6)

        lbl_list_title = QLabel("[TEŞHİS-01] BİRİM VE ENTEGRASYON DOĞRULAMA PAKETLERİ")
        lbl_list_title.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; letter-spacing: 0.6px; border: none;")
        lbox.addWidget(lbl_list_title)

        self.test_list = QListWidget()
        self.test_list.setStyleSheet("""
            QListWidget {
                background-color: #06090D;
                color: #C4D6E5;
                border: 1px solid #16222E;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #101A24;
            }
            QListWidget::item:hover {
                background-color: #0E1822;
            }
        """)
        lbox.addWidget(self.test_list)
        main_splitter.addWidget(left_frame)

        # Right: Terminal Output Console
        right_frame = QFrame()
        right_frame.setStyleSheet("background-color: #070B10; border: 1px solid #16222E;")
        rbox = QVBoxLayout(right_frame)
        rbox.setContentsMargins(8, 8, 8, 8)
        rbox.setSpacing(6)

        lbl_console_title = QLabel("[KONSOL-01] TEST ÇIKTISI VE ANALİTİK DOĞRULAMA GÜNLÜĞÜ")
        lbl_console_title.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; letter-spacing: 0.6px; border: none;")
        rbox.addWidget(lbl_console_title)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #05080C;
                color: #9FB6C8;
                border: 1px solid #16222E;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                padding: 6px;
            }
        """)
        rbox.addWidget(self.console_output)
        main_splitter.addWidget(right_frame)

        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 5)

        layout.addWidget(main_splitter, stretch=1)

        self._populate_static_test_list()

    def _populate_static_test_list(self) -> None:
        """Fills list with the standard 9 test cases."""
        test_definitions = [
            ("test_cv_dynamics", "Sabit Hızlı (Constant Velocity) 3B Kinematik Doğrulaması", "HAZIR"),
            ("test_radar_jacobian_numerical", "Analitik Radar Jacobi Matrisi vs Sayısal Sonlu Farklar", "HAZIR"),
            ("test_camera_jacobian_numerical", "Analitik Kamera Jacobi Matrisi vs Sayısal Sonlu Farklar", "HAZIR"),
            ("test_sensor_latency_and_timestamps", "Radar/Kamera Gecikmesi, Gauss Jitter ve Saat Kayması (+25ms)", "HAZIR"),
            ("test_priority_queue_arrival_ordering", "İşlemci Varış Sıralaması (t_receive) & OOSM Oluşumu", "HAZIR"),
            ("test_ekf_oosm_rollback_replay", "EKF Tamponu Geri Sarma (Rollback) & Sıralı Yeniden Oynatma", "HAZIR"),
            ("test_ekf_stale_oosm_dropped", "Tampon Sınırından Eski Ölçümlerin Güvenli Atılması (Stale Drop)", "HAZIR"),
            ("test_simulation_worker_precomputation", "Çoklu Mod (Mod A, B, C) Simülasyon Motoru Ön Hesaplaması", "HAZIR"),
            ("test_gui_window_initialization_and_frame_dispatch", "Taktik GUI Başlatma, 60 FPS Frame Yayını ve Zaman Çubuğu", "HAZIR"),
        ]

        self.test_list.clear()
        for test_id, desc, status in test_definitions:
            item_text = f"[{status}]  {test_id}\n     +-- {desc}"
            item = QListWidgetItem(item_text)
            item.setForeground(QColor("#7FA0B8"))
            self.test_list.addItem(item)

    def _start_tests(self) -> None:
        self.btn_run_tests.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_summary.setText("DURUM: [ÇALIŞIYOR] // Birim testleri koşturuluyor...")
        self.lbl_summary.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        self.console_output.setText(">> python -m unittest discover tests başlatıldı...\n")

        self.worker = TestRunnerWorker()
        self.worker.test_suite_finished.connect(self._on_tests_finished)
        self.worker.start()

    def _on_tests_finished(self, success: bool, output: str, passed: int, total: int) -> None:
        self.btn_run_tests.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.console_output.setText(output)

        if success:
            self.lbl_summary.setText(f"DURUM: [BAŞARILI] // Tüm doğrulama testleri geçti ({passed}/{total} PASS).")
            self.lbl_summary.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        else:
            self.lbl_summary.setText(f"DURUM: [HATA] // Bazı testlerde uyuşmazlık tespit edildi ({passed}/{total} PASS).")
            self.lbl_summary.setStyleSheet("color: #EF4444; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")

        for i in range(self.test_list.count()):
            item = self.test_list.item(i)
            cur_text = item.text()
            if success:
                new_text = cur_text.replace("[HAZIR]", "[GEÇTİ]").replace("[UYARI]", "[GEÇTİ]")
                item.setText(new_text)
                item.setForeground(QColor("#10B981"))
            else:
                new_text = cur_text.replace("[HAZIR]", "[TEST]").replace("[GEÇTİ]", "[TEST]")
                item.setText(new_text)
