"""
Tab 4: Live Sensor Latency, Clock Bias & EKF Parameter Configurator.
Allows real-time customization of Radar/Camera delays, jitter, clock offsets,
and EKF filter tuning parameters with instant re-simulation of synthetic data.
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QFrame,
    QScrollArea,
)

from src.gui.worker import SimulationWorker


class ConfigView(QWidget):
    """
    Sensor & Algorithm Parameters Configurator View.
    """
    applied_parameters = pyqtSignal()

    def __init__(self, worker: SimulationWorker, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker = worker

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(6)

        # Header Title Bar
        title_box = QFrame()
        title_box.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #182634;
                border-left: 3px solid #F59E0B;
                padding: 4px 8px;
            }
        """)
        t_layout = QHBoxLayout(title_box)
        t_layout.setContentsMargins(6, 4, 6, 4)

        header_lbl = QLabel("[MOD-04] SENTETİK SENSÖR GECİKMELERİ, SAAT KAYMASI VE EKF PARAMETRE KONFİGÜRASYONU")
        header_lbl.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; letter-spacing: 0.8px; border: none;")
        t_layout.addWidget(header_lbl)
        t_layout.addStretch()

        root_layout.addWidget(title_box)

        # Scroll Area for forms
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        form_layout = QVBoxLayout(container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(8)

        # --- Card 1: Radar Sensor Settings ---
        card_radar = self._create_card("[SENSÖR-01] RADAR SENSÖRÜ (1B MENZİL / RANGE) PARAMETRELERİ", "#10B981")
        grid_radar = QGridLayout()
        grid_radar.setSpacing(10)

        self.spin_radar_freq = self._add_double_spin(grid_radar, 0, 0, "Ölçüm Frekansı (Hz):", 1.0, 50.0, 10.0, 1.0, " Hz")
        self.spin_radar_sigma = self._add_double_spin(grid_radar, 0, 1, "Menzil Gürültüsü (sigma_r):", 0.5, 30.0, 6.0, 0.5, " m")
        self.spin_radar_latency = self._add_double_spin(grid_radar, 1, 0, "İletim Gecikmesi (Latency):", 0.0, 100.0, 8.0, 1.0, " ms")
        self.spin_radar_jitter = self._add_double_spin(grid_radar, 1, 1, "Gecikme Gauss Jitter (1-sigma):", 0.0, 20.0, 2.0, 0.5, " ms")

        card_radar.layout().addLayout(grid_radar)
        form_layout.addWidget(card_radar)

        # --- Card 2: Camera Sensor Settings ---
        card_cam = self._create_card("[SENSÖR-02] ELEKTRO-OPTİK / KAMERA (2B AZİMUT & ELEVASYON) PARAMETRELERİ", "#38BDF8")
        grid_cam = QGridLayout()
        grid_cam.setSpacing(10)

        self.spin_cam_freq = self._add_double_spin(grid_cam, 0, 0, "Ölçüm Frekansı (Hz):", 5.0, 100.0, 30.0, 5.0, " Hz")
        self.spin_cam_sigma = self._add_double_spin(grid_cam, 0, 1, "Açısal Gürültü (sigma_psi,lambda):", 0.001, 0.05, 0.004, 0.001, " rad")
        self.spin_cam_latency = self._add_double_spin(grid_cam, 1, 0, "Görüntü İşleme Gecikmesi:", 0.0, 250.0, 40.0, 5.0, " ms")
        self.spin_cam_jitter = self._add_double_spin(grid_cam, 1, 1, "Gecikme Gauss Jitter (1-sigma):", 0.0, 50.0, 5.0, 1.0, " ms")
        self.spin_cam_clock_bias = self._add_double_spin(grid_cam, 2, 0, "Donanımsal Saat Kayması (Clock Bias):", -100.0, 200.0, 25.0, 5.0, " ms")

        card_cam.layout().addLayout(grid_cam)
        form_layout.addWidget(card_cam)

        # --- Card 3: EKF Filter & Buffer Settings ---
        card_ekf = self._create_card("[FİLTRE-01] GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF) VE OOSM TAMPON AYARLARI", "#F59E0B")
        grid_ekf = QGridLayout()
        grid_ekf.setSpacing(10)

        self.spin_qc = self._add_double_spin(grid_ekf, 0, 0, "Süreç Gürültüsü Spektral Yoğunluğu (q_c):", 0.05, 10.0, 0.8, 0.1, " m^2/s^3")
        self.spin_buffer_size = self._add_int_spin(grid_ekf, 0, 1, "OOSM Durum Tamponu Kapasitesi (N Adım):", 10, 500, 150, 10, " adım")

        card_ekf.layout().addLayout(grid_ekf)
        form_layout.addWidget(card_ekf)
        form_layout.addStretch(1)

        scroll.setWidget(container)
        root_layout.addWidget(scroll, stretch=1)

        # --- Bottom Action Bar ---
        action_bar = QFrame()
        action_bar.setStyleSheet("background-color: #090E14; border: 1px solid #182634; padding: 6px;")
        act_layout = QHBoxLayout(action_bar)
        act_layout.setContentsMargins(6, 4, 6, 4)
        act_layout.setSpacing(12)

        self.btn_apply = QPushButton("[ UYGULA ] YENİ PARAMETRELERLE SENTETİK VERİ VE SİMÜLASYONU ÜRET")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #0D3B2A;
                color: #34D399;
                border: 1px solid #10B981;
                border-left: 3px solid #10B981;
                padding: 7px 16px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #134E38;
                color: #FFFFFF;
            }
        """)
        self.btn_apply.clicked.connect(self._apply_custom_parameters)
        act_layout.addWidget(self.btn_apply)

        self.btn_reset_defaults = QPushButton("[ VARSAYILAN ] FABRİKA DEĞERLERİNE DÖN")
        self.btn_reset_defaults.setStyleSheet("""
            QPushButton {
                background-color: #101923;
                color: #9FB6C8;
                border: 1px solid #223548;
                padding: 7px 14px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #182635;
                color: #FFFFFF;
                border: 1px solid #38BDF8;
            }
        """)
        self.btn_reset_defaults.clicked.connect(self._reset_to_defaults)
        act_layout.addWidget(self.btn_reset_defaults)

        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        act_layout.addWidget(self.lbl_feedback)

        act_layout.addStretch()
        root_layout.addWidget(action_bar)

    def _create_card(self, title: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #070B10;
                border: 1px solid #16222E;
                border-left: 3px solid {color};
                padding: 6px;
            }}
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 8)
        vbox.setSpacing(8)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {color}; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; letter-spacing: 0.6px; border: none;")
        vbox.addWidget(lbl)
        return card

    def _add_double_spin(
        self, grid: QGridLayout, row: int, col: int, label: str, min_v: float, max_v: float, def_v: float, step: float, suffix: str
    ) -> QDoubleSpinBox:
        vbox = QVBoxLayout()
        vbox.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #7FA0B8; font-family: 'Consolas', monospace; font-size: 9px; font-weight: bold; border: none;")
        vbox.addWidget(lbl)

        spin = QDoubleSpinBox()
        spin.setDecimals(3 if step < 0.01 else 1)
        spin.setRange(min_v, max_v)
        spin.setValue(def_v)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #0B121A;
                color: #10B981;
                border: 1px solid #1C2E40;
                padding: 4px 6px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        vbox.addWidget(spin)

        grid.addLayout(vbox, row, col)
        return spin

    def _add_int_spin(
        self, grid: QGridLayout, row: int, col: int, label: str, min_v: int, max_v: int, def_v: int, step: int, suffix: str
    ) -> QSpinBox:
        vbox = QVBoxLayout()
        vbox.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #7FA0B8; font-family: 'Consolas', monospace; font-size: 9px; font-weight: bold; border: none;")
        vbox.addWidget(lbl)

        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(def_v)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setStyleSheet("""
            QSpinBox {
                background-color: #0B121A;
                color: #10B981;
                border: 1px solid #1C2E40;
                padding: 4px 6px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        vbox.addWidget(spin)

        grid.addLayout(vbox, row, col)
        return spin

    def _reset_to_defaults(self) -> None:
        self.spin_radar_freq.setValue(10.0)
        self.spin_radar_sigma.setValue(6.0)
        self.spin_radar_latency.setValue(8.0)
        self.spin_radar_jitter.setValue(2.0)

        self.spin_cam_freq.setValue(30.0)
        self.spin_cam_sigma.setValue(0.004)
        self.spin_cam_latency.setValue(40.0)
        self.spin_cam_jitter.setValue(5.0)
        self.spin_cam_clock_bias.setValue(25.0)

        self.spin_qc.setValue(0.8)
        self.spin_buffer_size.setValue(150)
        self.lbl_feedback.setText("[BİLGİ] Değerler fabrika varsayılanlarına döndürüldü.")

    def _apply_custom_parameters(self) -> None:
        """Applies all sensor and EKF parameters to worker and triggers re-simulation."""
        self.worker.radar_freq_hz = float(self.spin_radar_freq.value())
        self.worker.radar_sigma_m = float(self.spin_radar_sigma.value())
        self.worker.radar_latency_s = float(self.spin_radar_latency.value()) / 1000.0
        self.worker.radar_jitter_s = float(self.spin_radar_jitter.value()) / 1000.0

        self.worker.cam_freq_hz = float(self.spin_cam_freq.value())
        self.worker.cam_sigma_rad = float(self.spin_cam_sigma.value())
        self.worker.cam_latency_s = float(self.spin_cam_latency.value()) / 1000.0
        self.worker.cam_jitter_s = float(self.spin_cam_jitter.value()) / 1000.0
        self.worker.cam_clock_bias_s = float(self.spin_cam_clock_bias.value()) / 1000.0

        self.worker.q_continuous = float(self.spin_qc.value())
        self.worker.max_buffer_size = int(self.spin_buffer_size.value())

        self.lbl_feedback.setText("[İŞLENİYOR] Sentetik veriler ve EKF yörüngeleri yeniden hesaplanıyor...")
        self.worker.recompute_all_modes()
        self.worker.reset()

        self.lbl_feedback.setText("[AKTİF] Yeni parametreler canlı sentetik veri akışına uygulandı.")
        self.applied_parameters.emit()
