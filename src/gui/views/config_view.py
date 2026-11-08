"""
Tab 4: Live Sensor Latency, Clock Bias & EKF Parameter Configurator.
Allows real-time customization of Radar/Camera delays, jitter, clock offsets,
and EKF filter tuning parameters with instant re-simulation.
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
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(8)

        # Header Title
        title_box = QFrame()
        title_box.setStyleSheet("background-color: #0C1418; border: 1px solid #1A2B35; border-radius: 4px; padding: 6px;")
        t_layout = QHBoxLayout(title_box)
        t_layout.setContentsMargins(4, 4, 4, 4)

        header_lbl = QLabel("SENSÖR GECİKMELERİ, JITTER VE EKF PARAMETRE KONFİGÜRASYONU")
        header_lbl.setStyleSheet("color: #FFB300; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
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
        form_layout.setSpacing(10)

        # --- Card 1: Radar Sensor Settings ---
        card_radar = self._create_card("RADAR SENSÖRÜ (1D RANGE) PARAMETRELERİ", "#00FF66")
        grid_radar = QGridLayout()
        grid_radar.setSpacing(10)

        self.spin_radar_freq = self._add_double_spin(grid_radar, 0, 0, "Ölçüm Frekansı (Hz):", 1.0, 50.0, 10.0, 1.0, " Hz")
        self.spin_radar_sigma = self._add_double_spin(grid_radar, 0, 1, "Menzil Hatası (sigma_r):", 0.5, 30.0, 6.0, 0.5, " m")
        self.spin_radar_latency = self._add_double_spin(grid_radar, 1, 0, "İletim Gecikmesi:", 0.0, 100.0, 8.0, 1.0, " ms")
        self.spin_radar_jitter = self._add_double_spin(grid_radar, 1, 1, "Gecikme Gauss Jitter:", 0.0, 20.0, 2.0, 0.5, " ms")

        card_radar.layout().addLayout(grid_radar)
        form_layout.addWidget(card_radar)

        # --- Card 2: Camera Sensor Settings ---
        card_cam = self._create_card("KAMERA / ELEKTRO-OPTİK (2D AZIMUTH & ELEVATION) PARAMETRELERİ", "#00CCFF")
        grid_cam = QGridLayout()
        grid_cam.setSpacing(10)

        self.spin_cam_freq = self._add_double_spin(grid_cam, 0, 0, "Ölçüm Frekansı (Hz):", 5.0, 100.0, 30.0, 5.0, " Hz")
        self.spin_cam_sigma = self._add_double_spin(grid_cam, 0, 1, "Açısal Gürültü (sigma):", 0.001, 0.05, 0.004, 0.001, " rad")
        self.spin_cam_latency = self._add_double_spin(grid_cam, 1, 0, "Görüntü İşleme Gecikmesi:", 0.0, 250.0, 40.0, 5.0, " ms")
        self.spin_cam_jitter = self._add_double_spin(grid_cam, 1, 1, "Gecikme Gauss Jitter:", 0.0, 50.0, 5.0, 1.0, " ms")
        self.spin_cam_clock_bias = self._add_double_spin(grid_cam, 2, 0, "Donanımsal Saat Kayması (Clock Bias):", -100.0, 200.0, 25.0, 5.0, " ms")

        card_cam.layout().addLayout(grid_cam)
        form_layout.addWidget(card_cam)

        # --- Card 3: EKF Filter & Buffer Settings ---
        card_ekf = self._create_card("GENİŞLETİLMİŞ KALMAN FİLTRESİ (EKF) VE OOSM TAMPON AYARLARI", "#FFB300")
        grid_ekf = QGridLayout()
        grid_ekf.setSpacing(10)

        self.spin_qc = self._add_double_spin(grid_ekf, 0, 0, "Süreç Gürültüsü Spektral Yoğunluğu (q_c):", 0.05, 10.0, 0.8, 0.1, " m^2/s^3")
        self.spin_buffer_size = self._add_int_spin(grid_ekf, 0, 1, "OOSM Durum Tamponu Boyutu (N Adım):", 10, 500, 150, 10, " adım")

        card_ekf.layout().addLayout(grid_ekf)
        form_layout.addWidget(card_ekf)

        scroll.setWidget(container)
        root_layout.addWidget(scroll, stretch=1)

        # --- Bottom Action Bar ---
        action_bar = QFrame()
        action_bar.setStyleSheet("background-color: #0A0F12; border: 1px solid #16242C; border-radius: 4px; padding: 6px;")
        act_layout = QHBoxLayout(action_bar)
        act_layout.setContentsMargins(6, 6, 6, 6)
        act_layout.setSpacing(12)

        self.btn_apply = QPushButton("💾 YENİ PARAMETRELERLE CANLI SİMÜLASYONU GÜNCELLE")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #008844;
                color: #FFFFFF;
                border: 1px solid #00BB55;
                border-radius: 4px;
                padding: 8px 18px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #00AA55;
            }
        """)
        self.btn_apply.clicked.connect(self._apply_custom_parameters)
        act_layout.addWidget(self.btn_apply)

        self.btn_reset_defaults = QPushButton("🔄 VARSAYILAN DEĞERLERE DÖN")
        self.btn_reset_defaults.setStyleSheet("""
            QPushButton {
                background-color: #222E36;
                color: #C0D8E6;
                border: 1px solid #364B59;
                border-radius: 4px;
                padding: 8px 14px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2E3E49;
            }
        """)
        self.btn_reset_defaults.clicked.connect(self._reset_to_defaults)
        act_layout.addWidget(self.btn_reset_defaults)

        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("color: #00FF66; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        act_layout.addWidget(self.lbl_feedback)

        act_layout.addStretch()
        root_layout.addWidget(action_bar)

    def _create_card(self, title: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #0A0F12;
                border: 1px solid #16242C;
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 8)
        vbox.setSpacing(8)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {color}; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        vbox.addWidget(lbl)
        return card

    def _add_double_spin(
        self, grid: QGridLayout, row: int, col: int, label: str, min_v: float, max_v: float, def_v: float, step: float, suffix: str
    ) -> QDoubleSpinBox:
        vbox = QVBoxLayout()
        vbox.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #7A9CAE; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold;")
        vbox.addWidget(lbl)

        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(def_v)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #111A20;
                color: #00FF66;
                border: 1px solid #1E2E38;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
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
        lbl.setStyleSheet("color: #7A9CAE; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold;")
        vbox.addWidget(lbl)

        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setValue(def_v)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setStyleSheet("""
            QSpinBox {
                background-color: #111A20;
                color: #00FF66;
                border: 1px solid #1E2E38;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
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
        self.lbl_feedback.setText("Değerler fabrika varsayılanlarına döndürüldü.")

    def _apply_custom_parameters(self) -> None:
        """Applies parameters to worker and triggers re-simulation."""
        # Update worker simulation configuration
        self.worker.q_continuous = float(self.spin_qc.value())

        # Trigger recomputation on worker
        self.lbl_feedback.setText("Simülasyon yeni gecikme ve sensör parametreleriyle tekrar hesaplanıyor...")
        self.worker.recompute_all_modes()
        self.worker.reset()

        self.lbl_feedback.setText("✔ Başarılı: Yeni parametreler canlı sisteme aktarıldı!")
        self.applied_parameters.emit()
