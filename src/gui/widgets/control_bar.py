"""
Tactical Playback Control Bar for UAV Sensor Fusion GUI.
Provides Play/Pause/Reset actions, Speed Multipliers (0.5x, 1x, 2x, Max),
Radio Mode Switcher (Mode A: Ideal, Mode B: Naive, Mode C: Rollback/Replay),
and interactive Timeline Scrubbing Slider in a sharp tactical deck layout.
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QLabel,
    QFrame,
)

from src.gui.worker import SimulationFrame


class PlaybackControlBar(QWidget):
    """
    Bottom tactical control deck containing timeline slider and simulation management controls.
    """
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    seek_requested = pyqtSignal(float)  # target time in seconds
    speed_changed = pyqtSignal(float)   # multiplier: 0.5, 1.0, 2.0, 0.0
    mode_selected = pyqtSignal(str)     # "MODE_A", "MODE_B", "MODE_C"

    def __init__(self, sim_duration: float = 20.0, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sim_duration = float(sim_duration)
        self._is_user_scrubbing = False

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        deck_frame = QFrame()
        deck_frame.setStyleSheet("""
            QFrame {
                background-color: #090E14;
                border: 1px solid #182634;
                border-top: 2px solid #1E3246;
            }
        """)
        main_layout = QVBoxLayout(deck_frame)
        main_layout.setContentsMargins(10, 6, 10, 6)
        main_layout.setSpacing(6)

        # ---------------- 1. Timeline Slider Row ----------------
        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)

        lbl_epoch_tag = QLabel("ZAMAN EKSENİ [EPOCH]:")
        lbl_epoch_tag.setStyleSheet("color: #68899E; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        slider_row.addWidget(lbl_epoch_tag)

        self.lbl_time_current = QLabel("T+00.00 s")
        self.lbl_time_current.setStyleSheet("color: #10B981; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; border: none;")
        slider_row.addWidget(self.lbl_time_current)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, int(self.sim_duration * 100))  # 10ms resolution
        self.time_slider.setValue(0)
        self.time_slider.setStyleSheet("""
            QSlider {
                border: none;
                background: transparent;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #0F1822;
                border: 1px solid #1E3042;
            }
            QSlider::sub-page:horizontal {
                background: #10B981;
            }
            QSlider::handle:horizontal {
                background: #E2EEF8;
                border: 1px solid #10B981;
                width: 10px;
                margin-top: -4px;
                margin-bottom: -4px;
            }
            QSlider::handle:horizontal:hover {
                background: #10B981;
            }
        """)
        self.time_slider.sliderPressed.connect(self._on_slider_pressed)
        self.time_slider.sliderReleased.connect(self._on_slider_released)
        self.time_slider.sliderMoved.connect(self._on_slider_moved)
        slider_row.addWidget(self.time_slider, stretch=1)

        self.lbl_time_total = QLabel(f"T+{self.sim_duration:05.2f} s")
        self.lbl_time_total.setStyleSheet("color: #68899E; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        slider_row.addWidget(self.lbl_time_total)

        main_layout.addLayout(slider_row)

        # ---------------- 2. Control Toolbar Row ----------------
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(12)

        # Play / Pause Button
        self.btn_play_pause = QPushButton("[ > ] SİMÜLASYONU BAŞLAT")
        self.btn_play_pause.setCheckable(True)
        self.btn_play_pause.setStyleSheet("""
            QPushButton {
                background-color: #0D3B2A;
                color: #34D399;
                border: 1px solid #10B981;
                border-left: 3px solid #10B981;
                padding: 5px 14px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #134E38;
                color: #FFFFFF;
            }
            QPushButton:checked {
                background-color: #4A2C06;
                color: #FBBF24;
                border: 1px solid #F59E0B;
                border-left: 3px solid #F59E0B;
            }
        """)
        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        toolbar_row.addWidget(self.btn_play_pause)

        # Reset Button
        self.btn_reset = QPushButton("[ 0 ] SIFIRLA")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #101923;
                color: #9FB6C8;
                border: 1px solid #223548;
                padding: 5px 12px;
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
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        toolbar_row.addWidget(self.btn_reset)

        # Speed Selector
        lbl_speed = QLabel("KOŞUM HIZI:")
        lbl_speed.setStyleSheet("color: #7FA0B8; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        toolbar_row.addWidget(lbl_speed)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x [YAVAŞ ANALİZ]", "1.0x [GERÇEK ZAMANLI]", "2.0x [HIZLANDIRILMIŞ]", "MAX [AZAMİ HIZ]"])
        self.combo_speed.setCurrentIndex(1)  # Default 1.0x
        self.combo_speed.setStyleSheet("""
            QComboBox {
                background-color: #0C131B;
                color: #38BDF8;
                border: 1px solid #1E3246;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #0C131B;
                color: #38BDF8;
                selection-background-color: #162B3D;
                border: 1px solid #1E3246;
            }
        """)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        toolbar_row.addWidget(self.combo_speed)

        # Separator Line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #1C2D3E;")
        toolbar_row.addWidget(sep)

        # Simulation Mode Radio Selector
        lbl_mode = QLabel("FÜZYON MİMARİSİ:")
        lbl_mode.setStyleSheet("color: #F59E0B; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; border: none;")
        toolbar_row.addWidget(lbl_mode)

        self.radio_group = QButtonGroup(self)
        self.rb_mode_a = QRadioButton("MOD A [İDEAL / 0ms]")
        self.rb_mode_b = QRadioButton("MOD B [NAIVE EKF]")
        self.rb_mode_c = QRadioButton("MOD C [OOSM ROLLBACK/REPLAY]")
        self.rb_mode_c.setChecked(True)  # Default Mode C

        radio_style = """
            QRadioButton {
                color: #C4D6E5;
                font-family: 'Consolas', monospace;
                font-size: 10px;
                font-weight: bold;
                spacing: 6px;
                border: none;
            }
            QRadioButton::indicator:checked {
                background-color: #10B981;
                border: 1px solid #FFFFFF;
                width: 9px;
                height: 9px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #0F1720;
                border: 1px solid #354E66;
                width: 9px;
                height: 9px;
            }
        """
        for rb in (self.rb_mode_a, self.rb_mode_b, self.rb_mode_c):
            rb.setStyleSheet(radio_style)
            self.radio_group.addButton(rb)
            toolbar_row.addWidget(rb)

        self.radio_group.buttonClicked.connect(self._on_mode_radio_clicked)

        toolbar_row.addStretch()
        main_layout.addLayout(toolbar_row)
        outer_layout.addWidget(deck_frame)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_play_pause_clicked(self) -> None:
        if self.btn_play_pause.isChecked():
            self.btn_play_pause.setText("[ || ] SİMÜLASYONU DURAKLAT")
            self.play_requested.emit()
        else:
            self.btn_play_pause.setText("[ > ] SİMÜLASYONU BAŞLAT")
            self.pause_requested.emit()

    def set_playing_state(self, is_playing: bool) -> None:
        """Updates button text when playback state changes externally."""
        self.btn_play_pause.setChecked(is_playing)
        if is_playing:
            self.btn_play_pause.setText("[ || ] SİMÜLASYONU DURAKLAT")
        else:
            self.btn_play_pause.setText("[ > ] SİMÜLASYONU BAŞLAT")

    def _on_reset_clicked(self) -> None:
        self.btn_play_pause.setChecked(False)
        self.btn_play_pause.setText("[ > ] SİMÜLASYONU BAŞLAT")
        self.time_slider.setValue(0)
        self.lbl_time_current.setText("T+00.00 s")
        self.reset_requested.emit()

    def _on_speed_changed(self, index: int) -> None:
        speeds = [0.5, 1.0, 2.0, 0.0]
        self.speed_changed.emit(speeds[index])

    def _on_mode_radio_clicked(self, button: QRadioButton) -> None:
        if button == self.rb_mode_a:
            self.mode_selected.emit("MODE_A")
        elif button == self.rb_mode_b:
            self.mode_selected.emit("MODE_B")
        elif button == self.rb_mode_c:
            self.mode_selected.emit("MODE_C")

    def _on_slider_pressed(self) -> None:
        self._is_user_scrubbing = True

    def _on_slider_released(self) -> None:
        self._is_user_scrubbing = False
        target_time = self.time_slider.value() / 100.0
        self.seek_requested.emit(target_time)

    def _on_slider_moved(self, value: int) -> None:
        target_time = value / 100.0
        self.lbl_time_current.setText(f"T+{target_time:05.2f} s")
        self.seek_requested.emit(target_time)

    def update_frame(self, frame: SimulationFrame) -> None:
        """Synchronizes slider and time label during active playback."""
        if not self._is_user_scrubbing:
            slider_val = int(frame.timestamp * 100)
            self.time_slider.setValue(slider_val)
            self.lbl_time_current.setText(f"T+{frame.timestamp:05.2f} s")
