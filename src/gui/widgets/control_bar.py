"""
Tactical Playback Control Bar for UAV Sensor Fusion GUI.
Provides Play/Pause/Reset actions, Speed Multipliers (0.5x, 1x, 2x, Max),
Radio Mode Switcher (Mode A: Ideal, Mode B: Naive, Mode C: Rollback/Replay),
and interactive Timeline Scrubbing Slider.
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
    Bottom control dock containing timeline slider and simulation management controls.
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

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 8)
        main_layout.setSpacing(6)

        # ---------------- 1. Timeline Slider Row ----------------
        slider_row = QHBoxLayout()
        slider_row.setSpacing(10)

        self.lbl_time_current = QLabel("00:00.0 s")
        self.lbl_time_current.setStyleSheet("color: #00FF66; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        slider_row.addWidget(self.lbl_time_current)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, int(self.sim_duration * 100))  # 10ms resolution
        self.time_slider.setValue(0)
        self.time_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #111A20;
                border: 1px solid #1E2E38;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00FF66;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #00FF66;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #00FF66;
            }
        """)
        self.time_slider.sliderPressed.connect(self._on_slider_pressed)
        self.time_slider.sliderReleased.connect(self._on_slider_released)
        self.time_slider.sliderMoved.connect(self._on_slider_moved)
        slider_row.addWidget(self.time_slider)

        self.lbl_time_total = QLabel(f"{self.sim_duration:04.1f} s")
        self.lbl_time_total.setStyleSheet("color: #6C8A99; font-family: 'Consolas', monospace; font-size: 11px;")
        slider_row.addWidget(self.lbl_time_total)

        main_layout.addLayout(slider_row)

        # ---------------- 2. Control Toolbar Row ----------------
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(16)

        # Play / Pause Button
        self.btn_play_pause = QPushButton("▶ BAŞLAT (PLAY)")
        self.btn_play_pause.setCheckable(True)
        self.btn_play_pause.setStyleSheet("""
            QPushButton {
                background-color: #00AA44;
                color: #FFFFFF;
                border: 1px solid #00DD55;
                border-radius: 4px;
                padding: 6px 14px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00CC55;
            }
            QPushButton:checked {
                background-color: #CC7A00;
                border: 1px solid #FFAA00;
            }
        """)
        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        toolbar_row.addWidget(self.btn_play_pause)

        # Reset Button
        self.btn_reset = QPushButton("⏹ SIFIRLA (RESET)")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #222E36;
                color: #DDEEFA;
                border: 1px solid #364B59;
                border-radius: 4px;
                padding: 6px 12px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2E3E49;
            }
        """)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        toolbar_row.addWidget(self.btn_reset)

        # Speed Selector
        lbl_speed = QLabel("HIZ:")
        lbl_speed.setStyleSheet("color: #8CAAB9; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        toolbar_row.addWidget(lbl_speed)

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.5x (Yavaş)", "1.0x (Canlı)", "2.0x (Hızlı)", "Azami (Max Speed)"])
        self.combo_speed.setCurrentIndex(1)  # Default 1.0x
        self.combo_speed.setStyleSheet("""
            QComboBox {
                background-color: #111A20;
                color: #00FF66;
                border: 1px solid #1E2E38;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 11px;
            }
            QComboBox QAbstractItemView {
                background-color: #111A20;
                color: #00FF66;
                selection-background-color: #1E3B2B;
            }
        """)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        toolbar_row.addWidget(self.combo_speed)

        # Separator Line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #203340;")
        toolbar_row.addWidget(sep)

        # Simulation Mode Radio Selector
        lbl_mode = QLabel("FÜZYON MODU:")
        lbl_mode.setStyleSheet("color: #FFB300; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        toolbar_row.addWidget(lbl_mode)

        self.radio_group = QButtonGroup(self)
        self.rb_mode_a = QRadioButton("Mod A (İdeal)")
        self.rb_mode_b = QRadioButton("Mod B (Naive EKF)")
        self.rb_mode_c = QRadioButton("Mod C (Rollback/OOSM Destekli)")
        self.rb_mode_c.setChecked(True)  # Default Mode C

        radio_style = """
            QRadioButton {
                color: #C0D8E6;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                spacing: 5px;
            }
            QRadioButton::indicator:checked {
                background-color: #00FF66;
                border: 2px solid #FFFFFF;
                border-radius: 6px;
                width: 10px;
                height: 10px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #1A2832;
                border: 1px solid #3B5466;
                border-radius: 6px;
                width: 10px;
                height: 10px;
            }
        """
        for rb, mode_id in [(self.rb_mode_a, "MODE_A"), (self.rb_mode_b, "MODE_B"), (self.rb_mode_c, "MODE_C")]:
            rb.setStyleSheet(radio_style)
            self.radio_group.addButton(rb)
            toolbar_row.addWidget(rb)

        self.radio_group.buttonClicked.connect(self._on_mode_radio_clicked)

        toolbar_row.addStretch()
        main_layout.addLayout(toolbar_row)

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_play_pause_clicked(self) -> None:
        if self.btn_play_pause.isChecked():
            self.btn_play_pause.setText("⏸ DURAKLAT (PAUSE)")
            self.play_requested.emit()
        else:
            self.btn_play_pause.setText("▶ BAŞLAT (PLAY)")
            self.pause_requested.emit()

    def set_playing_state(self, is_playing: bool) -> None:
        """Updates button text when playback state changes externally."""
        self.btn_play_pause.setChecked(is_playing)
        if is_playing:
            self.btn_play_pause.setText("⏸ DURAKLAT (PAUSE)")
        else:
            self.btn_play_pause.setText("▶ BAŞLAT (PLAY)")

    def _on_reset_clicked(self) -> None:
        self.btn_play_pause.setChecked(False)
        self.btn_play_pause.setText("▶ BAŞLAT (PLAY)")
        self.time_slider.setValue(0)
        self.lbl_time_current.setText("00:00.0 s")
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
        self.lbl_time_current.setText(f"{target_time:04.1f} s")
        self.seek_requested.emit(target_time)

    def update_frame(self, frame: SimulationFrame) -> None:
        """Synchronizes slider and time label during active playback."""
        if not self._is_user_scrubbing:
            slider_val = int(frame.timestamp * 100)
            self.time_slider.setValue(slider_val)
            self.lbl_time_current.setText(f"{frame.timestamp:04.1f} s")
