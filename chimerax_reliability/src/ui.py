"""GUI for half-map reliability coloring (Q-score-style volume pickers)."""

from __future__ import annotations

import textwrap

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)

from chimerax.core.errors import UserError
from chimerax.ui.gui import MainToolWindow


class _Layout(QVBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 2, 0, 0)
        self.setSpacing(4)


class _HLayout(QHBoxLayout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(4)


class VolumePicker(QToolButton):
    """Dropdown menu of open Volume models."""

    def __init__(self, session, label: str, on_change):
        super().__init__()
        self.session = session
        self._on_change = on_change
        self._volume = None
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMinimumWidth(160)
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(self._populate)
        self.setMenu(self._menu)
        self._set_label(None, label)

    @property
    def volume(self):
        return self._volume

    def _set_label(self, volume, placeholder: str) -> None:
        if volume is None:
            self.setText(placeholder)
            self.setToolTip(placeholder)
            return
        self.setText(f"#{volume.id_string}: {textwrap.shorten(volume.name, 14)}")
        self.setToolTip(f"#{volume.id_string}: {volume.name}")

    def _populate(self) -> None:
        from chimerax.map import Volume

        self._menu.clear()
        volumes = sorted(self.session.models.list(type=Volume), key=lambda m: m.id)
        if not volumes:
            action = self._menu.addAction("(no maps open)")
            action.setEnabled(False)
            return
        for vol in volumes:
            action = self._menu.addAction(f"#{vol.id_string}: {vol.name}")

            def _choose(_, v=vol):
                self._volume = v
                self._set_label(v, "")
                self._on_change(v)

            action.triggered.connect(_choose)

    def try_auto_select(self, volumes) -> None:
        if self._volume is not None:
            return
        if volumes:
            self._volume = volumes[0]
            self._set_label(volumes[0], "")


class MapReliabilityPanel(QFrame):
    def __init__(self, session):
        super().__init__()
        self.session = session
        layout = _Layout()
        self.setLayout(layout)

        row = _HLayout()
        row.addWidget(QLabel("Map:"))
        self.map_picker = VolumePicker(session, "Select map", self._map_changed)
        row.addWidget(self.map_picker)
        layout.addLayout(row)

        row = _HLayout()
        row.addWidget(QLabel("Half-map 1:"))
        self.half1_picker = VolumePicker(session, "Select half-map 1", self._update_work_info)
        row.addWidget(self.half1_picker)
        layout.addLayout(row)

        row = _HLayout()
        row.addWidget(QLabel("Half-map 2:"))
        self.half2_picker = VolumePicker(session, "Select half-map 2", self._update_work_info)
        row.addWidget(self.half2_picker)
        layout.addLayout(row)

        row = _HLayout()
        row.addWidget(QLabel("Contour:"))
        self.contour_spin = QDoubleSpinBox()
        self.contour_spin.setDecimals(4)
        self.contour_spin.setRange(-1e6, 1e6)
        self.contour_spin.setSingleStep(0.01)
        self.contour_spin.setValue(0.116)
        self.contour_spin.setToolTip("Density threshold for the macromolecular mask.")
        row.addWidget(self.contour_spin)
        row.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 21)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(5)
        row.addWidget(self.window_spin)
        layout.addLayout(row)

        row = _HLayout()
        row.addWidget(QLabel("Transparency:"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 95)
        self.transparency_slider.setValue(82)
        self.transparency_slider.setToolTip(
            "Surface transparency (0 = opaque, 95 = nearly invisible). "
            "Thesis figures use 82%."
        )
        self.transparency_value = QLabel("82%")
        self.transparency_slider.valueChanged.connect(
            lambda v: self.transparency_value.setText(f"{v}%")
        )
        row.addWidget(self.transparency_slider, stretch=1)
        row.addWidget(self.transparency_value)
        layout.addLayout(row)

        row = _HLayout()
        self.score_radio = QRadioButton("Reliability score")
        self.zones_radio = QRadioButton("Build zones")
        self.score_radio.setChecked(True)
        row.addWidget(self.score_radio)
        row.addWidget(self.zones_radio)
        layout.addLayout(row)

        row = _HLayout()
        self.run_button = QPushButton("Compute && Color")
        self.run_button.setToolTip(
            "Compute half-map reliability in ChimeraX (same math as cryoem_mrc), "
            "then color the deposited map surface."
        )
        self.run_button.clicked.connect(self._run)
        row.addWidget(self.run_button)
        layout.addLayout(row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.info_label = QLabel("Select the deposited map and both half-maps.")
        self.info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.info_label)

        self.contour_spin.valueChanged.connect(self._update_work_info)
        layout.addStretch()

    def _update_work_info(self, _value=None) -> None:
        from .volumes import describe_map_work

        reference = self.map_picker.volume
        half1 = self.half1_picker.volume
        half2 = self.half2_picker.volume
        if reference is None or half1 is None or half2 is None:
            self.info_label.setText("Select the deposited map and both half-maps.")
            return
        self.info_label.setText(describe_map_work([reference, half1, half2]))

    def _map_changed(self, volume) -> None:
        from .volumes import default_contour_level

        level = default_contour_level(volume)
        if level is not None:
            self.contour_spin.setValue(level)
        self._update_work_info()

    def _selected_mode(self) -> str:
        return "zones" if self.zones_radio.isChecked() else "score"

    def _run(self) -> None:
        reference = self.map_picker.volume
        half1 = self.half1_picker.volume
        half2 = self.half2_picker.volume
        if reference is None or half1 is None or half2 is None:
            raise UserError("Select the deposited map and both half-maps.")

        from .cmd import run_reliability_coloring

        self.run_button.setEnabled(False)
        self.status_label.setText("Starting...")

        def _progress(msg: str) -> None:
            self.status_label.setText(msg)
            QApplication.processEvents()

        try:
            run_reliability_coloring(
                self.session,
                reference,
                half1,
                half2,
                contour=float(self.contour_spin.value()),
                mode=self._selected_mode(),
                window=int(self.window_spin.value()),
                transparency=float(self.transparency_slider.value()),
                progress=_progress,
            )
        finally:
            self.run_button.setEnabled(True)
            self._update_work_info()


class MapReliabilityWindow(MainToolWindow):
    def __init__(self, tool_instance, **kw):
        super().__init__(tool_instance, **kw)
        layout = _Layout()
        self.ui_area.setLayout(layout)
        self.panel = MapReliabilityPanel(self.session)
        layout.addWidget(self.panel)
        self.manage(placement="side")

        from chimerax.map import Volume

        volumes = sorted(self.session.models.list(type=Volume), key=lambda m: m.id)
        if len(volumes) >= 3:
            self.panel.map_picker.try_auto_select([volumes[0]])
            self.panel.half1_picker.try_auto_select([volumes[1]])
            self.panel.half2_picker.try_auto_select([volumes[2]])
            self.panel._update_work_info()
