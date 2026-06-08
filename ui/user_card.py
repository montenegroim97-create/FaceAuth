"""
ui/user_card.py
---------------
Widget de tarjeta de usuario para el panel lateral.
Muestra el nombre, ID y estado en tiempo real (activo/inactivo).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from core.types import RegisteredUser
from utils.logger import get_logger

logger = get_logger(__name__)


class UserCard(QWidget):
    """
    Tarjeta visual para un usuario registrado en el sidebar.

    Señales:
        delete_requested: emitida con el user_id cuando se pide borrar
        add_photos_requested: emitida con el user_id para añadir más fotos
    """

    delete_requested = pyqtSignal(str)
    add_photos_requested = pyqtSignal(str)

    def __init__(
        self,
        user: RegisteredUser,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._user = user
        self._active = False
        self._build_ui()
        self._apply_inactive_style()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        layout.setSpacing(6)

        # Indicador de estado (punto verde/gris)
        self._indicator = QLabel("●")
        self._indicator.setFixedWidth(12)
        self._indicator.setStyleSheet("color: #d0d0d0; font-size: 13px;")
        layout.addWidget(self._indicator)

        # Info del usuario
        info_layout = QWidget()
        info_v = QHBoxLayout(info_layout)
        info_v.setContentsMargins(0, 0, 0, 0)
        info_v.setSpacing(4)

        self._name_label = QLabel(self._user.name)
        self._name_label.setStyleSheet("color: #2d3436; font-size: 12px; font-weight: bold;")

        self._id_label = QLabel(f"#{self._user.user_id}")
        self._id_label.setStyleSheet("color: #999999; font-size: 10px;")

        info_v.addWidget(self._name_label)
        info_v.addWidget(self._id_label)
        info_v.addStretch()
        layout.addWidget(info_layout, stretch=1)

        # Botón añadir fotos (+ pequeño)
        self._btn_add = QPushButton("+")
        self._btn_add.setFixedSize(22, 22)
        self._btn_add.setToolTip("Añadir más fotos a este usuario")
        self._btn_add.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #dcdde1;
                border-radius: 4px;
                color: #636e72;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                border-color: #00b894;
                color: #00b894;
                background: #00b89411;
            }
        """)
        self._btn_add.clicked.connect(lambda: self.add_photos_requested.emit(self._user.user_id))
        layout.addWidget(self._btn_add)

        # Botón eliminar (X pequeño)
        self._btn_delete = QPushButton("✕")
        self._btn_delete.setFixedSize(22, 22)
        self._btn_delete.setToolTip("Eliminar usuario")
        self._btn_delete.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #d0d0d0;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #e05050;
            }
        """)
        self._btn_delete.clicked.connect(lambda: self.delete_requested.emit(self._user.user_id))
        layout.addWidget(self._btn_delete)

    def set_active(self, active: bool) -> None:
        """Actualiza el estado visual de la tarjeta."""
        if self._active == active:
            return
        self._active = active
        if active:
            self._apply_active_style()
        else:
            self._apply_inactive_style()

    def _apply_active_style(self) -> None:
        """Estilo cuando el usuario está siendo reconocido."""
        self.setStyleSheet("""
            UserCard {
                background-color: #e8f8f5;
                border: 1px solid #00b894;
                border-radius: 8px;
            }
        """)
        self._indicator.setStyleSheet("color: #00b894; font-size: 13px;")
        self._name_label.setStyleSheet("color: #00b894; font-size: 12px; font-weight: bold;")

    def _apply_inactive_style(self) -> None:
        """Estilo cuando el usuario no está en frame."""
        self.setStyleSheet("""
            UserCard {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            UserCard:hover {
                border-color: #b0b0b0;
                background-color: #fafafa;
            }
        """)
        self._indicator.setStyleSheet("color: #d0d0d0; font-size: 13px;")
        self._name_label.setStyleSheet("color: #636e72; font-size: 12px; font-weight: bold;")
