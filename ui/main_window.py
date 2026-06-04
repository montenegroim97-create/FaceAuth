"""
ui/main_window.py
-----------------
Ventana principal PyQt6.

Layout:
┌─────────────────────────────────────────┬─────────────┐
│                                         │             │
│         Video Feed (webcam)             │  Sidebar    │
│         con anotaciones faciales        │  (usuarios  │
│                                         │  + acciones)│
│                                         │             │
└─────────────────────────────────────────┴─────────────┘

El pipeline de video corre en VideoWorker (QThread separado)
para no bloquear la UI de Qt.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config.loader import cfg
from core.camera import CameraCapture
from core.recognition_engine import RecognitionEngine
from core.types import FrameResult
from ui.registration_dialog import RegistrationDialog
from ui.user_card import UserCard
from ui.video_worker import VideoWorker
from utils.logger import get_logger
from utils.renderer import FrameRenderer

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """Ventana principal de FaceAuth MVP."""

    def __init__(self, engine: RecognitionEngine) -> None:
        super().__init__()
        self._engine = engine
        self._renderer = FrameRenderer()
        self._user_cards: dict[str, UserCard] = {}

        self._setup_window()
        self._build_ui()
        self._setup_worker()
        self._refresh_user_list()

    # ──────────────────────────────────────────────────────────
    # Configuración de ventana
    # ──────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        """Configura propiedades base de la ventana."""
        self.setWindowTitle(cfg.ui.window_title)
        self.setMinimumSize(900, 600)
        self.resize(cfg.ui.window_width, cfg.ui.window_height)
        self._apply_dark_theme()

    def _apply_dark_theme(self) -> None:
        """Aplica stylesheet moderno a la aplicación."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f1117;
            }
            QWidget {
                background-color: #0f1117;
                color: #e4e6f0;
            }
            QPushButton {
                background-color: #1e2030;
                color: #c8cbe0;
                border: 1px solid #2e3148;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #282b40;
                border-color: #00d4aa;
                color: #e4e6f0;
            }
            QPushButton:pressed {
                background-color: #00d4aa;
                color: #0f1117;
                border-color: #00d4aa;
            }
            QPushButton#btnRegister {
                background-color: #00d4aa22;
                border-color: #00d4aa;
                color: #00d4aa;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton#btnRegister:hover {
                background-color: #00d4aa44;
                color: #00ffcc;
            }
            QLabel#videoLabel {
                background-color: #080a12;
                border: 1px solid #1e2132;
                border-radius: 8px;
            }
            QLabel#titleLabel {
                color: #00d4aa;
                font-size: 16px;
                font-weight: bold;
                padding: 8px 0px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QStatusBar {
                background-color: #0a0c14;
                color: #6b7294;
                font-size: 11px;
                border-top: 1px solid #1e2132;
            }
        """)

    # ──────────────────────────────────────────────────────────
    # Construcción de UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Construye el layout principal."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Panel izquierdo: video ────────────────────────────
        video_panel = self._build_video_panel()
        main_layout.addWidget(video_panel, stretch=3)

        # ── Panel derecho: sidebar ────────────────────────────
        sidebar = self._build_sidebar()
        main_layout.addWidget(sidebar, stretch=0)

        # ── Status bar ────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Iniciando sistema...")

    def _build_video_panel(self) -> QWidget:
        """Panel de video en vivo."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Etiqueta de video
        self._video_label = QLabel()
        self._video_label.setObjectName("videoLabel")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._video_label.setMinimumSize(640, 480)

        # Placeholder inicial
        self._show_placeholder("Iniciando cámara...")

        layout.addWidget(self._video_label)
        return widget

    def _build_sidebar(self) -> QWidget:
        """Panel lateral con lista de usuarios y controles."""
        sidebar = QWidget()
        sidebar.setFixedWidth(cfg.ui.sidebar_width)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: #0c0e1a;
                border-left: 1px solid #1e2140;
            }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Título
        title = QLabel("FaceAuth")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtítulo
        subtitle = QLabel("Sistema Biométrico")
        subtitle.setStyleSheet("color: #6b7294; font-size: 11px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Separador
        layout.addWidget(self._make_separator())

        # Sección usuarios registrados
        users_label = QLabel("USUARIOS REGISTRADOS")
        users_label.setStyleSheet("color: #6b7294; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(users_label)

        # Scroll area para tarjetas de usuarios
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._users_container = QWidget()
        self._users_layout = QVBoxLayout(self._users_container)
        self._users_layout.setContentsMargins(0, 0, 0, 0)
        self._users_layout.setSpacing(4)
        self._users_layout.addStretch()

        self._scroll_area.setWidget(self._users_container)
        layout.addWidget(self._scroll_area, stretch=1)

        # Separador
        layout.addWidget(self._make_separator())

        # Botón de registro
        self._btn_register = QPushButton("+ Registrar Usuario")
        self._btn_register.setObjectName("btnRegister")
        self._btn_register.clicked.connect(self._open_registration)
        layout.addWidget(self._btn_register)

        # Métricas
        self._metrics_label = QLabel("FPS: --  |  Latencia: --ms")
        self._metrics_label.setStyleSheet("color: #6b7294; font-size: 10px;")
        self._metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._metrics_label)

        return sidebar

    @staticmethod
    def _make_separator() -> QWidget:
        """Línea separadora horizontal."""
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1e2140;")
        return sep

    # ──────────────────────────────────────────────────────────
    # Worker de video (QThread)
    # ──────────────────────────────────────────────────────────

    def _setup_worker(self) -> None:
        """Inicializa y arranca el worker de procesamiento de video."""
        self._worker_thread = QThread()
        self._worker = VideoWorker(self._engine)

        self._worker.moveToThread(self._worker_thread)

        # Conectar señales
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker_thread.started.connect(self._worker.run)

        self._worker_thread.start()
        logger.info("VideoWorker arrancado en QThread.")

    # ──────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────

    def _on_frame_ready(
        self,
        annotated_frame: np.ndarray,
        result: FrameResult,
    ) -> None:
        """Recibe un frame procesado y lo muestra en la UI."""
        # Mostrar frame
        self._display_frame(annotated_frame)

        # Actualizar métricas (display_fps = fluidez de video; fps = inferencia)
        ui_fps = result.display_fps if result.display_fps > 0 else result.fps
        if ui_fps > 0:
            infer_note = (
                f"  |  infer: {result.fps:.1f}fps"
                if result.fps > 0 and abs(ui_fps - result.fps) > 1
                else ""
            )
            self._metrics_label.setText(
                f"FPS: {ui_fps:.1f}{infer_note}  |  {result.processing_time_ms:.0f}ms"
            )

        # Actualizar status bar
        known = len(result.known_users)
        total = len(result.faces)
        if known > 0:
            names = ", ".join(f.user_name for f in result.known_users if f.user_name)
            self._status_bar.showMessage(f"Reconocido: {names}")
        elif total > 0:
            self._status_bar.showMessage(f"{total} rostro(s) detectado(s) — no identificados")
        else:
            self._status_bar.showMessage("Sin rostros detectados")

        # Actualizar estado de usuarios en tarjetas
        self._update_user_cards(result)

    def _on_worker_error(self, error: str) -> None:
        logger.error(f"Error en VideoWorker: {error}")
        self._status_bar.showMessage(f"Error: {error}")

    def _on_add_photos(self, user_id: str) -> None:
        """Abre diálogo para añadir más fotos a un usuario existente."""
        user = self._engine.database.get_user(user_id)
        if user is None:
            return

        self._worker.suspend_camera()
        try:
            dialog = RegistrationDialog(
                self._engine,
                parent=self,
                existing_user=user,
            )
            if dialog.exec():
                self._refresh_user_list()
                logger.info(f"Fotos añadidas para: {user.name}")
        finally:
            self._worker.resume_camera()
            self._show_placeholder("Reconectando cámara...")

    def _open_registration(self) -> None:
        """Abre el diálogo de registro de nuevos usuarios."""
        # Liberar la cámara del worker: Windows no permite dos apps en device 0
        self._worker.suspend_camera()
        try:
            dialog = RegistrationDialog(self._engine, parent=self)
            if dialog.exec():
                self._refresh_user_list()
                logger.info("Usuario registrado exitosamente.")
        finally:
            self._worker.resume_camera()
            self._show_placeholder("Reconectando cámara...")

    # ──────────────────────────────────────────────────────────
    # Helpers de UI
    # ──────────────────────────────────────────────────────────

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convierte frame BGR a QPixmap y lo muestra."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            qt_image = QImage(
                rgb.data,
                w, h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()

            label_size = self._video_label.size()
            pixmap = QPixmap.fromImage(qt_image).scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            self._video_label.setPixmap(pixmap)

        except Exception as e:
            logger.debug(f"Error displaying frame: {e}")

    def _show_placeholder(self, message: str) -> None:
        """Muestra un frame de placeholder en el label de video."""
        placeholder = self._renderer.render_loading(message=message)
        self._display_frame(placeholder)

    def _refresh_user_list(self) -> None:
        """Reconstruye la lista de tarjetas de usuarios."""
        for card in self._user_cards.values():
            card.setParent(None)
        self._user_cards.clear()

        # Quitar widgets previos (tarjetas, mensaje vacío, etc.)
        while self._users_layout.count() > 1:
            item = self._users_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        users = self._engine.database.users

        if not users:
            empty_label = QLabel("No hay usuarios registrados.\nUsa el botón de abajo.")
            empty_label.setStyleSheet("color: #6b7294; font-size: 11px; padding: 20px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setWordWrap(True)
            self._users_layout.insertWidget(0, empty_label)
            return

        for uid, user in users.items():
            card = UserCard(user, parent=self._users_container)
            card.delete_requested.connect(self._on_delete_user)
            card.add_photos_requested.connect(self._on_add_photos)
            self._user_cards[uid] = card
            # Insertar antes del stretch
            idx = self._users_layout.count() - 1
            self._users_layout.insertWidget(idx, card)

    def _update_user_cards(self, result: FrameResult) -> None:
        """Actualiza el estado visual de las tarjetas según resultado."""
        # Resetear todas a inactivo
        for card in self._user_cards.values():
            card.set_active(False)

        # Activar las que están en frame
        for face in result.known_users:
            if face.user_id and face.user_id in self._user_cards:
                self._user_cards[face.user_id].set_active(True)

    def _on_delete_user(self, user_id: str) -> None:
        """Maneja la solicitud de eliminar un usuario."""
        user = self._engine.database.get_user(user_id)
        if user is None:
            return

        reply = QMessageBox.question(
            self,
            "Confirmar eliminación",
            f"¿Eliminar usuario '{user.name}'?\nEsta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._engine.remove_user(user_id):
                self._refresh_user_list()
                logger.info(f"Usuario eliminado: {user.name}")

    # ──────────────────────────────────────────────────────────
    # Ciclo de vida
    # ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Limpieza al cerrar la ventana."""
        logger.info("Cerrando FaceAuth...")

        if hasattr(self, "_worker"):
            self._worker.stop()

        if hasattr(self, "_worker_thread"):
            self._worker_thread.quit()
            self._worker_thread.wait(3000)

        event.accept()
