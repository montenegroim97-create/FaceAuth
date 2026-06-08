"""
ui/registration_dialog.py
--------------------------
Diálogo de registro de nuevos usuarios.

Flujo:
1. Usuario introduce nombre y ID
2. Elige fuente: webcam o cargar imágenes
3. (Webcam) Captura automática de N fotos con guías
4. (Imágenes) Selector de archivos múltiples
5. Extracción de embeddings
6. Registro en FaceDatabase
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.loader import cfg, ConfigLoader
from core.camera import CameraCapture
from core.recognition_engine import RecognitionEngine
from core.types import RegisteredUser
from utils.logger import get_logger

logger = get_logger(__name__)


class RegistrationDialog(QDialog):
    """
    Diálogo modal para registrar un nuevo usuario biométrico
    o añadir más fotos a un usuario existente.
    Soporta registro por webcam y por carga de imágenes.
    """

    def __init__(
        self,
        engine: RecognitionEngine,
        parent: Optional[QWidget] = None,
        existing_user: Optional[RegisteredUser] = None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._existing_user = existing_user
        self._captured_embeddings: list[np.ndarray] = []
        self._camera: Optional[CameraCapture] = None
        self._capture_timer: Optional[QTimer] = None
        self._capturing = False
        self._photos_target = cfg.registration.photos_required

        if existing_user:
            title = f"Añadir fotos - {existing_user.name}"
            self._photos_target = max(4, cfg.registration.photos_required // 2)
        else:
            title = "Registrar Nuevo Usuario"

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(700, 550)
        self.resize(750, 580)
        self._apply_style()
        self._build_ui()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f6fa;
                color: #2d3436;
            }
            QTabWidget::pane {
                border: 1px solid #dcdde1;
                background-color: #ffffff;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #eef0f5;
                color: #636e72;
                padding: 8px 20px;
                border: 1px solid #dcdde1;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #00b894;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e0e2e8;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                color: #2d3436;
                padding: 7px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #00b894;
            }
            QPushButton {
                background-color: #ffffff;
                color: #2d3436;
                border: 1px solid #dcdde1;
                border-radius: 8px;
                padding: 7px 18px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #00b894;
            }
            QPushButton#btnCapture {
                background-color: #00b89422;
                border-color: #00b894;
                color: #00b894;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton#btnCapture:hover { background-color: #00b89444; }
            QPushButton#btnCapture:disabled {
                background-color: #e8f8f5;
                color: #8ac4b0;
                border-color: #8ac4b0;
            }
            QProgressBar {
                background-color: #eef0f5;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                text-align: center;
                color: #2d3436;
                height: 22px;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #00b894;
                border-radius: 5px;
            }
            QLabel#videoPreview {
                background-color: #ffffff;
                border: 1px solid #dcdde1;
                border-radius: 8px;
            }
        """)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 16)

        # ── Datos del usuario ─────────────────────────────────
        form_widget = self._build_form()
        layout.addWidget(form_widget)

        # ── Tabs: Webcam / Imágenes ───────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_webcam_tab(), "📷  Webcam")
        self._tabs.addTab(self._build_images_tab(), "🗂  Cargar Imágenes")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs, stretch=1)

        # ── Progreso ──────────────────────────────────────────
        progress_layout = QVBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, self._photos_target)
        self._progress_bar.setValue(0)
        target_text = f"{self._photos_target} muestras"
        self._progress_label = QLabel(f"0 / {target_text} capturadas")
        self._progress_label.setStyleSheet("color: #636e72; font-size: 11px;")
        progress_layout.addWidget(self._progress_bar)
        progress_layout.addWidget(self._progress_label)
        layout.addLayout(progress_layout)

        # ── Botones OK/Cancel ─────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_ok_text = "Añadir Fotos" if self._existing_user else "Registrar"
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(btn_ok_text)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setObjectName("btnCapture")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self._on_reject)
        layout.addWidget(buttons)

    def _build_form(self) -> QWidget:
        """Formulario de nombre y ID de usuario."""
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Nombre
        grid.addWidget(QLabel("Nombre completo:"), 0, 0)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Ej: Juan García")
        if self._existing_user:
            self._name_input.setText(self._existing_user.name)
            self._name_input.setReadOnly(True)
            self._name_input.setStyleSheet("color: #636e72;")
        grid.addWidget(self._name_input, 0, 1)

        # ID
        grid.addWidget(QLabel("ID de usuario:"), 1, 0)
        self._id_input = QLineEdit()
        if self._existing_user:
            self._id_input.setText(self._existing_user.user_id)
            self._id_input.setReadOnly(True)
            self._id_input.setStyleSheet("color: #636e72;")
        else:
            self._id_input.setPlaceholderText(f"Ej: user_{uuid.uuid4().hex[:6]}")
            self._id_input.setText(f"user_{uuid.uuid4().hex[:6]}")
        grid.addWidget(self._id_input, 1, 1)

        grid.setColumnStretch(1, 1)
        return widget

    def _build_webcam_tab(self) -> QWidget:
        """Tab de captura por webcam."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(12)

        # Preview de webcam
        left = QVBoxLayout()
        self._cam_preview = QLabel("Vista previa de cámara")
        self._cam_preview.setObjectName("videoPreview")
        self._cam_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_preview.setMinimumSize(320, 240)
        self._cam_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        left.addWidget(self._cam_preview)
        layout.addLayout(left, stretch=1)

        # Controles
        right = QVBoxLayout()
        right.setSpacing(10)

        # Prompt de pose
        self._pose_label = QLabel("Posiciona tu rostro frente a la cámara")
        self._pose_label.setWordWrap(True)
        self._pose_label.setStyleSheet(
            "color: #00b894; font-size: 12px; padding: 8px; "
            "background: #e8f8f5; border-radius: 6px;"
        )
        right.addWidget(self._pose_label)

        # Botón iniciar/detener captura
        self._btn_capture = QPushButton("▶  Iniciar Captura")
        self._btn_capture.setObjectName("btnCapture")
        self._btn_capture.clicked.connect(self._toggle_webcam_capture)
        right.addWidget(self._btn_capture)

        # Info
        extra = " adicionales" if self._existing_user else ""
        info = QLabel(
            f"Se capturarán {self._photos_target} fotos{extra} automáticamente.\n"
            "Mueve levemente la cabeza durante la captura\n"
            "para cubrir distintos ángulos."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #636e72; font-size: 10px;")
        right.addWidget(info)

        right.addStretch()
        layout.addLayout(right)

        return widget

    def _build_images_tab(self) -> QWidget:
        """Tab de carga de imágenes desde disco."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Instrucciones
        info = QLabel(
            f"Selecciona al menos {cfg.recognition.min_embeddings_per_user} imágenes "
            f"(recomendado: {cfg.registration.photos_recommended}+).\n"
            "Las imágenes deben mostrar el rostro claramente con buena iluminación."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #636e72; font-size: 11px;")
        layout.addWidget(info)

        # Botón seleccionar
        btn_select = QPushButton("Seleccionar Imágenes...")
        btn_select.clicked.connect(self._select_images)
        layout.addWidget(btn_select)

        # Preview de imágenes seleccionadas
        self._images_info_label = QLabel("Ninguna imagen seleccionada")
        self._images_info_label.setStyleSheet("color: #636e72; font-size: 11px;")
        self._images_info_label.setWordWrap(True)
        layout.addWidget(self._images_info_label)

        # Grid de thumbnails
        self._thumbnails_widget = QWidget()
        self._thumbnails_layout = QGridLayout(self._thumbnails_widget)
        self._thumbnails_layout.setSpacing(4)
        layout.addWidget(self._thumbnails_widget)

        layout.addStretch()
        return widget

    # ──────────────────────────────────────────────────────────
    # Lógica de captura por webcam
    # ──────────────────────────────────────────────────────────

    def _toggle_webcam_capture(self) -> None:
        if not self._capturing:
            self._start_webcam_capture()
        else:
            self._stop_webcam_capture()

    def _start_webcam_capture(self) -> None:
        """Inicia la captura automática por webcam."""
        self._camera = CameraCapture()
        if not self._camera.start():
            QMessageBox.warning(self, "Error", "No se pudo acceder a la cámara.")
            return

        self._capturing = True
        self._btn_capture.setText("⏹  Detener")

        # Timer para captura automática
        self._capture_timer = QTimer(self)
        self._capture_timer.timeout.connect(self._auto_capture_frame)
        self._capture_timer.start(cfg.registration.capture_delay_ms)

        # Timer para preview
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_camera_preview)
        self._preview_timer.start(33)  # ~30fps preview

        self._update_pose_prompt()

    def _stop_webcam_capture(self) -> None:
        """Detiene la captura por webcam."""
        self._capturing = False

        if self._capture_timer:
            self._capture_timer.stop()
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        if self._camera:
            self._camera.stop()
            self._camera = None

        self._btn_capture.setText("▶  Iniciar Captura")

    def _update_camera_preview(self) -> None:
        """Actualiza el preview de la cámara en el diálogo."""
        if not self._camera:
            return
        success, frame, _ = self._camera.read()
        if not success or frame is None:
            return

        # Convertir a QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self._cam_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_preview.setPixmap(pixmap)

    def _auto_capture_frame(self) -> None:
        """Captura automáticamente un frame y extrae su embedding."""
        if not self._camera or len(self._captured_embeddings) >= self._photos_target:
            self._stop_webcam_capture()
            return

        success, frame, _ = self._camera.read()
        if not success or frame is None:
            return

        embedding = self._engine.extract_embedding(frame)
        if embedding is not None:
            self._captured_embeddings.append(embedding)
            self._update_progress()
            self._update_pose_prompt()

            if len(self._captured_embeddings) >= self._photos_target:
                self._stop_webcam_capture()
                self._pose_label.setText("✓ Captura completada. Presiona Registrar.")

    def _update_pose_prompt(self) -> None:
        """Actualiza el mensaje de guía de pose."""
        prompts = cfg.registration.prompts
        idx = min(
            len(self._captured_embeddings),
            len(prompts) - 1,
        )
        self._pose_label.setText(prompts[idx])

    # ──────────────────────────────────────────────────────────
    # Lógica de carga de imágenes
    # ──────────────────────────────────────────────────────────

    def _select_images(self) -> None:
        """Abre selector de archivos y extrae embeddings."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar imágenes",
            str(Path.home()),
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp)",
        )

        if not files:
            return

        self._images_info_label.setText(f"Procesando {len(files)} imágenes...")
        self.repaint()

        # Limpiar embeddings anteriores de imágenes
        self._captured_embeddings = []
        valid_count = 0

        for path in files:
            frame = cv2.imread(path)
            if frame is None:
                continue

            embedding = self._engine.extract_embedding(frame)
            if embedding is not None:
                self._captured_embeddings.append(embedding)
                valid_count += 1

        self._images_info_label.setText(
            f"{valid_count}/{len(files)} imágenes con rostros detectados"
        )
        self._update_progress()
        self._add_thumbnails(files[:12])  # Mostrar hasta 12 thumbnails

    def _add_thumbnails(self, paths: list[str]) -> None:
        """Añade thumbnails de las imágenes seleccionadas."""
        # Limpiar thumbnails anteriores
        while self._thumbnails_layout.count():
            item = self._thumbnails_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        for i, path in enumerate(paths):
            frame = cv2.imread(path)
            if frame is None:
                continue
            thumb = cv2.resize(frame, (60, 60))
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            qi = QImage(rgb.data, 60, 60, 180, QImage.Format.Format_RGB888)
            lbl = QLabel()
            lbl.setPixmap(QPixmap.fromImage(qi))
            lbl.setFixedSize(64, 64)
            lbl.setStyleSheet("border: 1px solid #dcdde1; border-radius: 3px;")
            self._thumbnails_layout.addWidget(lbl, i // cols, i % cols)

    # ──────────────────────────────────────────────────────────
    # Progreso y validación
    # ──────────────────────────────────────────────────────────

    def _update_progress(self) -> None:
        """Actualiza la barra de progreso."""
        count = len(self._captured_embeddings)
        self._progress_bar.setValue(min(count, self._photos_target))
        self._progress_label.setText(
            f"{count} / {self._photos_target} muestras capturadas"
        )

    def _on_tab_changed(self, index: int) -> None:
        """Al cambiar de tab, detener captura webcam si estaba activa."""
        if index != 0 and self._capturing:
            self._stop_webcam_capture()

    # ──────────────────────────────────────────────────────────
    # Aceptar / Cancelar
    # ──────────────────────────────────────────────────────────

    def _on_accept(self) -> None:
        """Valida y registra el usuario."""
        if self._capturing:
            self._stop_webcam_capture()

        name = self._name_input.text().strip()
        user_id = self._id_input.text().strip()

        # Validaciones
        if not name:
            QMessageBox.warning(self, "Error", "El nombre no puede estar vacío.")
            return

        if not user_id:
            QMessageBox.warning(self, "Error", "El ID de usuario no puede estar vacío.")
            return

        min_emb = cfg.recognition.min_embeddings_per_user
        if len(self._captured_embeddings) < min_emb:
            QMessageBox.warning(
                self, "Muestras insuficientes",
                f"Se necesitan al menos {min_emb} muestras.\n"
                f"Tienes: {len(self._captured_embeddings)}.\n\n"
                "Captura más fotos por webcam o carga más imágenes.",
            )
            return

        # Registrar o añadir fotos
        if self._existing_user:
            success = self._engine.append_user_embeddings(
                user_id=user_id,
                embeddings=self._captured_embeddings,
            )
            msg_title = "Fotos añadidas"
            msg_body = (
                f"Se añadieron {len(self._captured_embeddings)} muestras a '{name}'.\n"
                f"Total: {self._existing_user.num_samples + len(self._captured_embeddings)} muestras."
            )
        else:
            success = self._engine.register_user(
                user_id=user_id,
                name=name,
                embeddings=self._captured_embeddings,
            )
            msg_title = "Éxito"
            msg_body = (
                f"Usuario '{name}' registrado correctamente.\n"
                f"Muestras: {len(self._captured_embeddings)}"
            )

        if success:
            QMessageBox.information(self, msg_title, msg_body)
            self.accept()
        else:
            QMessageBox.critical(
                self, "Error",
                "No se pudo completar la operación. Revisa los logs.",
            )

    def _on_reject(self) -> None:
        """Cancela y limpia recursos."""
        if self._capturing:
            self._stop_webcam_capture()
        self.reject()

    def closeEvent(self, event) -> None:
        if self._capturing:
            self._stop_webcam_capture()
        event.accept()
