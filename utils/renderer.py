"""
utils/renderer.py
-----------------
Renderizado de anotaciones sobre frames de video con OpenCV.

Dibuja:
- Bounding boxes con colores según identidad
- Nombre del usuario + porcentaje de confianza
- Indicador anti-spoofing
- FPS y latencia
- Indicador de estado del sistema
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config.loader import cfg
from core.types import DetectedFace, FrameResult, IdentityStatus, SpoofStatus
from utils.logger import get_logger

logger = get_logger(__name__)

# Fuente OpenCV
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_PLAIN


class FrameRenderer:
    """
    Renderizador de anotaciones faciales sobre frames BGR.
    Todos los métodos son estáticos / sin estado para facilitar
    su uso en cualquier contexto.
    """

    def __init__(self) -> None:
        self._show_confidence = cfg.ui.show_confidence
        self._show_spoof = cfg.ui.show_spoof_indicator
        self._show_fps = cfg.ui.show_fps
        self._show_landmarks = cfg.ui.show_landmarks
        self._bbox_thickness = cfg.ui.bbox_thickness
        self._font_scale = cfg.ui.font_scale

    def render(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """
        Dibuja todas las anotaciones sobre una copia del frame.

        Args:
            frame: Frame BGR original.
            result: FrameResult con rostros procesados.

        Returns:
            Frame anotado (nueva copia).
        """
        output = frame.copy()

        for face in result.faces:
            self._draw_face(output, face)

        if self._show_fps:
            self._draw_hud(output, result)

        return output

    def _draw_face(self, frame: np.ndarray, face: DetectedFace) -> None:
        """Dibuja bbox, etiqueta e indicadores para un rostro."""
        bbox = face.bbox
        color_bgr = face.display_color

        # ── Bounding box ──────────────────────────────────────
        # Solo dibujar si no es desconocido (ignorar desconocidos)
        if face.identity_status == IdentityStatus.UNKNOWN:
            # Desconocidos: solo un bbox gris muy tenue
            cv2.rectangle(
                frame,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (50, 50, 50),
                1,
            )
            return

        # Rostros conocidos / spoof / low confidence: bbox completo
        cv2.rectangle(
            frame,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color_bgr,
            self._bbox_thickness,
        )

        # Esquinas decorativas (estilo cyberpunk)
        self._draw_corner_marks(frame, bbox, color_bgr)

        # ── Etiqueta superior ────────────────────────────────
        label = face.display_label
        if label:
            self._draw_label(frame, bbox, label, color_bgr, face)

        # ── Indicador anti-spoofing ──────────────────────────
        if self._show_spoof and face.spoof_status != SpoofStatus.DISABLED:
            self._draw_spoof_indicator(frame, bbox, face)

        # ── Landmarks (opcional) ─────────────────────────────
        if self._show_landmarks and face.landmark is not None:
            self._draw_landmarks(frame, face)

    def _draw_label(
        self,
        frame: np.ndarray,
        bbox,
        label: str,
        color: tuple[int, int, int],
        face: DetectedFace,
    ) -> None:
        """Dibuja la etiqueta de nombre con fondo semi-transparente."""
        font_scale = self._font_scale
        thickness = 1
        padding = 6

        (tw, th), baseline = cv2.getTextSize(label, _FONT, font_scale, thickness)

        # Posición: encima del bbox
        x = bbox.x1
        y = bbox.y1 - padding

        # Si no cabe arriba, poner dentro
        if y - th - padding < 0:
            y = bbox.y1 + th + padding

        # Fondo semi-transparente
        bg_x1 = x - 2
        bg_y1 = y - th - padding
        bg_x2 = x + tw + padding
        bg_y2 = y + baseline

        # Clamp a bordes del frame
        h, w = frame.shape[:2]
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(w, bg_x2)
        bg_y2 = min(h, bg_y2)

        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Texto
        cv2.putText(
            frame, label,
            (x + 2, y),
            _FONT, font_scale, color, thickness,
            cv2.LINE_AA,
        )

        # Barra de confianza (si es usuario conocido)
        if face.identity_status == IdentityStatus.KNOWN and self._show_confidence:
            self._draw_confidence_bar(frame, bbox, face.identity_confidence, color)

    def _draw_confidence_bar(
        self,
        frame: np.ndarray,
        bbox,
        confidence: float,
        color: tuple[int, int, int],
    ) -> None:
        """Barra horizontal de confianza debajo del bbox."""
        bar_h = 4
        bar_y = bbox.y2 + 3
        bar_w = bbox.width

        h, w = frame.shape[:2]
        if bar_y + bar_h >= h:
            return

        # Fondo de la barra
        cv2.rectangle(
            frame,
            (bbox.x1, bar_y),
            (bbox.x2, bar_y + bar_h),
            (40, 40, 40), -1,
        )

        # Relleno proporcional a la confianza
        fill_w = int(bar_w * confidence)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bbox.x1, bar_y),
                (bbox.x1 + fill_w, bar_y + bar_h),
                color, -1,
            )

    def _draw_spoof_indicator(
        self,
        frame: np.ndarray,
        bbox,
        face: DetectedFace,
    ) -> None:
        """Indicador de estado anti-spoofing (esquina inferior derecha del bbox)."""
        if face.spoof_status == SpoofStatus.PENDING:
            icon = "?"
            color = (100, 100, 100)
        elif face.spoof_status == SpoofStatus.REAL:
            icon = "✓"
            color = (100, 212, 0)
        elif face.spoof_status == SpoofStatus.SPOOF:
            icon = "✗"
            color = (60, 60, 220)
        else:
            return

        x = bbox.x2 - 20
        y = bbox.y2 - 5
        cv2.putText(frame, icon, (x, y), _FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_corner_marks(
        self,
        frame: np.ndarray,
        bbox,
        color: tuple[int, int, int],
        length: int = 15,
        thickness: int = 2,
    ) -> None:
        """Marcas de esquina decorativas (estilo HUD)."""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        corners = [
            # (punto_inicio, punto_fin) para cada esquina
            ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
            ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
            ((x1, y2 - length), (x1, y2), (x1 + length, y2)),
            ((x2 - length, y2), (x2, y2), (x2, y2 - length)),
        ]

        for pts in corners:
            cv2.line(frame, pts[0], pts[1], color, thickness, cv2.LINE_AA)
            cv2.line(frame, pts[1], pts[2], color, thickness, cv2.LINE_AA)

    def _draw_landmarks(self, frame: np.ndarray, face: DetectedFace) -> None:
        """Dibuja los 5 puntos de referencia faciales."""
        if face.landmark is None:
            return
        colors_pts = [(0, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 0)]
        for i, pt in enumerate(face.landmark.points):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 2, colors_pts[i], -1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, result: FrameResult) -> None:
        """HUD de rendimiento en esquina superior izquierda."""
        h, w = frame.shape[:2]
        lines = []

        ui_fps = result.display_fps if result.display_fps > 0 else result.fps
        if ui_fps > 0:
            lines.append(f"FPS: {ui_fps:.1f}")
        if result.fps > 0 and abs(ui_fps - result.fps) > 1:
            lines.append(f"Infer: {result.fps:.1f}")
        if result.processing_time_ms > 0:
            lines.append(f"Latencia: {result.processing_time_ms:.0f}ms")

        known = len(result.known_users)
        total = len(result.faces)
        lines.append(f"Rostros: {total} | Conocidos: {known}")

        for i, line in enumerate(lines):
            y = 20 + i * 18
            # Fondo semi-transparente para el HUD
            cv2.putText(
                frame, line,
                (11, y + 1),
                _FONT_SMALL, 1.1, (10, 10, 10), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, line,
                (10, y),
                _FONT_SMALL, 1.1, (180, 230, 255), 1, cv2.LINE_AA,
            )

    def render_no_signal(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Frame de placeholder cuando la cámara no está disponible."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        msg = "Sin señal de cámara"
        (tw, th), _ = cv2.getTextSize(msg, _FONT, 0.8, 1)
        x = (width - tw) // 2
        y = (height + th) // 2
        cv2.putText(frame, msg, (x, y), _FONT, 0.8, (100, 100, 100), 1, cv2.LINE_AA)
        return frame

    def render_loading(
        self,
        width: int = 640,
        height: int = 480,
        message: str = "Cargando...",
    ) -> np.ndarray:
        """Frame de carga durante inicialización."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(message, _FONT, 0.7, 1)
        x = (width - tw) // 2
        y = (height + th) // 2
        cv2.putText(frame, message, (x, y), _FONT, 0.7, (0, 180, 120), 1, cv2.LINE_AA)
        return frame
