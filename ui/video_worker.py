"""
ui/video_worker.py
------------------
Worker de video que corre en QThread separado.

Responsabilidades:
1. Capturar frames de la webcam
2. Pasar frames al RecognitionEngine
3. Renderizar anotaciones
4. Emitir señal con el frame anotado + resultado

Por qué en QThread y no QRunnable:
- Necesitamos un loop continuo de captura
- Necesitamos controlar el ciclo de vida (stop)
- QThread es más apropiado para workers de larga duración
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from config.loader import cfg
from core.camera import CameraCapture
from core.recognition_engine import RecognitionEngine
from core.types import BoundingBox, DetectedFace, FrameResult, Landmark
from utils.logger import get_logger
from utils.renderer import FrameRenderer

logger = get_logger(__name__)


def _scale_frame_result(result: FrameResult, inv_scale: float) -> None:
    """Escala coordenadas de rostros al tamaño del frame de visualización."""
    if inv_scale == 1.0:
        return

    for face in result.faces:
        b = face.bbox
        face.bbox = BoundingBox(
            x1=int(b.x1 * inv_scale),
            y1=int(b.y1 * inv_scale),
            x2=int(b.x2 * inv_scale),
            y2=int(b.y2 * inv_scale),
            confidence=b.confidence,
        )
        if face.landmark is not None:
            face.landmark = Landmark(
                points=(face.landmark.points * inv_scale).astype(np.float32)
            )


class VideoWorker(QObject):
    """
    Worker de procesamiento de video.
    Debe ser movido a un QThread con moveToThread().

    Señales:
        frame_ready: emitida con (frame_anotado, FrameResult)
        error_signal: emitida con mensaje de error
    """

    frame_ready = pyqtSignal(np.ndarray, object)  # (frame BGR, FrameResult)
    error_signal = pyqtSignal(str)

    def __init__(self, engine: RecognitionEngine) -> None:
        super().__init__()
        self._engine = engine
        self._renderer = FrameRenderer()
        self._camera: Optional[CameraCapture] = None
        self._running = False
        self._suspended = False
        self._state_lock = threading.Lock()

        perf = getattr(cfg, "performance", None)
        self._inference_max_width = int(
            getattr(perf, "inference_max_width", 640) if perf else 640
        )
        self._min_emit_interval_ms = float(
            getattr(perf, "ui_emit_interval_ms", 33) if perf else 33
        )
        self._inference_min_interval_ms = float(
            getattr(perf, "inference_min_interval_ms", 150) if perf else 150
        )

        self._last_emit_time = 0.0
        self._last_inference_time = 0.0
        self._display_fps = 0.0
        self._display_fps_times: list[float] = []

        # Inferencia en hilo aparte: el video no se congela cuando hay rostros
        self._result_lock = threading.Lock()
        self._last_result: Optional[FrameResult] = None
        self._infer_busy = False

    def run(self) -> None:
        """
        Loop principal del worker.
        Llamado automáticamente cuando el QThread arranca.
        """
        logger.info("VideoWorker iniciando...")
        self._running = True

        if not self._open_camera():
            return

        logger.info("VideoWorker en ejecución.")

        while self._running:
            try:
                with self._state_lock:
                    suspended = self._suspended

                if suspended:
                    time.sleep(0.05)
                    continue

                if self._camera is None:
                    if not self._open_camera():
                        time.sleep(0.5)
                        continue

                success, frame, frame_id = self._camera.read()

                if not success or frame is None:
                    if self._camera and self._camera.error:
                        self.error_signal.emit(self._camera.error)
                        break
                    time.sleep(0.01)
                    continue

                now_ms = time.perf_counter() * 1000

                # ── Inferencia en segundo plano (no bloquea el preview) ──
                if (
                    self._engine.is_ready
                    and not self._infer_busy
                    and now_ms - self._last_inference_time >= self._inference_min_interval_ms
                ):
                    infer_frame, inv_scale = self._prepare_inference_frame(frame)
                    self._schedule_inference(infer_frame, frame_id, inv_scale)
                    self._last_inference_time = now_ms

                # ── Emitir preview fluido a la UI ─────────────────────
                if now_ms - self._last_emit_time >= self._min_emit_interval_ms:
                    with self._result_lock:
                        cached = self._last_result
                    if cached is not None:
                        annotated = self._renderer.render(frame, cached)
                        emit_result = cached
                    else:
                        annotated = frame
                        emit_result = FrameResult(frame_id=frame_id)

                    emit_result.display_fps = self._update_display_fps()
                    self.frame_ready.emit(annotated, emit_result)
                    self._last_emit_time = now_ms

            except Exception as e:
                logger.error(f"Error en VideoWorker loop: {e}", exc_info=True)
                time.sleep(0.1)

        if self._camera:
            self._camera.stop()
            self._camera = None

        logger.info("VideoWorker terminado.")

    def suspend_camera(self) -> None:
        """Libera la cámara para que otro componente (p. ej. registro) la use."""
        with self._state_lock:
            self._suspended = True
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        logger.info("VideoWorker: cámara suspendida.")

    def resume_camera(self) -> None:
        """Vuelve a tomar control de la cámara tras suspend_camera()."""
        with self._state_lock:
            self._suspended = False
        logger.info("VideoWorker: reanudando cámara...")

    def stop(self) -> None:
        """Detiene el worker de forma limpia."""
        self._running = False
        logger.info("VideoWorker detenido.")

    def _open_camera(self) -> bool:
        """Abre la cámara si no está activa."""
        if self._camera is not None and self._camera.is_running:
            return True

        self._camera = CameraCapture()
        if not self._camera.start():
            self.error_signal.emit("No se pudo abrir la cámara")
            self._camera = None
            return False
        return True

    def _schedule_inference(
        self,
        infer_frame: np.ndarray,
        frame_id: int,
        inv_scale: float,
    ) -> None:
        """Lanza process_frame en un hilo para no frenar la captura/UI."""
        self._infer_busy = True
        frame_copy = infer_frame.copy()

        def _run() -> None:
            try:
                result = self._engine.process_frame(frame_copy, frame_id)
                _scale_frame_result(result, inv_scale)
                with self._result_lock:
                    self._last_result = result
            except Exception as e:
                logger.error(f"Error en inferencia: {e}", exc_info=True)
            finally:
                self._infer_busy = False

        threading.Thread(
            target=_run,
            name="FaceInference",
            daemon=True,
        ).start()

    def _prepare_inference_frame(
        self, frame: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Reduce el frame para inferencia más rápida.

        Returns:
            (frame_para_inferencia, factor_inv_scale para coordenadas)
        """
        h, w = frame.shape[:2]
        max_w = self._inference_max_width
        if w <= max_w:
            return frame, 1.0

        scale = max_w / w
        new_h = int(h * scale)
        small = cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
        return small, 1.0 / scale

    def _update_display_fps(self) -> float:
        """Calcula FPS de la UI (frames mostrados por segundo)."""
        now = time.perf_counter()
        self._display_fps_times.append(now)
        if len(self._display_fps_times) > 30:
            self._display_fps_times.pop(0)
        if len(self._display_fps_times) >= 2:
            elapsed = self._display_fps_times[-1] - self._display_fps_times[0]
            if elapsed > 0:
                self._display_fps = (len(self._display_fps_times) - 1) / elapsed
        return self._display_fps
