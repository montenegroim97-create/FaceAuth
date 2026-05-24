"""
core/camera.py
--------------
Captura de webcam usando OpenCV en hilo dedicado.

El buffer de captura corre en un thread separado para evitar
que el procesamiento de frames bloquee la adquisición.
Esto garantiza que siempre tengamos el frame más reciente
disponible, descartando frames viejos automáticamente.

Diseño:
- Thread daemon de captura → escribe en buffer circular (size=1)
- Hilo principal → lee el último frame disponible
- Lock para acceso thread-safe al frame
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from config.loader import cfg
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraCapture:
    """
    Captura de webcam en hilo dedicado.

    Uso:
        cam = CameraCapture()
        if cam.start():
            frame = cam.read()
            cam.stop()
    """

    def __init__(self, device_id: Optional[int] = None) -> None:
        self._device_id = device_id if device_id is not None else cfg.camera.device_id
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

        # Config
        self._width = cfg.camera.width
        self._height = cfg.camera.height
        self._fps = cfg.camera.fps

    def start(self) -> bool:
        """
        Abre la cámara y arranca el hilo de captura.

        Returns:
            True si la cámara se abrió correctamente.
        """
        if self._running:
            return True

        logger.info(f"Abriendo cámara (device_id={self._device_id})...")

        try:
            # CAP_DSHOW en Windows evita latencia extra del backend DirectShow
            backend = cv2.CAP_DSHOW if self._is_windows() else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self._device_id, backend)

            if not self._cap.isOpened():
                logger.error(f"No se pudo abrir cámara {self._device_id}")
                return False

            # Configurar resolución y FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.camera.buffer_size)

            # Leer resolución real (puede diferir de lo solicitado)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Cámara lista: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

            # Capturar un frame de prueba
            ret, test_frame = self._cap.read()
            if not ret or test_frame is None:
                logger.error("La cámara no devuelve frames.")
                self._cap.release()
                return False

            with self._lock:
                self._frame = test_frame
                self._frame_id = 1

            # Arrancar hilo de captura
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="CameraCapture",
                daemon=True,
            )
            self._thread.start()
            logger.info("Hilo de captura iniciado.")
            return True

        except Exception as e:
            logger.error(f"Error abriendo cámara: {e}", exc_info=True)
            return False

    def _capture_loop(self) -> None:
        """
        Loop principal del hilo de captura.
        Lee frames continuamente y actualiza el buffer.
        """
        consecutive_errors = 0
        max_errors = 30

        while self._running:
            try:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logger.error(f"Cámara falló {max_errors} veces consecutivas.")
                        self._error = "Cámara desconectada"
                        self._running = False
                        break
                    time.sleep(0.033)
                    continue

                consecutive_errors = 0

                with self._lock:
                    self._frame = frame
                    self._frame_id += 1

            except Exception as e:
                logger.error(f"Error en capture_loop: {e}")
                consecutive_errors += 1
                time.sleep(0.1)

        logger.info("Hilo de captura terminado.")

    def read(self) -> tuple[bool, Optional[np.ndarray], int]:
        """
        Lee el último frame disponible.

        Returns:
            (success, frame_bgr, frame_id)
            success = False si no hay frame disponible.
        """
        with self._lock:
            if self._frame is None:
                return False, None, 0
            return True, self._frame.copy(), self._frame_id

    def stop(self) -> None:
        """Detiene la captura y libera recursos."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("CameraCapture detenida.")

    def get_frame_size(self) -> tuple[int, int]:
        """Retorna (width, height) del frame actual."""
        with self._lock:
            if self._frame is not None:
                h, w = self._frame.shape[:2]
                return w, h
        return self._width, self._height

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    @staticmethod
    def _is_windows() -> bool:
        import sys
        return sys.platform == "win32"

    def __enter__(self) -> CameraCapture:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
