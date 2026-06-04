"""
core/recognition_engine.py
---------------------------
Motor de reconocimiento facial: orquesta el pipeline completo.

Pipeline por frame:
  Frame BGR
    → Detector (SCRFD) → DetectedFace[]
    → Tracker (IoU)    → track_id asignado, identidad cacheada o PENDING
    → AntiSpoof        → spoof_status actualizado
    → Embedder check   → si PENDING: consultar FAISS
    → FAISS search     → IdentityStatus + user + confianza
    → FrameResult      → entregado a la UI

Decisión de re-identificación:
  - Tracks PENDING siempre consultan FAISS
  - Tracks con identidad conocida consultan cada reid_interval frames
  - Tracks SPOOF no consultan FAISS (ahorra cómputo)
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from config.loader import cfg
from core.anti_spoof import AntiSpoofing
from core.database import FaceDatabase
from core.detector import FaceDetector
from core.tracker import FaceTracker
from core.types import DetectedFace, FrameResult, IdentityStatus, SpoofStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class RecognitionEngine:
    """
    Orquestador del pipeline biométrico completo.

    Uso:
        engine = RecognitionEngine()
        engine.initialize()
        result = engine.process_frame(bgr_frame, frame_id)
    """

    def __init__(self) -> None:
        self._detector = FaceDetector()
        self._tracker = FaceTracker()
        self._anti_spoof = AntiSpoofing()
        self._database = FaceDatabase()
        self._initialized = False

        # Métricas de rendimiento
        self._frame_times: list[float] = []
        self._max_time_history = 30

    def initialize(self) -> bool:
        """
        Inicializa todos los componentes del pipeline.

        Returns:
            True si todos los componentes críticos están listos.
        """
        logger.info("=" * 60)
        logger.info("Inicializando RecognitionEngine...")
        logger.info("=" * 60)

        results = {}

        # Detector es crítico
        results["detector"] = self._detector.initialize()

        # Database es crítico
        results["database"] = self._database.initialize()

        # Anti-spoofing es importante pero no bloquea
        results["anti_spoof"] = self._anti_spoof.initialize()

        # Verificar componentes críticos
        if not results["detector"]:
            logger.critical("FaceDetector falló. El sistema no puede arrancar.")
            return False

        if not results["database"]:
            logger.critical("FaceDatabase falló. El sistema no puede arrancar.")
            return False

        if not results["anti_spoof"]:
            logger.warning(
                "AntiSpoofing no disponible. "
                "Descarga los modelos con: python scripts/download_models.py"
            )

        self._initialized = True
        logger.info("=" * 60)
        logger.info(f"RecognitionEngine listo.")
        logger.info(f"  Usuarios registrados: {self._database.user_count}")
        logger.info(f"  Anti-spoofing: {'✓' if results['anti_spoof'] else '✗ (sin modelos)'}")
        logger.info("=" * 60)
        return True

    # ──────────────────────────────────────────────────────────
    # Pipeline principal
    # ──────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> FrameResult:
        """
        Procesa un frame BGR completo a través del pipeline.

        Args:
            frame: Imagen BGR de OpenCV.
            frame_id: Número de frame (para métricas).

        Returns:
            FrameResult con todas las detecciones e identidades.
        """
        t_start = time.perf_counter()

        if not self._initialized:
            return FrameResult(frame_id=frame_id, error="Engine no inicializado")

        if frame is None or frame.size == 0:
            return FrameResult(frame_id=frame_id, error="Frame inválido")

        try:
            # ── Etapa 1: Detección SCRFD ─────────────────────────
            faces = self._detector.detect(frame)

            if not faces:
                return FrameResult(
                    frame_id=frame_id,
                    faces=[],
                    processing_time_ms=self._elapsed_ms(t_start),
                    fps=self._current_fps(),
                )

            # ── Etapa 2: Tracking IoU ────────────────────────────
            faces = self._tracker.update(faces)

            # ── Etapa 3: Anti-Spoofing ───────────────────────────
            faces = self._run_anti_spoof(faces)

            # ── Etapa 4: Reconocimiento (FAISS) ──────────────────
            faces = self._run_recognition(faces)

            # ── Etapa 5: Actualizar tracker con resultados ────────
            self._sync_tracker(faces)

            # ── Etapa 6: Limpiar tracks de anti-spoof ─────────────
            active_ids = self._tracker.get_active_track_ids()
            self._anti_spoof.cleanup_stale_tracks(active_ids)

            elapsed = self._elapsed_ms(t_start)
            self._record_time(elapsed)

            return FrameResult(
                frame_id=frame_id,
                faces=faces,
                processing_time_ms=elapsed,
                fps=self._current_fps(),
            )

        except Exception as e:
            logger.error(f"Error en process_frame #{frame_id}: {e}", exc_info=True)
            return FrameResult(
                frame_id=frame_id,
                error=str(e),
                processing_time_ms=self._elapsed_ms(t_start),
            )

    def _run_anti_spoof(self, faces: list[DetectedFace]) -> list[DetectedFace]:
        """Ejecuta anti-spoofing en todos los rostros detectados."""
        for face in faces:
            face = self._anti_spoof.analyze(face)
        return faces

    def _run_recognition(self, faces: list[DetectedFace]) -> list[DetectedFace]:
        """
        Ejecuta reconocimiento facial (FAISS search) solo en rostros
        que necesitan re-identificación.
        """
        for face in faces:
            # No identificar spoofs
            if face.spoof_status == SpoofStatus.SPOOF:
                face.identity_status = IdentityStatus.UNKNOWN
                continue

            # Solo re-identificar si está PENDING
            if face.identity_status != IdentityStatus.PENDING:
                continue

            if face.embedding is None:
                face.identity_status = IdentityStatus.UNKNOWN
                continue

            # Usar embedding promedio del tracker si está disponible
            track = self._tracker.get_track(face.track_id)
            embedding = None
            if track is not None:
                embedding = track.get_mean_embedding()
            if embedding is None:
                embedding = face.embedding

            # Consultar FAISS
            status, user, confidence, distance = self._database.search(embedding)

            face.identity_status = status
            face.identity_confidence = confidence
            face.identity_distance = distance

            if user is not None:
                face.user_id = user.user_id
                face.user_name = user.name
            else:
                face.user_id = None
                face.user_name = None

        return faces

    def _sync_tracker(self, faces: list[DetectedFace]) -> None:
        """Sincroniza los resultados de reconocimiento con el tracker."""
        for face in faces:
            if face.track_id < 0:
                continue

            self._tracker.update_identity(
                track_id=face.track_id,
                status=face.identity_status,
                user_id=face.user_id,
                user_name=face.user_name,
                confidence=face.identity_confidence,
                distance=face.identity_distance,
            )
            self._tracker.update_spoof(
                track_id=face.track_id,
                spoof_status=face.spoof_status,
                spoof_confidence=face.spoof_confidence,
            )

    # ──────────────────────────────────────────────────────────
    # Registro de usuarios (delegado a FaceDatabase)
    # ──────────────────────────────────────────────────────────

    def register_user(
        self,
        user_id: str,
        name: str,
        embeddings: list[np.ndarray],
        face_image_path: Optional[str] = None,
    ) -> bool:
        """Registra un nuevo usuario en la base de datos biométrica."""
        return self._database.add_user(user_id, name, embeddings, face_image_path)

    def remove_user(self, user_id: str) -> bool:
        """Elimina un usuario de la base de datos."""
        return self._database.remove_user(user_id)

    def append_user_embeddings(
        self,
        user_id: str,
        embeddings: list[np.ndarray],
    ) -> bool:
        """Añade más embeddings a un usuario existente."""
        return self._database.append_embeddings(user_id, embeddings)

    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae el embedding ArcFace del rostro más prominente en un frame.
        Usado durante el flujo de registro de usuarios.

        Returns:
            Embedding normalizado (512,) o None si no hay rostro.
        """
        faces = self._detector.detect(frame)
        if not faces:
            return None

        # Tomar el rostro más grande (más cercano a la cámara)
        best = max(faces, key=lambda f: f.bbox.area)
        return best.embedding

    def extract_embeddings_batch(
        self,
        frames: list[np.ndarray],
    ) -> list[Optional[np.ndarray]]:
        """Extrae embeddings de una lista de frames."""
        return [self.extract_embedding(f) for f in frames]

    # ──────────────────────────────────────────────────────────
    # Métricas
    # ──────────────────────────────────────────────────────────

    def _elapsed_ms(self, t_start: float) -> float:
        return (time.perf_counter() - t_start) * 1000

    def _record_time(self, elapsed_ms: float) -> None:
        self._frame_times.append(elapsed_ms)
        if len(self._frame_times) > self._max_time_history:
            self._frame_times.pop(0)

    def _current_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg_ms = np.mean(self._frame_times[-10:])
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self._frame_times:
            return 0.0
        return float(np.mean(self._frame_times))

    @property
    def database(self) -> FaceDatabase:
        return self._database

    @property
    def is_ready(self) -> bool:
        return self._initialized
