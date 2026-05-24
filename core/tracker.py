"""
core/tracker.py
---------------
Tracker multi-objeto basado en IoU para mantener identidades
consistentes entre frames y reducir llamadas a FAISS.

¿Por qué no usar DeepSORT o ByteTrack?
- Son dependencias pesadas innecesarias para ≤10 usuarios
- IoU Tracker es suficiente para casos con pocas personas
- Menor latencia, más predecible

Lógica central:
1. Cada rostro detectado se asigna a un "track" existente
   si IoU(bbox_nuevo, bbox_track) > threshold
2. Si no hay track compatible, se crea uno nuevo
3. Los tracks sin detección se mantienen N frames antes de eliminar
4. La identidad se re-evalúa cada K frames (reid_interval)
   para ahorrar cómputo FAISS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.loader import cfg
from core.types import BoundingBox, DetectedFace, IdentityStatus, SpoofStatus
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """Estado de un rostro trackeado a lo largo del tiempo."""
    track_id: int
    bbox: BoundingBox
    smoothed_bbox: BoundingBox          # Bbox suavizado para UI
    age: int = 0                        # Frames desde creación
    missing_frames: int = 0             # Frames sin detección
    frames_since_reid: int = 0          # Frames desde última re-identificación

    # Identidad actual
    identity_status: IdentityStatus = IdentityStatus.PENDING
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    identity_confidence: float = 0.0
    identity_distance: float = 1.0

    # Anti-spoofing
    spoof_status: SpoofStatus = SpoofStatus.PENDING
    spoof_confidence: float = 0.0

    # Historial de embeddings recientes (para promedio temporal)
    recent_embeddings: list[np.ndarray] = field(default_factory=list)
    _max_emb_history: int = 5

    def update_bbox(self, new_bbox: BoundingBox, alpha: float) -> None:
        """Actualiza bbox con suavizado exponencial."""
        self.bbox = new_bbox
        # Suavizado para reducir jitter visual
        sx1 = int(alpha * new_bbox.x1 + (1 - alpha) * self.smoothed_bbox.x1)
        sy1 = int(alpha * new_bbox.y1 + (1 - alpha) * self.smoothed_bbox.y1)
        sx2 = int(alpha * new_bbox.x2 + (1 - alpha) * self.smoothed_bbox.x2)
        sy2 = int(alpha * new_bbox.y2 + (1 - alpha) * self.smoothed_bbox.y2)
        self.smoothed_bbox = BoundingBox(sx1, sy1, sx2, sy2, new_bbox.confidence)

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Mantiene historial de los últimos N embeddings."""
        self.recent_embeddings.append(embedding.copy())
        if len(self.recent_embeddings) > self._max_emb_history:
            self.recent_embeddings.pop(0)

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Embedding promedio temporal para mayor estabilidad."""
        if not self.recent_embeddings:
            return None
        stacked = np.stack(self.recent_embeddings, axis=0)
        mean = np.mean(stacked, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-6 else mean

    def needs_reid(self) -> bool:
        """¿Es momento de re-evaluar la identidad?"""
        reid_interval = cfg.tracking.reid_interval
        return (
            self.frames_since_reid >= reid_interval
            or self.identity_status == IdentityStatus.PENDING
        )

    def copy_identity_to(self, face: DetectedFace) -> None:
        """Copia la identidad actual del track al DetectedFace."""
        face.track_id = self.track_id
        face.identity_status = self.identity_status
        face.user_id = self.user_id
        face.user_name = self.user_name
        face.identity_confidence = self.identity_confidence
        face.identity_distance = self.identity_distance
        face.spoof_status = self.spoof_status
        face.spoof_confidence = self.spoof_confidence
        face.bbox = self.smoothed_bbox  # Usar bbox suavizado


class FaceTracker:
    """
    Tracker IoU multi-objeto.

    El tracker desacopla la detección frame-a-frame del reconocimiento
    costoso, manteniendo la identidad de cada persona entre frames.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self._iou_thresh: float = cfg.tracking.iou_threshold
        self._max_missing: int = cfg.tracking.max_missing_frames
        self._smoothing: float = cfg.tracking.smoothing_alpha
        self._reid_interval: int = cfg.tracking.reid_interval
        self._enabled: bool = cfg.tracking.enabled

    def update(
        self,
        detected_faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """
        Actualiza el estado del tracker con nuevas detecciones.

        Args:
            detected_faces: Lista de rostros detectados en el frame actual.

        Returns:
            Lista de DetectedFace enriquecidos con track_id y
            marcados con needs_reid=True si deben re-identificarse.
        """
        if not self._enabled:
            # Sin tracking: asignar IDs temporales
            for i, face in enumerate(detected_faces):
                face.track_id = i
            return detected_faces

        # ── 1. Asociar detecciones a tracks existentes ──────────
        matched_det, matched_trk, unmatched_det, unmatched_trk = (
            self._associate(detected_faces, list(self._tracks.values()))
        )

        # ── 2. Actualizar tracks emparejados ─────────────────────
        for det_idx, trk_id in matched_det:
            face = detected_faces[det_idx]
            track = self._tracks[trk_id]

            track.update_bbox(face.bbox, self._smoothing)
            track.missing_frames = 0
            track.age += 1
            track.frames_since_reid += 1

            if face.embedding is not None:
                track.add_embedding(face.embedding)

            # Propagar identidad almacenada al face (sin re-identificar)
            track.copy_identity_to(face)

            # Señalizar si necesita re-identificación
            if track.needs_reid():
                face.identity_status = IdentityStatus.PENDING  # fuerza reid
                track.frames_since_reid = 0

        # ── 3. Crear tracks para detecciones nuevas ──────────────
        for det_idx in unmatched_det:
            face = detected_faces[det_idx]
            track = self._create_track(face)
            face.track_id = track.track_id
            face.identity_status = IdentityStatus.PENDING  # siempre reid en nuevo track

        # ── 4. Incrementar missing en tracks sin detección ───────
        for trk_id in unmatched_trk:
            self._tracks[trk_id].missing_frames += 1

        # ── 5. Eliminar tracks caducados ─────────────────────────
        self._prune_stale_tracks()

        return detected_faces

    def update_identity(
        self,
        track_id: int,
        status: IdentityStatus,
        user_id: Optional[str],
        user_name: Optional[str],
        confidence: float,
        distance: float,
    ) -> None:
        """
        Actualiza la identidad de un track después de re-identificación.
        Llamado por el Recognition Engine tras consultar FAISS.
        """
        if track_id in self._tracks:
            t = self._tracks[track_id]
            t.identity_status = status
            t.user_id = user_id
            t.user_name = user_name
            t.identity_confidence = confidence
            t.identity_distance = distance

    def update_spoof(
        self,
        track_id: int,
        spoof_status: SpoofStatus,
        spoof_confidence: float,
    ) -> None:
        """Actualiza el estado anti-spoofing de un track."""
        if track_id in self._tracks:
            t = self._tracks[track_id]
            t.spoof_status = spoof_status
            t.spoof_confidence = spoof_confidence

    def get_active_track_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    # ──────────────────────────────────────────────────────────
    # Asociación IoU (Hungarian simplificado greedy)
    # ──────────────────────────────────────────────────────────

    def _associate(
        self,
        detections: list[DetectedFace],
        tracks: list[Track],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
        """
        Asocia detecciones con tracks usando máximo IoU.

        Returns:
            matched_det: [(det_idx, track_id), ...]
            matched_trk: igual que matched_det
            unmatched_det: [det_idx, ...]  detecciones sin track
            unmatched_trk: [track_id, ...] tracks sin detección
        """
        if not tracks:
            return [], [], list(range(len(detections))), []

        if not detections:
            return [], [], [], [t.track_id for t in tracks]

        # Matriz IoU (dets x tracks)
        iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        for d_idx, face in enumerate(detections):
            for t_idx, track in enumerate(tracks):
                iou_matrix[d_idx, t_idx] = face.bbox.iou(track.bbox)

        matched_det: list[tuple[int, int]] = []
        matched_trk: list[tuple[int, int]] = []
        used_dets: set[int] = set()
        used_trks: set[int] = set()

        # Greedy: asignar el par de mayor IoU primero
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self._iou_thresh:
                break

            d_idx, t_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            trk_id = tracks[t_idx].track_id

            matched_det.append((int(d_idx), trk_id))
            matched_trk.append((int(d_idx), trk_id))
            used_dets.add(int(d_idx))
            used_trks.add(int(t_idx))

            # Marcar fila y columna usadas
            iou_matrix[int(d_idx), :] = -1
            iou_matrix[:, int(t_idx)] = -1

        unmatched_det = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trk = [
            tracks[i].track_id for i in range(len(tracks)) if i not in used_trks
        ]

        return matched_det, matched_trk, unmatched_det, unmatched_trk

    def _create_track(self, face: DetectedFace) -> Track:
        """Crea un nuevo Track para una detección sin match."""
        tid = self._next_id
        self._next_id += 1

        track = Track(
            track_id=tid,
            bbox=face.bbox,
            smoothed_bbox=BoundingBox(
                face.bbox.x1, face.bbox.y1,
                face.bbox.x2, face.bbox.y2,
                face.bbox.confidence,
            ),
        )

        if face.embedding is not None:
            track.add_embedding(face.embedding)

        self._tracks[tid] = track
        logger.debug(f"Nuevo track creado: id={tid}")
        return track

    def _prune_stale_tracks(self) -> None:
        """Elimina tracks que llevan demasiados frames sin detección."""
        to_delete = [
            tid for tid, t in self._tracks.items()
            if t.missing_frames > self._max_missing
        ]
        for tid in to_delete:
            del self._tracks[tid]
            logger.debug(f"Track eliminado por inactividad: id={tid}")
