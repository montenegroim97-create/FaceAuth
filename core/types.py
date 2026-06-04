"""
core/types.py
-------------
Dataclasses y tipos compartidos entre todos los módulos.
Un lugar único para las estructuras de datos del sistema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Enumeraciones
# ──────────────────────────────────────────────────────────────

class IdentityStatus(Enum):
    """Estado final de la identidad de un rostro detectado."""
    KNOWN = auto()          # Usuario registrado identificado
    UNKNOWN = auto()        # Persona desconocida
    LOW_CONFIDENCE = auto() # Zona gris, ignorar
    SPOOF = auto()          # Ataque de spoofing detectado
    PENDING = auto()        # Aún procesando


class SpoofStatus(Enum):
    """Resultado del análisis anti-spoofing."""
    REAL = auto()
    SPOOF = auto()
    PENDING = auto()        # No hay suficientes frames todavía
    DISABLED = auto()       # Anti-spoofing desactivado


# ──────────────────────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Bounding box de un rostro detectado."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union con otro bounding box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def scale(self, factor: float) -> BoundingBox:
        """Expande el bbox por un factor (útil para anti-spoof crop)."""
        cx, cy = self.center
        hw = int(self.width * factor / 2)
        hh = int(self.height * factor / 2)
        return BoundingBox(cx - hw, cy - hh, cx + hw, cy + hh, self.confidence)


@dataclass
class Landmark:
    """Puntos de referencia faciales (5 keypoints de ArcFace)."""
    points: np.ndarray  # shape (5, 2): ojo_izq, ojo_der, nariz, boca_izq, boca_der

    def __post_init__(self) -> None:
        if self.points.shape != (5, 2):
            raise ValueError(f"Se esperan 5 landmarks (x,y), got shape {self.points.shape}")


@dataclass
class DetectedFace:
    """
    Representa un rostro detectado en un frame.
    Es la unidad de datos que fluye por el pipeline.
    """
    bbox: BoundingBox
    landmark: Optional[Landmark] = None
    embedding: Optional[np.ndarray] = None      # ArcFace embedding (512,)
    det_score: float = 0.0                       # Score del detector SCRFD

    # --- Campos rellenados por etapas posteriores ---
    track_id: int = -1                           # ID del tracker
    identity_status: IdentityStatus = IdentityStatus.PENDING
    user_id: Optional[str] = None               # ID del usuario registrado
    user_name: Optional[str] = None             # Nombre para mostrar
    identity_confidence: float = 0.0            # 0.0 - 1.0
    identity_distance: float = 1.0              # Distancia coseno (menor = más similar)
    spoof_status: SpoofStatus = SpoofStatus.PENDING
    spoof_confidence: float = 0.0               # Probabilidad de ser REAL (0-1)

    # --- Crop del rostro (para anti-spoof y registro) ---
    face_crop: Optional[np.ndarray] = None      # BGR crop del rostro

    @property
    def is_identified(self) -> bool:
        return self.identity_status == IdentityStatus.KNOWN

    @property
    def is_unknown(self) -> bool:
        return self.identity_status == IdentityStatus.UNKNOWN

    @property
    def is_spoof(self) -> bool:
        return self.spoof_status == SpoofStatus.SPOOF

    @property
    def display_label(self) -> str:
        """Etiqueta a mostrar en la UI."""
        if self.is_spoof:
            return "⚠ SPOOF"
        if self.identity_status == IdentityStatus.KNOWN:
            pct = int(self.identity_confidence * 100)
            return f"{self.user_name} ({pct}%)"
        if self.identity_status == IdentityStatus.LOW_CONFIDENCE:
            return "..."
        return ""  # Desconocidos: sin etiqueta (ignorar visualmente)

    @property
    def display_color(self) -> tuple[int, int, int]:
        """Color BGR para bounding box."""
        if self.is_spoof:
            return (60, 60, 220)       # Rojo suave
        if self.identity_status == IdentityStatus.KNOWN:
            return (150, 212, 0)       # Verde teal
        if self.identity_status == IdentityStatus.LOW_CONFIDENCE:
            return (0, 180, 230)       # Naranja/ámbar
        return (60, 65, 75)            # Gris oscuro (desconocido)


@dataclass
class FrameResult:
    """Resultado completo del procesamiento de un frame."""
    frame_id: int
    faces: list[DetectedFace] = field(default_factory=list)
    processing_time_ms: float = 0.0
    fps: float = 0.0
    display_fps: float = 0.0
    error: Optional[str] = None

    @property
    def known_users(self) -> list[DetectedFace]:
        return [f for f in self.faces if f.is_identified]

    @property
    def unknown_faces(self) -> list[DetectedFace]:
        return [f for f in self.faces if f.is_unknown]


@dataclass
class RegisteredUser:
    """Usuario almacenado en la base de datos biométrica."""
    user_id: str
    name: str
    embedding: np.ndarray           # Embedding promedio normalizado (512,)
    num_samples: int = 0
    created_at: str = ""
    updated_at: str = ""
    face_image_path: Optional[str] = None  # Foto de referencia
