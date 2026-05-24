"""
core/anti_spoof.py
------------------
Módulo anti-spoofing usando MiniFASNet vía ONNX Runtime.

MiniFASNet (Silent-Face-Anti-Spoofing) detecta ataques 2D:
- Fotos impresas
- Pantallas/videos
- Máscaras planas

Limitaciones conocidas (aceptables para MVP):
- No detecta máscaras 3D de alta calidad
- No detecta deepfakes en tiempo real
- Requiere iluminación razonable

Arquitectura del análisis:
- Evaluamos N frames consecutivos
- Solo marcamos SPOOF si K frames consecutivos son spoof
- Solo marcamos REAL si M frames consecutivos son reales
- Esto reduce falsos positivos por frames ruidosos
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.loader import cfg, ConfigLoader
from core.types import DetectedFace, SpoofStatus
from utils.logger import get_logger

logger = get_logger(__name__)

# Tamaño de entrada para MiniFASNet
_INPUT_SIZE = (80, 80)

# Índice de clase "real" en la salida del modelo
# MiniFASNet: clase 1 = real, clase 0 = spoof
_REAL_CLASS_IDX = 1


class SpoofResult:
    """Resultado intermedio del análisis anti-spoof para un track."""

    def __init__(self, window_size: int = 10):
        self._scores: deque[float] = deque(maxlen=window_size)
        self._consecutive_real: int = 0
        self._consecutive_spoof: int = 0

    def add_score(self, real_prob: float) -> None:
        self._scores.append(real_prob)
        threshold = cfg.anti_spoofing.threshold
        if real_prob >= threshold:
            self._consecutive_real += 1
            self._consecutive_spoof = 0
        else:
            self._consecutive_spoof += 1
            self._consecutive_real = 0

    @property
    def mean_score(self) -> float:
        if not self._scores:
            return 0.5
        return float(np.mean(list(self._scores)))

    @property
    def consecutive_real(self) -> int:
        return self._consecutive_real

    @property
    def consecutive_spoof(self) -> int:
        return self._consecutive_spoof

    @property
    def sample_count(self) -> int:
        return len(self._scores)

    def reset(self) -> None:
        self._scores.clear()
        self._consecutive_real = 0
        self._consecutive_spoof = 0


class AntiSpoofing:
    """
    Detector de spoofing usando dos modelos MiniFASNet en ensemble.
    Usar dos modelos diferentes reduce significativamente los falsos positivos.
    """

    def __init__(self) -> None:
        self._models: list[object] = []          # ONNX InferenceSession(s)
        self._initialized = False
        self._enabled = cfg.anti_spoofing.enabled
        self._threshold = cfg.anti_spoofing.threshold
        self._real_frames_req = cfg.anti_spoofing.real_frames_required
        self._spoof_frames_req = cfg.anti_spoofing.spoof_frames_required

        # Estado por track_id
        self._track_results: dict[int, SpoofResult] = {}

    def initialize(self) -> bool:
        """
        Carga los modelos MiniFASNet ONNX.
        Intenta cargar ambos modelos; si solo uno está disponible, usa ese.
        """
        if not self._enabled:
            logger.info("Anti-spoofing deshabilitado en configuración.")
            self._initialized = True
            return True

        if self._initialized:
            return True

        logger.info("Inicializando AntiSpoofing (MiniFASNet)...")

        root = ConfigLoader().project_root()
        model_paths = [
            root / cfg.anti_spoofing.model_path,
            root / cfg.anti_spoofing.model_path_alt,
        ]

        loaded = 0
        for path in model_paths:
            session = self._load_model(path)
            if session is not None:
                self._models.append(session)
                loaded += 1
                logger.info(f"  ✓ Modelo cargado: {path.name}")
            else:
                logger.warning(f"  ✗ Modelo no disponible: {path.name}")

        if loaded == 0:
            logger.error(
                "Ningún modelo anti-spoofing disponible. "
                "Ejecuta: python scripts/download_models.py\n"
                "El sistema continuará SIN protección anti-spoofing."
            )
            # No bloquear el sistema, pero marcar todo como PENDING
            self._initialized = True
            return False

        logger.info(f"AntiSpoofing listo con {loaded}/2 modelos.")
        self._initialized = True
        return True

    def _load_model(self, path: Path) -> Optional[object]:
        """Carga un modelo ONNX con el proveedor apropiado."""
        if not path.exists():
            return None
        try:
            import onnxruntime as ort

            providers = self._build_providers()
            session = ort.InferenceSession(str(path), providers=providers)
            logger.debug(f"Modelo cargado: {path}, providers: {session.get_providers()}")
            return session
        except Exception as e:
            logger.warning(f"No se pudo cargar {path}: {e}")
            return None

    def _build_providers(self) -> list:
        """Proveedores ONNX para anti-spoofing (prioridad CPU para modelos pequeños)."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            available = []

        providers = []

        # Para modelos tan pequeños como MiniFASNet (80x80)
        # CPU es generalmente más rápido que el overhead de OpenVINO
        if "OpenVINOExecutionProvider" in available:
            providers.append((
                "OpenVINOExecutionProvider",
                {"device_type": "CPU", "precision": "FP32"},
            ))

        providers.append((
            "CPUExecutionProvider",
            {"intra_op_num_threads": 2},
        ))
        return providers

    def analyze(self, face: DetectedFace) -> DetectedFace:
        """
        Analiza un DetectedFace y actualiza su spoof_status.

        Args:
            face: DetectedFace con face_crop disponible.

        Returns:
            El mismo DetectedFace con spoof_status y spoof_confidence actualizados.
        """
        if not self._initialized:
            face.spoof_status = SpoofStatus.PENDING
            return face

        if not self._enabled or not self._models:
            face.spoof_status = SpoofStatus.DISABLED
            face.spoof_confidence = 1.0
            return face

        if face.face_crop is None or face.face_crop.size == 0:
            face.spoof_status = SpoofStatus.PENDING
            return face

        try:
            # Obtener/crear estado para este track
            track_id = face.track_id
            if track_id not in self._track_results:
                self._track_results[track_id] = SpoofResult()

            result = self._track_results[track_id]

            # Inferencia con ensemble de modelos
            real_prob = self._predict_ensemble(face.face_crop)
            result.add_score(real_prob)

            face.spoof_confidence = result.mean_score

            # Determinar estado basado en frames consecutivos
            if result.sample_count < 2:
                face.spoof_status = SpoofStatus.PENDING
            elif result.consecutive_spoof >= self._spoof_frames_req:
                face.spoof_status = SpoofStatus.SPOOF
                logger.warning(f"SPOOF detectado en track {track_id}, prob_real={real_prob:.3f}")
            elif result.consecutive_real >= self._real_frames_req:
                face.spoof_status = SpoofStatus.REAL
            else:
                face.spoof_status = SpoofStatus.PENDING

            return face

        except Exception as e:
            logger.debug(f"Error en anti-spoofing: {e}")
            face.spoof_status = SpoofStatus.PENDING
            face.spoof_confidence = 0.5
            return face

    def _predict_ensemble(self, crop: np.ndarray) -> float:
        """
        Ejecuta todos los modelos cargados y promedia las probabilidades.

        Returns:
            Probabilidad de que la imagen sea REAL (0.0 - 1.0).
        """
        probs: list[float] = []
        for session in self._models:
            prob = self._predict_single(session, crop)
            if prob is not None:
                probs.append(prob)

        if not probs:
            return 0.5

        return float(np.mean(probs))

    def _predict_single(self, session: object, crop: np.ndarray) -> Optional[float]:
        """
        Inferencia con un modelo MiniFASNet ONNX.

        MiniFASNet espera:
        - Input: (1, 3, 80, 80) float32, normalizado [-1, 1] o [0, 1]
        - Output: (1, 3) — 3 clases: spoof/real/background

        Returns:
            Probabilidad de clase "real" (índice 1).
        """
        try:
            # Preprocesamiento
            blob = self._preprocess(crop)

            # Inferencia
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: blob})

            # outputs[0] shape: (1, 3) — softmax sobre 3 clases
            logits = outputs[0][0]  # (3,)

            # Softmax manual (el modelo no siempre incluye softmax)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Clase 1 = real
            real_prob = float(probs[_REAL_CLASS_IDX])
            return real_prob

        except Exception as e:
            logger.debug(f"Error en predicción single model: {e}")
            return None

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """
        Preprocesa el crop para MiniFASNet.
        - Resize a 80x80
        - BGR → normalización
        - HWC → NCHW
        - Tipo float32
        """
        # Resize
        resized = cv2.resize(crop, _INPUT_SIZE)

        # Normalización a [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Transposición HWC → CHW y añadir batch dim
        chw = np.transpose(normalized, (2, 0, 1))  # (3, 80, 80)
        blob = np.expand_dims(chw, axis=0)          # (1, 3, 80, 80)

        return blob

    def reset_track(self, track_id: int) -> None:
        """Limpia el historial de un track (cuando desaparece del frame)."""
        self._track_results.pop(track_id, None)

    def cleanup_stale_tracks(self, active_track_ids: set[int]) -> None:
        """Elimina estados de tracks que ya no están activos."""
        stale = set(self._track_results.keys()) - active_track_ids
        for tid in stale:
            del self._track_results[tid]

    @property
    def is_ready(self) -> bool:
        return self._initialized
