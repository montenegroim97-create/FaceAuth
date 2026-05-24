"""
core/detector.py
----------------
Detección facial usando InsightFace con modelo SCRFD.
SCRFD (Sample and Computation Redistribution Face Detector)
es el detector más rápido y preciso del stack InsightFace.

Responsabilidades:
- Inicializar InsightFace con el proveedor de ejecución correcto
- Detectar rostros en cada frame
- Extraer bounding boxes, landmarks y scores
- Filtrar detecciones por tamaño mínimo
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.loader import cfg, ConfigLoader
from core.types import BoundingBox, DetectedFace, Landmark
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """
    Wrapper sobre InsightFace FaceAnalysis.

    InsightFace unifica detección (SCRFD) y reconocimiento (ArcFace)
    en un solo pipeline. Este módulo expone solo la detección.
    El reconocimiento se maneja en embedder.py.
    """

    def __init__(self) -> None:
        self._app: Optional[object] = None
        self._initialized = False
        self._det_size: tuple[int, int] = tuple(cfg.detection.det_size)  # type: ignore
        self._min_face_size: int = cfg.detection.min_face_size
        self._det_thresh: float = cfg.detection.det_thresh
        self._max_faces: int = cfg.detection.max_faces

    def initialize(self) -> bool:
        """
        Carga el modelo InsightFace. Llamar explícitamente antes
        de procesar frames para controlar el tiempo de inicio.

        Returns:
            True si la inicialización fue exitosa.
        """
        if self._initialized:
            return True

        logger.info("Inicializando FaceDetector (InsightFace SCRFD + ArcFace)...")
        t0 = time.perf_counter()

        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Determinar proveedores ONNX
            providers = self._build_providers()
            logger.info(f"Proveedores ONNX: {providers}")

            # FaceAnalysis carga SCRFD (detector) + ArcFace (embedder)
            # allowed_modules=['detection'] solo carga el detector aquí.
            # El embedder completo se necesita en embedder.py, pero
            # InsightFace comparte el mismo objeto app.
            # Usamos el modelo buffalo_l que incluye:
            #   - det_10g.onnx (SCRFD-10GF, el más preciso)
            #   - w600k_r50.onnx (ArcFace ResNet50)
            root_dir = str(ConfigLoader().project_root() / "models")

            self._app = FaceAnalysis(
                name=cfg.detection.model,
                root=root_dir,
                providers=providers,
            )
            self._app.prepare(
                ctx_id=0,
                det_thresh=self._det_thresh,
                det_size=self._det_size,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"FaceDetector listo en {elapsed:.0f}ms")
            self._initialized = True
            return True

        except ImportError as e:
            logger.error(f"InsightFace no instalado: {e}")
            logger.error("Ejecuta: pip install insightface")
            return False
        except Exception as e:
            logger.error(f"Error inicializando FaceDetector: {e}", exc_info=True)
            return False

    def _build_providers(self) -> list[str]:
        """
        Construye la lista de execution providers para ONNX Runtime.
        OpenVINO para Intel Arc/Iris, CPU como fallback.
        """
        provider = cfg.inference.execution_provider
        fallback = cfg.inference.fallback_provider

        available = self._get_available_providers()
        logger.debug(f"Proveedores ONNX disponibles: {available}")

        providers: list[str] = []

        if provider == "OpenVINO" and "OpenVINOExecutionProvider" in available:
            providers.append((
                "OpenVINOExecutionProvider",
                {
                    "device_type": "GPU",           # Intel Arc/Iris
                    "precision": "FP16",             # Más rápido en iGPU
                    "enable_opencl_throttling": False,
                },
            ))
            logger.info("OpenVINO GPU habilitado para Intel Arc/Iris")
        elif provider == "OpenVINO":
            logger.warning(
                "OpenVINOExecutionProvider no disponible. "
                "Instala: pip install onnxruntime-openvino"
            )

        # CPU siempre como fallback
        providers.append((
            "CPUExecutionProvider",
            {
                "intra_op_num_threads": cfg.inference.num_threads,
                "inter_op_num_threads": cfg.inference.inter_op_threads,
            },
        ))

        return providers

    @staticmethod
    def _get_available_providers() -> list[str]:
        """Lista los proveedores disponibles en el runtime actual."""
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Detecta todos los rostros en un frame BGR.

        Args:
            frame: Imagen BGR de OpenCV.

        Returns:
            Lista de DetectedFace con bbox, landmarks y embedding
            (InsightFace calcula embeddings en el mismo paso).
        """
        if not self._initialized:
            logger.warning("FaceDetector no inicializado. Llamar initialize() primero.")
            return []

        if frame is None or frame.size == 0:
            return []

        try:
            # InsightFace retorna lista de Face objects
            # Cada face tiene: bbox, kps (landmarks), det_score, embedding
            raw_faces = self._app.get(frame)

            detected: list[DetectedFace] = []
            for raw in raw_faces[:self._max_faces]:
                face = self._parse_face(raw, frame)
                if face is not None:
                    detected.append(face)

            return detected

        except Exception as e:
            logger.error(f"Error en detección facial: {e}", exc_info=True)
            return []

    def _parse_face(
        self,
        raw_face: object,
        frame: np.ndarray,
    ) -> Optional[DetectedFace]:
        """
        Convierte un InsightFace Face object en nuestro DetectedFace.

        Args:
            raw_face: Objeto Face de InsightFace.
            frame: Frame original (para extraer crop).

        Returns:
            DetectedFace o None si no pasa filtros de calidad.
        """
        try:
            # Bounding box [x1, y1, x2, y2]
            bbox_arr = raw_face.bbox.astype(int)
            bbox = BoundingBox(
                x1=max(0, bbox_arr[0]),
                y1=max(0, bbox_arr[1]),
                x2=min(frame.shape[1], bbox_arr[2]),
                y2=min(frame.shape[0], bbox_arr[3]),
                confidence=float(raw_face.det_score),
            )

            # Filtrar caras muy pequeñas
            if bbox.width < self._min_face_size or bbox.height < self._min_face_size:
                return None

            # Landmarks (5 puntos)
            landmark = None
            if hasattr(raw_face, "kps") and raw_face.kps is not None:
                landmark = Landmark(points=raw_face.kps.astype(np.float32))

            # Embedding ArcFace (calculado por InsightFace automáticamente)
            embedding = None
            if hasattr(raw_face, "embedding") and raw_face.embedding is not None:
                emb = raw_face.embedding.astype(np.float32)
                # Normalizar a norma unitaria (requisito cosine similarity)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                embedding = emb

            # Crop del rostro (para anti-spoofing)
            face_crop = self._extract_crop(frame, bbox)

            return DetectedFace(
                bbox=bbox,
                landmark=landmark,
                embedding=embedding,
                det_score=float(raw_face.det_score),
                face_crop=face_crop,
            )

        except Exception as e:
            logger.debug(f"Error parseando face: {e}")
            return None

    def _extract_crop(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        padding: float = 0.2,
    ) -> Optional[np.ndarray]:
        """
        Extrae y devuelve el crop del rostro con padding.
        El padding extra ayuda al anti-spoofing a ver contexto.
        """
        try:
            h, w = frame.shape[:2]
            pad_x = int(bbox.width * padding)
            pad_y = int(bbox.height * padding)

            x1 = max(0, bbox.x1 - pad_x)
            y1 = max(0, bbox.y1 - pad_y)
            x2 = min(w, bbox.x2 + pad_x)
            y2 = min(h, bbox.y2 + pad_y)

            return frame[y1:y2, x1:x2].copy()
        except Exception:
            return None

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def app(self) -> Optional[object]:
        """Expone el objeto FaceAnalysis de InsightFace (para embedder.py)."""
        return self._app
