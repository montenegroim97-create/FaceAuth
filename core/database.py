"""
core/database.py
----------------
Base de datos biométrica usando FAISS para búsqueda vectorial.

Responsabilidades:
- Almacenar embeddings de usuarios registrados
- Búsqueda por similitud coseno con FAISS FlatIP
- Persistencia en disco (index + metadata)
- CRUD de usuarios
- Cálculo de threshold y confianza

FAISS FlatIP:
- "Flat" = búsqueda exhaustiva exacta (no aproximada)
- "IP" = Inner Product
- Con embeddings L2-normalizados, IP equivale a coseno
- Para ≤10 usuarios: instantáneo (<0.1ms)
- Para escalar a 1000+: cambiar a IVFFlat
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.loader import cfg, ConfigLoader
from core.types import IdentityStatus, RegisteredUser
from utils.logger import get_logger

logger = get_logger(__name__)

# Dimensión de embeddings ArcFace
_EMB_DIM = 512


class FaceDatabase:
    """
    Base de datos vectorial de identidades registradas.
    Usa FAISS FlatIP internamente con embeddings L2-normalizados.

    Thread-safety: NO es thread-safe. El acceso desde múltiples
    hilos debe ser serializado externamente (el pipeline usa un
    solo hilo de inferencia).
    """

    def __init__(self) -> None:
        self._index: Optional[object] = None      # faiss.IndexFlatIP
        self._users: dict[str, RegisteredUser] = {}
        # Mapeo posición FAISS → user_id
        self._idx_to_user: list[str] = []

        root = ConfigLoader().project_root()
        self._index_path = root / cfg.faiss.index_path
        self._meta_path = root / cfg.faiss.metadata_path

        self._identity_thresh = cfg.recognition.identity_threshold
        self._unknown_thresh = cfg.recognition.unknown_threshold
        self._initialized = False

    def initialize(self) -> bool:
        """
        Inicializa FAISS y carga datos persistidos si existen.

        Returns:
            True si la inicialización fue exitosa.
        """
        if self._initialized:
            return True

        logger.info("Inicializando FaceDatabase (FAISS FlatIP)...")

        try:
            import faiss

            self._index = faiss.IndexFlatIP(_EMB_DIM)

            # Intentar cargar datos previos
            if self._index_path.exists() and self._meta_path.exists():
                self._load_from_disk()
            else:
                logger.info("Base de datos vacía. Registra usuarios para comenzar.")

            self._initialized = True
            logger.info(f"FaceDatabase lista. Usuarios registrados: {len(self._users)}")
            return True

        except ImportError:
            logger.error("FAISS no instalado. Ejecuta: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Error inicializando FaceDatabase: {e}", exc_info=True)
            return False

    # ──────────────────────────────────────────────────────────
    # Búsqueda (inferencia en tiempo real)
    # ──────────────────────────────────────────────────────────

    def search(
        self,
        embedding: np.ndarray,
    ) -> tuple[IdentityStatus, Optional[RegisteredUser], float, float]:
        """
        Busca el usuario más cercano para un embedding dado.

        Args:
            embedding: Vector ArcFace L2-normalizado (512,).

        Returns:
            Tupla de (IdentityStatus, RegisteredUser|None, confidence, distance).
            - IdentityStatus: KNOWN / UNKNOWN / LOW_CONFIDENCE
            - RegisteredUser: usuario encontrado o None
            - confidence: 0.0 - 1.0
            - distance: distancia coseno (0.0 = idéntico)
        """
        if not self._initialized or self._index is None:
            return IdentityStatus.UNKNOWN, None, 0.0, 1.0

        if len(self._users) == 0:
            return IdentityStatus.UNKNOWN, None, 0.0, 1.0

        if embedding is None or embedding.shape[0] != _EMB_DIM:
            return IdentityStatus.UNKNOWN, None, 0.0, 1.0

        try:
            # Asegurar normalización
            emb = self._normalize(embedding.reshape(1, -1))
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)

            # FAISS búsqueda: retorna (distances, indices)
            # Con FlatIP + vectores normalizados: distance = cosine similarity
            # rango: [-1, 1] donde 1 = idéntico
            distances, indices = self._index.search(emb, k=min(1, len(self._users)))

            if indices[0][0] == -1:
                return IdentityStatus.UNKNOWN, None, 0.0, 1.0

            # Convertir similitud coseno → distancia coseno
            cosine_sim = float(distances[0][0])
            cosine_dist = 1.0 - cosine_sim  # [0, 2], 0 = idéntico

            user_id = self._idx_to_user[indices[0][0]]
            user = self._users.get(user_id)

            logger.debug(
                f"Búsqueda FAISS: user_id={user_id}, "
                f"cosine_sim={cosine_sim:.4f}, cosine_dist={cosine_dist:.4f}"
            )

            # Aplicar thresholds
            status, confidence = self._classify(cosine_dist)

            return status, user if status == IdentityStatus.KNOWN else None, confidence, cosine_dist

        except Exception as e:
            logger.error(f"Error en búsqueda FAISS: {e}", exc_info=True)
            return IdentityStatus.UNKNOWN, None, 0.0, 1.0

    def _classify(self, distance: float) -> tuple[IdentityStatus, float]:
        """
        Clasifica la distancia coseno en KNOWN / LOW_CONFIDENCE / UNKNOWN.

        Zonas:
          [0.0 - identity_thresh]  → KNOWN (alta confianza)
          [identity_thresh - unknown_thresh] → LOW_CONFIDENCE (zona gris)
          [unknown_thresh - 2.0]   → UNKNOWN (definitivamente diferente)
        """
        if distance <= self._identity_thresh:
            # Convertir distancia a confianza: 0 dist → 100% conf
            confidence = 1.0 - (distance / self._identity_thresh)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            return IdentityStatus.KNOWN, confidence

        elif distance <= self._unknown_thresh:
            return IdentityStatus.LOW_CONFIDENCE, 0.0

        else:
            return IdentityStatus.UNKNOWN, 0.0

    # ──────────────────────────────────────────────────────────
    # Registro de usuarios
    # ──────────────────────────────────────────────────────────

    def add_user(
        self,
        user_id: str,
        name: str,
        embeddings: list[np.ndarray],
        face_image_path: Optional[str] = None,
    ) -> bool:
        """
        Registra un nuevo usuario con múltiples embeddings.

        Args:
            user_id: ID único del usuario (e.g., "user_001").
            name: Nombre para mostrar.
            embeddings: Lista de embeddings ArcFace para este usuario.
            face_image_path: Ruta a imagen de referencia opcional.

        Returns:
            True si el registro fue exitoso.
        """
        if not self._initialized:
            logger.error("Base de datos no inicializada.")
            return False

        if len(embeddings) < cfg.recognition.min_embeddings_per_user:
            logger.warning(
                f"Se requieren al menos {cfg.recognition.min_embeddings_per_user} "
                f"embeddings. Recibidos: {len(embeddings)}"
            )
            return False

        try:
            # Calcular embedding agregado
            agg_method = cfg.recognition.aggregation
            stacked = np.stack([self._normalize(e) for e in embeddings], axis=0)

            if agg_method == "median":
                mean_emb = np.median(stacked, axis=0)
            else:
                mean_emb = np.mean(stacked, axis=0)

            # Re-normalizar el embedding promedio
            mean_emb = self._normalize(mean_emb)

            # Si el usuario ya existe, actualizar
            if user_id in self._users:
                self._remove_user_from_index(user_id)
                logger.info(f"Actualizando usuario existente: {user_id}")

            # Crear objeto RegisteredUser
            now = datetime.now().isoformat()
            user = RegisteredUser(
                user_id=user_id,
                name=name,
                embedding=mean_emb,
                num_samples=len(embeddings),
                created_at=now if user_id not in self._users else self._users.get(user_id, RegisteredUser(user_id, name, mean_emb)).created_at,
                updated_at=now,
                face_image_path=face_image_path,
            )

            # Añadir al índice FAISS
            self._index.add(mean_emb.reshape(1, -1).astype(np.float32))
            self._idx_to_user.append(user_id)
            self._users[user_id] = user

            # Persistir
            self.save()

            logger.info(
                f"Usuario registrado: {name} (id={user_id}, "
                f"muestras={len(embeddings)}, "
                f"total_usuarios={len(self._users)})"
            )
            return True

        except Exception as e:
            logger.error(f"Error registrando usuario {user_id}: {e}", exc_info=True)
            return False

    def append_embeddings(self, user_id: str, new_embeddings: list[np.ndarray]) -> bool:
        """
        Añade más embeddings a un usuario existente usando promedio ponderado.
        No requiere almacenar todos los embeddings individuales.
        """
        if not self._initialized:
            logger.error("Base de datos no inicializada.")
            return False

        user = self._users.get(user_id)
        if user is None:
            logger.warning(f"Usuario no encontrado: {user_id}")
            return False

        try:
            old_count = user.num_samples
            old_emb = user.embedding

            # Normalizar y promediar nuevos embeddings
            stacked = np.stack([self._normalize(e) for e in new_embeddings], axis=0)
            new_mean = np.mean(stacked, axis=0)
            new_mean = self._normalize(new_mean)

            # Promedio ponderado
            total = old_count + len(new_embeddings)
            updated_emb = (old_emb * old_count + new_mean * len(new_embeddings)) / total
            updated_emb = self._normalize(updated_emb)

            # Actualizar índice FAISS
            self._remove_user_from_index(user_id)
            self._index.add(updated_emb.reshape(1, -1).astype(np.float32))
            self._idx_to_user.append(user_id)

            # Actualizar objeto de usuario
            user.embedding = updated_emb
            user.num_samples = total
            user.updated_at = datetime.now().isoformat()

            self.save()

            logger.info(
                f"Embeddings añadidos para {user.name} (id={user_id}): "
                f"{old_count} → {total} muestras"
            )
            return True

        except Exception as e:
            logger.error(f"Error añadiendo embeddings a {user_id}: {e}", exc_info=True)
            return False

    def remove_user(self, user_id: str) -> bool:
        """
        Elimina un usuario de la base de datos.
        FAISS FlatIP no soporta eliminación directa, reconstruimos el índice.
        """
        if user_id not in self._users:
            logger.warning(f"Usuario no encontrado: {user_id}")
            return False

        try:
            import faiss
            self._remove_user_from_index(user_id)
            del self._users[user_id]
            self.save()
            logger.info(f"Usuario eliminado: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error eliminando usuario {user_id}: {e}", exc_info=True)
            return False

    def _remove_user_from_index(self, user_id: str) -> None:
        """Reconstruye el índice FAISS excluyendo al usuario dado."""
        import faiss

        remaining_users = {k: v for k, v in self._users.items() if k != user_id}
        new_index = faiss.IndexFlatIP(_EMB_DIM)
        new_idx_map: list[str] = []

        for uid, user in remaining_users.items():
            new_index.add(user.embedding.reshape(1, -1).astype(np.float32))
            new_idx_map.append(uid)

        self._index = new_index
        self._idx_to_user = new_idx_map

    # ──────────────────────────────────────────────────────────
    # Persistencia
    # ──────────────────────────────────────────────────────────

    def save(self) -> bool:
        """Guarda el índice FAISS y metadata en disco."""
        try:
            import faiss

            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))

            meta = {
                "users": self._users,
                "idx_to_user": self._idx_to_user,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self._meta_path, "wb") as f:
                pickle.dump(meta, f)

            logger.debug(f"Base de datos guardada. Usuarios: {len(self._users)}")
            return True

        except Exception as e:
            logger.error(f"Error guardando base de datos: {e}", exc_info=True)
            return False

    def _load_from_disk(self) -> None:
        """Carga el índice FAISS y metadata desde disco."""
        try:
            import faiss

            self._index = faiss.read_index(str(self._index_path))

            with open(self._meta_path, "rb") as f:
                meta = pickle.load(f)

            self._users = meta.get("users", {})
            self._idx_to_user = meta.get("idx_to_user", [])

            logger.info(
                f"Base de datos cargada desde disco. "
                f"Usuarios: {len(self._users)}, "
                f"Vectores FAISS: {self._index.ntotal}"
            )

        except Exception as e:
            logger.warning(f"No se pudo cargar base de datos: {e}. Iniciando vacía.")
            import faiss
            self._index = faiss.IndexFlatIP(_EMB_DIM)
            self._users = {}
            self._idx_to_user = []

    # ──────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """Normalización L2 para usar Inner Product como coseno."""
        emb = embedding.astype(np.float32)
        if emb.ndim == 1:
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-6 else emb
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        return emb / norms

    @property
    def users(self) -> dict[str, RegisteredUser]:
        return dict(self._users)

    @property
    def user_count(self) -> int:
        return len(self._users)

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def get_user(self, user_id: str) -> Optional[RegisteredUser]:
        return self._users.get(user_id)

    def user_exists(self, user_id: str) -> bool:
        return user_id in self._users
