"""
config/loader.py
----------------
Carga y valida la configuración central desde settings.yaml.
Provee acceso tipado a todos los parámetros del sistema.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Directorio raíz del proyecto (dos niveles arriba de este archivo)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class _DotDict(dict):
    """Dict con acceso por atributos: cfg.camera.fps"""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict):
                return _DotDict(val)
            return val
        except KeyError:
            raise AttributeError(f"Config key not found: '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str) -> Path:
        """Retorna la ruta absoluta de un campo de ruta."""
        return PROJECT_ROOT / self[key]


class ConfigLoader:
    """
    Singleton que carga settings.yaml una sola vez y expone
    el árbol de configuración como _DotDict anidados.
    """

    _instance: ConfigLoader | None = None
    _config: _DotDict | None = None

    def __new__(cls) -> ConfigLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._load()

    def _load(self) -> None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado: {CONFIG_PATH}\n"
                "Asegúrate de ejecutar desde el directorio raíz del proyecto."
            )
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError("settings.yaml debe ser un mapa YAML válido.")

        ConfigLoader._config = _DotDict(raw)
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """Crea directorios necesarios si no existen."""
        dirs_to_ensure = [
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "data" / "embeddings",
            PROJECT_ROOT / "data" / "faces",
            PROJECT_ROOT / "models" / "antispoof",
        ]
        for d in dirs_to_ensure:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def cfg(self) -> _DotDict:
        """Acceso al árbol de configuración completo."""
        if self._config is None:
            self._load()
        return self._config  # type: ignore

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Acceso seguro con ruta de claves.
        Ej: loader.get('recognition', 'identity_threshold', default=0.40)
        """
        node: Any = self._config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def project_root(self) -> Path:
        return PROJECT_ROOT


# Instancia global — importar desde aquí
config = ConfigLoader()
cfg = config.cfg
