"""
utils/logger.py
---------------
Logger centralizado con colores, rotación de archivos y
niveles configurables. Un solo punto de configuración
para todo el sistema.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


_INITIALIZED = False


def setup_logger(
    name: str = "faceauth",
    level: str = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configura y retorna el logger raíz del sistema.
    Llamar una sola vez desde main.py; el resto de módulos
    usan get_logger().

    Args:
        name: Nombre del logger raíz.
        level: Nivel de logging ("DEBUG", "INFO", etc.).
        log_file: Ruta al archivo de log. None = solo consola.
        max_bytes: Tamaño máximo por archivo de log.
        backup_count: Número de archivos de backup.

    Returns:
        Logger configurado.
    """
    global _INITIALIZED
    logger = logging.getLogger(name)

    if _INITIALIZED:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.handlers.clear()

    # ── Formato ──────────────────────────────────────────────
    fmt_str = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # ── Handler consola ───────────────────────────────────────
    if _HAS_COLORLOG:
        color_fmt = (
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)-20s%(reset)s | %(message)s"
        )
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                color_fmt,
                datefmt=date_fmt,
                log_colors={
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(fmt_str, datefmt=date_fmt)
        )

    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # ── Handler archivo (opcional) ────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=date_fmt))
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    logger.propagate = False
    _INITIALIZED = True
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Retorna un sub-logger con el nombre del módulo.
    Uso: logger = get_logger(__name__)

    Args:
        module_name: Típicamente __name__ del módulo llamador.
    """
    # Limpia el prefijo del paquete para nombres más cortos
    short_name = module_name.split(".")[-1] if "." in module_name else module_name
    return logging.getLogger(f"faceauth.{short_name}")
