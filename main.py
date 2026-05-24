"""
main.py
-------
Punto de entrada de FaceAuth MVP.

Secuencia de arranque:
1. Configurar logging
2. Inicializar RecognitionEngine (carga modelos)
3. Crear aplicación PyQt6
4. Mostrar MainWindow
5. Ejecutar event loop
"""
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que el proyecto esté en el path antes de cualquier import
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    # ── 1. Logging ────────────────────────────────────────────
    from config.loader import cfg, ConfigLoader
    from utils.logger import setup_logger

    log_file = ConfigLoader().project_root() / cfg.system.log_file
    logger = setup_logger(
        name="faceauth",
        level=cfg.system.log_level,
        log_file=log_file,
    )

    logger.info("=" * 60)
    logger.info(f"FaceAuth MVP v{cfg.system.version} arrancando...")
    logger.info(f"Python: {sys.version}")
    logger.info("=" * 60)

    # ── 2. Verificar dependencias críticas ────────────────────
    missing = _check_dependencies()
    if missing:
        logger.critical(f"Dependencias faltantes: {missing}")
        logger.critical("Ejecuta: pip install -r requirements.txt")
        return 1

    # ── 3. Inicializar Engine ─────────────────────────────────
    from core.recognition_engine import RecognitionEngine

    engine = RecognitionEngine()
    if not engine.initialize():
        logger.critical("RecognitionEngine falló al inicializar. Abortando.")
        return 1

    # ── 4. Crear aplicación PyQt6 ─────────────────────────────
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        app = QApplication(sys.argv)
        app.setApplicationName(cfg.system.app_name)
        app.setApplicationVersion(cfg.system.version)

        # PyQt6 >= 6.7 eliminó AA_UseHighDpiPixmaps (alta resolución es automática)
        _hi_dpi_attr = getattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps", None)
        if _hi_dpi_attr is not None:
            app.setAttribute(_hi_dpi_attr, True)

    except ImportError as e:
        logger.critical(f"PyQt6 no disponible: {e}")
        logger.critical("Ejecuta: pip install PyQt6")
        return 1

    # ── 5. Ventana principal ──────────────────────────────────
    from ui.main_window import MainWindow

    window = MainWindow(engine)
    window.show()

    logger.info("UI lista. Iniciando event loop.")

    # ── 6. Event loop ─────────────────────────────────────────
    exit_code = app.exec()
    logger.info(f"FaceAuth cerrado. Exit code: {exit_code}")
    return exit_code


def _check_dependencies() -> list[str]:
    """Verifica que las dependencias críticas estén instaladas."""
    missing: list[str] = []

    checks = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "insightface": "insightface",
        "onnxruntime": "onnxruntime",
        "faiss": "faiss-cpu",
        "PyQt6": "PyQt6",
        "yaml": "pyyaml",
    }

    for module, package in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    return missing


if __name__ == "__main__":
    sys.exit(main())
