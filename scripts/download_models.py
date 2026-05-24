"""
scripts/download_models.py
--------------------------
Descarga los modelos ONNX necesarios para el sistema.

Modelos descargados:
1. InsightFace buffalo_l (SCRFD + ArcFace) - se descarga automáticamente
   por InsightFace en el primer uso.

2. MiniFASNet anti-spoofing (desde el repositorio oficial):
   - 2.7_80x80_MiniFASNetV2.pth
   - 4_0_0_80x80_MiniFASNetV1SE.pth

Uso:
    python scripts/download_models.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────
# URLs de los modelos MiniFASNet (repositorio oficial GitHub)
# Fuente: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
# ──────────────────────────────────────────────────────────────
ANTISPOOF_MODELS = [
    {
        "filename": "2.7_80x80_MiniFASNetV2.pth",
        "url": (
            "https://github.com/minivision-ai/"
            "Silent-Face-Anti-Spoofing/raw/master/resources/"
            "anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
        ),
        "description": "MiniFASNetV2 - Modelo principal anti-spoofing",
    },
    {
        "filename": "4_0_0_80x80_MiniFASNetV1SE.pth",
        "url": (
            "https://github.com/minivision-ai/"
            "Silent-Face-Anti-Spoofing/raw/master/resources/"
            "anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
        ),
        "description": "MiniFASNetV1SE - Modelo secundario (ensemble)",
    },
]

# Directorio de destino
ANTISPOOF_DIR = PROJECT_ROOT / "models" / "antispoof"


def download_file(url: str, dest: Path, description: str) -> bool:
    """
    Descarga un archivo con barra de progreso.

    Args:
        url: URL de descarga.
        dest: Ruta de destino.
        description: Descripción para mostrar.

    Returns:
        True si la descarga fue exitosa.
    """
    print(f"\n[DESCARGANDO] {description}")
    print(f"  URL: {url}")
    print(f"  Destino: {dest}")

    if dest.exists():
        print(f"  ✓ Ya existe, omitiendo.")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, int(downloaded * 100 / total_size))
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                mb_done = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r  [{bar}] {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print(f"\n  ✓ Descargado exitosamente.")
        return True

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        if dest.exists():
            dest.unlink()

        print(f"\n  DESCARGA MANUAL:")
        print(f"  1. Abre: {url}")
        print(f"  2. Guarda como: {dest}")
        return False


def verify_insightface_models() -> None:
    """
    Verifica que InsightFace pueda descargar sus modelos automáticamente.
    InsightFace descarga buffalo_l la primera vez que se llama prepare().
    """
    print("\n[INSIGHTFACE] Verificando modelos buffalo_l...")
    try:
        from insightface.app import FaceAnalysis
        models_dir = str(PROJECT_ROOT / "models")
        app = FaceAnalysis(name="buffalo_l", root=models_dir, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_thresh=0.5, det_size=(320, 320))
        print("  ✓ Modelos InsightFace buffalo_l disponibles.")
    except ImportError:
        print("  ✗ InsightFace no instalado. Ejecuta: pip install insightface")
    except Exception as e:
        print(f"  ⚠ InsightFace se descargará automáticamente en el primer uso: {e}")


def main() -> None:
    print("=" * 60)
    print("FaceAuth - Descarga de Modelos")
    print("=" * 60)

    # 1. Modelos anti-spoofing
    print("\n--- Modelos Anti-Spoofing (MiniFASNet) ---")
    antispoof_ok = 0
    for model_info in ANTISPOOF_MODELS:
        dest = ANTISPOOF_DIR / model_info["filename"]
        ok = download_file(
            url=model_info["url"],
            dest=dest,
            description=model_info["description"],
        )
        if ok:
            antispoof_ok += 1

    # 2. Modelos InsightFace
    print("\n--- Modelos InsightFace (SCRFD + ArcFace) ---")
    verify_insightface_models()

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN:")
    print(f"  Anti-spoofing: {antispoof_ok}/{len(ANTISPOOF_MODELS)} modelos disponibles")

    if antispoof_ok == 0:
        print("\n  ⚠ Los modelos anti-spoofing deben descargarse manualmente.")
        print("  Visita: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing")
        print(f"  Coloca los archivos .onnx en: {ANTISPOOF_DIR}")
        print("\n  El sistema funcionará SIN anti-spoofing hasta que los descargues.")
    elif antispoof_ok < len(ANTISPOOF_MODELS):
        print("\n  ⚠ Funciona con un solo modelo (ensemble incompleto).")
    else:
        print("\n  ✓ Todos los modelos listos.")

    print("=" * 60)
    print("\nSistema listo para ejecutar:")
    print("  python main.py")


if __name__ == "__main__":
    main()
