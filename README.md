# FaceAuth MVP
### Sistema de Reconocimiento Facial Biométrico en Tiempo Real
**Stack:** Python · InsightFace · SCRFD · ArcFace · FAISS · OpenVINO · PyQt6

---

## Estructura del Proyecto

```
faceauth/
├── main.py                         # Punto de entrada
├── requirements.txt
├── config/
│   ├── settings.yaml               # Toda la configuración (umbrales, cámara, etc.)
│   └── loader.py                   # Carga y acceso tipado a configuración
├── core/
│   ├── types.py                    # Dataclasses compartidos (DetectedFace, etc.)
│   ├── camera.py                   # Captura webcam en hilo dedicado
│   ├── detector.py                 # Detección facial SCRFD (InsightFace)
│   ├── anti_spoof.py               # Anti-spoofing MiniFASNet
│   ├── tracker.py                  # Tracker IoU multi-rostro
│   ├── database.py                 # Base de datos FAISS + persistencia
│   └── recognition_engine.py      # Orquestador del pipeline completo
├── ui/
│   ├── main_window.py              # Ventana principal PyQt6
│   ├── video_worker.py             # Worker de video en QThread
│   ├── registration_dialog.py     # Diálogo de registro de usuarios
│   └── user_card.py                # Widget tarjeta de usuario
├── utils/
│   ├── logger.py                   # Logger centralizado con colores
│   └── renderer.py                 # Renderizador de anotaciones OpenCV
├── scripts/
│   └── download_models.py          # Descarga modelos ONNX
├── models/
│   └── antispoof/                  # Modelos MiniFASNet .onnx
└── data/
    ├── embeddings/                 # FAISS index + metadata.pkl
    └── faces/                      # Imágenes de referencia
```

---

## Instalación Paso a Paso (Windows 10/11)

### Requisitos previos
- Python 3.10 o 3.11 (recomendado)
- Git (para clonar)
- Webcam USB o integrada

### 1. Clonar / descomprimir el proyecto
```cmd
cd C:\Users\TuUsuario\Desktop
:: Descomprime o clona aquí
cd faceauth
```

### 2. Crear entorno virtual
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias base
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Opcional pero recomendado) Habilitar OpenVINO para Intel Arc/Iris
```cmd
:: Desinstala onnxruntime estándar
pip uninstall onnxruntime -y

:: Instala onnxruntime con soporte OpenVINO
pip install onnxruntime-openvino==1.17.0
```

> **Nota:** También necesitas Intel OpenVINO Runtime instalado en el sistema.
> Descarga desde: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
> Instala la versión para Windows y ejecuta `setupvars.bat` antes de correr la app.

### 5. Descargar modelos
```cmd
python scripts/download_models.py
```

Los modelos InsightFace (SCRFD + ArcFace) se descargan automáticamente
en el primer uso. Los modelos anti-spoofing se descargan con el script.

### 6. Ejecutar
```cmd
python main.py
```

---

## Configuración (config/settings.yaml)

### Umbrales de reconocimiento (críticos)
```yaml
recognition:
  identity_threshold: 0.40  # < 0.40 = misma persona
  unknown_threshold: 0.55   # > 0.55 = definitivamente desconocido
```

**Cómo ajustar si hay falsos positivos:**
- Bajar `identity_threshold` a 0.35 → más estricto, menos falsos positivos
- Subir `identity_threshold` a 0.45 → más permisivo, mejor con accesorios

### Cámara
```yaml
camera:
  device_id: 0    # Cambiar a 1, 2... si tienes múltiples cámaras
  width: 1280
  height: 720
```

### Anti-spoofing
```yaml
anti_spoofing:
  enabled: true
  threshold: 0.85          # > 0.85 = real
  real_frames_required: 3  # Frames consecutivos reales para confirmar
```

---

## Uso del Sistema

### Primera vez: Registrar usuarios

1. Ejecuta `python main.py`
2. Clic en **"+ Registrar Usuario"**
3. Introduce nombre e ID
4. Elige **Webcam** o **Cargar Imágenes**
5. Captura al menos 8 fotos (15 recomendadas)
6. Clic en **Registrar**

### Reconocimiento en tiempo real

El sistema detecta automáticamente cuando una persona registrada
aparece frente a la cámara y muestra:
- **Bounding box verde** + nombre + porcentaje de confianza
- **Indicador anti-spoofing** (✓ real / ✗ spoof)
- **FPS y latencia** en esquina superior

Las personas desconocidas se muestran con un bbox gris muy tenue
y **sin etiqueta** (ignoradas intencionalmente).

---

## Rendimiento Esperado

| Hardware | FPS UI | Latencia Inferencia |
|---|---|---|
| Intel Arc A770 (OpenVINO) | 25-30 | ~80ms |
| Intel Iris Xe (OpenVINO) | 15-25 | ~150ms |
| CPU solo (i7 8-core) | 8-15 | ~200-400ms |

---

## Troubleshooting

**"No se pudo abrir cámara"**
→ Cambia `device_id` en settings.yaml a 1 o 2

**"Modelos InsightFace no disponibles"**
→ Asegúrate de tener internet en el primer uso (se descargan ~500MB)

**Falsos positivos (reconoce a desconocidos)**
→ Baja `identity_threshold` a 0.35 en settings.yaml

**El sistema no reconoce con gorra/gafas**
→ Registra fotos CON esos accesorios puestos. La variedad en el
  registro mejora significativamente el reconocimiento con oclusiones.

**Anti-spoofing marca real como spoof**
→ Sube `threshold` a 0.75 o `real_frames_required` a 2 en settings.yaml

---

## Arquitectura del Pipeline

```
Webcam
  │
  ▼
CameraCapture (thread) ──► frame BGR
  │
  ▼
FaceDetector (SCRFD)
  │  bbox + landmarks + embedding ArcFace
  ▼
FaceTracker (IoU)
  │  track_id asignado, cache de identidad
  ▼
AntiSpoofing (MiniFASNet ensemble)
  │  spoof_status: REAL / SPOOF / PENDING
  ▼
FaceDatabase (FAISS FlatIP)          ← solo si PENDING
  │  IdentityStatus + user + confidence
  ▼
FrameRenderer (OpenCV)
  │  frame anotado
  ▼
PyQt6 UI (QThread → señales)
```
