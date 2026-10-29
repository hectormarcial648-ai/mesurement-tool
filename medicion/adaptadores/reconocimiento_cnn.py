"""Adaptador de reconocimiento con red neuronal: extrae una "huella" (embedding) de un parche usando cualquier ONNX
de imágenes (p. ej. MobileNetV3-Small). Se enciende y apaga en caliente, como el segmentador.

El modelo debe recibir una imagen [1,3,H,W] (RGB, normalización ImageNet). Se usa su primera salida aplanada y normalizada
como huella (si es un clasificador, serán sus 1000 logits: sirve igual para comparar parecidos).
"""
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from medicion.dominio.piezas import para_cnn

MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESV = np.array([0.229, 0.224, 0.225], np.float32)


class ExtractorCnnConmutable:
    """Implementa PuertoExtractorCnn."""

    def __init__(self, dir_modelos, excluir=("u2netp.onnx", "u2net.onnx")):
        self.dir_modelos = Path(dir_modelos)
        self.excluir = set(excluir)       # modelos que son de segmentación, no de huellas
        self.modelo = None
        self.activo = False
        self.error = ""
        self._sesion = None
        self._entrada = None
        self._tam = 224
        self.ultimo_ms = None

    def _modelos(self):
        if not self.dir_modelos.is_dir():
            return []
        return sorted(p.name for p in self.dir_modelos.glob("*.onnx") if p.name not in self.excluir)

    def estado(self):
        return {"onnxruntime": importlib.util.find_spec("onnxruntime") is not None, "activo": bool(self.activo),
                "modelo": self.modelo, "modelos": self._modelos(), "error": self.error, "ultimo_ms": self.ultimo_ms,
                "dim": getattr(self, "_dim", None)}

    def encender(self, modelo=None):
        if importlib.util.find_spec("onnxruntime") is None:
            self.error = "Falta onnxruntime: pip install onnxruntime --break-system-packages (y reinicia el programa)"
            raise RuntimeError(self.error)
        nombre = Path(str(modelo or self.modelo or "")).name
        if not nombre:
            self.error = "Elige un modelo .onnx de la lista (cópialo a la carpeta del programa)"
            raise RuntimeError(self.error)
        ruta = self.dir_modelos / nombre
        if ruta.suffix.lower() != ".onnx" or not ruta.is_file():
            self.error = f"No existe el modelo «{nombre}» en {self.dir_modelos}"
            raise RuntimeError(self.error)
        if nombre != self.modelo or self._sesion is None:
            try:
                import onnxruntime as ort
                opciones = ort.SessionOptions()
                opciones.intra_op_num_threads = 2
                sesion = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
                ent = sesion.get_inputs()[0]
                forma = ent.shape
                if len(forma) != 4:
                    raise RuntimeError(f"la entrada del modelo no es una imagen [1,3,H,W] (forma {forma})")
                self._tam = int(forma[2]) if isinstance(forma[2], int) and forma[2] > 0 else 224
                self._sesion, self._entrada, self.modelo = sesion, ent.name, nombre
                self._dim = int(self.extraer(np.zeros((self._tam, self._tam, 3), np.uint8), _prueba=True).size)
            except Exception as e:  # noqa: BLE001
                self._sesion, self.activo = None, False
                self.error = str(e) or repr(e)
                raise RuntimeError(self.error) from e
        self.activo, self.error = True, ""

    def apagar(self):
        self.activo = False

    def extraer(self, parche_bgr, _prueba=False):
        import time
        if self._sesion is None:
            raise RuntimeError("La red de huellas no está cargada")
        img = para_cnn(parche_bgr, self._tam)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEDIA) / DESV
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        t0 = time.perf_counter()
        v = self._sesion.run(None, {self._entrada: x})[0].astype(np.float32).ravel()
        if not _prueba:
            self.ultimo_ms = (time.perf_counter() - t0) * 1000.0
        return v / (np.linalg.norm(v) + 1e-9)
