"""Adaptador de segmentación CONMUTABLE: reparte entre el método clásico y una red neuronal que se enciende,
se apaga y se ajusta EN CALIENTE (desde la interfaz), sin reiniciar el programa.

Implementa PuertoSegmentador (lo que usa el núcleo) y PuertoSegmentadorConmutable (lo que usa la interfaz).
Si la red falla estando encendida, se apaga sola y la medición sigue con el método clásico.
"""
import contextlib
import importlib.util
from pathlib import Path

from medicion.adaptadores.segmentadores.descarga import U2NETP, descargar_modelo

CONFIG_BASE = {"modelo": None, "modo": "restringir", "umbral": 0.5, "margen_px": 6, "cada": 10, "tam": 320,
               "salida": 0, "media": (0.485, 0.456, 0.406), "desv": (0.229, 0.224, 0.225)}
EN_VIVO = ("modo", "umbral", "margen_px", "cada")          # se aplican sin recargar el modelo
VARIANTES = ("clasico", "restringir", "rescatar")


class SegmentadorConmutable:
    def __init__(self, clasico, fabrica_red, dir_modelos, config=None, modelo_confiable=None):
        self.clasico = clasico
        self.fabrica_red = fabrica_red            # fabrica_red(config con 'modelo' = ruta) -> segmentador de red
        self.dir_modelos = Path(dir_modelos)
        self.config = dict(CONFIG_BASE)
        self.config.update({k: v for k, v in (config or {}).items() if k in CONFIG_BASE and v is not None})
        self._confiable = str(modelo_confiable) if modelo_confiable else None  # ruta absoluta dada por línea de comandos
        self.red = None
        self.activa = False
        self.error = ""
        self._forzado = None

    # ----- PuertoSegmentador ------------------------------------------------------------------------- #
    @property
    def _actual(self):
        if self._forzado == "clasico":
            return self.clasico
        if self._forzado in ("restringir", "rescatar"):
            return self.red
        return self.red if (self.activa and self.red is not None) else self.clasico

    @property
    def nombre(self):
        return self._actual.nombre

    @property
    def ultimo_clasico(self):
        return self._actual.ultimo_clasico

    def segmentar(self, frame, bg_lab, ref, min_diff, thr, erode):
        seg = self._actual
        if seg is self.clasico:
            return seg.segmentar(frame, bg_lab, ref, min_diff, thr, erode)
        try:
            return seg.segmentar(frame, bg_lab, ref, min_diff, thr, erode)
        except Exception as e:  # noqa: BLE001
            if self._forzado is not None:
                raise                                  # en una comparación el error se muestra tal cual
            self.activa, self.error = False, f"La red falló y se apagó (se sigue con el método clásico): {e}"
            return self.clasico.segmentar(frame, bg_lab, ref, min_diff, thr, erode)

    # ----- PuertoSegmentadorConmutable ----------------------------------------------------------------- #
    def _ruta_modelo(self):
        nombre = self.config["modelo"]
        if not nombre:
            raise RuntimeError("Elige un modelo .onnx de la lista (o pulsa «Descargar u2netp»)")
        if self._confiable and Path(str(nombre)).name == Path(self._confiable).name:
            return Path(self._confiable)
        ruta = self.dir_modelos / Path(str(nombre)).name      # solo nombres de archivo dentro de la carpeta del programa
        if ruta.suffix.lower() != ".onnx" or not ruta.is_file():
            raise RuntimeError(f"No existe el modelo «{ruta.name}» en {self.dir_modelos}")
        return ruta

    def _construir_red(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.error = "Falta onnxruntime: instálalo en la Pi con  pip install onnxruntime --break-system-packages  y reinicia el programa"
            raise RuntimeError(self.error)
        try:
            self.red = self.fabrica_red(dict(self.config, modelo=str(self._ruta_modelo())))
        except Exception as e:  # noqa: BLE001
            self.red, self.activa = None, False
            self.error = str(e) or repr(e)
            raise RuntimeError(self.error) from e

    def encender(self):
        if self.red is None:
            self._construir_red()
        self.activa, self.error = True, ""

    def apagar(self):
        self.activa = False

    def configurar(self, **kw):
        """Cambia ajustes. modelo/tam/salida/media/desv recargan la red; modo/umbral/margen_px/cada se aplican al instante."""
        nuevo = {}
        for k, v in kw.items():
            if k not in CONFIG_BASE or v in (None, ""):
                continue
            if k == "modo":
                if v not in ("restringir", "rescatar"):
                    raise ValueError("modo debe ser 'restringir' o 'rescatar'")
            elif k == "umbral":
                v = float(v)
                if not 0.05 <= v <= 0.95:
                    raise ValueError("el umbral debe estar entre 0.05 y 0.95")
            elif k in ("margen_px", "cada", "tam", "salida"):
                v = int(v)
                if (k == "margen_px" and not 0 <= v <= 60) or (k == "cada" and v < 1) or (k == "tam" and v < 32):
                    raise ValueError(f"valor fuera de rango para {k}")
            nuevo[k] = v
        if "modelo" in nuevo and nuevo["modelo"] != self.config["modelo"]:
            anterior = self.config["modelo"]
            self.config["modelo"] = nuevo["modelo"]
            try:
                self._ruta_modelo()                    # un modelo inexistente no cambia nada
            except RuntimeError:
                self.config["modelo"] = anterior
                raise
            self.config["modelo"] = anterior
        recargar = any(k not in EN_VIVO and nuevo[k] != self.config[k] for k in nuevo)
        self.config.update(nuevo)
        if self.red is not None:
            for k in EN_VIVO:
                if k in nuevo:
                    setattr(self.red, k, nuevo[k])
            if recargar:
                self.red = None
                if self.activa:
                    self.encender()

    def descargar_modelo(self):
        ruta = descargar_modelo(self.dir_modelos, U2NETP)
        if not self.config["modelo"]:
            self.config["modelo"] = ruta.name
        return ruta

    @contextlib.contextmanager
    def forzar(self, variante):
        """Usa temporalmente 'clasico', 'restringir' o 'rescatar' (para comparar), sin cambiar si la red está encendida."""
        if variante not in VARIANTES:
            raise ValueError(f"variante desconocida: {variante}")
        previo, modo_previo = self._forzado, None
        if variante != "clasico":
            if self.red is None:
                self._construir_red()
            modo_previo, self.red.modo = self.red.modo, variante
            self.red.reiniciar_cache()
        self._forzado = variante
        try:
            yield self
        finally:
            self._forzado = previo
            if modo_previo is not None:
                self.red.modo = modo_previo

    def estado(self):
        modelos = sorted(p.name for p in self.dir_modelos.glob("*.onnx")) if self.dir_modelos.is_dir() else []
        if self._confiable and Path(self._confiable).name not in modelos:
            modelos.append(Path(self._confiable).name)
        return {"onnxruntime": importlib.util.find_spec("onnxruntime") is not None, "activa": bool(self.activa),
                "modelo": Path(str(self.config["modelo"])).name if self.config["modelo"] else None, "modelos": modelos,
                "modo": self.config["modo"], "umbral": self.config["umbral"], "margen_px": self.config["margen_px"],
                "cada": self.config["cada"], "error": self.error,
                "ultimo_ms": getattr(self.red, "ultimo_ms", None), "llamadas": getattr(self.red, "llamadas_red", 0),
                "u2netp_descargado": (self.dir_modelos / U2NETP["nombre"]).is_file()}
