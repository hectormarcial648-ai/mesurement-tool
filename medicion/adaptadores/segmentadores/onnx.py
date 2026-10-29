"""Adaptador de segmentación con red neuronal (ONNX Runtime): la red PROPONE dónde está la pieza y
el método clásico AFINA el borde (subpíxel).

Por qué así: una red de segmentación entrega máscaras a baja resolución (p. ej. 320x320) cuyo borde tiene
incertidumbre de varios píxeles; sirve para decidir QUÉ es pieza (descartar sombras, reflejos, manchas), no para
ubicar el borde con precisión de centésimas de mm. Por eso, en el modo 'restringir' la máscara final es la del
método clásico limitada a la zona que la red marca como pieza (el borde sigue siendo el del método clásico).

LIMITACIÓN IMPORTANTE (medida en pruebas): una red de "objeto principal" como U²-Net decide qué es lo más llamativo de la
imagen; con VARIAS piezas en la mesa puede descartar las pequeñas o menos llamativas (en las pruebas descartó una pieza
legítima de 16x10 mm). Úsala con una sola pieza a la vez y artefactos de luz, o entrena un modelo con TUS piezas.

Costo: la red se ejecuta solo cuando la escena cambia o cada `cada` llamadas; en una Pi 4B mide su tiempo antes de usarla.

Modelos: cualquier ONNX con entrada [1,3,N,N] y una salida que sea un mapa 2D de "probabilidad de pieza".
Por defecto se asume U²-Net / U²-Netp (entrada 320, normalización ImageNet, salida d0). Para un modelo propio
(p. ej. una U-Net entrenada con tus piezas) basta ajustar --seg-tam, --seg-salida, --seg-media y --seg-desv.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from medicion.dominio.constantes import DIF_ESCALA
from medicion.dominio.vision import calcular_mascara


class SegmentadorOnnx:
    """Implementa PuertoSegmentador. Modos:
       restringir : máscara clásica ∩ zona de la red (recomendado: precisión del método clásico, robustez de la red)
       rescatar   : igual, pero si el método clásico no encuentra nada y la red sí, usa la máscara de la red
                    (menos precisa, solo como último recurso; desactiva el refinado en alta resolución)
    """

    def __init__(self, modelo, tam=320, salida=0, umbral=0.5, margen_px=6, modo="restringir",
                 media=(0.485, 0.456, 0.406), desv=(0.229, 0.224, 0.225), hilos=2, min_px=200, cada=10, cambio=4.0):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("Falta onnxruntime: pip install onnxruntime --break-system-packages") from e
        if not modelo or not Path(modelo).is_file():
            raise RuntimeError(f"No existe el modelo ONNX: {modelo!r} (usa --seg-modelo ruta.onnx)")
        if modo not in ("restringir", "rescatar"):
            raise ValueError("modo debe ser 'restringir' o 'rescatar'")
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = int(hilos)
        self.sesion = ort.InferenceSession(str(modelo), opciones, providers=["CPUExecutionProvider"])
        self.entrada = self.sesion.get_inputs()[0].name
        self.tam, self.salida, self.umbral = int(tam), int(salida), float(umbral)
        self.margen_px, self.modo, self.min_px = int(margen_px), modo, int(min_px)
        self.media = np.array(media, np.float32)
        self.desv = np.array(desv, np.float32)
        self.nombre = f"red ONNX ({Path(modelo).name}, modo {modo})"
        self.ultimo_clasico = True
        self.ultimo_ms = 0.0
        self.ultima_prob = None
        self.cada, self.cambio = max(1, int(cada)), float(cambio)
        self.llamadas_red = 0      # cuántas veces se ejecutó la red (para medir su costo real)
        self._prob, self._firma_img, self._n = None, None, 0

    def reiniciar_cache(self):
        """Fuerza que la próxima llamada vuelva a ejecutar la red."""
        self._prob, self._firma_img, self._n = None, None, 0

    @staticmethod
    def _firma(frame):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)

    def _prob_cache(self, frame):
        """Reutiliza el mapa de la red mientras la escena no cambie (y como mucho `cada` llamadas seguidas)."""
        f = self._firma(frame)
        reusar = (self._prob is not None and self._prob.shape == frame.shape[:2] and self._n < self.cada
                  and float(np.mean(np.abs(f - self._firma_img))) < self.cambio)
        if reusar:
            self._n += 1
            return self._prob
        self._prob = self.probabilidad(frame)
        self.llamadas_red += 1
        self._firma_img, self._n = f, 1
        return self._prob

    def probabilidad(self, frame):
        """Mapa float32 [0,1] del tamaño de `frame`: probabilidad de que cada píxel sea pieza."""
        img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if rgb.shape[0] > self.tam else cv2.INTER_LINEAR
        x = cv2.resize(rgb, (self.tam, self.tam), interpolation=interp).astype(np.float32) / 255.0
        x = ((x - self.media) / self.desv).transpose(2, 0, 1)[None].astype(np.float32)
        t0 = time.perf_counter()
        salida = self.sesion.run(None, {self.entrada: x})[self.salida]
        self.ultimo_ms = (time.perf_counter() - t0) * 1000.0
        p = np.squeeze(salida).astype(np.float32)
        if p.ndim != 2:
            raise RuntimeError(f"La salida {self.salida} del modelo no es un mapa 2D (forma {salida.shape})")
        if p.min() < 0.0 or p.max() > 1.0:  # logits u otra escala: normalizar a [0,1]
            p = (p - p.min()) / (float(p.max() - p.min()) + 1e-8)
        return cv2.resize(p, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    def segmentar(self, frame, bg_lab, ref, min_diff, thr, erode):
        bw_c, thr_u, dist_c = calcular_mascara(frame, bg_lab, min_diff, thr, erode, ref=ref)
        prob = self._prob_cache(frame)
        self.ultima_prob = prob
        red = (prob >= self.umbral).astype(np.uint8) * 255
        zona = red
        if self.margen_px > 0:
            k = 2 * self.margen_px + 1
            zona = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bw = cv2.bitwise_and(bw_c, zona)
        self.ultimo_clasico = True
        if self.modo == "rescatar" and cv2.countNonZero(bw) < self.min_px and cv2.countNonZero(red) >= self.min_px:
            self.ultimo_clasico = False
            return red, thr_u, prob * 255.0  # el mapa de la red hace de "diferencia" para el borde
        return bw, thr_u, dist_c
