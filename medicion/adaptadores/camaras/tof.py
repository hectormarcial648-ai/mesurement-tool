"""Adaptador de cámara: Arducam ToF."""
import cv2

import numpy as np


class CamaraToF:
    """Entrega fotogramas BGR agrandados a partir de la cámara ToF.

    modo "depth":     profundidad -> mapa de color en una ventana FIJA alrededor de la mesa
                      (los objetos más altos que la mesa cambian de color y se segmentan).
    modo "amplitude": imagen de amplitud/confianza en grises (se ve el tablero de ajedrez,
                      úsala para calibrar con h / k).
    La geometría es idéntica en ambos modos, así que la calibración sirve para los dos.
    """

    def __init__(self, rango_mm, up, span_mm, conf_min, modo, centro_mm=None):
        try:
            import ArducamDepthCamera as ac  # type: ignore
        except ImportError:
            raise SystemExit("Falta ArducamDepthCamera: pip install ArducamDepthCamera")
        self.ac = ac
        self.up = max(1, int(up))
        self.span = float(span_mm)
        self.conf_min = float(conf_min)
        self.modo = modo
        self.centro = centro_mm
        self.cam = ac.ArducamCamera()
        ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            raise RuntimeError(f"No se pudo abrir la cámara ToF (error {ret}). "
                               "Revisa cable CSI, fuente de 5V y reinicio tras instalar.")
        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            self.cam.close()
            raise RuntimeError(f"No se pudo iniciar la cámara ToF (error {ret})")
        self.cam.setControl(ac.Control.RANGE, int(rango_mm))
        info = self.cam.getCameraInfo()
        print(f"[tof] {info.width}x{info.height}, rango {int(rango_mm)} mm, imagen x{self.up}")
        if self.centro is None:  # "bloquear" la ventana de profundidad con la mesa a la vista
            vals = []
            for _ in range(15):
                d, c = self._leer_crudo()
                if d is not None:
                    m = (c >= self.conf_min) & np.isfinite(d) & (d > 0)
                    if m.any():
                        vals.append(float(np.median(d[m])))
            if not vals:
                raise RuntimeError("La cámara ToF no entrega mediciones válidas")
            self.centro = float(np.median(vals))
            print(f"[tof] Ventana de profundidad: {self.centro:.0f} mm ± {self.span:.0f} mm "
                  "(mesa al centro; cambia con --tof-centro-mm / --tof-span-mm)")

    def _leer_crudo(self):
        fr = self.cam.requestFrame(2000)
        if fr is None or not isinstance(fr, self.ac.DepthData):
            return None, None
        d = np.array(fr.depth_data, dtype=np.float32)
        c = np.array(fr.confidence_data, dtype=np.float32)
        self.cam.releaseFrame(fr)
        return d, c

    tipo = "tof"

    def descripcion(self):
        return "Cámara ToF Arducam"

    def info_captura(self):
        return f"up{self.up}"

    def alternar(self):
        self.modo = "amplitude" if self.modo == "depth" else "depth"
        return self.modo

    def read(self):
        d, c = self._leer_crudo()
        if d is None:
            return None
        if self.modo == "amplitude":
            top = max(float(np.percentile(c, 99)), 1.0)
            g = np.clip(c * (255.0 / top), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        else:
            valido = (c >= self.conf_min) & np.isfinite(d) & (d > 0)
            d = np.where(valido, d, self.centro).astype(np.float32)
            d = cv2.medianBlur(d, 3)
            t = np.clip((d - (self.centro - self.span)) / (2.0 * self.span), 0.0, 1.0)
            img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if self.up > 1:
            img = cv2.resize(img, None, fx=self.up, fy=self.up, interpolation=cv2.INTER_CUBIC)
        return img

    def close(self):
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
