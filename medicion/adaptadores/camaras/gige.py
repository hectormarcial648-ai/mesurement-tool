"""Adaptador de cámara: GigE Vision / PoE (Aravis)."""
import gc
import time

import numpy as np


def _alinear(v, paso=16):
    return max(paso, (int(v) // paso) * paso)


def _centrar(total, tam, paso=16):
    """Desplazamiento (alineado) para centrar un recorte `tam` en un sensor de `total` px."""
    return max(0, ((int(total) - int(tam)) // 2) // paso * paso)


class CamaraGigE:
    """Cámara GigE Vision con Aravis (configuración tomada de camara_vivo.py, que ya te funciona).

    Entrega arreglos 2-D uint8 (Mono8) del ROI centrado. Exposición y ganancia quedan FIJAS
    (necesario para que las medidas no cambien entre fotogramas).
    """

    def __init__(self, ip=None, roi_size=1280, roi_h=None, exposure=8000.0, gain=None,
                 decimation=1, buffers=8):
        try:
            import gi  # type: ignore
            gi.require_version("Aravis", "0.8")
            from gi.repository import Aravis  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Falta Aravis para Python ({e}).\n"
                             "  sudo apt install gir1.2-aravis-0.8 python3-gi\n"
                             "  (y no uses un venv sin --system-site-packages)")
        self.Aravis = Aravis
        Aravis.update_device_list()
        try:
            self.cam = Aravis.Camera.new(ip)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"No se pudo abrir la cámara GigE: {e}\n"
                             "Revisa que eth0 esté en 192.168.10.x y que arv-tool-0.8 la liste.")
        print(f"[gige] Conectado a: {self.cam.get_model_name()} (SN {self.cam.get_device_serial_number()})")
        self._buffers = buffers
        self.sensor = (5120, 5120)
        self._configurar(roi_size, roi_h, exposure, gain, decimation)
        self._abrir_flujo()

    def _abrir_flujo(self):
        """Crea el flujo con buffers del tamaño actual del ROI e inicia la adquisición."""
        Aravis = self.Aravis
        self.stream = self.cam.create_stream(None, None)
        payload = self.cam.get_payload()
        for _ in range(max(2, int(self._buffers))):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.cam.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
        self.cam.start_acquisition()
        self._sin_frame = 0

    def set_roi(self, roi_size, roi_h=None):
        """Cambia el recorte del sensor EN CALIENTE: detiene, reconfigura y reinicia la adquisición."""
        try:
            self.cam.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        self.stream = None  # soltar el flujo anterior: sus buffers tienen el tamaño viejo
        w_max, h_max = self.sensor
        if roi_size and roi_size > 0:
            w = min(_alinear(roi_size), w_max)
            h = min(_alinear(roi_h if roi_h else roi_size), h_max)
        else:
            w, h = w_max, h_max
        err = None
        try:
            self.cam.set_region(0, 0, w, h)
            self.cam.set_region(_centrar(w_max, w), _centrar(h_max, h), w, h)
        except Exception as e:  # noqa: BLE001
            err = e
        self._abrir_flujo()  # siempre dejar la cámara funcionando
        if err is not None:
            raise err
        x, y, w, h = self.cam.get_region()
        print(f"[gige] ROI: {w}x{h} en ({x},{y}) | {self.cam.get_payload() / 1e6:.2f} MB por fotograma")

    def origen_roi(self):
        x, y, _, _ = self.cam.get_region()
        return (x, y)

    def _configurar(self, roi_size, roi_h, exposure, gain, decimation):
        cam, Aravis = self.cam, self.Aravis
        dev = cam.get_device()
        try:
            cam.set_pixel_format_from_string("Mono8")
        except Exception as e:  # noqa: BLE001
            print(f"[gige] Aviso: no se pudo fijar Mono8: {e}")
        if decimation > 1:
            try:
                dev.set_integer_feature_value("DecimationHorizontal", int(decimation))
                dev.set_integer_feature_value("DecimationVertical", int(decimation))
                print(f"[gige] Decimation {decimation}x aplicado")
            except Exception as e:  # noqa: BLE001
                print(f"[gige] Aviso: decimation no disponible: {e}")
        try:
            w_max, h_max = cam.get_sensor_size()
        except Exception:  # noqa: BLE001
            w_max, h_max = 5120, 5120
        self.sensor = (w_max, h_max)
        if roi_size and roi_size > 0:
            w = min(_alinear(roi_size), w_max)
            h = min(_alinear(roi_h if roi_h else roi_size), h_max)
        else:
            w, h = w_max, h_max
        try:
            cam.set_region(0, 0, w, h)
            cam.set_region(_centrar(w_max, w), _centrar(h_max, h), w, h)
        except Exception as e:  # noqa: BLE001
            print(f"[gige] Aviso: no se pudo fijar el ROI: {e}")
        try:
            cam.set_exposure_time_auto(Aravis.Auto.OFF)
        except Exception:  # noqa: BLE001
            pass
        try:
            cam.set_exposure_time(float(exposure))
        except Exception as e:  # noqa: BLE001
            print(f"[gige] Aviso: exposición: {e}")
        if gain is not None:
            try:
                cam.set_gain_auto(Aravis.Auto.OFF)
            except Exception:  # noqa: BLE001
                pass
            try:
                cam.set_gain(float(gain))
            except Exception as e:  # noqa: BLE001
                print(f"[gige] Aviso: ganancia: {e}")
        try:
            cam.gv_auto_packet_size()
        except Exception as e:  # noqa: BLE001
            print(f"[gige] Aviso: packet size: {e}")
        x, y, w, h = cam.get_region()
        print(f"[gige] ROI: {w}x{h} en ({x},{y}) | exposición {cam.get_exposure_time():.0f} us | "
              f"ganancia {cam.get_gain():.1f} dB | {cam.get_payload() / 1e6:.2f} MB por fotograma")
        if w * h > 4_000_000:
            print(f"[gige] AVISO: {w * h / 1e6:.1f} MP por fotograma. Para procesarlo con fluidez en la Pi usa "
                  "--proc-scale 2/4 (procesa reducido y refina bordes a resolución completa) o baja --roi-size.")

    def read(self):
        buf = self.stream.timeout_pop_buffer(2000000)
        if buf is None:
            self._sin_frame += 1
            if self._sin_frame % 3 == 0:
                print("[gige] Esperando fotogramas... (revisa el cable PoE, la IP en 192.168.10.x y el packet size)")
            return None
        self._sin_frame = 0
        img = None
        try:
            if buf.get_status() == self.Aravis.BufferStatus.SUCCESS:
                w, h = buf.get_image_width(), buf.get_image_height()
                img = np.frombuffer(buf.get_data(), dtype=np.uint8, count=w * h).reshape((h, w)).copy()
        finally:
            self.stream.push_buffer(buf)  # devolver SIEMPRE el buffer al flujo
        return img

    tipo = "gige"

    def descripcion(self):
        try:
            return f"{self.cam.get_vendor_name()} {self.cam.get_model_name()} (PoE)"
        except Exception:  # noqa: BLE001
            return "Cámara PoE GigE"

    def set_exposicion(self, us):
        self.cam.set_exposure_time_auto(self.Aravis.Auto.OFF)
        self.cam.set_exposure_time(float(us))

    def set_ganancia(self, db):
        self.cam.set_gain_auto(self.Aravis.Auto.OFF)
        self.cam.set_gain(float(db))

    def geometria_sensor(self):
        """(ox, oy, k): px del sensor = ox + (px de imagen + 0.5) * k - 0.5 (sin decimation: k = 1)."""
        x, y = self.origen_roi()
        return (float(x), float(y), 1.0)

    def info_captura(self):
        """Texto que identifica la captura (ROI) para validar las calibraciones guardadas."""
        try:
            x, y, w, h = self.cam.get_region()
            return f"roi{x},{y}"
        except Exception:  # noqa: BLE001
            return ""

    def close(self):
        try:
            self.cam.stop_acquisition()
        except Exception:  # noqa: BLE001
            pass
        self.stream = None  # soltar flujo y cámara: así se puede volver a abrir
        self.cam = None
        gc.collect()
        time.sleep(0.5)
