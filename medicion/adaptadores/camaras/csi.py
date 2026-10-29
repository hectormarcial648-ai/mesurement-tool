"""Adaptador de cámara: CSI (picamera2)."""
import cv2
import time


class CamaraCSI:
    """Raspberry Pi HQ / Camera Module vía picamera2.

    Entrega imágenes BGR. Los tamaños que ofrece son los MODOS NATIVOS del sensor (sin reescalado
    del ISP). Enfoque (si el sensor tiene autofocus), exposición y balance de blancos quedan fijos.
    """
    tipo = "csi"

    def __init__(self, width=None, height=None, lock=True, lens_position=None):
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Falta picamera2 ({e}): sudo apt install python3-picamera2")
        try:
            from libcamera import controls  # type: ignore
        except Exception:  # noqa: BLE001
            controls = None
        self._controls = controls
        try:
            self.picam = Picamera2()
        except IndexError:  # picamera2 no encontró ninguna cámara CSI
            raise RuntimeError(
                "No se detecta ninguna cámara CSI (cinta flex). Apaga la Pi y revisa la cinta; prueba "
                "'rpicam-hello --list-cameras'. Si activaste la ToF, restaura /boot/firmware/config.txt "
                "(camera_auto_detect=1 y sin dtoverlay=arducam-pivariety) y reinicia.")
        self.lock, self.lens_position = lock, lens_position
        self.gris = False  # True = la imagen se entrega en escala de grises (mono)
        self.modelo = str(self.picam.camera_properties.get("Model", "?"))
        self._lock_vals = None
        self._lp = None
        self.size = None
        if not width or not height:
            width, height, desc = self._modo_nativo()
        else:
            desc = self._describir_modo(width, height) or "tamaño pedido: el ISP lo reescala (menos nítido)"
        print(f"[csi] Sensor {self.modelo}: {width}x{height} ({desc})")
        self._arrancar(width, height, primera=True)

    # ----- modos -------------------------------------------------------------- #
    def _modos_dict(self):
        return {tuple(m["size"]): m for m in self.picam.sensor_modes}

    def _describir_modo(self, w, h, por_tam=None):
        try:
            m = (por_tam or self._modos_dict()).get((w, h))
            if m is None:
                return None
            pa = self.picam.camera_properties.get("PixelArraySize")
            completo = bool(pa) and tuple(m.get("crop_limits", ())) == (0, 0, pa[0], pa[1])
            return (f"modo nativo, {'campo completo' if completo else 'recorte central'}, "
                    f"hasta {float(m.get('fps', 0)):.0f} fps")
        except Exception:  # noqa: BLE001
            return None

    def _modo_nativo(self, objetivo_px=1.4e6):
        try:
            por_tam = self._modos_dict()
            w, h = min(por_tam, key=lambda t: abs(t[0] * t[1] - objetivo_px))
            return w, h, self._describir_modo(w, h, por_tam)
        except Exception as e:  # noqa: BLE001
            print(f"[csi] No se pudieron leer los modos del sensor ({e}); se usa 1536x864")
            return 1536, 864, "tamaño por defecto"

    def modos(self):
        """Lista de modos nativos para la interfaz."""
        out = []
        try:
            pa = self.picam.camera_properties.get("PixelArraySize")
            for (w, h), m in sorted(self._modos_dict().items(), key=lambda t: t[0][0] * t[0][1]):
                completo = bool(pa) and tuple(m.get("crop_limits", ())) == (0, 0, pa[0], pa[1])
                out.append({"w": int(w), "h": int(h), "fps": float(m.get("fps", 0)), "completo": completo,
                            "nombre": f"{w}x{h} ({'campo completo' if completo else 'recorte'}, "
                                      f"{float(m.get('fps', 0)):.0f} fps, {w * h / 1e6:.1f} MP)"})
        except Exception:  # noqa: BLE001
            pass
        return out

    def geometria_sensor(self):
        """(ox, oy, k): px del sensor = ox + (px de imagen + 0.5) * k - 0.5."""
        try:
            m = self._modos_dict().get(tuple(self.size))
            x, y, cw, ch = m["crop_limits"]
            return (float(x), float(y), float(cw) / float(self.size[0]))
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 1.0)

    def info_captura(self):
        return self.modelo

    def descripcion(self):
        return f"Cámara CSI {self.modelo}"

    # ----- arranque / bloqueos ----------------------------------------------- #
    def _arrancar(self, w, h, primera):
        cfg = self.picam.create_video_configuration(main={"size": (w, h), "format": "RGB888"})
        self.picam.configure(cfg)
        self.picam.start()
        self.size = (w, h)
        time.sleep(1.5 if primera else 0.8)
        self._enfocar(primera)
        self._bloquear(primera)

    def _enfocar(self, primera):
        c = self._controls
        try:
            if "AfMode" not in self.picam.camera_controls:
                if primera:
                    print("[csi] Este sensor no tiene autofocus: enfoca el lente A MANO (anillo de enfoque) con la "
                          "pieza a la altura de trabajo y bloquéalo. Prueba: rpicam-hello -t 0")
                return
            if c is None:
                return
            if self._lp is None and self.lens_position is not None:
                self._lp = float(self.lens_position)
            if self._lp is None:  # primera vez: enfocar una vez y guardar la posición
                self.picam.set_controls({"AfMode": c.AfModeEnum.Auto})
                try:
                    self.picam.autofocus_cycle()
                except Exception:  # noqa: BLE001
                    time.sleep(3.0)
                self._lp = self.picam.capture_metadata().get("LensPosition")
            if self._lp is not None:
                self.picam.set_controls({"AfMode": c.AfModeEnum.Manual, "LensPosition": float(self._lp)})
                if primera:
                    print(f"[csi] Enfoque bloqueado (LensPosition={float(self._lp):.2f})")
            time.sleep(0.4)
        except Exception as e:  # noqa: BLE001
            print(f"[csi] No se pudo bloquear el enfoque: {e}")

    def _bloquear(self, primera):
        if not self.lock:
            return
        try:
            if self._lock_vals is None:  # primera vez: medir y fijar lo que eligió el auto-exposición
                time.sleep(1.0)
                md = self.picam.capture_metadata()
                self._lock_vals = {"ExposureTime": int(md["ExposureTime"]), "AnalogueGain": float(md["AnalogueGain"]),
                                   "ColourGains": tuple(md["ColourGains"])}
            self._aplicar_lock()
            if primera:
                print(f"[csi] Exposición bloqueada: {self._lock_vals['ExposureTime']} us, "
                      f"ganancia {self._lock_vals['AnalogueGain']:.2f}")
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001
            print(f"[csi] No se pudo bloquear exposición/AWB: {e}")

    def _aplicar_lock(self):
        v = self._lock_vals
        self.picam.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": int(v["ExposureTime"]),
                                 "AnalogueGain": float(v["AnalogueGain"]), "ColourGains": tuple(v["ColourGains"])})

    # ----- cambios en caliente ------------------------------------------------ #
    def set_modo(self, w, h):
        """Cambia el modo del sensor: detiene, reconfigura y reinicia (reaplicando enfoque y exposición)."""
        self.picam.stop()
        self._arrancar(int(w), int(h), primera=False)

    def set_exposicion(self, us):
        if self._lock_vals is None:
            raise RuntimeError("La exposición no está bloqueada (--no-lock): no se puede fijar a mano")
        self._lock_vals["ExposureTime"] = int(us)
        self._aplicar_lock()

    def set_ganancia(self, g):
        if self._lock_vals is None:
            raise RuntimeError("La exposición no está bloqueada (--no-lock): no se puede fijar a mano")
        self._lock_vals["AnalogueGain"] = float(g)
        self._aplicar_lock()

    def exposicion_ganancia(self):
        v = self._lock_vals
        return (float(v["ExposureTime"]), float(v["AnalogueGain"])) if v else (None, None)

    def set_gris(self, v):
        """Alterna color / escala de grises en tiempo real (sin reiniciar el sensor)."""
        self.gris = bool(v)

    def read(self):
        f = self.picam.capture_array()  # RGB888 de libcamera = orden BGR para OpenCV
        if self.gris and f is not None:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2GRAY if f.ndim == 3 and f.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        return f

    def close(self):
        try:
            self.picam.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.picam.close()  # libera la cámara para poder volver a abrirla
        except Exception:  # noqa: BLE001
            pass
