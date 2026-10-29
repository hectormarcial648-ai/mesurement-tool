"""Hilo dueño de cada cámara: captura, detección en vivo, video y órdenes de la web."""
import argparse
import queue
import threading
import time

import numpy as np
from medicion.adaptadores.web.util import codificar
from medicion.aplicacion.sistema import items_de_objs
from medicion.config import CAPTURAS_DIR


ACCIONES_MEDICION = {"medir", "verificar", "calibrar_ref", "calibrar_tablero", "lente_pose",
                     "lente_calcular", "fondo_capturar", "regla", "patron_capturar", "distancia_calibrar"}


class CamaraWorker(threading.Thread):
    """Una cámara + su `Sistema`. Todo acceso a ambos ocurre en ESTE hilo; el resto del programa
    (servidor web) solo lee resultados publicados y envía órdenes por `self.cola`."""

    def __init__(self, cid, fuente, args, orden, fabrica):
        super().__init__(daemon=True, name=f"cam-{cid}")
        self.cid, self.fuente, self.orden = cid, fuente, orden
        self.fabrica = fabrica  # arma (camara, sistema); la inyecta la composición
        self.args = argparse.Namespace(**vars(args))  # parámetros PROPIOS de esta cámara
        self.cola = queue.Queue()
        self.cond = threading.Condition()
        self._parar = threading.Event()
        self.cam = None
        self.sis = None
        self.estado_txt = "abriendo"
        self.error = ""
        self.ocupado = ""
        self.jpeg = None
        self.seq = 0
        self.ultimo_anotado = None
        self.objs = []
        self.resultado = None
        self.resultado_jpeg = None
        self.fps = 0.0
        self.vista_ancho = int(args.vista_ancho)
        self.vista_calidad = int(args.vista_calidad)
        self.vista_fps = float(args.vista_fps)
        self.tasa = float(args.tasa_vivo)
        self.hires_vivo = bool(args.hires_vivo)
        self.regla_n = 0
        self.encendida = True   # cámara prendida (abierta) o apagada (liberada)
        self.medicion = True    # sistema de medición prendido; apagado = solo inspección visual
        self.gris = False       # solo CSI: imagen en escala de grises
        self._nombre = ""
        self._estado_json = {"id": cid, "estado": "abriendo"}
        self._t_det = 0.0
        self._t_pub = 0.0
        self._t_est = 0.0
        self._t_fps = time.time()
        self._n_fps = 0
        self._fallos = 0
        self._t_frame = time.time()
        # reconocimiento de piezas (opcional: solo si la fábrica trae biblioteca)
        self.biblioteca = getattr(fabrica, "biblioteca", None)
        self.cnn = fabrica.crear_cnn() if hasattr(fabrica, "crear_cnn") else None
        self.repo_ver = fabrica.repo_veredictos(cid) if hasattr(fabrica, "repo_veredictos") else None
        self.reco_vivo, self.reco_modo, self.reco_auto = False, "geometrico", False
        self.reco = None               # última identificación en vivo
        self.pieza_sel = None          # pieza elegida a mano (para veredicto y para enseñar)
        self.ensenanza = None          # aprendizaje en curso
        self.ensenanza_fin = None      # resultado del último aprendizaje
        self.evaluacion = None
        self.control_cada_h = 8.0
        self._estable = (None, 0)
        self._receta_aplicada = None
        self._ctrl = (0.0, [])         # caché del historial de controles (instante, filas)

    # ----- ciclo de vida ------------------------------------------------------ #
    def parar(self):
        self._parar.set()

    def run(self):
        while not self._parar.is_set():
            if self.sis is None:
                if self.encendida:
                    self._abrir()
                if self.sis is None:  # apagada o no se pudo abrir: seguir atendiendo órdenes
                    self._publicar_estado(forzar=True)
                    self._atender(esperar=0.5)
                    continue
            self._atender()
            if self.sis is None:  # la orden apagó la cámara: volver al inicio del bucle
                continue
            try:
                self._ciclo()
            except Exception as e:  # noqa: BLE001
                self._fallos += 1
                print(f"[{self.cid}] error en el ciclo: {e!r}")
                if self._fallos >= 5:
                    self._cerrar(f"La cámara dejó de responder: {e}")
                time.sleep(0.2)
        self._cerrar(None)

    def _abrir(self):
        self.estado_txt, self.error = "abriendo", ""
        cam = None
        try:
            cam, sis = self.fabrica.armar(self.fuente, self.cid, self.args)  # si falla, ya cerró la cámara
            if hasattr(cam, "set_gris"):
                cam.set_gris(self.gris)
            sis.iniciar()
        except BaseException as e:  # noqa: BLE001  (las cámaras pueden lanzar SystemExit con el mensaje)
            if isinstance(e, KeyboardInterrupt):
                raise
            msg = str(e) or repr(e)
            if not isinstance(e, (SystemExit, RuntimeError)):
                msg = f"{type(e).__name__}: {msg}"
            self.error = msg
            self.estado_txt = "error"
            print(f"[{self.cid}] No se pudo abrir: {self.error}")
            try:
                if cam is not None:
                    cam.close()
            except Exception:  # noqa: BLE001
                pass
            return
        self.cam, self.sis = cam, sis
        self._nombre = cam.descripcion() if hasattr(cam, "descripcion") else self.fuente
        self.estado_txt, self._fallos, self._t_frame = "ok", 0, time.time()
        self._t_abierta = time.time()  # para avisar si se calibra con la cámara aún calentando
        print(f"[{self.cid}] abierta: {cam.descripcion() if hasattr(cam, 'descripcion') else self.fuente}")

    def _cerrar(self, motivo):
        try:
            if self.cam is not None:
                self.cam.close()
        except Exception:  # noqa: BLE001
            pass
        self.cam, self.sis = None, None
        if motivo:
            self.estado_txt, self.error = "error", motivo

    # ----- ciclo de captura ---------------------------------------------------- #
    def _ciclo(self):
        sis = self.sis
        frame = sis.leer()
        ahora = time.time()
        if frame is None:
            if ahora - self._t_frame > 8 and self.estado_txt == "ok":
                self.estado_txt = "sin_imagen"
                self.error = "La cámara no entrega imágenes (revisa el cable, la alimentación PoE y la IP). Pulsa Reintentar."
                print(f"[{self.cid}] {self.error}")
            self._publicar_estado()  # que la interfaz vea el estado aunque no lleguen imágenes
            time.sleep(0.02)
            return
        self._fallos, self._t_frame = 0, ahora
        if self.estado_txt == "sin_imagen":
            self.estado_txt, self.error = "ok", ""
            sis.aviso("La cámara volvió a entregar imágenes")
        self._n_fps += 1
        if ahora - self._t_fps >= 1.0:
            self.fps = self._n_fps / (ahora - self._t_fps)
            self._n_fps, self._t_fps = 0, ahora
        if not self.medicion:
            self.objs = []  # inspección visual: sin detección ni anotaciones
        elif ahora - self._t_det >= 1.0 / max(self.tasa, 0.1):
            self.objs = sis.detectar(frame, hires=self.hires_vivo)
            self._t_det = ahora
            self._tras_deteccion(frame, ahora)
        if ahora - self._t_pub >= 1.0 / max(self.vista_fps, 0.5):
            anot = sis.anotar(frame, items_de_objs(self.objs), pie=False, cabecera=False)
            self.ultimo_anotado = anot
            data = codificar(anot, self.vista_calidad, self.vista_ancho)
            if data:
                with self.cond:
                    self.jpeg = data
                    self.seq += 1
                    self.cond.notify_all()
            self._t_pub = ahora
        self._publicar_estado()

    def esperar_video(self, ultimo, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != ultimo or self._parar.is_set(), timeout=timeout)
            return self.seq, self.jpeg

    # ----- estado para la web --------------------------------------------------- #
    def _publicar_estado(self, forzar=False):
        ahora = time.time()
        if not forzar and ahora - self._t_est < 0.3:
            return
        self._t_est = ahora
        try:
            self._estado_json = self._construir_estado()
        except Exception as e:  # noqa: BLE001
            self._estado_json = {"id": self.cid, "estado": "error", "error": f"estado: {e!r}", "orden": self.orden}

    def csv_ruta(self):
        return self.sis.csv_path if self.sis is not None else self.fabrica.ruta_csv(self.fuente, self.cid)

    def estado(self):
        return self._estado_json

    def _construir_estado(self):
        e = {"id": self.cid, "fuente": self.fuente, "orden": self.orden, "estado": self.estado_txt,
             "error": self.error, "ocupado": self.ocupado, "fps": round(self.fps, 1), "nombre": self._nombre or self.fuente,
             "encendida": self.encendida, "medicion": self.medicion, "gris": self.gris,
             "color_soporta": self.fuente == "csi"}
        sis, cam = self.sis, self.cam
        if sis is None or cam is None:
            return e
        e.update(sis.estado_dict())
        e["nombre"] = cam.descripcion() if hasattr(cam, "descripcion") else self.fuente
        e["tipo"] = getattr(cam, "tipo", self.fuente)
        e["vista"] = {"ancho": self.vista_ancho, "calidad": self.vista_calidad, "fps": self.vista_fps}
        e["tasa"], e["hires_vivo"] = self.tasa, self.hires_vivo
        e["segmentador"] = sis.segmentador.nombre
        if self.biblioteca is not None:
            e["piezas"] = self._estado_piezas()
        if hasattr(sis.segmentador, "estado"):
            e["red"] = sis.segmentador.estado()
        if getattr(self, "comparacion", None) is not None:
            e["comparacion"] = self.comparacion
        e["calentamiento"] = {"min": (time.time() - getattr(self, "_t_abierta", time.time())) / 60.0,
                              "req": float(self.args.calentar_min)}
        u = sis.escala.unidad
        e["objetos"] = [{"tipo": o.get("tipo", "rect"), "largo": float(o["largo"]), "ancho": float(o["ancho"]),
                         "unidad": u} for o in self.objs]
        if self.resultado is not None:
            e["resultado"] = self.resultado
        e["mensajes"] = [{"t": t, "texto": m} for t, m in list(sis.log)[-40:]]
        # resoluciones disponibles según el tipo de cámara
        if hasattr(cam, "set_roi"):
            roi_actual = None
            try:
                roi_actual = list(cam.cam.get_region())
            except Exception:  # noqa: BLE001
                pass
            e["resoluciones"] = {"tipo": "gige", "actual": sis.preset_i, "sensor": list(cam.sensor), "roi": roi_actual,
                                 "lista": [{"i": i, "nombre": sis._nombre_preset(i), "roi": r, "escala": s}
                                           for i, (r, s) in enumerate(sis.presets)]}
        elif hasattr(cam, "set_modo"):
            e["resoluciones"] = {"tipo": "csi", "actual": list(cam.size), "lista": cam.modos()}
        else:
            e["resoluciones"] = None
        # exposición / ganancia
        ex = {"soporta": False}
        try:
            if hasattr(cam, "set_roi"):
                ex = {"soporta": True, "exposicion": float(cam.cam.get_exposure_time()), "ganancia": float(cam.cam.get_gain())}
                try:
                    ex["exp_lim"] = [float(v) for v in cam.cam.get_exposure_time_bounds()]
                    ex["gan_lim"] = [float(v) for v in cam.cam.get_gain_bounds()]
                except Exception:  # noqa: BLE001
                    pass
            elif hasattr(cam, "exposicion_ganancia"):
                x, g = cam.exposicion_ganancia()
                ex = {"soporta": x is not None, "exposicion": x, "ganancia": g}
        except Exception:  # noqa: BLE001
            pass
        e["exposicion"] = ex
        if hasattr(cam, "alternar"):
            e["tof"] = {"modo": cam.modo}
        return e

    # ----- órdenes -------------------------------------------------------------- #
    def enviar(self, accion, datos, timeout=240):
        """Llamado desde el servidor web: encola la orden y espera el resultado."""
        orden = {"accion": accion, "datos": datos or {}, "evento": threading.Event(), "resp": {}}
        self.cola.put(orden)
        if not orden["evento"].wait(timeout):
            return {"ok": False, "error": "La orden tardó demasiado (sigue en curso)"}
        return orden["resp"]

    def _atender(self, esperar=0.0):
        """Ejecuta las órdenes pendientes (opcionalmente esperando hasta `esperar` s por la primera)."""
        primera = True
        while True:
            try:
                if primera and esperar:
                    orden = self.cola.get(timeout=esperar)
                else:
                    orden = self.cola.get_nowait()
            except queue.Empty:
                break
            primera = False
            self._ejecutar(orden)

    def _ejecutar(self, orden):
        accion = orden["accion"]
        self.ocupado = accion
        self._publicar_estado(forzar=True)
        try:
            fn = getattr(self, "acc_" + accion, None)
            if fn is None:
                orden["resp"].update({"ok": False, "error": f"Acción desconocida: {accion}"})
            elif self.sis is None and accion not in ("reintentar", "camara", "medicion"):
                orden["resp"].update({"ok": False, "error": "La cámara está apagada o no abierta. " + (self.error or "")})
            elif not self.medicion and accion in ACCIONES_MEDICION:
                orden["resp"].update({"ok": False, "error": "El sistema de medición está apagado (inspección visual). Préndelo para usar esta función."})
            else:
                resp = fn(orden["datos"]) or {}
                resp.setdefault("ok", True)
                orden["resp"].update(resp)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.cid}] error en '{accion}': {e!r}")
            orden["resp"].update({"ok": False, "error": str(e) or repr(e)})
        finally:
            self.ocupado = ""
            self._publicar_estado(forzar=True)
            orden["evento"].set()

    def _msg(self):
        return self.sis.msg[0] if self.sis else ""

    def _resultado_img(self, frame, res):
        anot = self.sis.anotar(frame, res, pie=False, cabecera=False)
        self.resultado_jpeg = codificar(anot, 88)

    # ----- acciones ---------------------------------------------------------------- #
    def acc_camara(self, d):
        """Prende o apaga la cámara (al apagar se libera el dispositivo)."""
        on = bool(d.get("on", not self.encendida))
        if on:
            self.encendida = True
            if self.sis is not None and self.estado_txt == "ok":
                return {"mensaje": "La cámara ya está encendida"}
            if self.sis is not None:
                self._cerrar(None)
            self._abrir()
            return {"ok": self.sis is not None, "mensaje": self.error or "Cámara encendida"}
        self.encendida = False
        self._cerrar(None)
        self.estado_txt, self.error = "apagada", ""
        self.objs, self.fps, self.ultimo_anotado = [], 0.0, None
        with self.cond:
            self.jpeg = None
        return {"mensaje": "Cámara apagada"}

    def acc_medicion(self, d):
        """Prende o apaga el sistema de medición (apagado = solo video / inspección visual)."""
        self.medicion = bool(d.get("on", not self.medicion))
        self.objs = []
        if self.sis is not None:
            self.sis.aviso("Medición " + ("ENCENDIDA" if self.medicion else "APAGADA: solo inspección visual"))
        return {"mensaje": "Medición " + ("encendida" if self.medicion else "apagada (inspección visual)")}

    def acc_color(self, d):
        """CSI: alterna imagen a color / escala de grises en tiempo real."""
        cam, sis = self.cam, self.sis
        if not hasattr(cam, "set_gris"):
            return {"ok": False, "error": "Solo la cámara CSI permite alternar color / escala de grises"}
        self.gris = bool(d.get("gris", not self.gris))
        cam.set_gris(self.gris)
        # el fondo (capturado o adaptativo) era del otro tipo de imagen: se descarta
        sis.bg_bgr = sis.bg_lab = None
        sis._fondo_ref, sis._fondo_n = None, 0
        self.objs = []
        txt = "Imagen en ESCALA DE GRISES" if self.gris else "Imagen a COLOR"
        sis.aviso(txt + " (el fondo capturado se descartó)")
        return {"mensaje": txt}

    def acc_reintentar(self, d):
        self.encendida = True
        if self.sis is not None and self.estado_txt == "ok":
            return {"mensaje": "La cámara ya está funcionando"}
        if self.sis is not None:  # estaba abierta pero sin imágenes: cerrar y volver a abrir
            self._cerrar(None)
        self._abrir()
        return {"ok": self.sis is not None, "mensaje": self.error or "Cámara abierta"}

    def _aviso_calentamiento(self):
        t = (time.time() - getattr(self, "_t_abierta", time.time())) / 60.0
        req = float(self.args.calentar_min)
        if t < req:
            return (f" ⚠ La cámara lleva {t:.0f} min encendida: la imagen aún puede desplazarse por calentamiento. "
                    f"Lo ideal es esperar ~{req:.0f} min antes de calibrar.")
        return ""

    def acc_patron_capturar(self, d):
        sis = self.sis
        if not d.get("real_largo"):
            return {"ok": False, "error": "Escribe la medida REAL de la pieza patrón (lado largo, en mm)"}
        ok = sis.capturar_patron(d["real_largo"], d.get("real_ancho"), int(d.get("frames") or self.args.frames))
        return {"ok": bool(ok), "mensaje": self._msg() + self._aviso_calentamiento()}

    def acc_patron_calcular(self, d):
        ok = self.sis.ajustar_corr_lineal()
        return {"ok": bool(ok), "mensaje": self._msg()}

    def acc_patron_reiniciar(self, d):
        self.sis.patrones = []
        self.sis.aviso("Patrones descartados")
        return {"mensaje": self._msg()}

    def acc_corr_quitar(self, d):
        self.sis.reset_corr_lineal()
        self.sis.patrones = []
        self.sis.aviso("Corrección contra patrones quitada: se miden las dimensiones sin corregir")
        return {"mensaje": self._msg()}

    def acc_distancia_calibrar(self, d):
        sis = self.sis
        if not d.get("real_largo"):
            return {"ok": False, "error": "Escribe la medida REAL de la pieza patrón (lado largo, en mm)"}
        D = sis.calibrar_distancia(d["real_largo"], d.get("real_ancho"), d.get("altura_mm"), int(d.get("frames") or 5))
        return {"ok": D is not None, "mensaje": self._msg() + self._aviso_calentamiento()}

    def acc_prueba_iniciar(self, d):
        if not d.get("real_largo"):
            return {"ok": False, "error": "Escribe la medida REAL de la pieza de prueba (lado largo, en mm)"}
        self.sis.prueba_iniciar(d["real_largo"], d.get("real_ancho"), d.get("tol"))
        return {"mensaje": self._msg()}

    def acc_prueba_reiniciar(self, d):
        if self.sis.prueba is not None:
            self.sis.prueba["m"] = []
        self.sis.aviso("Mediciones de la prueba descartadas")
        return {"mensaje": self._msg()}

    def acc_prueba_detener(self, d):
        self.sis.prueba = None
        self.sis.aviso("Prueba de precisión terminada")
        return {"mensaje": self._msg()}

    def acc_medir(self, d):
        sis = self.sis
        n = int(d.get("frames") or self.args.frames)
        frame, res = sis.medir_promedio(n)
        if not res:
            return {"ok": False, "error": "No se detectó ningún objeto. Revisa la pieza, la luz y el fondo."}
        sis.registrar_csv(res)
        self._resultado_img(frame, res)
        filas = [{"tipo": (r["obj"]["tipo"] if r.get("obj") else "rect"), "largo": r["largo"], "ancho": r["ancho"],
                  "std_l": r["std_l"], "std_a": r["std_a"], "n": r["n"]} for r in res]
        extra = ""
        if sis.prueba is not None:
            sis.prueba_registrar(res[0])
            extra = f" | prueba de precisión: {len(sis.prueba['m'])} mediciones"
        pieza = self._pieza_para_medida(frame, res[0]) if self.biblioteca is not None else None
        if pieza and pieza.get("veredicto"):
            extra += f" | {pieza['nombre']}: {pieza['veredicto']['estado']}"
        self.resultado = {"t": time.time(), "unidad": sis.escala.unidad, "filas": filas, "origen": "medir",
                          "mensaje": f"Medición guardada en {sis.csv_path.name}{extra}", "pieza": pieza}
        sis.aviso(self.resultado["mensaje"])
        return {"ok": True, "resultado": self.resultado}

    def acc_verificar(self, d):
        sis = self.sis
        if d.get("largo") and d.get("ancho"):
            sis.set_check(float(d["largo"]), float(d["ancho"]))
        out, msg = sis.verificar()
        if not isinstance(out, tuple):
            return {"ok": False, "error": msg}
        frame, res = out
        self._resultado_img(frame, res)
        filas = [{"tipo": (r["obj"]["tipo"] if r.get("obj") else "rect"), "largo": r["largo"], "ancho": r["ancho"],
                  "std_l": r["std_l"], "std_a": r["std_a"], "n": r["n"]} for r in res]
        self.resultado = {"t": time.time(), "unidad": sis.escala.unidad, "filas": filas, "origen": "verificar",
                          "mensaje": msg}
        return {"ok": True, "resultado": self.resultado, "mensaje": msg}

    def acc_calibrar_ref(self, d):
        sis = self.sis
        if d.get("ref_mm"):
            sis.args.ref_mm = float(d["ref_mm"])
        if d.get("ref_alto_mm") is not None:
            sis.args.ref_alto_mm = float(d["ref_alto_mm"] or 0)
        sis.calibrar_ref()
        return {"ok": sis.escala.px_per_mm is not None, "mensaje": self._msg() + self._aviso_calentamiento()}

    def acc_calibrar_tablero(self, d):
        sis = self.sis
        sis.set_tablero(int(d.get("cols", 9)), int(d.get("filas", 6)), float(d.get("lado_mm", sis.args.square_mm)))
        sis.calibrar_plano()
        return {"ok": sis.escala.H is not None, "mensaje": self._msg() + self._aviso_calentamiento()}

    def acc_lente_pose(self, d):
        sis = self.sis
        sis.set_tablero(int(d.get("cols", sis.patron[0])), int(d.get("filas", sis.patron[1])),
                        float(d.get("lado_mm", sis.args.square_mm)))
        sis.capturar_pose_lente()
        return {"ok": True, "mensaje": self._msg()}

    def acc_lente_calcular(self, d):
        self.sis.finalizar_lente()
        return {"ok": self.sis.maps is not None, "mensaje": self._msg()}

    def acc_lente_reiniciar(self, d):
        self.sis.lente_obj, self.sis.lente_img = [], []
        self.sis.aviso("Poses de lente descartadas")
        return {"ok": True, "mensaje": self._msg()}

    def acc_fondo_capturar(self, d):
        self.sis.capturar_fondo()
        return {"ok": self.sis.bg_lab is not None, "mensaje": self._msg()}

    def acc_fondo_olvidar(self, d):
        self.sis.olvidar_fondo()
        return {"mensaje": self._msg()}

    def acc_altura(self, d):
        self.sis.fijar_altura(float(d.get("altura_mm") or 0))
        return {"mensaje": self._msg()}

    def acc_distancia(self, d):
        sis = self.sis
        if d.get("auto") or not d.get("cam_dist_mm"):
            sis.args.cam_dist_mm = None
            sis.aviso("Distancia lente-mesa: automática (desde el tablero)")
        else:
            sis.fijar_dist(float(d["cam_dist_mm"]))
        return {"mensaje": self._msg()}

    def acc_ref_alto(self, d):
        self.sis.args.ref_alto_mm = float(d.get("ref_alto_mm") or 0)
        self.sis.aviso(f"Altura de la pieza de referencia = {self.sis.args.ref_alto_mm:g} mm")
        return {"mensaje": self._msg()}

    def acc_huecos(self, d):
        self.sis.holes = bool(d.get("on", not self.sis.holes))
        self.sis.aviso("Huecos circulares: " + ("ACTIVADOS" if self.sis.holes else "desactivados"))
        return {"mensaje": self._msg()}

    def acc_silueta(self, d):
        self.sis.mostrar_silueta = bool(d.get("on", not self.sis.mostrar_silueta))
        return {"mensaje": "Silueta visible (cian punteada) con la altura: " + ("sí" if self.sis.mostrar_silueta else "no")}

    def acc_hires_vivo(self, d):
        self.hires_vivo = bool(d.get("on", not self.hires_vivo))
        return {"mensaje": "Bordes a resolución completa en vivo: " + ("sí" if self.hires_vivo else "no")}

    def acc_resolucion(self, d):
        sis, cam = self.sis, self.cam
        if hasattr(cam, "set_roi"):
            if "indice" in d:
                ok = sis.aplicar_resolucion_gige(int(d["indice"]) % len(sis.presets))
            else:
                ok = sis.aplicar_resolucion_gige(sis.agregar_preset(int(d.get("roi", 0)), int(d.get("escala", 1))))
        elif hasattr(cam, "set_modo"):
            ok = sis.cambiar_modo_csi(int(d["w"]), int(d["h"]))
        else:
            return {"ok": False, "error": "Esta cámara no permite cambiar la resolución"}
        self.objs, self.resultado = [], None
        return {"ok": bool(ok), "mensaje": self._msg()}

    def acc_exposicion(self, d):
        sis, cam = self.sis, self.cam
        if not hasattr(cam, "set_exposicion"):
            return {"ok": False, "error": "Esta cámara no permite ajustar la exposición"}
        cam.set_exposicion(float(d["exposicion"]))
        if sis.bg_lab is not None:
            sis.olvidar_fondo()
        sis.aviso(f"Exposición = {float(d['exposicion']):.0f} us (el fondo capturado se descartó: cambió la luz)")
        return {"mensaje": self._msg()}

    def acc_ganancia(self, d):
        sis, cam = self.sis, self.cam
        if not hasattr(cam, "set_ganancia"):
            return {"ok": False, "error": "Esta cámara no permite ajustar la ganancia"}
        cam.set_ganancia(float(d["ganancia"]))
        if sis.bg_lab is not None:
            sis.olvidar_fondo()
        sis.aviso(f"Ganancia = {float(d['ganancia']):.1f} (el fondo capturado se descartó: cambió la luz)")
        return {"mensaje": self._msg()}

    def acc_vista(self, d):
        if d.get("ancho"):
            self.vista_ancho = max(200, min(int(d["ancho"]), 3000))
        if d.get("calidad"):
            self.vista_calidad = max(20, min(int(d["calidad"]), 95))
        if d.get("fps"):
            self.vista_fps = max(1.0, min(float(d["fps"]), 30.0))
        if d.get("tasa"):
            self.tasa = max(0.2, min(float(d["tasa"]), 30.0))
        return {"mensaje": "Ajustes de video aplicados"}

    def acc_regla(self, d):
        sis = self.sis
        p1, p2 = d["p1"], d["p2"]
        dist, dx, dy = sis.distancia_puntos((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        if d.get("guardar"):
            self.regla_n += 1
            sis.registrar_regla(self.regla_n, dist)
        return {"ok": True, "d": dist, "dx": dx, "dy": dy, "unidad": sis.escala.unidad}

    def acc_guardar_captura(self, d):
        if self.ultimo_anotado is None:
            return {"ok": False, "error": "Todavía no hay imagen"}
        CAPTURAS_DIR.mkdir(exist_ok=True)
        nombre = f"{self.cid}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        data = codificar(self.ultimo_anotado, formato=".png")
        (CAPTURAS_DIR / nombre).write_bytes(data)
        return {"ok": True, "url": f"/capturas/{nombre}", "mensaje": f"Captura guardada: capturas/{nombre}"}

    # ----- red neuronal de segmentación (opcional) ------------------------------------------- #
    def _seg(self):
        seg = self.sis.segmentador
        if not hasattr(seg, "configurar"):
            raise RuntimeError("Este segmentador no se puede cambiar en caliente")
        return seg

    @staticmethod
    def _ajustes_red(d):
        return {k: d[k] for k in ("modelo", "modo", "umbral", "margen_px", "cada") if d.get(k) not in (None, "")}

    def acc_seg_configurar(self, d):
        seg = self._seg()
        seg.configurar(**self._ajustes_red(d))
        self.sis.aviso("Ajustes de la red guardados" + (" y aplicados" if seg.estado()["activa"] else ""))
        return {"mensaje": self._msg()}

    def acc_seg_encender(self, d):
        seg = self._seg()
        seg.configurar(**self._ajustes_red(d))
        seg.encender()
        self.objs = []
        self.sis.aviso(f"Red neuronal ENCENDIDA: {seg.nombre}", 6)
        return {"mensaje": self._msg()}

    def acc_seg_apagar(self, d):
        self._seg().apagar()
        self.objs = []
        self.sis.aviso("Red neuronal APAGADA: se usa el método clásico", 6)
        return {"mensaje": self._msg()}

    def acc_seg_descargar(self, d):
        ruta = self._seg().descargar_modelo()
        self.sis.aviso(f"Modelo descargado y verificado: {ruta.name}", 8)
        return {"mensaje": self._msg()}

    def acc_seg_comparar(self, d):
        """Detecta en UNA misma imagen con el método clásico y con la red (restringir / rescatar) y lo compara."""
        sis, seg = self.sis, self._seg()
        frame = None
        for _ in range(3):
            f = sis.leer()
            if f is not None:
                frame = f
        if frame is None:
            return {"ok": False, "error": "No hay imagen"}
        filas = []
        for var, etiqueta in (("clasico", "Clásico (sin red)"), ("restringir", "Red + clásico: restringir"),
                              ("rescatar", "Red + clásico: rescatar")):
            try:
                with seg.forzar(var):
                    t0 = time.perf_counter()
                    objs = sis.detectar(frame, fondo="fresco", hires=False)
                    ms = (time.perf_counter() - t0) * 1000.0
                    ms_red = getattr(seg.red, "ultimo_ms", None) if var != "clasico" else None
                filas.append({"variante": etiqueta, "n": len(objs), "ms": ms, "ms_red": ms_red,
                              "objetos": [{"tipo": o.get("tipo", "rect"), "largo": float(o["largo"]), "ancho": float(o["ancho"])}
                                          for o in objs]})
            except Exception as e:  # noqa: BLE001
                filas.append({"variante": etiqueta, "error": str(e) or repr(e)})
        self.comparacion = {"t": time.time(), "unidad": sis.escala.unidad, "filas": filas}
        return {"comparacion": self.comparacion, "mensaje": "Comparación lista (abajo)"}

    # ===== PIEZAS: reconocimiento, aprendizaje, receta, veredicto y control con la pieza maestra ===== #
    def _mm_por_px(self):
        e = self.sis.escala
        if not e.calibrada:
            return None
        try:
            p = e.a_unid([[0.0, 0.0], [100.0, 0.0]])
            return float(np.hypot(*(p[1] - p[0])) / 100.0)
        except Exception:  # noqa: BLE001
            return None

    def _cnn_activa(self):
        return self.cnn if (self.cnn is not None and self.cnn.activo) else None

    def _identificar(self, frame, obj):
        desc, motivo = self.biblioteca.descriptor(frame, obj, self._mm_por_px(), self._cnn_activa())
        if desc is None:
            return {"estado": "vacia", "id": None, "nombre": None, "candidatas": [], "nota": motivo}
        return self.biblioteca.identificar(self.cid, desc, self.reco_modo)

    def _tras_deteccion(self, frame, ahora):
        if self.biblioteca is None:
            return
        if self.ensenanza:
            self._paso_ensenanza(frame, ahora)
        if self.reco_vivo:
            self._reco_en_vivo(frame)

    def _reco_en_vivo(self, frame):
        if not self.objs:
            self.reco, self._estable = None, (None, 0)
            return
        obj = max(self.objs, key=lambda o: o.get("area_px", 0))
        try:
            r = self._identificar(frame, obj)
        except Exception as e:  # noqa: BLE001
            self.reco = {"estado": "vacia", "id": None, "nombre": None, "candidatas": [], "nota": f"error: {e}"}
            return
        r["n_objetos"] = len(self.objs)
        if r["estado"] == "reconocida" and self.sis.escala.calibrada:
            r["veredicto"] = self.biblioteca.veredicto_para(r["id"], {"largo": obj["largo"], "ancho": obj["ancho"], "tipo": obj.get("tipo", "rect")})
        self.reco = r
        if r["estado"] == "reconocida":
            pid, n = self._estable
            self._estable = (r["id"], n + 1) if pid == r["id"] else (r["id"], 1)
            if self.reco_auto and self._estable[1] >= 3 and self._receta_aplicada != r["id"]:
                self._aplicar_receta(r["id"])
        else:
            self._estable = (None, 0)

    def _aplicar_receta(self, pid):
        pieza = self.biblioteca.obtener(pid)
        if pieza is None:
            raise ValueError("La pieza no existe")
        rec = self.biblioteca.receta(pieza)
        sis = self.sis
        if rec["altura_mm"] is not None:
            sis.fijar_altura(float(rec["altura_mm"]))
        l, a = rec["check_mm"]
        if l and a:
            sis.set_check(float(l), float(a))
        self.pieza_sel, self._receta_aplicada = pid, pid
        sis.aviso(f"Receta cargada: {pieza['nombre']}" + (f" (altura {rec['altura_mm']:g} mm)" if rec["altura_mm"] is not None else "")
                  + ". La ESCALA no cambia: depende del montaje, no de la pieza.", 8)

    def _pieza_para_medida(self, frame, r0):
        """Identifica la pieza medida y la compara con su nominal (usa la medida promediada y corregida)."""
        obj = r0.get("obj")
        medidas = {"largo": float(r0["largo"]), "ancho": float(r0["ancho"]), "tipo": obj["tipo"] if obj else "rect"}
        ident, pid, origen, aviso = None, None, None, ""
        if self.reco_vivo and obj:
            ident = self._identificar(frame, obj)
            if ident["estado"] == "reconocida":
                pid, origen = ident["id"], "reconocida"
            elif ident["estado"] in ("desconocida", "ambigua"):
                aviso = "La pieza no se reconoció (" + ident["estado"] + ")"
        elif self.reco_vivo and not obj:
            aviso = "La pieza se movió durante la medición: no se pudo reconocer. Déjala quieta mientras mide."
        if pid is None and self.pieza_sel and self.biblioteca.obtener(self.pieza_sel):
            pid, origen = self.pieza_sel, "elegida a mano"
        info = {"reco": ident, "pid": pid, "origen": origen, "nombre": self.biblioteca.nombre(pid) if pid else None, "aviso": aviso}
        if pid and self.sis.escala.calibrada:
            ver = self.biblioteca.veredicto_para(pid, medidas)
            info["veredicto"] = ver
            if self.repo_ver is not None and ver:
                f = ver["filas"]
                nom = self.biblioteca.obtener(pid)
                self.repo_ver.registrar([[time.strftime("%Y-%m-%d %H:%M:%S"), info["nombre"], origen, f"{medidas['largo']:.3f}", f"{medidas['ancho']:.3f}",
                                          nom.get("largo") or nom.get("diametro") or "", nom.get("ancho") or "", ver.get("tol") or "",
                                          f"{f[0]['error']:+.3f}" if f else "", f"{f[1]['error']:+.3f}" if len(f) > 1 else "", ver["estado"]]])
        return info

    # ----- aprendizaje: corre dentro del ciclo de captura, así el video NO se detiene ------------------------------ #
    def _paso_ensenanza(self, frame, ahora):
        e = self.ensenanza
        if ahora < e["siguiente"]:
            return
        e["siguiente"] = ahora + e["intervalo"]
        e["intentos"] += 1
        motivo = None
        if len(self.objs) != 1:
            motivo = f"hay {len(self.objs)} objetos (debe haber solo la pieza)" if self.objs else "no se ve ninguna pieza"
        else:
            desc, mot = self.biblioteca.descriptor(frame, self.objs[0], self._mm_por_px(), self._cnn_activa())
            if desc is None:
                motivo = mot
            else:
                c = self.objs[0]["centro"]
                ca = np.asarray(self.objs[0].get("corners_px", [[0, 0]] * 4), float)
                v = ca[1] - ca[0]
                e["descs"].append(desc)
                e["poses"].append((float(c[0]), float(c[1]), float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)))
        if motivo:
            e["rechazos"][motivo] = e["rechazos"].get(motivo, 0) + 1
        if len(e["descs"]) >= e["total"] or e["intentos"] > e["total"] * 5:
            self._terminar_ensenanza()

    def _terminar_ensenanza(self, cancelado=False):
        e, self.ensenanza = self.ensenanza, None
        if e is None:
            return
        n = 0
        if e["descs"]:
            n = self.biblioteca.agregar_muestras(e["pid"], self.cid, e["descs"], miniatura=e["descs"][0]["parche"])
        avisos = []
        if len(e["poses"]) >= 3:
            p = np.array(e["poses"])
            if float(p[:, :2].std(axis=0).max()) < 12 and float(np.ptp(p[:, 2])) < 15:
                avisos.append("La pieza casi no se movió entre muestras: cambia su posición y su giro para que aprenda mejor")
        if n < e["total"] and not cancelado:
            avisos.append(f"Solo se pudieron guardar {n} de {e['total']} muestras (ver rechazos)")
        self.ensenanza_fin = {"t": time.time(), "pid": e["pid"], "nombre": e["nombre"], "guardadas": n, "total": e["total"],
                              "rechazos": e["rechazos"], "avisos": avisos, "cancelado": cancelado}
        self.evaluacion = None
        self.sis.aviso(f"Aprendizaje de «{e['nombre']}»: {n} muestras guardadas" + (" (cancelado)" if cancelado else ""), 8)

    def _estado_piezas(self):
        bib = self.biblioteca
        e = self.ensenanza
        ens = ({"activa": True, "nombre": e["nombre"], "hechas": len(e["descs"]), "total": e["total"], "rechazos": e["rechazos"]}
               if e else {"activa": False, "fin": self.ensenanza_fin})
        ahora = time.time()
        if ahora - self._ctrl[0] > 5.0:
            self._ctrl = (ahora, bib.controles(self.cid, 8))
        lista = bib.listar(self.cid)
        maestras = [{"id": p["id"], "nombre": p["nombre"]} for p in lista if p.get("maestra")]
        ult = self._ctrl[1][-1] if self._ctrl[1] else None
        hace_h = (ahora - float(ult["t"])) / 3600.0 if ult and ult.get("t") else None
        return {"lista": lista, "vivo": self.reco_vivo, "modo": self.reco_modo, "auto": self.reco_auto, "sel": self.pieza_sel,
                "reco": self.reco, "ensenanza": ens, "evaluacion": self.evaluacion,
                "control": {"maestras": maestras, "ultimos": self._ctrl[1], "cada_h": self.control_cada_h, "hace_h": hace_h,
                            "vencido": bool(maestras) and (hace_h is None or hace_h > self.control_cada_h)},
                "cnn": self.cnn.estado() if self.cnn is not None else None, "min_muestras": 5}

    def _bib(self):
        if self.biblioteca is None:
            raise RuntimeError("La biblioteca de piezas no está disponible")
        return self.biblioteca

    def acc_pieza_guardar(self, d):
        ficha = self._bib().guardar(d)
        self.sis.aviso(f"Pieza guardada: {ficha['nombre']}")
        return {"mensaje": f"Pieza guardada: {ficha['nombre']}", "id": ficha["id"]}

    def acc_pieza_eliminar(self, d):
        bib = self._bib()
        nombre = bib.nombre(d.get("id"))
        if nombre is None:
            return {"ok": False, "error": "La pieza no existe"}
        bib.eliminar(d["id"])
        if self.pieza_sel == d["id"]:
            self.pieza_sel = None
        return {"mensaje": f"Pieza eliminada: {nombre} (y sus muestras de TODAS las cámaras)"}

    def acc_pieza_seleccionar(self, d):
        self.pieza_sel = d.get("id") or None
        self._receta_aplicada = None
        return {"mensaje": "Pieza seleccionada" if self.pieza_sel else "Sin pieza seleccionada"}

    def acc_pieza_receta(self, d):
        self._bib()
        self._aplicar_receta(d["id"])
        return {"mensaje": self._msg()}

    def acc_pieza_ensenar(self, d):
        bib = self._bib()
        pieza = bib.obtener(d.get("id")) if d.get("id") else None
        if pieza is None:
            return {"ok": False, "error": "Elige primero una pieza de la biblioteca (o crea una)"}
        if not self.medicion:
            return {"ok": False, "error": "Enciende el sistema de medición: sin detección no se puede aprender"}
        if self.ensenanza:
            return {"ok": False, "error": "Ya hay un aprendizaje en curso"}
        n = max(3, min(int(d.get("n") or 20), 200))
        intervalo = max(0.1, min(float(d.get("intervalo") or 0.6), 10.0))
        self.ensenanza = {"pid": pieza["id"], "nombre": pieza["nombre"], "total": n, "descs": [], "poses": [], "rechazos": {},
                          "intentos": 0, "intervalo": intervalo, "siguiente": time.time() + 1.5}
        self.ensenanza_fin = None
        return {"mensaje": f"Aprendiendo «{pieza['nombre']}»: {n} muestras. Deja SOLO esa pieza y muévela y gírala entre muestras."}

    def acc_pieza_cancelar(self, d):
        if not self.ensenanza:
            return {"mensaje": "No hay aprendizaje en curso"}
        self._terminar_ensenanza(cancelado=True)
        return {"mensaje": self._msg()}

    def acc_pieza_borrar_muestras(self, d):
        bib = self._bib()
        if bib.obtener(d.get("id")) is None:
            return {"ok": False, "error": "La pieza no existe"}
        bib.borrar_muestras(d["id"], self.cid)
        self.evaluacion = None
        return {"mensaje": "Muestras de esta cámara borradas"}

    def acc_pieza_identificar(self, d):
        bib = self._bib()
        frame = None
        for _ in range(2):
            f = self.sis.leer()
            frame = f if f is not None else frame
        if frame is None:
            return {"ok": False, "error": "No hay imagen"}
        objs = self.sis.detectar(frame, hires=False)
        if not objs:
            self.reco = None
            return {"ok": False, "error": "No se detectó ninguna pieza"}
        obj = max(objs, key=lambda o: o.get("area_px", 0))
        r = self._identificar(frame, obj)
        r["n_objetos"] = len(objs)
        if r["estado"] == "reconocida" and self.sis.escala.calibrada:
            r["veredicto"] = bib.veredicto_para(r["id"], {"largo": obj["largo"], "ancho": obj["ancho"], "tipo": obj.get("tipo", "rect")})
        self.reco = r
        txt = {"reconocida": f"Reconocida: {r['nombre']}", "ambigua": "Ambigua: se parece a más de una pieza",
               "desconocida": "DESCONOCIDA: no se parece a ninguna pieza enseñada", "vacia": "No hay piezas enseñadas con suficientes muestras"}[r["estado"]]
        return {"mensaje": txt}

    def acc_reco_config(self, d):
        if "vivo" in d:
            self.reco_vivo = bool(d["vivo"])
            if not self.reco_vivo:
                self.reco, self._estable = None, (None, 0)
        if d.get("modo") in ("geometrico", "cnn", "ambos"):
            self.reco_modo = d["modo"]
        if "auto" in d:
            self.reco_auto = bool(d["auto"])
            self._receta_aplicada = None
        if d.get("control_cada_h"):
            self.control_cada_h = max(0.1, float(d["control_cada_h"]))
        return {"mensaje": "Ajustes de reconocimiento aplicados"}

    def acc_pieza_evaluar(self, d):
        self.evaluacion = self._bib().evaluar(self.cid)
        self.evaluacion["t"] = time.time()
        return {"mensaje": "Evaluación lista (abajo)", "evaluacion": self.evaluacion}

    def acc_maestra_verificar(self, d):
        bib, sis = self._bib(), self.sis
        pid = d.get("id") or (self.pieza_sel if (self.pieza_sel and (bib.obtener(self.pieza_sel) or {}).get("maestra")) else None)
        if not pid:
            ms = [p for p in bib.listar(self.cid) if p.get("maestra")]
            pid = ms[0]["id"] if len(ms) == 1 else None
        pieza = bib.obtener(pid) if pid else None
        if pieza is None or not pieza.get("maestra"):
            return {"ok": False, "error": "Elige una pieza marcada como MAESTRA en la biblioteca"}
        if not sis.escala.calibrada:
            return {"ok": False, "error": "Primero calibra la escala"}
        frame, res = sis.medir_promedio(int(d.get("frames") or self.args.frames))
        if not res:
            return {"ok": False, "error": "No se detectó ninguna pieza. Pon la pieza maestra sola a la vista."}
        r0 = res[0]
        medidas = {"largo": float(r0["largo"]), "ancho": float(r0["ancho"]), "tipo": r0["obj"]["tipo"] if r0.get("obj") else "rect"}
        try:
            ident = self._identificar(frame, r0["obj"]) if r0.get("obj") else None
        except Exception:  # noqa: BLE001
            ident = None
        if ident and ident["estado"] == "reconocida" and ident["id"] != pid:   # no se ensucia el historial con otra pieza
            return {"ok": False, "error": f"La pieza a la vista parece «{ident['nombre']}», no la maestra «{pieza['nombre']}». No se registró el control."}
        if ident and ident["estado"] in ("desconocida", "ambigua"):
            return {"ok": False, "error": "La pieza a la vista no se reconoce como la maestra (¿es otra pieza, o cambió mucho la luz o la cámara?). No se registró el control."}
        aviso = ""
        ver = bib.veredicto_para(pid, medidas)
        bib.registrar_control(self.cid, pid, medidas, ver)
        self._ctrl = (0.0, [])
        consejo = "" if ver["estado"] == "OK" else " La medida se alejó de la nominal de la maestra: la calibración puede haberse movido (¿se movió la cámara, la luz o el enfoque?). Revisa y recalibra."
        sis.aviso(f"Control con la maestra «{pieza['nombre']}»: {ver['estado']}", 8)
        return {"ok": True, "mensaje": f"Control: {ver['estado']}.{consejo}{aviso}",
                "control": {"pieza": pieza["nombre"], "veredicto": ver, "medidas": medidas}}

    def acc_cnn_encender(self, d):
        if self.cnn is None:
            return {"ok": False, "error": "Sin extractor de red"}
        self.cnn.encender(d.get("modelo") or None)
        self.sis.aviso(f"Red de huellas ENCENDIDA: {self.cnn.modelo}. Las piezas enseñadas ANTES no tienen huella de red: vuelve a enseñarlas.", 10)
        return {"mensaje": self._msg()}

    def acc_cnn_apagar(self, d):
        if self.cnn is not None:
            self.cnn.apagar()
        return {"mensaje": "Red de huellas apagada"}

    def acc_tof_imagen(self, d):
        if not hasattr(self.cam, "alternar"):
            return {"ok": False, "error": "Solo para la cámara ToF"}
        return {"mensaje": "Imagen ToF: " + self.cam.alternar()}
