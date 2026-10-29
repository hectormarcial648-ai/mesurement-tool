"""Servicio de aplicación: sesión de medición de UNA cámara (calibración, detección, medición, precisión)."""
import collections
import cv2
import re
import time

import numpy as np

from medicion.aplicacion.resoluciones import parse_resoluciones
from medicion.dominio.constantes import DIF_ESCALA, RADIO_MIN
from medicion.dominio.dibujo import FONT, texto
from medicion.dominio.escala import Escala
from medicion.dominio.vision import _es_mono, borde_subpixel, buscar_tablero, estimar_fondo, medir_contorno, puntos_tablero


class Sistema:
    def __init__(self, args, cam, cam_id="cam", repo_calibracion=None, repo_mediciones=None, segmentador=None):
        if repo_calibracion is None or repo_mediciones is None or segmentador is None:
            raise ValueError("Sistema necesita repo_calibracion, repo_mediciones y segmentador "
                             "(se inyectan desde medicion.composicion)")
        self.args = args
        self.cam = cam                       # PuertoCamara
        self.cam_id = cam_id
        self.repo = repo_calibracion         # PuertoRepoCalibracion
        self.repo_med = repo_mediciones      # PuertoRepoMediciones
        self.segmentador = segmentador       # PuertoSegmentador
        self.log = collections.deque(maxlen=300)
        self.corr_lin = {"s": 1.0, "e": 0.0, "n": 0, "rms": 0.0}  # corrección contra patrones (en unidades reales)
        self.patrones = []       # [{real:[...], med:[...], tipo, std}] capturados para calcular la corrección
        self._sin_corr = False   # True mientras se capturan patrones (se necesita la medida SIN corregir)
        self.prueba = None       # sesión de prueba de precisión
        self.escala = Escala()
        self.bg_bgr = None
        self.bg_lab = None
        self.maps = None
        self.lente_obj, self.lente_img = [], []
        self.size = None
        self.msg = ("", 0.0)
        self.patron = self._parse_patron(args.board)
        self.holes = bool(args.holes)
        self.K_eff = None      # matriz de la imagen ya corregida (lente calibrada)
        self.pose = None       # {"D", "foot", "aprox"} obtenido del tablero
        self.h_cal = 0.0       # altura del plano donde se calibró la escala (0 = mesa)
        self.mostrar_silueta = True  # con corrección por altura: dibujar también la silueta visible (cian punteada)
        self._fondo_ref = None
        self._fondo_n = 0
        self.proc_scale = max(1, int(getattr(args, "proc_scale", 1)))
        self.presets = parse_resoluciones(getattr(args, "resoluciones", ""))
        inicial = (int(getattr(args, "roi_size", 0) or 0), self.proc_scale)
        if inicial not in self.presets:
            self.presets.insert(0, inicial)
        self.preset_i = self.presets.index(inicial)
        self._calib_t = 0.0       # cuándo se hizo / cargó la calibración actual
        self._hi = None           # imagen de captura completa (solo cámara mono con --proc-scale > 1)
        self._hi_para = None      # fotograma de trabajo al que corresponde _hi
        self._hi_size = None
        self.K_cal = None         # K y distorsión de la lente (a escala de trabajo)
        self.dist_cal = None
        if args.cam_dist_mm and args.altura_mm and args.altura_mm >= args.cam_dist_mm:
            raise SystemExit("--altura-mm debe ser menor que --cam-dist-mm")

    @staticmethod
    def _parse_patron(s):
        try:
            c, r = s.lower().split("x")
            return (int(c), int(r))
        except Exception:
            raise SystemExit("--board debe tener el formato COLSxFILAS (esquinas internas), ej. 9x6")

    def aviso(self, texto_, seg=4.0):
        print(f"[{self.cam_id}] {texto_}")
        self.msg = (texto_, time.time() + seg)
        self.log.append((time.time(), texto_))

    # ----- inicio y persistencia ------------------------------------------ #
    def iniciar(self):
        frame = None
        for i in range(30):
            frame = self.cam.read()
            if frame is not None:
                break
            if i == 1:
                print("[cámara] Esperando el primer fotograma...")
            time.sleep(0.1)
        if frame is None:
            raise RuntimeError("La cámara no entrega imágenes")
        self._fijar_captura(frame, primera=True)

    def _fijar_captura(self, frame, primera=False):
        """Fija tamaños y estado según la captura actual y carga la lente (y calibración) guardadas para ella."""
        self._hi_size = frame.shape[:2]
        trabajo = self._a_trabajo(frame)
        self.size = (trabajo.shape[1], trabajo.shape[0])
        # todo lo que depende de la captura se reinicia (otro tamaño / campo de visión)
        self.maps, self.K_eff, self.K_cal, self.dist_cal = None, None, None, None
        self.escala.limpiar()
        self.pose, self.h_cal = None, 0.0
        self.bg_bgr = self.bg_lab = None
        self._fondo_ref, self._fondo_n = None, 0
        self.lente_obj, self.lente_img = [], []
        self._hi = self._hi_para = None
        self.corr_lin = {"s": 1.0, "e": 0.0, "n": 0, "rms": 0.0}
        self.patrones = []
        if self.proc_scale > 1:
            print(f"[cámara] Captura {frame.shape[1]}x{frame.shape[0]} -> trabajo {self.size[0]}x{self.size[1]} "
                  f"(÷{self.proc_scale}); los bordes se refinan en la captura completa")
            if not _es_mono(frame):
                print("[aviso] El refinado de bordes a resolución completa solo funciona con cámara monocroma; "
                      "aquí --proc-scale solo reduce la imagen.")
        else:
            print(f"[cámara] Resolución de trabajo: {self.size[0]}x{self.size[1]}")
        self._cargar_lente()
        if primera:
            self._cargar_calibracion()

    # ----- archivos de calibración POR CAPTURA ------------------------------ #
    def _slug(self):
        return re.sub(r"[^0-9A-Za-z]+", "_", self._firma()).strip("_")

    @property
    def csv_path(self):
        return self.repo_med.ruta()

    def _cargar_lente(self):
        try:
            d = self.repo.leer_lente(self._slug())
            if d is None:
                return
            if tuple(d["size"]) != self.size:
                print("[lente] Archivo de lente de otra resolución; se ignora.")
                return
            if d.get("firma") and str(d["firma"]) not in (self._firma(), self._firma(False)):
                print("[lente] El archivo de lente es de otra captura (ROI o --proc-scale distintos); se ignora.")
                return
            self._armar_mapas(d["K"], d["dist"], d["newK"])
            print(f"[lente] Corrección de distorsión activa (RMS {float(d['rms']):.3f} px)")
        except Exception as e:
            print(f"[lente] No se pudo cargar: {e}")

    def _armar_mapas(self, K, dist, newK):
        self.K_cal, self.dist_cal = np.array(K, np.float64), np.array(dist, np.float64)
        self.K_eff = np.array(newK, np.float64)
        self.maps = cv2.initUndistortRectifyMap(K, dist, None, newK, self.size, cv2.CV_16SC2)

    def _cargar_calibracion(self):
        try:
            rec = self.repo.leer_calibracion(self._slug())
            if rec is None:
                return
            d, mtime = rec
            if tuple(d.get("image_size", ())) != self.size:
                print("[calibración] Archivo antiguo o de otra resolución; recalibra (h o c).")
                return
            if d.get("firma") and d["firma"] not in (self._firma(), self._firma(False)):
                print("[calibración] Es de otra captura (ROI o --proc-scale distintos); recalibra (h o c).")
                return
            if bool(d.get("undistorted", False)) != (self.maps is not None):
                print("[calibración] No coincide con el estado de la lente; recalibra (h o c).")
                return
            if d.get("homography") is not None:
                self.escala.set_homografia(np.array(d["homography"]))
            elif d.get("px_per_mm"):
                self.escala.set_px_per_mm(d["px_per_mm"], d.get("ref_mm"))
                if self.args.ref_mm is None and d.get("ref_mm"):
                    self.args.ref_mm = float(d["ref_mm"])
            self.h_cal = float(d.get("h_cal", 0.0))
            if d.get("pose"):
                self.pose = d["pose"]
            cl = d.get("corr_lin")
            if cl and cl.get("n", 0) > 0:
                self.corr_lin = {"s": float(cl["s"]), "e": float(cl["e"]), "n": int(cl["n"]), "rms": float(cl.get("rms", 0.0))}
            self._calib_t = mtime
            print(f"[calibración] Cargada: {self.escala.modo}")
        except Exception as e:
            print(f"[calibración] No se pudo cargar: {e}")

    def _guardar_calibracion(self, **extra):
        data = {
            "image_size": list(self.size),
            "firma": self._firma(),
            "undistorted": self.maps is not None,
            "px_per_mm": self.escala.px_per_mm,
            "homography": self.escala.H.tolist() if self.escala.H is not None else None,
            "h_cal": self.h_cal,
            "pose": self.pose,
            "corr_lin": self.corr_lin if self.corr_lin.get("n", 0) > 0 else None,
            "fecha": time.ctime(),
        }
        data.update(extra)
        self.repo.guardar_calibracion(self._slug(), data)
        self._calib_t = time.time()

    def _set_fondo(self, bgr):
        self.bg_bgr = bgr
        if _es_mono(bgr):
            self.bg_lab = cv2.GaussianBlur(bgr.reshape(bgr.shape[:2]), (5, 5), 0)
        else:
            self.bg_lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)

    # ----- captura y detección --------------------------------------------- #
    # ----- cambio de resolución en caliente ---------------------------------- #
    def _nombre_preset(self, i=None):
        roi, sc = self.presets[self.preset_i if i is None else i]
        if not roi:
            roi = getattr(self.cam, "sensor", (0, 0))[0]
        return f"ROI {roi}" + (f", proceso 1/{sc}" if sc > 1 else "")

    def _geom(self):
        """(ox, oy, k): px del sensor = ox + (px de la imagen de trabajo + 0.5) * k - 0.5 (k incluye la reducción)."""
        g = getattr(self.cam, "geometria_sensor", None)
        ox, oy, k = g() if callable(g) else (0.0, 0.0, 1.0)
        return (ox, oy, k * self.proc_scale)

    def _convertir_calibracion(self, prev):
        """Traslada escala / homografía de la captura anterior a la actual (exacto cuando NO hay lente)."""
        if prev is None or prev["lente"] or self.maps is not None:
            return False

        def a_sensor(g):  # px de trabajo -> px del sensor (centros de píxel)
            ox, oy, k = g
            return np.array([[k, 0, 0.5 * k - 0.5 + ox], [0, k, 0.5 * k - 0.5 + oy], [0, 0, 1.0]])

        gA, gB = prev["geom"], self._geom()
        if prev["H"] is not None:
            self.escala.set_homografia(prev["H"] @ np.linalg.inv(a_sensor(gA)) @ a_sensor(gB))
        elif prev["ppm"]:
            self.escala.set_px_per_mm(prev["ppm"] * gA[2] / gB[2], prev["ref_mm"])  # px/mm del sensor es invariante
        else:
            return False
        self.h_cal, self.pose = prev["h_cal"], prev["pose"]
        self._guardar_calibracion(convertida_de=prev["firma"])
        return True

    def aplicar_captura(self, aplicar, revertir, nombre, proc=None):
        """Cambia la captura (ROI / modo) en caliente y deja la calibración coherente. Devuelve True si se aplicó."""
        prev = {"H": None if self.escala.H is None else self.escala.H.copy(), "ppm": self.escala.px_per_mm,
                "ref_mm": self.escala.ref_mm, "geom": self._geom(), "h_cal": self.h_cal, "pose": self.pose,
                "lente": self.maps is not None, "t": self._calib_t, "firma": self._firma()}
        self.aviso(f"Cambiando a {nombre}...", 3)
        try:
            aplicar()
            frame = None
            for _ in range(20):
                frame = self.cam.read()
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError("la cámara no entrega imágenes con esa configuración")
        except Exception as e:  # noqa: BLE001
            self.aviso(f"No se pudo cambiar la resolución: {e}", 8)
            try:  # volver a la anterior para no dejar la cámara en un estado raro
                revertir()
            except Exception:  # noqa: BLE001
                pass
            return False
        if proc is not None:
            self.proc_scale = proc
        self._fijar_captura(frame)
        t_b = self.repo.mtime_calibracion(self._slug())
        puede = (prev["H"] is not None or prev["ppm"]) and not prev["lente"] and self.maps is None
        convertida = False
        if puede and prev["t"] > t_b:  # la calibración que traía es más reciente que la guardada para esta captura
            convertida = self._convertir_calibracion(prev)
        if not convertida:
            self._cargar_calibracion()
        if not convertida and not self.escala.calibrada and puede:
            convertida = self._convertir_calibracion(prev)
        if convertida:
            estado = "calibración CONVERTIDA de la resolución anterior"
        elif self.escala.calibrada:
            estado = "calibración guardada para esta resolución"
        else:
            estado = "SIN calibrar: usa la pestaña Calibrar"
        nota = ""
        if prev["lente"] and self.maps is None:
            nota = " | la lente calibrada era de otra captura: calíbrala aquí"
        self.aviso(f"{nombre} | trabajo {self.size[0]}x{self.size[1]} | {estado}{nota}", 9)
        return True

    def aplicar_resolucion_gige(self, nuevo):
        """Pasa al preajuste `nuevo` (índice de self.presets) de la cámara GigE."""
        roi, sc = self.presets[nuevo]
        antes = self.presets[self.preset_i][0]
        ok = self.aplicar_captura(lambda: self.cam.set_roi(roi), lambda: self.cam.set_roi(antes),
                                  self._nombre_preset(nuevo), sc)
        if ok:
            self.preset_i = nuevo
        return ok

    def cambiar_resolucion(self, delta=1):
        if not hasattr(self.cam, "set_roi"):
            self.aviso("El cambio de resolución por preajustes solo existe en la cámara GigE")
            return False
        if len(self.presets) < 2:
            self.aviso("Solo hay una resolución: agrega más en --resoluciones (p. ej. 1280:1,2560:2,5120:4)")
            return False
        return self.aplicar_resolucion_gige((self.preset_i + delta) % len(self.presets))

    def cambiar_modo_csi(self, w, h):
        """Cambia el modo nativo del sensor de la cámara CSI."""
        antes = tuple(self.cam.size)
        return self.aplicar_captura(lambda: self.cam.set_modo(w, h), lambda: self.cam.set_modo(*antes), f"{w}x{h}")

    def _a_trabajo(self, f):
        """Reduce la captura N veces (promedio de bloques NxN: relación exacta con la captura)."""
        s = self.proc_scale
        if s <= 1:
            return f
        h, w = f.shape[:2]
        hh, ww = h // s, w // s
        return cv2.resize(f[:hh * s, :ww * s], (ww, hh), interpolation=cv2.INTER_AREA)

    def _firma(self, con_cam=True):
        """Identifica la captura (cámara, tamaño, escala y ROI): evita cargar una calibración de otro campo de visión."""
        info = getattr(self.cam, "info_captura", None)
        extra = info() if callable(info) else ""
        base = f"{self._hi_size[1]}x{self._hi_size[0]}/s{self.proc_scale}/{extra}"
        return f"{self.cam_id}/{base}" if con_cam else base

    def leer(self):
        f = self.cam.read()
        if f is None:
            return None
        if f.ndim == 3 and f.shape[2] == 4:
            f = np.ascontiguousarray(f[:, :, :3])
        hi = f
        f = self._a_trabajo(f)
        if self.maps is not None:
            f = cv2.remap(f, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
        self._hi = hi if (self.proc_scale > 1 and _es_mono(hi)) else None
        self._hi_para = f
        return f

    # ----- bordes a resolución completa -------------------------------------- #
    @staticmethod
    def _K_hi(K, s):
        """K de la imagen de trabajo -> K de la captura completa (centros de píxel)."""
        K = np.array(K, np.float64).copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] = s * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = s * (K[1, 2] + 0.5) - 0.5
        return K

    def _borde_hires(self, cnt, hueco, bg_lo, radio_lo):
        """Borde subpíxel en la captura completa (cámara mono con --proc-scale > 1).

        El contorno se halló a escala reducida; aquí se recorta una ventana alrededor de la pieza en
        resolución completa (con la lente corregida solo en ese recorte), se resta el fondo (el modelo
        suave se interpola) y se ubica el borde en el perfil de intensidad. Devuelve puntos en las
        coordenadas de la imagen de trabajo, o None si no se puede.
        """
        s, hi = self.proc_scale, self._hi
        pts = cnt.reshape(-1, 2).astype(np.float64)
        m = int(2 * radio_lo + 6)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        W, H = self.size
        cx0, cy0 = max(int(np.floor(x0)) - m, 0), max(int(np.floor(y0)) - m, 0)
        cx1, cy1 = min(int(np.ceil(x1)) + m + 1, W), min(int(np.ceil(y1)) + m + 1, H)
        cw, ch = cx1 - cx0, cy1 - cy0
        if cw < 8 or ch < 8 or cw * ch * s * s > 14e6:
            return None  # recorte inválido o demasiado grande para la Pi
        if self.maps is None:
            crop = hi[cy0 * s:cy1 * s, cx0 * s:cx1 * s]
        else:
            if self.K_cal is None or self.dist_cal is None or self.K_eff is None:
                return None
            newK = self._K_hi(self.K_eff, s)
            newK[0, 2] -= cx0 * s
            newK[1, 2] -= cy0 * s
            mx, my = cv2.initUndistortRectifyMap(self._K_hi(self.K_cal, s), self.dist_cal, None, newK,
                                                 (cw * s, ch * s), cv2.CV_32FC1)
            crop = cv2.remap(hi, mx, my, cv2.INTER_LINEAR)
        bg = cv2.resize(bg_lo[cy0:cy1, cx0:cx1].astype(np.float32), (cw * s, ch * s), interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(crop, (0, 0), float(np.clip(0.5 * s, 1.0, 3.0)))
        dist = np.abs(g.astype(np.float32) - bg) * DIF_ESCALA
        cnt_hi = ((pts + 0.5) * s - 0.5 - np.array([cx0 * s, cy0 * s])).reshape(-1, 1, 2)
        sub = borde_subpixel(dist, cnt_hi, hueco, radio=radio_lo * s, nmuest=int(2 * radio_lo * s) + 1)
        if sub is None:
            return None
        return (sub + np.array([cx0 * s, cy0 * s]) + 0.5) / s - 0.5

    # ----- geometría de la altura ------------------------------------------ #
    def _intrinsecas(self):
        """(K 3x3 o None, punto principal en px, ¿aproximada?)."""
        w, h = self.size
        if self.K_eff is not None:
            K = self.K_eff
            return K, (float(K[0, 2]), float(K[1, 2])), False
        if self.args.hfov_deg:
            f = (w / 2) / np.tan(np.deg2rad(self.args.hfov_deg) / 2)
            return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64), (w / 2, h / 2), True
        return None, (w / 2, h / 2), False

    def dist_camara(self):
        """(D en mm, fuente): distancia del lente al plano de la mesa."""
        if self.args.cam_dist_mm:
            return float(self.args.cam_dist_mm), "manual"
        if self.pose:
            return float(self.pose["D"]), "tablero" + ("~" if self.pose.get("aprox") else "")
        return None, None

    def _foot(self):
        """Pie de la cámara sobre el plano (origen de la ampliación), en las unidades actuales."""
        _, pp, _ = self._intrinsecas()
        if self.escala.H is not None and self.pose and self.pose.get("foot") is not None:
            return np.array(self.pose["foot"], np.float64)
        return self.escala.a_unid(pp)[0]

    def _corr(self, ctx="med"):
        """Parámetros de la corrección de altura, o None si no hay nada que corregir / falta D."""
        D, _ = self.dist_camara()
        if ctx == "ref":  # pieza de referencia: su cara superior es el plano de calibración
            hp = hc = float(self.args.ref_alto_mm or 0.0)
        else:
            hp, hc = float(self.args.altura_mm or 0.0), float(self.h_cal)
        if (hp <= 0 and hc <= 0) or D is None or D <= max(hp, hc) + 1.0:
            return None
        m_t, m_b = (D - hc) / (D - hp), (D - hc) / D
        if abs(m_t - 1) < 1e-6 and abs(m_b - 1) < 1e-6:
            return None
        return {"foot": self._foot(), "m_t": m_t, "m_b": m_b}

    def efecto_altura_pct(self):
        """% que la corrección por altura encoge (+) o agranda (-) el contorno respecto a la silueta visible; None si no actúa."""
        c = self._corr("med")
        return None if c is None else round(100.0 * (1.0 - 1.0 / c["m_t"]), 1)

    def detectar(self, frame, ctx="med", fondo="auto", hires=True):
        a = self.args
        ref = None
        if self.bg_lab is None:  # fondo adaptativo (se reajusta cada --fondo-cada fotogramas)
            if fondo == "fresco" or self._fondo_ref is None or self._fondo_n % max(1, a.fondo_cada) == 0:
                self._fondo_ref = estimar_fondo(frame)
            self._fondo_n += 1
            ref = self._fondo_ref
        bw, _, dist = self.segmentador.segmentar(frame, self.bg_lab, ref, a.min_diff, a.thr, a.erode)
        corr = self._corr(ctx)
        radio = float(np.clip(max(frame.shape[:2]) / 250.0, RADIO_MIN, 16.0))  # ventana del borde subpíxel
        bg_lo = self.bg_lab if self.bg_lab is not None else ref
        usar_hi = bool(hires and self._hi is not None and frame is self._hi_para and bg_lo is not None
                       and getattr(self.segmentador, "ultimo_clasico", True))
        modo = cv2.RETR_CCOMP if self.holes else cv2.RETR_EXTERNAL
        cnts, hier = cv2.findContours(bw, modo, cv2.CHAIN_APPROX_NONE)
        hier = hier[0] if hier is not None else None
        h_img, w_img = bw.shape
        objs = []
        for i, c in enumerate(cnts):
            es_hueco = bool(self.holes and hier is not None and hier[i][3] != -1)
            area = cv2.contourArea(c)
            if area < a.min_area or area > 0.8 * h_img * w_img:
                continue
            if not es_hueco:
                x, y, bw_, bh_ = cv2.boundingRect(c)
                if x <= 2 or y <= 2 or x + bw_ >= w_img - 2 or y + bh_ >= h_img - 2:
                    continue  # objeto cortado por el borde
            sub_hi = self._borde_hires(c, es_hueco, bg_lo, radio) if usar_hi else None
            o = medir_contorno(c, self.escala, es_hueco, a.circ_tol, not a.no_circles, dist,
                               None if es_hueco else corr, radio, sub_hi)
            if o is None:
                continue
            if o["ancho_px"] < a.min_side_px:
                continue  # tiras finas = ruido / restos de fondo, no piezas
            self._aplicar_corr_lineal(o)
            objs.append(o)
        objs.sort(key=lambda o: o["area_px"], reverse=True)
        return objs

    # ----- acciones --------------------------------------------------------- #
    def capturar_fondo(self, n=10):
        acc = None
        cont = 0
        for _ in range(n):
            f = self.leer()
            if f is None:
                continue
            acc = f.astype(np.float32) if acc is None else acc + f.astype(np.float32)
            cont += 1
        if cont == 0:
            self.aviso("No se pudo capturar el fondo")
            return
        bg = (acc / cont).astype(np.uint8)
        # Seguridad: si ya hay un objeto en la mesa, ese fondo causaría detecciones fantasma
        bw, _, _ = self.segmentador.segmentar(bg, None, None, self.args.min_diff, self.args.thr, 0)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hh, ww = bw.shape
        for c in cnts:
            x, y, w_, h_ = cv2.boundingRect(c)
            if cv2.contourArea(c) >= self.args.min_area and x > 2 and y > 2 and x + w_ < ww - 2 and y + h_ < hh - 2:
                self.aviso("Hay un objeto en la mesa: quítalo y pulsa b otra vez (fondo NO guardado)", 8)
                return
        self._set_fondo(bg)
        self.aviso("Fondo vacío capturado (solo para esta sesión)")

    def olvidar_fondo(self):
        self.bg_bgr = self.bg_lab = None
        self._fondo_ref = None
        self.aviso("Fondo automático (color de la mesa)")

    def calibrar_ref(self, n=10):
        if not self.args.ref_mm:
            self.aviso("Falta --ref-mm para calibrar con pieza")
            return
        esc_previa = self.escala
        self.escala = Escala()  # modo píxeles: la corrección de altura de la referencia se hace en px
        largos, cortos = [], []
        try:
            for i in range(n):
                f = self.leer()
                if f is None:
                    continue
                objs = self.detectar(f, ctx="ref", fondo="fresco" if i == 0 else "auto")
                if objs:
                    largos.append(objs[0]["largo"])
                    cortos.append(objs[0]["ancho"])
        finally:
            self.escala = esc_previa
        if len(largos) < max(3, n // 2):
            self.aviso("No se detectó la pieza de referencia de forma estable")
            return
        px = float(np.median(largos)) / self.args.ref_mm
        self.escala.set_px_per_mm(px, self.args.ref_mm)
        self.h_cal = float(self.args.ref_alto_mm or 0.0)
        self.reset_corr_lineal(guardar=False)
        self._guardar_calibracion(ref_mm=self.args.ref_mm)
        lado2 = float(np.median(cortos)) / px
        nota = ""
        if self.h_cal > 0:
            nota = " Altura de la referencia corregida." if self.dist_camara()[0] else \
                " OJO: falta la distancia lente-mesa (tecla d), la referencia no se pudo corregir por altura."
        self.aviso(
            f"Calibrado: lado largo = {self.args.ref_mm:g} mm (el corto sale {lado2:.1f} mm). "
            f"Si tu pieza no mide {self.args.ref_mm:g} mm, repite con --ref-mm correcto.{nota}", 10)

    def fijar_ref(self, valor):
        """Cambia la medida real de la pieza de referencia y recalibra si la ve."""
        if valor is None or valor <= 0:
            self.aviso("Valor inválido: escribe un número mayor que 0")
            return
        self.args.ref_mm = float(valor)
        f = self.leer()
        if f is not None and self.detectar(f):
            self.calibrar_ref()
        else:
            self.aviso(f"Referencia = {valor:g} mm. Pon la pieza y pulsa c", 6)

    def fijar_altura(self, valor):
        """Altura (grosor) de la pieza a medir, en mm. 0 = pieza plana."""
        if valor is None or valor < 0:
            self.aviso("Valor inválido: escribe un número (0 = pieza plana)")
            return
        D, fuente = self.dist_camara()
        if D is not None and valor >= D:
            self.aviso(f"La altura debe ser menor que la distancia lente-mesa ({D:.0f} mm)")
            return
        self.args.altura_mm = float(valor)
        if valor > 0 and D is None:
            self.aviso(f"Altura = {valor:g} mm, pero falta la distancia lente-mesa (tecla d) para corregir", 8)
        else:
            self.aviso(f"Altura de la pieza = {valor:g} mm" + (f" (D = {D:.0f} mm, {fuente})" if D else ""))

    def fijar_dist(self, valor):
        """Distancia del lente al plano de la mesa, en mm."""
        if valor is None or valor <= 0 or valor <= (self.args.altura_mm or 0) + 1:
            self.aviso("Valor inválido: debe ser mayor que la altura de la pieza")
            return
        self.args.cam_dist_mm = float(valor)
        self.aviso(f"Distancia lente-mesa = {valor:g} mm (manual)")

    def calibrar_plano(self):
        f = self.leer()
        if f is None:
            return
        self.aviso("Buscando tablero...", 2)
        c = buscar_tablero(f, self.patron)
        if c is None:
            self.aviso(f"No se encontró el tablero {self.patron[0]}x{self.patron[1]} (esquinas internas)")
            return
        obj = puntos_tablero(self.patron, self.args.square_mm)
        H, _ = cv2.findHomography(c, obj, 0)
        if H is None:
            self.aviso("No se pudo calcular la homografía")
            return
        self.escala.set_homografia(H)
        self.h_cal = 0.0  # el tablero va sobre la mesa
        proy = self.escala.a_unid(c)
        rms = float(np.sqrt(np.mean(np.sum((proy - obj) ** 2, axis=1))))
        # Pose de la cámara respecto a la mesa -> distancia lente-mesa y pie de la cámara
        self.pose = None
        K, _, aprox = self._intrinsecas()
        if K is not None:
            obj3 = np.zeros((len(c), 3), np.float64)
            obj3[:, :2] = obj
            pts = c.astype(np.float64).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(obj3, pts, K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(obj3, pts, K, None)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                C = -R.T @ tvec.ravel()
                self.pose = {"D": float(abs(C[2])), "foot": [float(C[0]), float(C[1])], "aprox": bool(aprox)}
        self.reset_corr_lineal(guardar=False)
        self._guardar_calibracion(board=f"{self.patron[0]}x{self.patron[1]}", square_mm=self.args.square_mm, rms_mm=rms)
        extra = "  (alto: revisa tablero plano / lente)" if rms > 0.3 else ""
        dtxt = ""
        if self.pose:
            dtxt = f" Distancia lente-mesa: {self.pose['D']:.0f} mm" + (" (aprox., con --hfov-deg)" if self.pose["aprox"] else "")
        else:
            dtxt = " (sin distancia lente-mesa: calibra la lente k/l, usa --hfov-deg o ponla con d)"
        self.aviso(f"Plano calibrado con tablero. Error RMS: {rms:.3f} mm{extra}.{dtxt}", 8)

    def capturar_pose_lente(self):
        f = self.cam.read()  # sin corregir: la lente se calcula sobre la imagen cruda
        if f is None:
            return
        if f.ndim == 3 and f.shape[2] == 4:
            f = np.ascontiguousarray(f[:, :, :3])
        f = self._a_trabajo(f)
        if self.maps is not None:
            nl, nc = self.repo.nombres(self._slug())
            self.aviso(f"Ya hay lente calibrada. Borra {nl} y {nc} para rehacerla.")
            return
        c = buscar_tablero(f, self.patron)
        if c is None:
            self.aviso("Tablero no encontrado: debe verse COMPLETO")
            return
        if self.lente_img:
            mov = np.mean(np.linalg.norm(c - self.lente_img[-1].reshape(-1, 2), axis=1))
            if mov < 20:
                self.aviso("Mueve/inclina el tablero antes de capturar otra pose")
                return
        obj3 = np.zeros((len(c), 3), np.float32)
        obj3[:, :2] = puntos_tablero(self.patron, self.args.square_mm)
        self.lente_obj.append(obj3)
        self.lente_img.append(c.reshape(-1, 1, 2).astype(np.float32))
        n = len(self.lente_img)
        self.aviso(f"Pose de lente {n}/{self.args.lens_poses}")
        if n >= self.args.lens_poses:
            self.finalizar_lente()

    def finalizar_lente(self):
        n = len(self.lente_img)
        if n < 8:
            self.aviso(f"Faltan poses ({n}/8 mínimo)")
            return
        # Modelo reducido (k1, k2; sin tangencial ni k3): estable con poses limitadas
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
        rms, K, dist, _, _ = cv2.calibrateCamera(
            self.lente_obj, self.lente_img, self.size, None, None, flags=flags
        )
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, self.size, 0, self.size)
        self.repo.guardar_lente(self._slug(), {"K": K, "dist": dist, "newK": newK, "size": tuple(self.size),
                                               "rms": float(rms), "firma": self._firma()})
        self._armar_mapas(K, dist, newK)
        # Todo lo anterior queda invalidado (cambió la geometría de la imagen)
        self.escala.limpiar()
        self.pose = None
        self.h_cal = 0.0
        self.repo.borrar_calibracion(self._slug())  # la calibración anterior ya no sirve
        self.bg_bgr = self.bg_lab = None
        self.lente_obj, self.lente_img = [], []
        extra = "  (alto: toma poses más inclinadas y variadas)" if rms > 1.0 else ""
        self.aviso(f"Lente calibrada (RMS {rms:.3f} px){extra}. Repite h/c y b.", 8)

    def medir_promedio(self, n=None):
        n = n or self.args.frames
        refs, series = [], []
        ult_frame, ult_objs = None, []
        for i in range(n):
            f = self.leer()
            if f is None:
                continue
            objs = self.detectar(f, fondo="fresco" if i == 0 else "auto")
            ult_frame, ult_objs = f, objs
            if not refs:
                if not objs:
                    continue
                refs = [o["centro"] for o in objs]
                series = [[] for _ in objs]
            for o in objs:
                d = [np.hypot(o["centro"][0] - r[0], o["centro"][1] - r[1]) for r in refs]
                j = int(np.argmin(d))
                if d[j] < 40:
                    series[j].append((o["largo"], o["ancho"]))
        res = []
        for j, s in enumerate(series):
            if len(s) < max(3, n // 3):
                continue
            a = np.array(s)
            d = [np.hypot(o["centro"][0] - refs[j][0], o["centro"][1] - refs[j][1]) for o in ult_objs]
            obj = ult_objs[int(np.argmin(d))] if d and min(d) < 40 else None
            res.append(
                {
                    "largo": float(np.median(a[:, 0])),
                    "ancho": float(np.median(a[:, 1])),
                    "std_l": float(np.std(a[:, 0])),
                    "std_a": float(np.std(a[:, 1])),
                    "n": len(s),
                    "obj": obj,
                }
            )
        return ult_frame, res

    def registrar_csv(self, res):
        ahora = time.strftime("%Y-%m-%d %H:%M:%S")
        filas = []
        for i, r in enumerate(res, 1):
            tipo = r["obj"]["tipo"] if r.get("obj") else "rect"
            filas.append([ahora, i, f"{r['largo']:.3f}", f"{r['ancho']:.3f}", self.escala.unidad,
                          f"{r['std_l']:.3f}", f"{r['std_a']:.3f}", r["n"], self.escala.modo, tipo,
                          *self._csv_altura()])
        self.repo_med.registrar(filas)

    def _csv_altura(self):
        D, _ = self.dist_camara()
        return [f"{self.args.altura_mm or 0:g}", f"{D:.1f}" if D else ""]

    def distancia_puntos(self, a, b):
        """Distancia entre dos puntos de imagen (px) en unidades reales. -> (d, dx, dy)"""
        u = self.escala.a_unid([a, b])
        corr = self._corr("med") if self.escala.calibrada else None
        k = 1.0 / corr["m_t"] if corr else 1.0  # puntos sobre la cara superior de la pieza
        dx, dy = (u[1] - u[0]) * k
        return float(np.hypot(dx, dy)), float(dx), float(dy)

    def registrar_regla(self, n, d):
        self._escribir_csv([[time.strftime("%Y-%m-%d %H:%M:%S"), f"reg-{n}", f"{d:.3f}", "",
                             self.escala.unidad, "", "", 1, self.escala.modo, "punto-punto",
                             *self._csv_altura()]])

    def verificar(self):
        if not self.args.check_mm:
            return None, "Falta --check-mm LARGO ANCHO para verificar"
        frame, res = self.medir_promedio()
        if not res:
            return frame, "No se detectó la pieza"
        r = res[0]
        el, ea = sorted(self.args.check_mm, reverse=True)
        msg = (
            f"Verif: {r['largo']:.2f}x{r['ancho']:.2f} vs {el:.2f}x{ea:.2f} {self.escala.unidad} | "
            f"error {r['largo'] - el:+.2f} / {r['ancho'] - ea:+.2f}"
        )
        return (frame, res), msg

    # ----- dibujo ----------------------------------------------------------- #
    def anotar(self, frame, items, pie=True, cabecera=True):
        out = cv2.cvtColor(frame.reshape(frame.shape[:2]), cv2.COLOR_GRAY2BGR) if _es_mono(frame) else frame.copy()
        fs = max(0.7, out.shape[1] / 1800.0)
        th = max(2, int(fs * 2))
        u = self.escala.unidad
        for i, it in enumerate(items, 1):
            o = it.get("obj")
            if o is None:
                continue
            if self.mostrar_silueta and o.get("alt_corr") and o.get("sil_px") is not None:  # lo que se VE, antes de corregir la altura
                _poli_punteado(out, o["sil_px"], (255, 255, 0), th, max(8, int(12 * fs)))
            cv2.polylines(out, [np.round(o["corners_px"]).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), th)
            tipo = o.get("tipo", "rect")
            if tipo == "circulo":
                etiqueta = f"#{i} D={it['largo']:.2f} {u}"
            elif tipo == "hueco":
                etiqueta = f"#{i} hueco D={it['largo']:.2f} {u}"
            else:
                etiqueta = f"#{i} {it['largo']:.2f} x {it['ancho']:.2f} {u}"
            if o.get("alt_corr"):
                etiqueta += " (h)"  # huella corregida por altura
            (tw, _), _ = cv2.getTextSize(etiqueta, FONT, fs, th)
            cx, cy = int(o["centro"][0]), int(o["centro"][1])
            if tipo != "rect":
                cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, int(18 * fs), th)
                cy += int(28 * fs)
            texto(out, etiqueta, (cx - tw // 2, cy), fs, th=th)
        if cabecera:
            bg = "capturado (sesión)" if self.bg_lab is not None else "automático"
            lente = "ok" if self.maps is not None else "no"
            proc = f" | Proceso: 1/{self.proc_scale}" if self.proc_scale > 1 else ""
            if hasattr(self.cam, "set_roi"):
                proc = f" | {self._nombre_preset()}"
            texto(out, f"Escala: {self.escala.modo} | Fondo: {bg} | Lente: {lente}{proc}", (10, int(30 * fs)), fs * 0.85, (0, 0, 255), th)
            if self.msg[1] > time.time():
                texto(out, self.msg[0], (10, int(65 * fs)), fs * 0.85, (0, 200, 255), th)
            hp = float(self.args.altura_mm or 0.0)
            if hp > 0 or self.h_cal > 0:
                D, fuente = self.dist_camara()
                if D:
                    txt, col = f"Altura: pieza {hp:g} mm | lente-mesa {D:.0f} mm ({fuente}) | corregida", (0, 200, 0)
                else:
                    txt, col = f"Altura: pieza {hp:g} mm SIN CORREGIR (falta distancia lente-mesa: tecla d)", (0, 0, 255)
                texto(out, txt, (10, int(100 * fs)), fs * 0.85, col, th)
        if pie:
            texto(out, "b fondo | c pieza | r ref | a altura | d dist | p regla | o huecos | z resol | h tablero | k/l lente | m medir | v verif | s guardar | q salir",
                  (10, out.shape[0] - 15), fs * 0.7, (255, 255, 255), max(1, th - 1))
        return out

    # ----- API para la interfaz web --------------------------------------------- #
    def set_tablero(self, cols, filas, lado_mm):
        if cols < 3 or filas < 3 or lado_mm <= 0:
            raise ValueError("Datos del tablero inválidos")
        self.patron = (int(cols), int(filas))
        self.args.board = f"{int(cols)}x{int(filas)}"
        self.args.square_mm = float(lado_mm)

    def set_check(self, largo, ancho):
        self.args.check_mm = [float(largo), float(ancho)]

    def agregar_preset(self, roi, sc):
        roi, sc = int(roi), int(sc)
        if roi < 0 or sc < 1:
            raise ValueError("El ROI debe ser >= 0 y la reducción >= 1")
        if (roi, sc) not in self.presets:
            self.presets.append((roi, sc))
        return self.presets.index((roi, sc))

    # ===== PRECISIÓN: corrección contra patrones, distancia lente-mesa y prueba de precisión ===== #
    def _aplicar_corr_lineal(self, o):
        """medido = s*real + e para dimensiones EXTERNAS; en huecos el desfase de borde tiene el signo contrario."""
        c = self.corr_lin
        if self._sin_corr or c["n"] == 0 or not self.escala.calibrada:
            return
        sg = -1.0 if o.get("tipo") == "hueco" else 1.0
        for k in ("largo", "ancho"):
            o[k] = (o[k] - sg * c["e"]) / c["s"]

    def reset_corr_lineal(self, guardar=True):
        self.corr_lin = {"s": 1.0, "e": 0.0, "n": 0, "rms": 0.0}
        if guardar and self.escala.calibrada:
            self._guardar_calibracion()

    def capturar_patron(self, real_l, real_a=None, n=None):
        """Mide (SIN corrección) una pieza de medida real conocida y la guarda para el ajuste."""
        if not self.escala.calibrada:
            self.aviso("Primero calibra la escala (pieza o tablero)")
            return False
        real_l = float(real_l)
        if real_a:
            real_a = float(real_a)
            if real_a > real_l:
                real_l, real_a = real_a, real_l
        self._sin_corr = True
        try:
            _, res = self.medir_promedio(n or self.args.frames)
        finally:
            self._sin_corr = False
        if not res:
            self.aviso("No se detectó ninguna pieza: revisa que esté sola y bien iluminada", 6)
            return False
        r = res[0]
        tipo = r["obj"]["tipo"] if r.get("obj") else "rect"
        pat = {"real": [real_l], "med": [r["largo"]], "tipo": tipo, "std": [r["std_l"]]}
        if real_a and tipo == "rect":
            pat["real"].append(real_a)
            pat["med"].append(r["ancho"])
            pat["std"].append(r["std_a"])
        self.patrones.append(pat)
        dif = ", ".join(f"{m - t:+.3f}" for m, t in zip(pat["med"], pat["real"]))
        self.aviso(f"Patrón {len(self.patrones)} capturado: real {pat['real']} mm, medido {[round(m, 3) for m in pat['med']]} "
                   f"(error sin corregir {dif} mm)", 8)
        return True

    def ajustar_corr_lineal(self):
        """Ajusta medido = s*real + e con los patrones capturados y la deja activa."""
        real = np.array([v for p in self.patrones for v in p["real"]], float)
        med = np.array([v for p in self.patrones for v in p["med"]], float)
        if len(real) < 2:
            self.aviso("Captura al menos 2 medidas de patrón (una pieza rectangular da 2: largo y ancho)", 7)
            return False
        if np.ptp(real) >= 3.0:  # hay variedad de tamaños: se ajustan escala Y desfase
            s_, e_ = np.polyfit(real, med, 1)
            modo = "escala + desfase"
        else:  # tamaños casi iguales: no se puede separar la escala; solo desfase
            s_, e_ = 1.0, float(np.mean(med - real))
            modo = "solo desfase (usa patrones de tamaños distintos para corregir también la escala)"
        if abs(s_ - 1.0) > 0.05 or abs(e_) > 2.0:
            self.aviso(f"Resultado dudoso (escala {s_:.4f}, desfase {e_:+.3f} mm): revisa las medidas reales escritas "
                       "y la calibración de escala. No se aplicó.", 10)
            return False
        resid = med - (s_ * real + e_)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        self.corr_lin = {"s": float(s_), "e": float(e_), "n": int(len(real)), "rms": rms}
        self._guardar_calibracion()
        antes = float(np.sqrt(np.mean((med - real) ** 2)))
        self.aviso(f"Corrección activa ({modo}): escala {s_:.5f}, desfase de borde {e_:+.3f} mm. "
                   f"Error típico de los patrones: {antes:.3f} -> {rms:.3f} mm", 12)
        return True

    def calibrar_distancia(self, real_l, real_a=None, altura_mm=None, n=5):
        """Halla la distancia lente-mesa D que hace que una pieza de medida y altura conocidas salga con su medida real."""
        if not self.escala.calibrada:
            self.aviso("Primero calibra la escala (pieza o tablero)")
            return None
        if altura_mm is not None:
            self.args.altura_mm = float(altura_mm)
        h = float(self.args.altura_mm or 0.0)
        if h <= 0:
            self.aviso("Indica la altura de la pieza patrón (mayor que 0)")
            return None
        real_l = float(real_l)
        real_a = float(real_a) if real_a else None
        frame = None
        for _ in range(max(1, int(n))):
            f = self.leer()
            if f is not None:
                frame = f
        if frame is None:
            self.aviso("No hay imagen")
            return None
        previo = self.args.cam_dist_mm

        def err(D):
            self.args.cam_dist_mm = float(D)
            objs = self.detectar(frame, fondo="auto")
            if not objs:
                return None
            o = objs[0]
            e = [o["largo"] - real_l]
            if real_a and o.get("tipo") == "rect":
                e.append(o["ancho"] - real_a)
            return float(np.mean(e))

        try:
            lo, hi = h + 5.0, 5000.0
            elo, ehi = err(lo), err(hi)
            if elo is None or ehi is None:
                self.args.cam_dist_mm = previo
                self.aviso("No se detectó la pieza patrón", 6)
                return None
            if elo * ehi > 0:
                self.args.cam_dist_mm = previo
                self.aviso("No hay ninguna distancia que haga coincidir la medida: revisa la medida real, la altura "
                           "y la calibración de escala", 9)
                return None
            D = 0.5 * (lo + hi)
            for _ in range(40):
                D = 0.5 * (lo + hi)
                e = err(D)
                if e is None:
                    break
                if abs(e) < 0.003 or (hi - lo) < 0.05:
                    break
                if e * elo > 0:
                    lo, elo = D, e
                else:
                    hi = D
            self.args.cam_dist_mm = float(D)
        except Exception:
            self.args.cam_dist_mm = previo
            raise
        otra = f" (el tablero daba ~{self.pose['D']:.0f} mm)" if self.pose and self.pose.get("D") else ""
        self.aviso(f"Distancia lente-mesa calibrada = {D:.1f} mm{otra}", 9)
        return float(D)

    # ----- prueba de precisión (repetibilidad y sesgo) ------------------------------ #
    def prueba_iniciar(self, real_l, real_a=None, tol=None):
        real_l = float(real_l)
        real_a = float(real_a) if real_a else None
        if real_a and real_a > real_l:
            real_l, real_a = real_a, real_l
        self.prueba = {"real_l": real_l, "real_a": real_a, "tol": float(tol) if tol else None, "m": []}
        self.aviso("Prueba de precisión iniciada: usa Medir varias veces (recoloca la pieza entre mediciones)", 8)

    def prueba_registrar(self, r):
        if not self.prueba:
            return
        self.prueba["m"].append({"l": float(r["largo"]), "a": float(r["ancho"]), "t": time.time()})

    def prueba_stats(self):
        pr = self.prueba
        if not pr:
            return None
        out = {"activa": True, "real_l": pr["real_l"], "real_a": pr["real_a"], "tol": pr["tol"], "n": len(pr["m"]),
               "ultimas": [{"l": m["l"], "a": m["a"]} for m in pr["m"][-12:]]}
        if len(pr["m"]) >= 1:
            def est(vals, real):
                v = np.array(vals, float)
                sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                return {"media": float(v.mean()), "sd": sd, "sesgo": float(v.mean() - real),
                        "max_abs": float(np.max(np.abs(v - real))), "rango": float(np.ptp(v))}
            out["largo"] = est([m["l"] for m in pr["m"]], pr["real_l"])
            if pr["real_a"]:
                out["ancho"] = est([m["a"] for m in pr["m"]], pr["real_a"])
            if pr["tol"] and len(pr["m"]) >= 5:
                for k in ("largo", "ancho"):
                    if k in out:
                        e = out[k]
                        e["pt"] = 6.0 * e["sd"] / (2.0 * pr["tol"]) * 100.0          # repetibilidad vs banda de tolerancia
                        e["err_esperado"] = abs(e["sesgo"]) + 3.0 * e["sd"]         # sesgo + 3 sigma
                        e["dentro"] = bool(e["err_esperado"] <= pr["tol"])
                        e["veredicto"] = "bueno" if e["pt"] < 10 else ("aceptable con vigilancia" if e["pt"] <= 30 else "no aceptable")
        return out

    def estado_dict(self):
        D, fuente = self.dist_camara()
        a = self.args
        return {
            "corr_lin": {"activa": self.corr_lin["n"] > 0, "s": self.corr_lin["s"], "e": self.corr_lin["e"],
                         "n": self.corr_lin["n"], "rms": self.corr_lin["rms"]},
            "patrones": [{"real": p["real"], "med": p["med"], "tipo": p["tipo"]} for p in self.patrones],
            "prueba": self.prueba_stats(),
            "segmentador": self.segmentador.nombre,
            "size": list(self.size) if self.size else None,
            "captura": [self._hi_size[1], self._hi_size[0]] if self._hi_size else None,
            "proc_scale": self.proc_scale,
            "escala": {"modo": self.escala.modo, "unidad": self.escala.unidad, "calibrada": self.escala.calibrada,
                       "ppm": self.escala.px_per_mm, "ref_mm": self.escala.ref_mm, "homografia": self.escala.H is not None},
            "fondo": "capturado" if self.bg_lab is not None else "automatico",
            "lente": {"activa": self.maps is not None, "poses": len(self.lente_img), "necesarias": a.lens_poses},
            "altura_mm": float(a.altura_mm or 0.0), "ref_alto_mm": float(a.ref_alto_mm or 0.0), "h_cal": float(self.h_cal),
            "dist": {"valor": D, "fuente": fuente, "manual": a.cam_dist_mm},
            "huecos": bool(self.holes),
            "ref_mm": a.ref_mm,
            "check_mm": list(a.check_mm) if a.check_mm else None,
            "tablero": {"cols": self.patron[0], "filas": self.patron[1], "lado_mm": a.square_mm},
            "corregida": self._corr("med") is not None,
            "efecto_altura_pct": self.efecto_altura_pct(),
            "silueta": bool(self.mostrar_silueta),
        }


def _poli_punteado(img, pts, color, th, seg=12):
    """Polígono cerrado con línea punteada (segmentos de `seg` px, alternando dibujado y hueco)."""
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        n = max(1, int(np.hypot(*(b - a)) // max(2, seg)))
        for k in range(0, n, 2):
            p0, p1 = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
            cv2.line(img, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), color, th)


def items_de_objs(objs):
    return [{"obj": o, "largo": o["largo"], "ancho": o["ancho"]} for o in objs]
