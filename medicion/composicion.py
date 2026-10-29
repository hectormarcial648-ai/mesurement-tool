"""Raíz de composición: aquí (y solo aquí) se eligen las implementaciones concretas de cada puerto."""
from medicion.adaptadores.almacenamiento.calibracion_archivos import RepoCalibracionArchivos
from medicion.adaptadores.almacenamiento.mediciones_csv import RepoMedicionesCsv
from medicion.adaptadores.almacenamiento.piezas_archivos import RepoPiezasArchivos
from medicion.adaptadores.camaras.csi import CamaraCSI
from medicion.adaptadores.camaras.gige import CamaraGigE
from medicion.adaptadores.camaras.tof import CamaraToF
from medicion.adaptadores.camaras.usb import CamaraUSB
from medicion.adaptadores.reconocimiento_cnn import ExtractorCnnConmutable
from medicion.adaptadores.segmentadores.clasico import SegmentadorClasico
from medicion.adaptadores.segmentadores.conmutable import SegmentadorConmutable
from medicion.aplicacion.biblioteca import BibliotecaPiezas
from medicion.aplicacion.sistema import Sistema
from medicion.config import BASE

# Mismos nombres que usaban las versiones anteriores: así se aprovechan calibraciones y CSV existentes.
ARCHIVOS_POR_CAMARA = {
    "gige": {"calib_legacy": BASE / "calibracion.json", "lente_legacy": BASE / "lente.npz", "csv": BASE / "mediciones.csv"},
    "csi": {"calib_legacy": BASE / "calibracion_csi.json", "lente_legacy": BASE / "lente_csi.npz", "csv": BASE / "mediciones_csi.csv"},
    "tof": {"calib_legacy": BASE / "calibracion_tof.json", "lente_legacy": BASE / "lente_tof.npz", "csv": BASE / "mediciones_tof.csv"},
}


ENC_VEREDICTOS = ["fecha", "pieza", "origen", "largo", "ancho", "nominal_largo", "nominal_ancho", "tol", "err_largo", "err_ancho", "veredicto"]


class Fabrica:
    """Arma una cámara + su Sistema con las implementaciones elegidas por la línea de comandos."""

    def __init__(self):
        # la biblioteca de piezas es UNA para todas las cámaras (las huellas se guardan por cámara)
        self.biblioteca = BibliotecaPiezas(RepoPiezasArchivos(BASE / "piezas"))

    def crear_cnn(self):
        return ExtractorCnnConmutable(BASE)

    def repo_veredictos(self, cid):
        return RepoMedicionesCsv(BASE / f"veredictos_{cid}.csv", encabezado=ENC_VEREDICTOS, marca="veredicto")

    def archivos(self, fuente, cid):
        return ARCHIVOS_POR_CAMARA.get(fuente, {"csv": BASE / f"mediciones_{cid}.csv",
                                                "calib_legacy": BASE / f"calibracion_{cid}.json",
                                                "lente_legacy": BASE / f"lente_{cid}.npz"})

    def ruta_csv(self, fuente, cid):
        return self.archivos(fuente, cid)["csv"]

    def crear_camara(self, fuente, a):
        if fuente == "gige":
            return CamaraGigE(a.ip, a.roi_size, a.roi_h, a.exposure, a.gain, a.decimation)
        if fuente == "csi":
            return CamaraCSI(a.width, a.height, lock=not a.no_lock, lens_position=a.lens_position)
        if fuente == "tof":
            return CamaraToF(a.tof_range, a.tof_up, a.tof_span_mm, a.tof_conf, a.tof_image, a.tof_centro_mm)
        if fuente.startswith("usb"):
            idx = fuente.split(":")[1] if ":" in fuente else "0"
            return CamaraUSB(int(idx), a.width or 1280, a.height or 720, lock=not a.no_lock)
        raise RuntimeError(f"Fuente de cámara desconocida: {fuente}")

    def crear_segmentador(self, a):
        """Siempre un segmentador conmutable: arranca con el método clásico y la red se enciende desde la interfaz
        (o ya encendida si se pidió --segmentador onnx)."""
        def fabrica_red(cfg):
            from medicion.adaptadores.segmentadores.onnx import SegmentadorOnnx  # import tardío: onnxruntime es opcional
            return SegmentadorOnnx(cfg["modelo"], tam=cfg["tam"], salida=cfg["salida"], umbral=cfg["umbral"],
                                   margen_px=cfg["margen_px"], modo=cfg["modo"], cada=cfg["cada"],
                                   media=tuple(cfg["media"]), desv=tuple(cfg["desv"]))
        cfg = {"modo": a.seg_modo, "umbral": a.seg_umbral, "margen_px": a.seg_margen_px, "cada": a.seg_cada,
               "tam": a.seg_tam, "salida": a.seg_salida,
               "media": tuple(float(v) for v in a.seg_media.split(",")), "desv": tuple(float(v) for v in a.seg_desv.split(","))}
        seg = SegmentadorConmutable(SegmentadorClasico(), fabrica_red, BASE, cfg, modelo_confiable=a.seg_modelo)
        if a.seg_modelo:
            seg.configurar(modelo=a.seg_modelo)
        elif (BASE / "u2netp.onnx").is_file():
            seg.configurar(modelo="u2netp.onnx")          # si ya está descargado, queda preseleccionado
        if getattr(a, "segmentador", "clasico") == "onnx":
            seg.encender()                                # pedido por línea de comandos: si falla, se avisa al arrancar
        return seg

    def armar(self, fuente, cid, args):
        """Devuelve (camara, sistema). Si algo falla, cierra la cámara que alcanzó a abrir."""
        cam = self.crear_camara(fuente, args)
        try:
            arch = self.archivos(fuente, cid)
            sis = Sistema(args, cam, cam_id=cid,
                          repo_calibracion=RepoCalibracionArchivos(BASE, arch["calib_legacy"], arch["lente_legacy"]),
                          repo_mediciones=RepoMedicionesCsv(arch["csv"]),
                          segmentador=self.crear_segmentador(args))
        except BaseException:
            try:
                cam.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        return cam, sis
