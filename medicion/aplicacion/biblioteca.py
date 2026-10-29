"""Servicio de aplicación: biblioteca de piezas.

Fichas con medidas nominales, aprendizaje de huellas por cámara, reconocimiento (con rechazo de piezas desconocidas),
veredicto contra la nominal, evaluación y control con pieza maestra. Solo depende del dominio y de los puertos.
"""
import threading
import time

import cv2
import numpy as np

from medicion.dominio.piezas import grupos, huella_geometrica, parche_alineado, pisos, veredicto
from medicion.dominio.reconocimiento import ClasificadorHuellas, combinar

MIN_MUESTRAS = 5          # menos que esto: la pieza todavía no entra en el reconocimiento
RECOMENDADAS = 15
CAMPOS_NUM = ("largo", "ancho", "diametro", "tol", "altura_mm")


class BibliotecaPiezas:
    def __init__(self, repo):
        self.repo = repo                  # PuertoRepoPiezas
        self._lock = threading.RLock()
        self.version = 0                  # sube con cada cambio: invalida cachés
        self._cache = {}                  # cam_id -> clasificadores ajustados
        self._lista = {}                  # cam_id -> (version, lista)

    def _cambio(self):
        self.version += 1

    # ----- fichas ------------------------------------------------------------------------------------------- #
    def listar(self, cam_id=None):
        with self._lock:
            hit = self._lista.get(cam_id)
            if hit and hit[0] == self.version:
                return hit[1]
            out = []
            for p in self.repo.listar():
                d = dict(p)
                if cam_id:
                    h = self.repo.leer_huellas(p["id"], cam_id)
                    d["muestras"] = 0 if h["geo"] is None else int(len(h["geo"]))
                    d["muestras_cnn"] = 0 if h["cnn"] is None else int(len(h["cnn"]))
                d["miniatura"] = bool(self.repo.tiene_miniatura(p["id"]))
                out.append(d)
            self._lista[cam_id] = (self.version, out)
            return out

    def obtener(self, pid):
        return self.repo.leer(pid)

    def nombre(self, pid):
        p = self.repo.leer(pid) if pid else None
        return p["nombre"] if p else None

    def guardar(self, datos):
        nombre = str(datos.get("nombre") or "").strip()      # None / vacío -> sin nombre (no el texto "None")
        if not nombre:
            raise ValueError("La pieza necesita un nombre")
        ficha = {"id": datos.get("id") or None, "nombre": nombre, "notas": str(datos.get("notas", "") or "").strip(),
                 "maestra": bool(datos.get("maestra"))}
        for k in CAMPOS_NUM:
            v = datos.get(k)
            ficha[k] = None if v in (None, "") else float(v)
            if ficha[k] is not None and ficha[k] < 0:
                raise ValueError(f"{k} no puede ser negativo")
        if ficha["maestra"] and not ((ficha["largo"] or ficha["diametro"]) and ficha["tol"]):
            raise ValueError("Una pieza maestra necesita su medida nominal y su tolerancia")
        with self._lock:
            out = self.repo.guardar(ficha)
            self._cambio()
        return out

    def eliminar(self, pid):
        with self._lock:
            self.repo.borrar(pid)
            self._cambio()

    # ----- descriptores y aprendizaje --------------------------------------------------------------------------- #
    def descriptor(self, frame, obj, mm_por_px=None, cnn=None):
        """-> (dict con 'geo', 'cnn', 'parche') o (None, motivo)."""
        parche, info = parche_alineado(frame, obj)
        if parche is None:
            return None, info
        geo = huella_geometrica(parche, info, mm_por_px)
        if geo is None:
            return None, "no se pudo segmentar la pieza dentro del recorte"
        d = {"geo": geo, "cnn": None, "parche": parche}
        if cnn is not None and cnn.estado().get("activo"):
            d["cnn"] = cnn.extraer(parche)
        return d, None

    def agregar_muestras(self, pid, cam_id, descs, miniatura=None):
        if not descs:
            return 0
        with self._lock:
            if self.repo.leer(pid) is None:
                raise ValueError("La pieza no existe")
            h = self.repo.leer_huellas(pid, cam_id)
            geo = np.vstack([x for x in ([h["geo"]] if h["geo"] is not None else []) + [np.array([d["geo"] for d in descs], np.float32)]])
            cnn_nuevas = [d["cnn"] for d in descs if d.get("cnn") is not None]
            cnn = h["cnn"]
            if cnn_nuevas:
                nuevas = np.array(cnn_nuevas, np.float32)
                cnn = nuevas if cnn is None else (np.vstack([cnn, nuevas]) if cnn.shape[1] == nuevas.shape[1] else nuevas)
            self.repo.guardar_huellas(pid, cam_id, geo, cnn)
            if miniatura is not None and not self.repo.tiene_miniatura(pid):
                ok, buf = cv2.imencode(".jpg", miniatura, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.repo.guardar_miniatura(pid, buf.tobytes())
            self._cambio()
            return len(descs)

    def borrar_muestras(self, pid, cam_id):
        with self._lock:
            self.repo.borrar_huellas(pid, cam_id)
            self._cambio()

    # ----- clasificadores ------------------------------------------------------------------------------------------ #
    def _clasif(self, cam_id):
        with self._lock:
            c = self._cache.get(cam_id)
            if c and c["v"] == self.version:
                return c
            datos_geo, datos_cnn, pobres = {}, {}, {}
            for p in self.repo.listar():
                h = self.repo.leer_huellas(p["id"], cam_id)
                n = 0 if h["geo"] is None else len(h["geo"])
                if n and n < MIN_MUESTRAS:
                    pobres[p["id"]] = n
                if h["geo"] is not None and n >= MIN_MUESTRAS:
                    datos_geo[p["id"]] = h["geo"]
                if h["cnn"] is not None and len(h["cnn"]) >= MIN_MUESTRAS:
                    datos_cnn[p["id"]] = h["cnn"]
            pa, pr = pisos()
            geo = ClasificadorHuellas("geo", piso_abs=pa, piso_rel=pr, grupos=grupos())
            geo.ajustar(datos_geo)
            cnn = None
            if datos_cnn:
                cnn = ClasificadorHuellas("cnn")
                cnn.ajustar(datos_cnn)
            c = {"v": self.version, "geo": geo, "cnn": cnn, "datos_geo": datos_geo, "datos_cnn": datos_cnn, "pobres": pobres}
            self._cache[cam_id] = c
            return c

    def _con_nombres(self, r):
        r = dict(r)
        r["nombre"] = self.nombre(r.get("id"))
        r["candidatas"] = [dict(c, nombre=self.nombre(c["id"])) for c in r.get("candidatas", [])]
        return r

    def identificar(self, cam_id, desc, modo="geometrico"):
        """modo: geometrico | cnn | ambos. -> dict con estado (reconocida|desconocida|ambigua|vacia), id, nombre, candidatas."""
        c = self._clasif(cam_id)
        r_geo = c["geo"].identificar(desc["geo"]) if c["datos_geo"] else {"estado": "vacia", "id": None, "candidatas": []}
        r_cnn = c["cnn"].identificar(desc["cnn"]) if (c["cnn"] is not None and desc.get("cnn") is not None) else None
        if modo == "geometrico":
            r = r_geo
        elif modo == "cnn":
            r = r_cnn if r_cnn is not None else {"estado": "vacia", "id": None, "candidatas": [], "nota": "faltan huellas de la red o la red está apagada"}
        else:
            r = combinar(r_geo, r_cnn) if r_cnn is not None else dict(r_geo, nota="sin red: solo geométrico")
        return self._con_nombres(r)

    def evaluar(self, cam_id):
        c = self._clasif(cam_id)
        out = {"avisos": [], "geo": None, "cnn": None, "piezas": {}}
        for pid, n in c["pobres"].items():
            out["avisos"].append(f"«{self.nombre(pid)}» tiene solo {n} muestras: necesita al menos {MIN_MUESTRAS} para entrar al reconocimiento")
        for espacio, clf, datos in (("geo", c["geo"], c["datos_geo"]), ("cnn", c["cnn"], c["datos_cnn"])):
            if clf is None or not datos:
                continue
            ev = clf.evaluar(datos)
            out[espacio] = {
                "total": ev["total"],
                "por_pieza": {pid: dict(r, nombre=self.nombre(pid), confundida_con={self.nombre(k): v for k, v in r["confundida_con"].items()})
                              for pid, r in ev["por_pieza"].items()},
                "separacion": [dict(s, a_nombre=self.nombre(s["a"]), b_nombre=self.nombre(s["b"])) for s in ev["separacion"]]}
            for pid, r in ev["por_pieza"].items():
                if r["n"] < RECOMENDADAS:
                    out["avisos"].append(f"[{espacio}] «{self.nombre(pid)}»: {r['n']} muestras; se recomiendan {RECOMENDADAS} o más, en distintas posiciones y giros")
                if r["n"] >= 3 and r["ok"] < 0.95 * r["n"]:
                    out["avisos"].append(f"[{espacio}] «{self.nombre(pid)}»: solo {r['ok']} de {r['n']} bien en la prueba dejar-una-fuera")
            for s in ev["separacion"]:
                if not s["separadas"]:
                    out["avisos"].append(f"[{espacio}] «{self.nombre(s['a'])}» y «{self.nombre(s['b'])}» se parecen demasiado (se pueden confundir): enseña más muestras o usa la red neuronal")
        if not out["geo"]:
            out["avisos"].append("Todavía no hay piezas con muestras suficientes en esta cámara")
        return out

    # ----- veredicto y receta --------------------------------------------------------------------------------------- #
    def veredicto_para(self, pid, medidas):
        pieza = self.repo.leer(pid)
        return None if pieza is None else veredicto(medidas, pieza)

    @staticmethod
    def receta(pieza):
        """Ajustes que una pieza conocida carga al reconocerse (la ESCALA nunca: depende del montaje, no de la pieza)."""
        return {"altura_mm": pieza.get("altura_mm"), "check_mm": (pieza.get("largo") or pieza.get("diametro"), pieza.get("ancho") or pieza.get("diametro")),
                "tol": pieza.get("tol")}

    # ----- control con pieza maestra --------------------------------------------------------------------------------- #
    def registrar_control(self, cam_id, pid, medidas, ver):
        fila = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "t": f"{time.time():.0f}", "pieza": pid,
                "largo": f"{medidas['largo']:.3f}", "ancho": f"{medidas['ancho']:.3f}", "tol": ver.get("tol") or "",
                "err_largo": f"{ver['filas'][0]['error']:+.3f}" if ver["filas"] else "",
                "err_ancho": f"{ver['filas'][1]['error']:+.3f}" if len(ver["filas"]) > 1 else "", "estado": ver["estado"]}
        self.repo.agregar_control(cam_id, fila)

    def controles(self, cam_id, n=10):
        return self.repo.leer_controles(cam_id, n)

    # ----- respaldo ---------------------------------------------------------------------------------------------------- #
    def exportar(self):
        return self.repo.exportar()

    def importar(self, datos):
        with self._lock:
            n = self.repo.importar(datos)
            self._cambio()
        return n
