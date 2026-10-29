"""Adaptador de almacenamiento: biblioteca de piezas en archivos (una carpeta por pieza).

    piezas/<id>/pieza.json            ficha: nombre, medidas nominales, tolerancia, altura, maestra...
    piezas/<id>/huellas_<cam>.npz     muestras aprendidas por CADA cámara (geo y, si hay, cnn)
    piezas/<id>/miniatura.jpg         imagen de referencia
    piezas/controles_<cam>.csv        historial de controles con la pieza maestra
"""
import csv
import io
import json
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

CAMPOS_CONTROL = ["fecha", "t", "pieza", "largo", "ancho", "err_largo", "err_ancho", "tol", "estado"]
_PERMITIDO = re.compile(r"^[a-z0-9][a-z0-9_-]*/(pieza\.json|miniatura\.jpg|huellas_[a-z0-9_-]+\.npz)$|^controles_[a-z0-9_-]+\.csv$")


def _slug(texto):
    s = re.sub(r"[^a-z0-9]+", "-", str(texto).lower().strip()).strip("-")
    return s or "pieza"


class RepoPiezasArchivos:
    """Implementa PuertoRepoPiezas."""

    def __init__(self, base):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _dir(self, pid):
        return self.base / _slug(pid)

    # ----- fichas ------------------------------------------------------------------------------------------- #
    def listar(self):
        with self._lock:
            out = []
            for d in sorted(p for p in self.base.iterdir() if p.is_dir()):
                f = d / "pieza.json"
                if f.is_file():
                    try:
                        out.append(json.loads(f.read_text(encoding="utf-8")))
                    except Exception:  # noqa: BLE001
                        pass
            return out

    def leer(self, pid):
        f = self._dir(pid) / "pieza.json"
        with self._lock:
            return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None

    def guardar(self, datos):
        with self._lock:
            datos = dict(datos)
            if not datos.get("id"):
                base = _slug(datos.get("nombre", "pieza"))
                pid, k = base, 2
                while (self.base / pid).exists():
                    pid, k = f"{base}-{k}", k + 1
                datos["id"] = pid
            datos["id"] = _slug(datos["id"])
            d = self._dir(datos["id"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "pieza.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
            return datos

    def borrar(self, pid):
        with self._lock:
            d = self._dir(pid)
            if d.is_dir():
                shutil.rmtree(d)

    # ----- huellas ------------------------------------------------------------------------------------------ #
    def leer_huellas(self, pid, cam_id):
        f = self._dir(pid) / f"huellas_{_slug(cam_id)}.npz"
        with self._lock:
            if not f.is_file():
                return {"geo": None, "cnn": None}
            with np.load(f) as d:
                return {"geo": d["geo"] if "geo" in d.files else None, "cnn": d["cnn"] if "cnn" in d.files else None}

    def guardar_huellas(self, pid, cam_id, geo, cnn):
        with self._lock:
            d = self._dir(pid)
            d.mkdir(parents=True, exist_ok=True)
            datos = {}
            if geo is not None and len(geo):
                datos["geo"] = np.asarray(geo, np.float32)
            if cnn is not None and len(cnn):
                datos["cnn"] = np.asarray(cnn, np.float32)
            np.savez(d / f"huellas_{_slug(cam_id)}.npz", **datos)

    def borrar_huellas(self, pid, cam_id=None):
        with self._lock:
            d = self._dir(pid)
            for f in ([d / f"huellas_{_slug(cam_id)}.npz"] if cam_id else list(d.glob("huellas_*.npz"))):
                if f.is_file():
                    f.unlink()

    def guardar_miniatura(self, pid, jpg):
        with self._lock:
            d = self._dir(pid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "miniatura.jpg").write_bytes(jpg)

    def tiene_miniatura(self, pid):
        return (self._dir(pid) / "miniatura.jpg").is_file()

    def leer_miniatura(self, pid):
        f = self._dir(pid) / "miniatura.jpg"
        return f.read_bytes() if f.is_file() else None

    # ----- controles con la pieza maestra --------------------------------------------------------------------- #
    def agregar_control(self, cam_id, fila):
        f = self.base / f"controles_{_slug(cam_id)}.csv"
        with self._lock:
            nuevo = not f.exists()
            with open(f, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=CAMPOS_CONTROL)
                if nuevo:
                    w.writeheader()
                w.writerow({k: fila.get(k, "") for k in CAMPOS_CONTROL})

    def leer_controles(self, cam_id, n=10):
        f = self.base / f"controles_{_slug(cam_id)}.csv"
        with self._lock:
            if not f.is_file():
                return []
            with open(f, newline="", encoding="utf-8") as fh:
                filas = list(csv.DictReader(fh))
        return filas[-n:]

    # ----- respaldo ------------------------------------------------------------------------------------------- #
    def exportar(self):
        buf = io.BytesIO()
        with self._lock, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(self.base.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(self.base).as_posix())
        return buf.getvalue()

    def importar(self, datos):
        """Fusiona un respaldo: solo acepta los nombres de archivo esperados (sin rutas peligrosas). Devuelve nº de piezas."""
        n = 0
        with self._lock, zipfile.ZipFile(io.BytesIO(datos)) as z:
            for info in z.infolist():
                nombre = info.filename
                if info.is_dir() or not _PERMITIDO.match(nombre) or info.file_size > 50_000_000:
                    continue
                contenido = z.read(info)
                if nombre.endswith("pieza.json"):  # la ficha debe ser válida y coincidir con su carpeta
                    try:
                        ficha = json.loads(contenido.decode("utf-8"))
                        if not (isinstance(ficha, dict) and ficha.get("nombre") and _slug(ficha.get("id", "")) == nombre.split("/")[0]):
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                destino = self.base / nombre
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(contenido)
                if nombre.endswith("pieza.json"):
                    n += 1
        return n
