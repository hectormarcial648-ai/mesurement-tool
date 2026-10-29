"""Adaptador de almacenamiento: calibración (JSON) y lente (NPZ) en archivos junto al programa."""
import json
from pathlib import Path

import numpy as np


class RepoCalibracionArchivos:
    """Implementa PuertoRepoCalibracion. Cada captura (cámara + tamaño + ROI) tiene sus propios archivos;
    se aceptan también los archivos 'antiguos' de versiones anteriores del programa (legacy)."""

    def __init__(self, base, legacy_calib=None, legacy_lente=None):
        self.base = Path(base)
        self.legacy_calib = Path(legacy_calib) if legacy_calib else None
        self.legacy_lente = Path(legacy_lente) if legacy_lente else None

    def _cal(self, clave):
        return self.base / f"calibracion_{clave}.json"

    def _len(self, clave):
        return self.base / f"lente_{clave}.npz"

    def leer_calibracion(self, clave):
        ruta = self._cal(clave)
        if not ruta.exists():
            ruta = self.legacy_calib  # archivo de versiones anteriores
        if ruta is None or not ruta.exists():
            return None
        return json.loads(ruta.read_text()), ruta.stat().st_mtime

    def guardar_calibracion(self, clave, datos):
        self._cal(clave).write_text(json.dumps(datos, indent=2))

    def borrar_calibracion(self, clave):
        for p in (self._cal(clave), self.legacy_calib):
            if p is not None and p.exists():
                p.unlink()

    def mtime_calibracion(self, clave):
        ruta = self._cal(clave)
        return ruta.stat().st_mtime if ruta.exists() else -1.0

    def leer_lente(self, clave):
        ruta = self._len(clave)
        if not ruta.exists():
            ruta = self.legacy_lente
        if ruta is None or not ruta.exists():
            return None
        with np.load(ruta) as d:
            return {"K": d["K"], "dist": d["dist"], "newK": d["newK"], "size": tuple(int(v) for v in d["size"]),
                    "rms": float(d["rms"]), "firma": str(d["firma"]) if "firma" in d.files else None}

    def guardar_lente(self, clave, datos):
        np.savez(self._len(clave), K=datos["K"], dist=datos["dist"], newK=datos["newK"],
                 size=np.array(datos["size"]), rms=datos["rms"], firma=np.array(datos["firma"]))

    def nombres(self, clave):
        return self._len(clave).name, self._cal(clave).name
