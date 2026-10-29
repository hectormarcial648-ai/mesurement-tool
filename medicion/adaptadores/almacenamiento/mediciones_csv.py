"""Adaptador de almacenamiento: registro de mediciones en un CSV."""
import csv
import time
from pathlib import Path

ENCABEZADO = ["fecha", "objeto", "largo", "ancho", "unidad", "std_largo", "std_ancho", "frames", "escala", "tipo",
              "altura_mm", "dist_mm"]


class RepoMedicionesCsv:
    """Implementa PuertoRepoMediciones. Si encuentra un CSV de una versión anterior (sin la columna altura_mm)
    lo respalda y empieza uno nuevo."""

    def __init__(self, ruta, encabezado=None, marca="altura_mm"):
        self._ruta = Path(ruta)
        self._encabezado = list(encabezado or ENCABEZADO)
        self._marca = marca           # columna que identifica el formato vigente

    def ruta(self):
        return self._ruta

    def _preparar(self):
        """True si hay que escribir encabezado."""
        if self._ruta.exists():
            with open(self._ruta, newline="") as fh:
                cab = fh.readline()
            if self._marca in cab:
                return False
            self._ruta.rename(self._ruta.with_name(f"mediciones_anterior_{int(time.time())}.csv"))
        return True

    def registrar(self, filas):
        nuevo = self._preparar()
        with open(self._ruta, "a", newline="") as fh:
            w = csv.writer(fh)
            if nuevo:
                w.writerow(self._encabezado)
            w.writerows(filas)
