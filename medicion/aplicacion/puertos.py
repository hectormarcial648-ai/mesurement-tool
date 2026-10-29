"""Puertos de la capa de aplicación: contratos que el núcleo necesita del exterior.

El núcleo (dominio + aplicación) solo conoce ESTOS contratos; las implementaciones concretas
(cámaras reales, archivos, red neuronal) viven en `medicion.adaptadores` y se inyectan en
`medicion.composicion`. Se usan `Protocol` (tipado estructural): un adaptador no necesita heredar de nada.
"""
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np


class PuertoCamara(Protocol):
    """Fuente de imágenes. Obligatorios: read() y close().

    Opcionales (se detectan con hasattr): descripcion(), info_captura(), geometria_sensor(),
    set_roi()/origen_roi() (recorte en caliente), set_modo()/modos() (modos del sensor),
    set_exposicion()/set_ganancia()/exposicion_ganancia(), set_gris(), alternar().
    """

    def read(self) -> Optional[np.ndarray]:
        """Siguiente imagen (BGR o gris 2D) o None si no hay."""

    def close(self) -> None: ...


class PuertoSegmentador(Protocol):
    """Separa la pieza del fondo. Devuelve (mascara 0/255 uint8, umbral usado, mapa de diferencia para el borde subpíxel)."""

    nombre: str
    ultimo_clasico: bool  # True si el mapa de diferencia viene del método clásico (permite el refinado en alta resolución)

    def segmentar(self, frame: np.ndarray, bg_lab, ref, min_diff: float, thr: float, erode: int): ...


class PuertoSegmentadorConmutable(PuertoSegmentador, Protocol):
    """Segmentador que se puede encender, apagar y ajustar en caliente (lo usa la interfaz)."""

    def encender(self) -> None: ...

    def apagar(self) -> None: ...

    def configurar(self, **ajustes) -> None: ...

    def estado(self) -> dict: ...


class PuertoExtractorCnn(Protocol):
    """Extractor de huellas con red neuronal (opcional). Convierte un parche BGR en un vector."""

    def estado(self) -> dict: ...

    def encender(self, modelo: Optional[str] = None) -> None: ...

    def apagar(self) -> None: ...

    def extraer(self, parche_bgr: np.ndarray) -> np.ndarray: ...


class PuertoRepoPiezas(Protocol):
    """Biblioteca de piezas: fichas (medidas nominales), huellas por cámara, controles y copia de respaldo."""

    def listar(self) -> list: ...

    def leer(self, pid: str) -> Optional[dict]: ...

    def guardar(self, datos: dict) -> dict: ...

    def borrar(self, pid: str) -> None: ...

    def leer_huellas(self, pid: str, cam_id: str) -> dict: ...

    def guardar_huellas(self, pid: str, cam_id: str, geo, cnn) -> None: ...

    def borrar_huellas(self, pid: str, cam_id: Optional[str] = None) -> None: ...

    def guardar_miniatura(self, pid: str, jpg: bytes) -> None: ...

    def leer_miniatura(self, pid: str) -> Optional[bytes]: ...

    def tiene_miniatura(self, pid: str) -> bool: ...

    def agregar_control(self, cam_id: str, fila: dict) -> None: ...

    def leer_controles(self, cam_id: str, n: int = 10) -> list: ...

    def exportar(self) -> bytes: ...

    def importar(self, datos: bytes) -> int: ...


class PuertoRepoCalibracion(Protocol):
    """Persistencia de la calibración y de la lente, por 'clave' de captura (cámara + tamaño + ROI)."""

    def leer_calibracion(self, clave: str) -> Optional[Tuple[dict, float]]:
        """(datos, instante de modificación) o None."""

    def guardar_calibracion(self, clave: str, datos: dict) -> None: ...

    def borrar_calibracion(self, clave: str) -> None: ...

    def mtime_calibracion(self, clave: str) -> float:
        """Instante de la calibración guardada para esa captura, o -1 si no hay."""

    def leer_lente(self, clave: str) -> Optional[dict]:
        """dict con K, dist, newK, size, rms, firma; o None."""

    def guardar_lente(self, clave: str, datos: dict) -> None: ...

    def nombres(self, clave: str) -> Tuple[str, str]:
        """(nombre del archivo de lente, nombre del de calibración): solo para mensajes al usuario."""


class PuertoRepoMediciones(Protocol):
    """Registro histórico de mediciones."""

    def registrar(self, filas: list) -> None: ...

    def ruta(self) -> Optional[Path]: ...
