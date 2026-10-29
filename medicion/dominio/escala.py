"""Escala: conversión de píxeles a unidades reales (px/mm u homografía)."""
import cv2

import numpy as np


class Escala:
    def __init__(self):
        self.px_per_mm = None
        self.ref_mm = None
        self.H = None
        self.Hinv = None

    def set_px_per_mm(self, v, ref_mm=None):
        self.px_per_mm = float(v)
        self.ref_mm = ref_mm
        self.H = self.Hinv = None

    def set_homografia(self, H):
        self.H = np.asarray(H, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.px_per_mm = None

    def limpiar(self):
        self.px_per_mm = None
        self.H = self.Hinv = None

    @property
    def calibrada(self):
        return self.H is not None or self.px_per_mm is not None

    @property
    def unidad(self):
        return "mm" if self.calibrada else "px"

    @property
    def modo(self):
        if self.H is not None:
            return "tablero (homografía)"
        if self.px_per_mm is not None:
            ref = f"{self.ref_mm:g} mm" if self.ref_mm else "?"
            return f"pieza ref = {ref} ({self.px_per_mm:.2f} px/mm)"
        return "SIN CALIBRAR"

    def a_unid(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.H is not None:
            return cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        if self.px_per_mm:
            return p / self.px_per_mm
        return p

    def a_px(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if self.Hinv is not None:
            return cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.Hinv).reshape(-1, 2)
        if self.px_per_mm:
            return p * self.px_per_mm
        return p
