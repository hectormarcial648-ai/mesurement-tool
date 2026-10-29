"""Adaptador de segmentación clásica: fondo adaptativo + distancia de color Lab (sin red neuronal)."""
from medicion.dominio.vision import calcular_mascara


class SegmentadorClasico:
    """Implementa PuertoSegmentador con el método de siempre. Es el predeterminado."""

    nombre = "clásico (fondo + color Lab)"
    ultimo_clasico = True

    def segmentar(self, frame, bg_lab, ref, min_diff, thr, erode):
        return calcular_mascara(frame, bg_lab, min_diff, thr, erode, ref=ref)
