"""Utilidades de serialización e imágenes para la web."""
import cv2
import json
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"no serializable: {type(o)}")


def a_json(obj):
    return json.dumps(obj, default=_json_default, ensure_ascii=False).encode("utf-8")


def codificar(img, calidad=80, ancho=None, formato=".jpg"):
    """Imagen -> bytes (JPG o PNG), opcionalmente reducida a `ancho` px."""
    if img is None:
        return None
    if ancho and img.shape[1] > ancho:
        img = cv2.resize(img, (int(ancho), int(img.shape[0] * ancho / img.shape[1])), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(calidad)] if formato == ".jpg" else []
    ok, buf = cv2.imencode(formato, img, params)
    return buf.tobytes() if ok else None
