"""Utilidades de texto sobre imágenes (OpenCV)."""
import cv2
import unicodedata


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _sin_acentos(s):
    """La fuente de OpenCV no dibuja acentos ni ñ: se convierten a ASCII."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def texto(img, s, org, fs, color=(0, 255, 255), th=2):
    s = _sin_acentos(s)
    cv2.putText(img, s, org, FONT, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, fs, color, th, cv2.LINE_AA)
