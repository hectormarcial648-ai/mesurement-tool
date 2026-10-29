"""Adaptador de cámara: USB (V4L2)."""
import cv2


class CamaraUSB:
    tipo = "usb"

    def __init__(self, idx=0, width=1280, height=720, lock=True):
        self.idx = int(idx)
        self.cap = cv2.VideoCapture(self.idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara USB {self.idx}")
        for _ in range(10):  # calentar
            self.cap.read()
        if lock:
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual en V4L2
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            except Exception:  # noqa: BLE001
                pass

    def descripcion(self):
        return f"Cámara USB {self.idx}"

    def info_captura(self):
        return f"usb{self.idx}"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()
