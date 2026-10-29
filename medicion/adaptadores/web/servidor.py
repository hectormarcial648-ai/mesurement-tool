"""Servidor web (solo biblioteca estándar): interfaz, video MJPEG y API JSON."""
import base64
import hmac
import json
import re
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

from medicion.adaptadores.web.util import a_json, codificar
from medicion.config import CAPTURAS_DIR


class Handler(BaseHTTPRequestHandler):
    server_version = "MedicionVision/1.0"
    workers = {}          # id -> CamaraWorker
    biblioteca = None     # servicio de la biblioteca de piezas (compartido)
    html = b""
    credencial = None     # "usuario:clave" si se pidió contraseña

    def log_message(self, fmt, *a):  # sin ruido por cada petición
        pass

    # ----- utilidades --------------------------------------------------------- #
    def _autorizado(self):
        if not self.credencial:
            return True
        cab = self.headers.get("Authorization", "")
        if cab.startswith("Basic "):
            try:
                dado = base64.b64decode(cab[6:]).decode("utf-8", "replace")
                if hmac.compare_digest(dado, self.credencial):
                    return True
            except Exception:  # noqa: BLE001
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Medicion por vision"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _enviar(self, codigo, tipo, datos, extra=None):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(datos)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(datos)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, obj, codigo=200):
        self._enviar(codigo, "application/json; charset=utf-8", a_json(obj))

    def _worker(self, cid):
        w = self.workers.get(cid)
        if w is None:
            self._json({"ok": False, "error": f"Cámara desconocida: {cid}"}, 404)
        return w

    # ----- GET -------------------------------------------------------------------- #
    def do_GET(self):
        if not self._autorizado():
            return
        u = urlparse(self.path)
        ruta = unquote(u.path)
        try:
            if ruta in ("/", "/index.html"):
                return self._enviar(200, "text/html; charset=utf-8", self.html)
            if ruta == "/api/estado":
                cams = sorted((w.estado() for w in self.workers.values()), key=lambda e: e.get("orden", 0))
                return self._json({"hora": time.time(), "camaras": cams})
            if ruta == "/api/salud":
                return self._json({"ok": True})
            m = re.fullmatch(r"/stream/([\w\-]+)\.mjpg", ruta)
            if m:
                w = self._worker(m.group(1))
                return self._stream(w) if w else None
            m = re.fullmatch(r"/snapshot/([\w\-]+)\.jpg", ruta)
            if m:
                w = self._worker(m.group(1))
                if not w:
                    return
                data = codificar(w.ultimo_anotado, 92)
                if data is None:
                    return self._json({"ok": False, "error": "Todavía no hay imagen"}, 503)
                return self._enviar(200, "image/jpeg", data)
            m = re.fullmatch(r"/resultado/([\w\-]+)\.jpg", ruta)
            if m:
                w = self._worker(m.group(1))
                if not w:
                    return
                if w.resultado_jpeg is None:
                    return self._json({"ok": False, "error": "Sin resultado"}, 404)
                return self._enviar(200, "image/jpeg", w.resultado_jpeg)
            m = re.fullmatch(r"/piezas/([\w\-]+)\.jpg", ruta)
            if m:
                jpg = self.biblioteca.repo.leer_miniatura(m.group(1)) if self.biblioteca else None
                if jpg is None:
                    return self._json({"ok": False, "error": "Sin miniatura"}, 404)
                return self._enviar(200, "image/jpeg", jpg)
            if ruta == "/api/piezas/exportar":
                if self.biblioteca is None:
                    return self._json({"ok": False, "error": "Sin biblioteca"}, 404)
                return self._enviar(200, "application/zip", self.biblioteca.exportar(),
                                    {"Content-Disposition": f'attachment; filename="biblioteca_piezas_{time.strftime("%Y%m%d_%H%M")}.zip"'})
            m = re.fullmatch(r"/capturas/([\w\-.]+\.png)", ruta)
            if m:
                f = CAPTURAS_DIR / Path(m.group(1)).name
                if not f.exists():
                    return self._json({"ok": False, "error": "No existe"}, 404)
                return self._enviar(200, "image/png", f.read_bytes(),
                                    {"Content-Disposition": f'inline; filename="{f.name}"'})
            m = re.fullmatch(r"/api/csv/([\w\-]+)", ruta)
            if m:
                w = self._worker(m.group(1))
                if not w:
                    return
                f = w.csv_ruta()
                if f is None or not Path(f).exists():
                    return self._json({"ok": False, "error": "Todavía no hay mediciones guardadas"}, 404)
                return self._enviar(200, "text/csv; charset=utf-8", Path(f).read_bytes(),
                                    {"Content-Disposition": f'attachment; filename="{Path(f).name}"'})
            return self._json({"ok": False, "error": "No encontrado"}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[web] error en GET {ruta}: {e!r}")
            try:
                self._json({"ok": False, "error": repr(e)}, 500)
            except Exception:  # noqa: BLE001
                pass

    def _stream(self, w):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        ultimo = -1
        try:
            while not w._parar.is_set():
                seq, data = w.esperar_video(ultimo, timeout=2.0)
                if data is None:
                    time.sleep(0.1)
                    continue
                ultimo = seq  # si no hubo imagen nueva en 2 s se reenvía la última (mantiene viva la conexión)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    # ----- POST -------------------------------------------------------------------- #
    def do_POST(self):
        if not self._autorizado():
            return
        ruta = unquote(urlparse(self.path).path)
        if ruta == "/api/piezas/importar":
            try:
                largo = int(self.headers.get("Content-Length", 0) or 0)
                if not 0 < largo <= 100_000_000:
                    raise ValueError("tamaño inválido")
                n = self.biblioteca.importar(self.rfile.read(largo))
                return self._json({"ok": True, "piezas": n, "mensaje": f"Biblioteca importada: {n} pieza(s)"})
            except Exception as e:  # noqa: BLE001
                return self._json({"ok": False, "error": f"No se pudo importar: {e}"}, 400)
        m = re.fullmatch(r"/api/([\w\-]+)/([\w\-]+)", ruta)
        if not m:
            return self._json({"ok": False, "error": "No encontrado"}, 404)
        w = self._worker(m.group(1))
        if not w:
            return
        try:
            largo = int(self.headers.get("Content-Length", 0) or 0)
            cuerpo = self.rfile.read(largo) if largo else b"{}"
            datos = json.loads(cuerpo.decode("utf-8") or "{}")
            if not isinstance(datos, dict):
                raise ValueError("el cuerpo debe ser un objeto JSON")
        except Exception as e:  # noqa: BLE001
            return self._json({"ok": False, "error": f"Petición inválida: {e}"}, 400)
        try:
            self._json(w.enviar(m.group(2), datos))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class Servidor(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def ips_locales():
    """IPs de la Pi para mostrar la dirección que hay que abrir en el navegador."""
    ips = []
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        ips = [i for i in out if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", i)]
    except Exception:  # noqa: BLE001
        pass
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ips = [s.getsockname()[0]]
            s.close()
        except Exception:  # noqa: BLE001
            ips = ["127.0.0.1"]
    return ips
