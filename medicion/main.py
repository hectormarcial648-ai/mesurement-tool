"""Punto de entrada: compone el sistema y arranca el servidor."""
import re
import signal

from medicion.adaptadores.web.servidor import Handler, Servidor, ips_locales
from medicion.adaptadores.web.ui import HTML
from medicion.adaptadores.web.worker import CamaraWorker
from medicion.composicion import Fabrica
from medicion.config import parse_args


def main(argv=None):
    args = parse_args(argv)
    fuentes = [f.strip().lower() for f in args.camaras.split(",") if f.strip()]
    if not fuentes:
        raise SystemExit("--camaras necesita al menos una fuente: gige, csi, tof o usb:N")
    if "csi" in fuentes and "tof" in fuentes:
        print("[aviso] La HQ (csi) y la ToF usan el MISMO puerto CSI de la Pi 4B: no pueden estar conectadas a la vez.")
    fabrica = Fabrica()
    workers = {}
    for i, f in enumerate(fuentes):
        cid = re.sub(r"[^0-9a-z]", "", f) or f"cam{i}"
        if cid in workers:
            cid = f"{cid}{i}"
        workers[cid] = CamaraWorker(cid, f, args, i, fabrica)
    Handler.workers = workers
    Handler.biblioteca = fabrica.biblioteca
    Handler.html = HTML.encode("utf-8")
    Handler.credencial = f"{args.usuario}:{args.clave}" if args.clave else None
    for w in workers.values():
        w.start()
    try:
        srv = Servidor((args.host, args.puerto), Handler)
    except OSError as e:
        for w in workers.values():
            w.parar()
        raise SystemExit(f"No se pudo abrir el puerto {args.puerto}: {e}")
    def _terminar(signum, frame):  # SIGTERM (systemd, kill): cerrar limpio igual que con Ctrl+C
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _terminar)
    except Exception:  # noqa: BLE001
        pass
    print("\n" + "=" * 64)
    print(" Medición por visión: abre en el navegador de tu PC (misma red):")
    for ip in ips_locales():
        print(f"     http://{ip}:{args.puerto}")
    print(" (Ctrl+C para salir)" + ("   [protegido con usuario y clave]" if args.clave else "   [sin contraseña: usa --clave]"))
    print("=" * 64 + "\n")
    try:
        srv.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nCerrando…")
        srv.shutdown()
        for w in workers.values():
            w.parar()
        for w in workers.values():
            w.join(timeout=5)
        srv.server_close()
