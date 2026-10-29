"""Descarga verificada de modelos ONNX (solo biblioteca estándar)."""
import hashlib
import urllib.request
from pathlib import Path

U2NETP = {
    "nombre": "u2netp.onnx",
    "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    "sha256": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
}


def descargar_modelo(destino_dir, info=U2NETP, timeout=60):
    """Baja el modelo a `destino_dir`, verifica su huella SHA-256 y devuelve la ruta. Lanza RuntimeError si falla."""
    destino = Path(destino_dir) / info["nombre"]
    parcial = destino.with_suffix(".part")
    try:
        h = hashlib.sha256()
        with urllib.request.urlopen(info["url"], timeout=timeout) as r, open(parcial, "wb") as f:
            while True:
                bloque = r.read(1 << 16)
                if not bloque:
                    break
                h.update(bloque)
                f.write(bloque)
        if h.hexdigest() != info["sha256"]:
            raise RuntimeError("La huella SHA-256 del archivo descargado no coincide: se descartó")
        parcial.replace(destino)
        return destino
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001  (sin internet, DNS, certificados, disco...)
        raise RuntimeError(f"No se pudo descargar el modelo ({type(e).__name__}: {e}). "
                           f"Si la Pi no tiene internet, copia {info['nombre']} a la carpeta del programa.") from e
    finally:
        if parcial.exists():
            parcial.unlink()
