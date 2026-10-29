#!/usr/bin/env python3
"""Genera medicion.pyz: el programa completo en UN solo archivo ejecutable (zipapp de la biblioteca estándar).

    python3 build.py            ->  medicion.pyz
    python3 medicion.pyz --camaras gige,csi --clave tuClave
"""
import shutil
import sys
import tempfile
import zipapp
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "medicion"
        shutil.copytree(RAIZ / "medicion", destino, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        salida = RAIZ / "medicion.pyz"
        zipapp.create_archive(tmp, salida, main="medicion.main:main", interpreter="/usr/bin/env python3", compressed=True)
    print(f"Generado {salida} ({salida.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
