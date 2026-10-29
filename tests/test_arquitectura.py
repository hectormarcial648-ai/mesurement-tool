"""Prueba de arquitectura: verifica la regla hexagonal de dependencias.

    dominio     -> solo biblioteca estándar, numpy, cv2 y medicion.dominio
    aplicacion  -> además medicion.aplicacion (NUNCA adaptadores, config ni composicion)
    adaptadores -> dominio, aplicacion, otros adaptadores y config
Ejecutar:  python3 tests/test_arquitectura.py
"""
import ast
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent / "medicion"
PERMITIDO = {
    "dominio": ("medicion.dominio",),
    "aplicacion": ("medicion.dominio", "medicion.aplicacion"),
    "adaptadores": ("medicion.dominio", "medicion.aplicacion", "medicion.adaptadores", "medicion.config"),
}


def importaciones(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    for n in ast.walk(arbol):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            yield n.module


def main():
    errores, revisados = [], 0
    for archivo in sorted(RAIZ.rglob("*.py")):
        capa = archivo.relative_to(RAIZ).parts[0]
        if capa not in PERMITIDO:
            continue
        revisados += 1
        for mod in importaciones(archivo):
            if mod.startswith("medicion") and not mod.startswith(PERMITIDO[capa]):
                errores.append(f"{archivo.relative_to(RAIZ.parent)}: la capa '{capa}' no puede importar {mod}")
    print(f"{revisados} archivos revisados.")
    for e in errores:
        print("  VIOLACIÓN:", e)
    print("ARQUITECTURA OK" if not errores else f"{len(errores)} violación(es)")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
