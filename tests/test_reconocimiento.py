"""Prueba autónoma del reconocimiento de piezas (solo dominio; sin cámaras ni servidor).

Dibuja un catálogo sintético de 6 piezas (dos de ellas distintas solo por sus huecos), aprende su huella con giros y
posiciones al azar, y comprueba: reconocimiento correcto, rechazo de piezas ajenas, invariancia al giro y tolerancia a un
error de escala del 3 %. Ejecutar:  python3 tests/test_reconocimiento.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medicion.dominio.piezas import grupos, huella_geometrica, parche_alineado, pisos
from medicion.dominio.reconocimiento import ClasificadorHuellas

N, PPM, SS = 700, 8.0, 3            # imagen, píxeles por mm, sobremuestreo
CAT = {"A": ("rect", dict(L=40, A=25)), "B": ("rect", dict(L=40, A=30)), "C": ("rect", dict(L=60, A=20)),
       "D": ("circ", dict(D=30)), "E": ("rect", dict(L=30, A=30, huecos=[(0, 0, 8)])),
       "F": ("rect", dict(L=40, A=25, huecos=[(-9, 0, 5), (9, 0, 5)]))}
AJENAS = {"33x22": ("rect", dict(L=33, A=22)), "50x35": ("rect", dict(L=50, A=35)), "círculo 22": ("circ", dict(D=22)),
          "40x25 con 1 hueco": ("rect", dict(L=40, A=25, huecos=[(0, 0, 5)]))}


def dibujar(tipo, p, ang, dx, dy, ruido=2.0):
    big = np.full((N * SS, N * SS), 150, np.uint8)
    cx, cy = (N / 2 + dx) * SS, (N / 2 + dy) * SS
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if tipo == "circ":
        cv2.circle(big, (int(cx), int(cy)), int(p["D"] / 2 * PPM * SS), 45, -1)
    else:
        L, A = p["L"] * PPM * SS, p["A"] * PPM * SS
        pts = np.array([[-L / 2, -A / 2], [L / 2, -A / 2], [L / 2, A / 2], [-L / 2, A / 2]]) @ R.T + [cx, cy]
        cv2.fillPoly(big, [np.round(pts).astype(np.int32)], 45)
    for hx, hy, hd in p.get("huecos", []):
        c = np.array([hx, hy]) * PPM * SS @ R.T + [cx, cy]
        cv2.circle(big, (int(c[0]), int(c[1])), int(hd / 2 * PPM * SS), 150, -1)
    img = cv2.GaussianBlur(cv2.resize(big, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.2)
    return np.clip(img + np.random.randn(N, N) * ruido, 0, 255).astype(np.uint8)


def objeto(img):
    """Detector mínimo: el contorno más grande -> el diccionario que usa el sistema (esquinas, centro, tipo)."""
    _, bw = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    c = max(cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    return {"corners_px": cv2.boxPoints(((cx, cy), (w, h), ang)), "centro": (cx, cy), "largo_px": max(w, h), "ancho_px": min(w, h), "tipo": "rect"}


def huella(tipo, p, rng, escala=1.0, **kw):
    img = dibujar(tipo, p, float(rng.uniform(0, 360)), float(rng.uniform(-90, 90)), float(rng.uniform(-90, 90)), **kw)
    o = objeto(img)
    parche, info = parche_alineado(img, o)
    return huella_geometrica(parche, info, escala / PPM)


def main():
    rng = np.random.default_rng(1)
    np.random.seed(1)
    entren = {k: np.array([huella(t, p, rng) for _ in range(20)]) for k, (t, p) in CAT.items()}
    pa, pr = pisos()
    clf = ClasificadorHuellas("geo", piso_abs=pa, piso_rel=pr, grupos=grupos())
    clf.ajustar(entren)
    fallos = []

    ok = tot = 0
    for k, (t, p) in CAT.items():
        for _ in range(12):
            r = clf.identificar(huella(t, p, rng))
            tot += 1
            ok += r["estado"] == "reconocida" and r["id"] == k
    print(f"Piezas conocidas bien reconocidas: {ok}/{tot}")
    if ok < 0.97 * tot:
        fallos.append("reconocimiento de piezas conocidas por debajo del 97 %")

    ajenas_ok = tot2 = 0
    for nombre, (t, p) in AJENAS.items():
        for _ in range(10):
            r = clf.identificar(huella(t, p, rng))
            tot2 += 1
            ajenas_ok += r["estado"] == "desconocida"
    print(f"Piezas ajenas rechazadas como DESCONOCIDAS: {ajenas_ok}/{tot2}")
    if ajenas_ok < 0.97 * tot2:
        fallos.append("se aceptan piezas ajenas")

    ok3 = tot3 = 0
    for k, (t, p) in CAT.items():
        for _ in range(8):
            r = clf.identificar(huella(t, p, rng, escala=1.03))
            tot3 += 1
            ok3 += r["estado"] == "reconocida" and r["id"] == k
    print(f"Con la escala 3 % mal calibrada: {ok3}/{tot3}")
    if ok3 < 0.95 * tot3:
        fallos.append("no tolera un error de escala del 3 %")

    ev = clf.evaluar(entren)
    peor = min(ev["separacion"], key=lambda s: s["dist"] / s["suma_radios"])
    print(f"Par más parecido: {peor['a']} ↔ {peor['b']} (separación {peor['dist'] / peor['suma_radios']:.1f}×; debe ser > 1)")
    if not all(s["separadas"] for s in ev["separacion"]):
        fallos.append("hay piezas del catálogo que no quedan separadas")
    print("RECONOCIMIENTO OK" if not fallos else "FALLOS: " + "; ".join(fallos))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
