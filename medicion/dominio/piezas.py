"""Dominio de piezas: parche alineado, huella geométrica/apariencia y veredicto contra la medida nominal.

Todo es puro (numpy + OpenCV): no conoce cámaras, archivos ni red.
"""
import math

import cv2
import numpy as np

# Nombres de los componentes de la huella (el orden importa: es el orden del vector).
FEATURES = (["aspecto", "solidez", "relleno", "circularidad"] + [f"hu{i}" for i in range(1, 8)] +
            ["huecos_n", "huecos_area"] + [f"radial{i}" for i in range(1, 7)] +
            ["brillo", "dispersion", "textura", "bordes", "color_a", "color_b", "largo_mm", "ancho_mm"])
DIM = len(FEATURES)

# Tolerancias mínimas de cada componente, para que un cambio normal de calibración o de luz NO haga "desconocida" a una pieza:
#  - tamaño en mm: 3 % del valor típico (un error de escala de ~2 % no la rechaza);
#  - brillo/textura/color: variaciones habituales al cambiar exposición o iluminación.
# Grupos de componentes: una pieza solo se acepta si NINGÚN grupo se aleja (evita que muchas coincidencias "diluyan" una
# diferencia importante, p. ej. solo el tamaño).
GRUPOS = {"forma": ["aspecto", "solidez", "relleno", "circularidad"] + [f"hu{i}" for i in range(1, 8)] + [f"radial{i}" for i in range(1, 7)],
          "huecos": ["huecos_n", "huecos_area"],
          "apariencia": ["brillo", "dispersion", "textura", "bordes", "color_a", "color_b"],
          "tamano": ["largo_mm", "ancho_mm"]}
PISO_REL = {"largo_mm": 0.03, "ancho_mm": 0.03}
PISO_ABS = {"brillo": 0.04, "dispersion": 0.02, "textura": 0.02, "bordes": 0.03, "color_a": 0.02, "color_b": 0.02}


# ----------------------------------------------------------------------------------------------- parche alineado
def parche_alineado(frame, obj, margen=1.3):
    """Recorta la pieza girada de modo que su lado largo quede horizontal. Devuelve (parche, info) o (None, motivo).

    obj: diccionario de un objeto medido (corners_px, centro, largo_px, ancho_px, tipo).
    """
    h, w = frame.shape[:2]
    if obj.get("tipo") == "circulo":
        d = float(obj["largo_px"])
        lado = int(math.ceil(d * margen))
        cx, cy = obj["centro"]
        M = np.float32([[1, 0, lado / 2 - cx], [0, 1, lado / 2 - cy]])
        parche = cv2.warpAffine(frame, M, (lado, lado), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return parche, {"largo_px": d, "ancho_px": d, "tipo": "circulo"}
    c = np.asarray(obj.get("corners_px"), np.float32)
    if c.shape != (4, 2):
        return None, "el objeto no tiene esquinas"
    e1, e2 = c[1] - c[0], c[2] - c[1]
    largo_v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    largo, ancho = float(max(np.linalg.norm(e1), np.linalg.norm(e2))), float(min(np.linalg.norm(e1), np.linalg.norm(e2)))
    if ancho < 8:
        return None, "pieza demasiado delgada"
    ang = math.degrees(math.atan2(largo_v[1], largo_v[0]))
    centro = c.mean(axis=0)
    wc, hc = int(math.ceil(largo * margen)), int(math.ceil(ancho * margen))
    M = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), ang, 1.0)
    M[0, 2] += wc / 2 - centro[0]
    M[1, 2] += hc / 2 - centro[1]
    parche = cv2.warpAffine(frame, M, (wc, hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    parche = _orientar(parche)
    return parche, {"largo_px": largo, "ancho_px": ancho, "tipo": obj.get("tipo", "rect")}


def _orientar(parche):
    """Elimina la ambigüedad de 180°: gira el parche para que la asimetría de la pieza (3.er momento en x) sea positiva."""
    gris = parche if parche.ndim == 2 else cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY)
    m, _ = _mascara_pieza(gris)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return parche
    dx = xs - xs.mean()
    asim = float((dx ** 3).mean()) / (float(dx.std()) ** 3 + 1e-9)
    return cv2.rotate(parche, cv2.ROTATE_180) if asim < -0.02 else parche


def para_cnn(parche, tam=224):
    """Parche -> imagen cuadrada tam x tam x 3 (BGR) conservando la proporción (rellena con el fondo)."""
    img = parche if parche.ndim == 3 else cv2.cvtColor(parche, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    borde = np.concatenate([img[0].reshape(-1, 3), img[-1].reshape(-1, 3), img[:, 0].reshape(-1, 3), img[:, -1].reshape(-1, 3)])
    fondo = np.median(borde, axis=0).astype(np.uint8)
    k = tam / max(h, w)
    nw, nh = max(1, int(round(w * k))), max(1, int(round(h * k)))
    peq = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.empty((tam, tam, 3), np.uint8)
    out[:] = fondo
    y0, x0 = (tam - nh) // 2, (tam - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = peq
    return out


# ---------------------------------------------------------------------------------------- huella geométrica
def _mascara_pieza(gris):
    borde = np.concatenate([gris[0], gris[-1], gris[:, 0], gris[:, -1]])
    fondo = float(np.median(borde))
    d = np.abs(gris.astype(np.float32) - fondo)
    d8 = np.clip(d * (255.0 / max(float(d.max()), 1.0)), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    umbral = max(float(t) * float(d.max()) / 255.0, 10.0)
    m = (d > umbral).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m, fondo


def huella_geometrica(parche, info, mm_por_px=None):
    """Vector de DIM componentes (float32, NaN = no disponible). Invariante a giro y reflejo."""
    gris = parche if parche.ndim == 2 else cv2.cvtColor(parche, cv2.COLOR_BGR2GRAY)
    f = np.full(DIM, np.nan, np.float32)
    m, fondo = _mascara_pieza(gris)
    cnts, jer = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    # contorno exterior más grande
    ext = [i for i in range(len(cnts)) if jer[0][i][3] < 0]
    i0 = max(ext, key=lambda i: cv2.contourArea(cnts[i]))
    c0 = cnts[i0]
    area = float(cv2.contourArea(c0))
    if area < 50:
        return None
    per = float(cv2.arcLength(c0, True))
    hull_a = float(cv2.contourArea(cv2.convexHull(c0)))
    x, y, w, h = cv2.boundingRect(c0)
    rect = cv2.minAreaRect(c0)
    rw, rh = max(rect[1]), max(min(rect[1]), 1e-6)
    f[0] = rh / max(rw, 1e-6)                                   # aspecto (<= 1)
    f[1] = area / max(hull_a, 1e-6)                             # solidez
    f[2] = area / max(rw * rh, 1e-6)                            # relleno del rectángulo mínimo
    f[3] = 4 * math.pi * area / max(per * per, 1e-6)            # circularidad
    hu = cv2.HuMoments(cv2.moments(c0)).flatten()
    f[4:11] = [-math.copysign(1.0, v) * math.log10(abs(v)) if abs(v) > 1e-30 else 0.0 for v in hu]
    # huecos (contornos hijos)
    huecos = [cv2.contourArea(cnts[i]) for i in range(len(cnts)) if jer[0][i][3] == i0 and cv2.contourArea(cnts[i]) > 0.004 * area]
    f[11] = float(len(huecos))
    f[12] = float(sum(huecos)) / area
    # firma radial (magnitudes de Fourier: invariantes a giro)
    M0 = cv2.moments(c0)
    cx, cy = M0["m10"] / M0["m00"], M0["m01"] / M0["m00"]
    pts = c0.reshape(-1, 2).astype(np.float64)
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    orden = np.argsort(ang)
    grid = np.linspace(-math.pi, math.pi, 128, endpoint=False)
    rr = np.interp(grid, ang[orden], r[orden], period=2 * math.pi)
    rr = rr / max(rr.mean(), 1e-6)
    F = np.abs(np.fft.rfft(rr)) / len(rr)
    f[13:19] = F[1:7]
    # apariencia dentro de la pieza
    relleno = np.zeros_like(m)
    cv2.drawContours(relleno, [c0], -1, 255, -1)
    interior = cv2.erode(relleno, np.ones((5, 5), np.uint8)) > 0
    if interior.sum() > 30:
        g = gris.astype(np.float32)
        f[19] = (float(g[interior].mean()) - fondo) / 255.0
        f[20] = float(g[interior].std()) / 255.0
        lap = cv2.Laplacian(gris, cv2.CV_32F)
        f[21] = float(lap[interior].std()) / 255.0
        f[22] = float((cv2.Canny(gris, 60, 140)[interior] > 0).mean())
        if parche.ndim == 3:
            lab = cv2.cvtColor(parche, cv2.COLOR_BGR2LAB).astype(np.float32)
            fondo_lab = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
            f[23], f[24] = (lab[interior, 1].mean() - fondo_lab[1]) / 255.0, (lab[interior, 2].mean() - fondo_lab[2]) / 255.0
    if mm_por_px:
        f[25] = float(info["largo_px"]) * mm_por_px
        f[26] = float(info["ancho_px"]) * mm_por_px
    return f


# ----------------------------------------------------------------------------------------------- veredicto
def veredicto(medidas, pieza):
    """Compara la medida con la nominal de la pieza.

    medidas: {"largo", "ancho", "tipo"};  pieza: {"largo", "ancho", "diametro", "tol"}.
    OK si |error| <= 0.8 * tol; LIMITE si <= tol; NO OK si > tol. El resultado global es el peor.
    """
    tol = pieza.get("tol")
    filas = []
    if medidas.get("tipo") == "circulo" and pieza.get("diametro"):
        pares = [("diámetro", medidas["largo"], pieza["diametro"])]
    else:
        pares = []
        if pieza.get("largo"):
            pares.append(("largo", medidas["largo"], pieza["largo"]))
        if pieza.get("ancho"):
            pares.append(("ancho", medidas["ancho"], pieza["ancho"]))
    peor = 0
    for nombre, med, nom in pares:
        err = float(med) - float(nom)
        if tol:
            nivel = 0 if abs(err) <= 0.8 * tol else (1 if abs(err) <= tol else 2)
        else:
            nivel = None
        if nivel is not None:
            peor = max(peor, nivel)
        filas.append({"dimension": nombre, "medido": float(med), "nominal": float(nom), "error": err, "nivel": nivel})
    if not filas:
        return {"estado": "SIN NOMINAL", "filas": [], "tol": tol}
    if not tol:
        return {"estado": "SIN TOLERANCIA", "filas": filas, "tol": None}
    return {"estado": ("OK", "LIMITE", "NO OK")[peor], "filas": filas, "tol": float(tol)}


def pisos():
    """(piso_abs, piso_rel) como vectores de longitud DIM, en el orden de FEATURES."""
    a = np.array([PISO_ABS.get(n, 0.0) for n in FEATURES], np.float64)
    r = np.array([PISO_REL.get(n, 0.0) for n in FEATURES], np.float64)
    return a, r


def grupos():
    """Lista de arrays de índices, uno por grupo de GRUPOS."""
    return [np.array([FEATURES.index(n) for n in nombres]) for nombres in GRUPOS.values()]
