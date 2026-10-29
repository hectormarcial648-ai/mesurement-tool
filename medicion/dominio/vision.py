"""Visión pura: fondo, máscara, bordes subpíxel, formas, altura y tablero. Sin cámaras, archivos ni red."""
import cv2

import numpy as np

from medicion.dominio.constantes import DIF_ESCALA, FRAC_UMBRAL_MONO, PESO_L


def _es_mono(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def _fondo_uniforme(lab):
    """Color (o nivel de gris) de fondo = mediana de una franja en el borde de la imagen."""
    h, w = lab.shape[:2]
    m = max(8, int(0.02 * min(h, w)))
    if lab.ndim == 2:
        franjas = np.concatenate([lab[:m, :].ravel(), lab[-m:, :].ravel(),
                                  lab[:, :m].ravel(), lab[:, -m:].ravel()])[::5]
        ref = np.empty_like(lab)
        ref[:] = np.uint8(np.median(franjas))
        return ref
    franjas = np.concatenate(
        [
            lab[:m, :].reshape(-1, 3),
            lab[-m:, :].reshape(-1, 3),
            lab[:, :m].reshape(-1, 3),
            lab[:, -m:].reshape(-1, 3),
        ]
    )[::5]
    med = np.median(franjas, axis=0).astype(np.uint8)
    ref = np.empty_like(lab)
    ref[:] = med
    return ref


def fondo_polinomial(lab, grado=3, iters=3, lado=160):
    """Superficie suave (por canal Lab) ajustada SOLO a píxeles de mesa.

    Inspirado en la normalización de iluminación de VisionMetrix (dividir por el fondo de baja
    frecuencia), pero sin su blur 61x61 (que se contamina con la propia pieza) ni CLAHE (que
    deforma el perfil del borde): aquí la pieza se excluye del ajuste, así degradados de luz,
    viñeteo y manchas suaves quedan absorbidos sin necesidad de capturar la mesa vacía.
    """
    h, w = lab.shape[:2]
    f = max(1, int(round(w / lado)))
    ps, hs = max(8, w // f), max(8, h // f)
    small = cv2.resize(lab, (ps, hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    if small.ndim == 2:  # imagen en gris: un solo canal
        small = small[:, :, None]
    nc = small.shape[2]
    ys, xs = np.mgrid[0:hs, 0:ps].astype(np.float32)
    x = (xs / max(ps - 1, 1) * 2 - 1).ravel()
    y = (ys / max(hs - 1, 1) * 2 - 1).ravel()
    A = np.stack([x ** i * y ** j for i in range(grado + 1) for j in range(grado + 1 - i)], axis=1)
    Y = small.reshape(-1, nc)

    def dist(Z, M):
        d = Z - M
        if nc == 1:
            return np.abs(d[:, 0])
        return np.sqrt((PESO_L * d[:, 0]) ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)

    bg = dist(Y, np.median(Y, axis=0)) < 25.0  # arranque: lo parecido al color dominante (la mesa)
    coef = None
    for _ in range(iters):
        if bg.sum() < 8 * A.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(A[bg], Y[bg], rcond=None)
        r = dist(Y, A @ coef)
        rb = r[bg]
        T = max(8.0, np.median(rb) + 4.0 * 1.4826 * np.median(np.abs(rb - np.median(rb))))
        obj = (r >= T).reshape(hs, ps).astype(np.uint8)
        obj = cv2.dilate(obj, np.ones((3, 3), np.uint8), iterations=2)  # excluye la penumbra
        bg = (obj == 0).ravel()
    if coef is None:
        return None
    return cv2.resize((A @ coef).reshape(hs, ps, nc).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def estimar_fondo(frame):
    """Superficie de fondo (float32; Lab 8 bits, o gris) de un fotograma, o None si no se pudo ajustar."""
    if _es_mono(frame):
        return fondo_polinomial(cv2.GaussianBlur(frame.reshape(frame.shape[:2]), (5, 5), 0))
    return fondo_polinomial(cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB))


def calcular_mascara(frame, bg_lab, min_diff, thr_manual, erode_px, ref=None):
    """Máscara del objeto por distancia de color Lab contra el fondo.

    Fondo: el capturado con `b` (bg_lab), o `ref` (superficie ya estimada), o una superficie
    polinomial ajustada al fotograma. Devuelve (máscara, umbral, mapa de distancia float32).
    """
    mono = _es_mono(frame)
    if mono:  # cámara monocroma: no hay color, se usa la luminosidad completa
        lab = cv2.GaussianBlur(frame.reshape(frame.shape[:2]), (5, 5), 0)
    else:
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB)
    if bg_lab is not None:
        ref = bg_lab.astype(np.float32)
    elif ref is None:
        ref = fondo_polinomial(lab)
        if ref is None:
            ref = _fondo_uniforme(lab).astype(np.float32)
    d = lab.astype(np.float32) - ref
    if mono:
        distf = np.abs(d) * DIF_ESCALA
    else:
        distf = np.sqrt((PESO_L * d[:, :, 0]) ** 2 + d[:, :, 1] ** 2 + d[:, :, 2] ** 2) * DIF_ESCALA
    d8 = np.clip(distf, 0, 255).astype(np.uint8)

    piso = min_diff * DIF_ESCALA
    if thr_manual and thr_manual > 0:
        thr = float(thr_manual)
    else:
        otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = d8 > max(otsu, piso)
        if fg.any():
            # umbral al 50% de la transición fondo->objeto => borde sin sesgo
            thr = max(piso, (FRAC_UMBRAL_MONO if mono else 0.5) * float(np.median(d8[fg])))
        else:
            thr = piso
    bw = (d8 > thr).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3)
    if erode_px > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        bw = cv2.erode(bw, ke)
    return bw, thr, distf


def borde_subpixel(dist, cnt, hueco=False, paso=3, radio=6.0, nmuest=None):
    """Reubica el contorno al cruce del 50% del perfil de intensidad a lo largo de su normal.

    Cada perfil usa SUS propios niveles (dentro/fuera), así el borde no depende del umbral global
    ni de la sombra o iluminación local. Devuelve puntos (m, 2) float en px, o None.
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n < 30:
        return None
    k = max(4, n // 80)
    idx = np.arange(0, n, paso)
    p = pts[idx]
    t = pts[(idx + k) % n] - pts[(idx - k) % n]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
    c = pts.mean(axis=0)
    if np.sum(nrm * (p - c)) < 0:
        nrm = -nrm  # normal hacia afuera del contorno
    if hueco:
        nrm = -nrm  # en un hueco, "fuera del objeto" es hacia el centro del hueco
    nmuest = nmuest or int(round(8 * radio)) + 1
    meseta = 0.42 * radio  # el desenfoque (en px) crece con la resolución: la ventana también
    s = np.linspace(-radio, radio, nmuest, dtype=np.float32)
    mx = (p[:, 0:1] + nrm[:, 0:1] * s[None, :]).astype(np.float32)
    my = (p[:, 1:2] + nrm[:, 1:2] * s[None, :]).astype(np.float32)
    prof = cv2.remap(dist, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hi = np.median(prof[:, s <= -meseta], axis=1)
    lo = np.median(prof[:, s >= meseta], axis=1)
    contraste = hi - lo
    a = prof - (0.5 * (hi + lo))[:, None]
    cruce = (a[:, :-1] >= 0) & (a[:, 1:] < 0)
    frac = a[:, :-1] / np.maximum(a[:, :-1] - a[:, 1:], 1e-6)
    pos = np.where(cruce, s[None, :-1] + frac * (s[1] - s[0]), np.inf)
    sc = pos[np.arange(len(p)), np.argmin(np.abs(pos), axis=1)]
    ok = np.isfinite(sc) & (np.abs(sc) < radio - 1.0) & (contraste > max(12.0, 0.3 * np.median(contraste)))
    if ok.sum() < max(20, 0.5 * len(p)):
        return None
    return (p[ok] + nrm[ok] * sc[ok, None]).astype(np.float64)


def _interseccion(p1, d1, p2, d2):
    cruz = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cruz) < 1e-9:
        return None
    w = p2 - p1
    s = (w[0] * d2[1] - w[1] * d2[0]) / cruz
    return p1 + s * d1


def refinar_cuadrilatero(pts, box, desp=0.5):
    """Ajusta 4 rectas a los lados del contorno y devuelve las 4 esquinas (subpíxel)."""
    c = box.mean(axis=0)
    lado_min = min(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4))
    if lado_min < 10:
        return None
    tol = max(2.0, 0.04 * lado_min)
    lineas = []
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        n = np.array([-u[1], u[0]])
        rel = pts - a
        t = (rel @ u) / L
        m = (np.abs(rel @ n) < tol) & (t > 0.15) & (t < 0.85)
        if m.sum() < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            pts[m].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).flatten()
        dirv = np.array([vx, vy], np.float64)
        p0 = np.array([x0, y0], np.float64)
        nn = np.array([-vy, vx], np.float64)
        if nn @ (p0 - c) < 0:
            nn = -nn
        p0 = p0 + desp * nn  # contorno por píxeles: el centro del píxel de borde está 0.5 px dentro
        lineas.append((p0, dirv))
    esquinas = np.zeros((4, 2), np.float64)
    for j in range(4):
        p1, d1 = lineas[(j - 1) % 4]
        p2, d2 = lineas[j]
        q = _interseccion(p1, d1, p2, d2)
        if q is None or np.linalg.norm(q - box[j]) > 0.15 * lado_min:
            return None
        esquinas[j] = q
    return esquinas


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def ajustar_circulo(pu):
    """Ajuste de círculo por mínimos cuadrados. Devuelve (centro, radio, error_relativo)."""
    m = pu.mean(axis=0)
    q = pu - m
    A = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
    b = q[:, 0] ** 2 + q[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    if r <= 0:
        return m, 0.0, 1.0
    res = np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r
    return np.array([cx, cy]) + m, r, float(np.std(res) / r)


def _escala_local(escala, c_px):
    """Unidades por píxel alrededor de un punto (sirve también con homografía)."""
    c = np.asarray(c_px, np.float64)
    u = escala.a_unid([c, c + [1.0, 0.0], c + [0.0, 1.0]])
    v1, v2 = u[1] - u[0], u[2] - u[0]
    return float(np.sqrt(abs(v1[0] * v2[1] - v1[1] * v2[0])))


def intentar_circulo(cnt, P, escala, tol, hueco, comp=0.5):
    """Círculo/hueco circular. P = puntos del borde (px). comp = medio píxel si P es contorno por píxeles."""
    if len(P) < 40:
        return None
    pu = escala.a_unid(P)
    c, r, err = ajustar_circulo(pu)
    if r <= 0 or err > tol:
        return None
    ratio = cv2.contourArea(pu.astype(np.float32)) / (np.pi * r * r)
    if not (0.93 < ratio < 1.07):
        return None
    c_px = escala.a_px(c)[0]
    r += (-comp if hueco else comp) * _escala_local(escala, c_px)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    poly_px = escala.a_px(c + r * np.column_stack([np.cos(ang), np.sin(ang)]))
    d_px = 2.0 * np.sqrt(cv2.contourArea(poly_px.astype(np.float32)) / np.pi)
    return {
        "tipo": "hueco" if hueco else "circulo",
        "corners_px": poly_px,
        "largo": 2 * r,
        "ancho": 2 * r,
        "largo_px": d_px,
        "ancho_px": d_px,
        "centro": (float(c_px[0]), float(c_px[1])),
        "area_px": float(cv2.contourArea(cnt)),
        "cuad": False,
    }


def _recortar(poly, n, h):
    """Recorta el polígono convexo `poly` (k, 2) con el semiplano  n·x <= h."""
    d = poly @ n - h
    if np.all(d <= 0):
        return poly
    if np.all(d > 0):
        return poly[:0]
    out = []
    k = len(poly)
    for i in range(k):
        a, b, da, db = poly[i], poly[(i + 1) % k], d[i], d[(i + 1) % k]
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            out.append(a + (da / (da - db)) * (b - a))
    return np.array(out) if out else poly[:0]


def corregir_altura(pts_u, foot, m_t, m_b, k=72):
    """Huella real de un prisma a partir de su silueta (corrige el efecto de la altura).

    Geometría: desde la cámara, todo lo que está a altura z se ve ampliado, respecto al pie de la
    cámara (`foot`, en unidades del plano de calibración), por m(z) = (D - h_cal) / (D - z). La
    silueta es el casco de la cara superior (m_t * F) y de la base (m_b * F), con m_t >= m_b.
    Con funciones de soporte (origen en `foot`):  h_S(u) = m_t * h_F(u)  si h_F > 0, y
    m_b * h_F(u) si h_F < 0. Se invierte esa relación y F se reconstruye como la intersección de
    semiplanos {x : u·x <= h_F(u)}, recortando un polígono (robusto al ruido, a diferencia de
    intersectar rectas casi paralelas). Exacta para huellas convexas con paredes verticales y
    corrige también el término que depende de DÓNDE está la pieza (no solo una escala global).
    """
    if m_t <= 0 or m_b <= 0:
        return None
    q = np.asarray(pts_u, np.float64) - foot
    hull = cv2.convexHull(q.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return None
    # direcciones: normales exactas de las aristas largas del casco (lados rectos) + rejilla uniforme
    per = float(np.sum(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1)))
    simple = cv2.approxPolyDP(hull.astype(np.float32), 0.004 * per, True).reshape(-1, 2).astype(np.float64)
    e = np.roll(simple, -1, axis=0) - simple
    c = hull.mean(axis=0)
    dirs = []
    for v, a in zip(e, simple):
        ln = np.linalg.norm(v)
        if ln < 0.05 * per:
            continue
        nrm = np.array([v[1], -v[0]]) / ln
        if np.dot(nrm, a + v / 2 - c) < 0:
            nrm = -nrm
        # afinar la dirección con TODOS los puntos de ese lado; tolerancia decreciente para quedarse
        # con el segmento dominante (las paredes verticales son casi colineales con el lado)
        p0, dv = a.copy(), v / ln
        for tol in (0.004 * per, 0.0015 * per, 0.0007 * per):
            rel = q - p0
            tp = rel @ dv
            near = (np.abs(rel @ np.array([dv[1], -dv[0]])) < tol) & (tp > -0.02 * ln) & (tp < 1.02 * ln)
            if near.sum() < 10:
                break
            vx, vy, x0, y0 = cv2.fitLine(q[near].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            dv = np.array([vx, vy], np.float64)
            if np.dot(dv, v) < 0:
                dv = -dv
            p0 = np.array([x0, y0], np.float64)
        n2 = np.array([dv[1], -dv[0]])
        nrm = n2 if np.dot(n2, nrm) > 0 else -n2
        dirs.append(nrm)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    U = np.vstack([np.array(dirs).reshape(-1, 2), np.column_stack([np.cos(t), np.sin(t)])])
    hS = (hull @ U.T).max(axis=0)
    hF = np.where(hS > 0, hS / m_t, hS / m_b)
    # polígono inicial holgado y recortes (primero las aristas exactas)
    lo, hi = hull.min(axis=0), hull.max(axis=0)
    ext = 2.5 * (hi - lo).max() + 1.0
    poly = np.array([[lo[0] - ext, lo[1] - ext], [hi[0] + ext, lo[1] - ext],
                     [hi[0] + ext, hi[1] + ext], [lo[0] - ext, hi[1] + ext]])
    for n_, h_ in zip(U, hF):
        poly = _recortar(poly, n_, h_)
        if len(poly) < 3:
            return None
    return cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2).astype(np.float64) + foot


def _medir_prisma(P, comp, escala, corr, circ_tol, circulos):
    S = escala.a_unid(P)
    F = corregir_altura(S, corr["foot"], corr["m_t"], corr["m_b"])
    if F is None or len(F) < 4:
        return None
    F32 = F.astype(np.float32)
    rect = cv2.minAreaRect(F32)
    (cx, cy), (w, h), _ = rect
    area_F = cv2.contourArea(F32)
    c_u = np.array([cx, cy])
    c_px = escala.a_px(c_u)[0]
    s = _escala_local(escala, c_px)
    if circulos and 0.70 < area_F / max(w * h, 1e-9) < 0.88:
        c, r, err = ajustar_circulo(F)
        if r > 0 and err <= circ_tol and 0.93 < area_F / (np.pi * r * r) < 1.07:
            r += comp * s
            ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            poly_px = escala.a_px(c + r * np.column_stack([np.cos(ang), np.sin(ang)]))
            d_px = 2.0 * np.sqrt(cv2.contourArea(poly_px.astype(np.float32)) / np.pi)
            return {"tipo": "circulo", "corners_px": poly_px, "largo": 2 * r, "ancho": 2 * r,
                    "largo_px": d_px, "ancho_px": d_px,
                    "centro": tuple(float(v) for v in escala.a_px(c)[0]), "cuad": False, "alt_corr": True}
    cpx = escala.a_px(cv2.boxPoints(rect))
    w, h = w + 2 * comp * s, h + 2 * comp * s
    w_px = (_dist(cpx[0], cpx[1]) + _dist(cpx[3], cpx[2])) / 2
    h_px = (_dist(cpx[1], cpx[2]) + _dist(cpx[0], cpx[3])) / 2
    return {"tipo": "rect", "corners_px": cpx, "largo": max(w, h), "ancho": min(w, h),
            "largo_px": max(w_px, h_px), "ancho_px": min(w_px, h_px),
            "centro": (float(c_px[0]), float(c_px[1])), "cuad": True, "alt_corr": True,
            "sil_px": cv2.boxPoints(cv2.minAreaRect(np.asarray(P, np.float32)))}  # silueta visible (lo que se ve en la imagen)


def medir_contorno(cnt, escala, hueco=False, circ_tol=0.02, circulos=True, dist=None, corr=None, radio=6.0,
                   sub_pts=None):
    """Mide un contorno. dist = mapa de distancia (borde subpíxel); corr = corrección por altura."""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float64)
    area_c = cv2.contourArea(cnt)
    area_r = max(rect[1][0] * rect[1][1], 1.0)
    M = cv2.moments(cnt)
    centro = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] > 0 else tuple(box.mean(axis=0))

    if sub_pts is not None:  # borde ya refinado (p. ej. en la imagen de captura completa)
        sub = sub_pts
    else:
        sub = borde_subpixel(dist, cnt, hueco, radio=radio) if dist is not None else None
    P, comp = (sub, 0.0) if sub is not None else (pts, 0.5)

    if corr is not None and not hueco:  # pieza con altura: silueta -> huella real
        o = _medir_prisma(P, comp, escala, corr, circ_tol, circulos)
        if o is not None:
            o["area_px"] = float(area_c)
            return o

    ratio = area_c / area_r
    if (circulos or hueco) and 0.70 < ratio < 0.88:  # un círculo ocupa ~78.5% de su caja
        circ = intentar_circulo(cnt, P, escala, circ_tol, hueco, comp)
        if circ is not None:
            return circ
    if hueco:
        return None  # los huecos solo se reportan si son circulares

    esq = refinar_cuadrilatero(P, box, comp) if ratio > 0.93 else None  # casi rectangular -> rectas
    if esq is not None:
        cu = escala.a_unid(esq)
        w = (_dist(cu[0], cu[1]) + _dist(cu[3], cu[2])) / 2
        h = (_dist(cu[1], cu[2]) + _dist(cu[0], cu[3])) / 2
        corners_px = esq
        w_px = (_dist(esq[0], esq[1]) + _dist(esq[3], esq[2])) / 2
        h_px = (_dist(esq[1], esq[2]) + _dist(esq[0], esq[3])) / 2
        cuad = True
    else:  # forma libre -> rectángulo mínimo en unidades reales
        r = cv2.minAreaRect(escala.a_unid(P).astype(np.float32))
        w, h = r[1]
        corners_px = escala.a_px(cv2.boxPoints(r))
        w_px, h_px = rect[1]
        cuad = False
    return {
        "corners_px": corners_px,
        "largo": max(w, h),
        "ancho": min(w, h),
        "largo_px": max(w_px, h_px),
        "ancho_px": min(w_px, h_px),
        "centro": centro,
        "area_px": float(area_c),
        "cuad": cuad,
        "tipo": "rect",
    }


def buscar_tablero(frame, patron):
    gris = frame.reshape(frame.shape[:2]) if _es_mono(frame) else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, corners = False, None
    try:
        ok, corners = cv2.findChessboardCornersSB(
            gris, patron, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
    except Exception:
        ok = False
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            gris, patron, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ok:
            corners = cv2.cornerSubPix(
                gris, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
            )
    if not ok:
        return None
    return corners.reshape(-1, 2).astype(np.float64)


def puntos_tablero(patron, cuadro_mm):
    cols, rows = patron
    return np.array(
        [[i * cuadro_mm, j * cuadro_mm] for j in range(rows) for i in range(cols)], np.float64
    )
