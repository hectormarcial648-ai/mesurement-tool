"""Dominio de reconocimiento: clasificador por huellas con rechazo de piezas DESCONOCIDAS (puro, solo numpy).

Dos espacios de huellas: 'geo' (vector de forma/apariencia, distancia en unidades de ruido) y 'cnn' (embedding,
distancia coseno). Cada pieza es un conjunto de muestras; se acepta una pieza si la distancia a su centroide cabe en su
radio de aceptación (calculado con las propias muestras). Si no cabe en ninguna: DESCONOCIDA.
"""
import warnings

import numpy as np


def _dist_geo(x, C, escala, grupos=None):
    """Distancia en unidades de ruido: máximo, sobre los grupos de componentes, del RMS del grupo (NaN se ignora)."""
    z = (x[None, :] - C) / escala[None, :]
    z2 = z * z
    ok = ~np.isnan(z2)
    z2 = np.where(ok, z2, 0.0)
    if not grupos:
        n = ok.sum(axis=1)
        return np.sqrt(z2.sum(axis=1) / np.maximum(n, 1)), n
    d = np.zeros(len(C))
    for g in grupos:
        n = ok[:, g].sum(axis=1)
        rms = np.sqrt(z2[:, g].sum(axis=1) / np.maximum(n, 1))
        d = np.maximum(d, np.where(n > 0, rms, 0.0))
    return d, ok.sum(axis=1)


def _dist_cos(x, C, escala=None):
    a = x / (np.linalg.norm(x) + 1e-9)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return 1.0 - Cn @ a, np.full(len(C), len(x))


class ClasificadorHuellas:
    """Ajuste y consulta para UN espacio de huellas ('geo' o 'cnn')."""

    def __init__(self, metrica="geo", factor_radio=1.6, piso_radio=None, min_muestras=5, piso_abs=None, piso_rel=None, grupos=None):
        self.piso_abs, self.piso_rel, self.grupos = piso_abs, piso_rel, grupos   # vectores por componente (solo métrica "geo")
        self.metrica = metrica
        self.factor_radio = factor_radio
        self.piso = piso_radio if piso_radio is not None else (2.2 if metrica == "geo" else 0.03)
        self.min_muestras = min_muestras
        self.ids, self.centros, self.radios, self.escala, self.n = [], None, None, None, {}

    # ----- ajuste ------------------------------------------------------------------------------------- #
    @staticmethod
    def _centro(X, metrica):
        if metrica == "geo":
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmedian(X, axis=0)
        v = X.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def _escalas(self, datos):
        """Escala por componente = ruido DENTRO de cada pieza (con un piso proporcional a la separación ENTRE piezas)."""
        todos = [X for X in datos.values() if len(X) >= 2]
        if self.metrica != "geo" or not todos:
            return None
        d = next(iter(datos.values())).shape[1]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dentro = np.nanmean([np.nanstd(X, axis=0) ** 2 for X in todos], axis=0) ** 0.5
            cents = np.array([np.nanmedian(X, axis=0) for X in datos.values()])
            entre = np.nanstd(cents, axis=0) if len(cents) > 1 else np.zeros(d)
            tipico = np.nanmedian(np.abs(cents), axis=0)
        dentro, entre, tipico = (np.nan_to_num(v, nan=0.0) for v in (dentro, entre, tipico))
        esc = np.sqrt(dentro ** 2 + (0.03 * entre) ** 2) + 1e-4
        if self.piso_abs is not None:
            esc = np.maximum(esc, self.piso_abs)
        if self.piso_rel is not None:
            esc = np.maximum(esc, self.piso_rel * tipico)
        return esc

    def ajustar(self, datos):
        """datos: {id: array (n, d)}. Calcula centroides, escalas y radios de aceptación."""
        self.ids = [k for k, X in datos.items() if len(X) >= 1]
        self.n = {k: int(len(datos[k])) for k in self.ids}
        if not self.ids:
            self.centros = self.radios = self.escala = None
            return
        self.escala = self._escalas({k: datos[k] for k in self.ids})
        if self.escala is None:
            self.escala = np.ones(next(iter(datos.values())).shape[1])
        self.centros = np.array([self._centro(datos[k], self.metrica) for k in self.ids])
        radios = []
        for i, k in enumerate(self.ids):
            X = datos[k]
            f = self._f
            if len(X) >= 3:  # radio con dejar-uno-fuera: la distancia de cada muestra al centroide SIN ella
                ds = []
                for j in range(len(X)):
                    resto = np.delete(X, j, axis=0)
                    c = self._centro(resto, self.metrica)
                    ds.append(f(X[j], c[None, :], self.escala)[0][0])
                r = self.factor_radio * float(np.percentile(ds, 95))
            else:
                r = self.piso * 2.0
            radios.append(max(r, self.piso))
        self.radios = np.array(radios)

    # ----- consulta --------------------------------------------------------------------------------------- #
    def _f(self, x, C, escala):
        if self.metrica == "geo":
            return _dist_geo(x, C, escala, self.grupos)
        return _dist_cos(x, C, escala)

    def distancias(self, x):
        return self._f(np.asarray(x, np.float32), self.centros, self.escala)

    def identificar(self, x, margen_ambiguo=1.25):
        """-> {'estado': reconocida|desconocida|ambigua|vacia, 'id', 'dist', 'radio', 'relativa', 'candidatas': [...]}"""
        if self.centros is None or not len(self.ids):
            return {"estado": "vacia", "id": None, "candidatas": []}
        d, n = self.distancias(x)
        rel = d / self.radios
        orden = np.argsort(rel)
        cand = [{"id": self.ids[i], "dist": float(d[i]), "radio": float(self.radios[i]), "relativa": float(rel[i])} for i in orden[:3]]
        mejor = cand[0]
        if mejor["relativa"] > 1.0:
            estado = "desconocida"
        elif len(cand) > 1 and cand[1]["relativa"] <= 1.0 and cand[1]["relativa"] < mejor["relativa"] * margen_ambiguo:
            estado = "ambigua"
        else:
            estado = "reconocida"
        return {"estado": estado, "id": mejor["id"] if estado in ("reconocida", "ambigua") else None, "dist": mejor["dist"],
                "radio": mejor["radio"], "relativa": mejor["relativa"], "candidatas": cand}

    # ----- evaluación ----------------------------------------------------------------------------------------- #
    def evaluar(self, datos):
        """Dejar-una-muestra-fuera: matriz de confusión y separación entre piezas (usa los datos con que se ajustó)."""
        ids = [k for k, X in datos.items() if len(X) >= 3]
        conf, tot = {}, 0
        res = {k: {"n": len(datos[k]), "ok": 0, "desconocida": 0, "ambigua": 0, "confundida_con": {}} for k in datos}
        for k in ids:
            X = datos[k]
            for j in range(len(X)):
                parcial = {kk: (np.delete(datos[kk], j, axis=0) if kk == k else datos[kk]) for kk in datos if len(datos[kk]) >= 2}
                if k not in parcial:
                    continue
                clf = ClasificadorHuellas(self.metrica, self.factor_radio, self.piso, self.min_muestras, self.piso_abs, self.piso_rel, self.grupos)
                clf.ajustar(parcial)
                r = clf.identificar(X[j])
                tot += 1
                if r["estado"] == "reconocida" and r["id"] == k:
                    res[k]["ok"] += 1
                elif r["estado"] == "desconocida":
                    res[k]["desconocida"] += 1
                elif r["estado"] == "ambigua":
                    res[k]["ambigua"] += 1
                    otro = r["candidatas"][1]["id"] if r["id"] == k else r["id"]
                    res[k]["confundida_con"][otro] = res[k]["confundida_con"].get(otro, 0) + 1
                else:
                    res[k]["confundida_con"][r["id"]] = res[k]["confundida_con"].get(r["id"], 0) + 1
        sep = []
        if self.centros is not None and len(self.ids) > 1:
            for a in range(len(self.ids)):
                d, _ = self.distancias(self.centros[a])
                for b in range(a + 1, len(self.ids)):
                    sep.append({"a": self.ids[a], "b": self.ids[b], "dist": float(d[b]),
                                "suma_radios": float(self.radios[a] + self.radios[b]),
                                "separadas": bool(d[b] > self.radios[a] + self.radios[b])})
        return {"por_pieza": res, "total": tot, "separacion": sep}


def combinar(r_geo, r_cnn):
    """Combina geo y cnn de forma CONSERVADORA: solo 'reconocida' si ambos coinciden; si discrepan -> 'ambigua'."""
    if r_cnn is None or r_cnn.get("estado") == "vacia":
        return r_geo
    if r_geo.get("estado") == "vacia":
        return r_cnn
    if r_geo["estado"] == "reconocida" and r_cnn["estado"] == "reconocida" and r_geo["id"] == r_cnn["id"]:
        out = dict(r_geo)
        out["relativa"] = max(r_geo["relativa"], r_cnn["relativa"])
        return out
    if r_geo["estado"] == "desconocida" and r_cnn["estado"] == "desconocida":
        return r_geo
    out = dict(r_geo)
    out["estado"] = "ambigua"
    out["id"] = None
    out["nota"] = "los descriptores no coinciden"
    return out
