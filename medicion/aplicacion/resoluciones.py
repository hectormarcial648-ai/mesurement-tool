"""Preajustes de resolución (ROI:reducción)."""

def parse_resoluciones(texto):
    """'1280:1,2560:2,5120:4' -> [(1280, 1), (2560, 2), (5120, 4)]  (ROI en px : reducción para procesar)."""
    pres = []
    for item in str(texto or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            a, b = item.split(":")
            r, sc = int(a), int(b)
            if r < 0 or sc < 1:
                raise ValueError
        except ValueError:
            raise SystemExit(f"--resoluciones: '{item}' no es válido. Usa ROI:REDUCCIÓN separados por comas, "
                             "p. ej. 1280:1,2560:2,5120:4")
        pres.append((r, sc))
    return pres
