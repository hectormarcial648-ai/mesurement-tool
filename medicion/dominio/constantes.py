"""Constantes del dominio de visión."""

DIF_ESCALA = 2.0   # factor para llevar la distancia de color a 0..255


RADIO_MIN = 8.0   # semiancho mínimo (px) de la ventana donde se ubica el borde subpíxel


FRAC_UMBRAL_MONO = 0.65  # en gris el umbral de la máscara se sube un poco: deja fuera la penumbra


PESO_L = 0.35      # peso de la luminosidad (bajo => sombras casi no cuentan)
