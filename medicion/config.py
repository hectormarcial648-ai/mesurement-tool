"""Configuración: directorio de datos, carpetas y línea de comandos (adaptador de entrada CLI)."""
import argparse
import os
from pathlib import Path


def _directorio_datos():
    """Donde viven calibraciones, CSV y capturas: MEDICION_DIR, o junto al .pyz, o junto a la carpeta del paquete."""
    env = os.environ.get("MEDICION_DIR")
    if env:
        return Path(env).resolve()
    f = Path(__file__).resolve()
    for p in f.parents:
        if p.suffix == ".pyz" and p.is_file():
            return p.parent
    return f.parent.parent


BASE = _directorio_datos()
CAPTURAS_DIR = BASE / "capturas"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Medición por visión con interfaz web (Raspberry Pi 4B)")
    ap.add_argument('--ip', default=None, help='gige: IP de la cámara (por defecto la primera detectada)')
    ap.add_argument('--exposure', type=float, default=8000, help='gige: exposición en us')
    ap.add_argument('--gain', type=float, default=None, help='gige: ganancia en dB')
    ap.add_argument('--roi-size', type=int, default=1280, help='gige: lado (px) del recorte central del sensor (0 = sensor completo; la Pi 4 no lo aguanta)')
    ap.add_argument('--roi-h', type=int, default=None, help='gige: alto del recorte (por defecto igual a --roi-size)')
    ap.add_argument('--decimation', type=int, default=1, help='gige: decimation NxN si la cámara lo soporta')
    ap.add_argument('--proc-scale', type=int, default=1, help='Procesa la imagen reducida N veces (p. ej. --roi-size 5120 --proc-scale 4). Con cámara mono, los bordes se refinan en la captura completa')
    ap.add_argument('--resoluciones', default='1280:1,2560:2,5120:4', help='gige: resoluciones a alternar con la tecla z, como ROI:REDUCCIÓN separadas por comas (0 = sensor completo)')
    ap.add_argument('--hires-vivo', action='store_true', help='Refinar bordes a resolución completa también en el video en vivo (más lento)')
    ap.add_argument('--tof-range', type=int, default=2000, choices=(2000, 4000), help='Rango ToF en mm')
    ap.add_argument('--tof-up', type=int, default=4, help='Factor de agrandado de la imagen ToF (240x180 x N)')
    ap.add_argument('--tof-span-mm', type=float, default=100.0, help='Semiancho de la ventana de profundidad alrededor de la mesa (mm)')
    ap.add_argument('--tof-centro-mm', type=float, default=None, help='Distancia a la mesa (mm); si no, se mide al arrancar')
    ap.add_argument('--tof-conf', type=float, default=30, help='Confianza mínima de un píxel de profundidad')
    ap.add_argument('--tof-image', choices=('depth', 'amplitude'), default='depth', help='Imagen que se analiza (tecla i para alternar)')
    ap.add_argument('--width', type=int, default=None, help='Solo CSI/USB. Por defecto: modo nativo del sensor (~1.4 MP). Más resolución = más lento')
    ap.add_argument('--height', type=int, default=None)
    ap.add_argument('--ref-mm', type=float, default=None, help='Lado largo de la pieza de referencia (tecla c)')
    ap.add_argument('--board', default='9x6', help='Esquinas internas del tablero COLSxFILAS')
    ap.add_argument('--square-mm', type=float, default=25.0, help='Lado de cada cuadro del tablero en mm')
    ap.add_argument('--lens-poses', type=int, default=12, help='Poses para calibrar la lente')
    ap.add_argument('--check-mm', type=float, nargs=2, default=None, metavar=('LARGO', 'ANCHO'), help='Medidas reales de la pieza patrón (tecla v)')
    ap.add_argument('--cam-dist-mm', type=float, default=None, help='Distancia lente-plano de trabajo')
    ap.add_argument('--altura-mm', '--obj-height-mm', dest='altura_mm', type=float, default=None, help='Altura (grosor) de la pieza a medir; también con la tecla a')
    ap.add_argument('--ref-alto-mm', type=float, default=0.0, help='Altura de la pieza de referencia usada con c (0 = plana)')
    ap.add_argument('--hfov-deg', type=float, default=None, help='Campo de visión horizontal (aprox.) para estimar la distancia sin calibrar la lente')
    ap.add_argument('--fondo-cada', type=int, default=15, help='Reajustar el fondo adaptativo cada N fotogramas')
    ap.add_argument('--min-area', type=float, default=3000, help='Área mínima en px²')
    ap.add_argument('--no-circles', action='store_true', help='No detectar circunferencias')
    ap.add_argument('--circ-tol', type=float, default=0.02, help='Tolerancia de circularidad (menor = más estricto)')
    ap.add_argument('--holes', action='store_true', help='Detectar huecos circulares desde el inicio')
    ap.add_argument('--min-side-px', type=float, default=20, help='Lado mínimo (px) para aceptar un objeto')
    ap.add_argument('--min-diff', type=float, default=18, help='Diferencia mínima de color vs fondo')
    ap.add_argument('--thr', type=float, default=0, help='Umbral manual 0-255 (0 = automático)')
    ap.add_argument('--erode', type=int, default=0, help='Px que se recortan hacia adentro (quita halo/sombra)')
    ap.add_argument('--frames', type=int, default=15, help='Fotogramas por medición promediada')
    ap.add_argument('--no-lock', action='store_true', help='No bloquear exposición/AWB')
    ap.add_argument('--lens-position', type=float, default=None, help='Enfoque manual (dioptrías, CSI v3)')
    ap.add_argument("--camaras", default="gige,csi",
                    help="Cámaras a abrir a la vez, separadas por comas: gige (PoE), csi (HQ por cinta flex), "
                         "tof (Arducam ToF; comparte el puerto CSI con la HQ) o usb:N")
    ap.add_argument("--host", default="0.0.0.0", help="Dirección donde escucha (0.0.0.0 = accesible desde la red)")
    ap.add_argument("--puerto", type=int, default=8080, help="Puerto del servidor web")
    ap.add_argument("--usuario", default="medicion", help="Usuario si se usa --clave")
    ap.add_argument("--clave", default=None, help="Si se indica, la interfaz pide usuario y clave")
    ap.add_argument("--tasa-vivo", type=float, default=4.0, help="Detecciones por segundo en el video en vivo")
    ap.add_argument("--vista-ancho", type=int, default=960, help="Ancho (px) del video que se envía al navegador")
    ap.add_argument("--vista-calidad", type=int, default=72, help="Calidad JPEG del video (20-95)")
    ap.add_argument("--vista-fps", type=float, default=10.0, help="Cuadros por segundo máximos del video")
    ap.add_argument("--calentar-min", type=float, default=30.0,
                    help="Minutos de calentamiento recomendados antes de calibrar (la imagen se desplaza mientras la cámara se calienta)")
    g = ap.add_argument_group("segmentación con red neuronal (opcional; requiere onnxruntime)")
    g.add_argument("--segmentador", choices=["clasico", "onnx"], default="clasico",
                   help="clasico = fondo + color (predeterminado); onnx = una red propone la pieza y el método clásico afina el borde")
    g.add_argument("--seg-modelo", default=None, help="Ruta del modelo .onnx (p. ej. u2netp.onnx)")
    g.add_argument("--seg-tam", type=int, default=320, help="Lado de la entrada de la red (px)")
    g.add_argument("--seg-salida", type=int, default=0, help="Índice de la salida del modelo que es el mapa de pieza")
    g.add_argument("--seg-umbral", type=float, default=0.5, help="Umbral de probabilidad de la red")
    g.add_argument("--seg-margen-px", type=int, default=6, help="Margen (px de trabajo) alrededor de la zona de la red")
    g.add_argument("--seg-modo", choices=["restringir", "rescatar"], default="restringir")
    g.add_argument("--seg-cada", type=int, default=10, help="Ejecutar la red como mucho cada N fotogramas (se reutiliza si la escena no cambia)")
    g.add_argument("--seg-media", default="0.485,0.456,0.406", help="Media de normalización (R,G,B)")
    g.add_argument("--seg-desv", default="0.229,0.224,0.225", help="Desviación de normalización (R,G,B)")
    return ap.parse_args(argv)
