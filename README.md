# Medición por visión — arquitectura hexagonal

Programa de medición con interfaz web para Raspberry Pi 4B (cámaras PoE GigE, CSI y ToF). Funciona igual que
`medir_web.py`; solo cambió la organización del código.

## Ejecutar
```bash
python3 medicion.pyz --camaras gige,csi --clave tuClave     # un solo archivo
python3 -m medicion --camaras gige,csi --clave tuClave      # desde el código fuente (esta carpeta)
python3 build.py                                            # regenera medicion.pyz
python3 tests/test_arquitectura.py                          # verifica la regla de dependencias
python3 tests/test_reconocimiento.py                        # reconocimiento de piezas con un catálogo sintético
```
Los datos (calibraciones, CSV, capturas) se guardan junto al `.pyz` (o junto a esta carpeta); `MEDICION_DIR` lo cambia.

## Estructura
```
medicion/
  dominio/          reglas de visión puras: escala, máscara, bordes subpíxel, formas, altura, tablero
                    (solo numpy/OpenCV; no conoce cámaras, archivos ni red)
  aplicacion/
    puertos.py      contratos: PuertoCamara, PuertoSegmentador, PuertoRepoCalibracion, PuertoRepoMediciones
    sistema.py      Sistema: sesión de medición de UNA cámara (calibración, detección, medición, precisión)
    biblioteca.py   BibliotecaPiezas: fichas, aprendizaje, reconocimiento, veredicto, pieza maestra
  adaptadores/
    camaras/        gige.py · csi.py · tof.py · usb.py
    almacenamiento/ calibracion_archivos.py (JSON/NPZ) · mediciones_csv.py · piezas_archivos.py (biblioteca)
    reconocimiento_cnn.py  extractor de huellas con red ONNX (opcional)
    segmentadores/  clasico.py (predeterminado) · onnx.py (red) · conmutable.py (enciende/apaga la red en caliente) · descarga.py
    web/            worker.py (hilo por cámara) · servidor.py (HTTP/MJPEG/API) · ui.html (interfaz)
  config.py         línea de comandos y rutas
  composicion.py    ÚNICO lugar donde se eligen las implementaciones concretas
  main.py           arranque
tests/test_arquitectura.py
```
**Regla:** las dependencias apuntan hacia adentro. `dominio` no importa nada del proyecto fuera de `dominio`;
`aplicacion` solo importa `dominio` y `aplicacion`; los `adaptadores` importan hacia adentro. La prueba
`tests/test_arquitectura.py` lo verifica automáticamente.

## Cómo extender
- **Otra cámara:** crea `adaptadores/camaras/mi_camara.py` con `read()` y `close()` (y, si aplica, `set_roi`,
  `set_modo`, `set_exposicion`, `geometria_sensor`... ver `PuertoCamara`) y regístrala en `composicion.crear_camara`.
- **Otro segmentador:** implementa `segmentar(frame, bg_lab, ref, min_diff, thr, erode) -> (mascara, umbral, mapa)` y
  `nombre`; regístralo en `composicion.crear_segmentador`.
- **Otro almacenamiento** (p. ej. SQL Server en vez de CSV): implementa `PuertoRepoMediciones` y cámbialo en `composicion.armar`.
- **Pruebas sin hardware:** inyecta una cámara falsa y repositorios en memoria al construir `Sistema`.

## Segmentación con red neuronal (opcional)
Se maneja desde la interfaz web, pestaña **Red neuronal** de cada cámara:
1. Pulsa **Descargar u2netp** (4.7 MB; se verifica su SHA-256) o copia tu propio `.onnx` a la carpeta del programa.
2. Pulsa **Encender red**. El botón pasa a **Apagar red** y aparece el chip «red: ON» en la cabecera de la cámara.
3. Ajusta modo, umbral, margen y «ejecutar cada N fotogramas», y pulsa **Aplicar ajustes** (se aplican al instante).
4. Pulsa **Comparar ahora** para ver, sobre una misma imagen, el método clásico contra la red (restringir / rescatar).

Requiere `onnxruntime` en la Pi: `pip install onnxruntime --break-system-packages` (la interfaz avisa si falta).
Por línea de comandos solo se decide el estado inicial: `--segmentador onnx --seg-modelo u2netp.onnx` la deja encendida al arrancar.

- La red **propone** dónde está la pieza y el método clásico **afina el borde**: la precisión sigue siendo la del método clásico.
- `restringir` (recomendado) quita lo que la red no marca como pieza; `rescatar` usa además la máscara de la red si el
  método clásico no halla nada (menos preciso: ~0.2 mm en las pruebas).
- **Limitación medida:** U²-Net es una red de "objeto principal": con varias piezas en la mesa puede descartar las pequeñas
  (descartó una de 16×10 mm). Úsala con una pieza a la vez, o entrena un modelo con tus piezas y conéctalo con
  `--seg-tam`, `--seg-salida`, `--seg-media`, `--seg-desv`.
- Si la red falla estando encendida, se apaga sola y la medición sigue con el método clásico.
- Costo: la red se ejecuta solo cuando la escena cambia o cada N fotogramas. Mide su tiempo en tu Pi.
- Los ajustes de la interfaz se pierden al reiniciar el programa.

## Reconocimiento de piezas (pestaña **Piezas**)
El sistema aprende piezas, las reconoce y carga sus datos de metrología. **Todo se maneja desde la interfaz**: crear fichas,
enseñar, evaluar, reconocer, controlar con la pieza maestra y hacer respaldo.

**Flujo:** 1) Crea la ficha (nombre, medida nominal, tolerancia, altura). 2) Selecciónala y pulsa *Empezar a enseñar*: deja SOLO esa
pieza y muévela y gírala mientras captura (el video no se detiene). 3) Repite por pieza y por cámara (la OPT y la HQ aprenden por
separado). 4) *Evaluar biblioteca* (prueba dejar-una-fuera; avisa qué piezas se confunden o necesitan más muestras).
5) Activa *Reconocer en vivo*; al pulsar **Medir** aparece el veredicto **OK / LÍMITE / NO OK** contra la nominal.

**Qué hace y qué NO hace**
- Reconoce por una *huella*: forma, huecos, tamaño en mm, brillo/textura/color (geométrica, sin descargar nada). Opcionalmente suma una
  red de imágenes en ONNX (modo `cnn` o `ambos`, el más estricto).
- Si la pieza no se parece a ninguna enseñada → **DESCONOCIDA** y no carga nada. Si se parece a dos → **AMBIGUA**.
- La **receta** carga altura y medida de verificación; **nunca la escala** (depende del montaje, no de la pieza). No se usa la nominal
  para calibrar: forzar la medida a la nominal escondería el error que quieres medir.
- No distingue variantes casi iguales (diferencias de tamaño menores a ~7 % sin otra diferencia de forma): esas se separan midiendo.
- **Pieza maestra** (certificada, fuera de producción): *Verificar con la pieza maestra* la mide y compara con su nominal; si se aleja,
  la calibración se movió. Solo se registra si la pieza a la vista es la maestra. Avisa cuando el control está vencido.
- Veredictos en `veredictos_<cámara>.csv`; controles en `piezas/controles_<cámara>.csv`.

**Datos en disco:** `piezas/<id>/pieza.json` (ficha), `huellas_<cámara>.npz` (muestras), `miniatura.jpg`. Respaldo: *Descargar
biblioteca (.zip)* / *Importar* (fusiona; rechaza rutas peligrosas y fichas inválidas).

**Red de imágenes (opcional):** copia un `.onnx` (entrada [1,3,224,224], p. ej. MobileNetV3-Small) a la carpeta del programa,
elígelo en *Red neuronal para las huellas*, enciéndela y **vuelve a enseñar** las piezas. `herramientas/exportar_mobilenetv3.py`
genera uno en una PC (no probado por el autor). Valídalo con *Evaluar*: solo se probó el cableado con un modelo de relleno,
no con una MobileNet real.

**Dónde está en la arquitectura:** `dominio/piezas.py` (parche alineado, huella, veredicto) y `dominio/reconocimiento.py` (clasificador con
rechazo de desconocidas) son puros; `aplicacion/biblioteca.py` orquesta; los puertos son `PuertoRepoPiezas` y `PuertoExtractorCnn`;
los adaptadores `almacenamiento/piezas_archivos.py` y `reconocimiento_cnn.py`; la parte web está en `worker.py`, `servidor.py` y `ui.html`.

**Pruebas:** `python3 tests/test_reconocimiento.py` (autónoma, sin hardware) y `python3 tests/test_arquitectura.py`.

## Pendiente
`aplicacion/sistema.py` sigue siendo grande (~1000 líneas): el siguiente paso natural es dividirlo en casos de uso
(calibración, medición, precisión).

## Corrección por altura: huella (verde) vs silueta (cian)

Con una altura distinta de 0 y la distancia lente-mesa conocida, el sistema reconstruye la **huella real** de la pieza
(el rectángulo **verde**, etiqueta «(h)»), que queda más cerca del centro de la imagen que el borde que se ve.
La **silueta visible** se dibuja en **cian punteado** (interruptor en la pestaña Altura). El chip «altura …» del encabezado
muestra cuánto encoge el contorno (p. ej. «contorno −12.5 %») y abre la pestaña al hacer clic.
Si la pieza es plana o la altura no es la real, usa «Quitar la corrección (altura 0)».
