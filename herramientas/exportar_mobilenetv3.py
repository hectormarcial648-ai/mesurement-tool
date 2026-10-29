#!/usr/bin/env python3
"""Exporta MobileNetV3-Small (preentrenada en ImageNet) a ONNX, para usarla como red de huellas.

Correr en una PC (no en la Pi):
    pip install torch torchvision onnx
    python3 exportar_mobilenetv3.py            ->  mobilenetv3_small.onnx   (cópialo a la carpeta del programa)

La salida es el vector de 576 componentes antes de las 1000 clases (la "huella" de la imagen).

AVISO: este script NO fue probado por quien lo escribió (requiere torch, que no se pudo instalar en su entorno).
Si falla, cualquier ONNX de imágenes con entrada [1,3,224,224] sirve: el programa usa su primera salida.
"""
import torch
import torchvision

modelo = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
modelo.classifier = torch.nn.Identity()          # features -> avgpool -> flatten: vector de 576
torch.onnx.export(modelo, torch.zeros(1, 3, 224, 224), "mobilenetv3_small.onnx", input_names=["input"],
                  output_names=["huella"], opset_version=13)
print("Listo: mobilenetv3_small.onnx")
