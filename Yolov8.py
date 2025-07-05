"""
Ejercicio 1 - Detección de Personas con YOLOv8

Este script utiliza un modelo preentrenado YOLOv8 (versión pequeña: yolov8n)
para procesar un video y mostrarlo cuadro por cuadro.

Objetivo del ejercicio:
- Detectar únicamente personas (clase 'person') en los frames del video.
- Dibujar sobre la imagen únicamente el bounding box más grande.
  La definición de 'más grande' queda a criterio del alumno (altura o área).

Este código actualmente carga el modelo y reproduce el video,
pero **no realiza aún la detección ni graficado del bounding box más grande**.

Autor: [Tu Nombre]
Fecha: [Fecha]
"""

import time
import cv2
from ultralytics import YOLO

# Ejercicio 1

# Se pide al alumno utilizar Yolov8 para poder detectar en video a personas, vale decir, solo la clase personas del total de 80 clases que puede
# detectar Yolov8, asimismo, luego de detectar solo personas, analizar los resultados de la inferencia y graficar sobre la imagen solo el bounding box más grande.
# Se puede entender como bounding box mas grande a que tiene mas altura o mayor area, queda a eleccion del alumno.
# Se recomienda revisar la clase de Yolov8 y la documentacion oficial de esta para poder extraer los parametros que contienen los bounding boxes
# https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode

# Cargando modelo de Yolov8
# model = YOLO("yolov8n.yaml")  # Se elige version de YOLO
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Cargar modelo preentrenado
model = YOLO("yolov8n.pt")  # Modelo liviano

# Cargar video
cap = cv2.VideoCapture("dance.mp4")

# Obtener información del video original
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar salida de video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hacer inferencia con resolución reducida
    results = model(frame, imgsz=320)

    boxes = results[0].boxes
    biggest_box = None
    max_height = 0

    for box in boxes:
        cls_id = int(box.cls[0])  # clase detectada
        if cls_id == 0:  # solo personas
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height_box = y2 - y1

            if height_box > max_height:
                max_height = height_box
                biggest_box = (x1, y1, x2, y2)

    # Dibujar la caja más alta y mostrar altura
    if biggest_box:
        x1, y1, x2, y2 = biggest_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Tallest person",
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Height: {max_height}px",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Mostrar en pantalla
    cv2.imshow("image", frame)

    # ⚠️ Guardar el frame procesado en el archivo
    out.write(frame)

    # Espera según FPS
    delay = int(1000 / fps)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
out.release()  # <- importante para cerrar el archivo correctamente
cv2.destroyAllWindows()

print("✅ Video guardado como 'output.mp4'")
# Descargar un video que contenga de 0 a mas personas
# cap = cv2.VideoCapture("Video.mp4")  # Se carga el video a elegir por el alumno

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     ### Solucion del alumno
#     cv2.imshow("image", frame)

#     time.sleep(0.02)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()  # Delete cap
# cv2.destroyAllWindows()  # Close all opencv windows
