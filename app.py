from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO

app = Flask(__name__)
model = YOLO("yolov8n.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model(frame, imgsz=320)
    boxes = results[0].boxes
    biggest_box = None
    max_height = 0

    for box in boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            if height > max_height:
                max_height = height
                biggest_box = (x1, y1, x2, y2)

    if biggest_box:
        x1, y1, x2, y2 = biggest_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "Tallest person",
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    _, buffer = cv2.imencode(".jpg", frame)
    return send_file(BytesIO(buffer.tobytes()), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
