# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🚗 Obstacle Detection System with YOLOv8")

# =========================
# Load YOLOv8 model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# =========================
# Class names
# =========================
class_names = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist"
}

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)


def estimate_distance(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)

    if area > 80000:
        distance = 3.0
        status = "Danger"
        color = "red"
    elif area > 30000:
        distance = 7.0
        status = "Warning"
        color = "yellow"
    else:
        distance = 15.0
        status = "Safe"
        color = "green"

    return distance, status, color


def run_detection(image):
    results = model(image, conf=0.25)

    draw = ImageDraw.Draw(image)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = class_names.get(cls_id, f"Class {cls_id}")

            distance, status, color = estimate_distance(x1, y1, x2, y2)

            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=4
            )

            text = f"{label} {conf:.2f} | {distance:.1f}m | {status}"

            draw.text(
                (x1, max(0, y1 - 20)),
                text,
                fill=color
            )

            detections.append({
                "Object": label,
                "Confidence": round(conf, 2),
                "Distance": f"{distance:.1f} m",
                "Status": status
            })

    return image, detections


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    result_image, detections = run_detection(image)

    col1, col2 = st.columns([2, 1])

    col1.image(result_image, use_container_width=True)

    col2.subheader("Detection Info")

    if len(detections) == 0:
        col2.warning("No object detected.")
    else:
        for i, det in enumerate(detections, 1):
            col2.markdown(f"### Object {i}")
            col2.write(f"Object: {det['Object']}")
            col2.write(f"Confidence: {det['Confidence']}")
            col2.write(f"Distance: {det['Distance']}")
            col2.write(f"Status: {det['Status']}")
            col2.markdown("---")
else:
    st.info("Please upload an image to start detection.")
