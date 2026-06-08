# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="Obstacle Detection System", layout="wide")

st.title("🚗 Obstacle Detection System")
st.write("YOLOv8-based Obstacle Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist"
}

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

def estimate_distance(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)

    if area > 80000:
        return 3.0, "Danger 🔴", "red"
    elif area > 30000:
        return 7.0, "Warning 🟡", "orange"
    else:
        return 15.0, "Safe 🟢", "green"

def detect_objects(image):
    results = model.predict(image, conf=0.25, verbose=False)

    draw = ImageDraw.Draw(image)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

            distance, status, color = estimate_distance(x1, y1, x2, y2)

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=4
            )

            draw.text(
                (x1, max(0, y1 - 18)),
                f"{label} {conf:.2f} | {distance:.1f}m | {status}",
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

    result_image, detections = detect_objects(image)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(result_image, use_column_width=True)

    with col2:
        st.subheader("Detection Info")

        if len(detections) == 0:
            st.warning("No object detected.")
        else:
            for i, det in enumerate(detections, start=1):
                st.markdown(f"### Object {i}")
                st.write(f"Object: {det['Object']}")
                st.write(f"Confidence: {det['Confidence']}")
                st.write(f"Distance: {det['Distance']}")
                st.write(f"Status: {det['Status']}")
                st.markdown("---")
else:
    st.info("Please upload an image.")
