# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO

st.set_page_config(page_title="Obstacle Detection System", layout="wide")

st.title("🚗 Obstacle Detection System")
st.write("YOLOv8 + Distance Estimation + Approaching Speed + TTC Risk Assessment")

@st.cache_resource
def load_model():
    return YOLO("best_camera.pt") 

model = load_model()

CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist"
}

st.sidebar.header("System Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01
)

approaching_speed = st.sidebar.slider(
    "Approaching Speed (m/s)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5
)

input_type = st.radio(
    "Choose Input Source",
    ["Upload Image", "Camera"]
)

input_file = None

if input_type == "Upload Image":
    input_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    input_file = st.camera_input("Take a picture")


def estimate_distance(x1, y1, x2, y2):
    """
    使用 Bounding Box 面積估算距離。
    框越大代表物體越近，框越小代表物體越遠。
    """

    area = max(1, (x2 - x1) * (y2 - y1))

    if area > 120000:
        return 2.0
    elif area > 70000:
        return 5.0
    elif area > 30000:
        return 10.0
    else:
        return 20.0


def assess_risk(distance_m, approaching_speed_mps):
    """
    根據距離與接近速度計算 TTC，並判斷危險程度。
    """

    if approaching_speed_mps <= 0:
        ttc = 999.0
    else:
        ttc = distance_m / approaching_speed_mps

    if distance_m <= 3 or ttc <= 1.0:
        return ttc, "Danger 🔴", "red"
    elif distance_m <= 10 or ttc <= 3.0:
        return ttc, "Warning 🟡", "orange"
    else:
        return ttc, "Safe 🟢", "green"


def detect_objects(image):
    results = model.predict(
        image,
        conf=confidence_threshold,
        imgsz=640,
        verbose=False
    )

    draw = ImageDraw.Draw(image)
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])

            if cls_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            conf = float(box.conf[0])
            label = CLASS_NAMES[cls_id]

            distance = estimate_distance(x1, y1, x2, y2)
            ttc, status, color = assess_risk(distance, approaching_speed)

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=4
            )

            text = (
                f"{label} {conf:.2f} | "
                f"Dist:{distance:.1f}m | "
                f"Speed:{approaching_speed:.1f}m/s | "
                f"TTC:{ttc:.1f}s | "
                f"{status}"
            )

            draw.text(
                (x1, max(0, y1 - 22)),
                text,
                fill=color
            )

            detections.append({
                "Object": label,
                "Confidence": round(conf, 2),
                "Distance": f"{distance:.1f} m",
                "Approaching Speed": f"{approaching_speed:.1f} m/s",
                "TTC": f"{ttc:.1f} s",
                "Status": status
            })

    return image, detections


with st.expander("Model Information"):
    st.write("Model Classes:", model.names)

if input_file is not None:
    image = Image.open(input_file).convert("RGB")

    result_image, detections = detect_objects(image)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection Result")
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
                st.write(f"Approaching Speed: {det['Approaching Speed']}")
                st.write(f"TTC: {det['TTC']}")
                st.write(f"Status: {det['Status']}")
                st.markdown("---")
else:
    st.info("Please upload an image or take a picture.")
