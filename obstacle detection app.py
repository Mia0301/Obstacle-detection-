# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:41:40 2026

@author: gersh
"""

import streamlit as st
from PIL import Image
import random
import cv2
import numpy as np

st.set_page_config(layout="wide")
st.title("🚗 Obstacle Detection System (Prototype)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

def fake_detection(img):
    h, w, _ = img.shape

    # 隨機框
    x1 = random.randint(0, w//2)
    y1 = random.randint(0, h//2)
    x2 = x1 + random.randint(100, 300)
    y2 = y1 + random.randint(100, 300)

    label = random.choice(["Car", "Pedestrian", "Cyclist"])
    conf = round(random.uniform(0.7, 0.95), 2)

    # 畫框
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    # 面積模擬距離（越大越近）
    area = (x2 - x1) * (y2 - y1)

    if area > 80000:
        distance = random.uniform(2, 5)
        status = "Danger 🔴"
        color = (0,0,255)
    elif area > 30000:
        distance = random.uniform(5, 10)
        status = "Warning 🟡"
        color = (0,255,255)
    else:
        distance = random.uniform(10, 20)
        status = "Safe 🟢"
        color = (0,255,0)

    distance = round(distance, 1)

    # 在畫面上寫資訊
    cv2.putText(img, f"{label} {conf}", (x1, y1-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(img, f"{distance}m", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, label, conf, distance, status

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img, label, conf, distance, status = fake_detection(img)

    col1, col2 = st.columns([2,1])

    # 左邊畫面
    col1.image(img, channels="BGR", use_container_width=True)

    # 右邊資訊
    col2.subheader("Detection Info")
    col2.write(f"Object: {label}")
    col2.write(f"Confidence: {conf}")
    col2.write(f"Distance: {distance} m")
    col2.write(f"Status: {status}")
