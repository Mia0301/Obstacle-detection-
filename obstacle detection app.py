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
import tempfile

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

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"{label} {conf}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    area = (x2-x1)*(y2-y1)

    if area > 80000:
        status = "Danger 🔴"
    elif area > 30000:
        status = "Warning 🟡"
    else:
        status = "Safe 🟢"

    return img, label, conf, status

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img, label, conf, status = fake_detection(img)

    col1, col2 = st.columns([2,1])

    col1.image(img, channels="BGR", use_container_width=True)

    col2.subheader("Detection Info")
    col2.write(f"Object: {label}")
    col2.write(f"Confidence: {conf}")
    col2.write(f"Status: {status}")
