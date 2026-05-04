# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:41:40 2026

@author: gersh
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Obstacle Detection UI Prototype", layout="wide")

st.title("🚗 Autonomous Driving Obstacle Detection System")
st.write("UI Prototype for Real-Time Road Condition Detection")

st.markdown("### System Flow")
st.info("Camera Input → YOLO Detection → Distance Estimation → Warning Output")

uploaded_file = st.file_uploader(
    "Upload a road image or video",
    type=["jpg", "jpeg", "png", "mp4"]
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Detection Preview")

    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.video(uploaded_file)
    else:
        st.warning("Please upload an image or video.")

with col2:
    st.subheader("Detection Information")

    st.markdown("**Detected Objects (Mock Data)**")
    st.write("🚘 Car | Confidence: 0.91 | Distance: 12m")
    st.success("Status: Safe")

    st.write("🚶 Pedestrian | Confidence: 0.82 | Distance: 6m")
    st.warning("Status: Warning")

    st.write("🚴 Cyclist | Confidence: 0.76 | Distance: 3m")
    st.error("Status: Danger")

st.markdown("---")
st.subheader("Planned Functions")
st.write("""
- Real-time camera or video input
- YOLO-based object detection
- Distance estimation using LiDAR or simulated distance
- Safety warning system
- Streamlit-based visualization interface
""")