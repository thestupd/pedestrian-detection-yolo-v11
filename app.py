import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('model.pt')

st.title("YOLO Object Detection")

# File uploader for images and videos
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.type in ['video/mp4', 'video/x-msvideo']:
        # Process video
        st.video(uploaded_file)
        # Save the uploaded video temporarily
        video_path = f"temp_video.{uploaded_file.type.split('/')[1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run inference on the video
        results = model.predict(source=video_path, iou=0.5, conf=0.7, save=True)
        st.success("Video processed successfully!")
        
        # Display the output video
        output_video_path = results[0].save_path  # Get the path of the saved video
        st.video(output_video_path)

    else:
        # Process image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Run inference on the image
        results = model.predict(source=image, iou=0.5, conf=0.7, save=True)
        st.success("Image processed successfully!")

        # Display the output image
        output_image_path = results[0].save_path  # Get the path of the saved image
        st.image(output_image_path, caption='Processed Image', use_column_width=True)