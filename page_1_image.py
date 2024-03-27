import os
import tempfile
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model
import librosa
import streamlit as st

def predict_video(tfile):
    return 1
def predict_image(file):
    return 1
def predict_audio(file):
    return 1
def get_file_extension(file):
    if source is not None:
    st.write(source.name)
    st.write("File extension:")
    st.write(Path(source.name).suffix)
    
file = st.file_uploader("Upload a video file" )

if file is not None:
    #file_extension = get_file_extension(file)
    
    #if file_extension in ['.jpg', '.png', '.jpeg']:
        st.write("Processing Image")

        prediction = predict_image(file)
    
        if prediction == 1:
            st.error("The image is a deepfake!")
        elif prediction == 0:
            st.success("The image is not a deepfake.")
        else:
            st.warning('Error: Face not detected')
            
 
