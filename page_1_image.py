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
    return 0
def predict_audio(file):
    return 
    
file = st.file_uploader("Upload a video file" )

if file is not None:
    #file_extension = get_file_extension(file)
    
    #if file_extension in ['.jpg', '.png', '.jpeg']:
    st.write("Processing Video")
    
    prediction = predict_image(file)
    
    if prediction == 1:
        st.error("The video is a deepfake!",icon='🚨')
    elif prediction == 0:
        st.success("The video is not a deepfake!",icon='✅')
    else:
        st.warning('Error: Face not detected')
        
 
