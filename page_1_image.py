import streamlit as st
import os
import tempfile
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model

def predict_image(file):
    return(1)

def predict_video(file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    cap = cv2.VideoCapture(tfile.name)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with st.spinner(f'Processing {num_frames} frames'):
        detector = MTCNN()
        
        frames_pred=[]
        frame_num=0
        
        while frame_num<10:
            ret, frame = cap.read()

            if not ret:
                break  # Break if no more frames to read

            faces = detector.detect_faces(frame)

            if faces:
                # Sort faces based on confidence in descending order
                faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)

                # Select the face with the highest confidence
                selected_face = faces[0]
                
                x, y, w, h = selected_face['box']
                face = frame[y:y+h, x:x+w]

                # Save face image
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                face = cv2.resize(face, (35, 35))
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                ret, face = cv2.threshold(face, 127, 255, cv2.THRESH_BINARY) 
                face=face/225 

                frames_pred.append(face)

        cap.release()

        X=np.array(frames_pred)

        model = load_model('./2DCNN_model.h5')
        pred=model.predict(X)
        print(pred)

        mean_prediction = np.mean(pred)

        y_predicted=np.where(mean_prediction>0.5, 1,0)
    
    return y_predicted
    

def predict_audio(file):
    return(1)

def get_file_extension(file):
    # Extract file extension from the uploaded file name
    file_name = file.name
    _, extension = file_name.split('.')
    return extension

file = st.file_uploader("Upload a video file" )

if file is not None:
    file_extension = get_file_extension(file)
    if file_extension in ['jpg','png']:
        st.write("Processing Image")

        prediction=predict_image(file)

        if prediction==1:
            st.error("The image is a deepfake!", icon="🚨")
        else:
            st.success("The image is not a deepfake.", icon="✅")
    if file_extension in ['mp4','avi','mov']:
        st.write("Processing Video")

        prediction=predict_video(file)

        if prediction==1:
            st.error("The video is a deepfake!", icon="🚨")
        else:
            st.success("The video is not a deepfake.", icon="✅")
    
    if file_extension in ['mp3','wav','flac']:
        st.write("Processing Audio")

        prediction=predict_audio(file)

        if prediction==1:
            st.error("The audio is a deepfake!", icon="🚨")
        else:
            st.success("The audio is not a deepfake.", icon="✅")
