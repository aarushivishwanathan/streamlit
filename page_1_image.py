import os
import tempfile
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from keras.models import load_model
import librosa
import streamlit as st

def predict_video(tfile):
    # Initialize prediction result as None
    #tfile = tempfile.NamedTemporaryFile(delete=False)
    #tfile.write(file.read())

    y_predicted = None
    
    frames_pred = []
    detector = MTCNN()
    cap=cv2.VideoCapture(tfile)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        faces = detector.detect_faces(frame)

        if faces:
            faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
            selected_face = faces[0]
            x, y, w, h = selected_face['box']
            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (35, 35))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            _, face = cv2.threshold(face, 127, 255, cv2.THRESH_BINARY) 
            face = face / 255.0 

            frames_pred.append(face)
    cap.release()

    X = np.array(frames_pred)

    model = load_model(r'C:\Sem 6 project\Models\2DCNN_model.h5')
    pred = model.predict(X)
    mean_prediction = np.mean(pred)
    y_predicted = np.where(mean_prediction > 0.5, 1, 0)
    
    return y_predicted  

def predict_image(file):
    tfile=cv2.imread(file)
    detector = MTCNN()
    faces = detector.detect_faces(tfile)

    if faces:
    # Sort faces based on confidence in descending order
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)

        # Select the face with the highest confidence
        selected_face = faces[0]

        x, y, w, h = selected_face['box']
        face = tfile[y:y+h, x:x+w]

        face = cv2.resize(face, (35, 35))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        ret, face = cv2.threshold(face, 127, 255, cv2.THRESH_BINARY) 
        face=face/225 
        face=face.reshape(1,35,35,1)

    model = load_model(r'C:\Sem 6 project\Models\2DCNN_model.h5')
    pred=model.predict(face)

    y_predicted=np.where(pred>0.5, 1,0)

    print(y_predicted)

    return y_predicted

def predict_audio(file):
    #tfile = tempfile.NamedTemporaryFile(delete=False)
    #tfile.write(file.read())

    y_predicted = None
    
    audio, sr = librosa.load(file, sr=16000, duration=6)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

    X = np.concatenate((mfccs, spectral_contrast, zcr, bandwidth), axis=0)
    X = X.flatten()
    X_flat = X.reshape(-1, X.shape[0])

    model = load_model(r'C:\Sem 6 project\Models\Audi_MLP.h5')
    pred = model.predict(X_flat)
    y_predicted = np.where(pred > 0.5, 1, 0)

    return y_predicted

def get_file_extension(file):
    # Extract file extension from the uploaded file name
    _, extension = os.path.splitext(file)
    return extension.lower()

file = st.file_uploader("Upload a video file" )

if file is not None:
    file_extension = get_file_extension(file)
    
    if file_extension in ['.jpg', '.png', '.jpeg']:
        st.write("Processing Image")

        prediction = predict_image(file)
        
        
        if prediction == 1:
            st.error("The image is a deepfake!")
        elif prediction == 0:
            st.success("The image is not a deepfake.")
        else:
            st.warning('Error: Face not detected')
            
    elif file_extension in ['.mp4', '.avi', '.mov']:
        st.write("Processing Video")

        prediction = predict_video(file)

        if prediction == 1:
            st.error("The video is a deepfake!")
        elif prediction == 0:
            st.success("The video is not a deepfake.")
        else:
            st.warning('Error: Face not detected')
    
    elif file_extension in ['.mp3', '.wav', '.flac']:
        st.write("Processing Audio")

        prediction = predict_audio(file)

        if prediction == 1:
            st.error("The audio is a deepfake!")
        else:
            st.success("The audio is not a deepfake.")
    else:
        st.warning("Unsupported file format.")
