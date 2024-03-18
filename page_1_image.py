import streamlit as st

def predict_image(image):
    return(1)
      

#with st.sidebar:
#    st.page_link("app.py", label="Home", icon="🏠")
#    st.page_link("pages/page_1_image.py", label="Image Deepfake Detection", icon="📷")
#    st.page_link("pages/page_2_video.py", label="Video Deepfake Detection", icon="🎥")
#    st.page_link("pages/page_3_audio.py", label="Audio Deepfake Detection", icon="🔉")
#    st.page_link("http://www.google.com", label="Report", icon="📄")

#st.title('Image Deepfake Detection')
xy = st.file_uploader("Upload a video file", type=["png", "jpg"])

if xy is not None:
    prediction=predict_image(xy)

    if prediction==1:
        st.info("The image is a deepfake!")
    else:
        st.info("The image is not a deepfake.")
else: 
    print("Error Processing image.")
