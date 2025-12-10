import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np

# Load model
model = tf.keras.models.load_model(r"C:\ARsl_\arsl_efficientnetb0_model.keras")
print("Model loaded successfully!")
# Image size and classes
IMG_SIZE = (224, 224)
class_names = ['ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي']

# Streamlit UI
st.title("Arabic Sign Language Letter Recognition")
st.write("Upload an image of a hand sign and the model will predict the Arabic letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    image = ImageOps.fit(image, IMG_SIZE, Image.ANTIALIAS)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Predicted Letter: **{predicted_class}**")
    st.write(f"Confidence: {confidence*100:.2f}%")
