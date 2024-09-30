import streamlit as st
import tensorflow as tf
from PIL import Image

import numpy as np

model = tf.keras.models.load_model('oral_cancer_densenet_model.h5')
class_labels = ["cancer", "non cancer"]


def preprocess_image(image):
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


st.title('Oral Cancer Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]

    max_index = np.argmax(prediction)
    confidence = prediction[max_index] * 100  # No typecasting needed here

    st.write(f"Confidence: {confidence:.2f}%")

    if confidence < 75 and class_labels[max_index] == "cancer":
        predicted = "non cancer"
    else:
        predicted = class_labels[max_index]

    st.write(f"Prediction: {predicted}")
