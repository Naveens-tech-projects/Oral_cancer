import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('best_model_VGG19.h5')
class_labels = ["cancer", "non cancer"]

st.title("Oral Cancer Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def preprocess_image(img):
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    processed_img = preprocess_image(img)
    st.image(img, caption='Uploaded Image', use_column_width=True)


    def predict():
        prediction = model.predict(processed_img)
        predicted_class = class_labels[int(np.round(prediction[0]))]
        confidence = prediction[0] * 100
        st.write(f"Prediction: {predicted_class}")


    if st.button("Predict"):
        predict()