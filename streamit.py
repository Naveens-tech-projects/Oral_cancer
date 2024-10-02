import streamlit as st
import sqlite3
import hashlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('best_model_VGG19.h5')
class_labels = ["cancerr", "non cancer"]

def create_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

create_database()
# Database connection and user management
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
user=""
def register_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        st.success("User registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose another.")
    conn.close()

def authenticate_user(username, password):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user

# Image preprocessing function
def preprocess_image(img):
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Cancer detection page

# Main application
def main():
    # st.title("Oral Cancer Detection and User Login")

    # Check if the user is logged in using session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # If the user is logged in, show the cancer detection page
    if st.session_state['logged_in']:
        show_cancer_detection()
    else:
        # Sidebar for login and register
        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Select an option", menu)

        if choice == "Register":
            st.subheader("Register")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Register"):
                register_user(username, password)

        elif choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            user=username
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                user = authenticate_user(username, password)
                if user:
                    st.session_state['logged_in'] = True
                    st.success(f"Welcome, {username}!")
                    st.rerun()  # Rerun the app to reflect the logged-in state
                else:
                    st.error("Invalidd username or password.")

def show_cancer_detection():
    st.subheader("Oral Cancer Detection")
    st.header(user+ "welcome")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        processed_img = preprocess_image(img)
        st.image(img, caption='Uploaded Image', use_column_width=True)


        def predict():
            prediction = model.predict(processed_img)
            predicted_class = class_labels[int(np.round(prediction[0]))]
            st.write(f"Prediction: {predicted_class}")

        if st.button("Predict"):
            predict()

    # Logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()


if __name__ == "__main__":
    main()
