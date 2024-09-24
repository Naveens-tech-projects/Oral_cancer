from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model
model = load_model('oral_cancer_densenet_model.h5')

# Path to the image to test
image_path = r'E:\akshaya_proj\037.jpeg'
    
# Preprocess the image
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Open the image
image = Image.open(image_path)

# Convert RGBA to RGB if necessary
if image.mode == 'RGBA':
    image = image.convert('RGB')

# Resize the image to match the input size expected by the model
image = image.resize((IMG_HEIGHT, IMG_WIDTH))

# Convert the image to a NumPy array and normalize pixel values to [0, 1]
image = np.array(image) / 255.0

# Add a batch dimension (since the model expects batches of images)
image = np.expand_dims(image, axis=0)

# Predict using the model
prediction = model.predict(image)


prediction = model.predict(image)

# Since prediction is a 2D array with shape (1, num_classes), access the first element
prediction = prediction[0]

# Find the index of the maximum value in the prediction array
max_index = np.argmax(prediction)

# Define class labels
class_labels = ["cancer", "not cancer"]

print(prediction)

# Output the prediction
print(f"Prediction: {class_labels[max_index]}")


