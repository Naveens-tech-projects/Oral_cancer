import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_dir_cancer = r'E:\Oral Cancer\Oral Cancer Dataset\CANCER'
dataset_dir_non_cancer = r'E:\Oral Cancer\Oral Cancer Dataset\NON CANCER'
train_dir = r'E:\Oral Cancer\Oral Cancer Dataset\train'
test_dir = r'E:\Oral Cancer\Oral Cancer Dataset\test'

# Create train and test directories for each class
os.makedirs(os.path.join(train_dir, 'CANCER'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'CANCER'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'NON CANCER'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'NON CANCER'), exist_ok=True)


# Function to split and copy files to train and test directories
def split_data(class_dir, class_name):
    images = os.listdir(class_dir)
    train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)

    # Copy train images
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))

    # Copy test images
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))


# Split the CANCER and non cancer datasets
split_data(dataset_dir_cancer, 'CANCER')
split_data(dataset_dir_non_cancer, 'NON CANCER')

print("Dataset split into train and test directories.")
