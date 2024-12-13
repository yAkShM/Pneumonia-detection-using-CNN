import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to extract dataset
def extract_dataset(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("Dataset extracted successfully!")

# Function to verify dataset structure
def verify_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            print(f"Path not found: {directory}")
            return False
        print(f"Path exists: {directory}")
    return True

# Function to display sample images
def display_sample_images(directory, class_name, num_images=8):
    class_dir = os.path.join(directory, class_name)
    image_files = os.listdir(class_dir)[:num_images]
    fig = plt.figure(figsize=(16, 8))
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(class_dir, image_file)
        ax = fig.add_subplot(2, 4, i + 1)
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('Off')
    plt.suptitle(f"Sample Images from Class: {class_name}", fontsize=16)
    plt.show()

# Function to define the model
def build_model(input_shape=(256, 256, 3), num_classes=2):
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to make predictions
def predict_image(model, image_path, class_labels, image_size=(256, 256)):
    img = keras.utils.load_img(image_path, target_size=image_size)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('Off')
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_label = class_labels[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_label} (Confidence: {np.max(prediction):.2f})")
    return predicted_label

# Define paths
zip_file_path = r'C:\Users\HP\Downloads\AI PROJECT\chest-xray-pneumonia.zip'
extract_to_path = r'C:\Users\HP\Downloads\AI PROJECT'
train_dir = os.path.join(extract_to_path, 'chest_xray', 'train')
test_dir = os.path.join(extract_to_path, 'chest_xray', 'test')
val_dir = os.path.join(extract_to_path, 'chest_xray', 'val')

# Extract and verify dataset
extract_dataset(zip_file_path, extract_to_path)
if not verify_directories(train_dir, test_dir, val_dir):
    exit()

# Display dataset statistics
classes = os.listdir(train_dir)
print("Classes in dataset:", classes)
for class_name in classes:
    class_dir = os.path.join(train_dir, class_name)
    print(f"{class_name}: {len(os.listdir(class_dir))} images in training set.")
    display_sample_images(train_dir, class_name)

# Load datasets
image_size = (256, 256)
Train = keras.utils.image_dataset_from_directory(train_dir, image_size=image_size, batch_size=32, label_mode="categorical")
Validation = keras.utils.image_dataset_from_directory(val_dir, image_size=image_size, batch_size=32, label_mode="categorical")
Test = keras.utils.image_dataset_from_directory(test_dir, image_size=image_size, batch_size=32, label_mode="categorical")

# Build, train, and evaluate the model
model = build_model(input_shape=(256, 256, 3), num_classes=len(classes))
model.summary()

history = model.fit(Train, validation_data=Validation, epochs=10)

# Plot training history
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(title="Loss over Epochs")
history_df[['accuracy', 'val_accuracy']].plot(title="Accuracy over Epochs")
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(Test)
print(f"Test Accuracy: {accuracy:.2%}")

# Save the trained model
model_path = os.path.join(extract_to_path, 'chest_xray_model.h5')
model.save(model_path)
print(f"Model saved at {model_path}")

# Predict on new images
test_images = [
    r"C:\Users\HP\Downloads\AI PROJECT\chest_xray\test\NORMAL\IM-0010-0001.jpeg",
    r"C:\Users\HP\Downloads\AI PROJECT\chest_xray\test\PNEUMONIA\person100_bacteria_478.jpeg"
]
for image_path in test_images:
    predict_image(model, image_path, classes)
