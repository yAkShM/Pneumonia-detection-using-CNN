import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras import layers
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Define paths
zip_file_path = r'C:\Users\HP\OneDrive\AI PROJECT.zip'
extract_to_path = r'C:\Users\HP\Downloads\AI PROJECT'

# Check if the ZIP file exists
if not os.path.exists(zip_file_path):
    raise FileNotFoundError(f"ZIP file not found at {zip_file_path}. Please check the file path.")

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    print("Extracting ZIP file...")
    zip_ref.extractall(extract_to_path)

# Define the expected directory structure
extracted_dir = os.path.join(extract_to_path, 'chest_xray')
train_dir = os.path.join(extracted_dir, 'train')
test_dir = os.path.join(extracted_dir, 'test')
val_dir = os.path.join(extracted_dir, 'val')

# Verify the directory structure
print("Verifying dataset structure...")
if not os.path.exists(extracted_dir):
    raise FileNotFoundError(f"Extracted folder 'chest_xray' not found in {extract_to_path}. Check the ZIP file contents.")

for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        print(f"Warning: Path not found: {directory}")
    else:
        print(f"Path exists: {directory}")

# Verify and list classes in the 'train' directory
if os.path.exists(train_dir):
    classes = os.listdir(train_dir)
    print("Classes in training dataset:", classes)

    # Define paths for NORMAL and PNEUMONIA
    normal_dir = os.path.join(train_dir, 'NORMAL')
    pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

    # Count and display images in each class
    if os.path.exists(normal_dir) and os.path.exists(pneumonia_dir):
        normal_names = os.listdir(normal_dir)
        pneumonia_names = os.listdir(pneumonia_dir)

        print(f'There are {len(normal_names)} normal images in the training dataset.')
        print(f'There are {len(pneumonia_names)} pneumonia images in the training dataset.')

        # Display sample images
        fig = plt.gcf()
        fig.set_size_inches(16, 8)

        pic_index = min(8, len(pneumonia_names))  # Ensure there are enough images
        pneumonia_images = [os.path.join(pneumonia_dir, fname) for fname in pneumonia_names[:pic_index]]

        for i, img_path in enumerate(pneumonia_images):
            sp = plt.subplot(2, 4, i + 1)
            sp.axis('Off')
            img = mpimg.imread(img_path)
            plt.imshow(img)
        plt.show()
    else:
        print("Error: NORMAL or PNEUMONIA folder is missing in the training dataset.")
else:
    print("Error: Training directory not found. Please check the extraction process.")

# Load datasets with error handling
def load_dataset(directory, name):
    if not os.path.exists(directory):
        print(f"Error: {name} directory not found at {directory}.")
        return None
    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="categorical",
        batch_size=32,
        image_size=(256, 256)
    )

Train = load_dataset(train_dir, "Training")
Test = load_dataset(test_dir, "Test")
Validation = load_dataset(val_dir, "Validation")

if Train is None or Test is None or Validation is None:
    raise RuntimeError("Failed to load one or more datasets. Please check the dataset structure.")

# Define the model
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
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
    layers.Dense(2, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(Train, epochs=10, validation_data=Validation)

# Plot training history
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Loss")
history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(Test)
print('Test dataset accuracy:', np.round(accuracy * 100, 2))

# Predict on new images
def predict_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    plt.imshow(img)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_probabilities = prediction[0]
    print("Predicted Class Probabilities:", class_probabilities)
    print("Prediction:", "Normal" if class_probabilities[0] > class_probabilities[1] else "Pneumonia")

# Test predictions
predict_image(r"C:\Users\HP\Downloads\AI PROJECT\chest_xray\test\NORMAL\IM-0010-0001.jpeg", model)
predict_image(r"C:\Users\HP\Downloads\AI PROJECT\chest_xray\test\PNEUMONIA\person100_bacteria_478.jpeg", model)