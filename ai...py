import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras import layers
import pandas as pd
import numpy as np
import gc

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Memory management
tf.keras.backend.clear_session()
gc.collect()

# Define paths
zip_file_path = r'C:\Users\HP\OneDrive\AI PROJECT.zip'
extract_to_path = r'C:\Users\HP\Downloads\AI PROJECT'

# Check and extract ZIP file if needed
if not os.path.exists(os.path.join(extract_to_path, 'chest_xray')):
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

# Define directories
extracted_dir = os.path.join(extract_to_path, 'chest_xray')
train_dir = os.path.join(extracted_dir, 'train')
test_dir = os.path.join(extracted_dir, 'test')
val_dir = os.path.join(extracted_dir, 'val')

# Verify directories
for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Path not found: {directory}")
    print(f"Path exists: {directory}")

# Count images and calculate class weights
normal_train = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
pneumonia_train = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
print(f'Training images - Normal: {normal_train}, Pneumonia: {pneumonia_train}')

total = normal_train + pneumonia_train
class_weights = {
    0: pneumonia_train/total,  # For normal class
    1: normal_train/total      # For pneumonia class
}

# Data loading function with memory optimization
def load_dataset(directory, name):
    return tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="categorical",
        batch_size=8,  # Reduced batch size
        image_size=(160, 160),  # Reduced image size
        shuffle=True
    ).prefetch(tf.data.AUTOTUNE)

# Load datasets
print("Loading datasets...")
Train = load_dataset(train_dir, "Training")
Test = load_dataset(test_dir, "Test")
Validation = load_dataset(val_dir, "Validation")

# Clear memory
gc.collect()

# Define a smaller model
model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='sigmoid')
])

# Compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Training with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train in smaller chunks
print("\nStarting training...")
history = model.fit(
    Train,
    epochs=5,  # Reduced epochs
    validation_data=Validation,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Clear memory again
gc.collect()

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Evaluate on test set
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(Test)
print(f'Test accuracy: {test_accuracy:.2%}')

# Function for single image prediction
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    result = "Normal" if prediction[0][0] > prediction[0][1] else "Pneumonia"
    confidence = max(prediction[0]) * 100
    
    plt.imshow(img)
    plt.title(f'Prediction: {result} ({confidence:.1f}% confident)')
    plt.axis('off')
    plt.show()

# Test predictions on sample images
print("\nTesting predictions on sample images...")
normal_test = os.path.join(test_dir, 'NORMAL', os.listdir(os.path.join(test_dir, 'NORMAL'))[0])
pneumonia_test = os.path.join(test_dir, 'PNEUMONIA', os.listdir(os.path.join(test_dir, 'PNEUMONIA'))[0])

predict_image(normal_test)
predict_image(pneumonia_test)

# Save the model
model.save('chest_xray_model.h5')
print("\nModel saved as 'chest_xray_model.h5'")