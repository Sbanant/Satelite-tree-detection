# Importing necessary libraries
import os
import numpy as np
import shutil
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set paths to the dataset and directories
DATASET = "../input/2750"
TRAIN_DIR = '../working/training'
TEST_DIR = '../working/testing'

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 64

# Automatically get the class labels from dataset folder names
LABELS = os.listdir(DATASET)
NUM_CLASSES = len(LABELS)

# Step 1: Create training and testing directories for organizing data
for path in (TRAIN_DIR, TEST_DIR):
    if not os.path.exists(path):
        os.mkdir(path)

# Step 2: Create class label subdirectories in train and test
for l in LABELS:
    if not os.path.exists(os.path.join(TRAIN_DIR, l)):
        os.mkdir(os.path.join(TRAIN_DIR, l))
    if not os.path.exists(os.path.join(TEST_DIR, l)):
        os.mkdir(os.path.join(TEST_DIR, l))

# Step 3: Split the dataset into training and testing (manually or use StratifiedShuffleSplit for better splitting)
from sklearn.model_selection import train_test_split

# For simplicity, let's split the dataset 80/20 train/test
for label in LABELS:
    img_dir = os.path.join(DATASET, label)
    img_list = os.listdir(img_dir)
    train_imgs, test_imgs = train_test_split(img_list, test_size=0.2, random_state=42)
    
    # Move training images to TRAIN_DIR
    for img in train_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TRAIN_DIR, label, img))
    
    # Move testing images to TEST_DIR
    for img in test_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TEST_DIR, label, img))

# Step 4: Data Augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Step 5: Create data generators for training and testing
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Step 6: Initialize ResNet50 model with ImageNet weights
# Set include_top=False to exclude the fully connected layer at the end
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Step 7: Add custom layers on top of ResNet50
x = Flatten()(base_model.output)  # Flatten the output of ResNet50
x = Dense(512, activation='relu')(x)  # Add a fully connected layer with 512 neurons
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax for multi-class classification

# Step 8: Define the complete model
model = Model(inputs=base_model.input, outputs=output_layer)

# Step 9: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 10: Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator)

# Step 11: Save the trained model
model.save('resnet50_trained_model.h5')

# Step 12: Evaluate the model performance
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Step 13: Plot training & validation accuracy and loss
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
