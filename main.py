import os
from PIL import Image

# Remove Corrupted Images
folders = ['dataset/train/cats', 'dataset/train/dogs']

for folder in folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            img = Image.open(path)
            img.verify()

        except:
            print("Deleting corrupted file:", path)
            os.remove(path)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataset Path
train_dir = 'dataset/train'

# Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data
val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# CNN Model
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
# model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=5
# )

# # Save Model
# model.save('models/cat_dog_model.h5')

# print("Model Trained Successfully")


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load Trained Model
model = load_model('models/cat_dog_model.h5')

# Load Test Image
img = image.load_img('test_images/test.jpg', target_size=(150,150))

# Convert Image to Array
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Prediction
prediction = model.predict(img_array)

# Output
if prediction[0][0] > 0.5:
    print("Dog Detected 🐶")
else:
    print("Cat Detected 🐱")