# train.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from keras.

# Constants
IMG_SIZE = 64 # small enough for fast training, big enough to capture detail
BATCH_SIZE = 32 # processes 32 images per step — good balance for RAM and speed
EPOCHS = 10 # one pass over all data = 1 epoch. 10 is enough to get good accuracy with ASL
DATASET_TRAIN = 'data/asl_alphabet_train'
MODEL_PATH = 'models/asl_model.h5'

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=False
)

train_gen = datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

# Simple CNN model
model = Sequential([
    # detects edges/shapes, relu adds non-linearity so model can learn complex patterns
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)), 
    # reduces dimensionality, keeps important features
    MaxPooling2D(2,2), 
    # more filters 
    Conv2D(64, (3,3), activation='relu'),
    # pooling again reduces size
    MaxPooling2D(2,2),
    # converts 2D feature maps into a flat 1D array for the fully connected layers
    Flatten(),
    # prevents overfitting by randomly turning off 50% of the neurons during training
    Dropout(0.5),
    # learns high-level patterns
    Dense(256, activation='relu'),
    # final layer, outputs probabilities for each class
    Dense(train_gen.num_classes, activation='softmax')
])

# adam is fast and adaptive, categorical_crossentropy used for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
# trains the model using batches, evaluating on validation set each epoch
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")


# Why Convolutional Neural Network?
# it is specifically designed for image data
# efficient - less computation, designed for large inputs like images
# proven architecture - CNNs are backbone of almost all successful image classifiers

# Why ImageGenerator?
# Loads images efficiently in batches
# Applies data augmentation
# Automatically map classes 