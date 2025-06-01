# train.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from PIL import Image, UnidentifiedImageError
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")

def validate_images_in_directory(directory):
    print(f"Validating images in: {directory}")
    bad_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            print(file)
            filepath = os.path.join(root, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Just verify image is valid
            except (UnidentifiedImageError, OSError) as e:
                print(f"[BAD IMAGE] {filepath}: {e}")
                bad_files.append(filepath)
    print(f"Validation complete. Found {len(bad_files)} bad file(s).")
    return bad_files

# Constants
IMG_SIZE = 64 # small enough for fast training, big enough to capture detail
BATCH_SIZE = 32 # processes 32 images per step — good balance for RAM and speed
EPOCHS = 30 # one pass over all data = 1 epoch. 10 is enough to get good accuracy with ASL
DATASET_TRAIN = './data/asl_data_split/train'
DATASET_VALIDATE = './data/asl_data_split/val'
MODEL_PATH = './models/asl_model2.keras'

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

# remove corrupted files
# bad_files = validate_images_in_directory(DATASET_TRAIN)
# for file in bad_files:
#     os.remove(file)
#     print(f"Removed corrupt file: {file}")

train_gen = datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical',
    shuffle=True
)

if not os.path.exists('models'):
    os.makedirs("models", exist_ok=True)
with open("./models/class_indices_asl.json", "w") as f:
    json.dump(train_gen.class_indices, f)

val_gen = datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

# Simple CNN model
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # detects edges/shapes, relu adds non-linearity so model can learn complex patterns
    Conv2D(32, (3,3), activation='relu', padding='same'), 
    # Add more convolutional layers
    BatchNormalization(),
    # reduces dimensionality, keeps important features
    MaxPooling2D(2,2), 
    
    # more filters 
    Conv2D(64, (3,3), activation='relu', padding='same'),
    # Add more convolutional layers
    BatchNormalization(),
    # pooling again reduces size
    MaxPooling2D(2,2),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # converts 2D feature maps into a flat 1D array for the fully connected layers
    Flatten(),
    # prevents overfitting by randomly turning off 50% of the neurons during training
    Dropout(0.5),
    # learns high-level patterns
    Dense(256, activation='relu'),
    Dropout(0.3),
    # final layer, outputs probabilities for each class
    Dense(train_gen.num_classes, activation='softmax')
])

# adam is fast and adaptive, categorical_crossentropy used for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
# Train
# learning rate scheduler for refined learning
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
# stops early if overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# trains the model using batches, evaluating on validation set each epoch
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[early_stop, lr_reduce])

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