import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import json

DATASET_DIR = 'data/asl_data'

X = []
y = []

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue
    for filename in os.listdir(label_path):
        if filename.endswith(".npy"):
            filepath = os.path.join(label_path, filename)
            sample = np.load(filepath)
            X.append(sample)
            y.append(label)
            
X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples")

letters_mask = np.char.isalpha(y)
numbers_mask = np.char.isnumeric(y)

X_letters = X[letters_mask]
y_letters = y[letters_mask]

X_numbers = X[numbers_mask]
y_numbers = y[numbers_mask]

# letters model
le_letters = LabelEncoder()
y_letters_encoded = to_categorical(le_letters.fit_transform(y_letters))
X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(
    X_letters, y_letters_encoded, test_size=0.2, random_state=42
)

# Numbers model
le_numbers = LabelEncoder()
y_numbers_encoded = to_categorical(le_numbers.fit_transform(y_numbers))

X_train_n, X_val_n, y_train_n, y_val_n = train_test_split(
    X_numbers, y_numbers_encoded, test_size=0.2, random_state=42
)

def build_model(output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Letters model
model_letters = build_model(y_train_l.shape[1])
model_letters.fit(X_train_l, y_train_l, epochs=20, validation_data=(X_val_l, y_val_l))

# Save it
model_letters.save("models/asl_letters.keras")

# Numbers model
model_numbers = build_model(y_train_n.shape[1])
model_numbers.fit(X_train_n, y_train_n, epochs=20, validation_data=(X_val_n, y_val_n))

# Save it
model_numbers.save("models/asl_numbers.keras")

# Letters
with open("models/class_indices_letters.json", "w") as f:
    json.dump({i: c for i, c in enumerate(le_letters.classes_)}, f)

# Numbers
with open("models/class_indices_numbers.json", "w") as f:
    json.dump({i: c for i, c in enumerate(le_numbers.classes_)}, f)