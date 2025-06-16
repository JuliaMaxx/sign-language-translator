import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Dataset path ===
DATASET_DIR = 'data/bsl_data'  # updated path

# === Load and prepare data ===
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
            if sample.shape[0] == 126:  # make sure it's 2-hand data
                X.append(sample)
                y.append(label)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Loaded {len(X)} BSL samples (2-hand landmarks)")

# === Separate letters and numbers ===
letters_mask = np.char.isalpha(y)
numbers_mask = np.char.isnumeric(y)

X_letters = X[letters_mask]
y_letters = y[letters_mask]

X_numbers = X[numbers_mask]
y_numbers = y[numbers_mask]

# === Encode & split ===
def prepare_data(X_data, y_data):
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y_data))
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_encoded, test_size=0.28, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3571, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, le

X_train_l, X_val_l, X_test_l, y_train_l, y_val_l, y_test_l, le_letters = prepare_data(X_letters, y_letters)
X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n, le_numbers = prepare_data(X_numbers, y_numbers)

# === Build model for 126-dim input ===
def build_model(output_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(126,)),  # Updated input size
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === Train & Save ===
os.makedirs("models", exist_ok=True)

# Letters
model_letters = build_model(y_train_l.shape[1])
model_letters.fit(X_train_l, y_train_l, epochs=20, validation_data=(X_val_l, y_val_l))
model_letters.save("models/bsl_letters.keras")
with open("models/class_indices_bsl_letters.json", "w") as f:
    json.dump({i: c for i, c in enumerate(le_letters.classes_)}, f)

# Numbers
model_numbers = build_model(y_train_n.shape[1])
model_numbers.fit(X_train_n, y_train_n, epochs=20, validation_data=(X_val_n, y_val_n))
model_numbers.save("models/bsl_numbers.keras")
with open("models/class_indices_bsl_numbers.json", "w") as f:
    json.dump({i: c for i, c in enumerate(le_numbers.classes_)}, f)

# === Evaluation helper ===
def evaluate(model, X_test, y_test, label_encoder, title, filename):
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(f"=== {title.upper()} CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{filename}.png")
    plt.close()

# === Evaluate both ===
evaluate(model_letters, X_test_l, y_test_l, le_letters, "Letters", "letters")
evaluate(model_numbers, X_test_n, y_test_n, le_numbers, "Numbers", "numbers")
