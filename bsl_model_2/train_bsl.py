import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
import joblib

# Load your extracted dataset
df = pd.read_csv('data/bsl_hand_landmarks.csv')

# Drop filename column — we don't need it for training
df.drop(columns=['filename'], inplace=True)

# Drop rows where any landmarks are missing
df.dropna(inplace=True)

counts = df['label'].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df['label'].isin(valid_classes)]

# Split into features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Save label encoder for later use in Flask app
joblib.dump(label_encoder, 'models/bsl_label_encoder_2.joblib')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Build the Keras model (simple MLP)
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Save the trained model
model.save('models/bsl_landmarks.keras')
print("Model saved successfully.")
