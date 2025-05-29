import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import datetime

# === Constants ===
DATASET_TEST = 'asl_alphabet_test'
IMG_SIZE = 64
BATCH_SIZE = 32
MODEL_PATH = '../models/asl_model.keras'
CLASS_INDEX_PATH = '../models/class_indices_asl.json'

# === Load class indices ===
with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

# === Load model ===
model = load_model(MODEL_PATH)

# === Load test data ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    DATASET_TEST,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === Predictions ===
pred_probs = model.predict(test_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes

# === Create output directory ===
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"classification_report_asl_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# === Classification Report ===
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_md = report_df.to_markdown()

# === Confusion Matrix ===
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
conf_img_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_img_path)
plt.close()

# === Write Markdown Report ===
md_path = os.path.join(output_dir, "report.md")
with open(md_path, "w") as f:
    f.write("# ASL Model Evaluation Report\n\n")
    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Confusion Matrix\n\n")
    f.write(f"![Confusion Matrix](confusion_matrix.png)\n\n")
    f.write("## Classification Report\n\n")
    f.write(report_md)

print(f"[✅] Markdown report saved to: {md_path}")
