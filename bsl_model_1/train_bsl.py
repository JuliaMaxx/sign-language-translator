import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def extract_landmarks_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        return np.array([coord for pt in lm.landmark for coord in (pt.x, pt.y, pt.z)])
    return None

def load_data_with_landmarks(data_dir):
    X = []
    y = []
    labels = sorted(os.listdir(data_dir))
    print(f"Loading landmarks from: {data_dir}")
    
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir): continue
        files = os.listdir(label_dir)
        print(f"  Label '{label}' with {len(files)} samples...")
        
        for file in tqdm(files, desc=f"    {label}", leave=False):
            img_path = os.path.join(label_dir, file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            landmarks = extract_landmarks_from_image(image)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
                
    return np.array(X), np.array(y)

print("Extracting landmarks for training and test sets...")
X_train, y_train = load_data_with_landmarks('data/train')
X_test, y_test = load_data_with_landmarks('data/test')

print("Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print("Training SVM on landmarks...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train_enc)
print("Training complete.")

print("Saving model and label encoder...")
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/bsl_landmark_svm.joblib')
joblib.dump(le, 'models/bsl_label_encoder.joblib')
print("Saved to 'models/' directory.")
