import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands

# Dataset path
dataset_path = './data/train'  # adjust this if needed

# Output storage
data = []

# Mediapipe Hands model
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:

    # Walk through each subfolder (label)
    for label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label)

        if not os.path.isdir(label_folder):
            continue

        for filename in os.listdir(label_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(label_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                # Pad second hand if only one detected
                if len(results.multi_hand_landmarks) == 1:
                    landmarks.extend([float('nan')] * 63)
            else:
                landmarks = [float('nan')] * 126

            # Save: label + landmarks
            row = [label, filename] + landmarks
            data.append(row)

# Prepare column names
columns = ['label', 'filename'] + [f'hand{i}_{axis}{j}' for i in range(1, 3) for j in range(21) for axis in ['x', 'y', 'z']]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv('bsl_hand_landmarks.csv', index=False)
