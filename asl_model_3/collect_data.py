import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# === Settings ===
SAVE_DIR = "data/asl_data"
NUM_SAMPLES = 2000
DETECTION_CONFIDENCE = 0.8

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

# === Webcam ===
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]  # Just first hand
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

print("[INFO] Press a key (like 'a') to start recording landmarks for that label.")
print("       Press ESC to quit at any time.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Show landmarks on screen
    if results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Hand Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    # ESC to quit
    if key == 27:
        break

    # A-Z or 0-9 key to label
    if 97 <= key <= 122 or 48 <= key <= 57:  # a-z or 0-9
        label = chr(key)
        print(f"[INFO] Starting capture for '{label.upper()}'")
        label_dir = os.path.join(SAVE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        count = 0
        while count < NUM_SAMPLES:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            landmarks = extract_landmarks(results)

            if landmarks is not None:
                filename = os.path.join(label_dir, f"{label}_{count}.npy")
                np.save(filename, landmarks)
                count += 1

                cv2.putText(frame, f"Recording '{label.upper()}' - {count}/{NUM_SAMPLES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"[INFO] Done capturing '{label.upper()}'")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
