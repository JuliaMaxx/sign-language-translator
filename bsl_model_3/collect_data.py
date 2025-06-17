import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# === Settings ===
SAVE_DIR = "data/bsl_data"
NUM_SAMPLES = 2000
DETECTION_CONFIDENCE = 0.8

# === Initialize MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow two hands
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    """
    Extract 2-hand landmarks into a single vector of shape (126,):
    [21 points * 3 (x,y,z) * 2 hands = 126 values].
    If one hand is missing, fill with zeros.
    """
    all_landmarks = []

    if results.multi_hand_landmarks:
        # Sort by handedness to keep consistency: Left hand first
        hand_order = sorted(zip(results.multi_handedness, results.multi_hand_landmarks),
                            key=lambda x: x[0].classification[0].label)

        for _, hand_landmarks in hand_order:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmarks)

    # If only one hand detected, pad with zeros for the second hand
    while len(all_landmarks) < 2:
        all_landmarks.append([0.0] * 63)  # 21 points * 3

    return np.array(all_landmarks).flatten()

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
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
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

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"[INFO] Done capturing '{label.upper()}'")

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()