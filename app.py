from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import json

app = Flask(__name__)

model = load_model('models/asl_model.keras')

with open('models/class_indices_asl.json') as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Create a black background frame same size as camera frame
        black_frame = np.zeros_like(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw only landmarks and connections on black background
                mp_drawing.draw_landmarks(
                    black_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3)
                )
                
                # Crop hand region from original frame
                h, w, _ = frame.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x_min = max(min(x_coords) - 20, 0)
                x_max = min(max(x_coords) + 20, w)
                y_min = max(min(y_coords) - 20, 0)
                y_max = min(max(y_coords) + 20, h)

                hand_crop = frame[y_min:y_max, x_min:x_max]

                # Predict letter
                if hand_crop.size > 0:
                    try:
                        hand_resized = cv2.resize(hand_crop, (64, 64))
                        hand_normalized = hand_resized.astype("float32") / 255.0
                        hand_input = np.expand_dims(hand_normalized, axis=0)

                        prediction = model.predict(hand_input, verbose=0)
                        predicted_class = index_to_class[np.argmax(prediction)]

                        print(f"[Predicted] {predicted_class}")

                        # Show prediction on black frame
                        cv2.putText(black_frame, predicted_class, (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

                    except Exception as e:
                        print("Prediction failed:", e)
                        
        ret, buffer = cv2.imencode('.jpg', black_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
    app.run(debug=True)