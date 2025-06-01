from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import json
import threading

app = Flask(__name__)

model_letters = load_model('models/asl_letters.keras')
model_numbers = load_model('models/asl_numbers.keras')

with open('models/class_indices_letters.json') as f:
    class_indices_letters = json.load(f)
with open('models/class_indices_numbers.json') as f:
    class_indices_numbers = json.load(f)

index_to_class_letters = {int(k): v for k, v in class_indices_letters.items()}
index_to_class_numbers = {int(k): v for k, v in class_indices_numbers.items()}

model_state = {
    "model": model_letters,
    "index_map": index_to_class_letters,
    "type": "letters"
}

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
               
def extract_landmarks(hand_landmarks):
    return np.array([coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])

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
                
                try:
                    landmarks = extract_landmarks(hand_landmarks)
                    input_data = np.expand_dims(landmarks, axis=0)

                    prediction = model_state["model"].predict(input_data, verbose=0)
                    predicted_class = model_state["index_map"][np.argmax(prediction)]
                    
                    print(f"[{model_state['type'].upper()} PREDICTION] {predicted_class}")

                    # Show prediction on black frame
                    cv2.putText(black_frame, f"{predicted_class}", (10, 70),
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
    
@app.route('/switch', methods=['POST'])
def switch():
    global model_state
    if model_state["type"] == "letters":
        model_state["model"] = model_numbers
        model_state["index_map"] = index_to_class_numbers
        model_state["type"] = "numbers"
    else:
        model_state["model"] = model_letters
        model_state["index_map"] = index_to_class_letters
        model_state["type"] = "letters"
    print(f"[MODEL] Switched to {model_state['type'].upper()}")
    return jsonify({"status": "ok", "model": model_state["type"]})

@app.route('/model')
def model_info():
    return jsonify({"model": model_state["type"]})
    
if __name__ == '__main__':
    app.run(debug=True)