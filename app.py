from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import json
from collections import deque, Counter
from symspellpy import SymSpell
from starlette.templating import Jinja2Templates
import socketio
import threading
import asyncio
import uvicorn

# Spelling
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# App and socket.io server
sio = socketio.AsyncServer(async_mode='asgi')

fastapi_app = FastAPI()
fastapi_app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app = socketio.ASGIApp(sio, fastapi_app)

main_loop = asyncio.new_event_loop()
asyncio.set_event_loop(main_loop)

# Globals
camera = cv2.VideoCapture(0)
frame_lock = threading.Lock()
prediction_buffer = deque(maxlen=15)

# Models
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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
               
def extract_landmarks(hand_landmarks):
    return np.array([coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)])

def correct_and_segment(text):
    suggestions = sym_spell.word_segmentation(text)
    return suggestions.corrected_string

# Get main loop
main_loop = asyncio.get_event_loop()

def generate_frames():
    global sentence
    prediction_buffer = deque(maxlen=15)
    last_prediction = None
    sentence = ""
    
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        # Create a black background frame same size as camera frame
        black_frame = np.full_like(frame, fill_value=(86, 86, 91))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw only landmarks and connections on black background
                mp_drawing.draw_landmarks(
                    black_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(237, 228, 228), thickness=5, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(188, 188, 200), thickness=3)
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

                    predicted_index = np.argmax(prediction)
                    predicted_char = model_state["index_map"][predicted_index]
                    
                    # wait for model to be certain withthe letter
                    prediction_buffer.append(predicted_char)
                    buffer_counts = Counter(prediction_buffer)
                    most_common, count = buffer_counts.most_common(1)[0]
                    if count >= 6 and most_common != last_prediction:
                        sentence += most_common
                        last_prediction = most_common
                        prediction_buffer.clear()
                        corrected = correct_and_segment(sentence)
                        print("Refined:", corrected)
                        
                        # Run emit on the main loop
                        asyncio.run_coroutine_threadsafe(
                            sio.emit('update_text', {'text': corrected}),
                            main_loop
                        )
                            
                except Exception as e:
                    print("Prediction failed:", e)
                        
        ret, buffer = cv2.imencode('.jpg', black_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@fastapi_app.get('/')
async def get_index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@fastapi_app.get('/video')
def video():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
    
@fastapi_app.post('/switch')
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
    return JSONResponse({"status": "ok", "model": model_state["type"]})

@fastapi_app.post('/clear')
def clear():
    global sentence
    sentence = ""
    return JSONResponse({"status": "ok"})

@fastapi_app.get('/model')
def model_info():
    return JSONResponse({"model": model_state["type"]})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)