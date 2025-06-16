import cv2
import os
import time
import threading

# Parameters
SAVE_PATH = "bsl_dataset"
NUM_CLASSES = 10
CAPTURE_KEY = ord('c')
EXIT_KEY = ord('q')
IMG_SIZE = (512, 512)
NUM_CAPTURES = 2000
CAPTURE_DELAY = 0.01  # Seconds between images

# Create folders
for i in range(NUM_CLASSES):
    os.makedirs(os.path.join(SAVE_PATH, str(i)), exist_ok=True)

# Choose label
label = input(f"Enter number (0-{NUM_CLASSES - 1}) to capture images for: ").strip()
assert label.isdigit() and 0 <= int(label) < NUM_CLASSES, "Invalid label"
label_path = os.path.join(SAVE_PATH, label)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam not found.")

# Optional: Boost resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Start file numbering from last saved index
count = len(os.listdir(label_path))
capturing = False

def capture_images(start_index):
    global count, capturing
    capturing = True
    print(f"[INFO] Starting background capture of {NUM_CAPTURES} images...")

    for i in range(NUM_CAPTURES):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Frame {i} failed to capture.")
            continue
        frame_resized = cv2.resize(frame, IMG_SIZE)
        filename = os.path.join(label_path, f"{count:04d}.jpg")
        cv2.imwrite(filename, frame_resized)
        print(f"[INFO] Saved {filename}")
        count += 1
        time.sleep(CAPTURE_DELAY)

    print("[INFO] Done capturing images.")
    capturing = False

print(f"[INFO] Press 'c' to start capturing {NUM_CAPTURES} images for label {label}, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    preview_frame = cv2.resize(frame, IMG_SIZE)
    status_text = f"Capturing: {'YES' if capturing else 'NO'}"
    cv2.putText(preview_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if capturing else (0, 0, 255), 2)
    cv2.imshow("BSL Capture", preview_frame)

    key = cv2.waitKey(1)
    if key == CAPTURE_KEY and not capturing:
        threading.Thread(target=capture_images, args=(count,), daemon=True).start()

    elif key == EXIT_KEY:
        print("[INFO] Exiting.")
        break

cap.release()
cv2.destroyAllWindows()
