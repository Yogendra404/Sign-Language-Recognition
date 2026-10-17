import time
import threading
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

MODEL_PATH = "models/sign_model.pkl"
CONF_THRESHOLD = 0.70   # minimum model confidence to accept a prediction
WINDOW = 10             # frames used for smoothing
STABLE_COUNT = 8        # how many of the last WINDOW frames must agree
COOLDOWN = 2.0          # seconds before the same word can be spoken again

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)


def normalize(row):
    """Must match the normalize() used in train_model.py."""
    pts = np.array(row, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


speaking = False


def speak(text):
    """Speak in a background thread so the video doesn't freeze."""
    global speaking
    if speaking:
        return

    def run():
        global speaking
        speaking = True
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        finally:
            speaking = False

    threading.Thread(target=run, daemon=True).start()


history = deque(maxlen=WINDOW)
current_word = ""
last_spoken = ""
last_spoken_time = 0.0
sentence = []

cap = cv2.VideoCapture(0)
print("Q = quit | C = clear sentence")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    label, conf = "", 0.0
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        raw = []
        for p in lm.landmark:
            raw += [p.x, p.y, p.z]
        x = normalize(raw).reshape(1, -1)

        probs = model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf >= CONF_THRESHOLD:
            label = model.classes_[idx]
        history.append(label)
    else:
        history.append("")

    # Smoothing: accept a word only if it dominates the recent frames
    word, votes = Counter(history).most_common(1)[0] if history else ("", 0)
    if word and votes >= STABLE_COUNT:
        current_word = word
        now = time.time()
        if word != last_spoken or (now - last_spoken_time) > COOLDOWN:
            speak(word)
            last_spoken, last_spoken_time = word, now
            if not sentence or sentence[-1] != word:
                sentence.append(word)
    elif not result.multi_hand_landmarks:
        current_word = ""

    # On-screen display
    cv2.putText(frame, f"Sign: {current_word or '...'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, " ".join(sentence[-6:]), (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence.clear()

cap.release()
cv2.destroyAllWindows()