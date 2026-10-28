import os
import time
import threading
import subprocess
import glob
from collections import deque, Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp

# ---------- Optional: .env + Groq (works without them) ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("Groq not available:", e)

# ---------- Settings ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_model():
    """Use MODEL_PATH from .env if valid, else auto-find a .pkl/.joblib in the project."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = env_path if os.path.isabs(env_path) else os.path.join(PROJECT_ROOT, env_path)
        if os.path.exists(p):
            return p
    patterns = ["models/*.pkl", "models/*.joblib", "*.pkl", "*.joblib",
                "src/*.pkl", "src/*.joblib", "**/*.pkl", "**/*.joblib"]
    for pat in patterns:
        hits = [h for h in glob.glob(os.path.join(PROJECT_ROOT, pat), recursive=True)
                if "venv" not in h.split(os.sep)]
        if hits:
            return hits[0]
    return None


MODEL_PATH = find_model()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
CONF_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", 0.8))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", 10))
SPEAK_COOLDOWN = 2.0  # seconds between spoken words


# ---------- Speech (Windows built-in voice, no extra install) ----------
def _speak(text):
    safe = text.replace("'", "")
    cmd = (
        "Add-Type -AssemblyName System.Speech; "
        f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
    )
    subprocess.run(["powershell", "-Command", cmd], creationflags=0x08000000)


def speak_async(text):
    threading.Thread(target=_speak, args=(text,), daemon=True).start()


# ---------- Sentence builder (Groq, with offline fallback) ----------
def build_sentence(signs):
    fallback = " ".join(signs)
    if groq_client is None or not signs:
        return fallback
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Convert these sign language words into one short, "
                               "natural English sentence. Reply with the sentence only.",
                },
                {"role": "user", "content": " ".join(signs)},
            ],
            max_tokens=40,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Groq error, using fallback:", e)
        return fallback


# ---------- Features ----------
# Matches the normalization used when the model was trained
# (same as normalize() in evaluate_live.py):
# subtract the wrist, then divide by the max absolute value.
def extract_features(hand_landmarks):
    feats = []
    for lm in hand_landmarks.landmark:
        feats.extend([lm.x, lm.y, lm.z])
    pts = np.array(feats, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten().reshape(1, -1)


def main():
    if MODEL_PATH is None:
        print("No model file (.pkl / .joblib) found. Run: python src/train_model.py")
        return
    print("Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("Model loaded. Classes:", list(model.classes_))

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera. Try CAMERA_INDEX=1")
        return

    recent = deque(maxlen=SMOOTHING_WINDOW)
    collected = []          # confirmed signs for the sentence
    last_spoken = ""
    last_time = 0.0
    sentence = ""
    current_label = ""

    print("Keys: s = build + speak sentence | c = clear | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_label = ""
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            X = extract_features(hand)
            probs = model.predict_proba(X)[0]
            idx = int(np.argmax(probs))
            label = str(model.classes_[idx])
            conf = float(probs[idx])

            if conf >= CONF_THRESHOLD:
                recent.append(label)
                current_label = f"{label} ({conf*100:.0f}%)"
            else:
                recent.append(None)

            # stable = same label for most of the window
            if len(recent) == SMOOTHING_WINDOW:
                common, count = Counter(recent).most_common(1)[0]
                if common is not None and count >= int(SMOOTHING_WINDOW * 0.8):
                    now = time.time()
                    if common != last_spoken and now - last_time > SPEAK_COOLDOWN:
                        print("Speaking:", common)
                        speak_async(common)
                        collected.append(common)
                        last_spoken = common
                        last_time = now
        else:
            recent.clear()
            last_spoken = ""  # allow the same sign again after hand leaves

        # ---------- On-screen text ----------
        cv2.putText(frame, f"Sign: {current_label}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Words: " + " ".join(collected[-6:]), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if sentence:
            cv2.putText(frame, sentence[:70], (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            collected.clear()
            sentence = ""
        elif key == ord("s") and collected:
            sentence = build_sentence(collected)
            print("Sentence:", sentence)
            speak_async(sentence)
            collected.clear()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()