import csv
import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/sign_model.pkl"
SIGNS = ["Hello", "Yes", "No", "ThankYou", "Please", "Sorry", "Drink"]
TRIALS = 10

model = joblib.load(MODEL_PATH)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)


def normalize(row):
    pts = np.array(row, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


results = {s: 0 for s in SIGNS}
log = []
cap = cv2.VideoCapture(0)
print("Make the sign shown on screen, then press SPACE to record. Q to quit.")

stop = False
for sign in SIGNS:
    trial = 0
    while trial < TRIALS and not stop:
        ok, frame = cap.read()
        if not ok:
            stop = True
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pred = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            raw = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
            pred = model.predict(normalize(raw).reshape(1, -1))[0]

        cv2.putText(frame, f"Show: {sign}  ({trial + 1}/{TRIALS})", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Model sees: {pred or '-'}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live Accuracy Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop = True
        elif key == 32 and pred is not None:   # SPACE
            correct = pred == sign
            results[sign] += int(correct)
            log.append([sign, pred, int(correct)])
            trial += 1
            print(f"{sign} #{trial}: predicted {pred} -> "
                  f"{'correct' if correct else 'WRONG'}")
    if stop:
        break

cap.release()
cv2.destroyAllWindows()

print("\n| Sign | Trials | Correct | Accuracy |")
print("|---|---|---|---|")
total_c = total_t = 0
for s in SIGNS:
    t = sum(1 for r in log if r[0] == s)
    if t == 0:
        continue
    c = results[s]
    total_c += c
    total_t += t
    print(f"| {s} | {t} | {c} | {c / t * 100:.0f}% |")
if total_t:
    print(f"| **Overall** | **{total_t}** | **{total_c}** | "
          f"**{total_c / total_t * 100:.0f}%** |")

with open("models/live_test_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["expected", "predicted", "correct"])
    w.writerows(log)
print("\nLog saved to models/live_test_log.csv")