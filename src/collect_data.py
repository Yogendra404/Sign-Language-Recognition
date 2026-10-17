import csv
import os
import cv2
import mediapipe as mp

# ---- Edit your signs here: key -> label ----
SIGNS = {
    ord('1'): "Hello",
    ord('2'): "Yes",
    ord('3'): "No",
    ord('4'): "ThankYou",
    ord('5'): "Please",
    ord('6'): "Sorry",
    ord('7'): "Drink",
}
CSV_PATH = "data/landmarks.csv"
TARGET = 250  # samples per sign

os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Load existing counts so you can resume across sessions
counts = {label: 0 for label in SIGNS.values()}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1

new_file = not os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)
if new_file:
    header = ["label"] + [f"{a}{i}" for i in range(21) for a in "xyz"]
    writer.writerow(header)

cap = cv2.VideoCapture(0)
print("Hold a sign key (1-7) to record samples. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        if key in SIGNS:
            label = SIGNS[key]
            row = [label]
            for p in lm.landmark:
                row += [p.x, p.y, p.z]
            writer.writerow(row)
            counts[label] += 1

    # On-screen info
    y = 25
    for k, label in SIGNS.items():
        done = counts[label] >= TARGET
        color = (0, 255, 0) if done else (255, 255, 255)
        cv2.putText(frame, f"{chr(k)}: {label} {counts[label]}/{TARGET}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25

    cv2.imshow("Collect Data", frame)

csv_file.close()
cap.release()
cv2.destroyAllWindows()