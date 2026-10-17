# AI-Based Real-Time Sign Language Recognition & Speech Conversion System

A real-time system that recognizes hand signs from a webcam and converts them to speech.

## How it works
1. **MediaPipe Hands** detects 21 hand landmarks from the webcam feed.
2. Landmarks are normalized (wrist-relative, size-scaled).
3. A **Random Forest** classifier predicts the sign.
4. Predictions are smoothed over recent frames, then spoken with **pyttsx3**.

## Supported signs
Hello, Yes, No, ThankYou, Please, Sorry, Drink

## Project structure
```
src/
  hand_detection.py     # Week 1 prototype: hand landmark detection
  collect_data.py       # record landmark samples for each sign
  train_model.py        # train and evaluate the classifier
  predict_realtime.py   # live recognition + speech
data/landmarks.csv      # collected dataset
models/                 # trained model + confusion matrix
requirements.txt
```

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Run all commands from the project root.

1. Collect data (hold keys 1-7 to record each sign, Q to quit):
   ```
   python src/collect_data.py
   ```
2. Train the model:
   ```
   python src/train_model.py
   ```
3. Run live recognition (C = clear sentence, Q = quit):
   ```
   python src/predict_realtime.py
   ```

## Results
- Test accuracy: 99.34% (602 test samples, 7 signs)
- Live accuracy: _add your own results here_

## Tech stack
Python, OpenCV, MediaPipe, scikit-learn, NumPy, pandas, pyttsx3

## Future scope
More signs, two-hand support, sequence models (LSTM) for moving signs, sentence formation, a graphical interface.

## Live Accuracy
- Live accuracy: 100% (70/70 correct, 10 trials per sign, webcam, normal indoor lighting)

| Sign     | Trials | Correct | Accuracy |
|----------|--------|---------|----------|
| Hello    | 10     | 10       | 100%      |
| Yes      | 10     | 10       | 100%      |
| No       | 10     | 10       | 100%      |
| ThankYou | 10     | 10       | 100%      |
| Please   | 10     | 10       | 100%      |
| Sorry    | 10     | 10       | 100%      |
| Drink    | 10     | 10       | 100%      |
|**Overall**| **70**| **70**   | **100%**  |

- Test accuracy (dataset split): 99.34% (602 test samples)
- Live accuracy (real-time webcam): 100% (70/70 trials)