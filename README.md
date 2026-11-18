# 😃 Facial Expression Recognition

A Convolutional Neural Network that recognises **7 facial expressions** (angry,
disgust, fear, happy, neutral, sad, surprise) from a face image, plus a small
Flask web app to try it on your own photos.

Built as a machine-learning project during my 3rd semester.

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | **57.7%** (7 classes; random guessing ≈ 14%) |
| Parameters | 422k |
| Input | 48×48 face image |

For reference, human accuracy on FER-2013 is about 65%, and state-of-the-art
models reach about 75%. The learning curves (see the notebook) show training and validation
accuracy plateauing together, so the model **underfits** rather than overfits.
A deeper network or transfer learning is the natural next step.

## Model

```
Input 48×48×3
→ Conv2D(32)  → MaxPool
→ Conv2D(64)  → MaxPool
→ Conv2D(128) → MaxPool
→ Conv2D(256) → MaxPool
→ Flatten → Dense(128, ReLU) → Dropout(0.5)
→ Dense(7, softmax)
```

Trained for 100 epochs with Adam, categorical cross-entropy and data
augmentation (rotation, shifts, shear, zoom, horizontal flip) on ~17k
training images. The full process is in
[`notebooks/FER_training.ipynb`](notebooks/FER_training.ipynb).

## Project structure

```
.
├── app.py                         # Flask app: preprocessing + /predict API
├── models/fer_cnn.h5              # Trained model
├── notebooks/FER_training.ipynb   # Data exploration, training, evaluation
├── templates/index.html           # Upload page with per-emotion probability bars
└── requirements.txt
```

## Run the app

**Requirements:** Python 3.9–3.11 (TensorFlow 2.15).

```bash
git clone https://github.com/belviskhoremk/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Open <http://127.0.0.1:5000> and upload a face photo.

> **Tip:** the model was trained on tightly cropped faces. Crop your photo
> around the face for much better results.

### API

```bash
curl -F file=@face.jpg http://127.0.0.1:5000/predict
```

```json
{
  "result": "happy",
  "confidence": 0.82,
  "probabilities": { "angry": 0.01, "disgust": 0.0, "fear": 0.02, "happy": 0.82, "neutral": 0.1, "sad": 0.03, "surprise": 0.02 }
}
```

## Tech stack

TensorFlow / Keras · Flask · Pillow · NumPy · Bootstrap
